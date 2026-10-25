"""
Command line scorer for the launch-day model.

  playodds --category Puzzle --name "Block Drop" --earlier-apps 3 --months-since-first-app 14
  playodds compare --a category=Puzzle "name=Block Drop" --b category=Tools
  playodds batch ideas.csv -o scored.csv
  playodds category Simulation
"""

import argparse
import json
import sys
from pathlib import Path

from playodds import __version__, report, tables
from playodds import model as PM

COMMANDS = ["score", "compare", "batch", "categories", "category"]
FORMATS = ["text", "json", "csv"]

EPILOG = """examples
  playodds --category Puzzle --name "Block Drop" --earlier-apps 3 --months-since-first-app 14
  playodds score --category Tools --html tools.html
  playodds compare --a category=Puzzle "name=Block Drop" --b category=Tools
  playodds batch --template > ideas.csv
  playodds batch ideas.csv -o scored.csv
  playodds categories --table --sort top10
  playodds category Simulation

Running playodds with flags and no command is the same as `playodds score`. Every number is an association in
launch-day data from 2016 to 2021 and none of it says what a change would cause. See docs/model_card.md."""

INPUT_HELP = [
    ("--category", "Play category. Run `playodds categories` for the list. Case does not matter and close misspellings "
                   "get suggestions."),
    ("--name", "app title as it will appear on Play. Only its length, word count, digits and non-ASCII characters "
               "are used. Default is empty."),
    ("--earlier-apps", "apps you already released on Play before this one, a whole number from 0. Default 0."),
    ("--same-day-apps", "other apps you release the same day, a whole number from 0. Default 0."),
    ("--apps-last-30-days", "apps you released in the 30 days before this one, a whole number from 0. Default 0."),
    ("--months-since-first-app", "months since your first Play app, 0 or more, 0 for a first app. Default 0."),
    ("--earlier-categories", "distinct categories among your earlier apps, a whole number from 0. Default is 1 if "
                             "you have earlier apps, otherwise 0."),
    ("--earlier-apps-same-category", "earlier apps in this same category, a whole number from 0. Default 0."),
    ("--builder-apps", "earlier Play apps that share your package name prefix, which is how app builder templates "
                       "show up. A whole number from 0. Default 0."),
]


def add_inputs(p):
    g = p.add_argument_group("the app (everything here is known on launch day)")
    for flag, text in INPUT_HELP:
        kind = float if flag == "--months-since-first-app" else str if flag in ("--category", "--name") else int
        g.add_argument(flag, type=kind, default=None, help=text, metavar=flag[2:].upper().replace("-", "_"))
    return g


def add_run(p, formats=FORMATS):
    g = p.add_argument_group("how to run it")
    g.add_argument("--months", type=float, default=12,
                   help="time window in months after launch, from 0.5 to 66 (default 12)")
    g.add_argument("--level", choices=["50", "80", "90"], default="80",
                   help="coverage of the intervals in percent (default 80)")
    g.add_argument("--format", choices=formats, default=formats[0], help="output format (default %(default)s)")
    g.add_argument("--model", default=None, help="path to a milestones.json (default is the one that ships)")


def build_parser():
    p = argparse.ArgumentParser(prog="playodds", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Odds that a new Google Play app reaches 10K, 100K or 1M installs, "
                                            "using only what you know on launch day.", epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"playodds {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    s = sub.add_parser("score", help="odds for one app idea (the default command)",
                       description="Odds that one app reaches 10K, 100K and 1M installs within a time window.")
    add_inputs(s)
    add_run(s)
    s.add_argument("--json", action="store_true", help="same as --format json")
    s.add_argument("--html", metavar="FILE", help="also write a self-contained HTML report")
    s.add_argument("--comparables", type=int, default=5, metavar="N",
                   help="how many real apps with a similar profile to list, 0 for none (default 5)")

    c = sub.add_parser("compare", help="two app ideas side by side",
                       description="Scores two ideas and shows the gap. Anything --b leaves out is copied from --a, "
                                   "so you can change one input at a time. The gap compares profiles of past apps. It "
                                   "is not what changing the input would do.")
    c.add_argument("--a", nargs="+", required=True, metavar="KEY=VALUE",
                   help="first idea, like category=Puzzle \"name=Block Drop\" earlier_apps=3. Keys: "
                        + ", ".join(tables.FIELDS))
    c.add_argument("--b", nargs="+", required=True, metavar="KEY=VALUE", help="second idea, same keys")
    add_run(c)
    c.add_argument("--html", metavar="FILE", help="also write a self-contained HTML report")

    b = sub.add_parser("batch", help="score many ideas from a CSV file",
                       description="Scores every row of a CSV. Columns are the app inputs (category is required, blank "
                                   "cells use the defaults). Other columns, like an id or a label, are copied through.")
    b.add_argument("file", nargs="?", help="CSV of ideas, or - for standard input")
    b.add_argument("-o", "--output", metavar="FILE", help="write here instead of standard output")
    b.add_argument("--template", action="store_true", help="print a starter CSV and stop")
    b.add_argument("--keep-going", action="store_true",
                   help="score the valid rows and put a message in an error column for the rest")
    b.add_argument("--sort", choices=["10k", "100k", "1m"], help="order the output by this chance, highest first")
    add_run(b, formats=["csv", "json"])

    k = sub.add_parser("categories", help="list the categories, or print the market table",
                       description="Lists the Play categories the model knows.")
    k.add_argument("--table", action="store_true", help="print the market table for all categories")
    k.add_argument("--sort", choices=list(tables.SORTS), default="first-app-10k",
                   help="column for --table, high to low except name (default first-app-10k)")
    k.add_argument("--format", choices=FORMATS, default="text", help="output format (default text)")
    k.add_argument("--model", default=None, help=argparse.SUPPRESS)

    q = sub.add_parser("category", help="market numbers for one category",
                       description="Concentration, supply, pooled time to each milestone and a first app's odds for one "
                                   "category, from the June 2021 scrape.")
    q.add_argument("name", help="a category, like Simulation (case does not matter)")
    q.add_argument("--format", choices=["text", "json"], default="text", help="output format (default text)")
    q.add_argument("--model", default=None, help=argparse.SUPPRESS)
    return p


def normalize_argv(argv):
    """Flags without a command mean `score`, and the old --list-categories flag still works."""
    if not argv:
        return ["-h"]
    if argv[0] in COMMANDS or argv[0] in ("-h", "--help", "--version"):
        return argv
    if "--list-categories" in argv:
        return ["categories"]
    return ["score", *argv]


def app_from_flags(a):
    raw = {"category": a.category, "name": a.name, "earlier_apps": a.earlier_apps, "same_day_apps": a.same_day_apps,
           "apps_last_30_days": a.apps_last_30_days, "months_since_first_app": a.months_since_first_app,
           "earlier_categories": a.earlier_categories, "earlier_apps_same_category": a.earlier_apps_same_category,
           "builder_apps": a.builder_apps}
    app = {k: v for k, v in raw.items() if v is not None}
    if "earlier_categories" not in app:
        app["earlier_categories"] = 1 if app.get("earlier_apps", 0) > 0 else 0
    return app


def app_from_spec(tokens):
    raw = {}
    for t in tokens:
        if "=" not in t:
            raise ValueError(f"expected KEY=VALUE, got {t!r}. Quote values with spaces, like \"name=Block Drop\"")
        k, v = t.split("=", 1)
        raw[k] = v
    return raw


def fmt_p(p):
    return f"{p:.2%}" if p < 0.01 else f"{p:.1%}"


def score_text(app, out, months, level, comps):
    name = app.get("name", "")
    lines = [f"{name} ({app['category']}), within {months:g} months of launch" if name
             else f"{app['category']}, within {months:g} months of launch"]
    for ms, r in out["milestones"].items():
        lines.append(f"  {tables.MILESTONE_LABEL[ms]:>4} installs  {fmt_p(r['p']):>6}   "
                     f"(coefficient range {fmt_p(r['lo'])} to {fmt_p(r['hi'])})")
    lo, hi = out["percentile_interval"]
    lines.append(f"\nAmong apps launched the same month, the {level}% interval for where it ends up runs from the "
                 f"{lo:.0%} to the {hi:.0%} mark, so it is wide.")
    if "installs_interval" in out:
        il, ih = out["installs_interval"]
        lines.append(f"For apps {months:g} months old in June 2021 that was about {il:,.0f} to {ih:,.0f} installs.")
    if comps:
        lines.append(f"\nReal {app['category']} apps with the closest launch-day profile:")
        for c in comps:
            lines.append(f"  {c['name'][:40]:40s} launched {c['released']}, {c['installs']:>12,} installs by June 2021")
    lines.append("\nThe coefficient range only says how well the model's coefficients are pinned down. It is too "
                 "narrow for one app.\nThese are patterns from apps launched 2016 to 2021. See docs/model_card.md.")
    return "\n".join(lines)


def write_html(path, model, apps, outs, months, level):
    Path(path).write_text(report.build(model, apps, outs, months, level), encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def run_score(a):
    model = PM.load(a.model)
    app = PM.prepare_app(app_from_flags(a), model)
    out = PM.score(app, months=a.months, model=model, level=a.level)
    out["comparables"] = PM.comparables(model, app, k=a.comparables) if a.comparables > 0 else []
    fmt = "json" if a.json else a.format
    if fmt == "json":
        print(json.dumps({"app": app, **out}, indent=2, default=float))
    elif fmt == "csv":
        sys.stdout.write(tables.csv_text([tables.flat_row(app, out)]))
    else:
        print(score_text(app, out, a.months, a.level, out["comparables"]))
    if a.html:
        write_html(a.html, model, [app], [out], a.months, a.level)


def run_compare(a):
    model = PM.load(a.model)
    spec_a = tables.clean_keys(app_from_spec(a.a))
    spec_b = {**spec_a, **tables.clean_keys(app_from_spec(a.b))}
    apps = [PM.prepare_app(tables.parse_app(s), model) for s in (spec_a, spec_b)]
    outs = [PM.score(x, months=a.months, model=model, level=a.level) for x in apps]
    if a.format == "json":
        print(json.dumps({"months": a.months, "level": a.level, "a": {"app": apps[0], **outs[0]},
                          "b": {"app": apps[1], **outs[1]}}, indent=2, default=float))
    elif a.format == "csv":
        sys.stdout.write(tables.csv_text([{"idea": lab, **tables.flat_row(x, o)}
                                          for lab, x, o in zip("AB", apps, outs)]))
    else:
        print(compare_text(*apps, outs, a.months, a.level))
    if a.html:
        write_html(a.html, model, apps, outs, a.months, a.level)


def label(app):
    return f"{app['name']} ({app['category']})" if app.get("name") else app["category"]


def compare_text(x, y, outs, months, level):
    w = max(12, len(label(x)), len(label(y)))
    lines = [f"Within {months:g} months of launch", "",
             f"{'':>14}  {'A: ' + label(x):<{w + 3}}  {'B: ' + label(y):<{w + 3}}  gap (A minus B)"]
    for ms in outs[0]["milestones"]:
        pa, pb = (o["milestones"][ms]["p"] for o in outs)
        ratio = f", {pa / pb:.2f}x" if pb > 0 else ""
        lines.append(f"{tables.MILESTONE_LABEL[ms] + ' installs':>14}  {fmt_p(pa):<{w + 3}}  {fmt_p(pb):<{w + 3}}  "
                     f"{100 * (pa - pb):+.2f} points{ratio}")
    lines.append(f"\nWhere each lands among apps launched the same month ({level}% interval)")
    for lab, o in zip("AB", outs):
        lo, hi = o["percentile_interval"]
        lines.append(f"  {lab}: {lo:.0%} to {hi:.0%} mark")
    diff = [k for k in tables.FIELDS if x.get(k, PM.INPUTS[k]) != y.get(k, PM.INPUTS[k])]
    lines.append("\nInputs that differ: " + (", ".join(f"{k} ({x.get(k, PM.INPUTS[k])} vs {y.get(k, PM.INPUTS[k])})"
                                                      for k in diff) or "none"))
    lines.append("\nThis compares two profiles of past apps. It does not say what changing an input would do.")
    return "\n".join(lines)


TEMPLATE = ("id,category,name,earlier_apps,same_day_apps,apps_last_30_days,months_since_first_app,earlier_categories,"
            "earlier_apps_same_category,builder_apps\n"
            "block-drop,Puzzle,Block Drop,3,0,0,14,2,1,0\n"
            "battery-saver,Tools,Battery Saver,,,,,,,\n")


def run_batch(a):
    if a.template:
        sys.stdout.write(TEMPLATE)
        return
    if not a.file:
        raise ValueError("give a CSV file, or use --template to get a starter file")
    model = PM.load(a.model)
    rows = tables.read_ideas(a.file)
    if not rows:
        raise ValueError("the file has a header but no ideas")
    results, errors = tables.score_rows(rows, model, a.months, a.level)
    if errors and not a.keep_going:
        shown = [f"  row {i}: {m}" for i, m in list(errors.items())[:20]]
        more = f"\n  and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"{len(errors)} of {len(rows)} rows have problems, nothing scored. Fix them or use "
                         "--keep-going\n" + "\n".join(shown) + more)
    recs, order = [], []
    for i, (raw, res) in enumerate(zip(rows, results), start=1):
        rec = dict(raw)
        if res is None:
            rec["error"] = errors[i]
            order.append(1.0)
            recs.append(rec)
            continue
        flat = tables.flat_row(*res)
        rec["category"] = flat["category"]
        rec.update({k: v for k, v in flat.items() if k not in tables.FIELDS})
        order.append(-flat.get(f"p_{a.sort.upper()}", 0) if a.sort else 0)
        recs.append(rec)
    if a.sort:
        recs = [r for _, r in sorted(zip(order, recs), key=lambda t: t[0])]
    if a.keep_going:
        for r in recs:
            r.setdefault("error", "")
    text = tables.csv_text(recs) if a.format == "csv" else json.dumps(recs, indent=2, default=float) + "\n"
    done = f"scored {len(rows) - len(errors)} of {len(rows)} ideas"
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8")
        print(f"{done}, wrote {a.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
        print(done, file=sys.stderr)


def run_categories(a):
    if not a.table:
        model = PM.load(a.model)
        cats = model["categories"]
        if a.format == "json":
            print(json.dumps(cats, indent=2))
        else:
            print("\n".join(cats))
        return
    t = tables.category_table(a.sort)
    if a.format == "json":
        print(t.to_json(orient="records", indent=2))
    elif a.format == "csv":
        sys.stdout.write(t.to_csv(index=False, float_format="%.4f"))
    else:
        w = max(len(c) for c in t["category"])
        print(f"{'category':<{w}}  {'apps':>8}  {'new 2020':>8}  {'top10':>6}  {'gini':>5}  {'12m 10K':>7}  "
              f"{'12m 100K':>8}  {'1st app 10K':>11}  {'1st app 100K':>12}")
        for r in t.itertuples():
            print(f"{r.category:<{w}}  {r.apps:>8,}  {r.new_apps_2020:>8,}  {r.top10_share:>6.1%}  {r.gini:>5.3f}  "
                  f"{r.npmle_12m_10K:>7.1%}  {r.npmle_12m_100K:>8.1%}  {r.first_app_12m_10K:>11.1%}  "
                  f"{r.first_app_12m_100K:>12.1%}")
        print("\n12m columns are the share of all apps that reached the milestone within a year (pooled over cohorts). "
              "1st app columns are the model's average for a developer's first apps. Past apps only.")


def run_category(a):
    model = PM.load(a.model)
    p = tables.category_profile(model, a.name)
    print(json.dumps(p, indent=2) if a.format == "json" else tables.category_text(p))


RUNNERS = {"score": run_score, "compare": run_compare, "batch": run_batch, "categories": run_categories,
           "category": run_category}


def main(argv=None):
    argv = normalize_argv(list(sys.argv[1:] if argv is None else argv))
    a = build_parser().parse_args(argv)
    if a.command is None:
        build_parser().print_help()
        return 0
    try:
        RUNNERS[a.command](a)
    except (ValueError, FileNotFoundError) as e:
        print(f"playodds: error: {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        sys.stderr.close()      # piped into head or similar
    return 0


if __name__ == "__main__":
    sys.exit(main())
