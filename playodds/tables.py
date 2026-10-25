"""Input parsing for batch and compare, flat output rows, and the category explorer."""

import csv
import difflib
import io
import json
import sys
from pathlib import Path

import pandas as pd

from playodds import model as PM

WHOLE = ["earlier_apps", "same_day_apps", "apps_last_30_days", "earlier_categories",
         "earlier_apps_same_category", "builder_apps"]
FIELDS = ["category", "name", *WHOLE[:3], "months_since_first_app", *WHOLE[3:]]
MILESTONE_LABEL = {"10000": "10K", "100000": "100K", "1000000": "1M"}


def clean_keys(raw):
    return {str(k).strip().lower().replace("-", "_"): v for k, v in raw.items()}


def parse_app(raw):
    """Turns text values (from a CSV row or key=value flags) into an app dict. Blank means default."""
    app = {}
    for key, val in raw.items():
        k = str(key).strip().lower().replace("-", "_")
        if k not in FIELDS:
            near = difflib.get_close_matches(k, FIELDS, n=1)
            raise ValueError(f"unknown input {key!r}." + (f" Did you mean {near[0]}?" if near else "")
                             + f" Inputs are {', '.join(FIELDS)}")
        val = "" if val is None else str(val).strip()
        if val == "":
            continue
        if k in ("category", "name"):
            app[k] = val
            continue
        try:
            num = float(val)
        except ValueError:
            raise ValueError(f"{k} must be a number (got {val!r})") from None
        if k in WHOLE:
            if num != int(num):
                raise ValueError(f"{k} must be a whole number (got {val!r})")
            num = int(num)
        app[k] = num
    if "earlier_categories" not in app:
        app["earlier_categories"] = 1 if app.get("earlier_apps", 0) > 0 else 0
    return app


def flat_row(app, out):
    """One score as a flat dict, used for CSV output."""
    row = {"category": app["category"], "name": app.get("name", ""), "months": out["months"]}
    for k in FIELDS[2:]:
        row[k] = app.get(k, PM.INPUTS[k])
    for ms, r in out["milestones"].items():
        lab = MILESTONE_LABEL[ms]
        row[f"p_{lab}"], row[f"p_{lab}_lo"], row[f"p_{lab}_hi"] = r["p"], r["lo"], r["hi"]
    row["launch_month_pct_lo"], row["launch_month_pct_hi"] = out["percentile_interval"]
    if "installs_interval" in out:
        row["installs_lo"], row["installs_hi"] = out["installs_interval"]
    return row


def write_csv(rows, fh):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in r.items()})


def csv_text(rows):
    buf = io.StringIO()
    write_csv(rows, buf)
    return buf.getvalue()


def read_ideas(path):
    """Rows of an ideas CSV as dicts of text. A '-' reads standard input."""
    if path != "-" and not Path(path).is_file():
        raise ValueError(f"cannot find the file {path!r}")
    fh = sys.stdin if path == "-" else open(path, newline="", encoding="utf-8-sig")
    try:
        rd = csv.DictReader(fh)
        if rd.fieldnames is None:
            raise ValueError("the file is empty. `playodds batch --template` prints a starter file")
        names = [c.strip().lower().replace("-", "_") for c in rd.fieldnames]
        names = [n if n in FIELDS else c for n, c in zip(names, rd.fieldnames)]
        if "category" not in names:
            raise ValueError("the file needs a category column. `playodds batch --template` prints a starter file")
        return [dict(zip(names, r.values())) for r in rd]
    finally:
        if fh is not sys.stdin:
            fh.close()


def score_rows(rows, model, months, level):
    """Scores each row. Returns (results, errors) where errors maps a row number (1 is the first idea) to a message."""
    results, errors = [], {}
    for i, raw in enumerate(rows, start=1):
        fields = {k: v for k, v in raw.items() if k in FIELDS}
        try:
            app = PM.prepare_app(parse_app(fields), model)
            out = PM.score(app, months=months, model=model, level=level)
        except ValueError as e:
            errors[i] = str(e)
            results.append(None)
            continue
        results.append((app, out))
    return results, errors


# ---------------------------------------------------------------------------
# category explorer
# ---------------------------------------------------------------------------
SORTS = {"name": "category", "apps": "apps", "new-apps": "new_apps_2020", "top10": "top10_share", "gini": "gini",
         "first-app-10k": "first_app_12m_10K", "first-app-100k": "first_app_12m_100K"}


def market_table():
    p = PM.data_path("market.csv")
    if p is None:
        raise FileNotFoundError("market.csv not found. Run from the repo, or reinstall with `pip install .`")
    return pd.read_csv(p)


def checks_data():
    p = PM.data_path("checks.json")
    return json.loads(p.read_text()) if p else {}


def category_profile(model, name):
    """Market numbers for one category, plus how the model's first-app average compares with the raw rate."""
    cat = PM.resolve_category(model, name)
    t = market_table()
    row = t[t["category"] == cat]
    if row.empty:
        raise ValueError(f"no market row for {cat}")
    r = row.iloc[0]
    n = len(t)
    rank = lambda col, high=True: int(t[col].rank(ascending=not high, method="min")[row.index[0]])
    out = {
        "category": cat,
        "apps": int(r["apps"]),
        "new_apps_2018": int(r["new_apps_2018"]),
        "new_apps_2020": int(r["new_apps_2020"]),
        "new_entrant_share": float(r["new_entrant_share"]),
        "top10_share": float(r["top10_share"]),
        "gini": float(r["gini"]),
        "median_installs_2020_launch": float(r["median_installs_2020_launch"]),
        "pooled_share_reached": {
            lab: {"12_months": float(r[f"npmle_12m_{lab}"]), "36_months": float(r[f"npmle_36m_{lab}"])}
            for lab in ["10K", "100K", "1M"]},
        "ranks_of": n,
        "rank_top10_share": rank("top10_share", high=False),
        "rank_first_app_10K": rank("first_app_12m_10K"),
    }
    fa = checks_data().get("first_apps_2019_2020", {}).get(cat)
    if fa:
        out["first_apps"] = {"model_average_at_12_months": fa["model_average_at_12_months"],
                             "apps_launched_2019_2020": int(r["first_apps_2019_2020"]),
                             "observed_ages_9_to_15_months": {
                                 "apps": fa["ages_9_to_15"]["apps"],
                                 **{lab: {"observed": fa["ages_9_to_15"][lab]["observed"],
                                          "lo": fa["ages_9_to_15"][lab]["lo"], "hi": fa["ages_9_to_15"][lab]["hi"],
                                          "model": fa["ages_9_to_15"][lab]["model"]} for lab in ["10K", "100K", "1M"]}}}
    return out


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def category_text(p):
    n = p["ranks_of"]
    lines = [f"{p['category']}",
             f"  {p['apps']:,} apps in the June 2021 scrape, {p['new_apps_2020']:,} launched in 2020 "
             f"({p['new_apps_2018']:,} in 2018)",
             f"  top 10 apps hold {pct(p['top10_share'])} of installs, number {p['rank_top10_share']} of {n} counting "
             f"from the least concentrated, Gini {p['gini']:.3f}",
             f"  median installs of a 2020 launch {p['median_installs_2020_launch']:,.0f}, "
             f"{pct(p['new_entrant_share'])} of installs go to apps launched in 2019 or 2020", "",
             "  Share of apps that reached each milestone (pooled over launch cohorts)",
             f"  {'':<13}{'10K':>8}{'100K':>8}{'1M':>8}"]
    for lab, key in [("by 12 months", "12_months"), ("by 36 months", "36_months")]:
        v = [p["pooled_share_reached"][m][key] for m in ["10K", "100K", "1M"]]
        lines.append(f"  {lab:<13}{pct(v[0]):>8}{pct(v[1]):>8}{pct(v[2], 2):>8}")
    fa = p.get("first_apps")
    if fa:
        o = fa["observed_ages_9_to_15_months"]
        m = fa["model_average_at_12_months"]
        lines += ["", f"  A developer's first app, launched 2019 or 2020 ({fa['apps_launched_2019_2020']:,} apps), "
                  f"number {p['rank_first_app_10K']} of {n} for the model's chance of 10K",
                  f"  {'':<13}{'10K':>8}{'100K':>8}{'1M':>8}",
                  f"  {'model, 12 mo':<13}{pct(m['10K']):>8}{pct(m['100K']):>8}{pct(m['1M'], 2):>8}",
                  f"  {'observed*':<13}{pct(o['10K']['observed']):>8}{pct(o['100K']['observed']):>8}"
                  f"{pct(o['1M']['observed'], 2):>8}",
                  f"  * raw share of the {o['apps']:,} first apps that were 9 to 15 months old in June 2021, "
                  f"95% intervals 10K {pct(o['10K']['lo'])} to {pct(o['10K']['hi'])}",
                  f"  The model is not calibrated per category, so treat gaps up to a factor of two as normal."]
    lines += ["", "  Past apps only. Nothing here says what would happen if you picked this category."]
    return "\n".join(lines)


def category_table(sort="first-app-10k"):
    t = market_table()
    col = SORTS[sort]
    t = t.sort_values(col, ascending=(col == "category"), kind="stable").reset_index(drop=True)
    return t[["category", "apps", "new_apps_2020", "top10_share", "gini", "npmle_12m_10K", "npmle_12m_100K",
              "first_app_12m_10K", "first_app_12m_100K"]]
