"""One self-contained HTML page for a scored idea, or for two ideas side by side."""

from html import escape

from playodds import model as PM
from playodds.tables import FIELDS, MILESTONE_LABEL

COLORS = ["var(--a)", "var(--b)"]
GRID = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 66]

CSS = """
:root{--bg:#fbfbfa;--fg:#1d2127;--muted:#5d6672;--line:#d9dde2;--card:#fff;--a:#2c6fbb;--b:#d9822b;--track:#eceef1}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e8eaed;--muted:#9aa3ad;--line:#2d333a;--card:#1b1f24;
--a:#6aa6e6;--b:#f0a45d;--track:#262b31}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:880px;margin:0 auto;padding:32px 20px 56px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.1rem;margin:36px 0 10px}
.sub{color:var(--muted);margin:0 0 20px}
table{border-collapse:collapse;width:100%;font-size:.93rem}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin-bottom:12px}
.row{display:grid;grid-template-columns:64px 1fr;gap:12px;align-items:center;margin:8px 0}
.lab{font-weight:600}.bars{display:grid;gap:6px}
.bar{position:relative;height:22px;background:var(--track);border-radius:4px}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:4px;opacity:.85}
.whisk{position:absolute;top:50%;height:2px;background:var(--fg);opacity:.7}
.cap{position:absolute;top:4px;bottom:4px;width:2px;background:var(--fg);opacity:.7}
.val{position:absolute;top:0;line-height:22px;font-size:.85rem;padding-left:6px;white-space:nowrap}
.band{position:absolute;top:3px;bottom:3px;border-radius:3px;opacity:.55}
.legend{display:flex;gap:16px;margin:0 0 10px;font-size:.9rem}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
.note{color:var(--muted);font-size:.9rem}ul{padding-left:20px}li{margin:4px 0}
@media print{body{background:#fff}main{padding:0}}
"""


def pct(x):
    return f"{100 * x:.2f}%" if x < 0.01 else f"{100 * x:.1f}%"


def nice_max(x):
    for m in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        if x <= m:
            return m
    return 1.0


def title_of(app, i, n):
    if n == 1:
        return app.get("name") or app["category"]
    return f"{'AB'[i]}. {app['name']} ({app['category']})" if app.get("name") else f"{'AB'[i]}. {app['category']}"


def inputs_table(apps):
    head = "".join(f"<th>{escape(title_of(a, i, len(apps)))}</th>" for i, a in enumerate(apps))
    rows = []
    for k in FIELDS:
        cells = "".join(f"<td>{escape(format(a.get(k, PM.INPUTS[k]), 'g') if isinstance(a.get(k, PM.INPUTS[k]), float) else str(a.get(k, PM.INPUTS[k])))}</td>" for a in apps)
        rows.append(f"<tr><th>{k.replace('_', ' ')}</th>{cells}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>"


def milestone_rows(outs, apps):
    rows = []
    for ms in outs[0]["milestones"]:
        top = nice_max(max(o["milestones"][ms]["hi"] for o in outs))
        bars = []
        for i, o in enumerate(outs):
            r = o["milestones"][ms]
            w = lambda v: 100 * min(v / top, 1)
            bars.append(
                f'<div class="bar"><div class="fill" style="width:{w(r["p"]):.2f}%;background:{COLORS[i]}"></div>'
                f'<div class="whisk" style="left:{w(r["lo"]):.2f}%;width:{w(r["hi"]) - w(r["lo"]):.2f}%"></div>'
                f'<div class="val" style="left:{max(w(r["hi"]), w(r["p"])):.2f}%">{pct(r["p"])}</div></div>')
        rows.append(f'<div class="row"><div class="lab">{MILESTONE_LABEL[ms]}</div><div class="bars">{"".join(bars)}</div></div>')
    return "".join(rows)


def placement_rows(outs, apps):
    rows = []
    for i, o in enumerate(outs):
        lo, hi = o["percentile_interval"]
        inst = ""
        if "installs_interval" in o:
            a, b = o["installs_interval"]
            inst = f"about {a:,.0f} to {b:,.0f} installs for apps that old in June 2021"
        rows.append(
            f'<div class="bar"><div class="band" style="left:{100 * lo:.1f}%;width:{100 * (hi - lo):.1f}%;'
            f'background:{COLORS[i]}"></div></div>'
            f'<div class="note" style="margin:2px 0 10px">{escape(title_of(apps[i], i, len(apps)))} lands between the '
            f'{lo:.0%} and {hi:.0%} mark, {inst}</div>')
    return "".join(rows)


def curve_svg(label, series, ymax):
    w, h, l, b, t, r = 300, 170, 38, 24, 10, 8
    px = lambda m: l + (w - l - r) * m / 66
    py = lambda p: t + (h - t - b) * (1 - min(p / ymax, 1))
    g = []
    for f in (0, 0.5, 1):
        y = py(ymax * f)
        g.append(f'<line x1="{l}" x2="{w - r}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)"/>'
                 f'<text x="{l - 5}" y="{y + 4:.1f}" text-anchor="end">{100 * ymax * f:.3g}%</text>')
    for m in (0, 12, 24, 36, 48, 66):
        g.append(f'<text x="{px(m):.1f}" y="{h - 6}" text-anchor="middle">{m}</text>')
    for i, ys in enumerate(series):
        pts = " ".join(f"{px(m):.1f},{py(p):.1f}" for m, p in zip(GRID, ys))
        g.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[i]}" stroke-width="2"/>')
    return (f'<div class="card"><div class="lab">{label}</div>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Chance of reaching {label} by age in months">{"".join(g)}'
            '</svg>'
            f'<div class="note">months since launch</div></div>')


def comparables_table(model, app):
    rows = PM.comparables(model, app)
    if not rows:
        return ""
    body = "".join(f"<tr><td>{escape(str(c['name'])[:48])}</td><td>{escape(str(c['released']))}</td>"
                   f"<td class='n'>{int(c['installs']):,}</td></tr>" for c in rows)
    return ("<table><tr><th>App</th><th>Launched</th><th class='n'>Installs by June 2021</th></tr>"
            f"{body}</table>")


def build(model, apps, outs, months, level):
    n = len(apps)
    heading = " versus ".join(escape(title_of(a, i, n)) for i, a in enumerate(apps)) if n > 1 \
        else escape(title_of(apps[0], 0, 1))
    legend = "".join(f'<span><span class="sw" style="background:{COLORS[i]}"></span>{escape(title_of(a, i, n))}</span>'
                     for i, a in enumerate(apps)) if n > 1 else ""
    curves = [PM.curve(model, a, GRID) for a in apps]
    charts = ""
    for ms in model["milestone_list"]:
        series = [c[ms] for c in curves]
        charts += curve_svg(MILESTONE_LABEL[ms], series, nice_max(max(max(s) for s in series)))
    comps = "".join(
        f"<h2>Real apps with a similar launch-day profile{' for ' + escape(title_of(a, i, n)) if n > 1 else ''}</h2>"
        + comparables_table(model, a) for i, a in enumerate(apps))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Launch-day odds, {heading}</title><style>{CSS}</style></head><body><main>
<h1>{heading}</h1>
<p class="sub">Chance of reaching each install milestone within {months:g} months of launch, from facts known on launch day.</p>
{f'<div class="legend">{legend}</div>' if legend else ''}
<h2>Chance of reaching each milestone</h2>
<div class="card">{milestone_rows(outs, apps)}
<p class="note">The bar is the model's estimate and the line is the coefficient range ({level}%), which only shows how well the
coefficients are pinned down. It is far too narrow for any single app.</p></div>
<h2>Where it ends up among apps launched the same month</h2>
<div class="card">{placement_rows(outs, apps)}
<p class="note">The {level}% conformal interval on a 0 to 100% scale, where 100% is the top of the month. It was checked on
apps the model never saw, and it is wide.</p></div>
<h2>Chance by age</h2>
<div class="charts">{charts}</div>
{comps}
<h2>Inputs</h2>
{inputs_table(apps)}
<h2>How to read this</h2>
<ul>
<li>These are patterns in Google Play apps launched from 2016 to 2021. They describe how similar past apps turned out and
say nothing about what any change to an input would cause.</li>
<li>Category and title are 2021 values, so part of what the model reads may have been edited after launch.</li>
<li>The model is not calibrated per category, and single categories can be off by a factor of two.</li>
<li>Details are in docs/model_card.md of the playodds repository.</li>
</ul>
</main></body></html>
"""
