"""
Where is there room for a new app? Category by category: demand (installs), supply (new apps),
how concentrated installs are (top 10 share, Gini, HHI), how fast new apps reach milestones (NPMLE
curves per category), and how a first-time developer's app does in its first year (from the
milestone model). Runs after study.milestones.

Run:  python -m study.market
"""

import gc

import numpy as np
import pandas as pd

from study import milestones as MS
from study.common import BLUE, DARK, FAST, FEATURES, GRAY, RED, RES, dirs, pct, plt, read, save, write

MIN_APPS = 50 if FAST else 2000      # categories smaller than this are left out of the table


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def hhi(x):
    s = np.asarray(x, float) / np.sum(x)
    return float(10_000 * np.sum(s ** 2))


SLIM = FEATURES.with_name("market_input.pkl")


def load_apps():
    """Four columns of the feature table, cached so the 760 MB file is read once."""
    if not SLIM.exists() or SLIM.stat().st_mtime < FEATURES.stat().st_mtime:
        pd.read_pickle(FEATURES)[["Category", "released", "installs", "age_days"]].to_pickle(SLIM)
        gc.collect()
    return pd.read_pickle(SLIM)


def main():
    dirs()
    df = load_apps()
    ms = read("milestones.json")
    first = ms["p12_by_category_first_app"]
    df["cat"] = df["Category"].astype(str)
    year = df["released"].dt.year
    w = df[(df["released"] >= MS.START) & (df["released"] < MS.END)]
    age = np.round(w["age_days"].to_numpy() / MS.MONTH, 2)

    rows = []
    curves = {}
    for c, g in df.groupby("cat"):
        inst = g["installs"].to_numpy()
        new18, new20 = int((year[g.index] == 2018).sum()), int((year[g.index] == 2020).sum())
        g20 = g[year[g.index] == 2020]
        recent = g[year[g.index].isin([2019, 2020])]
        wc = w["cat"].to_numpy() == c
        r = {"category": c, "apps": len(g), "installs": float(inst.sum()),
             "new_apps_2018": new18, "new_apps_2020": new20,
             "supply_growth_2018_to_2020": new20 / new18 if new18 else np.nan,
             "installs_to_2020_launches": float(g20["installs"].sum()),
             "installs_per_2020_launch": float(g20["installs"].mean()) if len(g20) else np.nan,
             "median_installs_2020_launch": float(g20["installs"].median()) if len(g20) else np.nan,
             "new_entrant_share": float(recent["installs"].sum() / inst.sum()),
             "top10_share": float(np.sort(inst)[::-1][:10].sum() / inst.sum()),
             "gini": gini(inst), "hhi": hhi(inst)}
        for m in MS.MILESTONES:
            a, F_ = MS.npmle(age[wc], (w["installs"].to_numpy()[wc] >= m).astype(float))
            r[f"npmle_12m_{MS.NAMES[m]}"] = float(MS.f_at(a, F_, 12.0))
            r[f"npmle_36m_{MS.NAMES[m]}"] = float(MS.f_at(a, F_, 36.0))
            if m == 100_000:
                curves[c] = (a, F_)
        f = first.get(c, {})
        r["first_app_12m_10K"] = f.get("10K", np.nan)
        r["first_app_12m_100K"] = f.get("100K", np.nan)
        r["first_apps_2019_2020"] = f.get("apps", 0)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("category")
    t = t[t["apps"] >= MIN_APPS].sort_values("first_app_12m_10K", ascending=False, na_position="last")
    t.round(6).to_csv(RES / "market.csv")
    write("market.json", {"categories": t.reset_index().to_dict(orient="records"),
                          "corr_top10_vs_first_app_10K": float(t["top10_share"].corr(t["first_app_12m_10K"],
                                                                                   method="spearman")),
                          "corr_supply_growth_vs_first_app_10K": float(
                              t["supply_growth_2018_to_2020"].corr(t["first_app_12m_10K"], method="spearman"))})
    figures(t, curves)
    print(t[["new_apps_2020", "installs_per_2020_launch", "top10_share", "gini", "hhi", "npmle_12m_100K",
             "first_app_12m_10K", "first_app_12m_100K"]].round(3).to_string())


def label_points(ax, t, x, y, names, color=DARK, avoid=()):
    """Put each label at the first of a few offsets where it doesn't land on a label already placed."""
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    placed = [a.get_window_extent(rend) for a in avoid]
    offsets = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"), (-5, -3, "right", "top"),
               (0, 8, "center", "bottom"), (0, -8, "center", "top"), (5, 14, "left", "bottom"), (5, -14, "left", "top")]
    for n in dict.fromkeys(names):
        if n not in t.index or not (np.isfinite(t.loc[n, x]) and np.isfinite(t.loc[n, y])):
            continue
        for dx, dy, ha, va in offsets:
            a = ax.annotate(n, (t.loc[n, x], t.loc[n, y]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                            color=color, ha=ha, va=va)
            box = a.get_window_extent(rend)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            if (dx, dy, ha, va) == offsets[-1]:
                placed.append(box)
            else:
                a.remove()


def figures(t, curves):
    # market map: supply vs demand, and break-in odds vs concentration
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    ax = axes[0]
    good = t["first_app_12m_10K"] >= t["first_app_12m_10K"].quantile(0.75)
    ax.scatter(t["new_apps_2020"], t["installs_to_2020_launches"], s=np.sqrt(t["apps"]) * 0.8,
               color=np.where(good, BLUE, GRAY), alpha=0.85, edgecolor="white", lw=0.5)
    lo, hi = t["new_apps_2020"].min() * 0.7, t["new_apps_2020"].max() * 1.4
    guides = []
    for k in [1e3, 1e4, 1e5]:
        ax.plot([lo, hi], [lo * k, hi * k], color=GRAY, lw=0.6, ls=":")
        guides.append(ax.text(hi, hi * k, f"{k:,.0f} per app", fontsize=7, color=GRAY, va="bottom", ha="right"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("New apps launched in 2020 (supply)")
    ax.set_ylabel("Installs those 2020 apps got by June 2021 (demand)")
    ax.set_title("Demand vs supply for new apps")
    top = list(t.sort_values("installs_per_2020_launch", ascending=False).index[:6])
    bottom = list(t.sort_values("installs_per_2020_launch").index[:4])
    label_points(ax, t, "new_apps_2020", "installs_to_2020_launches", top + bottom, avoid=guides)

    ax = axes[1]
    ax.scatter(t["top10_share"], t["first_app_12m_10K"], s=np.sqrt(t["apps"]) * 0.8,
               color=np.where(good, BLUE, GRAY), alpha=0.85, edgecolor="white", lw=0.5)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of the category's installs taken by its top 10 apps")
    ax.set_ylabel("First app's chance of 10K installs in year one")
    ax.set_title("How hard it is to break in (blue = top quarter)")
    label_points(ax, t, "top10_share", "first_app_12m_10K",
                 list(t.index[:6]) + list(t.index[-4:]) + list(t.sort_values("top10_share").index[-3:]))
    save(fig, "13_market_map.png")

    # time to 100K by category
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    order = t.sort_values("npmle_36m_100K", ascending=False).index
    show_top, show_bottom = list(order[:4]), list(order[-3:])
    curves = {c: (a[a <= 60], F_[a <= 60]) for c, (a, F_) in curves.items()}    # thin tails past 5 years are noise
    for c, (a, F_) in curves.items():
        if c not in t.index or c in show_top + show_bottom:
            continue
        ax.step(a, F_, where="post", color=GRAY, lw=0.5, alpha=0.35)
    shades = [BLUE, "#5468D4", "#7C8CE0", "#A3B0EC"]
    for c, col in zip(show_top, shades):
        a, F_ = curves[c]
        ax.step(a, F_, where="post", color=col, lw=1.8, label=c)
    for c, col in zip(show_bottom, [RED, "#C96A6A", "#DA9A9A"]):
        a, F_ = curves[c]
        ax.step(a, F_, where="post", color=col, lw=1.8, label=c)
    ax.set_xlim(0, 60)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Months since launch")
    ax.set_ylabel("Share of apps past 100K installs (NPMLE)")
    ax.set_title("Time to 100K installs by category (gray = the rest)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "14_category_curves.png")


if __name__ == "__main__":
    main()
