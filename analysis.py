"""
What actually gets an app to 100K installs? An analysis of 2.3M Google Play apps.

Setup:  ./get_data.sh            (downloads the data into data/)
Run:    python analysis.py       (prints every number in the README, saves charts to figures/)

Sections:
  1. Load and clean
  2. How concentrated installs are
  3. Monetization model vs. breaking out
  4. Developer experience (portfolio size)
  5. Category map: where new apps can still break out
  6. Model: can you spot a future hit from its listing? (trained on 2018, tested on 2019)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/google_playstore_2021.csv")
CACHE = Path("data/clean.pkl")
FIG, RES = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

HIT = 100_000           # "breaking out" = 100K+ installs
COHORT = (2018, 2019)   # apps old enough (1.5 to 3.5 years at scrape time) to have had a fair shot

BLUE, GRAY, DARK = "#2F4BC4", "#8A94A1", "#333333"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left",
})
pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
results = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. Load and clean
# --------------------------------------------------------------------------
def load():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    cols = ["App Id", "Category", "Maximum Installs", "Free", "Price", "Size", "Minimum Android",
            "Developer Id", "Developer Website", "Released", "Content Rating", "Privacy Policy",
            "Ad Supported", "In App Purchases", "Rating Count"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    n_raw = len(df)

    df = df.rename(columns={"Maximum Installs": "installs"})
    df["released"] = pd.to_datetime(df["Released"], format="%b %d, %Y", errors="coerce")
    df = df.dropna(subset=["released", "installs", "Category"])
    df = df.drop_duplicates("App Id")

    # "10M" -> 10.0, "512k" -> 0.5, "Varies with device" -> NaN
    size = df["Size"].astype(str).str.replace(",", "")
    df["size_mb"] = np.where(size.str.endswith("M"), pd.to_numeric(size.str[:-1], errors="coerce"),
                    np.where(size.str.endswith("k"), pd.to_numeric(size.str[:-1], errors="coerce") / 1024,
                             np.nan))
    # "5.0 and up" -> 5.0
    df["min_android"] = pd.to_numeric(df["Minimum Android"].astype(str).str.extract(r"^(\d+\.\d+)")[0],
                                      errors="coerce")
    df["paid"] = ~df["Free"].astype(bool)
    df["ads"] = df["Ad Supported"].astype(bool)
    df["iap"] = df["In App Purchases"].astype(bool)
    df["has_privacy"] = df["Privacy Policy"].notna()
    df["has_website"] = df["Developer Website"].notna()
    df["dev_apps"] = df.groupby("Developer Id")["App Id"].transform("count")
    df["model"] = np.select(
        [df["paid"], df["ads"] & df["iap"], df["ads"], df["iap"]],
        ["Paid upfront", "Ads + in-app purchases", "Ads only", "In-app purchases only"],
        "Free, no ads or purchases")

    keep = ["Category", "installs", "released", "size_mb", "min_android", "Content Rating",
            "paid", "ads", "iap", "has_privacy", "has_website", "dev_apps", "model"]
    df = df[keep]
    print(f"Rows: {n_raw:,} raw -> {len(df):,} apps after cleaning")
    df.to_pickle(CACHE)
    return df


# --------------------------------------------------------------------------
# 2. Concentration
# --------------------------------------------------------------------------
def concentration(df):
    s = np.sort(df["installs"].to_numpy())[::-1]
    total = s.sum()
    share = {p: s[: int(len(s) * p)].sum() / total for p in (0.001, 0.01, 0.10)}
    r = {
        "apps": int(len(s)),
        "median_installs": float(np.median(s)),
        "share_under_1k": float((s < 1_000).mean()),
        "share_100k_plus": float((s >= HIT).mean()),
        "share_1m_plus": float((s >= 1_000_000).mean()),
        "top_0.1pct_share": share[0.001], "top_1pct_share": share[0.01], "top_10pct_share": share[0.10],
    }
    results["concentration"] = r
    print(f"\nMedian app: {r['median_installs']:,.0f} installs. {r['share_under_1k']:.0%} have under 1,000.")
    print(f"Top 0.1% of apps: {share[0.001]:.0%} of all installs | top 1%: {share[0.01]:.0%} | top 10%: {share[0.10]:.0%}")
    print(f"Only {r['share_100k_plus']:.1%} of apps reach 100K installs, {r['share_1m_plus']:.1%} reach 1M")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = ["Top 0.1% of apps", "Top 1%", "Top 10%"]
    vals = [share[0.001], share[0.01], share[0.10]]
    ax.barh(labels[::-1], vals[::-1], color=[GRAY, GRAY, BLUE])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.0%} of all installs", va="center", fontsize=9, color=DARK)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(pct)
    ax.set_title(f"About {round(len(s) * 0.001, -2):,.0f} apps get {share[0.001]:.0%} of every install on the store")
    save(fig, "1_concentration.png")


def cohort(df):
    c = df[df["released"].dt.year.between(*COHORT)].copy()
    c["hit"] = c["installs"] >= HIT
    print(f"\nCohort: {len(c):,} apps released in {COHORT[0]} to {COHORT[1]}, "
          f"{c['hit'].mean():.1%} reached {HIT:,}+ installs")
    results["cohort"] = {"apps": int(len(c)), "hit_rate": float(c["hit"].mean())}
    return c


# --------------------------------------------------------------------------
# 3. Monetization
# --------------------------------------------------------------------------
def monetization(c):
    g = c.groupby("model")["hit"].agg(["size", "mean"]).sort_values("mean")
    results["monetization"] = {k: {"apps": int(r["size"]), "hit_rate": float(r["mean"])} for k, r in g.iterrows()}
    print("\nBreakout rate by monetization model:")
    print(g.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [BLUE if m == "Ads + in-app purchases" else GRAY for m in g.index]
    ax.barh(g.index, g["mean"], color=colors)
    for i, (v, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(v + 0.006, i, f"{v:.1%}  ({n:,} apps)", va="center", fontsize=8.5, color=DARK)
    ax.set_xlim(0, g["mean"].max() * 1.45)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of apps reaching 100K+ installs")
    ax.set_title("Apps that combine ads and in-app purchases break out far more often")
    save(fig, "2_monetization.png")


# --------------------------------------------------------------------------
# 4. Developer experience
# --------------------------------------------------------------------------
def developers(c):
    bins, labels = [0, 1, 5, 20, 100, np.inf], ["1 app", "2 to 5", "6 to 20", "21 to 100", "100+"]
    c = c.assign(port=pd.cut(c["dev_apps"], bins, labels=labels))
    g = c.groupby("port", observed=True)["hit"].agg(["size", "mean"])
    results["developers"] = {k: {"apps": int(r["size"]), "hit_rate": float(r["mean"])} for k, r in g.iterrows()}
    print("\nBreakout rate by how many apps the developer has on the store:")
    print(g.to_string())

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(g.index.astype(str), g["mean"], color=[BLUE if l == "6 to 20" else GRAY for l in g.index])
    for i, v in enumerate(g["mean"]):
        ax.text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=9, color=DARK)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Apps the developer has published")
    ax.set_ylabel("Share reaching 100K+ installs")
    ax.set_title("Experience helps, up to a point")
    save(fig, "3_developers.png")


# --------------------------------------------------------------------------
# 5. Category map
# --------------------------------------------------------------------------
def categories(df, c):
    def top10_share(s):
        s = np.sort(s.to_numpy())[::-1]
        return s[:10].sum() / s.sum()
    cat = pd.DataFrame({
        "top10_share": df.groupby("Category")["installs"].apply(top10_share),
        "hit_rate": c.groupby("Category")["hit"].mean(),
        "cohort_apps": c.groupby("Category").size(),
    }).dropna()
    cat = cat[cat["cohort_apps"] >= 1000]
    cat.sort_values("hit_rate", ascending=False).to_csv(RES / "categories.csv")
    results["categories"] = cat.round(4).to_dict(orient="index")
    print("\nCategories (breakout rate vs. how much the top 10 apps dominate):")
    print(cat.sort_values("hit_rate", ascending=False).round(3).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    open_ = (cat["hit_rate"] >= 0.15) & (cat["top10_share"] < 0.35)
    locked = cat["top10_share"] >= 0.6
    ax.scatter(cat["top10_share"], cat["hit_rate"], s=np.sqrt(cat["cohort_apps"]) * 1.2,
               color=np.where(open_, BLUE, GRAY), alpha=0.85, edgecolor="white", linewidth=0.6)
    show = set(cat[open_].index) | set(cat[locked].index) | {"Business", "Education", "Tools", "Food & Drink"}
    nudge = {"Role Playing": (6, -9), "Communication": (6, -10), "Social": (6, 4),
             "Video Players & Editors": (-10, 6), "Strategy": (-6, -11), "Comics": (6, 4), "Auto & Vehicles": (6, 4)}
    for name, r in cat.iterrows():
        if name in show:
            dx, dy = nudge.get(name, (6, 3))
            ha = "right" if dx < 0 else "left"
            ax.annotate(name, (r["top10_share"], r["hit_rate"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, ha=ha, color=BLUE if open_[name] else DARK)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of the category's installs taken by its top 10 apps")
    ax.set_ylabel("Share of new apps reaching 100K+ installs")
    ax.set_title("Where new apps still have room to break out (blue, top left)")
    save(fig, "4_category_map.png")


# --------------------------------------------------------------------------
# 6. Model: train on 2018 launches, test on 2019 launches
# --------------------------------------------------------------------------
def model(c):
    # only things you'd know at launch. No ratings, reviews or update history, since those come after
    num = ["size_mb", "min_android", "dev_apps"]
    flags = ["paid", "ads", "iap", "has_privacy", "has_website"]
    cats = ["Category", "Content Rating"]
    X = c[num + flags].astype(float).copy()
    for col in cats:
        X[col] = c[col].astype("category")
    y = c["hit"].astype(int)
    train = c["released"].dt.year == 2018
    test = c["released"].dt.year == 2019

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, categorical_features="from_dtype",
                                        random_state=0)
    gb.fit(X[train], y[train])
    p_gb = gb.predict_proba(X[test])[:, 1]

    # simple baseline: logistic regression on one-hot features
    Xd = pd.get_dummies(X, columns=cats, dtype=float)
    Xd[num] = Xd[num].fillna(Xd.loc[train, num].median())
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    lr.fit(Xd[train], y[train])
    p_lr = lr.predict_proba(Xd[test])[:, 1]

    auc_gb, auc_lr = roc_auc_score(y[test], p_gb), roc_auc_score(y[test], p_lr)

    # gain curve: if you could only look at the top X% of apps the model ranks highest,
    # what share of the real hits would you catch?
    order = np.argsort(-p_gb)
    hits = y[test].to_numpy()[order]
    cum = np.cumsum(hits) / hits.sum()
    frac = np.arange(1, len(hits) + 1) / len(hits)
    top10 = float(cum[int(0.10 * len(hits)) - 1])
    lift10 = top10 / 0.10

    results["model"] = {"train_apps": int(train.sum()), "test_apps": int(test.sum()),
                        "test_hit_rate": float(y[test].mean()), "auc_gradient_boosting": float(auc_gb),
                        "auc_logistic": float(auc_lr), "top10pct_captures": top10, "lift_top10pct": lift10}
    print(f"\nModel trained on {train.sum():,} apps from 2018, tested on {test.sum():,} apps from 2019")
    print(f"AUC  gradient boosting: {auc_gb:.3f}   logistic baseline: {auc_lr:.3f}")
    print(f"Top 10% of apps by predicted score contain {top10:.0%} of the real hits ({lift10:.1f}x random)")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    step = max(1, len(frac) // 2000)
    ax.plot(frac[::step], cum[::step], color=BLUE, lw=2.2, label="Model")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.4, ls="--", label="Random guessing")
    ax.scatter([0.10], [top10], color=BLUE, zorder=3)
    ax.annotate(f"Top 10% of listings catch\n{top10:.0%} of the real hits", (0.10, top10),
                xytext=(18, -32), textcoords="offset points", fontsize=9, color=DARK)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of 2019 apps reviewed, highest score first")
    ax.set_ylabel("Share of 100K+ hits found")
    ax.set_title("Trained on 2018 launches, tested on 2019 launches")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "5_model_gain.png")


if __name__ == "__main__":
    df = load()
    concentration(df)
    c = cohort(df)
    monetization(c)
    developers(c)
    categories(df, c)
    model(c)
    (RES / "summary.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nCharts saved to figures/, numbers saved to results/summary.json")
