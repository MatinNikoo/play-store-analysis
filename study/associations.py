"""
Monetization and listing flags against reaching the top 5% of the launch month. These are associations only.
The flags were read in June 2021, so an app that took off may have added in-app purchases afterwards.

This file fits adjusted risk ratios (modified Poisson, errors clustered by developer), E-values, the same ratios
for apps never updated after launch, and a few launch-day profiles scored with the exported milestone model.

Run:  python -m study.associations     (after study.milestones)
"""

import gc

import numpy as np
import pandas as pd

from playodds import model as PM
from study.common import BLUE, DARK, FAST, FEATURES, GRAY, MODELS, RED, SEED, dirs, pct, plt, save, write

REF = "Ads only"
GROUPS = ["Ads + in-app purchases", "In-app purchases only", "Free, no ads or purchases", "Paid upfront"]
DEV_BINS = [-1, 0, 4, 19, 99, np.inf]


COLS = ["Category", "released", "dev", "dev_prior_apps", "model", "has_privacy", "has_website", "iap", "top5",
        "never_updated"]
SLIM = FEATURES.with_name("associations_input.pkl")


def load_cohort():
    """The 2018 to 2019 cohort with only the columns used here, cached so the 760 MB feature table is read once."""
    if not SLIM.exists() or SLIM.stat().st_mtime < FEATURES.stat().st_mtime:
        df = pd.read_pickle(FEATURES)[COLS]
        df = df[df["released"].dt.year.between(2018, 2019)].reset_index(drop=True)
        df["Category"] = df["Category"].astype("category")
        df["model"] = df["model"].astype("category")
        df["dev"] = pd.factorize(df["dev"])[0]
        df.to_pickle(SLIM)
        del df
        gc.collect()
    return pd.read_pickle(SLIM)


def evalue(rr):
    """VanderWeele and Ding E-value for a risk ratio (or the CI limit nearest 1)."""
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def poisson_cluster(X, y, groups, chunk=200_000, iters=30):
    """Poisson regression with log link by IRLS, plus the cluster-robust sandwich covariance (same small
    sample factor as statsmodels). statsmodels keeps several float64 copies of an 815K x 80 design matrix,
    which does not fit in 7 GB, so this does the sums in chunks from a float32 matrix."""
    n, k = X.shape
    beta = np.zeros(k)
    beta[0] = np.log(y.mean())       # column 0 is the constant
    for _ in range(iters):
        A, b = np.zeros((k, k)), np.zeros(k)
        for i in range(0, n, chunk):
            x = X[i:i + chunk].astype(float)
            eta = x @ beta
            mu = np.exp(eta)
            z = eta + (y[i:i + chunk] - mu) / mu
            A += (x * mu[:, None]).T @ x
            b += (x * mu[:, None]).T @ z
        new = np.linalg.solve(A, b)
        done = np.max(np.abs(new - beta)) < 1e-9
        beta = new
        if done:
            break
    A = np.zeros((k, k))
    codes = pd.factorize(groups)[0]
    G = codes.max() + 1
    S = np.zeros((G, k))
    for i in range(0, n, chunk):
        x = X[i:i + chunk].astype(float)
        mu = np.exp(x @ beta)
        A += (x * mu[:, None]).T @ x
        sc = x * (y[i:i + chunk] - mu)[:, None]
        for j in range(k):
            S[:, j] += np.bincount(codes[i:i + chunk], weights=sc[:, j], minlength=G)
    bread = np.linalg.inv(A)
    cov = bread @ (S.T @ S) @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta, cov


def design(c):
    X = pd.DataFrame(index=c.index)
    for g in GROUPS:
        X[g] = (c["model"].astype(str) == g).astype(np.float32)
    X["Privacy policy"] = c["has_privacy"].astype(np.float32)
    X["Developer website"] = c["has_website"].astype(np.float32)
    fe = pd.get_dummies(pd.DataFrame({
        "cat": c["Category"].astype(str), "month": c["released"].dt.to_period("M").astype(str),
        "dev": pd.cut(c["dev_prior_apps"], DEV_BINS).astype(str)}), drop_first=True, dtype=np.float32)
    X = pd.concat([X, fe], axis=1)
    X.insert(0, "const", np.float32(1.0))
    return X


def fit_rr(c, y="top5"):
    X = design(c)
    X = X.loc[:, X.std() > 0].copy()
    X["const"] = np.float32(1.0)
    X = X[["const"] + [k for k in X.columns if k != "const"]]       # constant first, as poisson_cluster expects
    beta, cov = poisson_cluster(X.to_numpy(), c[y].to_numpy(float), c["dev"].to_numpy())
    out = {}
    names = list(X.columns)
    for k in GROUPS + ["Privacy policy", "Developer website"]:
        if k not in names:
            continue
        j = names.index(k)
        if c.loc[X[k].to_numpy() == 1, y].sum() < 5:
            out[k] = {"rr": None, "note": "fewer than 5 breakout apps in this group", "apps": int(X[k].sum())}
            continue
        b, se = beta[j], np.sqrt(cov[j, j])
        rr, lo, hi = np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        near = 1.0 if lo <= 1 <= hi else (lo if rr > 1 else hi)
        out[k] = {"rr": float(rr), "lo": float(lo), "hi": float(hi), "e_value": evalue(rr),
                  "e_value_ci": 1.0 if near == 1.0 else evalue(near), "apps": int(X[k].sum())}
    del X
    gc.collect()
    return out


def main():
    dirs()
    c = load_cohort()
    res = {"population": "apps launched 2018-2019", "apps": len(c), "outcome": "top 5% of launch month",
           "reference": REF}

    raw = c.groupby(c["model"].astype(str))["top5"].agg(["size", "mean"])
    res["raw_rates"] = {k: {"apps": int(v["size"]), "rate": float(v["mean"])} for k, v in raw.iterrows()}
    res["adjusted_all"] = fit_rr(c)
    print("all done")

    # timing: listing closest to launch day (never updated) vs updated apps
    res["never_updated_share"] = {"top5": float(c.loc[c["top5"] == 1, "never_updated"].mean()),
                                  "rest": float(c.loc[c["top5"] == 0, "never_updated"].mean())}
    nu = c[c["never_updated"] == 1]
    up = c[c["never_updated"] == 0]
    res["never_updated_apps"] = len(nu)
    res["adjusted_never_updated"] = fit_rr(nu)
    res["adjusted_updated"] = fit_rr(up)
    res["iap_share_by_update"] = {
        "never_updated_top5": float(nu.loc[nu["top5"] == 1, "iap"].mean()),
        "updated_top5": float(up.loc[up["top5"] == 1, "iap"].mean()),
        "never_updated_rest": float(nu.loc[nu["top5"] == 0, "iap"].mean()),
        "updated_rest": float(up.loc[up["top5"] == 0, "iap"].mean())}
    print("timing done")

    res["playbook"] = playbook()
    write("associations.json", res)
    figures(res)
    for k in ["adjusted_all", "adjusted_never_updated", "adjusted_updated"]:
        print(k)
        for g, v in res[k].items():
            if v["rr"] is None:
                continue
            print(f"  {g:28s} RR {v['rr']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]  E {v['e_value']:.2f} "
                  f"(CI {v['e_value_ci']:.2f})")


# --------------------------------------------------------------------------
# playbook: launch-time profiles scored with the exported model
# --------------------------------------------------------------------------
PROFILES = [
    ("First app, Tools", {"category": "Tools", "name": "Battery Saver"}),
    ("Same, 5th to 20th app", {"category": "Tools", "name": "Battery Saver", "earlier_apps": 8,
                               "months_since_first_app": 24, "earlier_categories": 2,
                               "earlier_apps_same_category": 3}),
    ("Same, 101st app or later", {"category": "Tools", "name": "Battery Saver", "earlier_apps": 150,
                                  "months_since_first_app": 36, "earlier_categories": 6,
                                  "earlier_apps_same_category": 40, "apps_last_30_days": 10}),
    ("First app, 10 more the same day", {"category": "Tools", "name": "Battery Saver", "same_day_apps": 10}),
    ("First app on an app builder", {"category": "Tools", "name": "Battery Saver", "builder_apps": 5000}),
    ("First app, Simulation", {"category": "Simulation", "name": "Truck Driver"}),
    ("First app, Business", {"category": "Business", "name": "Invoice Maker"}),
    ("First app, Education", {"category": "Education", "name": "Math Practice"}),
]


def playbook():
    model = PM.load(MODELS / "milestones.json")
    out = []
    for label, app in PROFILES:
        if app["category"] not in model["categories"]:
            continue
        s = PM.score(app, months=12, model=model, level="90")
        out.append({"profile": label, "app": app, **{k: v for k, v in s["milestones"].items()}})
    return out


def figures(res):
    # adjusted risk ratios, all apps vs never-updated apps
    fig, ax = plt.subplots(figsize=(8, 4.4))
    keys = [k for k in GROUPS + ["Privacy policy", "Developer website"]
            if res["adjusted_all"].get(k, {}).get("rr") and res["adjusted_never_updated"].get(k, {}).get("rr")]
    y = np.arange(len(keys))
    for off, block, col, lab in [(-0.17, "adjusted_all", BLUE, "All 2018-2019 apps"),
                                 (0.17, "adjusted_never_updated", RED, "Never updated after launch")]:
        r = res[block]
        v = [r[k]["rr"] for k in keys]
        err = [[r[k]["rr"] - r[k]["lo"] for k in keys], [r[k]["hi"] - r[k]["rr"] for k in keys]]
        ax.errorbar(v, y + off, xerr=err, fmt="o", color=col, ms=5, capsize=2, label=lab)
    ax.axvline(1, color=DARK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yticks(y, [k if k in ["Privacy policy", "Developer website"] else f"{k} (vs ads only)" for k in keys],
                  fontsize=8.5)
    ax.invert_yaxis()
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.set_xlabel("Adjusted ratio of top 5% rates (category, launch month, developer history)")
    ax.set_title("2021 listing flags and breakout (associations only)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "15_associations.png")

    # playbook profiles
    pb = res["playbook"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(pb))
    v = [p["10000"]["p"] for p in pb]
    err = [[p["10000"]["p"] - p["10000"]["lo"] for p in pb], [p["10000"]["hi"] - p["10000"]["p"] for p in pb]]
    ax.errorbar(v, y, xerr=err, fmt="o", color=BLUE, ms=6, capsize=3)
    for i, p in enumerate(pb):
        ax.text(p["10000"]["p"], i - 0.28, f"{p['10000']['p']:.1%}", ha="center", fontsize=8, color=DARK)
    ax.set_yticks(y, [p["profile"] for p in pb], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlim(0, min(1.0, max(p["10000"]["hi"] for p in pb) * 1.2))
    ax.set_xlabel("Model's chance of 10K installs within 12 months (bars show parameter uncertainty only)")
    ax.set_title("Example launch profiles scored by the milestone model")
    save(fig, "16_playbook.png")


if __name__ == "__main__":
    main()
