"""
Ranks new launches using only launch-time information, and compares that with the old model on the same split.
The target is the top 5% of installs among apps released the same month. Also calibration, stability, subgroups,
distribution shift, the leakage ablation and an error analysis.

Run:  python -m study.ranking        (about an hour on 2 cores, it is resumable, see --fresh and --eval)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from study import features as F
from study import metrics as M
from study.common import BLUE, DARK, FAST, FEATURES, GRAY, LIGHT, RED, SEED, dirs, pct, plt, save, write

PREDS = Path(str(FEATURES).replace(".pkl", "_ranking_preds.pkl"))
B = 50 if FAST else 500
SUB_FRAC = 0.4      # ablation and sensitivity fits use 40% of training developers to keep the runtime sane
SEED_FRAC = 0.5     # each seed refit uses a different half of the training developers
DEV_BINS = [-1, 0, 4, 19, 99, np.inf]
DEV_LABELS = ["First app", "1 to 4 earlier", "5 to 19 earlier", "20 to 99 earlier", "100+ earlier"]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def matrix(c, feats):
    X = pd.DataFrame(index=c.index)
    for f in feats:
        X[f] = c[f].astype("category") if f in F.CATEGORICAL else c[f].astype(float)
    return X


def gbm(seed=SEED):
    return HistGradientBoostingClassifier(
        max_iter=40 if FAST else 250, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        categorical_features="from_dtype", random_state=seed)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def logit_model(X):
    cats = [f for f in X.columns if f in F.CATEGORICAL]
    binary = [f for f in X.columns if f not in cats and X[f].nunique() <= 2]
    nums = [f for f in X.columns if f not in cats and f not in binary]
    spline = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(slog),
                           SplineTransformer(n_knots=5, degree=3, knots="quantile"), StandardScaler())
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ("num", spline, nums)])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000))


def fit_predict(kind, train, tests, feats, y, seed=SEED):
    Xtr = matrix(train, feats)
    m = gbm(seed) if kind == "gbm" else logit_model(Xtr)
    m.fit(Xtr, train[y].to_numpy())
    return [m.predict_proba(matrix(t, feats))[:, 1] for t in tests]


def launch_fit(kind, train, tests, feats, y, seed=SEED):
    F.check_launch_only(feats)     # leakage guard: refuses anything not known on launch day
    return fit_predict(kind, train, tests, feats, y, seed)


def lookup(train, tests, keys, y, m=50):
    """Baseline: training rate for each cell of `keys`, shrunk toward the overall rate."""
    base = train[y].mean()
    g = train.groupby(keys, observed=True)[y].agg(["sum", "size"])
    rate = (g["sum"] + m * base) / (g["size"] + m)
    out = []
    for t in tests:
        idx = pd.MultiIndex.from_frame(t[keys]) if len(keys) > 1 else t[keys[0]]
        out.append(rate.reindex(idx).fillna(base).to_numpy())
    return out


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------
def summary(y, p, base):
    cal = M.calibration(y, p)
    return {"auc": M.auc(y, p), "brier": M.brier(y, p), "brier_skill": M.brier_skill(y, p, base),
            "cal_intercept": cal["intercept"], "cal_slope": cal["slope"], "ece": M.ece(y, p),
            "top10_captures": M.captured(y, p, 0.10), "mean_pred": float(np.mean(p)), "rate": float(np.mean(y))}


def by_group(test, y, preds, col, min_pos=20):
    rows = []
    for g, d in test.groupby(col, observed=True):
        if d[y].sum() < min_pos or (1 - d[y]).sum() < min_pos:
            continue
        r = {"group": str(g), "apps": len(d), "rate": float(d[y].mean())}
        for k, p in preds.items():
            pp = p[d.index.to_numpy()]
            r[f"auc_{k}"] = M.auc(d[y], pp)
            r[f"mean_pred_{k}"] = float(pp.mean())
        rows.append(r)
    return rows


def setup():
    df = pd.read_pickle(FEATURES)
    df["top1"] = F.top_in_month(df, 0.01)
    df["top10"] = F.top_in_month(df, 0.10)
    df["dev_bin"] = pd.cut(df["dev_prior_apps"], DEV_BINS, labels=DEV_LABELS)
    train, test, later = split(df)
    shift_tests = {yr: df[df["released"].dt.year == yr].reset_index(drop=True) for yr in (2018, 2019, 2020)}
    res = {"split": {"train": "released 2018", "test": "released 2019", "train_apps": len(train),
                     "test_apps": len(test), "shared_developers": int(len(set(train["dev"]) & set(test["dev"]))),
                     "test_developers": int(test["dev"].nunique()),
                     "ablation_train_apps": len(dev_sample(train, SUB_FRAC, SEED + 10))}}
    return df, train, test, later, shift_tests, res


def split(df, train_years=(2018,), test_year=2019, later_year=2020):
    """Temporal split by launch date. Developers can show up on both sides (their history is
    launch-time information), apps can't."""
    year = df["released"].dt.year
    train = df[year.isin(train_years)].reset_index(drop=True)
    test = df[year == test_year].reset_index(drop=True)
    later = df[year == later_year].reset_index(drop=True)
    return train, test, later


def dev_sample(train, frac, seed):
    devs = train["dev"].unique()
    keep = set(np.random.default_rng(seed).choice(devs, int(frac * len(devs)), replace=False))
    return train[train["dev"].isin(keep)].reset_index(drop=True)


def main(fresh=False):
    dirs()
    df, train, test, later, shift_tests, res = setup()
    year = df["released"].dt.year
    # predictions are saved after every fit, so a stopped run picks up where it left off
    P = {} if fresh or not PREDS.exists() else pickle.load(open(PREDS, "rb"))

    def run(keys, fn):
        if all(k in P for k in keys):
            return
        for k, v in zip(keys, fn()):
            P[k] = v
        pickle.dump(P, open(PREDS, "wb"))
        print("done", keys[0], flush=True)

    # ---- main comparison, primary outcome (top 5% of launch month) ----
    y = "top5"
    tests = [test, later]
    run(["gbm_launch", "gbm_launch_2020"], lambda: launch_fit("gbm", train, tests, F.LAUNCH, y))
    run(["gbm_strict"], lambda: launch_fit("gbm", train, [test], F.STRICT, y))
    run(["logit_launch"], lambda: launch_fit("logit", train, [test], F.LAUNCH, y))
    run(["gbm_old", "gbm_old_2020"], lambda: fit_predict("gbm", train, tests, F.OLD, y))
    run(["logit_old"], lambda: fit_predict("logit", train, [test], F.OLD, y))
    run(["base_category"], lambda: lookup(train, [test], ["Category"], y))
    run(["base_dev"], lambda: lookup(train, [test], ["dev_bin"], y))
    run(["base_category_dev"], lambda: lookup(train, [test], ["Category", "dev_bin"], y))

    # ---- old outcome (100K by June 2021) for reference ----
    run(["gbm_old_100k"], lambda: fit_predict("gbm", train, [test], F.OLD, "hit100k"))
    run(["gbm_launch_100k"], lambda: launch_fit("gbm", train, [test], F.LAUNCH, "hit100k"))

    # ---- leakage ablation on a 40% developer sample (both reference models refit on the same sample) ----
    sub = dev_sample(train, SUB_FRAC, SEED + 10)
    ablation = {
        "old features (all)": F.OLD,
        "old minus monetization": [f for f in F.OLD if f not in F.MONETIZATION],
        "old minus monetization and listing": [f for f in F.OLD if f not in F.MONETIZATION + F.LISTING],
        "old minus all snapshot fields": [f for f in F.OLD if f not in F.SNAPSHOT],
        "monetization only": ["paid", "ads", "iap"],
        "launch features": F.LAUNCH,
        "launch minus title": [f for f in F.LAUNCH if not f.startswith("name_")],
        "launch minus category": [f for f in F.LAUNCH if f != "Category"],
        "launch + monetization": F.LAUNCH + ["paid", "ads", "iap"],
        "launch + all snapshot fields": F.LAUNCH + F.SNAPSHOT,
        "launch + earlier apps' 2021 success": F.LAUNCH + F.LEAKY_DEV,
        "launch + post-launch fields": F.LAUNCH + F.POST,
    }
    for name, feats in ablation.items():
        run(["abl:" + name], lambda: fit_predict("gbm", sub, [test], feats, y))

    # ---- outcome sensitivity: top 1% and top 10% of the launch month (same sample) ----
    for k in ["top1", "top10"]:
        run([f"gbm_launch_{k}"], lambda: launch_fit("gbm", sub, [test], F.LAUNCH, k))
        run([f"gbm_old_{k}"], lambda: fit_predict("gbm", sub, [test], F.OLD, k))

    # ---- seeds: refit on a different half of the training developers each time ----
    for s in range(1, 4):
        half = dev_sample(train, SEED_FRAC, s)
        run([f"seed{s}_launch"], lambda: launch_fit("gbm", half, [test], F.LAUNCH, y, seed=s))
        run([f"seed{s}_old"], lambda: fit_predict("gbm", half, [test], F.OLD, y, seed=s))

    # ---- distribution shift: train on 2016-2017, test on each later year ----
    old_train = df[year.between(2016, 2017)].reset_index(drop=True)
    run([f"shift_launch_{yr}" for yr in shift_tests],
        lambda: launch_fit("gbm", old_train, list(shift_tests.values()), F.LAUNCH, y))
    run([f"shift_old_{yr}" for yr in shift_tests],
        lambda: fit_predict("gbm", old_train, list(shift_tests.values()), F.OLD, y))
    P["shift_rate_train"] = old_train[y].mean()
    pickle.dump(P, open(PREDS, "wb"))
    print("all fits done", flush=True)
    evaluate(df, train, test, later, shift_tests, P, res)


def evaluate(df, train, test, later, shift_tests, P, res):
    y = "top5"
    yt = test[y].to_numpy()
    base = train[y].mean()

    # main table with developer-clustered bootstrap CIs and paired differences vs the launch model
    names = ["gbm_launch", "logit_launch", "gbm_strict", "base_category_dev", "base_category", "base_dev",
             "gbm_old", "logit_old"]
    boot = M.bootstrap_auc(yt, {k: P[k] for k in names}, test["dev"], B=B, seed=SEED)
    res["main"] = {k: {**summary(yt, P[k], base), **boot[k]} for k in names}
    res["main_100k"] = {}
    yo = test["hit100k"].to_numpy()
    for k in ["gbm_old_100k", "gbm_launch_100k"]:
        r = summary(yo, P[k], train["hit100k"].mean())
        r["within_month_auc"] = M.stratified_auc(yo, P[k], test["released"].dt.month)
        res["main_100k"][k] = r
    # top 5% outcome scored inside each month too, so it's clear the horizon isn't driving it
    res["main"]["gbm_launch"]["within_month_auc"] = M.stratified_auc(yt, P["gbm_launch"], test["released"].dt.month)
    res["main"]["gbm_old"]["within_month_auc"] = M.stratified_auc(yt, P["gbm_old"], test["released"].dt.month)

    # the live simulator's population (known size, 4 main content ratings) gets the same comparison
    sim = test["Content Rating"].isin(["Everyone", "Everyone 10+", "Teen", "Mature 17+"]) & test["size_mb"].notna()
    res["simulator_population_test"] = {"apps": int(sim.sum()),
                                        "auc_gbm_launch": M.auc(yt[sim], P["gbm_launch"][sim]),
                                        "auc_gbm_old": M.auc(yt[sim], P["gbm_old"][sim])}

    # ablation
    abl_boot = M.bootstrap_auc(yt, {k[4:]: v for k, v in P.items() if k.startswith("abl:")},
                               test["dev"], B=max(20, B // 5), seed=SEED + 1)
    res["ablation"] = {k: v["auc"] for k, v in abl_boot.items()}
    res["ablation_ci"] = abl_boot
    gap = res["ablation"]["old features (all)"] - res["ablation"]["launch features"]
    old_lift = res["ablation"]["old features (all)"] - 0.5
    res["launch_gap_share_of_old_auc_above_half"] = gap / old_lift
    # how much of the old model's lift over chance sat in fields only seen in the 2021 snapshot
    res["snapshot_share_of_old_auc_above_half"] = (res["ablation"]["old features (all)"]
                                                   - res["ablation"]["old minus all snapshot fields"]) / old_lift

    res["outcome_sensitivity"] = {}
    for k in ["top1", "top10"]:
        yk = test[k].to_numpy()
        res["outcome_sensitivity"][k] = {"rate": float(yk.mean()), "auc_launch": M.auc(yk, P[f"gbm_launch_{k}"]),
                                         "auc_old": M.auc(yk, P[f"gbm_old_{k}"])}

    seeds = {m: [M.auc(yt, P[f"seed{s}_{m}"]) for s in range(1, 4)] for m in ["launch", "old"]}
    res["seeds"] = {m: {"aucs": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                        "min": float(np.min(v)), "max": float(np.max(v))} for m, v in seeds.items()}

    # subgroups
    test = test.assign(quarter="2019 Q" + test["released"].dt.quarter.astype(str))
    pr = {"launch": P["gbm_launch"], "old": P["gbm_old"]}
    res["by_category"] = by_group(test, y, pr, "Category")
    res["by_developer"] = by_group(test, y, pr, "dev_bin")
    res["by_quarter"] = by_group(test, y, pr, "quarter")
    res["by_game"] = by_group(test.assign(kind=np.where(test["game"] == 1, "Games", "Apps")), y, pr, "kind")

    # distribution shift
    res["shift"] = {"train": "released 2016-2017", "train_rate": float(P["shift_rate_train"])}
    for yr, t in shift_tests.items():
        yy = t[y].to_numpy()
        res["shift"][str(yr)] = {m: {"auc": M.auc(yy, P[f"shift_{m}_{yr}"]), **M.calibration(yy, P[f"shift_{m}_{yr}"]),
                                     "mean_pred": float(P[f"shift_{m}_{yr}"].mean())}
                                 for m in ["launch", "old"]}
        res["shift"][str(yr)]["rate"] = float(yy.mean())
    yl = later[y].to_numpy()
    res["shift"]["train_2018_test_2020"] = {"launch": M.auc(yl, P["gbm_launch_2020"]), "old": M.auc(yl, P["gbm_old_2020"])}

    res["reliability"] = {k: M.reliability(yt, P[k], 10) for k in ["gbm_launch", "gbm_old", "logit_launch"]}
    res["errors"] = errors(test, P["gbm_launch"], y)
    write("ranking.json", res)
    figures(res, test, P)
    print_summary(res)


# --------------------------------------------------------------------------
# error analysis
# --------------------------------------------------------------------------
def errors(test, p, y):
    t = test.assign(p=p)
    cut = np.quantile(p, 0.90)
    t["flagged"] = t["p"] >= cut
    hits = t[t[y] == 1]
    caught = hits[hits["flagged"]]
    missed = hits[t.loc[hits.index, "p"] < np.median(p)]
    false_alarm = t[t["flagged"] & (t[y] == 0)]

    def profile(d):
        return {"apps": len(d), "first_app": float((d["dev_prior_apps"] == 0).mean()),
                "prolific_100plus": float((d["dev_prior_apps"] >= 100).mean()),
                "games": float(d["game"].mean()), "never_updated": float(d["never_updated"].mean()),
                "has_iap_2021": float(d["iap"].mean()), "has_ads_2021": float(d["ads"].mean()),
                "bulk_same_day_5plus": float((d["dev_same_day"] >= 5).mean()),
                "builder_prefix_1000plus": float((d["pkg_prior"] >= 1000).mean()),
                "median_installs": float(d["installs"].median())}

    out = {"flag_cutoff_score": float(cut), "caught_hits": profile(caught), "missed_hits": profile(missed),
           "false_alarms": profile(false_alarm), "all_test": profile(t)}
    out["recall_top10_by_developer"] = {str(g): float(d["flagged"].mean())
                                        for g, d in hits.groupby("dev_bin", observed=True)}
    out["recall_top10_by_kind"] = {"Games": float(hits.loc[hits["game"] == 1, "flagged"].mean()),
                                   "Apps": float(hits.loc[hits["game"] == 0, "flagged"].mean())}
    out["recall_top10_by_iap_2021"] = {"In-app purchases in 2021": float(hits.loc[hits["iap"], "flagged"].mean()),
                                       "No in-app purchases in 2021": float(hits.loc[~hits["iap"], "flagged"].mean())}
    out["recall_top10_overall"] = float(hits["flagged"].mean())
    out["precision_top10"] = float(t.loc[t["flagged"], y].mean())
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figures(res, test, P):
    # reliability: old vs launch-time model
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for k, col, lab in [("gbm_old", GRAY, "Old model (2021 snapshot fields)"), ("gbm_launch", BLUE, "Launch-time model")]:
        r = res["reliability"][k]
        x = [b["pred"] for b in r]
        o = [b["obs"] for b in r]
        err = [[b["obs"] - b["lo"] for b in r], [b["hi"] - b["obs"] for b in r]]
        ax.errorbar(x, o, yerr=err, color=col, marker="o", ms=4, lw=1.6, capsize=2, label=lab)
    m = max(max(b["pred"] for b in res["reliability"][k]) for k in ["gbm_old", "gbm_launch"]) * 1.08
    ax.plot([0, m], [0, m], color=DARK, lw=0.8, ls=":", label="Perfect calibration")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Predicted chance of a top 5% launch (deciles of 2019 apps)")
    ax.set_ylabel("Share that actually made the top 5%")
    L, O = res["main"]["gbm_launch"], res["main"]["gbm_old"]
    ax.set_title(f"Calibration on 2019 launches (slope {L['cal_slope']:.2f} launch, {O['cal_slope']:.2f} old)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "6_calibration.png")

    # leakage ablation
    a = res["ablation_ci"]
    rows = [("Old model, all 2021 fields", a["old features (all)"]["auc"], a["old features (all)"]),
            ("Old minus monetization", a["old minus monetization"]["auc"], a["old minus monetization"]),
            ("Old minus monetization and listing", a["old minus monetization and listing"]["auc"],
             a["old minus monetization and listing"]),
            ("Old minus every snapshot field", a["old minus all snapshot fields"]["auc"], a["old minus all snapshot fields"]),
            ("Launch-time model", a["launch features"]["auc"], a["launch features"]),
            ("Launch minus title", a["launch minus title"]["auc"], a["launch minus title"]),
            ("Launch minus category", a["launch minus category"]["auc"], a["launch minus category"]),
            ("Launch + monetization (leaky)", a["launch + monetization"]["auc"], a["launch + monetization"]),
            ("Launch + every snapshot field (leaky)", a["launch + all snapshot fields"]["auc"],
             a["launch + all snapshot fields"]),
            ("Launch + earlier apps' 2021 installs (leaky)", a["launch + earlier apps' 2021 success"]["auc"],
             a["launch + earlier apps' 2021 success"]),
            ("Launch + ratings and updates (leaky)", a["launch + post-launch fields"]["auc"], a["launch + post-launch fields"])]
    fig, ax = plt.subplots(figsize=(9, 5.3))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [BLUE if l.startswith("Launch-time") else (RED if "leaky" in l else (LIGHT if l.startswith("Launch minus") else GRAY)) for l in labels]
    ax.barh(labels, [v - 0.5 for v in vals], left=0.5, color=colors, height=0.62)
    for i, (v, r) in enumerate(zip(vals, [r[2] for r in rows][::-1])):
        txt = f"{v:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]"
        ax.text(v + 0.004, i, txt, va="center", fontsize=8, color=DARK)
    ax.set_xlim(0.5, 1.15)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUC on 2019 launches, top 5% of launch month (95% CI)")
    ax.set_title("Where the old AUC came from")
    save(fig, "7_leakage_ablation.png")

    # distribution shift
    s = res["shift"]
    yrs = ["2018", "2019", "2020"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for m, col, lab in [("old", GRAY, "Old model"), ("launch", BLUE, "Launch-time model")]:
        axes[0].plot(yrs, [s[y][m]["auc"] for y in yrs], marker="o", color=col, lw=2, label=lab)
        axes[1].plot(yrs, [s[y][m]["mean_pred"] for y in yrs], marker="o", color=col, lw=2, label=lab)
    axes[1].plot(yrs, [s[y]["rate"] for y in yrs], marker="s", color=DARK, lw=1.2, ls="--", label="Actual rate")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC by launch year")
    axes[1].yaxis.set_major_formatter(pct)
    axes[1].set_ylim(0, max(0.1, max(s[y]["old"]["mean_pred"] for y in yrs) * 1.3))
    axes[1].set_ylabel("Mean predicted vs actual")
    axes[1].set_title("Average prediction vs actual")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Launch year (models trained on 2016-2017 launches)")
    save(fig, "8_shift.png")

    # error analysis
    e = res["errors"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    rd = e["recall_top10_by_developer"]
    axes[0].bar([k.replace(" earlier", "\nearlier") for k in rd], list(rd.values()), color=[BLUE if v == min(rd.values()) else GRAY for v in rd.values()])
    for i, v in enumerate(rd.values()):
        axes[0].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9, color=DARK)
    axes[0].axhline(e["recall_top10_overall"], color=DARK, lw=0.8, ls=":")
    axes[0].yaxis.set_major_formatter(pct)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Share of real top 5% apps in the model's top 10%")
    axes[0].set_title("Share of hits in the model's top 10%, by developer history")
    groups = ["first_app", "games", "never_updated", "has_iap_2021", "prolific_100plus", "bulk_same_day_5plus"]
    labs = ["First app", "Game", "Never updated", "In-app purchases (2021)", "Developer had 100+ earlier apps",
            "5+ apps released the same day"]
    yy = np.arange(len(groups))
    h = 0.38
    ax = axes[1]
    h = 0.27
    ax.barh(yy - h, [e["missed_hits"][g] for g in groups], h, color=RED, label="Missed hits (bottom half of scores)")
    ax.barh(yy, [e["false_alarms"][g] for g in groups], h, color=GRAY, label="False alarms (top 10%, not a hit)")
    ax.barh(yy + h, [e["all_test"][g] for g in groups], h, color=LIGHT, label="All 2019 launches")
    ax.set_yticks(yy, labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Share of the group")
    ax.set_title("What the mistakes look like")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "9_errors.png")


def print_summary(res):
    print("\nMain comparison, 2019 launches, top 5% of launch month")
    for k, v in res["main"].items():
        print(f"  {k:20s} AUC {v['auc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]  Brier {v['brier']:.4f}  "
              f"slope {v['cal_slope']:.2f}  int {v['cal_intercept']:+.2f}  top10 {v['top10_captures']:.0%}")
    print("Ablation:")
    for k, v in res["ablation"].items():
        print(f"  {k:40s} {v:.3f}")
    print("Seeds:", {k: (round(v['mean'], 3), round(v['sd'], 4)) for k, v in res["seeds"].items()})
    print("Shift:", {y: (round(res['shift'][y]['launch']['auc'], 3), round(res['shift'][y]['old']['auc'], 3))
                     for y in ["2018", "2019", "2020"]})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        # re-score saved predictions without refitting
        dirs()
        df, train, test, later, shift_tests, res = setup()
        evaluate(df, train, test, later, shift_tests, pickle.load(open(PREDS, "rb")), res)
    else:
        main(fresh="--fresh" in sys.argv)
