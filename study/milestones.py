"""
How long does a new app take to reach 10K, 100K and 1M installs?

The scrape is one snapshot, so each app is seen once at its age in June 2021, with a yes or no for whether it has
crossed the milestone (current status data). This file fits the NPMLE of the time to milestone with subsampling
intervals, fits current status GLMs on launch-day inputs, validates them forward in time and on held-out launch
months, builds conformal intervals for the launch-month percentile, and exports models/milestones.json for the
playodds package.

Run:  python -m study.milestones     (about 15 minutes on 2 cores, peak memory about 5.3 GB)
"""

import gc
import gzip
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from playodds import model as PM
from study import features as F
from study import metrics as M
from study.common import BLUE, DARK, FAST, FEATURES, GRAY, LIGHT, MODELS, RED, SEED, dirs, pct, plt, save, write

MILESTONES = [10_000, 100_000, 1_000_000]
NAMES = {10_000: "10K", 100_000: "100K", 1_000_000: "1M"}
START, END = "2016-01-01", "2021-06-01"      # launches with at least two weeks before the scrape
CUTOFF = "2019-06-01"                          # forward validation: fit before this, test after
FIT_N = 2_500 if FAST else 400_000              # apps per GLM fit (a random sample, to stay polite on memory)
LINKS = {"cloglog": sm.families.links.CLogLog, "probit": sm.families.links.Probit, "logit": sm.families.links.Logit}
MONTH = 30.44
EVAL_MONTHS = [6, 12, 24, 36, 48, 60]
LEVELS = {"50": 0.50, "80": 0.80, "90": 0.90}
Z = {"50": 0.6745, "80": 1.2816, "90": 1.6449}

# which availability-table feature each scorer input comes from (checked by the leakage guard test)
INPUT_SOURCE = {"category": "Category", "earlier_apps": "dev_prior_apps", "same_day_apps": "dev_same_day",
                "apps_last_30_days": "dev_prior_30d", "months_since_first_app": "dev_days_active",
                "earlier_categories": "dev_prior_cats", "earlier_apps_same_category": "dev_prior_same_cat",
                "builder_apps": "pkg_prior", "name_len": "name_len", "name_words": "name_words",
                "name_nonascii": "name_nonascii", "name_digit": "name_digit"}


def inputs(df):
    return pd.DataFrame({
        "category": df["Category"].astype(str).to_numpy(), "name": df["name"].to_numpy(),
        "earlier_apps": df["dev_prior_apps"].to_numpy(), "same_day_apps": df["dev_same_day"].to_numpy(),
        "apps_last_30_days": df["dev_prior_30d"].to_numpy(),
        "months_since_first_app": df["dev_days_active"].to_numpy() / MONTH,
        "earlier_categories": df["dev_prior_cats"].to_numpy(),
        "earlier_apps_same_category": df["dev_prior_same_cat"].to_numpy(),
        "builder_apps": df["pkg_prior"].to_numpy()}, index=df.index)


def make_terms(categories, ref):
    terms = [{"name": f"category={c}", "input": "category", "kind": "onehot", "level": c}
             for c in categories if c != ref]
    for lo, hi in [(1, 1), (2, 4), (5, 19), (20, 99), (100, 1e9)]:
        terms.append({"name": f"earlier_apps={lo}-{int(hi) if hi < 1e9 else 'up'}", "input": "earlier_apps",
                      "kind": "bin", "lo": lo, "hi": hi})
    terms += [
        {"name": "log1p(same_day_apps)", "input": "same_day_apps", "kind": "log1p", "cap": 400},
        {"name": "log1p(apps_last_30_days)", "input": "apps_last_30_days", "kind": "log1p", "cap": 400},
        {"name": "log1p(months_since_first_app)", "input": "months_since_first_app", "kind": "log1p", "cap": 200},
        {"name": "earlier_categories", "input": "earlier_categories", "kind": "linear", "scale": 1, "cap": 20},
        {"name": "log1p(earlier_apps_same_category)", "input": "earlier_apps_same_category", "kind": "log1p",
         "cap": 10000},
        {"name": "log1p(builder_apps)", "input": "builder_apps", "kind": "log1p", "cap": 50000},
        {"name": "name_len/10", "input": "name_len", "kind": "linear", "scale": 10, "cap": 50},
        {"name": "name_words", "input": "name_words", "kind": "linear", "scale": 1, "cap": 16},
        {"name": "name_nonascii", "input": "name_nonascii", "kind": "linear", "scale": 1, "cap": 1},
        {"name": "name_digit", "input": "name_digit", "kind": "linear", "scale": 1, "cap": 1},
    ]
    return terms


# --------------------------------------------------------------------------
# NPMLE for current status data
# --------------------------------------------------------------------------
def pava(y, w):
    """Weighted isotonic (non-decreasing) regression by pool adjacent violators."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            sizes.append(s1 + s2)
    return np.repeat(vals, sizes)


def npmle(age, delta):
    """NPMLE of F(t) = P(T <= t) from current status data (age at observation, crossed yet).
    Returns the distinct ages and F at each one."""
    t = pd.DataFrame({"a": np.asarray(age), "d": np.asarray(delta, float)}).groupby("a")["d"].agg(["sum", "size"])
    return t.index.to_numpy(), pava((t["sum"] / t["size"]).to_numpy(), t["size"].to_numpy().astype(float))


def f_at(ages, F_, t):
    i = np.searchsorted(ages, t, side="right") - 1
    return np.where(i >= 0, F_[np.clip(i, 0, None)], 0.0)


def npmle_ci(age, delta, groups, points, B, frac=0.1, seed=SEED):
    """Cube-root subsampling CIs (Politis and Romano): resample whole developers without replacement."""
    a0, F0 = npmle(age, delta)
    est = f_at(a0, F0, points)
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    G = codes.max() + 1
    rng = np.random.default_rng(seed)
    n = len(age)
    draws = []
    for _ in range(B):
        keep = np.zeros(G, bool)
        keep[rng.choice(G, int(frac * G), replace=False)] = True
        m = keep[codes]
        a, F_ = npmle(age[m], delta[m])
        draws.append(m.sum() ** (1 / 3) * (f_at(a, F_, points) - est))
    q = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return est, np.clip(est - q[1] / n ** (1 / 3), 0, 1), np.clip(est - q[0] / n ** (1 / 3), 0, 1)


# --------------------------------------------------------------------------
# parametric current status GLMs
# --------------------------------------------------------------------------
def design(X, age_m):
    return np.column_stack([np.ones(len(age_m)), np.log(age_m), X])


def fit_glm(X, age_m, d, link, groups=None):
    fam = sm.families.Binomial(link=LINKS[link]())
    mod = sm.GLM(d, design(X.astype(float), age_m), family=fam)
    if groups is None:
        r = mod.fit(maxiter=100)
    else:
        r = mod.fit(maxiter=100, cov_type="cluster", cov_kwds={"groups": groups})
    r.remove_data()           # statsmodels keeps several copies of the design matrix otherwise
    del mod
    gc.collect()
    return r


def predict(r, X, age_m, link):
    p = np.asarray(r.params)
    return PM.inv_link(p[0] + p[1] * np.log(age_m) + X @ p[2:].astype(X.dtype), link)


def as_export(r):
    p = np.asarray(r.params)
    return {"intercept": float(p[0]), "log_time": float(p[1]), "coef": p[2:].tolist(),
            "cov": np.asarray(r.cov_params()).tolist()}


def eval_block(d, p, months):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = -np.mean(d * np.log(p) + (1 - d) * np.log(1 - p))
    cal = M.calibration(d, p)
    return {"log_loss": float(ll), "brier": M.brier(d, p), "auc": M.auc(d, p),
            "within_month_auc": M.stratified_auc(d, p, months), "cal_intercept": cal["intercept"],
            "cal_slope": cal["slope"], "mean_pred": float(p.mean()), "rate": float(np.mean(d))}


KEEP = ["app_id", "name", "dev", "Category", "released", "age_days", "installs", "month_pct", "game",
        "dev_prior_apps", "dev_same_day", "dev_prior_30d", "dev_days_active", "dev_prior_cats",
        "dev_prior_same_cat", "pkg_prior"]
SLIM = FEATURES.with_name("milestone_input.pkl")


def load_window():
    """Only the 16 columns and the launch window this stage needs. The full feature table is 760 MB on disk and
    several GB in memory, so it is read once to write a slim copy, and the slim copy is used from then on."""
    if not SLIM.exists() or SLIM.stat().st_mtime < FEATURES.stat().st_mtime:
        df = pd.read_pickle(FEATURES)[KEEP]
        df = df[(df["released"] >= START) & (df["released"] < END)].reset_index(drop=True)
        df["Category"] = df["Category"].astype("category")
        df.to_pickle(SLIM)
        del df
        gc.collect()
    return pd.read_pickle(SLIM)


def scores(X, coef, chunk=200_000):
    """X @ coef in float64 without making a float64 copy of the whole matrix."""
    coef = np.asarray(coef, float)
    return np.concatenate([X[i:i + chunk].astype(float) @ coef for i in range(0, len(X), chunk)])


def sample(idx, n, seed):
    idx = np.asarray(idx)
    if len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def main():
    dirs()
    w = load_window()
    w["age_m"] = np.maximum(w["age_days"].to_numpy(), 1) / MONTH
    w["month"] = w["released"].dt.to_period("M").astype(str)
    cats = sorted(w["Category"].astype(str).unique())
    ref = w["Category"].astype(str).value_counts().index[0]
    terms = make_terms(cats, ref)
    for t in terms:
        F.check_launch_only([INPUT_SOURCE[t["input"]]])
    X = PM.encode_frame(inputs(w), terms, dtype=np.float32)   # float32 to keep memory down
    # drop terms that never vary in this window (a category with no apps, say)
    varies = X.min(axis=0) != X.max(axis=0)
    terms = [t for t, v in zip(terms, varies) if v]
    X = X[:, varies]
    dev = pd.factorize(w["dev"])[0]
    comp = comparables_sample(w)
    w = w.drop(columns=["dev", "app_id"])     # long strings, not needed past this point
    gc.collect()
    age = w["age_m"].to_numpy()
    D = {m: (w["installs"].to_numpy() >= m).astype(float) for m in MILESTONES}
    res = {"window": f"launches {START} to {END}", "apps": len(w), "reference_category": ref,
           "fit_sample": FIT_N}

    # NPMLE curves for each milestone, plus subsampling CIs at a few ages
    B = 20 if FAST else 200
    res["npmle"] = {}
    curves = {}
    for m in MILESTONES:
        a, F_ = npmle(np.round(w["age_days"].to_numpy() / MONTH, 2), D[m])
        curves[m] = (a, F_)
        est, lo, hi = npmle_ci(np.round(w["age_days"].to_numpy() / MONTH, 2), D[m], dev,
                               np.array(EVAL_MONTHS, float), B)
        res["npmle"][NAMES[m]] = {str(t): {"F": e, "lo": l, "hi": h} for t, e, l, h in zip(EVAL_MONTHS, est, lo, hi)}
    print("npmle done")

    # validation: forward in time and interleaved months, three links, plus an age-only model
    fwd_tr = np.where(w["released"] < CUTOFF)[0]
    fwd_te = np.where(w["released"] >= CUTOFF)[0]
    mnum = w["released"].dt.year * 12 + w["released"].dt.month
    inter_te = np.where(mnum % 4 == 0)[0]
    inter_tr = np.where(mnum % 4 != 0)[0]
    res["validation"] = {}
    by_month = {}
    for vname, tr, te in [("forward", fwd_tr, fwd_te), ("interleaved", inter_tr, inter_te)]:
        tr_s = sample(tr, FIT_N, SEED)
        res["validation"][vname] = {"train_apps_sampled": len(tr_s), "test_apps": len(te)}
        for m in MILESTONES:
            r_m = {}
            for link in LINKS:
                r = fit_glm(X[tr_s], age[tr_s], D[m][tr_s], link)
                p = predict(r, X[te], age[te], link)
                r_m[link] = eval_block(D[m][te], p, w["month"].to_numpy()[te])
                del r
                gc.collect()
                if vname == "forward":
                    by_month[(m, link)] = pd.DataFrame({"month": w["month"].to_numpy()[te], "p": p,
                                                        "d": D[m][te]}).groupby("month").mean()
                    # calibration for apps close to one year old, the horizon the scorer reports
                    yr = (age[te] >= 9) & (age[te] <= 15)
                    r_m[link]["reliability_9_to_15_months"] = M.reliability(D[m][te][yr], p[yr], 10)
                    r_m[link]["calibration_9_to_15_months"] = M.calibration(D[m][te][yr], p[yr])
            r0 = sm.GLM(D[m][tr_s], design(np.empty((len(tr_s), 0)), age[tr_s]),
                        family=sm.families.Binomial(link=LINKS["cloglog"]())).fit()
            p0 = PM.inv_link(r0.params[0] + r0.params[1] * np.log(age[te]), "cloglog")
            r_m["age_only_cloglog"] = eval_block(D[m][te], p0, w["month"].to_numpy()[te])
            if vname == "interleaved":
                a_, F_ = npmle(np.round(age[tr], 2), D[m][tr])
                r_m["npmle_age_only"] = eval_block(D[m][te], np.interp(age[te], a_, F_),
                                                   w["month"].to_numpy()[te])
            res["validation"][vname][NAMES[m]] = r_m
            gc.collect()
            print(vname, NAMES[m], {k: round(v["log_loss"], 4) for k, v in r_m.items()})

    # pick the link with the lowest held-out log loss, averaged over both checks and all milestones
    score = {l: np.mean([res["validation"][v][NAMES[m]][l]["log_loss"] for v in ["forward", "interleaved"]
                         for m in MILESTONES]) for l in LINKS}
    link = min(score, key=score.get)
    res["link_log_loss"] = score
    res["link"] = link

    # final fit on a sample of the whole window, with developer-clustered covariance
    fit_idx = sample(np.arange(len(w)), FIT_N, SEED + 1)
    fits = {m: fit_glm(X[fit_idx], age[fit_idx], D[m][fit_idx], link, groups=dev[fit_idx]) for m in MILESTONES}
    print("final fits done")

    model = {
        "format": "playodds-milestones/1",
        "description": "P(an app reaches N installs within t months of launch) from launch-day facts. "
                       "Current status GLM fit to the June 2021 Google Play scrape.",
        "link": link, "time_unit": "months", "categories": cats, "reference_category": ref,
        "inputs": PM.INPUTS, "terms": terms,
        "milestone_list": [str(m) for m in MILESTONES], "index_milestone": str(100_000),
        "milestones": {str(m): as_export(fits[m]) for m in MILESTONES},
        "z": Z,
    }

    # conformal intervals for the launch-month percentile, calibrated on apps the final fit never saw
    rest = np.setdiff1d(np.arange(len(w)), fit_idx)
    rng = np.random.default_rng(SEED + 2)
    is_cal = rng.random(len(rest)) < 0.5
    cal_idx, test_idx = rest[is_cal], rest[~is_cal]
    s_all = scores(X, model["milestones"]["100000"]["coef"])
    pctl = w["month_pct"].to_numpy()
    model["conformal"] = {lv: conformal_table(s_all[cal_idx], pctl[cal_idx], a) for lv, a in LEVELS.items()}
    res["conformal"] = coverage(w, s_all, pctl, model["conformal"], test_idx, "random held-out apps")

    # same check forward in time: score with the forward model, calibrate on the last pre-cutoff year
    tr_s = sample(fwd_tr, FIT_N, SEED)
    r_fwd = fit_glm(X[tr_s], age[tr_s], D[100_000][tr_s], link)
    s_fwd = scores(X, np.asarray(r_fwd.params)[2:])
    cal_f = np.setdiff1d(np.where((w["released"] >= "2018-06-01") & (w["released"] < CUTOFF))[0], tr_s)
    tab_f = {lv: conformal_table(s_fwd[cal_f], pctl[cal_f], a) for lv, a in LEVELS.items()}
    res["conformal_forward"] = coverage(w, s_fwd, pctl, tab_f, fwd_te, "launches after June 2019")
    res["conformal_forward"]["by_year"] = {
        str(y): {lv: float(inside(s_fwd[idx], pctl[idx], tab_f[lv]).mean()) for lv in LEVELS}
        for y in [2019, 2020, 2021] for idx in [fwd_te[w["released"].dt.year.to_numpy()[fwd_te] == y]] if len(idx)}

    # install counts at each launch-month percentile, read off the cohort that is t months old now
    model["percentile_installs"] = {}
    snap = pd.Timestamp("2021-06-15")
    for t in [6, 12, 24, 36, 48]:
        mon = (snap - pd.DateOffset(months=t)).to_period("M").strftime("%Y-%m")
        inst = w.loc[w["month"] == mon, "installs"].to_numpy()
        if len(inst) < 50:
            continue
        q = np.arange(0, 101)
        model["percentile_installs"][str(t)] = {"cohort": mon, "pct": q.tolist(),
                                                "installs": np.percentile(inst, q).round(1).tolist()}

    model["meta"] = {"snapshot": "June 2021", "launch_window": [START, END], "apps_in_window": len(w),
                     "fit_apps": len(fit_idx), "conformal_calibration_apps": len(cal_idx),
                     "validation": {v: {NAMES[m]: res["validation"][v][NAMES[m]][link]["auc"] for m in MILESTONES}
                                    for v in ["forward", "interleaved"]},
                     "notes": "Parameter intervals come from developer-clustered covariance. They do not cover "
                              "model error or the stationarity assumption (see docs/model_card.md)."}
    (MODELS / "milestones.json").write_text(json.dumps(model, indent=1))

    # golden predictions: statsmodels' own predictions for real apps, which the exported file must match
    gold = sample(test_idx, 200, SEED + 3)
    ex = []
    for i in gold:
        app = {k: (v.item() if hasattr(v, "item") else v) for k, v in inputs(w.iloc[[i]]).iloc[0].items()}
        months = float(np.round(age[i], 3))
        ex.append({"app": app, "months": months,
                   "p": {str(m): float(fits[m].predict(design(PM.encode_frame(pd.DataFrame([app]), terms),
                                                              np.array([months])))[0]) for m in MILESTONES}})
    (MODELS / "examples.json").write_text(json.dumps(ex, indent=1))

    # violations of P(10K) >= P(100K) >= P(1M) before the scorer enforces the order
    P12 = {m: PM.inv_link(PM.eta(model["milestones"][str(m)], X[test_idx], 12.0), link) for m in MILESTONES}
    res["order_violations_12m"] = float(np.mean((P12[100_000] > P12[10_000] + 1e-12) |
                                                (P12[1_000_000] > P12[100_000] + 1e-12)))

    # effects as time ratios: how much faster (below 1) or slower apps with this trait reach the milestone
    res["time_ratios"] = {NAMES[m]: time_ratios(fits[m], terms) for m in MILESTONES}
    res["p12_by_category_first_app"] = first_app_by_category(w, X, model, cats)
    res["marginal_fit"] = {NAMES[m]: marginal_curve(model, m, X) for m in MILESTONES}

    comparables(comp, X, model)
    write("milestones.json", res)
    figures(res, curves, by_month, link, w)
    print(json.dumps({k: res[k] for k in ["link", "link_log_loss", "order_violations_12m"]}, indent=1))


def conformal_table(s, y, level, bins=20):
    """Mondrian split conformal: within each score bin, the finite-sample corrected lower and upper
    quantiles of the launch-month percentile."""
    edges = np.quantile(s, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.searchsorted(edges, s, side="right")
    a = (1 - level) / 2
    lo, hi = [], []
    for k in range(bins):
        v = np.sort(y[b == k])
        n = len(v)
        i_lo = int(np.floor(a * (n + 1))) - 1
        i_hi = int(np.ceil((1 - a) * (n + 1))) - 1
        lo.append(float(v[i_lo]) if i_lo >= 0 else 0.0)
        hi.append(float(v[i_hi]) if i_hi < n else 1.0)
    return {"edges": edges.tolist(), "lo": lo, "hi": hi, "level": level}


def inside(s, y, tab):
    b = np.clip(np.searchsorted(tab["edges"], s, side="right"), 0, len(tab["lo"]) - 1)
    return (y >= np.asarray(tab["lo"])[b]) & (y <= np.asarray(tab["hi"])[b])


def coverage(w, s, y, tabs, idx, label):
    out = {"test": label, "apps": len(idx)}
    for lv, tab in tabs.items():
        hit = inside(s[idx], y[idx], tab)
        out[lv] = {"target": tab["level"], "coverage": float(hit.mean()),
                   "mean_width": float((np.asarray(tab["hi"]) - np.asarray(tab["lo"]))[
                       np.clip(np.searchsorted(tab["edges"], s[idx], side="right"), 0, len(tab["lo"]) - 1)].mean())}
    t = w.iloc[idx]
    hit80 = inside(s[idx], y[idx], tabs["80"])
    groups = {"developer": pd.cut(t["dev_prior_apps"], [-1, 0, 4, 19, 99, np.inf],
                                  labels=["First app", "1 to 4", "5 to 19", "20 to 99", "100+"]).astype(str).to_numpy(),
              "kind": np.where(t["game"] == 1, "Games", "Apps"),
              "category": t["Category"].astype(str).to_numpy()}
    out["by_80"] = {g: {str(k): float(hit80[v == k].mean()) for k in np.unique(v) if (v == k).sum() >= 200}
                    for g, v in groups.items()}
    return out


def time_ratios(r, terms):
    """For an AFT reading of the model: log T = (G^-1(U) - a - x'beta) / b, so a term multiplies
    the time to milestone by exp(-beta / b). Delta method CI from the clustered covariance."""
    p = np.asarray(r.params)
    V = np.asarray(r.cov_params())
    b = p[1]
    out = {}
    for j, t in enumerate(terms):
        k = j + 2
        tr = -p[k] / b
        g = np.zeros(len(p))
        g[k] = -1 / b
        g[1] = p[k] / b ** 2
        se = np.sqrt(g @ V @ g)
        out[t["name"]] = {"time_ratio": float(np.exp(tr)), "lo": float(np.exp(tr - 1.96 * se)),
                          "hi": float(np.exp(tr + 1.96 * se)), "coef": float(p[k]),
                          "coef_se": float(np.sqrt(V[k, k]))}
    return out


def first_app_by_category(w, X, model, cats):
    """Average predicted 12-month odds for a developer's first app in each category (2019-2020 launches)."""
    m = (w["dev_prior_apps"].to_numpy() == 0) & w["released"].dt.year.isin([2019, 2020]).to_numpy()
    out = {}
    for c in cats:
        idx = np.where(m & (w["Category"].astype(str).to_numpy() == c))[0]
        if len(idx) < (10 if FAST else 100):
            continue
        out[c] = {"apps": len(idx)}
        for ms in MILESTONES:
            mm = model["milestones"][str(ms)]
            out[c][NAMES[ms]] = float(PM.inv_link(PM.eta(mm, X[idx], 12.0), model["link"]).mean())
    return out


def marginal_curve(model, m, X, n=40_000):
    idx = sample(np.arange(len(X)), n, SEED + 4)
    ts = np.geomspace(0.5, 66, 60)
    mm = model["milestones"][str(m)]
    return {"t": ts.tolist(), "F": [float(PM.inv_link(PM.eta(mm, X[idx], t), model["link"]).mean()) for t in ts]}


def comparables_sample(w, per_cat=300):
    """Up to per_cat random 2018 to 2020 apps per category, taken early so the id and name columns can be freed."""
    m = w["released"].between("2018-01-01", "2020-12-31")
    rows = []
    for c, g in w[m].groupby(w.loc[m, "Category"].astype(str)):
        rows.append(g.sample(min(per_cat, len(g)), random_state=SEED))
    return pd.concat(rows)[["app_id", "name", "Category", "released", "dev_prior_apps", "installs", "age_days"]]


def comparables(t, X, model):
    s = scores(X[t.index.to_numpy()], model["milestones"][model["index_milestone"]]["coef"])
    out = pd.DataFrame({"app_id": t["app_id"], "name": t["name"], "category": t["Category"].astype(str),
                        "released": t["released"].dt.strftime("%Y-%m"), "earlier_apps": t["dev_prior_apps"],
                        "installs": t["installs"], "months_old": (t["age_days"] / MONTH).round(1),
                        "score": np.round(s, 6)})
    with gzip.open(MODELS / "comparables.csv.gz", "wt", encoding="utf-8") as f:
        out.to_csv(f, index=False)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figures(res, curves, by_month, link, w):
    # time to milestone curves
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cols = {10_000: GRAY, 100_000: BLUE, 1_000_000: DARK}
    for m in MILESTONES:
        a, F_ = curves[m]
        ax.step(a, F_, where="post", color=cols[m], lw=1.8, label=f"{NAMES[m]} installs (NPMLE)")
        mf = res["marginal_fit"][NAMES[m]]
        ax.plot(mf["t"], mf["F"], color=cols[m], lw=1, ls="--")
        for t in [12, 36]:
            r = res["npmle"][NAMES[m]][str(t)]
            ax.errorbar([t], [r["F"]], yerr=[[r["F"] - r["lo"]], [r["hi"] - r["F"]]], color=cols[m], capsize=3,
                        fmt="o", ms=4)
    ax.plot([], [], color=DARK, lw=1, ls="--", label=f"Model average ({link})")
    ax.set_xlim(0, 60)       # the last few months hold a handful of apps and jump around
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Months since launch")
    ax.set_ylabel("Share of apps that got there")
    r12 = res["npmle"]["100K"]["12"]["F"]
    ax.set_title(f"Pooled over cohorts, {r12:.1%} reach 100K installs within a year")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    save(fig, "10_milestone_curves.png")

    # forward validation: predicted vs actual by launch month, and calibration near one year
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    fwd = res["validation"]["forward"]
    for j, m in enumerate(MILESTONES):
        ax = axes[0, j]
        t = by_month[(m, link)]
        x = pd.PeriodIndex(t.index, freq="M").to_timestamp()
        ax.plot(x, t["d"], color=DARK, lw=1.6, label="Actual")
        ax.plot(x, t["p"], color=BLUE, lw=1.6, ls="--", label="Predicted")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
        ax.set_title(f"{NAMES[m]} installs, by launch month")
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.set_ylim(0, None)
        ax = axes[1, j]
        r = fwd[NAMES[m]][link]["reliability_9_to_15_months"]
        pr, ob = [b["pred"] for b in r], [b["obs"] for b in r]
        err = [[b["obs"] - b["lo"] for b in r], [b["hi"] - b["obs"] for b in r]]
        ax.errorbar(pr, ob, yerr=err, fmt="o", color=BLUE, ms=4, capsize=2)
        top = max(max(pr), max(ob)) * 1.1
        ax.plot([0, top], [0, top], color=DARK, lw=0.8, ls=":")
        fmt = plt.FuncFormatter(lambda v, _: f"{100 * v:g}%")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("Predicted (apps 9 to 15 months old)")
        cal = fwd[NAMES[m]][link]["calibration_9_to_15_months"]
        ax.set_title(f"Calibration near one year, slope {cal['slope']:.2f}")
    axes[0, 0].set_ylabel("Share reached by June 2021")
    axes[1, 0].set_ylabel("Actual share")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fit on launches before June 2019, tested on the two years after", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    save(fig, "11_forward_validation.png")

    # conformal coverage
    c = res["conformal"]["by_80"]
    cf = res["conformal_forward"]
    order = [k for k in ["First app", "1 to 4", "5 to 19", "20 to 99", "100+"] if k in c["developer"]]
    kinds = [k for k in ["Apps", "Games"] if k in c["kind"]]
    labels = (["All held-out apps", "Launched after June 2019"] + [f"Earlier apps: {k}" for k in order] +
              [{"Apps": "Apps (not games)", "Games": "Games"}[k] for k in kinds])
    vals = ([res["conformal"]["80"]["coverage"], cf["80"]["coverage"]] + [c["developer"][k] for k in order] +
            [c["kind"][k] for k in kinds])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(labels[::-1], vals[::-1], color=[BLUE if i >= len(labels) - 2 else GRAY for i in range(len(labels))])
    ax.axvline(0.8, color=RED, lw=1.2, ls="--")
    for i, v in enumerate(vals[::-1]):
        ax.text(max(v, 0.8) + 0.008, i, f"{v:.1%}", va="center", fontsize=8.5, color=DARK)
    ax.set_xlim(0.5, 1.0)
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of apps whose real launch-month percentile fell inside the 80% interval")
    ax.set_title(f"80% intervals covered {min(vals):.0%} to {max(vals):.0%} across groups")
    save(fig, "12_interval_coverage.png")


if __name__ == "__main__":
    main()
