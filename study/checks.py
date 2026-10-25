"""
Checks of the milestone model against raw rates in the data, so the README can quote observed numbers next to
model numbers.

  cohort_rates   share past each milestone for apps at a given age, next to the pooled NPMLE and the model average
  by_category    model average vs observed, apps 9 to 15 months old, every category
  first_apps     the same for a developer's first apps launched 2019 and 2020 (the README market table)
  npmle_interval_simulation   how often the subsampling intervals cover the truth when I know it

Run:  python -m study.checks     (after study.milestones)
"""

import numpy as np
import pandas as pd

from playodds import model as PM
from study import milestones as MS
from study.common import FAST, MODELS, dirs, read, write

MIN_N = 20 if FAST else 200


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def simulate_interval_coverage(n, seeds, B):
    """Current status data from a Weibull time to milestone (shape 0.8, scale 30 months), ages uniform 0.5 to 66,
    developers of 4 apps each. Share of (seed, age) pairs where the 95% interval holds the true F."""
    pts = np.array([6.0, 12.0, 36.0])
    truth = 1 - np.exp(-(pts / 30.0) ** 0.8)
    hits = []
    for seed in range(seeds):
        rng = np.random.default_rng(300 + seed)
        age = rng.uniform(0.5, 66, n)
        d = (30.0 * rng.weibull(0.8, n) <= age).astype(float)
        est, lo, hi = MS.npmle_ci(np.round(age, 2), d, np.arange(n) // 4, pts, B)
        hits += list((lo <= truth) & (truth <= hi))
    return {"apps": n, "seeds": seeds, "draws": B, "nominal": 0.95, "coverage": float(np.mean(hits))}


def block(w, X, model, idx):
    """Observed share past each milestone and the model's average at each app's own age."""
    out = {"apps": int(len(idx))}
    if len(idx) == 0:
        return out
    age = w["age_m"].to_numpy()[idx]
    for ms in MS.MILESTONES:
        d = (w["installs"].to_numpy()[idx] >= ms)
        m = model["milestones"][str(ms)]
        eta = m["intercept"] + m["log_time"] * np.log(age) + X[idx].astype(float) @ np.asarray(m["coef"])
        pred = float(PM.inv_link(eta, model["link"]).mean())
        lo, hi = wilson(d.sum(), len(d))
        out[MS.NAMES[ms]] = {"observed": float(d.mean()), "lo": float(lo), "hi": float(hi), "model": pred,
                             "model_inside_interval": bool(lo <= pred <= hi)}
    return out


def main():
    dirs()
    model = PM.load(MODELS / "milestones.json")
    ms_res = read("milestones.json")
    w = MS.load_window()
    w["age_m"] = np.maximum(w["age_days"].to_numpy(), 1) / MS.MONTH
    X = PM.encode_frame(MS.inputs(w), model["terms"], dtype=np.float32)
    age = w["age_m"].to_numpy()
    cat = w["Category"].astype(str).to_numpy()
    first = w["dev_prior_apps"].to_numpy() == 0
    yr = w["released"].dt.year.to_numpy()
    res = {"note": "observed = raw share past the milestone, model = model average at each app's own age, "
                   "intervals are Wilson intervals for the observed share (apps treated as independent)"}

    res["cohort_rates"] = {}
    for lo, hi in [(5, 7), (8, 10), (11, 13), (14, 16), (17, 19), (23, 25)]:
        idx = np.where((age >= lo) & (age <= hi))[0]
        r = block(w, X, model, idx)
        r["launch_months"] = [str(w["released"].iloc[idx].min().date()), str(w["released"].iloc[idx].max().date())]
        r["pooled_npmle_at_midpoint"] = {k: ms_res["npmle"][k].get(str((lo + hi) // 2), {}).get("F")
                                         for k in ["10K", "100K", "1M"]}
        res["cohort_rates"][f"{lo} to {hi} months"] = r

    big = {}
    for c in sorted(set(cat)):
        idx = np.where((cat == c) & (age >= 9) & (age <= 15))[0]
        if len(idx) >= MIN_N:
            big[c] = block(w, X, model, idx)
    res["by_category_9_to_15_months"] = big
    for name in ["10K", "100K", "1M"] if big else []:
        ratio = np.array([v[name]["observed"] / v[name]["model"] for v in big.values() if v[name]["model"] > 0])
        inside = np.mean([v[name]["model_inside_interval"] for v in big.values()])
        res.setdefault("by_category_summary", {})[name] = {
            "categories": len(big), "share_with_model_inside_interval": float(inside),
            "median_observed_over_model": float(np.median(ratio)), "min": float(ratio.min()), "max": float(ratio.max())}

    fa = {}
    base = (first & np.isin(yr, [2019, 2020]))
    for c in sorted(set(cat)):
        allidx = np.where(base & (cat == c))[0]
        if len(allidx) < MIN_N:
            continue
        near = allidx[(age[allidx] >= 9) & (age[allidx] <= 15)]
        r = {"all_ages": block(w, X, model, allidx), "ages_9_to_15": block(w, X, model, near)}
        r["model_average_at_12_months"] = {k: v for k, v in ms_res["p12_by_category_first_app"].get(c, {}).items()
                                           if k != "apps"}
        fa[c] = r
    res["first_apps_2019_2020"] = fa
    res["npmle_interval_simulation"] = simulate_interval_coverage(20_000 if FAST else 200_000, 4 if FAST else 20,
                                                                  10 if FAST else 100)
    write("checks.json", res)
    for c in ["Simulation", "Business"]:
        if c in fa:
            print(c, {k: (v["apps"], round(v["10K"]["observed"], 3), round(v["10K"]["model"], 3))
                      for k, v in [("all", fa[c]["all_ages"]), ("9-15", fa[c]["ages_9_to_15"])]},
                  fa[c]["model_average_at_12_months"])
    print(res.get("by_category_summary"))
    print({k: (v["apps"], round(v["100K"]["observed"], 4), round(v["100K"]["model"], 4)) for k, v in
           res["cohort_rates"].items()})


if __name__ == "__main__":
    main()
