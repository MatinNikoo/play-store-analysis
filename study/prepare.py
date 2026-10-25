"""
Builds the study table once: cleaned apps + launch-time features + outcomes, saved to
data/study_features.pkl. Also writes results/study/populations.json, which pins down which
app count belongs to which population (815,231 vs 799,763 and so on).

Run:  python -m study.prepare
"""

import pandas as pd

from study import data, features as F
from study.common import FEATURES, RES, dirs, write

SIM_RATINGS = ["Everyone", "Everyone 10+", "Teen", "Mature 17+"]


def populations(df):
    counts = dict(df.attrs.get("counts", {}))
    year = df["released"].dt.year
    c = df[year.between(2018, 2019)]
    rated = c["Content Rating"].isin(SIM_RATINGS)
    sized = c["size_mb"].notna()
    sim = c[rated & sized]
    m = df[(df["released"] >= "2016-01-01") & (df["released"] < "2021-06-01")]
    return {
        **counts,
        "cleaned_apps": int(len(df)),
        "cohort_2018_2019": int(len(c)),
        "cohort_2018": int((c["released"].dt.year == 2018).sum()),
        "cohort_2019": int((c["released"].dt.year == 2019).sum()),
        "simulator_population": int(len(sim)),
        "simulator_2018": int((sim["released"].dt.year == 2018).sum()),
        "simulator_2019": int((sim["released"].dt.year == 2019).sum()),
        "cohort_dropped_size_unknown": int((~sized).sum()),
        "cohort_dropped_rating_other": int((~rated).sum()),
        "cohort_dropped_both": int((~sized & ~rated).sum()),
        "milestone_window_2016_01_to_2021_05": int(len(m)),
        "developers": int(df["dev"].nunique()),
        "first_release": str(df["released"].min().date()), "last_release": str(df["released"].max().date()),
        "share_never_updated": float((df["updated"] <= df["released"]).mean()),
        "share_size_unknown": float(df["size_mb"].isna().mean()),
        "share_free": float((~df["paid"]).mean()), "share_ads_2021": float(df["ads"].mean()),
        "share_iap_2021": float(df["iap"].mean()), "apps_with_zero_installs": int((df["installs"] == 0).sum()),
        "apps_by_year": {int(k): int(v) for k, v in year.value_counts().sort_index().items()},
        "by_launch_month": by_month(df),
    }


def by_month(df):
    """Apps, median installs, 100K rate and the top 5% cutoff for each launch month from 2016 on."""
    d = df[df["released"] >= "2016-01-01"]
    g = d.groupby(d["released"].dt.to_period("M").astype(str))["installs"]
    t = pd.DataFrame({"apps": g.size(), "median_installs": g.median(), "rate_100k": g.apply(lambda s: (s >= 100_000).mean()),
                      "top5_cutoff": g.quantile(0.95)})
    return {m: {k: float(v) for k, v in r.items()} for m, r in t.iterrows()}


def build(df):
    df = F.add_launch_features(df)
    df = F.add_outcomes(df)
    df = F.add_leaky_features(df)
    df["never_updated"] = (df["updated"] <= df["released"]).astype(int)
    df["game"] = df["Category"].isin(F.GAMES).astype(int)
    return df


def main():
    dirs()
    df = data.load()
    pops = populations(df)
    df = build(df)
    pops["top5_rate_2018"] = float(df.loc[df["released"].dt.year == 2018, "top5"].mean())
    pops["top5_rate_2019"] = float(df.loc[df["released"].dt.year == 2019, "top5"].mean())
    write("populations.json", pops)
    F.availability_table().to_csv(RES / "feature_availability.csv", index=False)
    FEATURES.parent.mkdir(exist_ok=True)
    df.to_pickle(FEATURES)
    print(pd.Series(pops).to_string())
    print(f"Saved {FEATURES}")


if __name__ == "__main__":
    main()
