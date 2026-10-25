"""
Feature availability and launch-time features.

The scrape is one snapshot from June 2021. Most columns describe the app as it looked in 2021,
not on launch day. AVAILABILITY says, for every feature any model here uses, when it could
first have been known and whether success could have changed it. The launch-time model is only
allowed features marked "launch" (tests/test_leakage_guard.py enforces that).
"""

import numpy as np
import pandas as pd

# known: "launch"   fixed or set on launch day and visible then
#        "snapshot" only seen in the June 2021 scrape, could have changed after launch
#        "after"    only exists after launch (it is partly the outcome)
# reverse: can success plausibly change it? none / low / medium / high
AVAILABILITY = {
    # launch-time features built here
    "Category":           ("Category", "launch", "low", "picked at launch, can be changed later but rarely is"),
    "rel_month":          ("Released", "launch", "none", "release date never changes"),
    "rel_dow":            ("Released", "launch", "none", "release date never changes"),
    "dev_prior_apps":     ("Developer Id + Released", "launch", "none", "apps by the same developer released earlier"),
    "dev_same_day":       ("Developer Id + Released", "launch", "none", "other apps the developer released the same day"),
    "dev_prior_30d":      ("Developer Id + Released", "launch", "none", "apps the developer released in the 30 days before"),
    "dev_days_active":    ("Developer Id + Released", "launch", "none", "days since the developer's first app"),
    "dev_prior_cats":     ("Developer Id + Category", "launch", "low", "distinct categories among earlier apps"),
    "dev_prior_same_cat": ("Developer Id + Category", "launch", "low", "earlier apps in the same category"),
    "pkg_prior":          ("App Id + Released", "launch", "none", "earlier apps sharing the package prefix (app builders)"),
    "pkg_segments":       ("App Id", "launch", "none", "package names can never change"),
    "pkg_len":            ("App Id", "launch", "none", "package names can never change"),
    "name_len":           ("App Name", "launch", "low", "title can be edited later (store optimization)"),
    "name_words":         ("App Name", "launch", "low", "title can be edited later"),
    "name_nonascii":      ("App Name", "launch", "low", "title can be edited later"),
    "name_digit":         ("App Name", "launch", "low", "title can be edited later"),
    # what the old model and the simulator used, all read off the 2021 snapshot
    "dev_apps":           ("Developer Id + Released", "launch", "none", "old feature, same as dev_prior_apps + 1"),
    "Content Rating":     ("Content Rating", "snapshot", "low", "questionnaire is redone when features like chat get added"),
    "size_mb":            ("Size", "snapshot", "high", "apps grow with every update and hits update more"),
    "min_android":        ("Minimum Android", "snapshot", "medium", "raised in updates, hits update more"),
    "paid":               ("Free", "snapshot", "medium", "price can change, paid apps often go free"),
    "price":              ("Price", "snapshot", "medium", "price can change"),
    "ads":                ("Ad Supported", "snapshot", "high", "ads are easy to add once an app has traffic"),
    "iap":                ("In App Purchases", "snapshot", "high", "purchases get added after an app has users"),
    "has_privacy":        ("Privacy Policy", "snapshot", "high", "policies got added as Google required them and as apps grew"),
    "has_website":        ("Developer Website", "snapshot", "medium", "developers with a hit build a site"),
    # leaky developer track record: earlier apps' installs are counted in 2021, after this launch
    "dev_prior_hit_share": ("Developer Id + Maximum Installs", "after", "high", "earlier apps' installs measured in 2021"),
    "dev_prior_max_installs": ("Developer Id + Maximum Installs", "after", "high", "earlier apps' installs measured in 2021"),
    # clearly after launch
    "rating":             ("Rating", "after", "high", "needs users"),
    "rating_count":       ("Rating Count", "after", "high", "basically a second install count"),
    "editors_choice":     ("Editors Choice", "after", "high", "given to apps that already did well"),
    "days_since_update":  ("Last Updated", "after", "high", "hits keep getting updated"),
}

LAUNCH = ["Category", "rel_month", "rel_dow", "dev_prior_apps", "dev_same_day", "dev_prior_30d",
          "dev_days_active", "dev_prior_cats", "dev_prior_same_cat", "pkg_prior", "pkg_segments", "pkg_len",
          "name_len", "name_words", "name_nonascii", "name_digit"]
# only things that can never change after launch
STRICT = [f for f in LAUNCH if f != "Category" and not f.startswith("name_")]
OLD = ["size_mb", "min_android", "dev_apps", "paid", "ads", "iap", "has_privacy", "has_website",
       "Category", "Content Rating"]
MONETIZATION = ["paid", "price", "ads", "iap"]
LISTING = ["has_privacy", "has_website"]
TECH = ["size_mb", "min_android"]
SNAPSHOT = MONETIZATION + LISTING + TECH + ["Content Rating"]
LEAKY_DEV = ["dev_prior_hit_share", "dev_prior_max_installs"]
POST = ["rating", "rating_count", "editors_choice", "days_since_update"]
CATEGORICAL = ["Category", "Content Rating"]

GAMES = ["Action", "Adventure", "Arcade", "Board", "Card", "Casino", "Casual", "Educational", "Music",
         "Puzzle", "Racing", "Role Playing", "Simulation", "Strategy", "Trivia", "Word"]


def known(feature):
    return AVAILABILITY[feature][1]


def check_launch_only(features):
    """Raises if any feature isn't known at launch. Called before every launch-time fit."""
    bad = [f for f in features if f not in AVAILABILITY or known(f) != "launch"]
    if bad:
        raise ValueError(f"not available at launch: {bad}")
    return True


def availability_table():
    rows = []
    for f, (src, k, rev, note) in AVAILABILITY.items():
        rows.append({"feature": f, "source column": src, "first known": k, "success can change it": rev,
                     "old model": f in OLD, "launch model": f in LAUNCH, "note": note})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# developer history, using only apps released strictly before each app
# --------------------------------------------------------------------------
def _earlier(group_codes, days, values=None, window=None):
    """For each row, count (or sum `values` over) rows in the same group released strictly earlier.
    With window=w, only rows released in the w days before. Works on any row order."""
    order = np.lexsort((days, group_codes))
    g, d = group_codes[order], days[order]
    key = g.astype(np.int64) * 100_000 + d          # sorted, and groups never overlap
    first = np.searchsorted(key, g.astype(np.int64) * 100_000, "left")
    lo = first if window is None else np.maximum(first, np.searchsorted(key, key - window, "left"))
    hi = np.searchsorted(key, key, "left")            # strictly earlier day
    if values is None:
        out_sorted = hi - lo
    else:
        cs = np.concatenate([[0.0], np.cumsum(values[order].astype(float))])
        out_sorted = cs[hi] - cs[lo]
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def pkg_prefix(app_ids):
    # first two parts of the package name, or three when the second is a country-style "com"/"co"
    parts = app_ids.str.lower().str.split(".")
    two = parts.str[:2].str.join(".")
    three = parts.str[:3].str.join(".")
    generic = parts.str[1].isin(["com", "co", "net", "org", "ac", "gov", "or", "ne", "go"])
    return two.where(~generic, three)


def add_launch_features(df):
    """Adds every launch-time feature. Uses only the app's own fixed fields and earlier releases."""
    df = df.copy()
    days = (df["released"] - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    dev = pd.factorize(df["dev"])[0]

    df["rel_month"] = df["released"].dt.month
    df["rel_dow"] = df["released"].dt.dayofweek

    df["dev_prior_apps"] = _earlier(dev, days)
    df["dev_apps"] = df["dev_prior_apps"] + 1
    same = pd.Series(dev).groupby([dev, days]).transform("size").to_numpy()
    df["dev_same_day"] = same - 1
    df["dev_prior_30d"] = _earlier(dev, days, window=30)
    first_day = pd.Series(days).groupby(dev).transform("min").to_numpy()
    df["dev_days_active"] = days - first_day

    # distinct earlier categories: the first release of each (developer, category) pair
    pair = pd.factorize(pd.Series(dev).astype(str) + "|" + df["Category"].astype(str).to_numpy())[0]
    pair_first = pd.Series(days).groupby(pair).transform("min").to_numpy()
    cand = days == pair_first
    is_first = cand & ~pd.Series(np.where(cand, pair, -1 - np.arange(len(pair)))).duplicated().to_numpy()
    df["dev_prior_cats"] = _earlier(dev, days, values=is_first.astype(float)).astype(int)
    df["dev_prior_same_cat"] = _earlier(pair, days)

    pref = pd.factorize(pkg_prefix(df["app_id"]))[0]
    df["pkg_prior"] = _earlier(pref, days)
    df["pkg_segments"] = df["app_id"].str.count(r"\.") + 1
    df["pkg_len"] = df["app_id"].str.len()

    name = df["name"]
    df["name_len"] = name.str.len()
    df["name_words"] = name.str.split().str.len().fillna(0).astype(int)
    df["name_nonascii"] = name.map(lambda s: any(ord(ch) > 127 for ch in s)).astype(int)
    df["name_digit"] = name.str.contains(r"[0-9]", regex=True).astype(int)
    return df


def add_leaky_features(df, hit_col="hit100k"):
    """Developer track record using earlier apps' 2021 installs. Not known at launch, only used to
    show how much this kind of leak inflates the score."""
    df = df.copy()
    days = (df["released"] - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    dev = pd.factorize(df["dev"])[0]
    hits = _earlier(dev, days, values=df[hit_col].to_numpy().astype(float))
    n = df["dev_prior_apps"].to_numpy()
    df["dev_prior_hit_share"] = np.where(n > 0, hits / np.maximum(n, 1), np.nan)
    # max of earlier installs: cumulative max within developer, shifted to strictly earlier days
    order = np.lexsort((days, dev))
    s = pd.DataFrame({"dev": dev[order], "day": days[order], "x": np.log10(df["installs"].to_numpy()[order] + 1)})
    daymax = s.groupby(["dev", "day"])["x"].max()
    prev = daymax.groupby(level=0).cummax().groupby(level=0).shift(1)
    m = prev.reindex(pd.MultiIndex.from_arrays([s["dev"], s["day"]])).to_numpy()
    out = np.empty(len(df))
    out[order] = m
    df["dev_prior_max_installs"] = out
    df["days_since_update"] = (df["scraped"] - df["updated"]).dt.days
    return df


# --------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------
def add_outcomes(df, top=0.05, hit=100_000):
    """hit100k: 100K+ installs by June 2021 (the old outcome, horizon depends on release date).
    top5: in the top `top` share of installs among all apps released the same calendar month.
    Every app in a month had the same time to grow, so this one compares like with like."""
    df = df.copy()
    df["hit100k"] = (df["installs"] >= hit).astype(int)
    df["top5"] = top_in_month(df, top)
    month = df["released"].dt.to_period("M")
    df["month_pct"] = df.groupby(month)["installs"].rank(pct=True, method="average")
    return df


def top_in_month(df, top):
    month = df["released"].dt.to_period("M")
    cut = df.groupby(month)["installs"].transform(lambda s: s.quantile(1 - top))
    return (df["installs"] >= cut).astype(int)
