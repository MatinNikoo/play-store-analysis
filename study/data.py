"""
Loads the raw scrape and cleans it with the same rules as analysis.py, but keeps the extra
columns the launch-time study needs (app id, developer id, name, last updated, ratings).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/google_playstore_2021.csv")
CACHE = Path("data/study.pkl")

COLS = ["App Name", "App Id", "Category", "Rating", "Rating Count", "Maximum Installs", "Free", "Price",
        "Size", "Minimum Android", "Developer Id", "Developer Website", "Released", "Last Updated",
        "Content Rating", "Privacy Policy", "Ad Supported", "In App Purchases", "Editors Choice",
        "Scraped Time"]


def as_bool(s):
    # the csv stores True/False, but a fixture or a different pandas version can hand back strings
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


def clean(raw):
    """Same filters as analysis.py: drop rows with no release date, installs or category, then
    drop duplicate app ids (first row wins)."""
    df = raw.rename(columns={"Maximum Installs": "installs"})
    df["released"] = pd.to_datetime(df["Released"], format="%b %d, %Y", errors="coerce")
    counts = {"raw_rows": len(df)}
    df = df.dropna(subset=["released", "installs", "Category"])
    counts["after_dropping_missing"] = len(df)
    df = df.drop_duplicates("App Id").reset_index(drop=True)
    counts["after_dropping_duplicates"] = len(df)

    df["updated"] = pd.to_datetime(df["Last Updated"], format="%b %d, %Y", errors="coerce")
    df["scraped"] = pd.to_datetime(df["Scraped Time"].astype(str).str[:10], errors="coerce").dt.normalize()
    df["scraped"] = df["scraped"].fillna(pd.Timestamp("2021-06-16"))
    df["age_days"] = (df["scraped"] - df["released"]).dt.days.clip(lower=0)

    size = df["Size"].astype(str).str.replace(",", "")
    df["size_mb"] = np.where(size.str.endswith("M"), pd.to_numeric(size.str[:-1], errors="coerce"),
                    np.where(size.str.endswith("k"), pd.to_numeric(size.str[:-1], errors="coerce") / 1024,
                             np.nan))
    df["min_android"] = pd.to_numeric(df["Minimum Android"].astype(str).str.extract(r"^(\d+\.\d+)")[0],
                                      errors="coerce")
    df["paid"] = ~as_bool(df["Free"])
    df["price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    df["ads"] = as_bool(df["Ad Supported"])
    df["iap"] = as_bool(df["In App Purchases"])
    df["editors_choice"] = as_bool(df["Editors Choice"])
    df["has_privacy"] = df["Privacy Policy"].notna()
    df["has_website"] = df["Developer Website"].notna()
    df["rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["Rating Count"], errors="coerce")
    df["model"] = np.select(
        [df["paid"], df["ads"] & df["iap"], df["ads"], df["iap"]],
        ["Paid upfront", "Ads + in-app purchases", "Ads only", "In-app purchases only"],
        "Free, no ads or purchases")

    # a handful of apps have no developer id. Treat each as its own developer
    dev = df["Developer Id"].astype(object)
    df["dev"] = dev.where(dev.notna(), "missing:" + df["App Id"].astype(str))
    df["name"] = df["App Name"].fillna("").astype(str)
    df["app_id"] = df["App Id"].astype(str)

    keep = ["app_id", "name", "dev", "Category", "Content Rating", "installs", "released", "updated", "scraped",
            "age_days", "size_mb", "min_android", "paid", "price", "ads", "iap", "editors_choice",
            "has_privacy", "has_website", "rating", "rating_count", "model"]
    df = df[keep].copy()
    df.attrs["counts"] = counts
    return df


def load(path=RAW, cache=CACHE):
    if cache is not None and Path(cache).exists():
        return pd.read_pickle(cache)
    raw = pd.read_csv(path, usecols=COLS, low_memory=False)
    df = clean(raw)
    for col in ["Category", "Content Rating", "model"]:
        df[col] = df[col].astype("category")
    if cache is not None:
        df.to_pickle(cache)
    return df
