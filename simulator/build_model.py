"""
Builds the model behind the "Will your app hit 100K?" simulator.

Run from the repo root after analysis.py:   python simulator/build_model.py
Writes simulator/model.json, which the simulator page reads in the browser.

Why logistic regression and not the gradient boosting model from analysis.py:
it has to run in a browser with no server, and a logistic model is just a list
of weights. It scores 0.84 AUC on the 2019 holdout, close to the 0.86 from
gradient boosting, and every weight can be read and explained.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HIT = 100_000
OUT = Path("simulator/model.json")

SIZE_BINS = [-1, 5, 15, 40, 100, np.inf]
SIZE_LABELS = ["Under 5 MB", "5 to 15 MB", "15 to 40 MB", "40 to 100 MB", "Over 100 MB"]
DEV_BINS = [0, 1, 5, 20, 100, np.inf]
DEV_LABELS = ["This is my first app", "2 to 5", "6 to 20", "21 to 100", "More than 100"]
RATINGS = ["Everyone", "Everyone 10+", "Teen", "Mature 17+"]
FEATURES = ["Category", "model", "size", "dev", "rating"]


def prep(df):
    c = df[df["released"].dt.year.between(2018, 2019)].copy()
    c = c[c["Content Rating"].isin(RATINGS) & c["size_mb"].notna()]
    c["hit"] = (c["installs"] >= HIT).astype(int)
    c["size"] = pd.cut(c["size_mb"], SIZE_BINS, labels=SIZE_LABELS).astype(str)
    c["dev"] = pd.cut(c["dev_apps"], DEV_BINS, labels=DEV_LABELS).astype(str)
    c["rating"] = c["Content Rating"]
    return c


def design(c, columns=None):
    X = pd.get_dummies(c[FEATURES], prefix_sep="=", dtype=float)
    X["privacy=yes"] = c["has_privacy"].astype(float)
    X["website=yes"] = c["has_website"].astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def fit(X, y):
    return LogisticRegression(C=1.0, max_iter=5000).fit(X, y)


def main():
    c = prep(pd.read_pickle("data/clean.pkl"))
    X = design(c)
    y = c["hit"]

    # honest check first: train on 2018 launches, score 2019 launches
    train = c["released"].dt.year == 2018
    test = ~train
    m_check = fit(X[train], y[train])
    p_test = m_check.predict_proba(X[test])[:, 1]
    auc = roc_auc_score(y[test], p_test)

    # calibration on the 2019 holdout, in 5 buckets of predicted probability
    bins = pd.qcut(p_test, 5, duplicates="drop")
    calib = (pd.DataFrame({"pred": p_test, "real": y[test].to_numpy()})
               .groupby(bins, observed=True).mean())
    print(f"Holdout AUC (train 2018, test 2019): {auc:.3f}")
    print("Calibration on 2019 apps (mean predicted vs actual):")
    print(calib.round(3).to_string())

    # final model uses both years so the weights see as much data as possible
    m = fit(X, y)
    weights = {col: round(float(w), 5) for col, w in zip(X.columns, m.coef_[0])}

    # real breakout rates for "apps like yours" (category x monetization)
    cells = (c.groupby(["Category", "model"])["hit"].agg(["size", "mean"])
               .reset_index())
    cells = {f"{r.Category}|{r.model}": [int(r["size"]), round(float(r["mean"]), 4)]
             for _, r in cells.iterrows()}

    cat_rates = c.groupby("Category")["hit"].mean().round(4).to_dict()

    model = {
        "intercept": round(float(m.intercept_[0]), 5),
        "weights": weights,
        "options": {
            "Category": sorted(c["Category"].unique().tolist()),
            "model": ["Ads + in-app purchases", "In-app purchases only", "Ads only",
                      "Free, no ads or purchases", "Paid upfront"],
            "size": SIZE_LABELS,
            "dev": DEV_LABELS,
            "rating": RATINGS,
        },
        "cells": cells,
        "category_rates": cat_rates,
        "meta": {
            "apps": int(len(c)),
            "base_rate": round(float(y.mean()), 4),
            "holdout_auc": round(float(auc), 3),
            "train_apps_2018": int(train.sum()),
            "test_apps_2019": int(test.sum()),
        },
    }
    OUT.write_text(json.dumps(model, separators=(",", ":")))
    print(f"\nSaved {OUT} ({OUT.stat().st_size / 1024:.0f} KB), {len(c):,} apps, "
          f"base rate {y.mean():.1%}")


if __name__ == "__main__":
    main()
