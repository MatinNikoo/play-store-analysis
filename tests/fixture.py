"""
A small fake Play Store scrape in the exact raw format of the real csv (same columns, "10M" sizes,
"Feb 26, 2020" dates, True/False flags). Installs grow with age and depend on developer skill and
category, so the models have something real to find. Used by the tests and by CI.
"""

import numpy as np
import pandas as pd

CATS = ["Action", "Puzzle", "Simulation", "Business", "Tools", "Education", "Communication", "Food & Drink"]
CAT_EFFECT = np.array([1.0, 0.6, 1.2, -1.0, 0.2, -0.4, 0.0, -0.8])
SCRAPE = pd.Timestamp("2021-06-15")


def make_raw(n=8000, devs=900, seed=0):
    rng = np.random.default_rng(seed)
    dev = rng.integers(0, devs, n)
    skill = rng.normal(0, 1, devs)[dev]
    start = pd.Timestamp("2014-01-01")
    days = rng.integers(0, (SCRAPE - start).days - 10, n)
    released = start + pd.to_timedelta(days, unit="D")
    cat = rng.integers(0, len(CATS), n)
    age_m = (SCRAPE - released).days / 30.4
    # time to reach installs grows like a log-normal process, so older apps have more
    log_inst = 5.0 + 1.1 * skill + CAT_EFFECT[cat] + 1.3 * np.log1p(age_m) + rng.normal(0, 1.6, n)
    installs = np.floor(np.exp(log_inst)).astype(np.int64)
    hit = installs > 50_000
    ads = rng.random(n) < 0.4 + 0.3 * hit
    iap = rng.random(n) < 0.05 + 0.3 * hit
    paid = rng.random(n) < 0.03
    size = np.where(rng.random(n) < 0.05, "Varies with device",
                    np.where(rng.random(n) < 0.1, [f"{x:.0f}k" for x in rng.uniform(100, 900, n)],
                             [f"{x:.1f}M" for x in rng.lognormal(2.5, 1, n)]))
    builder = rng.random(n) < 0.15
    app_id = np.where(builder, [f"com.andromo.dev{d}.app{i}" for i, d in enumerate(dev)],
                      [f"com.studio{d}.game{i}" for i, d in enumerate(dev)])
    updated = released + pd.to_timedelta(np.where(rng.random(n) < 0.3 + 0.5 * hit,
                                                  rng.integers(0, 400, n), 0), unit="D")
    updated = pd.DatetimeIndex(np.minimum(updated.values, SCRAPE.to_datetime64()))
    names = [("Super " * (i % 3) + f"App {i}" + (" é" if i % 11 == 0 else "")).strip() for i in range(n)]
    raw = pd.DataFrame({
        "App Name": names, "App Id": app_id, "Category": np.array(CATS)[cat],
        "Rating": np.round(rng.uniform(0, 5, n), 1), "Rating Count": (installs * 0.01).astype(int),
        "Installs": [f"{x:,}+" for x in installs], "Minimum Installs": installs.astype(float),
        "Maximum Installs": installs, "Free": ~paid, "Price": np.where(paid, 1.99, 0.0), "Currency": "USD",
        "Size": size, "Minimum Android": rng.choice(["4.1 and up", "5.0 and up", "Varies with device"], n),
        "Developer Id": [f"Dev {d}" for d in dev],
        "Developer Website": np.where(rng.random(n) < 0.5, "https://x.example", None),
        "Developer Email": "a@b.c", "Released": released.strftime("%b %d, %Y"),
        "Last Updated": updated.strftime("%b %d, %Y"),
        "Content Rating": rng.choice(["Everyone", "Teen", "Mature 17+", "Everyone 10+", "Unrated"], n,
                                     p=[0.7, 0.15, 0.08, 0.06, 0.01]),
        "Privacy Policy": np.where(rng.random(n) < 0.4 + 0.3 * hit, "https://p.example", None),
        "Ad Supported": ads, "In App Purchases": iap, "Editors Choice": rng.random(n) < 0.001,
        "Scraped Time": "2021-06-15 20:19:35",
    })
    # a few rows the cleaning has to drop: missing release date and a duplicate app id
    raw.loc[0:4, "Released"] = np.nan
    raw = pd.concat([raw, raw.iloc[[10, 11]]], ignore_index=True)
    return raw
