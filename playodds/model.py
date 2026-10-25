"""
Scores an app idea with the exported milestone model (models/milestones.json).

The model file is plain JSON so the same numbers can run in Python, in a browser (models/score.js)
or anywhere else. models/FORMAT.md documents every field.
"""

import difflib
import gzip
import json
import math
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("PLAYODDS_MODELS", HERE.parent / "models"))


def data_path(name):
    """Finds a shipped data file. PLAYODDS_MODELS wins, then the copy inside an installed package, then the repo."""
    homes = [Path(os.environ["PLAYODDS_MODELS"])] if "PLAYODDS_MODELS" in os.environ else []
    homes += [HERE / "_data", HERE.parent / "models", HERE.parent / "results" / "study"]
    for h in homes:
        if (h / name).exists():
            return h / name
    return None

# what a person can tell us on launch day, with defaults for a first app
INPUTS = {
    "category": None,
    "name": "",
    "earlier_apps": 0,              # apps you already released on Google Play
    "same_day_apps": 0,             # other apps you're releasing the same day
    "apps_last_30_days": 0,         # apps you released in the 30 days before this one
    "months_since_first_app": 0.0,  # 0 if this is your first app
    "earlier_categories": 0,        # distinct categories among your earlier apps
    "earlier_apps_same_category": 0,
    "builder_apps": 0,              # earlier apps on Play sharing your package prefix (app builders)
}


def name_parts(name):
    name = name or ""
    return {"name_len": len(name), "name_words": len(name.split()),
            "name_nonascii": int(any(ord(ch) > 127 for ch in name)),
            "name_digit": int(re.search(r"[0-9]", name) is not None)}


def raw_values(app):
    """Turns one app (dict of INPUTS) into the numbers the terms read."""
    a = {**INPUTS, **app}
    v = {k: a[k] for k in INPUTS if k != "name"}
    v.update(name_parts(a["name"]))
    return v


def term_value(term, v):
    x = v[term["input"]]
    kind = term["kind"]
    if kind == "onehot":
        return 1.0 if x == term["level"] else 0.0
    x = float(x)
    if kind == "bin":
        return 1.0 if term["lo"] <= x <= term["hi"] else 0.0
    if kind == "log1p":
        return math.log1p(min(max(x, 0.0), term["cap"]))
    if kind == "linear":
        return min(max(x, 0.0), term["cap"]) / term["scale"]
    raise ValueError(kind)


def encode(app, terms):
    v = raw_values(app)
    return np.array([term_value(t, v) for t in terms])


def encode_frame(df, terms, dtype=float):
    """Vectorized encode for a DataFrame whose columns are the INPUTS names (training side)."""
    names = df["name"].fillna("").astype(str)
    extra = {"name_len": names.str.len().to_numpy(),
             "name_words": names.map(lambda s: len(s.split())).to_numpy(),
             "name_nonascii": names.map(lambda s: int(any(ord(ch) > 127 for ch in s))).to_numpy(),
             "name_digit": names.str.contains(r"[0-9]", regex=True).astype(int).to_numpy()}
    out = np.empty((len(df), len(terms)), dtype=dtype)     # filled one column at a time to keep memory down
    for j, t in enumerate(terms):
        x = extra[t["input"]] if t["input"] in extra else df[t["input"]].to_numpy()
        k = t["kind"]
        if k == "onehot":
            out[:, j] = np.asarray(x).astype(str) == t["level"]
            continue
        x = np.asarray(x, dtype=float)
        if k == "bin":
            out[:, j] = (x >= t["lo"]) & (x <= t["hi"])
        elif k == "log1p":
            out[:, j] = np.log1p(np.clip(x, 0.0, t["cap"]))
        elif k == "linear":
            out[:, j] = np.clip(x, 0.0, t["cap"]) / t["scale"]
        else:
            raise ValueError(k)
    return out


# --------------------------------------------------------------------------
# links: P(reached by time t) = inv_link(intercept + log_time * log(t) + coef . x)
# --------------------------------------------------------------------------
def inv_link(eta, link):
    eta = np.asarray(eta, float)
    if link == "cloglog":
        return -np.expm1(-np.exp(np.minimum(eta, 30)))
    if link == "logit":
        return 1 / (1 + np.exp(-eta))
    if link == "probit":
        return 0.5 * _erfc(-eta / math.sqrt(2))
    raise ValueError(link)


_erfc = np.vectorize(math.erfc, otypes=[float])


def eta(m, x, months):
    return m["intercept"] + m["log_time"] * np.log(months) + np.asarray(x) @ np.asarray(m["coef"])


MONTHS_RANGE = (0.5, 66.0)       # ages the model was fit on, in months


def check_app(app, months):
    """Plain errors for inputs the model can't read, instead of silent clipping or a math error."""
    if not (isinstance(months, (int, float)) and MONTHS_RANGE[0] <= months <= MONTHS_RANGE[1]):
        raise ValueError(f"months must be between {MONTHS_RANGE[0]:g} and {MONTHS_RANGE[1]:g}, the ages the model "
                         f"was fit on (got {months!r})")
    for k in INPUTS:
        if k in ("category", "name"):
            continue
        v = app.get(k, INPUTS[k])
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v or v < 0:
            raise ValueError(f"{k} must be a number that is 0 or more (got {v!r})")


def prob(model, milestone, x, months, z=None):
    """Chance of reaching `milestone` installs within `months`. With z (like 1.645) also returns the
    interval from the developer-clustered covariance of the fit (parameter uncertainty only)."""
    m = model["milestones"][str(milestone)]
    e = eta(m, x, months)
    p = float(inv_link(e, model["link"]))
    if z is None:
        return p
    g = np.concatenate([[1.0, math.log(months)], np.asarray(x, float)])
    se = math.sqrt(max(float(g @ np.asarray(m["cov"]) @ g), 0.0))
    return p, float(inv_link(e - z * se, model["link"])), float(inv_link(e + z * se, model["link"]))


def score_index(model, x):
    """Launch-time strength of the app, without the time part. Higher means faster growth."""
    return float(np.asarray(x) @ np.asarray(model["milestones"][model["index_milestone"]]["coef"]))


def percentile_interval(model, s, level="80"):
    """Conformal interval for where the app ends up among apps launched the same month."""
    c = model["conformal"][level]
    i = int(np.searchsorted(c["edges"], s, side="right"))
    i = min(max(i, 0), len(c["lo"]) - 1)
    return c["lo"][i], c["hi"][i]


def installs_at(model, percentile, months):
    """Install count at a launch-month percentile, read off the apps that were `months` old in June 2021."""
    tab = model["percentile_installs"][str(months)]
    return float(np.interp(percentile * 100, tab["pct"], tab["installs"]))


def load(path=None):
    path = Path(path) if path else data_path("milestones.json")
    if path is None or not path.exists():
        raise FileNotFoundError(f"no model file found{f' at {path}' if path else ''}. Point PLAYODDS_MODELS at a "
                                "folder that has milestones.json, or pass --model")
    return json.loads(path.read_text())


def resolve_category(model, name):
    """Matches a category ignoring case and 'and' for '&'. Raises a ValueError that suggests close names."""
    cats = model["categories"]
    if name is None or not str(name).strip():
        raise ValueError("category is required. Run `playodds categories` to see the options")
    key = " ".join(str(name).lower().replace(" and ", " & ").split())
    by_key = {c.lower(): c for c in cats}
    if key in by_key:
        return by_key[key]
    close = difflib.get_close_matches(key, list(by_key), n=3, cutoff=0.6)
    close += [k for k in by_key if k.startswith(key) and k not in close]
    hint = f" Did you mean {' or '.join(by_key[k] for k in close[:3])}?" if close else ""
    raise ValueError(f"unknown category {str(name)!r}.{hint} Run `playodds categories` to see all {len(cats)}")


def prepare_app(app, model):
    """The app dict with the category matched to the model's spelling."""
    return {**app, "category": resolve_category(model, app.get("category"))}


def curve(model, app, months):
    """Chance of each milestone at each age in `months`, with the same 1M <= 100K <= 10K ordering as score()."""
    x = encode(app, model["terms"])
    out = {ms: [] for ms in model["milestone_list"]}
    for t in months:
        prev = None
        for ms in model["milestone_list"]:
            p = prob(model, ms, x, t)
            p = p if prev is None else min(p, prev)
            out[ms].append(p)
            prev = p
    return out


def comparables(model, app, k=5, path=None):
    """Real apps in the same category whose launch-time score was closest to this one."""
    path = Path(path) if path else data_path("comparables.csv.gz")
    if path is None or not path.exists():
        return []
    app = prepare_app(app, model)
    import pandas as pd
    with gzip.open(path, "rt", encoding="utf-8") as f:
        t = pd.read_csv(f)
    s = score_index(model, encode(app, model["terms"]))
    t = t[t["category"] == app["category"]]
    t = t.assign(gap=(t["score"] - s).abs()).sort_values(["gap", "released"], ascending=[True, False])
    return t.head(k).drop(columns="gap").to_dict(orient="records")


def score(app, months=12, model=None, level="80"):
    """Everything the CLI prints, as a dict."""
    model = model or load()
    check_app(app, months)
    app = prepare_app(app, model)
    x = encode(app, model["terms"])
    z = model["z"][level]
    out = {"months": months, "milestones": {}}
    prev = None
    for ms in model["milestone_list"]:
        p, lo, hi = prob(model, ms, x, months, z=z)
        if prev is not None:
            # each milestone is fit on its own, so keep 1M <= 100K <= 10K
            p, lo, hi = min(p, prev["p"]), min(lo, prev["lo"]), min(hi, prev["hi"])
        out["milestones"][ms] = prev = {"p": p, "lo": lo, "hi": hi}
    s = score_index(model, x)
    plo, phi = percentile_interval(model, s, level)
    out["percentile_interval"] = [plo, phi]
    key = int(months) if float(months).is_integer() else None
    if str(key) in model["percentile_installs"]:
        out["installs_interval"] = [installs_at(model, plo, key), installs_at(model, phi, key)]
    out["score"] = s
    return out
