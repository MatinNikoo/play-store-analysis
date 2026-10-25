"""Paths, plot style and small helpers shared by the study scripts."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# STUDY_OUT lets the tests run the whole pipeline on a tiny fixture without touching the real outputs
OUT = Path(os.environ.get("STUDY_OUT", "."))
RES = OUT / "results" / "study"
FIG = OUT / "figures"
MODELS = OUT / "models"
FEATURES = Path(os.environ.get("STUDY_FEATURES", "data/study_features.pkl"))
FAST = os.environ.get("STUDY_FAST") == "1"     # fixture runs: fewer bootstraps, smaller models
SEED = 0

BLUE, GRAY, DARK, LIGHT, RED = "#2F4BC4", "#8A94A1", "#333333", "#C9D2F5", "#B23A3A"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left",
})
pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


def dirs():
    for d in (RES, FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write(name, obj):
    (RES / name).write_text(json.dumps(_clean(obj), indent=2))


def read(name):
    return json.loads((RES / name).read_text())
