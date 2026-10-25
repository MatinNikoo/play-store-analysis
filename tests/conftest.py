import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixture import make_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return make_raw()


@pytest.fixture(scope="session")
def built(raw):
    from study import data, prepare
    df = data.clean(raw)
    for col in ["Category", "Content Rating", "model"]:
        df[col] = df[col].astype("category")
    return prepare.build(df)


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, built):
    """Runs every study stage on the fixture in a temp folder, the same way `make all` does on real data."""
    out = tmp_path_factory.mktemp("run")
    feats = out / "feat.pkl"
    built.to_pickle(feats)
    env = {**os.environ, "STUDY_FAST": "1", "STUDY_OUT": str(out), "STUDY_FEATURES": str(feats),
           "PLAYODDS_MODELS": str(out / "models"), "OMP_NUM_THREADS": "2", "PYTHONPATH": str(ROOT)}
    for stage in ["study.ranking", "study.milestones", "study.checks", "study.market", "study.associations"]:
        r = subprocess.run([sys.executable, "-m", stage], cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, f"{stage} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    return out
