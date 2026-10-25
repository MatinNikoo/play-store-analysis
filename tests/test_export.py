"""The exported model file has to give the same numbers as the fitted statsmodels GLM, in Python and in
JavaScript. Runs on the fixture export, and on the real export in models/ when it exists."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from playodds import model as PM

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def exports(pipeline):
    out = [pipeline / "models"]
    if (ROOT / "models" / "milestones.json").exists() and (ROOT / "models" / "examples.json").exists():
        out.append(ROOT / "models")
    return out


def test_python_scorer_reproduces_statsmodels(pipeline):
    for d in exports(pipeline):
        model = PM.load(d / "milestones.json")
        ex = json.loads((d / "examples.json").read_text())
        assert len(ex) > 20
        worst = 0.0
        for e in ex:
            x = PM.encode(e["app"], model["terms"])
            for ms, p in e["p"].items():
                worst = max(worst, abs(PM.prob(model, ms, x, e["months"]) - p))
        assert worst < 1e-9, (d, worst)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_javascript_scorer_matches_python(pipeline):
    for d in exports(pipeline):
        model = PM.load(d / "milestones.json")
        ex = json.loads((d / "examples.json").read_text())
        r = subprocess.run([NODE, str(ROOT / "models" / "score.js"), str(d / "milestones.json"),
                            str(d / "examples.json")], capture_output=True, text=True, check=True)
        js = json.loads(r.stdout)
        for e, j in zip(ex, js):
            for ms, p in e["p"].items():
                assert j["p"][ms] == pytest.approx(p, abs=1e-9)
            py = PM.score(e["app"], months=12, model=model, level="80")
            for ms in model["milestone_list"]:
                for k in ["p", "lo", "hi"]:
                    assert j["scored"]["milestones"][ms][k] == pytest.approx(py["milestones"][ms][k], abs=1e-9)
            assert j["scored"]["percentile_interval"] == pytest.approx(py["percentile_interval"], abs=1e-12)
            if "installs_interval" in py:
                assert j["scored"]["installs_interval"] == pytest.approx(py["installs_interval"], rel=1e-9)


def test_encode_one_matches_encode_frame(pipeline):
    import pandas as pd
    model = PM.load(pipeline / "models" / "milestones.json")
    ex = json.loads((pipeline / "models" / "examples.json").read_text())
    apps = [e["app"] for e in ex] + [{"category": model["categories"][0], "name": "Ünïcode 3 words"}]
    frame = PM.encode_frame(pd.DataFrame([{**PM.INPUTS, **a} for a in apps]), model["terms"])
    one = np.array([PM.encode(a, model["terms"]) for a in apps])
    np.testing.assert_allclose(frame, one, atol=1e-12)


def test_format_fields(pipeline):
    model = PM.load(pipeline / "models" / "milestones.json")
    assert model["format"] == "playodds-milestones/1"
    assert model["link"] in {"cloglog", "probit", "logit"}
    k = len(model["terms"])
    for ms in model["milestone_list"]:
        m = model["milestones"][ms]
        assert len(m["coef"]) == k
        cov = np.array(m["cov"])
        assert cov.shape == (k + 2, k + 2)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(cov) > -1e-10)
    for lv, c in model["conformal"].items():
        assert len(c["lo"]) == len(c["edges"]) + 1
        assert all(0 <= l <= h <= 1 for l, h in zip(c["lo"], c["hi"]))
