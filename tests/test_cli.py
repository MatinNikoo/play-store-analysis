import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from playodds import model as PM

ROOT = Path(__file__).resolve().parents[1]


def run(pipeline, *args):
    env = {**os.environ, "PLAYODDS_MODELS": str(pipeline / "models"), "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-m", "playodds", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def test_cli_text(pipeline):
    model = PM.load(pipeline / "models" / "milestones.json")
    r = run(pipeline, "--category", model["categories"][0], "--name", "Block Drop", "--earlier-apps", "3",
            "--months-since-first-app", "14")
    assert r.returncode == 0, r.stderr
    assert "10K installs" in r.stdout and "1M installs" in r.stdout
    assert "too narrow" in r.stdout and "coefficient range" in r.stdout


def test_cli_json_is_ordered_and_bounded(pipeline):
    model = PM.load(pipeline / "models" / "milestones.json")
    r = run(pipeline, "--category", model["categories"][1], "--json", "--level", "90")
    out = json.loads(r.stdout)
    m = out["milestones"]
    assert 0 <= m["1000000"]["p"] <= m["100000"]["p"] <= m["10000"]["p"] <= 1
    for v in m.values():
        assert v["lo"] <= v["p"] <= v["hi"]
    lo, hi = out["percentile_interval"]
    assert 0 <= lo <= hi <= 1
    assert isinstance(out["comparables"], list)
    for c in out["comparables"]:
        assert c["category"] == model["categories"][1]


def test_cli_rejects_unknown_category(pipeline):
    r = run(pipeline, "--category", "Not A Category")
    assert r.returncode == 2
    assert "unknown category" in r.stderr


def test_cli_lists_categories(pipeline):
    r = run(pipeline, "--list-categories")
    assert r.returncode == 0
    assert len(r.stdout.split("\n")) >= 5


def test_more_time_never_lowers_the_odds(pipeline):
    model = PM.load(pipeline / "models" / "milestones.json")
    app = {"category": model["categories"][0], "earlier_apps": 2}
    p = [PM.score(app, months=t, model=model)["milestones"]["10000"]["p"] for t in [3, 6, 12, 24, 48]]
    assert p == sorted(p)


def test_score_rejects_bad_category(pipeline):
    model = PM.load(pipeline / "models" / "milestones.json")
    with pytest.raises(ValueError):
        PM.score({"category": "nope"}, model=model)


@pytest.mark.parametrize("flag,value", [("--months", "0"), ("--months", "120"), ("--months", "-3"),
                                        ("--months-since-first-app", "-3"), ("--earlier-apps", "-1"),
                                        ("--same-day-apps", "-2")])
def test_cli_rejects_values_outside_what_the_model_saw(pipeline, flag, value):
    model = PM.load(pipeline / "models" / "milestones.json")
    r = run(pipeline, "--category", model["categories"][0], f"{flag}={value}")
    assert r.returncode == 2
    assert "Traceback" not in r.stderr and "warning" not in r.stderr.lower()
    assert flag.lstrip("-").replace("-", "_") in r.stderr or "months must be" in r.stderr


def test_score_accepts_the_edges_of_the_fitted_range(pipeline):
    model = PM.load(pipeline / "models" / "milestones.json")
    app = {"category": model["categories"][0]}
    for t in [0.5, 66]:
        assert 0 <= PM.score(app, months=t, model=model)["milestones"]["10000"]["p"] <= 1
