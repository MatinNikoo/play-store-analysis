"""End to end on the fake dataset: the old analysis script and every study stage run and write sane outputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_analysis_script_runs_on_fixture(tmp_path, raw):
    (tmp_path / "data").mkdir()
    raw.to_csv(tmp_path / "data" / "google_playstore_2021.csv", index=False)
    r = subprocess.run([sys.executable, str(ROOT / "analysis.py")], cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "MPLBACKEND": "Agg"})
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert s["concentration"]["apps"] == len(raw) - 7
    assert 0.5 < s["model"]["auc_gradient_boosting"] <= 1
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5


def test_study_outputs(pipeline):
    res = pipeline / "results" / "study"
    rk = json.loads((res / "ranking.json").read_text())
    for k, v in rk["main"].items():
        assert 0 <= v["auc"] <= 1 and v["lo"] <= v["auc"] <= v["hi"], k
    assert rk["main"]["gbm_launch"]["rate"] == rk["main"]["gbm_old"]["rate"]   # same test apps
    ms = json.loads((res / "milestones.json").read_text())
    assert ms["link"] in {"cloglog", "probit", "logit"}
    for name in ["10K", "100K", "1M"]:
        f = [ms["npmle"][name][t]["F"] for t in ["6", "12", "24", "36"]]
        assert f == sorted(f)
    assert 0.6 < ms["conformal"]["80"]["coverage"] < 0.95
    ck = json.loads((res / "checks.json").read_text())
    r = ck["cohort_rates"]["11 to 13 months"]
    assert 0 <= r["100K"]["observed"] <= r["10K"]["observed"] <= 1 and r["100K"]["lo"] <= r["100K"]["observed"]
    assert 0 <= ck["npmle_interval_simulation"]["coverage"] <= 1
    assert "abl:launch minus title" not in rk and "launch minus title" in rk["ablation"]
    assert (res / "market.csv").exists()
    a = json.loads((res / "associations.json").read_text())
    assert a["playbook"]
    figs = {p.name for p in (pipeline / "figures").glob("*.png")}
    for i in range(6, 17):
        assert any(f.startswith(f"{i}_") for f in figs), i
