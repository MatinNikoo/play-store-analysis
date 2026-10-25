import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from study import metrics as M


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn_with_ties_and_weights(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(500) < 0.2
    s = np.round(rng.normal(size=500) + y, 1)          # rounding makes lots of ties
    w = rng.integers(0, 4, 500).astype(float)
    assert M.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert M.auc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_auc_edge_cases():
    y = np.array([0, 0, 1, 1])
    assert M.auc(y, [1, 2, 3, 4]) == 1.0
    assert M.auc(y, [4, 3, 2, 1]) == 0.0
    assert M.auc(y, [1, 1, 1, 1]) == 0.5
    assert np.isnan(M.auc([1, 1], [0.2, 0.3]))


def test_stratified_auc_only_compares_within_strata():
    # inside each stratum the score is perfect, but strata are shifted so pooled AUC is worse
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    g = np.array([0, 0, 1, 1])
    assert M.stratified_auc(y, s, g) == 1.0
    assert M.auc(y, s) < 1.0


def test_calibration_slope_and_intercept():
    rng = np.random.default_rng(0)
    p = expit(rng.normal(-2.5, 1.0, 200_000))
    y = (rng.random(len(p)) < p).astype(float)
    good = M.calibration(y, p)
    assert good["slope"] == pytest.approx(1.0, abs=0.03)
    assert good["intercept"] == pytest.approx(0.0, abs=0.03)
    # predictions twice as extreme on the logit scale -> slope near 0.5
    assert M.calibration(y, expit(2 * logit(p)))["slope"] == pytest.approx(0.5, abs=0.03)
    # predictions too high across the board -> negative intercept
    assert M.calibration(y, expit(logit(p) + 0.7))["intercept"] == pytest.approx(-0.7, abs=0.04)


def test_brier_skill_and_reliability():
    rng = np.random.default_rng(1)
    y = (rng.random(10_000) < 0.1).astype(float)
    assert M.brier_skill(y, np.full_like(y, 0.1), 0.1) == pytest.approx(0.0)
    bins = M.reliability(y, rng.random(10_000), 10)
    assert sum(b["n"] for b in bins) == 10_000
    assert all(b["lo"] <= b["obs"] <= b["hi"] for b in bins)


def test_captured():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.arange(10)[::-1].astype(float)               # app 0 ranked first, app 9 last
    assert M.captured(y, s, 0.10) == 0.5
    assert M.captured(y, s, 1.0) == 1.0


def test_cluster_bootstrap_ci():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.2).astype(float)
    s1 = y + rng.normal(0, 1, 3000)
    s2 = y + rng.normal(0, 3, 3000)
    groups = rng.integers(0, 300, 3000)
    r = M.bootstrap_auc(y, {"a": s1, "b": s2}, groups, B=200)
    assert r["a"]["lo"] < r["a"]["auc"] < r["a"]["hi"]
    assert r["a"]["diff_vs_a"] == 0 and r["a"]["diff_lo"] == 0
    assert r["b"]["diff_hi"] < 0                        # b is clearly worse
