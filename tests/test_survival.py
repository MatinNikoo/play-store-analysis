"""Checks the current status (one snapshot) machinery on data where the truth is known."""

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from study import milestones as MS


@pytest.mark.parametrize("seed", range(3))
def test_pava_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(300)
    w = rng.integers(1, 5, 300).astype(float)
    ours = MS.pava(y, w)
    ref = IsotonicRegression().fit(np.arange(300), y, sample_weight=w).predict(np.arange(300))
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    assert np.all(np.diff(ours) >= -1e-15)


def simulate(n, seed, shape=0.8, scale=30.0, beta=0.0, x=None):
    """Weibull time to milestone, seen once at a uniform age: current status data."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.5, 66, n)
    x = np.zeros(n) if x is None else x
    t = scale * np.exp(-beta * x / shape) * rng.weibull(shape, n)
    return age, (t <= age).astype(float), x


def test_npmle_recovers_known_curve():
    age, d, _ = simulate(60_000, 0)
    a, F_ = MS.npmle(np.round(age, 1), d)
    truth = lambda t: 1 - np.exp(-(t / 30.0) ** 0.8)
    for t in [6, 12, 24, 48]:
        assert MS.f_at(a, F_, t) == pytest.approx(truth(t), abs=0.03)


def test_subsampling_interval_covers_truth():
    pts = np.array([12.0, 36.0])
    truth = 1 - np.exp(-(pts / 30.0) ** 0.8)
    covered = []
    for seed in range(12):
        age, d, _ = simulate(60_000, 200 + seed)
        groups = np.arange(len(age)) // 4
        est, lo, hi = MS.npmle_ci(np.round(age, 1), d, groups, pts, B=60)
        assert np.all(lo <= est) and np.all(est <= hi)
        covered += list((lo <= truth) & (truth <= hi))
    # nominal 95%. These intervals run short in finite samples (about 80% here, see results/study/checks.json),
    # so the test only guards against them being useless
    assert np.mean(covered) >= 0.7


def _cohort_data(n, seed, drift):
    """Age is launch age. With drift, newer launches (small age) reach the milestone faster."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.5, 66, n)
    scale = 30.0 * (0.5 + 0.5 * age / 66) if drift else 30.0
    t = scale * rng.weibull(0.8, n)
    return age, (t <= age).astype(float)


@pytest.mark.parametrize("drift", [False, True])
def test_forward_check_flags_cohort_drift(drift):
    """Fit on old launches (ages 24 and up), score young ones. The Weibull form extrapolates fine when nothing
    drifts, and the average prediction is clearly off when newer cohorts behave differently."""
    age, d = _cohort_data(150_000, 7, drift)
    old, young = age >= 24, age < 24
    r = MS.fit_glm(np.empty((old.sum(), 0)), age[old], d[old], "cloglog")
    p = MS.predict(r, np.empty((young.sum(), 0)), age[young], "cloglog")
    gap = d[young].mean() - p.mean()
    if drift:
        assert abs(gap) > 0.015
    else:
        assert abs(gap) < 0.006


def test_cloglog_glm_recovers_weibull_parameters():
    # Weibull PH: P(T <= a | x) = 1 - exp(-exp(shape * log a - shape * log scale + beta x))
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, 40_000).astype(float)
    age, d, _ = simulate(40_000, 3, beta=0.7, x=x)
    r = MS.fit_glm(x[:, None], age, d, "cloglog")
    assert r.params[1] == pytest.approx(0.8, abs=0.05)                     # shape
    assert r.params[0] == pytest.approx(-0.8 * np.log(30.0), abs=0.1)      # -shape * log(scale)
    assert r.params[2] == pytest.approx(0.7, abs=0.06)
    tr = MS.time_ratios(r, [{"name": "x"}])["x"]
    assert tr["time_ratio"] < 1                                            # x reaches it faster
    assert tr["lo"] < np.exp(-0.7 / 0.8) < tr["hi"]


def test_conformal_table_covers_on_exchangeable_data():
    rng = np.random.default_rng(4)
    s = rng.normal(size=40_000)
    y = 1 / (1 + np.exp(-(s + rng.normal(size=40_000))))      # percentile-like target that depends on s
    tab = MS.conformal_table(s[:20_000], y[:20_000], 0.8)
    cov = MS.inside(s[20_000:], y[20_000:], tab).mean()
    assert cov == pytest.approx(0.8, abs=0.015)
    assert len(tab["lo"]) == 20 and all(l <= h for l, h in zip(tab["lo"], tab["hi"]))


def test_age_only_npmle_beats_nothing_on_holdout():
    age, d, _ = simulate(30_000, 5)
    a, F_ = MS.npmle(np.round(age[:20_000], 1), d[:20_000])
    p = np.clip(np.interp(age[20_000:], a, F_), 1e-6, 1 - 1e-6)
    ll = -np.mean(d[20_000:] * np.log(p) + (1 - d[20_000:]) * np.log(1 - p))
    base = d[:20_000].mean()
    ll0 = -np.mean(d[20_000:] * np.log(base) + (1 - d[20_000:]) * np.log(1 - base))
    assert ll < ll0
