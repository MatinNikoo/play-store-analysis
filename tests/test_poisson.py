import numpy as np
import statsmodels.api as sm

from study.associations import poisson_cluster


def test_poisson_cluster_matches_statsmodels():
    rng = np.random.default_rng(3)
    n, g = 6000, 900
    groups = rng.integers(0, g, n)
    X = np.column_stack([np.ones(n), rng.integers(0, 2, n), rng.integers(0, 2, n), rng.normal(size=n)])
    y = rng.poisson(np.exp(-2.5 + 0.7 * X[:, 1] - 0.4 * X[:, 2] + 0.3 * X[:, 3] + rng.normal(0, 0.3, g)[groups])).clip(0, 1)
    ref = sm.GLM(y.astype(float), X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    beta, cov = poisson_cluster(X.astype(np.float32), y.astype(float), groups, chunk=2500)
    assert np.allclose(beta, ref.params, atol=2e-5)
    assert np.allclose(np.sqrt(np.diag(cov)), np.sqrt(np.diag(ref.cov_params())), rtol=1e-3)
