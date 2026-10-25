"""
Metric code. Written out by hand (instead of only calling sklearn) so the bootstrap can reuse one sort
across hundreds of resamples and so tests can check every piece on small cases.
"""

import numpy as np
from scipy.special import expit, logit


def _ranks(s):
    uniq, inv = np.unique(s, return_inverse=True)
    return inv, len(uniq)


def auc(y, s, w=None, _inv=None):
    """Weighted ROC AUC, ties count half. Same answer as sklearn's roc_auc_score(sample_weight=w)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    inv, k = _ranks(s) if _inv is None else _inv
    pos = np.bincount(inv, weights=w * y, minlength=k)
    neg = np.bincount(inv, weights=w * (1 - y), minlength=k)
    below = np.cumsum(neg) - neg
    den = pos.sum() * neg.sum()
    if den == 0:
        return np.nan
    return float((pos * (below + 0.5 * neg)).sum() / den)


def stratified_auc(y, s, strata):
    """AUC computed inside each stratum (like release month) and pooled by number of pos/neg pairs.
    It only compares apps that had the same time to grow."""
    y, s, strata = np.asarray(y), np.asarray(s), np.asarray(strata)
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        npos, nneg = y[m].sum(), (1 - y[m]).sum()
        if npos == 0 or nneg == 0:
            continue
        num += auc(y[m], s[m]) * npos * nneg
        den += npos * nneg
    return num / den


def captured(y, s, frac=0.10):
    """Share of all positives found in the top `frac` of scores."""
    y = np.asarray(y)
    k = int(round(frac * len(y)))
    top = np.argsort(-np.asarray(s), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p, base):
    """1 - Brier / Brier of always predicting `base`. Above 0 means better than the base rate."""
    y = np.asarray(y, float)
    return 1 - brier(y, p) / brier(y, np.full_like(y, base))


def _newton_logit(y, X, offset, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = expit(X @ beta + offset)
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def calibration(y, p):
    """Calibration intercept and slope. Slope: fit logit P(y) = a + b logit(p), report b (1 is ideal,
    under 1 means predictions are too extreme). Intercept: calibration in the large, fit with the
    slope fixed at 1 (0 is ideal, negative means predictions are too high on average)."""
    y = np.asarray(y, float)
    lp = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    a_b = _newton_logit(y, np.column_stack([np.ones_like(lp), lp]), np.zeros_like(lp))
    a = _newton_logit(y, np.ones((len(y), 1)), lp)
    return {"intercept": float(a[0]), "slope": float(a_b[1])}


def reliability(y, p, bins=10):
    """Mean prediction vs observed rate in equal-count bins, with Wilson 95% intervals."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    out = []
    for idx in np.array_split(order, bins):
        n, k = len(idx), y[idx].sum()
        r = k / n
        z = 1.96
        mid = (r + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        out.append({"n": int(n), "pred": float(p[idx].mean()), "obs": float(r), "lo": float(mid - half),
                    "hi": float(mid + half)})
    return out


def ece(y, p, bins=10):
    r = reliability(y, p, bins)
    n = sum(b["n"] for b in r)
    return float(sum(b["n"] / n * abs(b["pred"] - b["obs"]) for b in r))


# --------------------------------------------------------------------------
# bootstrap that resamples whole developers, since apps by one developer aren't independent
# --------------------------------------------------------------------------
def cluster_weights(groups, B, seed=0):
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    G = codes.max() + 1
    rng = np.random.default_rng(seed)
    for _ in range(B):
        counts = np.bincount(rng.integers(0, G, G), minlength=G)
        yield counts[codes].astype(float)


def bootstrap_auc(y, scores, groups, B=500, seed=0):
    """AUC with developer-clustered percentile 95% CIs for several score vectors at once, plus
    the paired difference of each against the first one (same resamples)."""
    y = np.asarray(y, float)
    invs = {k: _ranks(s) for k, s in scores.items()}
    point = {k: auc(y, s, _inv=invs[k]) for k, s in scores.items()}
    draws = {k: [] for k in scores}
    for w in cluster_weights(groups, B, seed):
        for k in scores:
            draws[k].append(auc(y, None, w, _inv=invs[k]))
    first = next(iter(scores))
    out = {}
    for k in scores:
        d = np.array(draws[k])
        diff = d - np.array(draws[first])
        out[k] = {"auc": point[k], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                  "se": float(d.std(ddof=1)),
                  f"diff_vs_{first}": point[k] - point[first],
                  "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5))}
    return out
