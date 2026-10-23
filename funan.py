"""
Fuzzy natural neighbors for outlier detection -- Python implementation.

Reference: R. Saltos, R. Weber, W. Pedrycz, "Fuzzy natural neighbors for outlier
detection", Applied Soft Computing 186 (2026) 114114.

Implements FuNaN I, FuNaN II, FuNaN III, FuDSR (the paper's four methods) and the
crisp DSR baseline, following the paper's equations and the authors' official MATLAB
code (github.com/rjsaltos1989/FuNaNOutlier).

All computations are done in row blocks so that only one n x n integer matrix
(the neighbour ranks) is kept in memory. For n = 10,000 that is ~400 MB.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def normalize(X, method="minmax"):
    """Scale the data. 'minmax' = MATLAB normalize(X,'range'); 'zscore' = standardization."""
    X = np.asarray(X, dtype=np.float64)
    if method == "minmax":
        lo, hi = X.min(axis=0), X.max(axis=0)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)  # constant columns -> 0
        return (X - lo) / rng
    if method == "zscore":
        mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)  # MATLAB uses the n-1 std
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd
    if method == "none":
        return X
    raise ValueError(f"unknown normalization: {method}")


# ---------------------------------------------------------------------------
# Step 1: neighbour ranks  r_ij  (Eq. 10)
# ---------------------------------------------------------------------------
def neighbour_ranks(X, block=1000):
    """
    R[i, j] = position of x_j in the distance-sorted neighbour list of x_i.
    The point itself gets rank 0, its nearest neighbour rank 1, ..., farthest rank n-1.
    Ties are broken by index (stable sort), like MATLAB's knnsearch.
    """
    n = X.shape[0]
    R = np.empty((n, n), dtype=np.int32)
    ranks = np.arange(n, dtype=np.int32)
    for s in range(0, n, block):
        e = min(s + block, n)
        D = cdist(X[s:e], X)
        D[np.arange(e - s), np.arange(s, e)] = -1.0  # make sure x_i is first in its own list
        order = np.argsort(D, axis=1, kind="stable")
        R[np.arange(s, e)[:, None], order] = ranks[None, :]
    return R


def membership_block(R, s, e):
    """
    Fuzzy natural neighborhood memberships for rows s..e-1 (Eqs. 10-12):
        mu_ij = 1 / r_ij,   upsilon_ij = min(mu_ij, mu_ji) = 1 / max(r_ij, r_ji).
    The diagonal (a point with itself) gets 0 so it never counts as a friend.
    """
    Rmax = np.maximum(R[s:e, :], R[:, s:e].T).astype(np.float64)
    with np.errstate(divide="ignore"):
        U = 1.0 / Rmax
    U[np.arange(e - s), np.arange(s, e)] = 0.0
    return U


# ---------------------------------------------------------------------------
# Step 2: per-point statistics needed by the four detectors
# ---------------------------------------------------------------------------
def fuzzy_statistics(X, R, block=1000, need_dsr=False):
    """
    Returns a dict with, for every point x_i:
      card   : fuzzy cardinality  sum_j upsilon_ij                     (Eq. 14, FuNaN I)
      psi    : sum of cumulative sums of sorted memberships              (Eq. 15, FuNaN II)
      fudsr_num : sum_j (1 - upsilon_ij) d(x_i, x_j)                     (Eq. 20 numerator)
      sumW   : sum_j 1 / (1 + d(x_i, x_j)^2), self included             (Eq. 17 with m = 2)
      dsr    : (optional) crisp DSR, mean distance to natural neighbours (Eq. 4)
    """
    n = X.shape[0]
    card = np.empty(n)
    psi = np.empty(n)
    fudsr_num = np.empty(n)
    sumW = np.empty(n)
    # weight of the k-th largest membership inside psi: it appears in (n - k) cumulative sums
    w = (n - np.arange(1, n + 1)).astype(np.float64)

    lam = natural_eigenvalue(R) if need_dsr else None
    dsr = np.full(n, np.nan) if need_dsr else None

    for s in range(0, n, block):
        e = min(s + block, n)
        U = membership_block(R, s, e)
        D = cdist(X[s:e], X)

        card[s:e] = U.sum(axis=1)
        U_sorted = -np.sort(-U, axis=1)            # descending; the self-zero ends up last
        psi[s:e] = U_sorted @ w
        fudsr_num[s:e] = ((1.0 - U) * D).sum(axis=1)  # self term is 0 because d_ii = 0
        sumW[s:e] = (1.0 / (1.0 + D ** 2)).sum(axis=1)

        if need_dsr:
            # Theorem 1: crisp natural neighbours = alpha-cut at alpha = 1/lambda
            NN = U >= 1.0 / lam - 1e-12
            cnt = NN.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                dsr[s:e] = (D * NN).sum(axis=1) / cnt   # 0/0 -> NaN, as in the MATLAB code

    out = dict(card=card, psi=psi, fudsr_num=fudsr_num, sumW=sumW)
    if need_dsr:
        out["dsr"] = dsr
        out["lambda"] = lam
    return out


def natural_eigenvalue(R):
    """
    Natural neighbour eigenvalue lambda found by NaN Search (Algorithm 1, Definition 3).
    In round r every point adds its r-th nearest neighbour; x_j gets its first reverse
    neighbour in round min_i R[i, j]. Stop when nobody is left without a reverse
    neighbour or when that number stops decreasing.
    """
    n = R.shape[0]
    Rm = R.astype(np.int64)
    Rm[np.diag_indices(n)] = n + 1                # ignore self
    first_round = Rm.min(axis=0)                  # round in which x_j first becomes someone's neighbour
    counts = np.bincount(first_round, minlength=n + 2)
    without = n
    for r in range(1, n):
        new_without = without - counts[r]
        if new_without == 0 or new_without == without:
            return r
        without = new_without
    return n - 1


# ---------------------------------------------------------------------------
# Step 3: the four detectors (higher score = more outlying for all of them)
# ---------------------------------------------------------------------------
def harmonic(n):
    return np.sum(1.0 / np.arange(1, n + 1))


def funan_scores(stats, n, gammas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), m=2.0):
    """Turn the per-point statistics into outlier scores."""
    expo = 2.0 / (m - 1.0)
    scores = {
        # FuNaN I / II: low cardinality = few friends = outlier, so use the reciprocal
        "FuNaN I": 1.0 / stats["card"],
        "FuNaN II": 1.0 / stats["psi"],
        # FuDSR (Eq. 20)
        "FuDSR": stats["fudsr_num"] / stats["card"],
    }
    # FuNaN III (Eqs. 16-19)
    A, B = n / 2.0, n * (harmonic(n) - 1.0)
    mu_L = 1.0 - ((stats["psi"] - A) / (B - A)) ** expo           # "few friends"
    mu_F = 1.0 - ((stats["sumW"] - 1.0) / (n - 1.0)) ** expo      # "far away"
    scores["FuNaN III"] = {g: (mu_L ** g) * (mu_F ** (1.0 - g)) for g in gammas}
    if "dsr" in stats:
        scores["DSR"] = stats["dsr"]
    return scores


# ---------------------------------------------------------------------------
# Evaluation metrics (match MATLAB perfcurve as used by the authors)
# ---------------------------------------------------------------------------
def auc(y, score):
    """Area under the ROC curve. Points with NaN scores are dropped (as perfcurve does)."""
    keep = ~np.isnan(score)
    return roc_auc_score(y[keep], score[keep])


def max_precision(y, score):
    """
    The 'Precision' reported in the paper: the authors' code takes max(Prec(2:end))
    from perfcurve, i.e. the highest precision reached at any score threshold
    (flag every point scoring >= t, for each distinct score t).
    """
    keep = ~np.isnan(score)
    y, score = y[keep], score[keep]
    order = np.argsort(-score, kind="stable")
    s_sorted, y_sorted = score[order], y[order]
    tp = np.cumsum(y_sorted == 1)
    flagged = np.arange(1, len(y_sorted) + 1)
    # only evaluate at the last position of each group of tied scores (= a threshold)
    last_of_group = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    return float(np.max(tp[last_of_group] / flagged[last_of_group]))


def evaluate(y, scores):
    """AUC and precision for every method; FuNaN III is averaged over its gamma values."""
    res = {}
    for name, sc in scores.items():
        if isinstance(sc, dict):  # FuNaN III: one score vector per gamma
            aucs = [auc(y, v) for v in sc.values()]
            precs = [max_precision(y, v) for v in sc.values()]
            res[name] = dict(AUC=float(np.mean(aucs)), Precision=float(np.mean(precs)),
                             AUC_per_gamma=dict(zip(sc.keys(), aucs)),
                             Precision_per_gamma=dict(zip(sc.keys(), precs)))
        else:
            res[name] = dict(AUC=float(auc(y, sc)), Precision=max_precision(y, sc))
    return res


def run_all(X, y, normalization="minmax", with_dsr=True, block=1000,
            gammas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), m=2.0):
    """Full pipeline for one dataset."""
    Xn = normalize(X, normalization)
    R = neighbour_ranks(Xn, block)
    stats = fuzzy_statistics(Xn, R, block, need_dsr=with_dsr)
    del R
    scores = funan_scores(stats, Xn.shape[0], gammas, m)
    res = evaluate(np.asarray(y).ravel().astype(int), scores)
    if with_dsr:
        res["_lambda"] = stats["lambda"]
    return res, scores
