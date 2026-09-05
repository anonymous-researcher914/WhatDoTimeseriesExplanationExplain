r"""
omic.py — On-Manifold Interventional Model-Dependency Consistency.

Implements exactly the paper's definitions:
  - omic           (Eq. 3-4)
  - OMIC ranking curve OMIC_k (Eq. 5-10)
  - AUOMIC (normalized trapezoidal area under OMIC curve) (Eq. 11-12)

At ranking level k the retained top-k claimed edges E_k^+ are
compared against the FULL complement E_k^- = E \ E_k^+ (Eq. 8). The only
stochasticity is in Delta itself (the on-manifold intervention estimate), not in
any edge subsample.

Delta(e) is supplied by intervention.delta_effect (on-manifold conditional
intervention). This module is pure scoring: it consumes precomputed Delta values.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Sequence, Hashable


def _phi(x: float, y: float) -> float:
    """phi(x,y) = 1 if x>y, 1/2 if x==y, 0 if x<y  (Eq. 4)."""
    if x > y:
        return 1.0
    if x == y:
        return 0.5
    return 0.0


def _concordance(delta_plus: np.ndarray, delta_minus: np.ndarray) -> float:
    """
    Mean tie-aware concordance (1/|A||B|) sum_a sum_b phi(a,b)  over a in
    delta_plus, b in delta_minus. Vectorized. Returns value in [0,1].
    Undefined (NaN) if either set is empty.
    """
    a = np.asarray(delta_plus, dtype=float)
    b = np.asarray(delta_minus, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    # pairwise comparison matrix
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return (gt + 0.5 * eq) / (a.size * b.size)


def omic_support(delta_present: np.ndarray, delta_absent: np.ndarray) -> float:
    """
    omic = (1/|E+||E-|) sum_{e+ in E+} sum_{e- in E-} phi(Delta(e+), Delta(e-)).
    In [0,1]; 0.5 = chance. NaN if E+ or E- empty (anti-gaming: a method that
    claims nothing or everything is unscored).
    """
    return _concordance(delta_present, delta_absent)


def omic_ranking_curve(
    claimed_edges: Sequence[Hashable],
    attribution: dict,          # a(e), magnitude used for ranking (Eq. 5)
    delta: dict,                # Delta(e) for every candidate edge e in E
    candidate_edges: Sequence[Hashable],
    rho_max: float = 0.75,
):
    r"""
    Compute the OMIC ranking curve (Eq. 9) and the rho grid (Eq. 6).

    - Rank the CLAIMED edges by |a(e)| descending (Eq. 5).
    - For k = 1..floor(rho_max*|E+|):
        E_k^+ = top-k claimed edges           (Eq. 7)
        E_k^- = E \ E_k^+                      (Eq. 8)  [FULL complement, no sample]
        OMIC_k = concordance(Delta over E_k^+, Delta over E_k^-)   (Eq. 9)
    Returns (rho_grid, omic_values) as np.ndarrays, each length K.

    Notes:
      * candidate_edges is the full E; E_k^- includes claimed-absent edges AND
        the not-yet-retained claimed edges, exactly as E \ E_k^+.
      * ranking is within the explanation support E+ only (Eq. 5-7).
    """
    Eplus = list(claimed_edges)
    n_plus = len(Eplus)
    if n_plus == 0:
        return np.array([]), np.array([])

    # Eq. 5: rank claimed edges by |a(e)| descending
    ranked = sorted(Eplus, key=lambda e: abs(attribution[e]), reverse=True)

    E = list(candidate_edges)
    delta_all = np.array([delta[e] for e in E], dtype=float)
    # index of each edge in E for fast complement masking
    idx_of = {e: i for i, e in enumerate(E)}

    K = int(np.floor(rho_max * n_plus))
    K = max(K, 1)

    rho = np.array([k / n_plus for k in range(1, K + 1)], dtype=float)  # Eq. 6
    omic = np.empty(K, dtype=float)

    for k in range(1, K + 1):
        Ek_plus = ranked[:k]                                   # Eq. 7
        plus_mask = np.zeros(len(E), dtype=bool)
        for e in Ek_plus:
            plus_mask[idx_of[e]] = True
        d_plus = delta_all[plus_mask]
        d_minus = delta_all[~plus_mask]                        # Eq. 8: E \ E_k^+
        omic[k - 1] = _concordance(d_plus, d_minus)            # Eq. 9

    return rho, omic

def omic_ranking_curve_restricted(
    claimed_edges: Sequence[Hashable],
    attribution: dict,
    delta: dict,
    rho_max: float = 0.75,
):
    r"""Same construction as `omic_ranking_curve` (Eq. 5-7, 9-10), except E_k^- = the claimed
    edges NOT yet retained at level k (Eplus[k:] after ranking), instead of the full complement
    E \ E_k^+. No `candidate_edges` argument -- this variant never looks outside E+.

    Returns (rho_grid, omic_values), same shapes/semantics as `omic_ranking_curve`. NaN at any
    k where the "remaining claimed" side is empty (k == n_plus, i.e. rho_max*n_plus rounds up
    to the whole claim) -- `auomic` already drops NaN levels, consistent with the rest of the
    module's anti-gaming convention.
    """
    Eplus = list(claimed_edges)
    n_plus = len(Eplus)
    if n_plus == 0:
        return np.array([]), np.array([])

    ranked = sorted(Eplus, key=lambda e: abs(attribution[e]), reverse=True)
    delta_ranked = np.array([delta[e] for e in ranked], dtype=float)

    K = max(int(np.floor(rho_max * n_plus)), 1)
    rho = np.array([k / n_plus for k in range(1, K + 1)], dtype=float)
    omic = np.empty(K, dtype=float)
    for k in range(1, K + 1):
        d_plus = delta_ranked[:k]
        d_minus = delta_ranked[k:]  # remaining CLAIMED edges only -- H2's own restriction
        omic[k - 1] = _concordance(d_plus, d_minus)
    return rho, omic


def kendall_tau_restricted(claimed_edges: Sequence[Hashable], attribution: dict, delta: dict) -> float:
    """Kendall's tau between a(e)
    (the ranking attribution) and Delta(e), computed ONLY over the claimed set E+ -- "did it
    order its own claim correctly", a single s-independent number (+1 for a claim ranked
    exactly like Delta, -1 for the exact reverse, 0 for no relationship). NaN for |E+| < 2
    (tau undefined) or if every Delta value tied (no variation to rank against).
    """
    Eplus = list(claimed_edges)
    if len(Eplus) < 2:
        return np.nan
    a = np.array([abs(attribution[e]) for e in Eplus], dtype=float)
    d = np.array([delta[e] for e in Eplus], dtype=float)
    n = len(Eplus)
    concordant = discordant = 0
    for p in range(n):
        for q in range(p + 1, n):
            sa = np.sign(a[p] - a[q])
            sd = np.sign(d[p] - d[q])
            if sa == 0 or sd == 0:
                continue
            if sa == sd:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return np.nan
    return (concordant - discordant) / denom


def auomic(rho: np.ndarray, omic: np.ndarray) -> float:
    """
    AUOMIC_{rho_max} =
        [ sum_{k=1}^{K-1} (OMIC_k + OMIC_{k+1})/2 * (rho_{k+1} - rho_k) ]
        / (rho_K - rho_1)                                          (Eq. 11)

    i.e. trapezoidal integral of OMIC over [rho_1, rho_K], normalized by the
    rho-interval length so AUOMIC is a rho-average of OMIC in [0,1].
    rho_1 = 1/|E+| (Eq. 12), rho_K = rho_max (approx, discretized).
    Returns NaN if fewer than 2 valid points.
    """
    rho = np.asarray(rho, dtype=float)
    omic = np.asarray(omic, dtype=float)
    valid = ~np.isnan(omic)
    rho, omic = rho[valid], omic[valid]
    if rho.size < 2:
        return np.nan
    trap = np.sum((omic[:-1] + omic[1:]) / 2.0 * (rho[1:] - rho[:-1]))
    span = rho[-1] - rho[0]             # rho_K - rho_1
    if span <= 0:
        return np.nan
    return trap / span

def evaluate(
    claimed_edges,
    attribution,
    delta,
    candidate_edges,
    rho_max: float = 0.75,
):
    """Return dict with omic, the ranking curve, and AUOMIC."""
    Eplus = list(claimed_edges)
    Eall = list(candidate_edges)
    Eminus = [e for e in Eall if e not in set(Eplus)]

    d_present = np.array([delta[e] for e in Eplus], dtype=float)
    d_absent = np.array([delta[e] for e in Eminus], dtype=float)

    rho, omic = omic_ranking_curve(
        claimed_edges, attribution, delta, candidate_edges, rho_max=rho_max
    )
    return {
        "omic_support": omic_support(d_present, d_absent),   # Eq. 3
        "rho": rho,                                          # Eq. 6
        "omic_curve": omic,                                  # Eq. 9-10
        "auomic": auomic(rho, omic),                         # Eq. 11
    }