"""OMIC_tot / AUOMIC_tot + mediation -- a THIN wrapper: all scoring
is `dagfaith.omic.evaluate`, unchanged; only the per-edge effect vector fed to it differs
(Delta_dir vs Delta_tot, from `dagfaith.intervention`/`dagfaith.intervention_tot`).

OMIC_tot ALONE does not expose
mediation -- a mediated edge scores WELL on it, by design (it IS predictive of the outcome,
just not through a direct structural path). The mediation is the GAP per-edge Med(e) = Delta_tot(e) - Delta_dir(e).
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from dagfaith.omic import evaluate


def evaluate_dir_tot(
    claimed_edges: Sequence[Hashable],
    attribution: dict,
    delta_dir: dict,
    delta_tot: dict,
    candidate_edges: Sequence[Hashable],
    rho_max: float = 0.75,
) -> dict:
    """Run `dagfaith.omic.evaluate` twice against the SAME claimed_edges/attribution/candidate
    set -- once with Delta_dir, once with Delta_tot -- and report both plus the mediation gap.
    No new scoring logic: `dagfaith.omic`'s OMIC_support/omic_ranking_curve/AUOMIC are reused
    verbatim, just fed a different `delta` dict each call.
    """
    dir_result = evaluate(claimed_edges, attribution, delta_dir, candidate_edges, rho_max=rho_max)
    tot_result = evaluate(claimed_edges, attribution, delta_tot, candidate_edges, rho_max=rho_max)
    return {
        "omic_support_dir": dir_result["omic_support"],
        "rho_dir": dir_result["rho"],
        "omic_curve_dir": dir_result["omic_curve"],
        "auomic_dir": dir_result["auomic"],
        "omic_support_tot": tot_result["omic_support"],
        "rho_tot": tot_result["rho"],
        "omic_curve_tot": tot_result["omic_curve"],
        "auomic_tot": tot_result["auomic"],
        "gap": tot_result["auomic"] - dir_result["auomic"],
    }


def mediation(
    delta_dir: dict,
    delta_tot: dict,
    claimed_edges: Sequence[Hashable] | None = None,
    *,
    m: float = 0.5,
    m_grid: np.ndarray | Sequence[float] | None = None,
) -> dict:
    """Per-edge and aggregate mediation diagnostics.

    - Med(e) = Delta_tot(e) - Delta_dir(e), for every edge Delta_tot has a value for.
    - mediation set (at the given `m`) = { e in E+ : Med(e)/Delta_tot(e) >= m }. Restricted to
      `claimed_edges` (E+) when given; over all scored edges otherwise.
    - mediation_fraction = |mediation set| / |E+| at `m` -- the B3 headline number, now a
      STATED threshold rather than an implicit one.
    - fraction_curve/m_grid: the SAME fraction computed at every m in `m_grid` (default a 21-
      point grid over [0, 1]) -- report the curve beside the headline value, not instead of it
      (F2's own instruction), so a reader can see how threshold-sensitive the headline is.
    """
    common = [e for e in delta_tot if e in delta_dir]
    med = {e: float(delta_tot[e] - delta_dir[e]) for e in common}
    ratio = {e: med[e] / delta_tot[e] for e in common if delta_tot[e] != 0}

    universe = list(claimed_edges) if claimed_edges is not None else common
    universe = [e for e in universe if e in ratio]

    grid = np.asarray(m_grid if m_grid is not None else np.linspace(0.0, 1.0, 21), dtype=float)
    if not universe:
        return {
            "med": med, "ratio": ratio, "mediation_set": [], "mediation_fraction": float("nan"),
            "m": m, "m_grid": grid, "fraction_curve": np.full(grid.shape, np.nan),
        }

    r = np.array([ratio[e] for e in universe], dtype=float)
    mediation_set = [e for e in universe if ratio[e] >= m]
    fraction = len(mediation_set) / len(universe)
    fraction_curve = np.array([(r >= mm).mean() for mm in grid])
    return {
        "med": med, "ratio": ratio, "mediation_set": mediation_set, "mediation_fraction": fraction,
        "m": m, "m_grid": grid, "fraction_curve": fraction_curve,
    }


def delta_std(delta: dict, cond_sd_by_source: dict) -> dict:
    """Eq.(delta-std): the standardized effect
        Delta_std(e) = Delta(e) / sd(q_bullet(X_s))
    -- removes the conditional-spread NUISANCE SCALE from a raw Delta(e): Delta(e) is
    (roughly) |sensitivity| * E|on-manifold shift|, and the shift's own typical size scales
    with the source cell's conditional standard deviation, so two edges with identical TRUE
    sensitivity but different conditional variance at their source cell get different raw
    Delta purely from that nuisance scale (H5's "homoscedasticity accident", generalized).

    Args:
        delta: {edge: Delta(e)}, edge = (i, t, j).
        cond_sd_by_source: {(i, t): sd} -- keyed by SOURCE CELL only (shared across every
            target j at that cell), e.g. from
            `dagfaith.cond_baseline.AnalyticGaussianConditional.conditional_std(i, t, D, T)`.

    Returns {edge: Delta_std(e)} -- only for edges whose source cell has a positive sd in
    `cond_sd_by_source` (silently drops the rest, matching `evaluate`'s own "score what you
    can" convention rather than raising on a partial `cond_sd_by_source`).
    """
    out = {}
    for e, v in delta.items():
        sd = cond_sd_by_source.get((e[0], e[1]))
        if sd and sd > 0:
            out[e] = v / sd
    return out
