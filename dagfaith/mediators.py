"""Mediator partition + joint mediator conditional (omic_total.md, Task A1) -- the hard part
the total-effect intervention rests on: drawing several mediator cells JOINTLY given a new
source value and a fixed pre-context, on-manifold, rather than intervening one cell in
isolation the way `dagfaith.intervention.delta_effect` (the direct-effect primitive) does.

Cell convention: (i, t) = (variable index, window position), matching `dagfaith.dbn`/
`dagfaith.cond_baseline`'s own (D, T) layout; flatten index = t*D + i.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

Cell = tuple[int, int]  # (i, t)


def partition(s: Cell, D: int, T: int) -> tuple[list[Cell], list[Cell]]:
    """Split every OTHER cell in a (D, T) window into pre-context and mediators of source
    cell s = (i, t):

        M(s)   = {(k, t'): t < t' <= T-1}      -- temporally-LATER cells, every variable k
        pre(s) = {(k, t'): t' <= t} \\ {s}      -- earlier-or-contemporaneous cells, minus s

    Contemporaneous cells (t' == t, k != i) go to pre-context (conservative default: without a
    known causal order among simultaneous variables, treat them as fixed context rather than
    guessing they are mediators). A caller with domain knowledge that a contemporaneous cell IS
    a mediator (e.g. a same-timestep structural equation) should build pre/M by hand instead of
    calling this function -- see `dagfaith.dbn.scenario_I`'s X1->X2 relation for exactly this
    case, tested directly in tests/test_tot.py rather than through `partition`.
    """
    i, t = s
    pre = [(k, tp) for tp in range(t + 1) for k in range(D) if (k, tp) != (i, t)]
    M = [(k, tp) for tp in range(t + 1, T) for k in range(D)]
    return pre, M


def flat_index(cells: Sequence[Cell], D: int) -> np.ndarray:
    """(i, t) cells -> flatten indices t*D+i (dagfaith.cond_baseline's own convention);
    shared with dagfaith.intervention_tot's source/pre-context index bookkeeping."""
    return np.array([t * D + i for (i, t) in cells], dtype=int)


def joint_cond_sampler(
    x: np.ndarray,
    s: Cell,
    v: np.ndarray,
    pre: list[Cell],
    M: list[Cell],
    model,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Counterfactual (residual-preserving) response of M(s) to X_s: v, holding pre(s) fixed.

        x_M= E[M(s) | X_s=v, pre(s)] + (x_M^obs - E[M(s) | X_s=x_s^obs, pre(s)])

    This is deterministic given (x, v) and stays on-manifold (it is still a valid conditional
    mean plus that row's own genuine residual noise -- not a constant baseline swap). Crucially,
    it reproduces x_M EXACTLY when v == x_s^obs (no-op intervention -> zero effect, always), and
    for a mediator whose conditional mean does not depend on X_s at all (the true relevance-zero
    case), E[.|v] == E[.|x_s^obs] so the residual cancels perfectly and x_M^new == x_M^obs
    exactly -- Delta_tot correctly collapses to 0 for a causally unrelated source, matching
    Delta_dir there, rather than picking up phantom resampling variance.

    Two `model` backends (omic_total.md A1's options (a)/(b)):

      (a) VAR / Gaussian oracle -- `model` exposes `.joint_conditional(given_idx, given_vals,
          target_idx, rng)` (dagfaith.cond_baseline.AnalyticGaussianConditional). Exact,
          closed-form: one Schur-complement mean for every mediator cell at once. Use for B1.

      (b) sequential single-cell -- `model` is a plain per-cell cond_sampler(x, i, t) -> v that
          also exposes `.mean(x, i, t)` (dagfaith.dbn's empirical-Gaussian sampler carries one;
          a trained conditional would need to). M(s) is handled one cell at a time in temporal
          order, each conditional mean computed against a working copy of the window (earlier
          mediators/source already updated, later ones still at their observed values) --
          approximate (marginalizes not-yet-updated later mediators via their STALE values
          rather than a true joint), but reuses machinery that already exists for real data
          (B2). If `model` has no `.mean` (a bare stochastic-only cond_sampler), falls back to
          a fresh stochastic redraw at that cell (the residual-cancellation guarantee above does
          NOT hold in that fallback -- document this at the call site).

    `rng` is accepted for interface symmetry with `dagfaith.intervention.delta_effect`'s
    cond_sampler convention but is UNUSED here: the only randomness `delta_tot` needs is in
    drawing v itself (the source), not in the mediators' response to it.

    Args:
        x: (batch, T, D) ORIGINAL windows (both the observed X_s/M(s) values used to compute
            residuals, and pre(s)'s fixed values, are read from here).
        s: (i, t) source cell.
        v: (batch,) new source values.
        pre: pre-context cells (held fixed at their values in `x`).
        M: mediator cells to update.
        model: see above.
        rng: unused; kept for interface symmetry (see docstring).

    Returns:
        (batch, T, D) windows with X_s set to v and every M(s) cell counterfactually updated;
        pre(s) left at its original value in `x`.
    """
    i, t = s
    x = np.asarray(x, dtype=float)
    batch, T, D = x.shape
    x_new = x.copy()
    x_new[:, t, i] = v
    x_s_obs = x[:, t, i]

    if not M:
        return x_new  # A1 guard: no mediators to update (e.g. s at the last window position)

    if hasattr(model, "joint_conditional"):
        pre_idx = flat_index(pre, D)
        pre_vals = np.stack([x[:, tp, k] for (k, tp) in pre], axis=1) if pre else np.zeros((batch, 0))
        given_idx = np.concatenate([pre_idx, np.array([t * D + i], dtype=int)])
        target_idx = flat_index(M, D)

        given_vals_obs = np.concatenate([pre_vals, x_s_obs[:, None]], axis=1)
        given_vals_new = np.concatenate([pre_vals, v[:, None]], axis=1)
        mean_obs = model.joint_conditional(given_idx, given_vals_obs, target_idx, rng=None)
        mean_new = model.joint_conditional(given_idx, given_vals_new, target_idx, rng=None)

        x_M_obs = np.stack([x[:, tp, k] for (k, tp) in M], axis=1)
        x_M_new = mean_new + (x_M_obs - mean_obs)  # residual-preserving counterfactual shift
        for col, (k, tp) in enumerate(M):
            x_new[:, tp, k] = x_M_new[:, col]
        return x_new

    order = sorted(M, key=lambda cell: cell[1])  # temporal order
    for (k, tp) in order:
        if hasattr(model, "mean"):
            mean_obs = np.asarray(model.mean(x, k, tp), dtype=float)
            mean_new = np.asarray(model.mean(x_new, k, tp), dtype=float)
            x_new[:, tp, k] = mean_new + (x[:, tp, k] - mean_obs)
        else:
            x_new[:, tp, k] = np.asarray(model(x_new, k, tp), dtype=float)
    return x_new
