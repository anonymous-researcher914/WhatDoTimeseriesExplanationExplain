"""The TOTAL-effect on-manifold intervention .
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from dagfaith import mediators
from dagfaith.intervention import Edge, _discrepancy, delta_effect

delta_dir = delta_effect


def delta_tot(
    f: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    edge: Edge,
    model,
    D: int,
    T: int,
    *,
    pre: list[tuple[int, int]] | None = None,
    M: list[tuple[int, int]] | None = None,
    w: int | None = None,
    d: str | Callable = "abs",
    B: int = 32,
    rng: np.random.Generator | None = None,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    """The on-manifold TOTAL-effect intervention of edge e = (i, t, j), Eq.(delta-tot):

        1. draw v ~ p(X_i^t | pre(s))                          [source given pre-context only]
        2. draw x_M ~ p(M(s) | X_i^t=v, pre(s))                 [mediators respond, on-manifold]
        3. d(f(x)_j, f(v, x_M, pre(s))_j), averaged over X_eval and B draws

    pre(s) stays FIXED throughout every draw. `model` is the same object
    `dagfaith.mediators.joint_cond_sampler` takes: an `AnalyticGaussianConditional`
    (`.joint_conditional`, option (a): exact, closed-form -- use for the VAR oracle, B1) or a
    plain per-cell `cond_sampler(x, i, t) -> v` (option (b): sequential single-cell fallback --
    approximate, since step 1's source-given-pre draw then also conditions on whatever stale
    values still sit in the not-yet-resampled mediator cells rather than marginalizing them
    out; document this at any B2/real-data call site).

    `pre`/`M` default to `dagfaith.mediators.partition(edge[:2], D, T)`'s temporal split;
    pass them explicitly to override the "contemporaneous -> pre-context" default (e.g.
    `dagfaith.dbn.scenario_I`'s X1->X2 relation is a known SAME-timestep mediation the generic
    partition would otherwise misclassify as pre-context -- see tests/test_tot.py).

    A1 guard: if M(s) is empty (source at the last window position, nothing downstream to
    mediate through), Delta_tot is DEFINED to equal Delta_dir exactly (calls `delta_dir`
    directly) rather than computed via the source-given-pre-only draw, so the two never
    spuriously diverge just from using a different conditioning set for the source itself.

    Returns:
        delta: scalar, the mean effect over X_eval and the B draws.
        delta_per_sample: (n_eval,) the per-x_eval-window mean effect.
    """
    rng = rng if rng is not None else np.random.default_rng()
    i, t, j = edge[0], edge[1], edge[2]
    s = (i, t)

    if pre is None or M is None:
        auto_pre, auto_M = mediators.partition(s, D, T)
        pre = auto_pre if pre is None else pre
        M = auto_M if M is None else M

    X_eval = np.asarray(X_eval, dtype=float)
    n_eval = X_eval.shape[0]
    if n_eval == 0:
        return float("nan"), np.zeros(0, dtype=float)

    if not M:
        cond_sampler = model.as_cond_sampler(rng) if hasattr(model, "as_cond_sampler") else model
        return delta_dir(f, X_eval, edge, cond_sampler, w=w, d=d, B=B, rng=rng, project=project)

    f_base = np.asarray(f(X_eval), dtype=float)
    D_win = X_eval.shape[2]
    pre_idx = mediators.flat_index(pre, D_win)
    joint = hasattr(model, "joint_conditional")

    per_draw = np.zeros((B, n_eval), dtype=float)
    for b in range(B):
        if joint:
            pre_vals = (
                np.stack([X_eval[:, tp, k] for (k, tp) in pre], axis=1)
                if pre else np.zeros((n_eval, 0))
            )
            v = model.joint_conditional(
                pre_idx, pre_vals, np.array([t * D_win + i], dtype=int), rng=rng
            )[:, 0]
        else:
            # option (b): approximate -- conditions on the full complement (incl. stale,
            # not-yet-resampled mediator cells), not pre(s) alone; see docstring above.
            v = np.asarray(model(X_eval, i, t), dtype=float)

        X_int = mediators.joint_cond_sampler(X_eval, s, v, pre, M, model, rng=rng)
        if project is not None:
            X_int = project(X_int)
        f_int = np.asarray(f(X_int), dtype=float)
        per_draw[b] = _discrepancy(f_base, f_int, j, w, d)

    delta_per_sample = per_draw.mean(axis=0)
    delta = float(delta_per_sample.mean())
    return delta, delta_per_sample


def delta_tot_dict(
    f: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    candidate_edges,
    model,
    D: int,
    T: int,
    **kwargs,
) -> dict:
    """{edge: Delta_tot(e)} for every edge in `candidate_edges` -- the total-effect counterpart
    to `dagfaith.intervention.delta_dict`, feeding `dagfaith.omic.evaluate`/`metrics_tot`."""
    return {e: delta_tot(f, X_eval, e, model, D, T, **kwargs)[0] for e in candidate_edges}

def delta_tot_uncoupled(
    f: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    edge: Edge,
    model,
    D: int,
    T: int,
    *,
    pre: list[tuple[int, int]] | None = None,
    M: list[tuple[int, int]] | None = None,
    w: int | None = None,
    d: str | Callable = "abs",
    B: int = 32,
    rng: np.random.Generator | None = None,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    """THIS ESTIMATOR EXISTS TO BE REFUTED. Do not use it to report a real Delta_tot number --
    it is the paper-as-submitted definition (omic_new.md's finding (i)), kept only as an
    explicitly-labelled reference/ablation so the case against it is a runnable comparison, not
    an assertion (Prop. 3, C2).

    d( f(x), f(v, x_M, pre) ) with x_M ~ P_M(v) drawn as a FRESH, INDEPENDENT sample -- same
    outer form as `delta_tot`, but the mediator response is NOT residual-preserving (contrast
    `dagfaith.mediators.joint_cond_sampler`'s coupled x_M^cf(v) = E[M|X_s=v,pre] + (x_M^obs -
    E[M|X_s=x_s^obs,pre])). Both arms are UNCOUPLED here, so resampling M(s) registers as an
    "effect" on its own, independent of whether the source truly influences it:

      - A source with NO direct path and NO influence on its mediators still scores
        Delta_tot_uncoupled > 0 whenever f depends on M(s) at all (the mediators get a genuinely
        fresh, independent realization each of the B draws, which f sees as "changing").
      - It grows (mechanically, not causally) with |M(s)| = (T-t)*D -- MORE mediator cells means
        MORE resampled noise for f to react to -- so it decays with window position t even for a
        variable with zero true mediation, confounding any spatial reading of a Med(e) heatmap.

    `delta_tot`'s own coupled estimator does not have either problem (see its docstring / C1's
    three properties) -- that comparison, not this docstring, is the actual argument; see
    `experiments/run_uncoupled_artifact.py`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    i, t, j = edge[0], edge[1], edge[2]
    s = (i, t)

    if pre is None or M is None:
        auto_pre, auto_M = mediators.partition(s, D, T)
        pre = auto_pre if pre is None else pre
        M = auto_M if M is None else M

    X_eval = np.asarray(X_eval, dtype=float)
    n_eval = X_eval.shape[0]
    if n_eval == 0:
        return float("nan"), np.zeros(0, dtype=float)

    if not M:
        cond_sampler = model.as_cond_sampler(rng) if hasattr(model, "as_cond_sampler") else model
        return delta_dir(f, X_eval, edge, cond_sampler, w=w, d=d, B=B, rng=rng, project=project)

    f_base = np.asarray(f(X_eval), dtype=float)
    D_win = X_eval.shape[2]
    pre_idx = mediators.flat_index(pre, D_win)
    M_idx = mediators.flat_index(M, D_win)
    joint = hasattr(model, "joint_conditional")

    per_draw = np.zeros((B, n_eval), dtype=float)
    for b in range(B):
        if joint:
            pre_vals = (
                np.stack([X_eval[:, tp, k] for (k, tp) in pre], axis=1)
                if pre else np.zeros((n_eval, 0))
            )
            v = model.joint_conditional(pre_idx, pre_vals, np.array([t * D_win + i], dtype=int), rng=rng)[:, 0]
            given_idx = np.concatenate([pre_idx, np.array([t * D_win + i], dtype=int)])
            given_vals = np.concatenate([pre_vals, v[:, None]], axis=1)
            # FRESH, independent draw -- the uncoupled contrast this function exists to refute.
            x_M = model.joint_conditional(given_idx, given_vals, M_idx, rng=rng)
            X_int = X_eval.copy()
            X_int[:, t, i] = v
            for col, (k, tp) in enumerate(M):
                X_int[:, tp, k] = x_M[:, col]
        else:
            v = np.asarray(model(X_eval, i, t), dtype=float)
            X_int = X_eval.copy()
            X_int[:, t, i] = v
            for (k, tp) in sorted(M, key=lambda cell: cell[1]):
                X_int[:, tp, k] = np.asarray(model(X_int, k, tp), dtype=float)  # fresh stochastic draw

        if project is not None:
            X_int = project(X_int)
        f_int = np.asarray(f(X_int), dtype=float)
        per_draw[b] = _discrepancy(f_base, f_int, j, w, d)

    delta_per_sample = per_draw.mean(axis=0)
    delta = float(delta_per_sample.mean())
    return delta, delta_per_sample


def delta_tot_uncoupled_dict(
    f: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    candidate_edges,
    model,
    D: int,
    T: int,
    **kwargs,
) -> dict:
    """{edge: Delta_tot_uncoupled(e)} -- REFERENCE-ONLY, see `delta_tot_uncoupled`."""
    return {e: delta_tot_uncoupled(f, X_eval, e, model, D, T, **kwargs)[0] for e in candidate_edges}
