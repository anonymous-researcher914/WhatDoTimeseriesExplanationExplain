"""The on-manifold interventional effect Delta(e).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Edge = tuple[int, int, int]  # (i, t1, j): source var, source window-position, target coord


def _discrepancy(f_base: np.ndarray, f_int: np.ndarray, j: int, w: int | None, d) -> np.ndarray:
    """d(f(x)_{j,w}, f(x^{i<-v})_{j,w}) for a batch of (base, intervened) forecast pairs.

    f_base/f_int: (n, D_out) if w is None, else (n, D_out, W).
    """
    if w is None:
        a, b = f_base[:, j], f_int[:, j]
    else:
        a, b = f_base[:, j, w], f_int[:, j, w]

    if d == "abs":
        return np.abs(b - a)
    if callable(d):
        return np.asarray(d(a, b), dtype=float)
    raise ValueError(f"unsupported discrepancy d={d!r} (use 'abs' or a callable(a, b) -> array)")


def delta_effect(
    f: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    edge: Edge,
    cond_sampler,
    *,
    w: int | None = None,
    d: str | Callable = "abs",
    B: int = 32,
    rng: np.random.Generator | None = None,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    """The on-manifold interventional effect of edge e = (i, t1, j) (source variable i, source
    window-position t1, target coordinate j -- dbn.py's own B[i, t, j]/Gstar[i, t, j] indexing,
    a raw window position, not a "lag"):

        Delta(e) = E_{x~X_eval} E_{v~p(X_i^{t1}|x_-)}[ d(f(x)_{j,w}, f(x^{i<-v})_{j,w}) ]

    v is drawn ON-MANIFOLD from cond_sampler(x, i, t1) -- NEVER a fixed baseline (a constant
    swap would push x off-manifold and reintroduce the very C1 error OMIC is meant to detect).

    Scenario II hook: when the generator provides `project_to_manifold` (its degenerate
    manifold X2 = delta*X1 means "condition on the rest of the window" alone can't be used to
    draw v; its own cond_sampler instead draws a fresh marginal for X1 and expects the caller
    to re-establish the constraint), pass it as `project` -- applied to x after the swap.

    Args:
        f: black-box forecaster, X:(n,T,D) numpy -> (n,D_out) [or (n,D_out,W) if `w` is used].
        X_eval: (n, T, D) windows to evaluate the effect over.
        edge: (i, t1, j).
        w: optional horizon index into a (n, D_out, W) forecast; None for (n, D_out).
        d: "abs" (|Delta mean forecast|) or a callable(a, b) -> array for a custom/
            distributional discrepancy (e.g. TV between predictive dists).
        B: number of on-manifold draws per x.
        project: optional callable applied to the intervened window (e.g. Scenario II's
            project_to_manifold).

    Returns:
        delta: scalar, the mean effect over X_eval and the B draws.
        delta_per_sample: (n_eval,) the per-x_eval-window mean effect (averaged over the B
            draws only) -- e.g. for a per-edge standard error.
    """
    rng = rng if rng is not None else np.random.default_rng()
    i, t1, j = edge[0], edge[1], edge[2]

    X_eval = np.asarray(X_eval, dtype=float)
    n_eval = X_eval.shape[0]
    if n_eval == 0:
        return float("nan"), np.zeros(0, dtype=float)

    f_base = np.asarray(f(X_eval), dtype=float)

    per_draw = np.zeros((B, n_eval), dtype=float)
    for b in range(B):
        v = np.asarray(cond_sampler(X_eval, i, t1), dtype=float)
        X_int = X_eval.copy()
        X_int[:, t1, i] = v
        if project is not None:
            X_int = project(X_int)
        f_int = np.asarray(f(X_int), dtype=float)
        per_draw[b] = _discrepancy(f_base, f_int, j, w, d)

    delta_per_sample = per_draw.mean(axis=0)
    delta = float(delta_per_sample.mean())
    return delta, delta_per_sample


def delta_dict(
    f: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    candidate_edges,
    cond_sampler,
    **kwargs,
) -> dict:
    """{edge: Delta(e)} for every edge in `candidate_edges`, via `delta_effect`.

    The batch convenience `dagfaith.omic`'s new (Eq. 3-12) API needs: `omic_ranking_curve`/
    `evaluate` consume a precomputed `delta` dict keyed by the SAME candidate edges used to
    build `attribution`/`claimed_edges`, and (per the paper's Eq. 8) score every ranking level
    against the FULL complement E \\ E_k^+ -- so, unlike the old S^- subsample, Delta must be
    computed for every edge in the candidate set, not just the claimed ones.

    kwargs are forwarded to `delta_effect` (w, d, B, rng, project).
    """
    return {e: delta_effect(f, X_eval, e, cond_sampler, **kwargs)[0] for e in candidate_edges}
