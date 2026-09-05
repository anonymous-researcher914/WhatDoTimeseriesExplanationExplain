"""Conditional baseline and interpolation path.

Everything downstream (teig.py, telrp.py) is built on \\bar{x}_s^c = E[x_s | x_{-s}] -- the
CONDITIONAL baseline at a source cell s = (i, t) (i: variable index, t: window position,
matching dagfaith.dbn's own (i, t)/(D, T, D_out) convention) -- and the path that interpolates
from that baseline to the observed value, holding the REST of the window fixed (NOT carried
along the data manifold; see `path`'s docstring).

`cond_model` is any object exposing a deterministic `.mean(x, i, t) -> (batch,)` -- what
`dagfaith.dbn.sample_dbn`/`scenario_I`/`scenario_II`'s own `cond_sampler` now carries as an
attribute (the mean of `_make_empirical_cond_sampler`'s empirical-Gaussian regression, with
zero draw noise), or one of this module's `AnalyticGaussianConditional` instances built from a
KNOWN covariance.
"""
from __future__ import annotations

import numpy as np

Cell = tuple[int, int] 

def cond_mean(x: np.ndarray, s: Cell, cond_model) -> np.ndarray:
    """\\bar{x}_s^c = E[x_s | x_{-s}] for a batch of windows x: (batch, T, D)."""
    if not hasattr(cond_model, "mean"):
        raise TypeError(
            "cond_model must expose a deterministic `.mean(x, i, t)` -- e.g. the cond_sampler "
            "returned by dagfaith.dbn.sample_dbn/scenario_I/scenario_II (which carries one as "
            "an attribute), or an AnalyticGaussianConditional from this module."
        )
    i, t = s
    return np.asarray(cond_model.mean(x, i, t), dtype=float)


def cond_support_ok(x: np.ndarray, s: Cell, cond_model, tol: float = 1e-6) -> np.ndarray:
    """Is \\bar{x}_s^c in the conditional support of x_s | x_{-s}?

    Theorem 1 (C1) needs the conditional baseline to lie in the support it conditions on. Every
    conditional this module/dbn.py produces is Gaussian (or a Gaussian point-mass limit), whose
    support is convex, so the baseline is always admissible -- this returns all-True unless
    `cond_model` exposes its own `support_ok(x, i, t)` for a genuinely non-convex conditional
    support, in which case that check is deferred to. Callers (teig.py's `teig_edge`) use this
    to skip/flag edges/instances where the assumption fails, rather than assuming it holds.
    """
    if hasattr(cond_model, "support_ok"):
        i, t = s
        return np.asarray(cond_model.support_ok(x, i, t), dtype=bool)
    batch = np.asarray(x).shape[0]
    return np.ones(batch, dtype=bool)


def path(x: np.ndarray, s: Cell, alpha, cond_model) -> np.ndarray:
    """x with x_s replaced by \\bar{x}_s^c + alpha * (x_s - \\bar{x}_s^c), the REST of the window
    held FIXED.

    This is NOT the value carried along the data manifold (an earlier, wrong formulation) --
    holding the complement fixed while only x_s moves is the correct path for Theorem 1's
    conditional-baseline path integral (teig_telrp.md line 37-39).

    Args:
        x: (batch, T, D) windows.
        s: (i, t) source cell.
        alpha: scalar, or an array broadcastable against x's batch dimension (batch,) -- used to
            build one interpolation step per window in a batch, or a single shared alpha for the
            whole batch.
        cond_model: see module docstring.

    Returns:
        (batch, T, D) windows with only cell (t, i) changed.
    """
    i, t = s
    x = np.asarray(x, dtype=float)
    baseline = cond_mean(x, s, cond_model)  # (batch,)
    alpha = np.asarray(alpha, dtype=float)
    x_s = x[:, t, i]
    interp = baseline + alpha * (x_s - baseline)
    out = x.copy()
    out[:, t, i] = interp
    return out


class AnalyticGaussianConditional:
    """E[x_s | x_{-s}] for a jointly-Gaussian window with a KNOWN covariance Sigma (T*D, T*D),
    flatten index = t*D + i (matches dbn.py's own `_make_empirical_cond_sampler` convention).
    Exact -- no sample-fit estimation error, unlike `_make_empirical_cond_sampler`'s regression
    fit off a finite sample. Gaussian conditionals are always convex, so `support_ok` is
    unconditionally True.
    """

    def __init__(self, Sigma: np.ndarray, mu: np.ndarray | None = None, ridge: float = 1e-10):
        Sigma = np.asarray(Sigma, dtype=float)
        n = Sigma.shape[0]
        if Sigma.shape != (n, n):
            raise ValueError("Sigma must be square")
        self.Sigma = Sigma
        self.mu = np.zeros(n) if mu is None else np.asarray(mu, dtype=float)
        self.ridge = ridge
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def _weights(self, target_idx: int, n: int):
        if target_idx not in self._cache:
            rest = np.array([k for k in range(n) if k != target_idx])
            Sbb = self.Sigma[np.ix_(rest, rest)] + self.ridge * np.eye(len(rest))
            Sab = self.Sigma[target_idx, rest]
            w = np.linalg.solve(Sbb, Sab)
            b = self.mu[target_idx] - w @ self.mu[rest]
            cond_var = max(float(self.Sigma[target_idx, target_idx] - Sab @ w), 0.0)
            self._cache[target_idx] = (w, rest, b, cond_var)
        return self._cache[target_idx]

    def mean(self, x: np.ndarray, i: int, t: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        batch, T, D = x.shape
        n = T * D
        target_idx = t * D + i
        w, rest, b, _cond_var = self._weights(target_idx, n)
        flat = x.reshape(batch, n)
        return flat[:, rest] @ w + b

    def support_ok(self, x: np.ndarray, i: int, t: int) -> np.ndarray:
        batch = np.asarray(x).shape[0]
        return np.ones(batch, dtype=bool)

    def conditional_std(self, i: int, t: int, D: int, T: int) -> float:
        """sd(q_bullet(X_s)) -- omic_new.md D3, Eq.(delta-std): the conditional standard
        deviation of X_i^t given the rest of the window (the SAME leave-one-out conditional
        `.mean`/`.sample` use), exact (sqrt of the Schur-complement conditional variance, no
        sample-fit error). `dagfaith.metrics_tot`/experiments feed this to `delta_std` to
        remove the "conditional spread is a nuisance scale" confound from a raw Delta(e)."""
        n = T * D
        target_idx = t * D + i
        _w, _rest, _b, cond_var = self._weights(target_idx, n)
        return float(np.sqrt(cond_var))

    def sample(self, x: np.ndarray, i: int, t: int, rng: np.random.Generator) -> np.ndarray:
        """A single draw v ~ p(x_i^t | x_-) from the EXACT Gaussian conditional (mean +/-
        sqrt(cond_var) noise, cond_var from the Schur complement of the known Sigma) -- the
        `cond_sampler(x, i, t) -> v` callable `dagfaith.intervention.delta_effect` needs; unlike
        `.mean`, this is stochastic, matching what a real on-manifold intervention draws."""
        x = np.asarray(x, dtype=float)
        batch, T, D = x.shape
        n = T * D
        target_idx = t * D + i
        _w, _rest, _b, cond_var = self._weights(target_idx, n)
        mean = self.mean(x, i, t)
        return mean + rng.normal(scale=np.sqrt(cond_var), size=batch)

    def joint_conditional(
        self,
        given_idx,
        given_vals: np.ndarray,
        target_idx,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Exact joint Gaussian conditional over an ARBITRARY subset `target_idx`, given an
        ARBITRARY subset `given_idx` fixed at `given_vals` -- the Schur complement generalized
        from `.mean`/`.sample`'s "condition on all D*T-1 other cells" to "condition on any
        subset". This is the closed-form joint conditional `dagfaith.mediators.joint_cond_sampler`
        needs for the TOTAL-effect intervention (omic_total.md A1 option (a)): drawing several
        mediator cells JOINTLY given the new source value and a fixed pre-context, rather than
        one cell against the rest of the window.

        Args:
            given_idx: (m,) flatten indices (t*D+i) that are fixed.
            given_vals: (batch, m) their fixed values.
            target_idx: (k,) flatten indices to draw jointly.
            rng: if None, returns the conditional MEAN (deterministic, no noise) -- e.g. for a
                sanity check; if given, draws one stochastic joint sample per batch row.

        Returns:
            (batch, k) conditional mean, or a stochastic joint draw if `rng` is given.
        """
        given_idx = np.asarray(given_idx, dtype=int)
        target_idx = np.asarray(target_idx, dtype=int)
        given_vals = np.asarray(given_vals, dtype=float)
        batch = given_vals.shape[0]

        if given_idx.size == 0:
            cond_mean = np.broadcast_to(self.mu[target_idx], (batch, target_idx.size)).copy()
            cond_cov = self.Sigma[np.ix_(target_idx, target_idx)]
        else:
            Sgg = self.Sigma[np.ix_(given_idx, given_idx)] + self.ridge * np.eye(given_idx.size)
            Sgt = self.Sigma[np.ix_(given_idx, target_idx)]
            Stt = self.Sigma[np.ix_(target_idx, target_idx)]
            W = np.linalg.solve(Sgg, Sgt)                       # (m, k)
            cond_mean = self.mu[target_idx] + (given_vals - self.mu[given_idx]) @ W  # (batch, k)
            cond_cov = Stt - Sgt.T @ W

        if rng is None:
            return cond_mean

        cond_cov = 0.5 * (cond_cov + cond_cov.T) + 1e-10 * np.eye(target_idx.size)
        L = np.linalg.cholesky(cond_cov)
        z = rng.normal(size=(batch, target_idx.size))
        return cond_mean + z @ L.T

    def as_cond_sampler(self, rng: np.random.Generator):
        """Wrap `.sample` as a plain `cond_sampler(x, i, t) -> v` closure over `rng` -- the
        exact-conditional counterpart to `dagfaith.dbn._make_empirical_cond_sampler`'s returned
        callable, for `delta_effect`/`dagfaith.omic.evaluate` callers that want the analytic
        oracle's EXACT conditional (E1's soundness check) rather than a sample-fit estimate (E2)."""

        def cond_sampler(x: np.ndarray, i: int, t: int) -> np.ndarray:
            return self.sample(x, i, t, rng)

        cond_sampler.mean = self.mean
        return cond_sampler


def scenario_I_analytic_cond_model(delta: float, eps_std: float) -> AnalyticGaussianConditional:
    """Exact E[x_s|x_-s] for `dagfaith.dbn.scenario_I`'s generative model: X1 ~ N(0,1),
    X2 = delta*X1 + eps, eps ~ N(0, eps_std^2). D=2, T=1 (index 0 = X1, index 1 = X2, matching
    scenario_I's own convention)."""
    Sigma = np.array([[1.0, delta], [delta, delta**2 + eps_std**2]])
    return AnalyticGaussianConditional(Sigma)


def scenario_II_analytic_cond_model(delta: float) -> AnalyticGaussianConditional:
    """Exact, DEGENERATE (point-mass) E[x_s|x_-s] for `dagfaith.dbn.scenario_II`'s generative
    model: X2 = delta*X1 with NO noise. Conditioning on either variable pins the other exactly
    (E[X1|X2] = X2/delta = X1 on supp(p)) -- this is the C1 counterexample's conditional root.
    """
    Sigma = np.array([[1.0, delta], [delta, delta**2]])
    return AnalyticGaussianConditional(Sigma)


def analytic_gaussian_cond_model_for_ar_inputs(
    D: int, T: int, ar: float = 0.5, noise_std=0.1
) -> AnalyticGaussianConditional:
    """Exact E[x_s|x_-s] for `dagfaith.dbn.sample_dbn`'s default window-generation process, when
    every variable has nonzero process noise: each variable evolves as an independent,
    non-stationary-start AR(1) chain (dbn.py's own `ar = 0.5`). Builds the (T*D, T*D) covariance
    directly (block-diagonal across variables, since they evolve independently) instead of
    fitting it from a finite sample, so both the gradient (via `dagfaith.oracle.AnalyticOracle`)
    and the conditional baseline are exact -- teig_telrp.md Task 4's soundness/recovery
    experiment ("gradients and the conditional baseline are exact").

    NOTE: does not cover dbn.py's DETERMINISTIC (`noise_std[d] == 0`, tied-to-variable-0)
    mechanism used to build a manifold at D>2 (Scenario II generalized). A deterministic linear
    relation has R^2 = 1, so `_make_empirical_cond_sampler`'s regression fit recovers it EXACTLY
    even from a finite sample (there is no noise for an estimator to have variance against) --
    use `sample_dbn`'s own returned `cond_sampler` (its `.mean`) for that case instead of this
    helper.
    """
    nstd = np.broadcast_to(np.asarray(noise_std, float), (D,)).astype(float)
    if np.any(nstd <= 0.0):
        raise ValueError(
            "analytic_gaussian_cond_model_for_ar_inputs assumes every variable has nonzero "
            "process noise; use sample_dbn's own empirical cond_sampler.mean for deterministic "
            "(tied) variables instead"
        )
    Sigma = np.zeros((T * D, T * D))
    for d in range(D):
        var = np.zeros(T)
        var[0] = nstd[d] ** 2
        for t in range(1, T):
            var[t] = ar**2 * var[t - 1] + nstd[d] ** 2
        for s_t in range(T):
            for t_t in range(T):
                Sigma[s_t * D + d, t_t * D + d] = ar ** abs(t_t - s_t) * var[min(s_t, t_t)]
    return AnalyticGaussianConditional(Sigma)
