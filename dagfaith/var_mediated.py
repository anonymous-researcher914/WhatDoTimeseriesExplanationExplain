"""VAR(1) with a PLANTED mediator -- the controlled generative model
for the direct-vs-total conflation demonstration: known ground truth for which input cells
influence the forecast target directly vs. only through a mediator.

Fixed D=4 layout (variable index):
    0 = X1  the MEDIATED source: influences Y only by driving X2 (index 1) one step later.
    1 = X2  the mediator: X2_t = m * X1_{t-1} + ar * X2_{t-1} + noise, and Y reads X2 at the
            LAST window position -- so X1@t=0 has Delta_dir(->Y) == 0 (Y's structural formula
            never mentions X1) but Delta_tot(->Y) > 0 (moving X1 moves X2, which Y reads).
    2 = X3  the DIRECT driver: self-AR only, Y reads X3 at the last window position directly.
    3 = X4  filler: self-AR only, no influence on Y at all (neither direct nor mediated).

The whole (window, target) system is linear-Gaussian by construction, so the window's exact
(T*D, T*D) covariance is derived analytically from the VAR(1) recursion (`_var1_window_covariance`)
rather than fit from a finite sample -- an EXACT oracle conditional (omic_total.md A1 option (a)),
matching how `dagfaith.dbn`'s own scenario_I/II analytic conditionals avoid sample-fit error.
"""
from __future__ import annotations

import numpy as np

from dagfaith.cond_baseline import AnalyticGaussianConditional


def _var1_window_covariance(A: np.ndarray, Q: np.ndarray, T: int, Sigma0: np.ndarray | None = None) -> np.ndarray:
    """Exact (T*D, T*D) covariance of a window Z_0..Z_{T-1} from Z_0 ~ N(0, Sigma0),
    Z_t = A Z_{t-1} + eps_t, eps_t ~ N(0, Q) i.i.d. Flatten index = t*D + i (matches
    `dagfaith.cond_baseline`'s own convention).

    Cov(Z_t, Z_t) = A Cov(Z_{t-1}, Z_{t-1}) A^T + Q   (the usual Lyapunov recursion)
    Cov(Z_t, Z_s) = A^{t-s} Cov(Z_s, Z_s)             for t > s  (future noise is independent of Z_s)
    """
    D = A.shape[0]
    Sigma0 = Q.copy() if Sigma0 is None else Sigma0
    P = [Sigma0]
    for _ in range(1, T):
        P.append(A @ P[-1] @ A.T + Q)

    Apow: dict[int, np.ndarray] = {0: np.eye(D)}

    def apow(k: int) -> np.ndarray:
        if k not in Apow:
            Apow[k] = apow(k - 1) @ A
        return Apow[k]

    Sigma = np.zeros((T * D, T * D))
    for t in range(T):
        for u in range(T):
            if t == u:
                block = P[t]
            elif t > u:
                block = apow(t - u) @ P[u]
            else:
                block = (apow(u - t) @ P[t]).T
            Sigma[t * D:(t + 1) * D, u * D:(u + 1) * D] = block
    return Sigma


def sample_var_mediated(
    n: int = 500,
    T: int = 3,
    m: float = 0.7,
    ar: float = 0.5,
    beta: float = 1.0,
    gamma: float = 1.0,
    noise_std: float = 0.3,
    target_noise_std: float = 0.05,
    seed: int = 0,
) -> dict:
    """Sample the planted-mediator VAR (see module docstring for the fixed D=4 layout).

    Args:
        n: number of windows.
        T: window length (>= 2, so the mediator has a later position to occupy after t=0).
        m: X1(t-1) -> X2(t) coupling strength (the planted mediation edge).
        ar: each variable's own AR(1) self-coupling.
        beta: X2@(T-1) -> Y structural weight (mediator's own direct edge into f).
        gamma: X3@(T-1) -> Y structural weight (the known direct edge).
        noise_std: per-variable process noise (shared across all 4 variables).
        target_noise_std: Y's own observation noise (f itself, used for delta effects, excludes
            this -- it is added only to the returned Y column, for realism as training data).
        seed: RNG seed.

    Returns a dict: X (n,T,4), Y (n,1), f (deterministic ground-truth callable, no target
    noise), B (D,T,1) direct structural coefficients, Gstar_dir/Gstar_tot boolean graphs,
    cond_model (AnalyticGaussianConditional over the window, exact), mediator_edge/direct_edge
    (the two edges of interest, dbn.py's (i,t,j) convention), params.
    """
    if T < 2:
        raise ValueError("sample_var_mediated needs T >= 2 (mediator occupies a later position)")
    D = 4
    rng = np.random.default_rng(seed)

    A = np.zeros((D, D))
    A[0, 0] = ar
    A[1, 1] = ar
    A[1, 0] = m       # planted mediation edge: X2_t <- X1_{t-1}
    A[2, 2] = ar
    A[3, 3] = ar
    Q = (noise_std ** 2) * np.eye(D)

    Sigma_window = _var1_window_covariance(A, Q, T, Sigma0=Q)
    mu = np.zeros(T * D)
    X = rng.multivariate_normal(mu, Sigma_window, size=n).reshape(n, T, D)

    def f(x: np.ndarray) -> np.ndarray:
        """(n, T, D) -> (n, 1) -- dagfaith.intervention's documented f(x) -> (n, D_out) shape."""
        x = np.asarray(x, dtype=float)
        return (beta * x[..., T - 1, 1] + gamma * x[..., T - 1, 2])[..., None]

    Y = f(X) + rng.normal(scale=target_noise_std, size=(n, 1))

    B = np.zeros((D, T, 1))
    B[1, T - 1, 0] = beta
    B[2, T - 1, 0] = gamma
    Gstar_dir = B != 0.0

    Gstar_tot = Gstar_dir.copy()
    Gstar_tot[0, 0, 0] = True  # X1@t=0 -> Y: no direct edge, but a nonzero total effect

    cond_model = AnalyticGaussianConditional(Sigma_window, mu)

    return {
        "X": X, "Y": Y, "f": f, "B": B,
        "Gstar_dir": Gstar_dir, "Gstar_tot": Gstar_tot,
        "cond_model": cond_model,
        "A": A, "Q": Q, "Sigma_window": Sigma_window,
        "D": D, "T": T, "D_out": 1,
        "mediator_edge": (0, 0, 0),
        "direct_edge": (2, T - 1, 0),
        "params": dict(
            m=m, ar=ar, beta=beta, gamma=gamma,
            noise_std=noise_std, target_noise_std=target_noise_std, n=n, T=T, seed=seed,
        ),
    }


def oracle_effect_targets(gen: dict, candidate_edges) -> tuple[dict, dict]:
    """True DIRECT and TOTAL oracle targets, derived ANALYTICALLY from `gen`'s own A (VAR(1)
    companion) and B (direct readout) matrices -- omic_new.md D1's "do NOT sample-fit them".

    true_direct(i,t,j=0) = |B[i,t,0]| -- the generator's own direct structural coefficient
        (paper's |A_ell[j,i]| specialized to this generator's single-lag, multi-component
        LINEAR READOUT Y = beta*Z_1[T-1] + gamma*Z_2[T-1], rather than a further next-step VAR
        forecast -- B already IS this generator's direct-effect ground truth by construction).

    true_total(i,t,j=0) = |w . A^h[:, i]|, h = T-1-t, w = (0, beta, gamma, 0, ...) the readout
        weight vector -- generalizes the paper's impulse-response coefficient Psi_h[j,i]
        (Psi_0=I, Psi_h = sum_ell A_ell Psi_{h-ell}, here just A^h since p=1) to a
        WEIGHTED-SUM readout: how much a perturbation at (i,t) propagates through the VAR(1)
        recursion into window position T-1, weighted by how much the target actually reads
        each propagated component there. At h=0 (t=T-1) this reduces to |w[i]| = true_direct
        exactly, matching Delta_tot=Delta_dir at the last window position (the A1 guard).
    """
    A, B, T, D = gen["A"], gen["B"], gen["T"], gen["D"]
    beta, gamma = gen["params"]["beta"], gen["params"]["gamma"]
    w = np.zeros(D)
    w[1] = beta
    w[2] = gamma

    Apow: dict[int, np.ndarray] = {0: np.eye(D)}

    def apow(h: int) -> np.ndarray:
        if h not in Apow:
            Apow[h] = apow(h - 1) @ A
        return Apow[h]

    true_direct, true_total = {}, {}
    for e in candidate_edges:
        i, t, j = e
        true_direct[e] = float(abs(B[i, t, j]))
        h = T - 1 - t
        true_total[e] = float(abs(w @ apow(h)[:, i]))
    return true_direct, true_total
