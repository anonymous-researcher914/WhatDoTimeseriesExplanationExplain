"""Data generation for MDDAG-recovery experiments.

Encodes dependencies in the INPUT -> FORECAST-TARGET structure of the MDDAG
(Definition 2): a window of T input timesteps over D variables feeds a forecast
target, and edges run only from input positions to targets. Supports:

  * a general sparse DBN over input positions -> target(s), with PER-VARIABLE /
    per-position noise so that specific structural equations can be made
    DETERMINISTIC (required for Scenario II's degenerate manifold);
  * an explicit `mediation` option (an input drives the target only through a
    mediator input), the structural motif behind Scenario I;
  * exact reproduction of Scenarios I and II, each returning the analytic models
    f1/f2 (Scenario I) and f1/f3 (Scenario II) together with their true MDDAGs.

Graph convention (input->target): G has shape (D, T, D_out).  G[i, t, j] is True
iff input X_t^{(i)} has a DIRECT edge to forecast target j.  The analytic
`gf_*` helpers return graphs in this convention so SHD/F1 line up with the
explainers' induced graphs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def sample_dbn(
    D: int,
    T: int,
    n: int,
    D_out: int = 1,
    sparsity: float = 0.3,
    nonlinear: bool = False,
    noise_std=0.1,
    seed: int = 0,
):
    """Sample a DBN in the input->forecast-target convention.

    The window X has shape (n, T, D).  A forecast target Y has shape (n, D_out),
    a function of the whole window through a coefficient tensor B of shape
    (D, T, D_out): Y_j = sum_{i,t} B[i,t,j] X_t^{(i)} (+ small tanh if nonlinear)
    (+ target noise).

    Args:
        D:        number of input variables.
        T:        window length (input positions t = 0..T-1).
        n:        number of windows.
        D_out:    number of forecast-target variables.
        sparsity: fraction of (D*T*D_out) input->target edges that are nonzero.
        nonlinear: add a small tanh nonlinearity to the target mean.
        noise_std: scalar, OR array of shape (D,) giving per-INPUT-VARIABLE
                   process noise on the window, OR the string 'input_free' to
                   draw the window i.i.d. standard normal.  Per-variable zeros
                   make that variable deterministic given its parents (used for
                   Scenario II).  A separate small target noise is always added
                   to Y unless target_noise is overridden via the scenarios.
        seed:     RNG seed.

    Returns:
        X:      (n, T, D) input windows.
        Y:      (n, D_out) forecast targets.
        B:      (D, T, D_out) input->target coefficient tensor.
        Gstar:  (D, T, D_out) boolean input->target data graph (B != 0).
        cond_sampler: fn(x, i, t) -> (batch,) on-manifold draws of X_t^{(i)}
                      given the rest of the window (empirical-Gaussian).
    """
    rng = np.random.default_rng(seed)
    if not (0 < sparsity < 1):
        raise ValueError("sparsity must lie strictly between 0 and 1")

    if isinstance(noise_std, str) and noise_std == "input_free":
        X = rng.normal(size=(n, T, D))
    else:
        nstd = np.broadcast_to(np.asarray(noise_std, float), (D,)).astype(float)
        X = np.zeros((n, T, D))
        ar = 0.5  # mild autocorrelation so mediation/on-manifold tests bite
        for t in range(T):
            for d in range(D):
                prev = X[:, t - 1, d] if t > 0 else 0.0
                X[:, t, d] = ar * prev + rng.normal(scale=max(nstd[d], 0.0), size=n)
                if nstd[d] == 0.0 and t == 0:
                    # fully deterministic variable needs a seed value; tie it to
                    # variable 0 so it lies on a manifold (Scenario-II style)
                    X[:, t, d] = X[:, t, 0]

    B = np.zeros((D, T, D_out))
    total = D * T * D_out
    k = max(1, min(total - 1, round(total * sparsity)))
    for idx in rng.choice(total, size=k, replace=False):
        i = idx // (T * D_out)
        rem = idx % (T * D_out)
        t = rem // D_out
        j = rem % D_out
        B[i, t, j] = rng.normal(scale=0.8)

    Gstar = B != 0.0

    Y = np.einsum("itj,nti->nj", B, X)
    if nonlinear:
        Y = Y + 0.1 * np.tanh(Y)
    Y = Y + rng.normal(scale=0.1, size=(n, D_out))

    cond_sampler = _make_empirical_cond_sampler(X, rng)
    return X, Y, B, Gstar, cond_sampler


def _make_empirical_cond_sampler(X: np.ndarray, rng: np.random.Generator):
    """Gaussian conditional p(X_t^{(i)} | rest of window) fit from the sample.

    Uses the empirical covariance of the flattened window; exact for
    linear-Gaussian data, a linear-Gaussian approximation otherwise. Returns a
    callable cond_sampler(x, i, t).
    """
    n, T, D = X.shape
    flat = X.reshape(n, T * D)
    mu = flat.mean(axis=0)
    Sigma = np.cov(flat, rowvar=False) + 1e-8 * np.eye(T * D)
    cache: dict[int, tuple[np.ndarray, float, np.ndarray, float]] = {}

    def _weights(target_idx: int):
        if target_idx not in cache:
            rest = np.array([k for k in range(T * D) if k != target_idx])
            Srr = Sigma[np.ix_(rest, rest)]
            Str = Sigma[target_idx, rest]
            w = np.linalg.solve(Srr, Str)              # regression weights
            cond_var = Sigma[target_idx, target_idx] - Str @ w
            cond_var = max(float(cond_var), 1e-10)
            b = mu[target_idx] - w @ mu[rest]          # intercept
            cache[target_idx] = (w, cond_var, rest, b)
        return cache[target_idx]

    def _mean(x: np.ndarray, i: int, t: int) -> np.ndarray:
        x = np.asarray(x, float)
        batch, _Tw, Dw = x.shape
        target_idx = t * Dw + i
        w, _cond_var, rest, b = _weights(target_idx)
        xf = x.reshape(batch, x.shape[1] * Dw)
        return xf[:, rest] @ w + b

    def cond_sampler(x: np.ndarray, i: int, t: int) -> np.ndarray:
        x = np.asarray(x, float)
        batch, Tw, Dw = x.shape
        target_idx = t * Dw + i
        _w, cond_var, _rest, _b = _weights(target_idx)
        mean = _mean(x, i, t)
        return mean + rng.normal(scale=np.sqrt(cond_var), size=batch)
    cond_sampler.mean = _mean
    return cond_sampler

def scenario_I(n: int, delta: float = 0.8, beta: float = 1.0, gamma: float = 1.0,
               eps_std: float = 1.0, seed: int = 0):
    """Scenario I: X2 = delta*X1 + eps  (autocorrelation), target driven via X2.

    Inputs live at a single input position t=0 with D=2 variables
    (X^(1)=index 0, X^(2)=index 1); the forecast target is X^(3) (D_out=1).

    Returns X (n,1,2), and the two analytic models with their MDDAGs:
        f1(x) = beta * x2                      # X1 fully mediated -> NO X1->tgt edge
        f2(x) = beta * x2 + gamma * x1         # direct X1->tgt edge present
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = delta * x1 + rng.normal(scale=eps_std, size=n)   # eps NOT zero: full support
    X = np.stack([x1, x2], axis=1)[:, None, :]            # (n, 1, 2)

    def f1(x):
        x = np.asarray(x, float)
        return beta * x[..., 0, 1]                        # beta * X2
    def f2(x):
        x = np.asarray(x, float)
        return beta * x[..., 0, 1] + gamma * x[..., 0, 0]  # + gamma * X1

    # MDDAG in (D=2, T=1, D_out=1) convention: edge iff DIRECT influence.
    Gf1 = np.zeros((2, 1, 1), bool); Gf1[1, 0, 0] = True                 # only X2->tgt
    Gf2 = np.zeros((2, 1, 1), bool); Gf2[1, 0, 0] = True; Gf2[0, 0, 0] = True  # X2->tgt, X1->tgt

    cond_sampler = _make_empirical_cond_sampler(X, rng)
    return {
        "X": X, "f1": f1, "f2": f2, "Gf1": Gf1, "Gf2": Gf2,
        "cond_sampler": cond_sampler,
        "params": dict(delta=delta, beta=beta, gamma=gamma, eps_std=eps_std),
    }


class _AffineTarget(nn.Module):
    """(batch, D=2, T=1) -> (batch, D_out=1), the torch-differentiable form of scenario_I's
    and scenario_II's f1/f2/f3 -- needed because dagfaith.explainers requires an nn.Module
    forecaster it can backpropagate through (or repeatedly query); the scenario_* functions'
    own f1/f2/f3 are plain numpy functions with no autograd graph.

    mediated=False: beta*x2 + gamma*x1        (Scenario I's f1 [gamma=0] / f2)
    mediated=True:  beta*x2 + gamma*(x2 - delta*x1)   (Scenario II's f1 [gamma=0] / f3)
    """

    def __init__(self, beta: float, gamma: float = 0.0, delta: float = 0.0, mediated: bool = False):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mediated = mediated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0, -1]
        x2 = x[:, 1, -1]
        if self.mediated:
            out = self.beta * x2 + self.gamma * (x2 - self.delta * x1)
        else:
            out = self.beta * x2 + self.gamma * x1
        return out.unsqueeze(-1)


def scenario_I_torch(beta: float = 1.0, gamma: float = 1.0) -> tuple[nn.Module, nn.Module]:
    """Torch-differentiable f1/f2 for Scenario I -- see `scenario_I`'s docstring for the
    formulas and true graphs (Gf1, Gf2). Same parameter defaults as `scenario_I`."""
    return _AffineTarget(beta=beta, gamma=0.0), _AffineTarget(beta=beta, gamma=gamma)

def scenario_II(n: int, delta: float = 0.8, beta: float = 1.0, gamma: float = 1.0,
                seed: int = 0):
    """Scenario II: X2 = delta*X1 (NO noise) -> data on a 1-D manifold (a line).

    Returns X (n,1,2), and the two analytic models with their MDDAGs:
        f1(x) = beta * x2
        f3(x) = beta * x2 + gamma * (x2 - delta*x1)   # == f1 ON supp(p); off it, differs

    On supp(p): x2 - delta*x1 == 0, so f3 == f1 and X1 has NO edge to the target
    in EITHER model -> Gf3 == Gf1.  A gradient method nonetheless sees
    d f3/d x1 = -gamma*delta != 0 (off-manifold sensitivity), the C1 failure.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = delta * x1                                       # DETERMINISTIC: on the line
    X = np.stack([x1, x2], axis=1)[:, None, :]            # (n, 1, 2)

    def f1(x):
        x = np.asarray(x, float)
        return beta * x[..., 0, 1]
    def f3(x):
        x = np.asarray(x, float)
        return beta * x[..., 0, 1] + gamma * (x[..., 0, 1] - delta * x[..., 0, 0])

    # On supp(p) both models depend on the target only through X2; X1 edge absent.
    Gf1 = np.zeros((2, 1, 1), bool); Gf1[1, 0, 0] = True
    Gf3 = Gf1.copy()                                     # Markov-equivalent on supp(p)

    # NOTE: the empirical Gaussian conditional is DEGENERATE here (X2 perfectly
    # determined by X1). The on-manifold intervention must respect the line:
    # replacing X1 by v must also set X2 = delta*v. Provide a manifold-aware
    # sampler rather than the generic one.
    def cond_sampler(x, i, t):
        x = np.asarray(x, float)
        batch = x.shape[0]
        # draw a new x1 from its marginal, then project onto the manifold
        v = rng.normal(size=batch)
        return v  # caller sets X1<-v; use project_to_manifold to fix X2

    def project_to_manifold(x):
        """Enforce x2 = delta*x1 after any intervention on x1 (keeps data on supp(p))."""
        x = np.asarray(x, float).copy()
        x[..., 0, 1] = delta * x[..., 0, 0]
        return x

    return {
        "X": X, "f1": f1, "f3": f3, "Gf1": Gf1, "Gf3": Gf3,
        "cond_sampler": cond_sampler, "project_to_manifold": project_to_manifold,
        "params": dict(delta=delta, beta=beta, gamma=gamma),
    }


def scenario_II_torch(beta: float = 1.0, gamma: float = 1.0, delta: float = 0.8) -> tuple[nn.Module, nn.Module]:
    """Torch-differentiable f1/f3 for Scenario II -- see `scenario_II`'s docstring for the
    formulas and true graphs (Gf1, Gf3). Same parameter defaults as `scenario_II`."""
    return _AffineTarget(beta=beta, gamma=0.0), _AffineTarget(beta=beta, gamma=gamma, delta=delta, mediated=True)


def scenarioII_gradient_witness(delta: float = 0.8, gamma: float = 1.0) -> float:
    """The off-manifold sensitivity d f3 / d x1 = -gamma*delta that a gradient
    method reports for X1 despite f3 == f1 on supp(p). Nonzero => C1 violation."""
    return -gamma * delta


if __name__ == "__main__":
    # smoke test
    X, Y, B, G, cs = sample_dbn(D=5, T=3, n=200, sparsity=0.3, seed=0)
    print("general DBN:", X.shape, Y.shape, "edges", int(G.sum()))
    v = cs(X[:8], i=1, t=0); print("cond draw:", v.shape)

    s1 = scenario_I(1000)
    print("Scenario I  f1 X1-edge:", s1["Gf1"][0,0,0], " f2 X1-edge:", s1["Gf2"][0,0,0])
    s2 = scenario_II(1000)
    print("Scenario II Gf1==Gf3:", bool((s2["Gf1"]==s2["Gf3"]).all()),
          " gradient witness:", scenarioII_gradient_witness())