"""Analytic oracle models for DAG-faithfulness recovery experiments."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AnalyticOracle(nn.Module):
    """f(x) = E[X_t | past] using the TRUE coefficients A.

    Implemented as a torch module (fixed, non-trainable weights) so it is both an exact
    numeric oracle (via `numpy_forward`, used as the black-box `f` for gf_intervention) and a
    differentiable model (via `forward`) that gradient-based explainers (IG, Saliency,
    Dynamask, ...) can attribute through directly. For this f, G_f = supp(A) = Gstar exactly.

    A: (D, L, D_out). D_out need not equal D -- the forecast target doesn't have to be the
    same variable set as the input window (e.g. dbn.py's general sample_dbn/scenario_I/
    scenario_II, where D_out is an independent forecast target).
    """

    def __init__(self, A: np.ndarray, nonlinear: bool = False):
        super().__init__()
        A = np.asarray(A, dtype=np.float32)
        self.D, self.L, self.D_out = A.shape
        self.register_buffer("A", torch.as_tensor(A))
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, D, T) features-by-time, matching the captum/WinIT convention.
        Returns: (batch, D_out) forecast for every output coordinate."""
        batch, D, T = x.shape
        out = x.new_zeros((batch, self.D_out))
        for lag in range(self.L):
            t_idx = T - 1 - lag
            if t_idx < 0:
                continue
            out = out + x[:, :, t_idx] @ self.A[:, lag, :]
        if self.nonlinear:
            out = out + torch.tanh(out) * 0.1
        return out

    def numpy_forward(self, X: np.ndarray) -> np.ndarray:
        """X: (n, T, D) -> (n, D), the black-box numeric convention used by gf_intervention."""
        x = torch.as_tensor(np.asarray(X, dtype=np.float32)).permute(0, 2, 1)
        with torch.no_grad():
            out = self.forward(x)
        return out.numpy()


def analytic_oracle(A: np.ndarray, nonlinear: bool = False) -> AnalyticOracle:
    """Return the analytic conditional-mean oracle module for the given coefficient tensor."""
    return AnalyticOracle(A, nonlinear=nonlinear)


def gf_analytic(A: np.ndarray) -> np.ndarray:
    """Exact graph for the analytic oracle: support of the coefficient tensor."""
    return np.asarray(A != 0, dtype=bool)


class RawWindowOracle(nn.Module):
    """f(x) = sum_{i,t} B[i,t,j] x_t^{(i)} (+ 0.1*tanh if nonlinear) -- `dagfaith.dbn`'s OWN
    raw (variable, window-position) convention, as opposed to `AnalyticOracle`'s lag-reversed
    (A[d', lag, d]) convention. x: (batch, D, T). Used by teig_telrp.md's Task 4/5 scripts
    (`experiments/run_recovery.py`, `experiments/run_faithfulness.py`), since
    `dagfaith.cond_baseline`/`dagfaith.teig`'s cond_model machinery is built around dbn.py's raw
    (i, t) convention, not the lag one.
    """

    def __init__(self, B: np.ndarray, nonlinear: bool = False):
        super().__init__()
        self.register_buffer("B", torch.as_tensor(np.asarray(B, dtype=np.float32)))
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("ndt,dtj->nj", x, self.B)
        if self.nonlinear:
            out = out + 0.1 * torch.tanh(out)
        return out

    def numpy_forward(self, X: np.ndarray) -> np.ndarray:
        """X: (n, T, D) -> (n, D_out), dbn.py's own raw convention (no permute needed)."""
        x = torch.as_tensor(np.asarray(X, dtype=np.float32)).permute(0, 2, 1)
        with torch.no_grad():
            out = self.forward(x)
        return out.numpy()
