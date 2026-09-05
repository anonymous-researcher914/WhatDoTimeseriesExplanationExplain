#!/usr/bin/env python
"""
experiments/comparison_realdata.py
────────────────────────────────────────────────────────────────────────────────
AUC and prediction-change comparison on real forecasting datasets.

Methods
-------
  KARMA    – edge-removal AUC (edges ranked by ρ; skipped when D > MAX_KARMA_D)
  IG       – Integrated Gradients
  TimeShap – KernelSHAP over (feature, time) cells
  TIMING   – temporality-aware Integrated Gradients

"""

from __future__ import annotations
import abc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if not hasattr(np, "trapz"):  # NumPy ≥ 2.0 removed np.trapz
    np.trapz = np.trapezoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from captum.attr import IntegratedGradients

from karma.causal_recovery.edge_contribution import (
    compute_variable_importance,
    compute_variable_importance_multistep,
)
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.sampling import SuffixPool

_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(_ROOT))

# TS-MuLe is not on PyPI; clone github.com/dbvis-ukon/ts-mule into ./ts-mule
_TSMULE_PATH = str(_ROOT / "ts-mule")
if _TSMULE_PATH not in sys.path:
    sys.path.insert(0, _TSMULE_PATH)


def set_global_seed(seed: int) -> None:
    """Seed python's random, numpy's legacy global RNG, and torch (CPU + CUDA).

    Covers every randomness source in this pipeline that reads global RNG
    state rather than an explicit generator: TS-MuLe's perturbation sampling
    (np.random.choice), TIMING's segment sampling (torch.rand/randint), FIT's
    counterfactual sampling (torch.randn), and fresh GRU/LSTM/TCN training
    (DataLoader shuffling + weight init) when no checkpoint is loaded. Does
    NOT cover KARMA's TreeKernelEstimator, which takes its own explicit
    np.random.Generator — run_karma seeds that separately via its own `seed`
    argument.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    D: int
    T: int  # sequence_length
    pred_horizon: int
    ckpt_prefix: str = "" 
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    num_epochs: int = 50
    # KARMA params
    N: int = 2
    eps: float = 0.05
    lam: float = 0.01
    M: int = 100
    n_pool_min: int = 3
    K_max: int = 3

    def __post_init__(self):
        if not self.ckpt_prefix:
            self.ckpt_prefix = self.name


DATASETS: dict[str, DatasetConfig] = {
    "beijing_pm25": DatasetConfig(
        name="beijing_pm25",
        data_dir="data/generated/beijing_pm25",
        D=11,
        T=24,
        pred_horizon=1,
        M=100,
        K_max=24,
        ckpt_prefix="bpm25",
        lam=0.05,
        eps=0.001,
    ),
}

MAX_KARMA_D = 101  # skip KARMA for D > this (state space too large)
N_AUC_STEPS = 10  # progressive removal steps
N_TEST_MAX = 200  # max test samples for attribution (capped for speed)
N_TEST_LARGE_D = 50  # further reduced cap when D > 20




class TorchModel(nn.Module, abc.ABC):
    """
    Class extends torch.nn.Module. Mainly for user to specify the forward with a ``return_all``
    option. The model is supposed to accept inputs of shape (num_samples, num_features, num_times).
    If return_all is True, the output should be of shape (num_samples, num_states, num_times).
    Otherwise, the output should be of shape (num_samples, num_states)
    """

    def __init__(self, feature_size, num_states, hidden_size, device):
        """
        Constructor

        Args:
            feature_size:
               The number of features the model is accepting.
            num_states:
               The number of output nodes.
            hidden_size:
               The hidden size of the model
            device:
               The torch device the model is on.
        """
        super().__init__()
        self.feature_size = feature_size
        self.num_states = num_states
        self.hidden_size = hidden_size
        if self.num_states > 1:
            activation = torch.nn.Softmax(dim=1)
        else:
            activation = torch.nn.Sigmoid()
        self.activation = activation
        self.device = device

    @abc.abstractmethod
    def forward(self, input, return_all=True):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """

    def predict(self, input, return_all=True):
        """
        Apply the activation after the forward function.

            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        return self.activation(self.forward(input, return_all=return_all))

class GRUForecastModel(TorchModel):
    """GRU regression model supporting single- and multi-step forecasting.

    forward(x: (B, D, T), return_all=False) → (B, D)       [next-step, all callers]
    forward(x: (B, D, T), return_all=True)  → (B, D, T)    [rolling, WinIT/DynaMask]
    predict_multistep(x: (B, D, T))         → (B, H, D)    [KARMA multi-step oracle]
    """

    def __init__(
        self,
        D: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        super().__init__(
            feature_size=D, num_states=D, hidden_size=hidden_size, device=device
        )
        self.D = D
        self.forecast_steps = forecast_steps
        self.gru = nn.GRU(
            D,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, forecast_steps * D)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        x_td = x.permute(0, 2, 1)  # (B, T, D)
        out, _ = self.gru(x_td)  # (B, T, hidden)
        if return_all:
            return self.fc(out)[..., : self.D].permute(0, 2, 1)  # (B, D, T)
        return self.fc(out[:, -1, :])[..., : self.D]  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last hidden state."""
        x_td = x.permute(0, 2, 1)
        out, _ = self.gru(x_td)
        raw = self.fc(out[:, -1, :])  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


def load_dataset(cfg: DatasetConfig):
    base = Path(cfg.data_dir)
    X_train = np.load(base / "X_train.npy")  # (N, T, D)
    X_val = np.load(base / "X_val.npy")
    X_test = np.load(base / "X_test.npy")
    y_train = np.load(base / "y_train.npy")  # (N, pred_horizon, D)
    y_val = np.load(base / "y_val.npy")
    y_test = np.load(base / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def reconstruct_raw_series(X_windows: np.ndarray) -> np.ndarray:
    """Reconstruct raw (T_total, D) time series from (N, W, D) windows.

    Windows are assumed to be consecutive (stride=1).  X_windows[0] gives the
    first W values; each subsequent window adds one new observation at the end.
    """
    first = X_windows[0]  # (W, D)
    rest = X_windows[1:, -1, :]  # (N-1, D)
    return np.concatenate([first, rest], axis=0)  # (W + N - 1, D)


def train_gru(
    model: GRUForecastModel,
    X_train: np.ndarray,  # (N, T, D)
    y_train: np.ndarray,  # (N, pred_horizon, D)
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DatasetConfig,
    device,
    ckpt_path: Optional[Path] = None,
) -> GRUForecastModel:
    multistep = int(y_train.shape[1]) > 1  # driven by actual data, not cfg
    X_tr = torch.from_numpy(X_train.transpose(0, 2, 1).astype(np.float32))  # (N, D, T)
    y_tr = torch.from_numpy(
        y_train.astype(np.float32) if multistep else y_train[:, 0, :].astype(np.float32)
    )  # (N, actual_horizon, D) or (N, D)
    X_v = torch.from_numpy(X_val.transpose(0, 2, 1).astype(np.float32))
    y_v = torch.from_numpy(
        y_val.astype(np.float32) if multistep else y_val[:, 0, :].astype(np.float32)
    )

    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=cfg.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_val = float("inf")
    best_state = None
    patience = 10

    model = model.to(device)
    for epoch in range(cfg.num_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = (
                model.predict_multistep(xb)
                if multistep
                else model(xb, return_all=False)
            )
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = (
                    model.predict_multistep(xb)
                    if multistep
                    else model(xb, return_all=False)
                )
                val_losses.append(F.mse_loss(pred, yb).item())
        vl = float(np.mean(val_losses))
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 10
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            patience -= 1
            if patience == 0:
                break

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: val_mse={vl:.5f}")

    model.load_state_dict(best_state)
    return model


def get_or_train_gru(
    cfg: DatasetConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    ckpt_dir: Path,
) -> GRUForecastModel:
    ckpt_path = ckpt_dir / f"gru_{cfg.name}.pt"
    actual_horizon = int(y_train.shape[1])  # ground truth from data, not cfg
    model = GRUForecastModel(
        D=cfg.D,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        forecast_steps=actual_horizon,
        device=device,
    )
    if ckpt_path.exists():
        print(f"  Loading GRU from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
    else:
        print(f"  Training GRU ({cfg.num_epochs} epochs max)…")
        model = train_gru(model, X_train, y_train, X_val, y_val, cfg, device, ckpt_path)
    model.eval()
    return model




def _orig_preds(x_test: np.ndarray, model: GRUForecastModel, device) -> np.ndarray:
    """Predict on x_test → (B, D) for single-step models.

    For multistep-trained models (forecast_steps > 1), uses
    model.predict_multistep() to get the full (B, H, D) horizon instead of
    model(..., return_all=False), which — even on a multistep-trained model —
    silently truncates to just the first forecast step (see
    LSTMForecastModel._apply_head / TCNForecastModel._apply_head: "take
    first forecast step's D outputs"). Every removal-based metric below
    (feature_removal_auc, timestep_removal_auc, ...) calls this for both the
    unmasked and masked predictions, so as long as both
    sides go through the same branch here, the H>1 case falls out for free —
    np.abs(a - b).mean() and MSE both work unchanged on (B, H, D) arrays.
    """
    x = torch.from_numpy(x_test.astype(np.float32)).to(device)
    with torch.no_grad():
        if getattr(model, "forecast_steps", 1) > 1 and hasattr(
            model, "predict_multistep"
        ):
            return model.predict_multistep(x).cpu().numpy()  # (B, H, D)
        return model(x, return_all=False).cpu().numpy()  # (B, D)



def karma_edges_to_attr(edges: list, D: int, T: int) -> np.ndarray:
    """Convert KARMA edges to a (D, T) attribution matrix.

    Edge {"src": s, "lag": l, "rho": r} maps to window position (s, T-l).
    Multiple edges sharing the same (src, lag) are combined by max ρ.
    Returns a (D, T) array broadcastable to (B, D, T) for feature_removal_auc.
    """
    attr = np.zeros((D, T), dtype=np.float32)
    for e in edges:
        src, lag, rho = e["src"], e["lag"], e["rho"]
        t_idx = T - lag  # lag=1 → last timestep, lag=2 → second-to-last, …
        if 0 <= t_idx < T:
            attr[src, t_idx] = max(attr[src, t_idx], rho)
    return attr


def _multistep_target(model, x: torch.Tensor, tau: list = None) -> torch.Tensor:
    """(B, D) target for black-box/gradient explainers that only ever call the
    model with return_all=False and expect a single (B, D) tensor.

    For forecast_steps > 1 models, model(x, return_all=False) silently
    truncates to just the first forecast step (see LSTMForecastModel/
    TCNForecastModel._apply_head). This instead uses predict_multistep and
    averages over the tau forecast steps (default: all steps), so the
    resulting target genuinely reflects the whole horizon rather than step 0
    alone. Falls back to model(x, return_all=False) for single-step models.
    """
    if getattr(model, "forecast_steps", 1) > 1 and hasattr(model, "predict_multistep"):
        multi = model.predict_multistep(x)  # (B, forecast_steps, D)
        steps = tau if tau is not None else list(range(multi.shape[1]))
        return multi[:, steps, :].mean(dim=1)  # (B, D)
    return model(x, return_all=False)


def run_timeshap(
    x_test: np.ndarray,  # (B, D, T)
    model: GRUForecastModel,
    device,
    nsamples: int = 200,
    tau: list = None,
) -> np.ndarray:
    """TimeShap cell-level attribution → (B, D, T)."""
    # timeshap 1.0.4's TimeShapKernel(Kernel) still imports shap's PRIVATE `Kernel` class
    # (shap.explainers._kernel.Kernel), renamed to the now-public `KernelExplainer` in shap
    # 0.43+ with no compatibility alias kept. An older shap with `Kernel` still exists (<0.43)
    # would fix the import, but those releases predate NumPy 2 support and crash on import
    # under this repo's numpy>=1.24 (installed: 2.x) -- not worth a repo-wide NumPy downgrade
    # for one baseline method. `KernelExplainer` is the same implementation under its current
    # name (confirmed: TimeShapKernel runs correctly against it), so alias it in instead.
    import shap.explainers._kernel as _shap_kernel_mod
    if not hasattr(_shap_kernel_mod, "Kernel"):
        _shap_kernel_mod.Kernel = _shap_kernel_mod.KernelExplainer

    from timeshap.explainer.kernel import TimeShapKernel

    B, D_feat, T = x_test.shape
    x_td = x_test.transpose(0, 2, 1).astype(np.float32)  # (B, T, D)
    background = np.zeros((1, T, D_feat), dtype=np.float32)
    model.eval()

    # LassoLarsIC (used inside TimeShap's kernel solver) requires
    # nsamples > n_features = T * D.  Silently floor up when needed.
    n_features = T * D_feat
    if nsamples <= n_features:
        nsamples_eff = n_features + max(50, n_features // 4)
        print(
            f"    [TimeShap] nsamples={nsamples} < n_features={n_features}; "
            f"raising to {nsamples_eff}"
        )
        nsamples = nsamples_eff

    def model_fn(x: np.ndarray) -> np.ndarray:
        # x: (N, T, D) numpy → (N, D, T) torch → scalar (N,)
        x_dt = torch.from_numpy(x.transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            return _multistep_target(model, x_dt, tau).mean(dim=-1).cpu().numpy()

    varying = (list(range(T)), list(range(D_feat)))
    attr = np.zeros((B, D_feat, T), dtype=np.float32)
    for b in range(B):
        kernel = TimeShapKernel(
            model_fn, background, rs=42, mode="cell", varying=varying
        )
        sv = kernel.shap_values(x_td[b : b + 1], pruning_idx=0, nsamples=nsamples)
        # sv: (T*D,) ordered (t0_d0, t0_d1, …, t1_d0, …) → reshape (T, D) → (D, T)
        attr[b] = np.abs(sv.reshape(T, D_feat).T)
    return attr







def run_ig(
    x_test: np.ndarray, model: GRUForecastModel, device, tau: list = None
) -> np.ndarray:
    """Integrated Gradients → (B, D, T).  Zero baseline, sum over D outputs.

    For forecast_steps > 1 models, targets _multistep_target (mean over tau
    forecast steps via predict_multistep) instead of model(return_all=False),
    which would otherwise silently attribute only the first forecast step.
    predict_multistep is a plain differentiable forward pass, so it works
    unchanged under IG's path integral (Captum evaluates it at every
    interpolated point along the baseline->input path, not just the input).
    """
    model.eval()

    def _scalar_forward(x: torch.Tensor) -> torch.Tensor:
        return _multistep_target(model, x, tau).sum(dim=-1)  # (B,)

    ig = IntegratedGradients(_scalar_forward)
    x_t = torch.from_numpy(x_test.astype(np.float32)).to(device)
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t))  # (B, D, T)
    torch.backends.cudnn.enabled = orig_cudnn
    return np.abs(attr.detach().cpu().numpy())


def _timing_segment_ig(
    forward_func,
    x: torch.Tensor,
    target_channel: int,
    n_samples: int,
    num_segments: int,
    min_seg_len: int,
    max_seg_len: int | None,
) -> torch.Tensor:
    """Segment-based, temporality-aware Integrated Gradients: the core algorithm behind TIMING
    (Yoon et al., "TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation",
    ICML 2025 Spotlight -- github.com/drumpt/TIMING, attribution/explainers.py::OUR). Ported
    from their `attribute_random_time_segments_one_dim_same_for_batch`, generalized from
    "gather one class logit" (classification) to "read one regression output coordinate" --
    the same segment-masked-interpolation/gradient-accumulation core, not a simplification.

    Instead of IG's usual per-coordinate straight-line path, interpolation steps randomly hold
    out whole contiguous (feature, time-segment) blocks at the input value rather than the
    interpolated one, so the path respects temporal structure instead of moving every point
    independently.

    x: (batch, D, T). Returns the (batch, D, T) attribution for output coordinate
    `target_channel`.
    """
    batch, D, T = x.shape
    baselines = x.mean(dim=0, keepdim=True).expand_as(x)
    max_seg_len = T if max_seg_len is None else min(T, max_seg_len)

    dev = x.device
    alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=dev).view(
        -1, 1, 1, 1
    )
    expanded_inputs = x.unsqueeze(0)
    expanded_baselines = baselines.unsqueeze(0)
    interpolated = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)

    dims = torch.randint(0, D, (n_samples, batch, num_segments), device=dev)
    seg_lens = torch.randint(
        min_seg_len, max_seg_len + 1, (n_samples, batch, num_segments), device=dev
    )
    t_starts = (
        torch.rand(n_samples, batch, num_segments, device=dev) * (T - seg_lens)
    ).long()

    time_mask = torch.ones_like(interpolated)
    batch_indices = torch.arange(batch, device=dev)
    sample_indices = torch.arange(n_samples, device=dev)
    for s in range(num_segments):
        max_len = int(seg_lens[:, :, s].max().item())
        base_range = torch.arange(max_len, device=dev).view(1, 1, -1)
        indices = t_starts[:, :, s].unsqueeze(-1) + base_range
        end_points = (t_starts[:, :, s] + seg_lens[:, :, s]).unsqueeze(-1)
        valid = (indices < end_points) & (indices < T)
        idx = (indices * valid).clamp(0, T - 1)
        time_mask[
            sample_indices.view(-1, 1, 1),
            batch_indices.view(1, -1, 1),
            dims[:, :, s].unsqueeze(-1),
            idx,
        ] = 0

    fixed_inputs = expanded_inputs.detach()
    masked_inputs = time_mask * interpolated + (1 - time_mask) * fixed_inputs
    masked_inputs.requires_grad_(True)

    preds = forward_func(masked_inputs.reshape(-1, D, T))
    preds = preds.reshape(n_samples, batch, -1)
    target = preds[:, :, target_channel]
    total = target.sum()

    (grad,) = torch.autograd.grad(outputs=total, inputs=masked_inputs)
    grad = grad * time_mask
    grads = grad.sum(dim=0)
    denom = time_mask.sum(dim=0) + 1e-8
    return grads * (x - baselines) / denom


def run_timing(
    x_test: np.ndarray,  # (B, D, T)
    model,
    device,
    n_samples: int = 50,
    tau: list = None,
) -> np.ndarray:
    """TIMING attribution → (B, D, T) (Jang et al., ICML 2025).

    Temporality-aware integrated gradients: randomly fixes positions during
    path integration so gradients are conditioned on temporal context.
    Adapted here for multi-output regression by summing all outputs.

    For forecast_steps > 1 models, forward_func targets _multistep_target
    (mean over tau forecast steps) instead of the model directly, which
    would otherwise silently attribute only the first forecast step —
    _timing_segment_ig only ever calls forward_func(x) -> (N, D) and reads
    off target_channel, so the (B, D)-shaped _multistep_target output is a
    drop-in replacement.
    """
    model.eval()
    _, D, T = x_test.shape

    x = torch.from_numpy(x_test.astype(np.float32)).to(device)  # (B, D, T)
    forward_func = lambda xin: _multistep_target(model, xin, tau)  # noqa: E731
    per_output = []
    for d_out in range(D):
        attr = _timing_segment_ig(
            forward_func,
            x,
            d_out,
            n_samples,
            num_segments=3,
            min_seg_len=1,
            max_seg_len=1,
        )
        scores_dt = attr.abs().mean(dim=0).detach().cpu().numpy()
        per_output.append(scores_dt)
    return np.stack(per_output, axis=0)  # (D, T)

def run_karma(
    X_train: np.ndarray,  # (N, T, D) windowed
    X_val: np.ndarray,  # (N, T, D) windowed
    model: GRUForecastModel,
    cfg: DatasetConfig,
    device,
    verbose: bool = True,
    seed: int = 42,
    tau: list = None,
) -> tuple[list, int, np.ndarray]:
    """KARMA pipeline.  Returns (edges, K_star, b_star).

    b_star: (W-K*, D) certified baseline for non-blanket prefix positions.

    tau : forecast step indices for the joint multistep variable importance
          (compute_variable_importance_multistep). Only used when the fitted
          estimator's horizon > 1 (i.e. the oracle's forecast_steps > 1 —
          NOT cfg.pred_horizon, which is a display-only field that can be
          stale relative to the checkpoint actually loaded). None pools all
          horizon steps jointly. Ignored when horizon == 1.
    """
    X_train_raw = reconstruct_raw_series(X_train)  # (T_raw, D)
    X_val_raw = reconstruct_raw_series(X_val)
    W = cfg.T  # use dataset window size as KARMA oracle window

    model.eval()

    def f_oracle(window: np.ndarray) -> np.ndarray:
        is_single = window.ndim == 2
        w = window[np.newaxis] if is_single else window  # (B, W, D)
        x = torch.from_numpy(w.transpose(0, 2, 1).astype(np.float32)).to(device)
        with torch.no_grad():
            if getattr(model, "forecast_steps", 1) > 1:
                pred = (
                    model.predict_multistep(x).cpu().numpy()
                )  # (B, forecast_steps, D)
            else:
                pred = model(x, return_all=False).cpu().numpy()  # (B, D)
        return pred[0] if is_single else pred

    disc = Discretiser(N=cfg.N)
    disc.fit(X_train_raw)

    T_val_raw = len(X_val_raw)
    X_val_windows = np.stack([X_val_raw[t : t + W] for t in range(T_val_raw - W)])

    result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train_raw,
        X_val=X_val_windows,
        disc=disc,
        W=W,
        eps=cfg.eps,
        K_max=cfg.K_max,
        loss="regression",
        verbose=verbose,
        seed=seed,
    )
    K_star = result["K_star"]
    b_star = result["b_star"]
    pi_star = result["pi_star"]
    if verbose:
        print(f"  K* = {K_star},  Δ_pred = {result['Delta_A1']:.4f}")

    pool = SuffixPool(disc=disc, K=K_star, W=W)
    pool.build(X_train_raw)

    tree = TreeKernelEstimator(
        disc,
        K=K_star,
        W=W,
        M=cfg.M,
        n_pool=cfg.n_pool_min,
        pool=pool,
        b_star=b_star,
        rng=np.random.default_rng(seed),
    )
    tree.fit(
        f=f_oracle,
        pi_star=pi_star,
        verbose=verbose,
        mega_batch=torch.cuda.is_available(),
    )

    horizon = getattr(tree, "horizon", 1)
    if horizon > 1:
        vi = compute_variable_importance_multistep(
            tree, pi_star, disc, tau=tau, lam=cfg.lam
        )
    else:
        vi = compute_variable_importance(tree, pi_star, disc, lam=cfg.lam)
    return vi["edges"], K_star, b_star


def _scalar_numpy_forward(model, device, tau: list = None):
    """(n, T, D) numpy -> (n, 1) numpy: sum-over-output-coordinates target, for methods whose
    OWN attribution is already a single (i, t) score with no per-target breakdown (IG, TimeShap,
    TS-MuLe, ShapTime, TIMING, KARMA) -- their OMIC candidate edges stay `(i, t, 0)` (dummy
    target), so `delta_effect`'s `j=0` must index a (n, 1) black-box forecast. Wrapped for
    dagfaith.intervention.delta_effect's own convention (X: (n, T, D), NOT (D, T)-transposed).

    NOT used by TEIG's own OMIC scoring: TEIG is an edge-level method with a genuine per-target
    score at every (source cell, target variable) pair (see run_teig's docstring for why summing
    those away would defeat the point) -- see _pertarget_numpy_forward instead.
    """
    model.eval()

    def f_np(X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(X, dtype=np.float32).transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            return _multistep_target(model, x, tau).sum(dim=-1, keepdim=True).cpu().numpy()

    return f_np