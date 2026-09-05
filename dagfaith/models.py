"""Simple forecasting models for the DAG-faithfulness experiments.

This module provides a minimal training interface for three model families:
- GRU
- LSTM
- TCN

The train() helper accepts a 3D tensor of shape (n_samples, seq_len, n_features)
and a target vector of shape (n_samples,). It returns a trained PyTorch module and
its validation forecast error.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class TCNForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=-1)
        return self.fc(x).squeeze(-1)


def _prepare_inputs(X, y, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, seq_len, n_features)")
    if y.ndim != 1:
        raise ValueError("y must have shape (n_samples,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    return X, y


def train(model_type: str, X, y, epochs: int = 20, device: str = "cpu"):
    """Train a simple forecasting model and return the trained model plus validation loss.

    Args:
        model_type: one of {gru, lstm, tcn}
        X: array-like of shape (n_samples, seq_len, n_features)
        y: array-like of shape (n_samples,)
        epochs: number of training epochs
        device: torch device name, e.g. 'cpu' or 'cuda'

    Returns:
        model: trained torch.nn.Module
        val_error: mean absolute error on a held-out split
    """
    if model_type not in {"gru", "lstm", "tcn"}:
        raise ValueError("model_type must be one of {'gru', 'lstm', 'tcn'}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    X_t, y_t = _prepare_inputs(X, y, device)
    n = X_t.shape[0]
    split = max(1, int(0.2 * n))
    train_x = X_t[:-split]
    train_y = y_t[:-split]
    val_x = X_t[-split:]
    val_y = y_t[-split:]

    input_size = X_t.shape[-1]
    if model_type == "gru":
        model = GRUForecaster(input_size=input_size).to(device)
    elif model_type == "lstm":
        model = LSTMForecaster(input_size=input_size).to(device)
    else:
        model = TCNForecaster(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(val_x)
        val_error = float(torch.mean(torch.abs(val_pred - val_y)).cpu().item())

    return model, val_error


class MultiOutputForecaster(nn.Module):
    """Bundles D independently-trained single-output forecasters (one per output coordinate)
    into a single (batch, D, T) -> (batch, D) model, for use as gf_intervention's black-box f.

    Each submodel was trained to predict X[:, -1, d] from X[:, :-1, :], so this module applies
    the same slicing to any window it is given.
    """

    def __init__(self, submodels):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, D, T) features-by-time. Returns (batch, D)."""
        history = x[:, :, :-1].permute(0, 2, 1)  # (batch, T-1, D)
        outs = [m(history) for m in self.submodels]
        return torch.stack(outs, dim=1)

    def numpy_forward(self, X: np.ndarray) -> np.ndarray:
        """X: (n, T, D) -> (n, D)."""
        device = next(self.parameters()).device
        history = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)[:, :-1, :]
        self.eval()
        with torch.no_grad():
            outs = [m(history) for m in self.submodels]
        return torch.stack(outs, dim=1).cpu().numpy()


def train_forecaster(model_type: str, X, epochs: int = 20, device: str = "cpu"):
    """Train one single-output model per output coordinate, predicting X[:, -1, d] from
    X[:, :-1, :]. Returns a MultiOutputForecaster plus per-coordinate validation errors
    (report these so the reader knows the model is worth explaining, per E3)."""
    X = np.asarray(X, dtype=np.float32)
    _, _, D = X.shape
    inputs = X[:, :-1, :]

    submodels = []
    val_errors = np.zeros(D, dtype=float)
    for d in range(D):
        y = X[:, -1, d]
        model, val_error = train(model_type, inputs, y, epochs=epochs, device=device)
        submodels.append(model)
        val_errors[d] = val_error

    forecaster = MultiOutputForecaster(submodels).to(device)
    return forecaster, val_errors


class TargetForecaster(nn.Module):
    """Bundles D_out independently-trained submodels -- one per column of `dagfaith.dbn`'s own
    forecast target Y (n, D_out) -- into a single (batch, D, T) -> (batch, D_out) model:
    dbn.py's raw (variable, window-position) convention (matches `dagfaith.oracle.RawWindowOracle`),
    as opposed to `MultiOutputForecaster`'s next-step-within-window convention.

    Each submodel was trained to predict Y[:, j] from the FULL window X (all T positions -- Y is
    an independent target, not X's own future, so there is no "last step" to hold out here).
    This is the "real model" black box for TEIG/OMIC to evaluate on dbn.py data, as opposed to
    `RawWindowOracle`'s exact linear oracle.
    """

    def __init__(self, submodels):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, D, T) features-by-time (RawWindowOracle's own convention). Returns
        (batch, D_out)."""
        window = x.permute(0, 2, 1)  # (batch, T, D), the submodels' own batch_first convention
        outs = [m(window) for m in self.submodels]
        return torch.stack(outs, dim=1)

    def numpy_forward(self, X: np.ndarray) -> np.ndarray:
        """X: (n, T, D) -> (n, D_out), dbn.py's own raw convention (no permute needed)."""
        device = next(self.parameters()).device
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        self.eval()
        with torch.no_grad():
            outs = [m(x) for m in self.submodels]
        return torch.stack(outs, dim=1).cpu().numpy()


def train_target_forecaster(model_type: str, X, Y, epochs: int = 20, device: str = "cpu"):
    """Train one single-output model per column of `dagfaith.dbn.sample_dbn`'s own forecast
    target Y (n, D_out), each predicting Y[:, j] from the FULL input window X (n, T, D). Returns
    a TargetForecaster plus per-coordinate validation errors (report these so the reader knows
    the model is worth explaining, per teig_telrp.md/omic_iclr.md's trained-model tasks)."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    D_out = Y.shape[1]

    submodels = []
    val_errors = np.zeros(D_out, dtype=float)
    for j in range(D_out):
        model, val_error = train(model_type, X, Y[:, j], epochs=epochs, device=device)
        submodels.append(model)
        val_errors[j] = val_error

    forecaster = TargetForecaster(submodels).to(device)
    return forecaster, val_errors
