"""Real-data loading and an empirical on-manifold conditional sampler.

Real data has no known A, so `empirical_cond_sampler` estimates the joint window mean/covariance directly
from observed windows and conditions via the same Schur-complement machinery -- a
linear-Gaussian APPROXIMATION to the true (unknown) conditional. This is the same honest
caveat `sample_dbn` already carries for its nonlinear regime: report interventional-consistency
results here as resting on that approximation, not as an exact on-manifold guarantee.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dagfaith.dbn import _make_empirical_cond_sampler


def empirical_cond_sampler(X: np.ndarray, seed: int | None = None):
    """Fit a linear-Gaussian on-manifold conditional sampler directly from observed windows.

    Thin wrapper around `dagfaith.dbn._make_empirical_cond_sampler` (same Schur-complement fit
    off the empirical window covariance) so real-data callers get the SAME raw (i, t) cell
    convention -- and the same `.mean(x, i, t)` attribute `dagfaith.cond_baseline`/`teig.teig`
    need for the conditional baseline -- as `dagfaith.dbn.sample_dbn`'s own returned sampler,
    rather than a second, separately-maintained (and previously lag-indexed, `.mean`-less)
    implementation.

    Args:
        X: (n, T, D) windows to fit on.
        seed: seed for the sampler's own stochastic draws.
    """
    rng = np.random.default_rng(seed)
    return _make_empirical_cond_sampler(np.asarray(X, dtype=float), rng)


def load_windowed_series(
    path: str | Path,
    T: int,
    stride: int = 1,
    date_column: str | None = "date",
    standardize: bool = True,
    cond_sampler_seed: int | None = 0,
):
    """Load a real multivariate time series CSV and window it for dagfaith's (n, T, D)
    convention, with an empirically-fit on-manifold conditional sampler.

    Args:
        path: CSV path. All columns except `date_column` are treated as numeric variables.
        T: window length.
        stride: step between consecutive window start points.
        date_column: name of a non-numeric timestamp column to drop, if present.
        standardize: z-score each variable (recommended -- real variables are rarely on a
            common scale, unlike `sample_dbn`'s synthetic unit-scale process).

    Returns:
        X: (n, T, D) windows.
        columns: the D variable names, in array order.
        cond_sampler: fit via `empirical_cond_sampler` on these same windows.
    """
    df = pd.read_csv(path)
    if date_column is not None and date_column in df.columns:
        df = df.drop(columns=[date_column])
    columns = list(df.columns)
    series = df.to_numpy(dtype=float)

    if standardize:
        mean = series.mean(axis=0, keepdims=True)
        std = series.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        series = (series - mean) / std

    T_total, D = series.shape
    n = (T_total - T) // stride + 1
    if n <= 0:
        raise ValueError(f"window T={T} is longer than the available series ({T_total} steps)")
    X = np.stack([series[i * stride : i * stride + T] for i in range(n)], axis=0)

    cond_sampler = empirical_cond_sampler(X, seed=cond_sampler_seed)
    return X, columns, cond_sampler
