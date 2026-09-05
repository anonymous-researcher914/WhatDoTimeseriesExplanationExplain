"""Shared DataLoader construction for every dataset directory under data/generated/<name>/,
all produced by the same preprocessing convention (see metadata.json in each): X_{split}.npy
shape (N, sequence_length, D), y_{split}.npy shape (N, forecast_steps, D), split in
{train, val, test}, already windowed and scaled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloaders(data_dir: str, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(data_dir)

    def _load(split: str) -> TensorDataset:
        X = np.load(data_dir / f"X_{split}.npy")
        y = np.load(data_dir / f"y_{split}.npy")
        return TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))

    train_loader = DataLoader(_load("train"), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_load("val"), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_load("test"), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
