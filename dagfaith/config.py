"""Global seeding and config loader."""
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(path=None):
    """Load and return the YAML config at path, defaulting to configs/default.yaml."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def seed_everything(seed: int = 42):
    """Set all random seeds (Python, NumPy, PyTorch, CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def results_dir(cfg=None) -> Path:
    """Return the results directory path, creating it if necessary."""
    if cfg is None:
        cfg = load_config()
    d = Path(cfg.get("results_dir", "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d
