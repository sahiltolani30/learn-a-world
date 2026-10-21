"""
Shared utilities for Learn-a-World.
Provides device auto-selection (MPS/CUDA/CPU), Google Drive / local storage paths,
config loading, and reproducible random seeding.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import yaml

def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Selects optimal PyTorch compute device automatically:
    CUDA (Colab/Linux) -> MPS (Mac Apple Silicon) -> CPU fallback.
    """
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_output_dir(
    local_dir: str = "./checkpoints",
    colab_subpath: str = "MyDrive/learn-a-world/checkpoints"
) -> Path:
    """
    Resolves output directory:
    - In Google Colab with mounted Drive: /content/drive/{colab_subpath}
    - Locally: {local_dir}
    Creates the directory if it does not exist.
    """
    colab_drive_path = Path("/content/drive")
    if colab_drive_path.exists():
        target_dir = colab_drive_path / colab_subpath
    else:
        target_dir = Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS deterministic mode if supported
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
