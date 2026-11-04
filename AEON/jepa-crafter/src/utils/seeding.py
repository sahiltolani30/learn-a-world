"""Seeding utilities for reproducible runs."""
import os
import random
from typing import Dict, Any, Optional
import numpy as np
import torch


def seed_everything(seed: int) -> Dict[str, Any]:
    """Seed Python, NumPy, PyTorch (CPU and CUDA/MPS if available).
    
    Returns a dictionary of recorded seeds and initial RNG states.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    return {
        "seed": seed,
        "torch_seed": torch.initial_seed(),
    }


def get_rng_states() -> Dict[str, Any]:
    """Capture current RNG states for checkpointing."""
    states = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states from checkpoint."""
    if "python_rng_state" in states:
        random.setstate(states["python_rng_state"])
    if "numpy_rng_state" in states:
        np.random.set_state(states["numpy_rng_state"])
    if "torch_rng_state" in states:
        try:
            t_rng = states["torch_rng_state"]
            if torch.is_tensor(t_rng):
                t_rng = t_rng.cpu().byte()
            torch.set_rng_state(t_rng)
        except Exception:
            pass
    if torch.cuda.is_available() and "torch_cuda_rng_state" in states:
        try:
            torch.cuda.set_rng_state_all(states["torch_cuda_rng_state"])
        except Exception:
            pass
