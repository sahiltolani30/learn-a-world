"""Safety, integrity, and tensor boundary validation utilities."""
from typing import Any, Tuple
import numpy as np
import torch


def assert_finite_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert that a PyTorch tensor contains only finite numbers (no NaN or Inf)."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor for {name}, got {type(tensor)}")
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}!")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}!")


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[Any, ...], name: str = "tensor") -> None:
    """Assert that a PyTorch tensor has the expected shape (None in expected_shape matches any dimension)."""
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(f"Shape rank mismatch for {name}: expected {expected_shape}, got {tuple(tensor.shape)}")
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"Shape mismatch for {name} at dim {i}: expected {expected}, got {actual} (full shape: {tuple(tensor.shape)})")


def assert_observation_contract(obs: np.ndarray, expected_shape: Tuple[int, int, int] = (64, 64, 3)) -> None:
    """Assert observation contract: uint8 numpy array with exact shape [64, 64, 3]."""
    if not isinstance(obs, np.ndarray):
        raise TypeError(f"Observation must be np.ndarray, got {type(obs)}")
    if obs.shape != expected_shape:
        raise ValueError(f"Observation shape mismatch: expected {expected_shape}, got {obs.shape}")
    if obs.dtype != np.uint8:
        raise ValueError(f"Observation dtype mismatch: expected uint8, got {obs.dtype}")


def assert_action_token_contract(token: int, num_actions: int = 17) -> None:
    """Assert opaque action token contract: int in [0, num_actions - 1]."""
    if not isinstance(token, (int, np.integer)):
        raise TypeError(f"Action token must be an integer, got {type(token)}")
    if token < 0 or token >= num_actions:
        raise ValueError(f"Action token must be in [0, {num_actions - 1}], got {token}")
