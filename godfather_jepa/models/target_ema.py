"""
Polyak Target Exponential Moving Average (EMA) Updater for Godfather JEPA.

Maintains stop-gradient target networks for non-contrastive representation learning
and target critics for stabilized reinforcement learning value bootstrapping.

Guarantees:
1. Strict parameter freeze: Target parameters have requires_grad=False.
2. Eval mode preservation: Target network stays in eval() mode across model.train().
3. Strict stop-gradient guarantee: Zero gradients backpropagate through target outputs
   to either target parameters or input observations.
4. Polyak momentum schedule: Built-in linear and cosine schedules tau in [0.990, 0.999].
5. Buffer synchronization: Registered buffers synchronize across online and target modules.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_polyak_tau(
    step: int,
    total_steps: int,
    tau_min: float = 0.990,
    tau_max: float = 0.999,
    schedule: str = "linear",
) -> float:
    """
    Computes scheduled Polyak momentum factor tau in [tau_min, tau_max].

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        tau_min: Initial Polyak momentum (default 0.990).
        tau_max: Asymptotic Polyak momentum (default 0.999).
        schedule: Schedule type ('linear' or 'cosine').

    Returns:
        tau: Scaled float momentum in [tau_min, tau_max].
    """
    if not (0.0 <= tau_min <= 1.0):
        raise ValueError(f"tau_min must be in [0.0, 1.0], got {tau_min}")
    if not (0.0 <= tau_max <= 1.0):
        raise ValueError(f"tau_max must be in [0.0, 1.0], got {tau_max}")
    if tau_min > tau_max:
        raise ValueError(f"tau_min ({tau_min}) cannot exceed tau_max ({tau_max})")

    if total_steps <= 0:
        return tau_max

    prog = min(max(float(step) / float(total_steps), 0.0), 1.0)
    if schedule == "cosine":
        return float(tau_max - (tau_max - tau_min) * 0.5 * (1.0 + math.cos(math.pi * prog)))
    elif schedule == "linear":
        return float(tau_min + (tau_max - tau_min) * prog)
    else:
        raise ValueError(f"Unknown momentum schedule '{schedule}'. Supported: 'linear', 'cosine'")


class PolyakMomentumScheduler:
    """
    Stateful momentum scheduler managing step count and tau progression.
    """

    def __init__(
        self,
        total_steps: int = 100_000,
        tau_min: float = 0.990,
        tau_max: float = 0.999,
        schedule: str = "linear",
    ):
        self.total_steps = total_steps
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.schedule = schedule
        self.current_step = 0

    def get_tau(self, step: Optional[int] = None) -> float:
        s = self.current_step if step is None else step
        return compute_polyak_tau(
            s,
            self.total_steps,
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            schedule=self.schedule,
        )

    def step(self) -> float:
        tau = self.get_tau(self.current_step)
        self.current_step += 1
        return tau

    def reset(self) -> None:
        self.current_step = 0


def update_target_ema(online: nn.Module, target: nn.Module, tau: float) -> None:
    """
    Updates target module parameters via Polyak exponential moving average:
        theta_target = tau * theta_target + (1 - tau) * theta_online

    Also synchronizes any registered buffers from online to target.

    Args:
        online: Source online PyTorch neural network module.
        target: Destination target PyTorch neural network module.
        tau: Polyak momentum factor in [0.0, 1.0]. Typically 0.990 to 0.999.
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"Polyak tau momentum must be in [0.0, 1.0], got {tau}")

    with torch.no_grad():
        # 1. Update parameters via Polyak momentum interpolation
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.mul_(tau).add_(p_online, alpha=1.0 - tau)

        # 2. Synchronize buffers directly
        for b_online, b_target in zip(online.buffers(), target.buffers()):
            b_target.copy_(b_online)


class TargetEMA(nn.Module):
    """
    Wrapper holding an online module and an EMA target module with frozen gradients.

    Guarantees:
    - Target parameters always have `requires_grad=False`.
    - Target is kept in `eval()` mode even when `train()` is called on this container.
    - Calling target forward produces completely detached tensors with zero gradients
      backpropagating to input tensors.
    """

    def __init__(self, online_module: nn.Module, initial_tau: float = 0.996):
        super().__init__()
        self.online = online_module
        self.target = copy.deepcopy(online_module)
        self.tau = initial_tau

        # Ensure target network gradients are completely disabled
        for param in self.target.parameters():
            param.requires_grad = False
        self.target.eval()

        # Wrap target forward with a strict detachment boundary
        orig_target_forward = self.target.forward

        def _safe_target_forward(*args: Any, **kwargs: Any) -> Any:
            with torch.no_grad():
                out = orig_target_forward(*args, **kwargs)
                return self._detach_output(out)

        self.target.forward = _safe_target_forward

    @staticmethod
    def _detach_output(out: Any) -> Any:
        """Recursively detaches all output tensors."""
        if isinstance(out, torch.Tensor):
            return out.detach()
        elif isinstance(out, tuple):
            return tuple(TargetEMA._detach_output(item) for item in out)
        elif isinstance(out, list):
            return [TargetEMA._detach_output(item) for item in out]
        elif isinstance(out, dict):
            return {k: TargetEMA._detach_output(v) for k, v in out.items()}
        return out

    def train(self, mode: bool = True) -> TargetEMA:
        """
        Sets training mode for online network, while strictly enforcing eval mode
        for target network.
        """
        super().train(mode)
        self.target.eval()
        return self

    def update(self, tau: Optional[float] = None) -> None:
        """Applies EMA update from online network to target network."""
        current_tau = self.tau if tau is None else tau
        update_target_ema(self.online, self.target, current_tau)

    def update_with_schedule(
        self,
        step: int,
        total_steps: int,
        tau_min: float = 0.990,
        tau_max: float = 0.999,
        schedule: str = "linear",
    ) -> float:
        """
        Computes tau from schedule, updates target network, and stores current tau.
        """
        current_tau = compute_polyak_tau(step, total_steps, tau_min=tau_min, tau_max=tau_max, schedule=schedule)
        self.tau = current_tau
        self.update(current_tau)
        return current_tau

    def forward(self, *args: Any, use_target: bool = False, **kwargs: Any) -> Any:
        """
        Forwards input through either online or target network.

        Args:
            *args: Positional arguments for underlying module.
            use_target: If True, forwards through target network with stop-gradient guarantee.
                        If False, forwards through online network with active gradients.
            **kwargs: Keyword arguments for underlying module.
        """
        if use_target:
            with torch.no_grad():
                return self.target(*args, **kwargs)
        return self.online(*args, **kwargs)

    def forward_target(self, *args: Any, **kwargs: Any) -> Any:
        """Explicit shortcut for target forward with stop-gradient isolation."""
        with torch.no_grad():
            return self.target(*args, **kwargs)

    def forward_online(self, *args: Any, **kwargs: Any) -> Any:
        """Explicit shortcut for online forward with active autograd graph."""
        return self.online(*args, **kwargs)


__all__ = [
    "update_target_ema",
    "compute_polyak_tau",
    "PolyakMomentumScheduler",
    "TargetEMA",
]
