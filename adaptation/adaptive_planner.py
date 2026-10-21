"""
Test-Time Adaptive JEPA Agent for Phase 4.
Core Scientific Mechanism:
When world dynamics secretly shift at test time, the agent adapts its latent predictor
purely using self-supervised prediction error ||z_hat_{t+1} - sg(z_bar_{t+1})||^2.
ZERO environment reward signal is accessed or utilized during adaptation.
"""

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.jepa import JEPAWorldModel
from planning.mpc import LatentMPCPlanner

class AdaptiveJEPAAgent:
    """
    Test-time adaptive agent that updates its world model on the fly
    solely from self-supervised visual prediction errors.
    """

    def __init__(
        self,
        model: JEPAWorldModel,
        horizon: int = 7,
        num_candidates: int = 128,
        lr: float = 1e-3,
        window_size: int = 64,
        steps_per_update: int = 2,
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.window_size = window_size
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size

        # 1. Adaptive Model & Optimizer (updates only predictor dynamics)
        self.model = copy.deepcopy(model).to(self.device)
        self.model.eval()
        self.optimizer = optim.AdamW(
            self.model.predictor.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )

        # 2. Frozen Control Baseline (never updates)
        self.frozen_model = copy.deepcopy(model).to(self.device)
        self.frozen_model.eval()

        # 3. Planners for both adaptive and frozen models
        self.planner = LatentMPCPlanner(
            model=self.model,
            horizon=horizon,
            num_candidates=num_candidates,
            elite_ratio=0.15,
            device=self.device,
        )
        self.frozen_planner = LatentMPCPlanner(
            model=self.frozen_model,
            horizon=horizon,
            num_candidates=num_candidates,
            elite_ratio=0.15,
            device=self.device,
        )

        # Sliding window buffer storing (obs, action, next_obs)
        self.sliding_window: deque = deque(maxlen=window_size)
        self.adaptation_errors: List[float] = []

    def reset_planners(self) -> None:
        """Resets planning trajectory history."""
        self.planner.reset_history()
        self.frozen_planner.reset_history()

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Converts observation array to tensor."""
        if obs.ndim == 3:
            t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif obs.ndim == 4:
            t = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255.0
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
        return t.to(self.device)

    def act(
        self,
        obs: np.ndarray,
        goal_obs: Optional[np.ndarray] = None,
        use_adaptive: bool = True,
    ) -> int:
        """
        Plans action using either the adaptive or frozen world model.
        """
        if use_adaptive:
            return self.planner.plan(obs, goal_obs)
        return self.frozen_planner.plan(obs, goal_obs)

    def observe_transition(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        # NOTICE: No 'reward' argument exists in this function.
        # Adaptation is mathematically guaranteed to be 100% self-supervised.
    ) -> float:
        """
        Receives transition, records prediction error, and performs self-supervised adaptation.
        Returns the instantaneous prediction error before the adaptation step.
        """
        x_t = self._preprocess_obs(obs)
        x_next = self._preprocess_obs(next_obs)
        act_tensor = torch.tensor([int(action)], device=self.device)

        # 1. Compute pre-update prediction error
        self.model.eval()
        with torch.no_grad():
            z_t = self.model.encode(x_t, use_target=False)
            z_pred = self.model.predictor(z_t, act_tensor)
            z_target = self.model.encode(x_next, use_target=True)
            error = torch.norm(z_pred - z_target, p=2).item()

        self.adaptation_errors.append(error)

        # 2. Add transition to recent sliding window
        self.sliding_window.append((x_t, act_tensor, x_next))

        # 3. Perform 1-3 self-supervised gradient steps on the predictor
        if len(self.sliding_window) >= 4:
            self.model.train()
            for _ in range(self.steps_per_update):
                # Sample batch from recent sliding window
                indices = np.random.choice(
                    len(self.sliding_window),
                    size=min(self.batch_size, len(self.sliding_window)),
                    replace=True,
                )
                b_xt = torch.cat([self.sliding_window[i][0] for i in indices], dim=0)
                b_act = torch.cat([self.sliding_window[i][1] for i in indices], dim=0)
                b_xnext = torch.cat([self.sliding_window[i][2] for i in indices], dim=0)

                with torch.no_grad():
                    b_zt = self.model.encode(b_xt, use_target=False)
                    b_ztarget = self.model.encode(b_xnext, use_target=True)

                self.optimizer.zero_grad()
                b_zpred = self.model.predictor(b_zt, b_act)
                loss = nn.functional.mse_loss(b_zpred, b_ztarget)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.predictor.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.model.eval()

        return error
