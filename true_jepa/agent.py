"""Strict visual control interface for the new JEPA model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from true_jepa.model import TrueJEPA


class VisualJEPAAgent:
    """Pixel-only recurrent agent with the repository-required act signature."""

    def __init__(self, model: TrueJEPA, device: str = "cpu", stochastic: bool = True) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.stochastic = bool(stochastic)
        self.hidden: Optional[Tensor] = None
        self.previous_action: Optional[int] = None

    def reset(self) -> None:
        self.hidden = None
        self.previous_action = None

    @torch.no_grad()
    def act(self, obs_uint8: np.ndarray, reward_prev: float, step: int) -> int:
        """Select an action from pixels, previous reward, and public step index only."""

        if int(step) == 0 or self.hidden is None:
            self.reset()
        obs = torch.from_numpy(np.ascontiguousarray(obs_uint8)).unsqueeze(0).to(self.device)
        frame = self.model.encode(obs)
        if self.hidden is None:
            self.hidden = self.model.initial_belief(1, self.device)
        elif self.previous_action is not None:
            previous = torch.tensor([self.previous_action], device=self.device)
            reward = torch.tensor([float(reward_prev)], device=self.device)
            self.hidden = self.model.belief_step(frame, previous, reward, self.hidden)
        logits = self.model.policy_logits(frame, self.hidden)
        if self.stochastic:
            action = torch.distributions.Categorical(logits=logits).sample()
        else:
            action = torch.argmax(logits, dim=-1)
        self.previous_action = int(action.item())
        return self.previous_action
