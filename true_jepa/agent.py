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


class BatchedVisualJEPAAgent:
    """One shared model controlling independent environments in one forward pass."""

    def __init__(self, model: TrueJEPA, batch_size: int, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.hidden = self.model.initial_belief(self.batch_size, self.device)
        self.previous_action = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def reset(self, indices: Optional[np.ndarray] = None) -> None:
        if indices is None:
            self.hidden.zero_()
            self.previous_action.zero_()
            return
        selected = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.hidden[selected] = 0.0
        self.previous_action[selected] = 0

    @torch.no_grad()
    def act(
        self,
        obs_uint8: np.ndarray,
        reward_prev: np.ndarray,
        step: np.ndarray,
    ) -> np.ndarray:
        """Return one action per environment using one batched inference call."""

        observations = torch.from_numpy(np.ascontiguousarray(obs_uint8)).to(self.device)
        rewards = torch.as_tensor(reward_prev, dtype=torch.float32, device=self.device).reshape(-1)
        steps = torch.as_tensor(step, dtype=torch.long, device=self.device).reshape(-1)
        if observations.shape[0] != self.batch_size:
            raise ValueError("observation batch does not match agent batch_size")
        fresh = steps == 0
        if bool(fresh.any()):
            self.reset(torch.nonzero(fresh, as_tuple=False).flatten().cpu().numpy())
        frames = self.model.encode(observations)
        if bool((~fresh).any()):
            updated = self.model.belief_step(frames, self.previous_action, rewards, self.hidden)
            self.hidden = torch.where((~fresh).unsqueeze(1), updated, self.hidden)
        logits = self.model.policy_logits(frames, self.hidden)
        actions = torch.distributions.Categorical(logits=logits).sample()
        self.previous_action.copy_(actions)
        return actions.cpu().numpy().astype(np.int64)

    def record_actions(self, actions: np.ndarray) -> None:
        """Replace policy actions with the actions actually executed."""

        self.previous_action.copy_(torch.as_tensor(actions, dtype=torch.long, device=self.device))
