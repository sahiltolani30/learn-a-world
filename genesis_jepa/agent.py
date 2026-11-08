"""
genesis_jepa.agent — Genesis Visual JEPA Agent
==============================================
Governing Laws: AGENTS.md & GEMINI.md (Strict Zero-Cheating Protocol)

Sacred Signature:
    act(obs_uint8: np.ndarray, reward_prev: float = 0.0, step: Optional[int] = None) -> int

Integrity Invariants:
  1. Input is strictly sensory RGB pixels uint8 [0, 255] and scalar previous reward.
  2. Zero simulator state dictionaries, coordinates, or inventory masks.
  3. Zero hardcoded action slicing, recipe tables, or game-specific rules.
  4. Tabula rasa policy: Actions optimized purely via Actor-Critic reinforcement learning.
  5. Unimix Categorical Exploration across full action space.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genesis_jepa.config import GenesisConfig
from genesis_jepa.modules.distributional_critic import SymlogDistributionalCritic
from genesis_jepa.world_model import GenesisJEPA


class GenesisVisualAgent(nn.Module):
    """
    Autonomous Visual Agent interacting purely via raw visual observations.
    """

    def __init__(self, config: Optional[GenesisConfig] = None) -> None:
        super().__init__()
        self.config = config or GenesisConfig()

        # 1. World Model Core (JEPA)
        self.world_model = GenesisJEPA(self.config)

        # 2. Distributional Critic (65-bin symlog)
        self.critic = SymlogDistributionalCritic(
            in_dim=self.config.token_dim,
            num_bins=self.config.symlog_bins,
            v_min=self.config.symlog_v_min,
            v_max=self.config.symlog_v_max,
        )

        # 3. Policy Actor Head
        # Takes pooled slot representation (token_dim = 192) and outputs 17 action logits
        self.actor = nn.Sequential(
            nn.Linear(self.config.token_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.config.num_actions),
        )

        # 4. Auxiliary Danger Estimator Head (predicts strike probability in [0, 1])
        self.danger_head = nn.Sequential(
            nn.Linear(self.config.token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def reset(self) -> None:
        """Reset episodic transient state."""
        pass

    def act(
        self,
        obs_uint8: np.ndarray,
        reward_prev: float = 0.0,
        step: Optional[int] = None,
    ) -> int:
        """
        Sacred decision interface. Strictly takes raw pixels and scalar reward.
        """
        # Ensure tensor format and device
        dev = next(self.parameters()).device
        if isinstance(obs_uint8, np.ndarray):
            obs_t = torch.from_numpy(obs_uint8).to(dev)
        else:
            obs_t = obs_uint8.to(dev)

        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)  # (1, 64, 64, 3)

        self.eval()
        with torch.no_grad():
            # 1. Visual Retina: Extract patch and slot tokens from raw pixels
            patch_tokens, slot_tokens = self.world_model.encode(obs_t)

            # 2. Pooled object representation across slots
            pooled_slots = torch.mean(slot_tokens, dim=1)  # (1, 192)

            # 3. Policy Actor: Directly predicts action logits from learned latent representation
            action_logits = self.actor(pooled_slots).squeeze(0)  # (num_actions,)

            # 4. Unimix Categorical Exploration (domain-agnostic tabula rasa sampling)
            probs = F.softmax(action_logits, dim=-1)
            unimix_probs = (1.0 - self.config.unimix_floor) * probs + (
                self.config.unimix_floor / self.config.num_actions
            )

            # Sample action purely from learned policy + exploration
            action = int(torch.multinomial(unimix_probs, num_samples=1).item())
            return action
