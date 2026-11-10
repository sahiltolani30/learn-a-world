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

    def evaluate_1step_lookahead(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Rolls forward 1 step in latent space for all discrete actions in parallel.
        Evaluates the resulting future state using the 65-bin distributional critic.
        Returns:
            q_values: Expected return for each action, shape (num_actions,)
        """
        dev = patch_tokens.device
        num_act = self.config.num_actions

        # Repeat initial state across all actions: (num_act, 64, token_dim)
        z_rep = patch_tokens.repeat(num_act, 1, 1)
        act_cand = torch.arange(num_act, device=dev)

        next_patches, _, _ = self.world_model.predictor(z_rep, act_cand)
        next_slots = self.world_model.encoder.slot_attention(next_patches)
        next_pooled = torch.mean(next_slots, dim=1)  # (num_act, token_dim)
        q_vals = self.critic.predict_scalar(next_pooled)  # (num_act,)
        return q_vals

    def evaluate_multistep_rollout(
        self,
        patch_tokens: torch.Tensor,
        prior_logits: torch.Tensor,
        horizon: int = 3,
        num_candidates: int = 16,
    ) -> torch.Tensor:
        """
        Simulates candidate multi-step action trajectories through the JEPA predictor.
        Aggregates future expected return to evaluate candidate first actions.
        """
        dev = patch_tokens.device
        num_act = self.config.num_actions

        # 1. Base scores from 1-step lookahead across all actions
        q_1step = self.evaluate_1step_lookahead(patch_tokens)

        if horizon <= 1:
            return q_1step

        # 2. For multi-step: sample candidate future action sequences of shape (K, horizon)
        prior_probs = F.softmax(prior_logits, dim=-1)
        first_actions = torch.arange(min(num_act, num_candidates), device=dev)
        if num_candidates > num_act:
            extra = torch.multinomial(prior_probs, num_samples=num_candidates - num_act, replacement=True)
            first_actions = torch.cat([first_actions, extra])

        K = len(first_actions)
        subsequent_actions = torch.multinomial(
            prior_probs.repeat(K, 1),
            num_samples=horizon - 1,
            replacement=True,
        )  # (K, H - 1)
        action_seq = torch.cat([first_actions.unsqueeze(1), subsequent_actions], dim=1)  # (K, H)

        # 3. Roll out multi-step trajectory in JEPA latent space
        z_curr = patch_tokens.repeat(K, 1, 1)  # (K, 64, D)
        cumulative_return = torch.zeros(K, device=dev)
        discount = 1.0

        for h in range(horizon):
            act_step = action_seq[:, h]
            z_curr, _, _ = self.world_model.predictor(z_curr, act_step)
            if h == horizon - 1 or h == 0:
                slots = self.world_model.encoder.slot_attention(z_curr)
                pooled = torch.mean(slots, dim=1)
                val_step = self.critic.predict_scalar(pooled)
                cumulative_return += (discount * val_step)
            discount *= self.config.gamma

        # 4. Map trajectory returns back to first actions
        q_multistep = q_1step.clone()
        for idx in range(K):
            a0 = int(first_actions[idx].item())
            q_multistep[a0] = max(float(q_multistep[a0]), float(cumulative_return[idx]))

        return q_multistep

    def act(
        self,
        obs_uint8: np.ndarray,
        reward_prev: float = 0.0,
        step: Optional[int] = None,
    ) -> int:
        """
        Sacred decision interface. Strictly takes raw pixels and scalar reward.
        Connects the World Model Predictor & Critic to choose actions via Latent Planning.
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

            # 3. Policy Actor Prior
            actor_logits = self.actor(pooled_slots).squeeze(0)  # (num_actions,)

            # 4. Action Decision: Latent Model-Predictive Planning
            mode = getattr(self.config, "planning_mode", "hybrid_mppi")

            if mode == "reflex":
                # Reflex baseline: MLP pattern matching without lookahead
                effective_logits = actor_logits

            elif mode == "lookahead_1step":
                # Pure 1-step mental simulation: Q(s, a) = V(Predictor(s, a))
                q_vals = self.evaluate_1step_lookahead(patch_tokens)
                effective_logits = q_vals / max(self.config.plan_temperature, 0.01)

            elif mode == "multistep_rollout":
                # Multi-step mental simulation over horizon H
                q_vals = self.evaluate_multistep_rollout(
                    patch_tokens,
                    actor_logits,
                    horizon=self.config.plan_horizon,
                    num_candidates=self.config.plan_num_candidates,
                )
                effective_logits = q_vals / max(self.config.plan_temperature, 0.01)

            elif mode == "hybrid_mppi":
                # Hybrid MPPI: Policy Actor Prior + JEPA Lookahead Return
                q_vals = self.evaluate_1step_lookahead(patch_tokens)
                effective_logits = actor_logits + (q_vals / max(self.config.plan_temperature, 0.01))

            else:
                effective_logits = actor_logits

            # 5. Unimix Categorical Exploration (domain-agnostic tabula rasa sampling)
            probs = F.softmax(effective_logits, dim=-1)
            unimix_probs = (1.0 - self.config.unimix_floor) * probs + (
                self.config.unimix_floor / self.config.num_actions
            )

            # Sample action purely from planned policy + exploration
            action = int(torch.multinomial(unimix_probs, num_samples=1).item())
            return action
