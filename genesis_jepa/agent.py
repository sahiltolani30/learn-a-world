"""
genesis_jepa.agent — Genesis Visual JEPA Agent
==============================================
Governing Laws: AGENTS.md & GEMINI.md (Strict Zero-Cheating Protocol)

Sacred Signature:
    act(obs_uint8: np.ndarray, reward_prev: float = 0.0, step: Optional[int] = None) -> int

Integrity Invariants:
  1. Input is strictly sensory RGB pixels uint8 [0, 255] and scalar previous reward.
  2. Zero simulator state dictionaries, coordinates, or inventory masks.
  3. Anti-Sleep Hazard Lock (bars Action 6 when L_dark >= 0.40).
  4. Cost-Sensitive Bayes Risk Decider (evasion cutoff tau* <= 0.082).
  5. Unimix Categorical Exploration (epsilon_u = 0.01).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genesis_jepa.config import GenesisConfig
from genesis_jepa.modules.causal_budgeter import UpstreamDeficitBudgeter
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

        # 4. Survival & Danger Estimator Head (predicts strike probability in [0, 1])
        self.danger_head = nn.Sequential(
            nn.Linear(self.config.token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        # Initialize danger head with safe baseline prior (P(hazard) ~ 0.029 < tau* = 0.082)
        with torch.no_grad():
            self.danger_head[-2].weight.mul_(0.01)
            self.danger_head[-2].bias.fill_(-3.5)

        # 5. Upstream Deficit Budgeter
        self.budgeter = UpstreamDeficitBudgeter()

        # Diurnal online extreme trackers for circadian darkness index
        self.nu_day = 1.0
        self.nu_night = 0.1
        self.beta_tracker = 1e-4

        # Estimated internal health (for dynamic Bayes cutoff)
        self.estimated_health = 9.0

    def reset(self) -> None:
        """Reset episodic transient state."""
        self.estimated_health = 9.0

    def compute_circadian_darkness(self, patch_tokens: torch.Tensor) -> float:
        """
        Derives cheat-free circadian darkness index L_dark purely from median patch token norms.
        """
        with torch.no_grad():
            token_norms = torch.norm(patch_tokens, p=2, dim=-1)  # (B, 64)
            median_norm = float(torch.median(token_norms).item())

            # Update moving extrema
            self.nu_day = max(self.nu_day, median_norm)
            self.nu_night = min(self.nu_night, median_norm)

            denom = max(self.nu_day - self.nu_night, 1e-4)
            l_dark = float(np.clip((self.nu_day - median_norm) / denom, 0.0, 1.0))
            return l_dark

    def compute_dynamic_bayes_cutoff(self, health: float) -> float:
        """
        Dynamically calculates Bayes-optimal risk cutoff tau*(H):
            Full health (H=9): tau* <= 0.082
            Critical health (H<=2): tau* <= 0.012
        """
        c_false = 1.0
        c_mitigated = 1.0
        h_clamped = max(min(health, 9.0), 1.0)
        c_harm = 100.0 * (((10.0 - h_clamped) / 9.0) ** 1.85) + 11.20
        tau = c_false / ((c_harm - c_mitigated) + c_false)
        return float(min(tau, self.config.bayes_risk_cutoff_max))

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
            # 1. Visual Retina: Extract patch and slot tokens
            patch_tokens, slot_tokens = self.world_model.encode(obs_t)

            # 2. Pooled object representation across slots
            pooled_slots = torch.mean(slot_tokens, dim=1)  # (1, 192)

            # 3. Circadian Darkness Perception
            l_dark = self.compute_circadian_darkness(patch_tokens)

            # 4. Instantaneous Threat Risk Assessment
            danger_risk = float(self.danger_head(pooled_slots).item())

            # 5. Actor Policy Logits
            action_logits = self.actor(pooled_slots).squeeze(0)  # (17,)

            # 6. Anti-Sleep Hazard Lock: Action 6 (sleep) is banned in open / dangerous conditions
            if l_dark >= self.config.night_darkness_threshold or danger_risk > self.config.danger_threshold:
                action_logits[6] = -1e9  # Bar Action 6 strictly

            # 7. Cost-Sensitive Bayes Risk Decision: Dodge if threat exceeds threshold
            tau_star = self.compute_dynamic_bayes_cutoff(self.estimated_health)
            if danger_risk > tau_star:
                # Evasive Reflex: prioritize locomotion dodges (Actions 1..4)
                evasive_logits = action_logits[1:5]
                chosen_move = int(torch.argmax(evasive_logits).item()) + 1
                return chosen_move

            # 8. Unimix Categorical Exploration (epsilon floor)
            probs = F.softmax(action_logits, dim=-1)
            unimix_probs = (1.0 - self.config.unimix_floor) * probs + (self.config.unimix_floor / self.config.num_actions)

            # Re-enforce safety lock: strictly banned actions (logit <= -1e8) must NEVER be sampled
            banned_mask = (action_logits <= -1e8)
            unimix_probs[banned_mask] = 0.0
            unimix_probs = unimix_probs / unimix_probs.sum()

            # Sample action
            action = int(torch.multinomial(unimix_probs, num_samples=1).item())
            return action
