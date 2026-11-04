"""Active probe scorer, dual value heads, and exploration policy for MOSAIC Phase P3.

Balances epistemic information gain against survival risk and returnability.
Maintains separate task and information value streams.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mosaic.src.api import validate_action, validate_observation, validate_reward, validate_step
from mosaic.src.models.world_model import MosaicWorldModel


class DualValueHeads(nn.Module):
    """Maintains independent task (extrinsic) and information (intrinsic) value heads.

    Separating the heads prevents non-stationary epistemic bonuses from corrupting
    survival and task reward credit assignment.
    """

    def __init__(self, belief_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.belief_dim = belief_dim

        # Extrinsic task value head: V_task(b)
        self.task_head = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Intrinsic information value head: V_info(b)
        self.info_head = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, belief: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            belief: [B, belief_dim]
        Returns:
            v_task: [B, 1] predicted extrinsic task value
            v_info: [B, 1] predicted intrinsic information value
        """
        return self.task_head(belief), self.info_head(belief)


class ActiveProbeScorer:
    """Scores candidate actions by expected epistemic information gain minus survival risk."""

    def __init__(
        self,
        world_model: MosaicWorldModel,
        beta_epistemic: float = 1.0,
        rho_risk: float = 5.0,
        step_cost: float = 0.01,
        max_survival_risk: float = 0.8,
    ) -> None:
        self.world_model = world_model
        self.beta_epistemic = float(beta_epistemic)
        self.rho_risk = float(rho_risk)
        self.step_cost = float(step_cost)
        self.max_survival_risk = float(max_survival_risk)

    def score_candidate_actions(
        self,
        belief: torch.Tensor,
    ) -> Dict[str, Any]:
        """Evaluate all candidate action tokens at the given belief state.

        Args:
            belief: [1, belief_dim] current predictive belief
        Returns:
            dict containing:
                scores: [num_actions] probe scores J_probe(a)
                disagreements: [num_actions] epistemic disagreements
                death_risks: [num_actions] predicted termination probabilities
                rewards: [num_actions] predicted immediate rewards
                best_action: integer token with highest probe score
        """
        device = belief.device
        num_actions = self.world_model.num_actions

        # Expand belief across all possible candidate actions
        b_expanded = belief.expand(num_actions, -1)
        actions = torch.arange(num_actions, dtype=torch.long, device=device)

        with torch.no_grad():
            dyn_out = self.world_model.imagine_transition(b_expanded, actions)

        disagreements = dyn_out["disagreement"]  # [num_actions]
        continue_logits = dyn_out["continue_logit_mean"].squeeze(-1)  # [num_actions]
        pred_rewards = dyn_out["reward_mean"].squeeze(-1)  # [num_actions]

        # Predicted death risk: P(terminated) = 1 - sigmoid(continue_logit)
        p_death = 1.0 - torch.sigmoid(continue_logits)

        # J_probe(a) = beta * D(b, a) - rho * P_death(b, a) - cost
        # If death risk exceeds max_survival_risk, apply an extra heavy penalty
        risk_penalty = self.rho_risk * p_death
        death_mask = (p_death > self.max_survival_risk).float()
        scores = (
            self.beta_epistemic * disagreements
            - risk_penalty
            - self.step_cost
            - 10.0 * death_mask
        )

        best_action = int(torch.argmax(scores).item())

        return {
            "scores": scores.cpu().numpy(),
            "disagreements": disagreements.cpu().numpy(),
            "death_risks": p_death.cpu().numpy(),
            "pred_rewards": pred_rewards.cpu().numpy(),
            "best_action": best_action,
        }


class ActiveProbePolicy:
    """Visual policy implementing the strict act(obs, reward, step) contract with active probing.

    Arbitrates between epistemic information seeking and survival constraints.
    Supports shadow-mode evaluation to log probe traces without modifying actions.
    """

    def __init__(
        self,
        world_model: MosaicWorldModel,
        mode: str = "active_probe",
        beta_epistemic: float = 1.0,
        rho_risk: float = 5.0,
        step_cost: float = 0.01,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.world_model = world_model
        self.num_actions = world_model.num_actions
        self.mode = mode
        self.device = device if device is not None else next(world_model.parameters()).device
        self.rng = np.random.default_rng(seed)

        self.scorer = ActiveProbeScorer(
            world_model=world_model,
            beta_epistemic=beta_epistemic,
            rho_risk=rho_risk,
            step_cost=step_cost,
        )

        # Private recurrent state
        self._h: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._last_action = 0
        self.trace_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset private recurrent state at episode boundary."""
        h_0, z_0 = self.world_model.belief.initial_state(1, device=self.device)
        self._h = h_0
        self._z = z_0
        self._last_action = 0

    def act(self, obs_uint8: np.ndarray, reward_prev: float, step: int) -> int:
        """Public control API: act(obs_uint8, reward_prev, step) -> int."""
        obs = validate_observation(obs_uint8)
        r_prev = validate_reward(reward_prev)
        cur_step = validate_step(step)

        if cur_step == 0 or self._h is None or self._z is None:
            self.reset()

        # Format observation tensor: [1, 3, 64, 64] float in [0, 1]
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        a_prev_tensor = torch.tensor([self._last_action], dtype=torch.long, device=self.device)
        r_prev_tensor = torch.tensor([[r_prev]], dtype=torch.float32, device=self.device)

        # Update belief
        with torch.no_grad():
            b_t, self._h, _, _, _ = self.world_model.step_belief(
                obs_uint8_or_float=obs_tensor,
                action_token_prev=a_prev_tensor,
                reward_prev=r_prev_tensor,
                h_prev=self._h,
                z_prev=self._z,
                use_posterior=True,
            )
            self._z = b_t[:, self.world_model.hidden_dim :]

            # Score all candidate actions
            score_out = self.scorer.score_candidate_actions(b_t)

        scores = score_out["scores"]
        disagreements = score_out["disagreements"]
        death_risks = score_out["death_risks"]
        pred_rewards = score_out["pred_rewards"]

        if self.mode == "active_probe":
            selected_action = score_out["best_action"]
        elif self.mode == "disagreement_only":
            # Raw curiosity without death penalty
            selected_action = int(np.argmax(disagreements))
        elif self.mode == "task_only":
            # Immediate predicted reward
            selected_action = int(np.argmax(pred_rewards))
        elif self.mode == "random":
            selected_action = int(self.rng.integers(0, self.num_actions))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Store shadow trace for offline inspection
        trace_entry = {
            "step": cur_step,
            "chosen_action": selected_action,
            "best_probe_action": score_out["best_action"],
            "disagreement": float(disagreements[selected_action]),
            "death_risk": float(death_risks[selected_action]),
            "pred_reward": float(pred_rewards[selected_action]),
        }
        self.trace_history.append(trace_entry)
        self._last_action = selected_action

        return validate_action(selected_action, self.num_actions)
