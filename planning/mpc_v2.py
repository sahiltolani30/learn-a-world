"""
Reward-Value MPC Planner for Learn-a-World v2.
Replaces curiosity-dispersion planning with expected task return maximization:
Score = sum_{t<H} gamma^t * r_hat_t + gamma^H * q_mean(z_H, a_H) - lambda_u * q_std(z_H, a_H) + 0.02 * novelty

Features:
- Dual Candidate Generation: 50% from PolicyPrior rolled autoregressively, 50% from CEM categorical.
- Per-candidate mask simulation: prevents achievement double-counting.
- Uncertainty Penalization: lambda_u * q_std penalizes out-of-distribution trajectories.
- Debug decomposition: returns {reward_sum, terminal_value, uncertainty_penalty} per step.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from models.jepa_v2 import JEPAv2
from planning.agent_v2 import AgentBelief

class RewardValuePlanner:
    """
    TD-MPC2 / Dreamer-style model predictive control planner using
    per-achievement rewards and terminal Q-ensemble returns.
    """

    def __init__(
        self,
        model: JEPAv2,
        horizon: int = 10,
        num_candidates: int = 256,
        cem_iterations: int = 3,
        elite_ratio: float = 0.15,
        gamma: float = 0.99,
        uncertainty_weight: float = 0.5,
        novelty_weight: float = 0.02,
        prior_mix: float = 0.5,
        num_actions: int = 17,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight
        self.novelty_weight = novelty_weight
        self.prior_mix = prior_mix
        self.num_actions = num_actions
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def plan(
        self,
        obs: np.ndarray,
        belief: AgentBelief,
        goal: Optional[np.ndarray] = None,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Plans next action given current observation and agent belief mask.
        obs: [64, 64, 3] uint8 or float32 observation
        belief: AgentBelief tracking unlocked mask without ground-truth info
        Returns: (action, debug_decomposition)
        """
        # If goal-conditioned request (GridMaze), delegate to legacy latent planner
        if goal is not None:
            from planning.mpc import LatentMPCPlanner
            legacy_planner = LatentMPCPlanner(
                model=self.model.base_model,
                horizon=self.horizon,
                num_candidates=self.num_candidates,
                elite_ratio=self.elite_ratio,
                num_actions=self.num_actions,
                device=self.device,
            )
            return legacy_planner.plan(obs, goal), {"reward_sum": 0.0, "terminal_value": 0.0, "uncertainty_penalty": 0.0}

        self.model.eval()

        # 1. Encode root observation
        if isinstance(obs, np.ndarray):
            if obs.shape[-1] == 3:  # [H, W, C]
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float()
            else:
                obs_t = torch.from_numpy(obs).unsqueeze(0).float()
            if obs_t.max() > 1.0:
                obs_t = obs_t / 255.0
        else:
            obs_t = obs.float()
            if obs_t.dim() == 3:
                obs_t = obs_t.unsqueeze(0)
            if obs_t.max() > 1.0:
                obs_t = obs_t / 255.0

        obs_t = obs_t.to(self.device)

        with torch.no_grad():
            z_0 = self.model.encode(obs_t, use_target=True)  # [1, 128]
            m_0 = belief.get_mask_tensor(device=self.device).unsqueeze(0)  # [1, 22]

            C = self.num_candidates
            N_prior = int(round(C * self.prior_mix))
            N_cem = C - N_prior

            # Initialize categorical CEM action distribution [H, 17]
            action_dist = torch.ones(self.horizon, self.num_actions, device=self.device) / self.num_actions

            best_action = 0
            best_debug: Dict[str, float] = {
                "reward_sum": 0.0,
                "terminal_value": 0.0,
                "uncertainty_penalty": 0.0,
                "best_score": 0.0,
            }

            for iteration in range(self.cem_iterations):
                # 2. Sample N_prior candidates autoregressively from PolicyPrior
                prior_actions = torch.empty((N_prior, self.horizon), dtype=torch.long, device=self.device)
                z_rollout = z_0.expand(N_prior, -1)
                m_rollout = m_0.expand(N_prior, -1).clone()

                for h in range(self.horizon):
                    a_h = self.model.sample_prior_action(z_rollout, m_rollout, temperature=1.0)
                    prior_actions[:, h] = a_h
                    p_h = self.model.event_head.predict_proba(z_rollout, a_h, m_rollout)
                    m_rollout = torch.where(p_h > 0.5, torch.tensor(1.0, device=self.device), m_rollout)
                    z_rollout = self.model.predict_next(z_rollout, a_h)

                # 3. Sample N_cem candidates from CEM categorical distribution
                cem_actions = torch.empty((N_cem, self.horizon), dtype=torch.long, device=self.device)
                for h in range(self.horizon):
                    probs_h = action_dist[h]
                    cem_actions[:, h] = torch.multinomial(probs_h, num_samples=N_cem, replacement=True)

                # Combined candidates: [C, horizon]
                actions_seq = torch.cat([prior_actions, cem_actions], dim=0)

                # 4. Rollout and score all C candidates
                z_t = z_0.expand(C, -1)
                m_t = m_0.expand(C, -1).clone()
                cumulative_reward = torch.zeros(C, device=self.device)

                for t in range(self.horizon):
                    a_t = actions_seq[:, t]
                    r_hat, decomp = self.model.predict_reward(z_t, a_t, m_t, return_decomposition=True)
                    cumulative_reward = cumulative_reward + (self.gamma ** t) * r_hat

                    # Simulated belief progression
                    p_t = decomp["event_probs"]
                    m_t = torch.where(p_t > 0.5, torch.tensor(1.0, device=self.device), m_t)
                    z_t = self.model.predict_next(z_t, a_t)

                # Terminal Q-value and uncertainty penalty at H
                a_terminal = self.model.sample_prior_action(z_t, m_t, temperature=1.0)
                q_mean_term = self.model.q_mean(z_t, a_terminal)
                q_std_term = self.model.q_std(z_t, a_terminal)

                terminal_value = (self.gamma ** self.horizon) * q_mean_term
                uncertainty_penalty = self.uncertainty_weight * q_std_term
                novelty = torch.norm(z_t - z_0, dim=-1)

                scores = cumulative_reward + terminal_value - uncertainty_penalty + self.novelty_weight * novelty

                # 5. Elite selection and CEM refit
                num_elites = max(1, int(round(C * self.elite_ratio)))
                elite_indices = torch.topk(scores, k=num_elites).indices
                elite_seqs = actions_seq[elite_indices]

                # Update categorical action distribution with Laplace smoothing
                for h in range(self.horizon):
                    counts = torch.bincount(elite_seqs[:, h], minlength=self.num_actions).float()
                    action_dist[h] = (counts + 0.1) / (num_elites + 0.1 * self.num_actions)

                # Track best sequence
                best_cand_idx = elite_indices[0].item()
                best_action = int(actions_seq[best_cand_idx, 0].item())
                best_debug = {
                    "reward_sum": float(cumulative_reward[best_cand_idx].item()),
                    "terminal_value": float(terminal_value[best_cand_idx].item()),
                    "uncertainty_penalty": float(uncertainty_penalty[best_cand_idx].item()),
                    "best_score": float(scores[best_cand_idx].item()),
                    "mean_q_std": float(q_std_term.mean().item()),
                }

        return best_action, best_debug
