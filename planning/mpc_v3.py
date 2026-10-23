"""
Recurrent Reward-Value MPC Planner for Learn-a-World v3 (R2 + R4).
Extends MPC planning with recurrent belief propagation through imagined trajectories:
Score = sum_{t<H} gamma^t * r_hat_t + gamma^H * q_min(s_H, a_H) - lambda_u * q_std(s_H, a_H) + lambda_nov * novelty

Features:
- Recurrent State Unrolls: Each candidate propagates both visual latent z_t and memory h_t.
- Prior-Guided Dual Sampling: 50% candidates generated autoregressively by PolicyPrior, 50% CEM.
- Clipped Double-Q: Uses q_min to prevent terminal value overestimation.
- Epistemic Uncertainty Penalization: lambda_u * q_std penalizes uncertain paths.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3


class RecurrentMPCPlanner:
    """
    TD-MPC2 / Dreamer-style model predictive control planner for JEPA v3
    using recurrent belief states and Clipped Double-Q terminal returns.
    """

    def __init__(
        self,
        model: JEPAv3,
        horizon: int = 10,
        num_candidates: int = 64,
        cem_iterations: int = 2,
        elite_ratio: float = 0.15,
        gamma: float = 0.99,
        uncertainty_weight: float = 0.5,
        novelty_weight: float = 0.02,
        prior_mix: float = 0.5,
        num_actions: int = 17,
        use_calibration: bool = True,
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
        self.use_calibration = use_calibration
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def plan(
        self,
        obs: np.ndarray,
        belief: AgentBeliefV3,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Plans next action given current visual observation and stateful belief.
        obs: [64, 64, 3] uint8 or float observation
        belief: AgentBeliefV3 holding mask m_0 and recurrent memory h_0
        Returns: (best_action, debug_metrics)
        """
        self.model.eval()

        # 1. Encode root visual observation
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
            z_0 = self.model.encode(obs_t, use_target=False)  # [1, 128]
            h_0 = belief.get_h_tensor(device=self.device).unsqueeze(0)  # [1, 256]
            m_0 = belief.get_mask_tensor(device=self.device).unsqueeze(0)  # [1, 22]
            s_0 = torch.cat([z_0, h_0], dim=-1)  # [1, 384]

            C = self.num_candidates
            N_prior = int(round(C * self.prior_mix))
            N_cem = C - N_prior

            # Initialize categorical CEM action distribution [H, 17]
            action_dist = torch.ones(self.horizon, self.num_actions, device=self.device) / self.num_actions

            best_action = 0
            best_debug: Dict[str, float] = {}

            for iteration in range(self.cem_iterations):
                # 2. Sample N_prior candidates autoregressively using PolicyPrior
                prior_actions = torch.empty((N_prior, self.horizon), dtype=torch.long, device=self.device)
                z_roll = z_0.expand(N_prior, -1)
                h_roll = h_0.expand(N_prior, -1)
                s_roll = s_0.expand(N_prior, -1)
                m_roll = m_0.expand(N_prior, -1).clone()

                for h_step in range(self.horizon):
                    a_h = self.model.sample_prior_action(s_roll, m_roll, temperature=1.0)
                    prior_actions[:, h_step] = a_h
                    p_h = self.model.event_head.predict_proba(s_roll, a_h, m_roll, use_calibration=self.use_calibration)
                    m_roll = torch.where(p_h > 0.5, torch.tensor(1.0, device=self.device), m_roll)
                    z_roll, h_roll, s_roll = self.model.predict_next_step(z_roll, h_roll, a_h)

                # 3. Sample N_cem candidates from CEM categorical distribution
                cem_actions = torch.empty((N_cem, self.horizon), dtype=torch.long, device=self.device)
                for h_step in range(self.horizon):
                    probs_h = action_dist[h_step]
                    cem_actions[:, h_step] = torch.multinomial(probs_h, num_samples=N_cem, replacement=True)

                # Combine candidates: [C, horizon]
                actions_seq = torch.cat([prior_actions, cem_actions], dim=0)

                # 4. Rollout and score all C candidates through recurrent dynamics
                z_t = z_0.expand(C, -1)
                h_t = h_0.expand(C, -1)
                s_t = s_0.expand(C, -1)
                m_t = m_0.expand(C, -1).clone()
                cumulative_reward = torch.zeros(C, device=self.device)

                for t in range(self.horizon):
                    a_t = actions_seq[:, t]
                    r_hat, decomp = self.model.predict_reward(
                        s_t, a_t, m_t, return_decomposition=True, use_calibration=self.use_calibration
                    )
                    cumulative_reward = cumulative_reward + (self.gamma ** t) * r_hat

                    # Simulated belief progression
                    p_t = decomp["event_probs"]
                    m_t = torch.where(p_t > 0.5, torch.tensor(1.0, device=self.device), m_t)
                    z_t, h_t, s_t = self.model.predict_next_step(z_t, h_t, a_t)

                # 5. Terminal Clipped Double-Q evaluation at horizon H
                a_terminal = self.model.sample_prior_action(s_t, m_t, temperature=1.0)
                q_term = self.model.q_min(s_t, a_terminal, use_target=True)
                q_std_term = self.model.q_std(s_t, a_terminal, use_target=True)

                terminal_value = (self.gamma ** self.horizon) * q_term
                uncertainty_penalty = self.uncertainty_weight * q_std_term
                novelty = torch.norm(z_t - z_0, dim=-1)

                scores = cumulative_reward + terminal_value - uncertainty_penalty + self.novelty_weight * novelty

                # 6. Elite selection and CEM refitting
                num_elites = max(1, int(round(C * self.elite_ratio)))
                elite_indices = torch.topk(scores, k=num_elites).indices
                elite_seqs = actions_seq[elite_indices]

                for h_step in range(self.horizon):
                    counts = torch.bincount(elite_seqs[:, h_step], minlength=self.num_actions).float()
                    action_dist[h_step] = (counts + 0.1) / (num_elites + 0.1 * self.num_actions)

                # Track best candidate
                best_idx = elite_indices[0].item()
                best_action = int(actions_seq[best_idx, 0].item())
                best_debug = {
                    "reward_sum": float(cumulative_reward[best_idx].item()),
                    "terminal_value": float(terminal_value[best_idx].item()),
                    "uncertainty_penalty": float(uncertainty_penalty[best_idx].item()),
                    "best_score": float(scores[best_idx].item()),
                    "mean_q_std": float(q_std_term.mean().item()),
                }

        return best_action, best_debug
