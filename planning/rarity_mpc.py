"""
planning/rarity_mpc.py — Curiosity + Rarity Augmented Recurrent MPC Planner (T2 Part 4)

Extends RecurrentMPCPlanner to optimize a tri-objective reward inside imagined rollouts:
    r_total = lambda_task * r_task + lambda_cur * r_curiosity + lambda_rare * r_rarity

Components:
1. r_task: Predicted achievement unlock reward + health change (calibrated via EventHeadCalibrator)
2. r_curiosity: Epistemic disagreement across Plan2Explore predictors in latent space
3. r_rarity: Inverse-frequency weighted bonus sum_k w_k * P_k(s, a) for currently rare achievements
4. Terminal Q: Clipped Double-Q terminal value with epistemic uncertainty penalization

Standing Rules:
- All achievement indexing is integer-only (0 to 21).
- Strictly ZERO achievement names in code logic.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from models.curiosity import Plan2ExploreCuriosity
from models.jepa_v3 import JEPAv3
from models.rarity import RarityWeightScheduler
from planning.agent_v3 import AgentBeliefV3


class CuriosityRarityMPCPlanner:
    """
    Model predictive control planner for Learn-a-World v3 Tranche T2 collection.
    Integrates task rewards, curiosity disagreement, and rarity weights into CEM planning.
    """

    def __init__(
        self,
        model: JEPAv3,
        curiosity: Plan2ExploreCuriosity,
        rarity_scheduler: RarityWeightScheduler,
        lambda_task: float = 0.3,
        lambda_cur: float = 1.0,
        lambda_rare: float = 0.5,
        horizon: int = 10,
        num_candidates: int = 64,
        cem_iterations: int = 2,
        elite_ratio: float = 0.15,
        gamma: float = 0.99,
        uncertainty_weight: float = 0.5,
        prior_mix: float = 0.5,
        num_actions: int = 17,
        use_calibration: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.curiosity = curiosity
        self.rarity_scheduler = rarity_scheduler
        self.lambda_task = float(lambda_task)
        self.lambda_cur = float(lambda_cur)
        self.lambda_rare = float(lambda_rare)

        self.horizon = horizon
        self.num_candidates = num_candidates
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight
        self.prior_mix = prior_mix
        self.num_actions = num_actions
        self.use_calibration = use_calibration
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.model.to(self.device)
        self.curiosity.to(self.device)

    def plan(
        self,
        obs: np.ndarray,
        belief: AgentBeliefV3,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Plans next action given root visual observation and belief state.
        Returns: (best_action, debug_metrics)
        """
        self.model.eval()
        self.curiosity.eval()

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

            # Pre-fetch rarity weights tensor
            rarity_weights = self.rarity_scheduler.get_weights_tensor(self.device)  # [22]

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
                cum_task_reward = torch.zeros(C, device=self.device)
                cum_curiosity_bonus = torch.zeros(C, device=self.device)
                cum_rarity_bonus = torch.zeros(C, device=self.device)

                for t in range(self.horizon):
                    a_t = actions_seq[:, t]

                    # A. Task reward (achievements + health)
                    r_task, decomp = self.model.predict_reward(
                        s_t, a_t, m_t, return_decomposition=True, use_calibration=self.use_calibration
                    )
                    p_events = decomp["event_probs"]  # [C, 22]

                    # B. Curiosity bonus (disagreement across 3 predictors)
                    r_cur = self.curiosity.curiosity_bonus(z_t, h_t, a_t)  # [C]

                    # C. Rarity bonus: sum_k w_k * p_k * (1 - m_k)
                    locked_mask = 1.0 - m_t.float()  # [C, 22]
                    r_rare = torch.sum(p_events * locked_mask * rarity_weights.unsqueeze(0), dim=-1)  # [C]

                    # Tri-objective composite step reward
                    r_step = self.lambda_task * r_task + self.lambda_cur * r_cur + self.lambda_rare * r_rare
                    discount = self.gamma ** t

                    cumulative_reward = cumulative_reward + discount * r_step
                    cum_task_reward = cum_task_reward + discount * r_task
                    cum_curiosity_bonus = cum_curiosity_bonus + discount * r_cur
                    cum_rarity_bonus = cum_rarity_bonus + discount * r_rare

                    # Belief progression & dynamics transition
                    m_t = torch.where(p_events > 0.5, torch.tensor(1.0, device=self.device), m_t)
                    z_t, h_t, s_t = self.model.predict_next_step(z_t, h_t, a_t)

                # 5. Terminal Clipped Double-Q evaluation at horizon H
                a_terminal = self.model.sample_prior_action(s_t, m_t, temperature=1.0)
                q_term = self.model.q_min(s_t, a_terminal, use_target=True)
                q_std_term = self.model.q_std(s_t, a_terminal, use_target=True)

                terminal_value = (self.gamma ** self.horizon) * q_term
                uncertainty_penalty = self.uncertainty_weight * q_std_term

                scores = cumulative_reward + terminal_value - uncertainty_penalty

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
                    "total_score": float(scores[best_idx].item()),
                    "cum_total_reward": float(cumulative_reward[best_idx].item()),
                    "task_reward_sum": float(cum_task_reward[best_idx].item()),
                    "curiosity_bonus_sum": float(cum_curiosity_bonus[best_idx].item()),
                    "rarity_bonus_sum": float(cum_rarity_bonus[best_idx].item()),
                    "terminal_value": float(terminal_value[best_idx].item()),
                    "uncertainty_penalty": float(uncertainty_penalty[best_idx].item()),
                    "mean_q_std": float(q_std_term.mean().item()),
                }

        return best_action, best_debug
