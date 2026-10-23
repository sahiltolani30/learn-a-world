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
    Integrates task rewards, curiosity disagreement, rarity weights, and epistemic uncertainty
    into CEM planning with running scale (EMA) normalization.
    """

    def __init__(
        self,
        model: JEPAv3,
        curiosity: Plan2ExploreCuriosity,
        rarity_scheduler: RarityWeightScheduler,
        lambda_task: float = 1.0,
        lambda_cur: float = 0.5,
        lambda_rare: float = 1.0,
        lambda_q: float = 0.3,
        cap_q: float = 0.5,
        horizon: int = 10,
        num_candidates: int = 64,
        cem_iterations: int = 2,
        elite_ratio: float = 0.15,
        gamma: float = 0.99,
        uncertainty_weight: float = 0.0,  # Deprecated terminal penalty: kept 0 for API compatibility
        prior_mix: float = 0.5,
        num_actions: int = 17,
        use_calibration: bool = True,
        ema_alpha: float = 0.01,
        ema_clamp_min: float = 0.05,
        ema_clamp_max: float = 20.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.curiosity = curiosity
        self.rarity_scheduler = rarity_scheduler
        self.lambda_task = float(lambda_task)
        self.lambda_cur = float(lambda_cur)
        self.lambda_rare = float(lambda_rare)
        self.lambda_q = float(lambda_q)
        self.cap_q = float(cap_q)

        self.horizon = horizon
        self.num_candidates = num_candidates
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.gamma = gamma
        self.uncertainty_weight = float(uncertainty_weight)
        self.prior_mix = prior_mix
        self.num_actions = num_actions
        self.use_calibration = use_calibration
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Running scale (EMA) normalization trackers
        self.ema_alpha = float(ema_alpha)
        self.ema_clamp_min = float(ema_clamp_min)
        self.ema_clamp_max = float(ema_clamp_max)

        self.ema_task: Optional[float] = None
        self.ema_cur: Optional[float] = None
        self.ema_rare: Optional[float] = None
        self.ema_q: Optional[float] = None

        self.init_ema_task: Optional[float] = None
        self.init_ema_cur: Optional[float] = None
        self.init_ema_rare: Optional[float] = None
        self.init_ema_q: Optional[float] = None

        self.model.to(self.device)
        self.curiosity.to(self.device)

    def _ema_normalize(self, val: torch.Tensor, name: str) -> torch.Tensor:
        """
        Normalizes a reward tensor by running EMA of mean absolute value.
        Clamps divisor to [ema_clamp_min * init, ema_clamp_max * init].
        """
        batch_mean = float(torch.mean(torch.abs(val.detach())).item())
        if batch_mean < 1e-6:
            batch_mean = 1e-4

        ema_attr = f"ema_{name}"
        init_attr = f"init_ema_{name}"

        curr_ema = getattr(self, ema_attr)
        if curr_ema is None:
            setattr(self, ema_attr, batch_mean)
            setattr(self, init_attr, batch_mean)
            curr_ema = batch_mean
            init_val = batch_mean
        else:
            curr_ema = (1.0 - self.ema_alpha) * curr_ema + self.ema_alpha * batch_mean
            setattr(self, ema_attr, curr_ema)
            init_val = getattr(self, init_attr)

        min_div = self.ema_clamp_min * init_val
        max_div = self.ema_clamp_max * init_val
        clamped_div = max(min_div, min(max_div, curr_ema))

        return val / clamped_div

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
                cum_unc_bonus = torch.zeros(C, device=self.device)

                cum_task_weighted = torch.zeros(C, device=self.device)
                cum_cur_weighted = torch.zeros(C, device=self.device)
                cum_rare_weighted = torch.zeros(C, device=self.device)
                cum_unc_weighted = torch.zeros(C, device=self.device)

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

                    # D. Per-step epistemic Q-uncertainty bonus (capped at cap_q)
                    q_std_step = self.model.q_std(s_t, a_t, use_target=False)  # [C]
                    r_unc = torch.clamp(q_std_step, max=self.cap_q)

                    # Running scale (EMA) normalization
                    r_task_norm = self._ema_normalize(r_task, "task")
                    r_cur_norm = self._ema_normalize(r_cur, "cur")
                    r_rare_norm = self._ema_normalize(r_rare, "rare")
                    r_unc_norm = self._ema_normalize(r_unc, "q")

                    # Weighted step bonus components
                    w_task_t = self.lambda_task * r_task_norm
                    w_cur_t = self.lambda_cur * r_cur_norm
                    w_rare_t = self.lambda_rare * r_rare_norm
                    w_unc_t = self.lambda_q * r_unc_norm

                    # Tri-objective + uncertainty composite step reward
                    r_step = w_task_t + w_cur_t + w_rare_t + w_unc_t
                    discount = self.gamma ** t

                    cumulative_reward = cumulative_reward + discount * r_step
                    cum_task_reward = cum_task_reward + discount * r_task
                    cum_curiosity_bonus = cum_curiosity_bonus + discount * r_cur
                    cum_rarity_bonus = cum_rarity_bonus + discount * r_rare
                    cum_unc_bonus = cum_unc_bonus + discount * r_unc

                    cum_task_weighted = cum_task_weighted + discount * w_task_t
                    cum_cur_weighted = cum_cur_weighted + discount * w_cur_t
                    cum_rare_weighted = cum_rare_weighted + discount * w_rare_t
                    cum_unc_weighted = cum_unc_weighted + discount * w_unc_t

                    # Belief progression & dynamics transition
                    m_t = torch.where(p_events > 0.5, torch.tensor(1.0, device=self.device), m_t)
                    z_t, h_t, s_t = self.model.predict_next_step(z_t, h_t, a_t)

                # 5. Terminal Clipped Double-Q evaluation at horizon H (no penalty)
                a_terminal = self.model.sample_prior_action(s_t, m_t, temperature=1.0)
                q_term = self.model.q_min(s_t, a_terminal, use_target=True)
                terminal_value = (self.gamma ** self.horizon) * q_term

                scores = cumulative_reward + terminal_value

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

                w_task_best = float(cum_task_weighted[best_idx].item())
                w_cur_best = float(cum_cur_weighted[best_idx].item())
                w_rare_best = float(cum_rare_weighted[best_idx].item())
                w_unc_best = float(cum_unc_weighted[best_idx].item())
                w_total = abs(w_task_best) + abs(w_cur_best) + abs(w_rare_best) + abs(w_unc_best)
                denom = max(1e-8, w_total)

                best_debug = {
                    "total_score": float(scores[best_idx].item()),
                    "cum_total_reward": float(cumulative_reward[best_idx].item()),
                    "task_reward_sum": float(cum_task_reward[best_idx].item()),
                    "curiosity_bonus_sum": float(cum_curiosity_bonus[best_idx].item()),
                    "rarity_bonus_sum": float(cum_rarity_bonus[best_idx].item()),
                    "unc_bonus_sum": float(cum_unc_bonus[best_idx].item()),
                    "weighted_task_sum": w_task_best,
                    "weighted_cur_sum": w_cur_best,
                    "weighted_rare_sum": w_rare_best,
                    "weighted_unc_sum": w_unc_best,
                    "weighted_task_share": float(abs(w_task_best) / denom),
                    "weighted_cur_share": float(abs(w_cur_best) / denom),
                    "weighted_rare_share": float(abs(w_rare_best) / denom),
                    "weighted_unc_share": float(abs(w_unc_best) / denom),
                    "terminal_value": float(terminal_value[best_idx].item()),
                    "ema_task": float(self.ema_task) if self.ema_task is not None else 1.0,
                    "ema_cur": float(self.ema_cur) if self.ema_cur is not None else 1.0,
                    "ema_rare": float(self.ema_rare) if self.ema_rare is not None else 1.0,
                    "ema_q": float(self.ema_q) if self.ema_q is not None else 1.0,
                }

        return best_action, best_debug

