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
        prior_temperature: float = 1.0,
        action_persistence: float = 0.5,
        num_actions: int = 17,
        use_calibration: bool = True,
        denormalize_q: bool = False,
        tte_weight: float = 0.0,
        standardize_scores: bool = False,
        frontier_threshold: float = 0.25,
        common_threshold: float = 0.50,
        event_thresholds: Optional[Union[float, torch.Tensor]] = None,
        vital_weight: float = 0.5,
        death_risk_factor: float = 0.0,
        harvest_bonus: float = 0.15,
        semantic_memory: Optional[Any] = None,
        memory_sim_threshold: float = 0.58,
        memory_inject_ratio: float = 0.50,
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
        self.prior_temperature = prior_temperature
        self.action_persistence = action_persistence
        self.num_actions = num_actions
        self.use_calibration = use_calibration
        self.denormalize_q = denormalize_q
        self.tte_weight = tte_weight
        self.standardize_scores = standardize_scores
        self.vital_weight = vital_weight
        self.death_risk_factor = death_risk_factor
        self.harvest_bonus = harvest_bonus
        self.semantic_memory = semantic_memory
        self.memory_sim_threshold = memory_sim_threshold
        self.memory_inject_ratio = memory_inject_ratio
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Adaptive Frontier Event Thresholds
        # Common survival achievements (sapling, wood, drink, cow, plant, sleep) use common_threshold (0.50)
        # Sparse frontier achievements (table, pickaxe, sword, stone, furnace, iron, etc.) use frontier_threshold (0.25)
        if event_thresholds is not None:
            if isinstance(event_thresholds, (int, float)):
                t = torch.full((1, 22), float(event_thresholds), device=self.device)
            else:
                t = event_thresholds.to(self.device).float().reshape(1, 22)
            self.event_thresholds = t
        else:
            t = torch.full((1, 22), common_threshold, device=self.device)
            frontier_indices = [0, 1, 3, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20]
            t[0, frontier_indices] = frontier_threshold
            self.event_thresholds = t

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
            m_bin_0 = belief.get_mask_tensor(device=self.device).unsqueeze(0)  # [1, 22]
            m_soft_0 = (
                belief.get_soft_mask_tensor(device=self.device).unsqueeze(0)
                if hasattr(belief, "get_soft_mask_tensor")
                else m_bin_0.clone()
            )  # [1, 22]
            s_0 = torch.cat([z_0, h_0], dim=-1)  # [1, 384]
            # Environmental luminance to condition night survival dynamics (Crafter night is < 0.35)
            luminance = obs_t.mean()
            night_factor = torch.clamp((0.35 - luminance) / 0.15, min=0.0, max=1.0)

            C = self.num_candidates
            N_prior = int(round(C * self.prior_mix))
            N_cem = C - N_prior

            # Query Long-Term Semantic Memory for learned affordances
            retrieved_action = None
            match_sim = 0.0
            target_milestone = None
            if self.semantic_memory is not None and len(self.semantic_memory) > 0:
                retrieved_action, match_sim, target_milestone = self.semantic_memory.query(
                    z_t=z_0,
                    active_inventory=belief.get_binary_mask(),
                    sim_threshold=self.memory_sim_threshold,
                )

            # Landmark anchor guidance (Working Memory: keep agent oriented near table if tools needed)
            table_anchor = None
            if hasattr(belief, "working_memory") and belief.working_memory.has_anchor("place_table"):
                if m_bin_0[0, 15] < 0.5 or m_bin_0[0, 16] < 0.5:
                    anc = belief.working_memory.get_anchor("place_table")
                    if anc is not None:
                        table_anchor = anc.to(self.device).flatten().unsqueeze(0)

            # Initialize categorical CEM action distribution [H, 17]
            action_dist = torch.ones(self.horizon, self.num_actions, device=self.device) / self.num_actions

            best_action = 0
            best_debug: Dict[str, float] = {}

            for iteration in range(self.cem_iterations):
                # 2. Sample N_prior candidates autoregressively using PolicyPrior with directional persistence
                prior_actions = torch.empty((N_prior, self.horizon), dtype=torch.long, device=self.device)
                z_roll = z_0.expand(N_prior, -1)
                h_roll = h_0.expand(N_prior, -1)
                s_roll = s_0.expand(N_prior, -1)
                m_bin_roll = m_bin_0.expand(N_prior, -1).clone()

                for h_step in range(self.horizon):
                    a_h = self.model.sample_prior_action(s_roll, m_bin_roll, temperature=self.prior_temperature)
                    if self.action_persistence > 0.0 and h_step > 0:
                        repeat_mask = torch.rand(N_prior, device=self.device) < self.action_persistence
                        prev_actions = prior_actions[:, h_step - 1]
                        move_mask = (prev_actions >= 1) & (prev_actions <= 4)
                        a_h = torch.where(repeat_mask & move_mask, prev_actions, a_h)

                    # Inject retrieved affordance action at step 0 during initial iteration
                    if h_step == 0 and retrieved_action is not None and iteration == 0:
                        n_inject = max(1, int(N_prior * self.memory_inject_ratio))
                        a_h[:n_inject] = retrieved_action

                    prior_actions[:, h_step] = a_h
                    p_h = self.model.event_head.predict_proba(s_roll, a_h, m_bin_roll, use_calibration=self.use_calibration)
                    m_bin_roll = torch.where(p_h > self.event_thresholds, torch.tensor(1.0, device=self.device), m_bin_roll)
                    z_roll, h_roll, s_roll = self.model.predict_next_step(z_roll, h_roll, a_h)

                # 3. Sample N_cem candidates from CEM categorical distribution with directional movement persistence
                cem_actions = torch.empty((N_cem, self.horizon), dtype=torch.long, device=self.device)
                for h_step in range(self.horizon):
                    probs_h = action_dist[h_step]
                    fresh_actions = torch.multinomial(probs_h, num_samples=N_cem, replacement=True)
                    if self.action_persistence > 0.0 and h_step > 0:
                        repeat_mask = torch.rand(N_cem, device=self.device) < self.action_persistence
                        prev_actions = cem_actions[:, h_step - 1]
                        move_mask = (prev_actions >= 1) & (prev_actions <= 4)
                        cem_actions[:, h_step] = torch.where(repeat_mask & move_mask, prev_actions, fresh_actions)
                    else:
                        cem_actions[:, h_step] = fresh_actions

                    # Inject retrieved affordance into CEM candidates at step 0 during initial iteration
                    if h_step == 0 and retrieved_action is not None and iteration == 0:
                        n_inject_cem = max(1, int(N_cem * self.memory_inject_ratio))
                        cem_actions[:n_inject_cem, 0] = retrieved_action

                # Combine candidates: [C, horizon]
                actions_seq = torch.cat([prior_actions, cem_actions], dim=0)

                # 4. Rollout and score all C candidates through recurrent dynamics
                z_t = z_0.expand(C, -1)
                h_t = h_0.expand(C, -1)
                s_t = s_0.expand(C, -1)
                m_bin_t = m_bin_0.expand(C, -1).clone()
                m_soft_t = m_soft_0.expand(C, -1).clone()
                cumulative_reward = torch.zeros(C, device=self.device)
                gamma_prod = torch.ones(C, device=self.device)

                for t in range(self.horizon):
                    a_t = actions_seq[:, t]
                    r_hat, decomp = self.model.predict_reward(
                        s_t,
                        a_t,
                        m_bin_t,
                        m_soft=m_soft_t,
                        return_decomposition=True,
                        use_calibration=self.use_calibration,
                    )
                    p_t = decomp["event_probs"]
                    health_delta = decomp["health_reward"]

                    # Compute newly unlocked achievements in imagination using adaptive thresholds
                    newly_unlocked = (p_t > self.event_thresholds) & (m_bin_t < 0.5)
                    discrete_ach_reward = newly_unlocked.float().sum(dim=-1)
                    soft_ach_reward = torch.clamp(p_t - m_soft_t.float(), min=0.0).sum(dim=-1)
                    effective_ach_reward = torch.maximum(discrete_ach_reward, soft_ach_reward)

                    # Repeat harvest utility: reward harvesting wood even if unlocked, enabling table & tool crafting
                    harvest_wood = (p_t[:, 6] > self.event_thresholds[0, 6]) & (m_bin_t[:, 6] >= 0.5)
                    repeat_harvest_reward = harvest_wood.float() * self.harvest_bonus

                    # Scaled vital reward:
                    # Daylight hunger/thirst (-0.05 to -0.10) carries standard vital cost
                    # Acute lethal combat damage (health_delta <= -0.12, e.g. zombie bite / arrow) penalizes paths toward monsters
                    lethal_combat_damage = torch.clamp(health_delta + 0.12, max=0.0)
                    vital_reward = self.vital_weight * health_delta + 1.5 * lethal_combat_damage

                    # Night survival bonus: in darkness, reward health restoration (e.g. resting/sleeping through night)
                    night_rest_reward = night_factor * torch.clamp(health_delta, min=0.0) * 0.5

                    # Affordance guidance bonus: reward executing the retrieved milestone action
                    affordance_reward = 0.0
                    if retrieved_action is not None and t == 0:
                        affordance_reward = (a_t == retrieved_action).float() * (0.35 * match_sim)

                    # Landmark anchor persistence bonus: prevent wandering away from table before tool craft
                    landmark_reward = 0.0
                    if table_anchor is not None:
                        table_sim = F.cosine_similarity(z_t, table_anchor, dim=-1)
                        landmark_reward = 0.15 * torch.clamp(table_sim, min=0.0)

                    step_reward = (
                        effective_ach_reward
                        + repeat_harvest_reward
                        + vital_reward
                        + night_rest_reward
                        + affordance_reward
                        + landmark_reward
                    )
                    cumulative_reward = cumulative_reward + gamma_prod * step_reward

                    # Continuation discounting: state-dependent survival in imagination
                    # Decay continuation probability ONLY on severe lethal combat damage, avoiding daylight hunger panic
                    p_continue = torch.clamp(1.0 + 2.0 * lethal_combat_damage, min=0.0, max=1.0)
                    gamma_prod = gamma_prod * (self.gamma * p_continue)

                    # Simulated belief progression:
                    # Soft mask updated by running maximum of predicted probability (used for reward)
                    m_soft_t = torch.maximum(m_soft_t, p_t)
                    # Binary mask updated with adaptive frontier thresholds (for model input)
                    m_bin_t = torch.where(p_t > self.event_thresholds, torch.tensor(1.0, device=self.device), m_bin_t)
                    z_t, h_t, s_t = self.model.predict_next_step(z_t, h_t, a_t)

                # 5. Terminal Clipped Double-Q evaluation at horizon H discounted by cumulative continuation
                a_terminal = self.model.sample_prior_action(s_t, m_bin_t, temperature=self.prior_temperature)
                q_term = self.model.q_min(s_t, a_terminal, use_target=True)
                q_std_term = self.model.q_std(s_t, a_terminal, use_target=True)

                if self.denormalize_q and hasattr(self.model, "return_normalizer") and self.model.return_normalizer.initialized:
                    q_term = self.model.return_normalizer.denormalize(q_term)
                    norm = self.model.return_normalizer
                    scale = max(norm.high - norm.low, 1.0)
                    q_std_term = q_std_term / (norm.target_max * 0.8) * scale

                terminal_value = gamma_prod * q_term
                raw_uncertainty = q_std_term
                raw_novelty = torch.norm(z_t - z_0, dim=-1)

                raw_tte = torch.zeros_like(terminal_value)
                if hasattr(self.model, "time_to_event_head") and self.model.time_to_event_head is not None:
                    tau_0 = self.model.time_to_event_head(s_0)  # [1, 22]
                    tau_H = self.model.time_to_event_head(s_t)  # [C, 22]
                    unachieved_mask = (1.0 - m_soft_0).expand(C, -1)
                    prog = ((tau_H - tau_0) * unachieved_mask).clamp(min=0.0).sum(dim=-1)
                    raw_tte = prog

                if self.standardize_scores:
                    def _standardize(term: torch.Tensor) -> torch.Tensor:
                        std = term.std()
                        if std < 1e-8:
                            return torch.zeros_like(term)
                        return (term - term.mean()) / std

                    z_reward = _standardize(cumulative_reward)
                    z_term = _standardize(terminal_value)
                    z_unc = _standardize(raw_uncertainty)
                    z_nov = _standardize(raw_novelty)
                    z_tte = _standardize(raw_tte)

                    scores = z_reward + z_term - z_unc + z_nov + z_tte
                else:
                    uncertainty_penalty = self.uncertainty_weight * raw_uncertainty
                    novelty = self.novelty_weight * raw_novelty
                    tte_gain = self.tte_weight * raw_tte
                    scores = cumulative_reward + terminal_value - uncertainty_penalty + novelty + tte_gain

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
                    "uncertainty_penalty": float(raw_uncertainty[best_idx].item() if self.standardize_scores else uncertainty_penalty[best_idx].item()),
                    "tte_gain": float(raw_tte[best_idx].item() if self.standardize_scores else tte_gain[best_idx].item()),
                    "best_score": float(scores[best_idx].item()),
                    "mean_q_std": float(q_std_term.mean().item()),
                    "memory_retrieved_action": float(retrieved_action if retrieved_action is not None else -1),
                    "memory_match_sim": float(match_sim),
                }
                if self.standardize_scores:
                    best_debug.update({
                        "z_reward": float(z_reward[best_idx].item()),
                        "z_terminal": float(z_term[best_idx].item()),
                        "z_uncertainty": float(z_unc[best_idx].item()),
                        "z_novelty": float(z_nov[best_idx].item()),
                        "z_tte": float(z_tte[best_idx].item()),
                    })

        return best_action, best_debug
