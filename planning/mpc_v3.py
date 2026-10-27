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
from eval.metrics import CRAFTER_ACHIEVEMENTS


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
        use_landmark_guidance: bool = True,
        use_policy_prior: bool = True,
        strict_inference: bool = True,
        use_transition_model: bool = False,
        transition_scoring: bool = False,
        transition_weight: float = 0.5,
        transition_done_penalty: float = 2.0,
        transition_resource_weight: float = 0.25,
        compositional_planner: Optional[Any] = None,
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
        self.strict_inference = strict_inference
        self.use_transition_model = use_transition_model
        self.transition_scoring = transition_scoring
        self.transition_weight = transition_weight
        self.transition_done_penalty = transition_done_penalty
        self.transition_resource_weight = transition_resource_weight
        self.compositional_planner = compositional_planner
        if self.strict_inference and semantic_memory is not None:
            raise RuntimeError(
                "Crafter semantic memory with hardcoded subgoals/recipes cannot be attached in strict inference mode. "
                "Pass semantic_memory=None or set strict_inference=False."
            )
        self.semantic_memory = semantic_memory
        self.memory_sim_threshold = memory_sim_threshold
        self.memory_inject_ratio = memory_inject_ratio
        self.use_landmark_guidance = use_landmark_guidance
        self.use_policy_prior = use_policy_prior
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Adaptive Frontier Event Thresholds
        if event_thresholds is not None:
            if isinstance(event_thresholds, (int, float)):
                t = torch.full((1, 22), float(event_thresholds), device=self.device)
            else:
                t = event_thresholds.to(self.device).float().reshape(1, 22)
            self.event_thresholds = t
        else:
            if self.strict_inference:
                # Uniform learned/calibrated threshold across all achievements in strict mode
                t = torch.full((1, 22), frontier_threshold, device=self.device)
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
            z_0 = self.model.encode(obs_t, use_target=True)  # [1, 128] — target encoder: matches event/health-head training, belief updates, and consolidation keys (plan Stage 1 encoder unification)
            h_0 = belief.get_h_tensor(device=self.device).unsqueeze(0)  # [1, 256]
            m_bin_0 = belief.get_mask_tensor(device=self.device).unsqueeze(0)  # [1, 22]
            m_soft_0 = (
                belief.get_soft_mask_tensor(device=self.device).unsqueeze(0)
                if hasattr(belief, "get_soft_mask_tensor")
                else m_bin_0.clone()
            )  # [1, 22]
            s_0 = torch.cat([z_0, h_0], dim=-1)  # [1, 384]

            C = self.num_candidates
            N_prior = int(round(C * self.prior_mix)) if self.use_policy_prior else 0
            N_cem = C - N_prior

            # Query Long-Term Semantic Memory (Experience Diary) for learned affordances
            retrieved_action = None
            match_sim = 0.0
            target_milestone = None
            q_memory_all = None
            sim_memory_all = None
            active_subgoal = None
            if self.strict_inference:
                inv_context = None
            else:
                inv_context = belief.get_inv_vector() if hasattr(belief, "get_inv_vector") else belief.get_binary_mask()
            if self.semantic_memory is not None and len(self.semantic_memory) > 0:
                retrieved_action, match_sim, target_milestone = self.semantic_memory.query(
                    z_t=z_0,
                    active_inventory=inv_context,
                    sim_threshold=self.memory_sim_threshold,
                )
                if hasattr(self.semantic_memory, "query_all_actions"):
                    q_memory_all, sim_memory_all = self.semantic_memory.query_all_actions(
                        z_t=z_0,
                        active_inventory=inv_context,
                    )
                if hasattr(self.semantic_memory, "deduce_active_subgoal"):
                    achieved_set = set()
                    if hasattr(belief, "working_memory") and hasattr(belief.working_memory, "active_inventory"):
                        for ach_i, ach_n in enumerate(CRAFTER_ACHIEVEMENTS):
                            if belief.working_memory.active_inventory[ach_i] > 0:
                                achieved_set.add(ach_n)
                    active_subgoal = self.semantic_memory.deduce_active_subgoal(
                        current_inv=inv_context,
                        achieved_names=achieved_set,
                    )
            elif self.compositional_planner is not None:
                x_hat_root = None
                if hasattr(self.model, "predict_state"):
                    x_hat_root, _ = self.model.predict_state(s_0)
                subgoal_info = self.compositional_planner.plan_subgoal(
                    obs_latent=z_0,
                    grounded_state=x_hat_root if x_hat_root is not None else torch.zeros(16, device=self.device),
                )
                if subgoal_info is not None:
                    active_subgoal = {
                        "status": subgoal_info["status"],
                        "producer_action": subgoal_info.get("producer_action", subgoal_info["recommended_action"]),
                        "producer_key": subgoal_info.get("visual_key", z_0),
                        "producer_return": subgoal_info.get("expected_return", 1.0),
                        "target_action": subgoal_info["recommended_action"],
                        "target_key": subgoal_info.get("visual_key", z_0),
                        "target_name": subgoal_info.get("prototype_id", "phase2_subgoal"),
                    }
                    if subgoal_info.get("visual_sim", 0.0) >= 0.35:
                        retrieved_action = subgoal_info["recommended_action"]
                        match_sim = float(subgoal_info.get("visual_sim", 0.5))

            # Landmark anchor guidance (Working Memory: keep agent oriented near placed structure until breakthrough)
            structure_anchor = None
            structure_offset = None
            if hasattr(belief, "working_memory") and belief.working_memory.has_anchor("structure"):
                anc = belief.working_memory.get_anchor("structure")
                if anc is not None:
                    structure_anchor = anc.to(self.device).flatten().unsqueeze(0)
                if hasattr(belief.working_memory, "get_anchor_offset"):
                    structure_offset = belief.working_memory.get_anchor_offset("structure")

            # Passive visual landmark spotting for stone deposits (Bug 3 Fix)
            if (
                hasattr(belief, "working_memory")
                and not belief.working_memory.has_anchor("stone")
                and self.semantic_memory is not None
                and hasattr(self.semantic_memory, "get_prototype_key")
            ):
                stone_proto = self.semantic_memory.get_prototype_key("collect_stone")
                if stone_proto is not None:
                    sim_stone = float(F.cosine_similarity(z_0.flatten().unsqueeze(0), stone_proto.to(self.device).flatten().unsqueeze(0)).item())
                    if sim_stone >= 0.65:
                        belief.working_memory.set_anchor("stone", z_0.flatten().cpu())

            # Stone landmark anchor guidance (Working Memory: return to spotted rock once pickaxe is crafted)
            stone_anchor = None
            stone_offset = None
            if hasattr(belief, "working_memory") and belief.working_memory.has_anchor("stone"):
                anc_s = belief.working_memory.get_anchor("stone")
                if anc_s is not None:
                    stone_anchor = anc_s.to(self.device).flatten().unsqueeze(0)
                if hasattr(belief.working_memory, "get_anchor_offset"):
                    stone_offset = belief.working_memory.get_anchor_offset("stone")

            # Backpack inventory checks for landmark gating (0: wood, 6: wood_pickaxe)
            has_wood = False
            has_pickaxe = False
            if not self.strict_inference and inv_context is not None:
                if torch.is_tensor(inv_context):
                    inv_cpu = inv_context.detach().cpu().flatten()
                else:
                    inv_cpu = torch.from_numpy(np.array(inv_context, dtype=np.float32).flatten())
                if len(inv_cpu) >= 1:
                    has_wood = bool(inv_cpu[0] >= 1.0)
                if len(inv_cpu) >= 7:
                    has_pickaxe = bool(inv_cpu[6] >= 1.0)

            # Action displacement mapping for dead-reckoning:
            # 1: Left (-1, 0), 2: Right (1, 0), 3: Up (0, -1), 4: Down (0, 1)
            action_deltas_tensor = torch.zeros((self.num_actions, 2), dtype=torch.float32, device=self.device)
            action_deltas_tensor[1] = torch.tensor([-1.0, 0.0], device=self.device)
            action_deltas_tensor[2] = torch.tensor([1.0, 0.0], device=self.device)
            action_deltas_tensor[3] = torch.tensor([0.0, -1.0], device=self.device)
            action_deltas_tensor[4] = torch.tensor([0.0, 1.0], device=self.device)

            # Determine action injection: direct affordance or backward subgoal intention
            inject_action = retrieved_action
            if inject_action is None and active_subgoal is not None:
                if active_subgoal["status"] == "HARVESTING_INPUT":
                    prod_key = active_subgoal["producer_key"].to(self.device)
                    cue_sim = float(F.cosine_similarity(z_0.flatten().unsqueeze(0), prod_key.flatten().unsqueeze(0)).item())
                    if cue_sim >= 0.35:
                        inject_action = active_subgoal["producer_action"]
                elif active_subgoal["status"] == "READY_TO_EXECUTE":
                    tgt_key = active_subgoal["target_key"].to(self.device)
                    cue_sim = float(F.cosine_similarity(z_0.flatten().unsqueeze(0), tgt_key.flatten().unsqueeze(0)).item())
                    struct_sim_curr = 0.0
                    if self.use_landmark_guidance and structure_anchor is not None:
                        struct_sim_curr = float(F.cosine_similarity(z_0.flatten().unsqueeze(0), structure_anchor.flatten().unsqueeze(0)).item())
                    if cue_sim >= 0.35 or struct_sim_curr >= 0.50:
                        inject_action = active_subgoal["target_action"]

            # Initialize categorical CEM action distribution [H, 17]
            action_dist = torch.ones(self.horizon, self.num_actions, device=self.device) / self.num_actions

            best_action = 0
            best_debug: Dict[str, float] = {}

            for iteration in range(self.cem_iterations):
                # 2. Sample N_prior candidates autoregressively using PolicyPrior with action persistence
                prior_actions = torch.empty((N_prior, self.horizon), dtype=torch.long, device=self.device)
                z_roll = z_0.expand(N_prior, -1)
                h_roll = h_0.expand(N_prior, -1)
                s_roll = s_0.expand(N_prior, -1)
                m_bin_roll = m_bin_0.expand(N_prior, -1).clone()

                for h_step in range(self.horizon):
                    a_h = self.model.sample_prior_action(s_roll, m_bin_roll, temperature=self.prior_temperature)
                    if self.action_persistence > 0.0 and h_step > 0:
                        repeat_mask = torch.rand(N_prior, device=self.device) < self.action_persistence
                        a_h = torch.where(repeat_mask, prior_actions[:, h_step - 1], a_h)

                    # Inject retrieved affordance action at step 0 during initial iteration
                    if h_step == 0 and inject_action is not None and iteration == 0:
                        n_inject = max(1, int(N_prior * self.memory_inject_ratio))
                        a_h[:n_inject] = inject_action

                    prior_actions[:, h_step] = a_h
                    p_h = self.model.event_head.predict_proba(s_roll, a_h, m_bin_roll, use_calibration=self.use_calibration)
                    m_bin_roll = torch.where(p_h > self.event_thresholds, torch.tensor(1.0, device=self.device), m_bin_roll)
                    z_roll, h_roll, s_roll = self.model.predict_next_step(z_roll, h_roll, a_h)

                # 3. Sample N_cem candidates from CEM categorical distribution with action persistence
                cem_actions = torch.empty((N_cem, self.horizon), dtype=torch.long, device=self.device)
                for h_step in range(self.horizon):
                    probs_h = action_dist[h_step]
                    fresh_actions = torch.multinomial(probs_h, num_samples=N_cem, replacement=True)
                    if self.action_persistence > 0.0 and h_step > 0:
                        repeat_mask = torch.rand(N_cem, device=self.device) < self.action_persistence
                        cem_actions[:, h_step] = torch.where(repeat_mask, cem_actions[:, h_step - 1], fresh_actions)
                    else:
                        cem_actions[:, h_step] = fresh_actions

                    # Inject retrieved affordance into CEM candidates at step 0 during initial iteration
                    if h_step == 0 and inject_action is not None and iteration == 0:
                        n_inject_cem = max(1, int(N_cem * self.memory_inject_ratio))
                        cem_actions[:n_inject_cem, 0] = inject_action

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

                pos_struct_t = None
                if structure_offset is not None:
                    pos_struct_t = torch.tensor(structure_offset, dtype=torch.float32, device=self.device).unsqueeze(0).expand(C, -1).clone()

                pos_stone_t = None
                if stone_offset is not None:
                    pos_stone_t = torch.tensor(stone_offset, dtype=torch.float32, device=self.device).unsqueeze(0).expand(C, -1).clone()

                # Transition-aware state tracking per imagined candidate branch
                x_hat_0 = None
                x_hat_t = None
                transition_preds_history = []
                if self.use_transition_model and hasattr(self.model, "predict_state"):
                    x_hat_0, _ = self.model.predict_state(s_0)
                    # Each candidate maintains its own independent predicted state:
                    x_hat_t = x_hat_0.expand(C, -1).clone()

                for t in range(self.horizon):
                    a_t = actions_seq[:, t]
                    delta_move = action_deltas_tensor[a_t]
                    if pos_struct_t is not None:
                        pos_struct_t = pos_struct_t + delta_move
                    if pos_stone_t is not None:
                        pos_stone_t = pos_stone_t + delta_move

                    # Transition head imagined consequence prediction (Shadow mode: per-candidate state tracking)
                    if self.use_transition_model and hasattr(self.model, "predict_transition"):
                        trans = self.model.predict_transition(s_t, a_t)
                        delta_hat = trans["state_delta"]
                        reward_hat = trans["reward"]
                        done_prob = trans["done_prob"]
                        uncertainty = trans["uncertainty"]

                        if not torch.isfinite(delta_hat).all():
                            raise FloatingPointError("Predicted state delta contains NaN or Inf")
                        if not torch.isfinite(reward_hat).all():
                            raise FloatingPointError("Predicted transition reward contains NaN or Inf")
                        if ((done_prob < 0.0) | (done_prob > 1.0)).any():
                            raise ValueError("Predicted done probability outside [0, 1]")
                        if (uncertainty < 0.0).any():
                            raise ValueError("Predicted transition uncertainty must be non-negative")

                        if x_hat_t is not None:
                            x_hat_t = x_hat_t + delta_hat
                            if not torch.isfinite(x_hat_t).all():
                                raise FloatingPointError("Imagined candidate state contains NaN or Inf")

                        if iteration == self.cem_iterations - 1:
                            transition_preds_history.append({
                                "step": t,
                                "delta_hat": delta_hat,
                                "reward_hat": reward_hat,
                                "done_prob": done_prob,
                                "uncertainty": uncertainty,
                            })

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

                    # Scaled vital reward:
                    # Daylight hunger/thirst (-0.05 to -0.10) carries standard vital cost
                    # Acute lethal combat damage (health_delta <= -0.12, e.g. zombie bite / arrow) penalizes paths toward monsters
                    lethal_combat_damage = torch.clamp(health_delta + 0.12, max=0.0)
                    vital_reward = self.vital_weight * health_delta + 1.5 * lethal_combat_damage

                    # Affordance guidance bonus from Neural Episodic Memory (Pillar 4):
                    affordance_reward = 0.0
                    if q_memory_all is not None and t == 0:
                        q_a = q_memory_all[a_t]  # [C]
                        sim_a = sim_memory_all[a_t]  # [C]
                        # Reward actions with high historical return in similar visual/backpack states
                        # Fruitless/failed actions (q_a == 0.0) receive 0.0
                        affordance_reward = 0.25 * torch.clamp(q_a, min=0.0, max=2.5) * sim_a
                    elif retrieved_action is not None and t == 0:
                        affordance_reward = (a_t == retrieved_action).float() * (0.35 * match_sim)

                    # Sub-Goal Affordance Pull (Autonomous Backward Goal Chaining):
                    subgoal_reward = 0.0
                    if active_subgoal is not None:
                        if active_subgoal["status"] == "HARVESTING_INPUT":
                            prod_act = active_subgoal["producer_action"]
                            prod_key = active_subgoal["producer_key"].to(self.device).flatten().unsqueeze(0)
                            sim_cue = torch.clamp(F.cosine_similarity(z_t, prod_key, dim=-1), min=0.0)
                            # Pull agent towards producer visual cues while executing producer action
                            action_reward = (a_t == prod_act).float() * (0.25 * sim_cue * min(active_subgoal["producer_return"], 2.0))
                            # Continuous visual navigation gradient pulling candidate trajectories toward harvest cues (trees/water)
                            nav_reward = 0.10 * sim_cue
                            subgoal_reward = action_reward + nav_reward
                        elif active_subgoal["status"] == "READY_TO_EXECUTE":
                            tgt_act = active_subgoal["target_action"]
                            tgt_key = active_subgoal["target_key"].to(self.device).flatten().unsqueeze(0)
                            sim_cue = torch.clamp(F.cosine_similarity(z_t, tgt_key, dim=-1), min=0.0)
                            # Multi-step crafting affordance: rewards crafting at any point in the lookahead horizon
                            subgoal_reward = (a_t == tgt_act).float() * (0.35 * sim_cue)

                    # Landmark anchor guidance (Preserved in privileged mode; enabled in strict mode via learned state readout)
                    landmark_reward = 0.0
                    if not self.strict_inference:
                        if self.use_landmark_guidance and structure_anchor is not None:
                            struct_sim = F.cosine_similarity(z_t, structure_anchor, dim=-1)
                            if has_wood:
                                # We have wood (>= 1)! Pull HARD back to the table to craft pickaxe/sword
                                # Visual alignment pull when table is on-screen
                                vis_pull = 0.30 * torch.clamp(struct_sim, min=0.0)
                                # Dead-reckoning spatial pull when table is off-screen
                                offset_pull = 0.0
                                if pos_struct_t is not None:
                                    dist_to_table = torch.norm(pos_struct_t, dim=-1)
                                    offset_pull = -0.10 * dist_to_table
                                landmark_reward = vis_pull + offset_pull
                            else:
                                # We have 0 wood! Do NOT glue agent to the table with +1.5 reward!
                                # Push slightly away so the agent explores to harvest trees
                                landmark_reward = -0.05 * torch.clamp(struct_sim - 0.85, min=0.0)

                        # Stone landmark pull (Bug 3 Fix: Return to spotted rock once pickaxe is in hand)
                        if self.use_landmark_guidance and stone_anchor is not None and has_pickaxe:
                            stone_sim = F.cosine_similarity(z_t, stone_anchor, dim=-1)
                            vis_pull = 0.30 * torch.clamp(stone_sim, min=0.0)
                            offset_pull = 0.0
                            if pos_stone_t is not None:
                                dist_to_stone = torch.norm(pos_stone_t, dim=-1)
                                offset_pull = -0.10 * dist_to_stone
                            landmark_reward = landmark_reward + vis_pull + offset_pull
                    elif self.strict_inference and self.use_transition_model and self.transition_scoring and x_hat_t is not None:
                        # Strict-mode landmark guidance using strictly learned state readout & imagined dynamics
                        # Zero privileged environment inventory inspection!
                        if self.use_landmark_guidance and structure_anchor is not None:
                            struct_sim = F.cosine_similarity(z_t, structure_anchor, dim=-1)
                            cand_has_wood = (x_hat_t[:, 0] >= 0.5)
                            vis_pull = 0.30 * torch.clamp(struct_sim, min=0.0)
                            offset_pull = 0.0
                            if pos_struct_t is not None:
                                dist_to_table = torch.norm(pos_struct_t, dim=-1)
                                offset_pull = -0.10 * dist_to_table
                            wood_landmark = torch.where(
                                cand_has_wood,
                                vis_pull + offset_pull,
                                -0.05 * torch.clamp(struct_sim - 0.85, min=0.0),
                            )
                            landmark_reward = landmark_reward + wood_landmark

                        # Stone landmark pull conditioned on strictly learned pickaxe prediction
                        if self.use_landmark_guidance and stone_anchor is not None:
                            cand_has_pickaxe = (x_hat_t[:, 6] >= 0.5)
                            stone_sim = F.cosine_similarity(z_t, stone_anchor, dim=-1)
                            vis_pull = 0.30 * torch.clamp(stone_sim, min=0.0)
                            offset_pull = 0.0
                            if pos_stone_t is not None:
                                dist_to_stone = torch.norm(pos_stone_t, dim=-1)
                                offset_pull = -0.10 * dist_to_stone
                            stone_landmark = torch.where(
                                cand_has_pickaxe,
                                vis_pull + offset_pull,
                                torch.zeros_like(vis_pull),
                            )
                            landmark_reward = landmark_reward + stone_landmark

                    # Transition head active consequence scoring (Ablation C / Gate 10)
                    consequence_reward = 0.0
                    if self.use_transition_model and self.transition_scoring and hasattr(self.model, "predict_transition"):
                        # 1. Model-predicted scalar reward (aligned with env reward)
                        r_trans = self.transition_weight * reward_hat.reshape(-1)
                        # 2. Death avoidance penalty: penalize imagined transitions leading to termination
                        done_pen = - self.transition_done_penalty * done_prob.reshape(-1)
                        # 3. Resource consequence progress: reward transitions predicting positive resource acquisition (wood, stone, tools)
                        res_prog = self.transition_resource_weight * torch.clamp(delta_hat, min=0.0).sum(dim=-1).reshape(-1)
                        consequence_reward = r_trans + done_pen + res_prog

                    step_reward = (
                        effective_ach_reward
                        + vital_reward
                        + affordance_reward
                        + subgoal_reward
                        + landmark_reward
                        + consequence_reward
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
                    "memory_best_q": float(q_memory_all.max().item()) if q_memory_all is not None else 0.0,
                    "active_subgoal_status": str(active_subgoal.get("status", "") if active_subgoal else ""),
                    "active_subgoal_target": str(active_subgoal.get("target_name", "") if active_subgoal else ""),
                    "structure_offset_x": float(structure_offset[0]) if structure_offset is not None else 0.0,
                    "structure_offset_y": float(structure_offset[1]) if structure_offset is not None else 0.0,
                    "stone_offset_x": float(stone_offset[0]) if stone_offset is not None else 0.0,
                    "stone_offset_y": float(stone_offset[1]) if stone_offset is not None else 0.0,
                }
                if self.use_transition_model:
                    best_debug.update({
                        "transition_model_used": True,
                        "transition_scoring_active": bool(self.transition_scoring),
                        "state_estimate": x_hat_0[0].detach().cpu().numpy().tolist() if x_hat_0 is not None else [],
                        "predicted_delta": transition_preds_history[0]["delta_hat"][best_idx].detach().cpu().numpy().tolist() if transition_preds_history else [],
                        "predicted_reward": float(transition_preds_history[0]["reward_hat"][best_idx].item()) if transition_preds_history else 0.0,
                        "predicted_done_probability": float(transition_preds_history[0]["done_prob"][best_idx].item()) if transition_preds_history else 0.0,
                        "predicted_uncertainty": float(transition_preds_history[0]["uncertainty"][best_idx].mean().item()) if transition_preds_history else 0.0,
                    })
                else:
                    best_debug["transition_model_used"] = False
                    best_debug["transition_scoring_active"] = False
                if self.standardize_scores:
                    best_debug.update({
                        "z_reward": float(z_reward[best_idx].item()),
                        "z_terminal": float(z_term[best_idx].item()),
                        "z_uncertainty": float(z_unc[best_idx].item()),
                        "z_novelty": float(z_nov[best_idx].item()),
                        "z_tte": float(z_tte[best_idx].item()),
                    })

        return best_action, best_debug
