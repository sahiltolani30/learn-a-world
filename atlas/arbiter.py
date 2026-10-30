"""AtlasAgent: central decision-maker and mode arbiter.

Hierarchy:
1. Safety override (ContinueNet reactive survival)
2. PROBE mode (calibrated event-head affordance ranking at keyframe)
3. NAVIGATE mode (closed-loop OptionNet guidance toward goal)
4. FALLBACK / Explore (BC prior + EpistemicNet disagreement)

Strictness invariant: act() consumes ONLY visual observations and realized reward.
No engine inventory, coordinates, or achievement vectors enter the decision process.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from atlas.backbone import FrozenBackbone
from atlas.belief import AchievementBelief
from atlas.config import (
    CRAFTER_ACTION_NAMES,
    LATENT_DIM,
    NUM_ACHIEVEMENTS,
    NUM_ACTIONS,
    RECURRENT_DIM,
    STATE_DIM,
    AtlasConfig,
)
from atlas.epistemic import EpistemicNet, knn_novelty
from atlas.goals import GoalBank, GoalEntry, PredecessorLift
from atlas.placemap import TopoMap
from atlas.readout import VitalsReadout
from atlas.safety import ContinueNet
from atlas.skills import OptionNet, ReachabilityNet


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a / norm_a, b / norm_b))


# V2.2-F11-fix: NO hardcoded recipe table exists anywhere in ATLAS. Material requirements
# are LEARNED: the frozen StateDeltaHead predicts each action's inventory change (log1p scale);
# suppression fires only when the learned predicted cost exceeds the learned readout holdings.


def compute_material_suppression(deltas, have, margin: float = 0.2) -> "np.ndarray":
    """Learned material gating (V2.2-F11-fix). PURE function, no recipe knowledge.

    deltas: [17, 16] predicted inventory change per action (log1p scale) from the
            frozen StateDeltaHead. Negative = the action consumes.
    have:   [16] current holdings estimated by the learned readout (log1p scale).
    Returns bool [17]: True where any ITEM dim (0..11) has learned cost > have + margin.
    Item dims only: vitals (12..15) are ContinueNet's responsibility, never gated here.
    """
    costs = np.clip(-deltas, 0.0, None)          # consumption per action, learned
    deficit = costs[:, :12] > (have[None, :12] + margin)
    return deficit.any(axis=-1)


class AtlasAgent:
    """ATLAS autonomous agent."""

    def __init__(
        self,
        backbone: FrozenBackbone,
        belief: Optional[AchievementBelief] = None,
        topomap: Optional[TopoMap] = None,
        goals: Optional[GoalBank] = None,
        option_net: Optional[OptionNet] = None,
        reach_net: Optional[ReachabilityNet] = None,
        epistemic_net: Optional[EpistemicNet] = None,
        safety_net: Optional[ContinueNet] = None,
        readout_net: Optional[VitalsReadout] = None,
        critic_net: Optional[Any] = None,
        cfg: Optional[AtlasConfig] = None,
    ):
        self.cfg = cfg or AtlasConfig()
        self.backbone = backbone
        self.device = self.backbone.device

        self.belief = belief or AchievementBelief.from_config(self.cfg.belief)
        self.topomap = topomap or TopoMap(self.cfg.placemap)
        self.goals = goals or GoalBank(self.cfg.goals)
        self.lift = PredecessorLift()

        self.option_net = option_net or OptionNet(self.cfg.skills).to(self.device)
        self.reach_net = reach_net or ReachabilityNet(self.cfg.skills).to(self.device)
        self.epistemic_net = epistemic_net or EpistemicNet(self.cfg.epistemic).to(self.device)
        self.safety_net = safety_net or ContinueNet(self.cfg.safety).to(self.device)
        self.readout_net = readout_net or VitalsReadout(self.cfg.readout).to(self.device)
        self.critic_net = critic_net

        # Set evaluation mode for play-time inference
        self.option_net.eval()
        self.reach_net.eval()
        self.epistemic_net.eval()
        self.safety_net.eval()
        if self.readout_net is not None:
            self.readout_net.eval()
        if self.critic_net is not None:
            self.critic_net.eval()

        # Schedules
        self.epsilon: float = self.cfg.collect.epsilon_start
        self.beta_t: float = self.cfg.collect.beta_start
        self.current_iter: int = 0

        # Episode-local state
        self.h_prev: torch.Tensor = torch.zeros(1, RECURRENT_DIM, device=self.device)
        self.a_prev: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.last_event_probs: Optional[torch.Tensor] = None
        self.h_memory: Deque[np.ndarray] = deque(maxlen=self.cfg.epistemic.knn_capacity)

        # Mode machine
        self.mode: str = "FALLBACK"
        self.prev_mode: str = "FALLBACK"
        self.active_goal: Optional[GoalEntry] = None
        self.goal_score: float = 0.0
        self.probe_steps_remaining: int = 0
        self.wander_steps_remaining: int = 0
        self.probe_unlocked: bool = False
        self.frozen_memory: bool = False
        self.nav_steps: int = 0
        self.stagnation_counter: int = 0
        self.progress_best: float = -1.0
        self.current_option_budget: int = self.cfg.skills.max_option_steps
        self.current_goal_mode: str = "none"
        self.last_probe_decision: Optional[Dict[str, Any]] = None
        self.episode_artifact_views: Dict[int, np.ndarray] = {}

        # V3 F3 & F8: Quantity-aware subgoals and craft execution
        self.target_goal_pending: Optional[GoalEntry] = None
        self.material_subgoal_cycles: int = 0
        self.item_event_map: Optional[Dict[int, Optional[int]]] = None

        # V3 Instrumentation (C2 & F6 telemetry)
        self.probe_id: int = 0
        self.probe_newly_unlocked: List[int] = []
        self.last_abort_reason: str = "none"

        # In-progress episode transition buffer for collector
        self.episode_buffer: list[Dict[str, Any]] = []

    def reset_episode(self, seed: Optional[int] = None) -> None:
        """Resets agent state at episode boundaries."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.belief.reset()
        self.topomap.reset_episode()
        self.h_prev = torch.zeros(1, RECURRENT_DIM, device=self.device)
        self.a_prev = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.last_event_probs = None
        self.h_memory.clear()

        self.mode = "FALLBACK"
        self.prev_mode = "FALLBACK"
        self.active_goal = None
        self.goal_score = 0.0
        self.probe_steps_remaining = 0
        self.wander_steps_remaining = 0
        self.probe_unlocked = False
        self.nav_steps = 0
        self.stagnation_counter = 0
        self.progress_best = -1.0
        self.current_option_budget = self.cfg.skills.max_option_steps
        self.current_goal_mode = "none"
        self.last_probe_decision = None
        self.episode_artifact_views.clear()
        self.target_goal_pending = None
        self.material_subgoal_cycles = 0
        self.probe_newly_unlocked.clear()
        self.last_abort_reason = "none"
        self.episode_buffer.clear()

    @torch.no_grad()
    def act(
        self,
        obs_uint8: np.ndarray,
        reward_prev: float,
        step: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """Selects action given uint8 visual observation and previous realized reward.

        Enforces strictness: no info / inventory / cheat inputs allowed in signature.
        """
        # 1) Belief update from previous step's prediction and realized reward
        newly_unlocked = []
        if step > 0 and self.last_event_probs is not None:
            newly_unlocked = self.belief.update(self.last_event_probs, reward_prev)
            if self.prev_mode == "PROBE":
                if len(newly_unlocked) > 0:
                    self.probe_unlocked = True
                self.probe_newly_unlocked.extend(newly_unlocked)

        # 2) Encode observation and recurrent step
        z = self.backbone.encode(obs_uint8)  # [1, 128]
        if step == 0:
            h = torch.zeros(1, RECURRENT_DIM, device=self.device)
        else:
            h = self.backbone.recurrent_step(z, self.a_prev, self.h_prev)  # [1, 256]

        s = self.backbone.composite(z, h)  # [1, 384]

        z_np = z.squeeze(0).cpu().numpy()
        h_np = h.squeeze(0).cpu().numpy()
        s_np = s.squeeze(0).cpu().numpy()

        # Record tagged artifact views for newly unlocked achievements (V2.2 F11)
        if step > 0 and newly_unlocked:
            for dim in newly_unlocked:
                self.episode_artifact_views[dim] = z_np.copy()

        # Update TopoMap (gated by frozen_memory)
        if not self.frozen_memory:
            self.topomap.observe(z_np, h_np, step)
        self.h_memory.append(h_np)

        # 3) Mode machine & candidate action selection
        mask_t = self.belief.get_mask_tensor(self.device).unsqueeze(0)  # [1, 22]
        all_event_probs = self.backbone.event_logits_all_actions(s, mask_t).squeeze(0)  # [17, 22]
        arrival_cos = 0.0
        psi_val = 0.0

        # Helper functions for goal selection
        def _psi_fn(s_vec: np.ndarray, g_vec: np.ndarray) -> float:
            s_tensor = torch.from_numpy(s_vec).float().to(self.device).unsqueeze(0)
            g_tensor = torch.from_numpy(g_vec).float().to(self.device).unsqueeze(0)
            return float(self.reach_net(s_tensor, g_tensor).item())

        def _danger_fn(s_vec: np.ndarray, g_vec: np.ndarray) -> float:
            s_tensor = torch.from_numpy(s_vec).float().to(self.device).unsqueeze(0)
            risks = self.safety_net.risk_all_actions(s_tensor)
            return float(risks.mean().item())

        # Determine mode
        if self.probe_steps_remaining > 0:
            self.mode = "PROBE"
            self.probe_steps_remaining -= 1
        elif self.wander_steps_remaining > 0:
            self.mode = "WANDER"
            self.wander_steps_remaining -= 1
        elif self.active_goal is not None:
            # Ensure option budget is scaled for active goal (V2.2 F9)
            if getattr(self, "_active_goal_id", None) != id(self.active_goal):
                self._active_goal_id = id(self.active_goal)
                g_key_t = torch.from_numpy(self.active_goal.key).float().to(self.device).unsqueeze(0)
                psi_val = float(self.reach_net(s, g_key_t).item())
                base_budget = self.cfg.skills.max_option_steps
                scaled_budget = base_budget * (1.0 + 2.0 * max(0.0, 1.0 - psi_val))
                b_min = min(getattr(self.cfg.skills, "option_budget_min", 25), base_budget)
                b_max = max(getattr(self.cfg.skills, "option_budget_max", 60), base_budget)
                self.current_option_budget = int(min(b_max, max(b_min, round(scaled_budget))))

            g_key_np = self.active_goal.key
            arrival_cos = _cosine(z_np, g_key_np)
            if arrival_cos >= self.cfg.skills.arrival_cosine:
                # Arrived at goal! Transition to PROBE (V3 C2: reset probe attribution)
                self.mode = "PROBE"
                self.probe_steps_remaining = self.cfg.goals.probe_steps - 1
                self.probe_unlocked = False
                self.probe_id += 1
                self.probe_newly_unlocked.clear()
            else:
                g_key_t = torch.from_numpy(g_key_np).float().to(self.device).unsqueeze(0)
                psi_val = float(self.reach_net(s, g_key_t).item())
                # V3 F6: progress_t = max(psi(s_t, g), cos(z_t, g))
                progress_t = max(psi_val, arrival_cos)
                if progress_t >= self.progress_best + 0.01:
                    self.progress_best = progress_t
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

                self.nav_steps += 1
                stagnated = (
                    self.stagnation_counter >= self.cfg.skills.stagnation_patience
                    and arrival_cos < self.cfg.skills.arrival_cosine
                )
                timed_out = (self.nav_steps >= self.current_option_budget)
                if stagnated or timed_out:
                    self.last_abort_reason = "stagnation" if stagnated else "timeout"
                    if not self.frozen_memory:
                        self.goals.mark_used(g_key_np, success=False, current_iter=self.current_iter)
                    self.active_goal = None
                    self.mode = "WANDER"
                    self.wander_steps_remaining = self.cfg.skills.wander_steps - 1
                else:
                    self.mode = "NAVIGATE"
        else:
            self.mode = "FALLBACK"

        # If idle / fallback, try selecting a new goal from bank
        if self.mode not in ("PROBE", "WANDER") and self.active_goal is None:
            best_goal, score = self.goals.select(
                z_np,
                h_np,
                self.belief.m_soft,
                _psi_fn,
                _danger_fn,
                lift=self.lift,
                current_iter=self.current_iter,
                beta_t=self.beta_t,
                topomap=self.topomap,
                belief=self.belief,
                episode_artifact_views=self.episode_artifact_views,
                critic=self.critic_net,
                readout_net=self.readout_net,
            )

            # V3 F3 & F8: Quantity-aware subgoals and craft execution
            if (
                best_goal is not None
                and score > 0.0
                and getattr(self.cfg.goals, "material_gating_learned", True)
                and getattr(self.backbone, "has_learned_delta_head", False)
                and self.readout_net is not None
            ):
                target_dims = list(best_goal.event_dims)
                craft_predecessors = []
                for d in target_dims:
                    for p in range(NUM_ACHIEVEMENTS):
                        if p != d and self.lift.ci_low[p, d] > getattr(self.cfg.goals, "lift_ci_threshold", 1.2):
                            craft_predecessors.append((p, float(self.lift.ci_low[p, d]), d))

                if craft_predecessors:
                    if best_goal.action_profile is not None and np.max(best_goal.action_profile[:, target_dims]) > 0.0:
                        top_a = int(np.argmax(np.max(best_goal.action_profile[:, target_dims], axis=-1)))
                    elif target_dims:
                        top_a = int(torch.argmax(all_event_probs[:, target_dims].max(dim=-1).values).item())
                    else:
                        top_a = 0
                    with torch.no_grad():
                        a_cand = torch.tensor([top_a], device=self.device)
                        delta = self.backbone.inventory_delta(s, a_cand).squeeze(0).cpu().numpy()
                        mean_x, _ = self.readout_net(s)
                        have = mean_x.squeeze(0).detach().cpu().numpy()

                    cost = np.clip(-delta[:12], 0.0, None)
                    deficit_mask = cost > (have[:12] + self.cfg.goals.material_gate_margin)

                    if np.any(deficit_mask):
                        # V3 F3: Pick single largest deficit dim d* = argmax_i (cost[i] - have[i])
                        deficit_diffs = cost - have[:12]
                        d_star = int(np.argmax(deficit_diffs))
                        max_deficit = float(deficit_diffs[d_star])
                        max_c = getattr(self.cfg.goals, "max_cycles", 4)
                        cycles = int(min(max_c, max(1, np.ceil(max_deficit))))

                        # Find goal CLOUD = all bank entries whose event_dims contain ItemEventMap(d*)
                        pred_goals = []
                        if getattr(self, "item_event_map", None) is not None:
                            mapped_ev = self.item_event_map.get(d_star, None)
                            if mapped_ev is not None:
                                pred_goals = [g for g in self.goals.entries if mapped_ev in g.event_dims]
                        else:
                            # Fallback if ItemEventMap not yet learned: predecessors from lift graph
                            craft_predecessors.sort(key=lambda item: item[1], reverse=True)
                            for p, ci_val, d in craft_predecessors:
                                pg = [g for g in self.goals.entries if p in g.event_dims]
                                if pg:
                                    pred_goals = pg
                                    break

                        if pred_goals:
                            m_soft_arr = np.asarray(self.belief.m_soft, dtype=np.float32).reshape(-1)
                            locked_dims_tmp = [i for i in range(NUM_ACHIEVEMENTS) if m_soft_arr[i] < 0.5]
                            best_pred = max(
                                pred_goals,
                                key=lambda g: self.goals._compute_goal_score(
                                    g, s_np, locked_dims_tmp, _psi_fn, _danger_fn, None, self.lift, self.current_iter, self.beta_t, critic=self.critic_net
                                )
                            )
                            self.target_goal_pending = best_goal
                            self.material_subgoal_cycles = cycles
                            best_goal = best_pred
                            self.current_goal_mode = "material_subgoal"
                    else:
                        # Materials satisfied! Check F8 Craft-execution step
                        top_affordance = float(all_event_probs[top_a, target_dims].max().item())
                        is_unbuilt = any(not self.belief.has(d) for d in target_dims)
                        aff_min = getattr(self.cfg.goals, "probe_affordance_min", 0.05)
                        if is_unbuilt and top_affordance >= aff_min:
                            self.mode = "PROBE"
                            self.probe_steps_remaining = self.cfg.goals.probe_steps - 1
                            self.probe_unlocked = False
                            self.probe_id += 1
                            self.probe_newly_unlocked.clear()
                            self.current_goal_mode = "execute_craft"
                            self.active_goal = best_goal
                            self.nav_steps = 0
                            self.stagnation_counter = 0
                            self.progress_best = -1.0

            if best_goal is not None and score > 0.0 and self.mode != "PROBE":
                self.active_goal = best_goal
                self.goal_score = score
                if self.current_goal_mode not in ("material_subgoal", "execute_craft"):
                    self.current_goal_mode = getattr(self.goals, "last_selection_mode", "none")

                # Reachability-scaled option budget (V2.2 F9)
                g_key_t = torch.from_numpy(best_goal.key).float().to(self.device).unsqueeze(0)
                psi_val = float(self.reach_net(s, g_key_t).item())
                b_min = getattr(self.cfg.skills, "option_budget_min", 25)
                b_max = getattr(self.cfg.skills, "option_budget_max", 60)
                scaled_budget = self.cfg.skills.max_option_steps * (1.0 + 2.0 * max(0.0, 1.0 - psi_val))
                self.current_option_budget = int(min(b_max, max(b_min, round(scaled_budget))))

                self.nav_steps = 0
                self.stagnation_counter = 0
                self.progress_best = -1.0
                arrival_cos = _cosine(z_np, best_goal.key)
                if arrival_cos >= self.cfg.skills.arrival_cosine:
                    self.mode = "PROBE"
                    self.probe_steps_remaining = self.cfg.goals.probe_steps - 1
                    self.probe_unlocked = False
                    self.probe_id += 1
                    self.probe_newly_unlocked.clear()
                else:
                    self.mode = "NAVIGATE"

        # Select intended action by mode
        if self.mode == "PROBE":
            # argmax_a [ max_{i: m_soft_i < 0.5} p(a, i) + probe_entropy * softmax(disagreement(s, a)) ]
            m_soft = self.belief.m_soft
            locked_dims = [i for i in range(NUM_ACHIEVEMENTS) if m_soft[i] < 0.5]
            if locked_dims:
                affordance = all_event_probs[:, locked_dims].max(dim=-1).values
            else:
                affordance = all_event_probs.max(dim=-1).values

            d_all = self.epistemic_net.disagreement_all_actions(s).squeeze(0)  # [17]

            # V3 F4: Confidence-gated exploration focus in blind contexts
            if float(affordance.max().item()) < 0.05:
                blind_boost = getattr(self.cfg.goals, "blind_probe_boost", 1.0)
                curiosity_weight = self.cfg.goals.probe_entropy * (1.0 + blind_boost * float(d_all.mean().item()))
            else:
                curiosity_weight = self.cfg.goals.probe_entropy

            probe_score = affordance + curiosity_weight * F.softmax(d_all, dim=-1)

            # V2.2-F11-fix / V3 F5: learned material gating
            learned_gate_info = None
            suppress = None
            if (
                self.cfg.goals.material_gating_learned
                and getattr(self.backbone, "has_learned_delta_head", False)
                and self.readout_net is not None
            ):
                with torch.no_grad():
                    a_all = torch.arange(NUM_ACTIONS, device=self.device)
                    s_rep = s.expand(NUM_ACTIONS, -1)
                    deltas = self.backbone.inventory_delta(s_rep, a_all).cpu().numpy()
                    mean_x, _ = self.readout_net(s)
                    have = mean_x.squeeze(0).detach().cpu().numpy()
                    suppress = compute_material_suppression(
                        deltas, have, margin=self.cfg.goals.material_gate_margin
                    )
                    probe_score[suppress] -= 100.0
                    suppressed_cnt = int(np.sum(suppress))
                    learned_gate_info = {
                        "suppressed": suppress.tolist(),
                        "have": have.tolist(),
                        "suppressed_count": suppressed_cnt,
                    }

            a_intended = int(torch.argmax(probe_score).item())

            self.last_probe_decision = {
                "step": step,
                "goal_kind": self.active_goal.kind if self.active_goal else "none",
                "goal_dim": list(self.active_goal.event_dims) if self.active_goal else [],
                "blended_scores": probe_score.detach().cpu().numpy().tolist(),
                "probe_event_probs": all_event_probs.detach().cpu().to(torch.float16).numpy().tolist(),
                "arrival_cos": float(arrival_cos),
                "matched_bank": False,
                "material_gate_learned": learned_gate_info,
                "suppressed_count": int(np.sum(suppress)) if suppress is not None else 0,
                "probe_id": self.probe_id,
                "unlocked_dims": list(self.probe_newly_unlocked),
            }

            if self.probe_steps_remaining == 0 and self.active_goal is not None:
                if not self.frozen_memory:
                    matched = self.goals.mark_used(self.active_goal.key, success=self.probe_unlocked, current_iter=self.current_iter)
                    if self.last_probe_decision is not None:
                        self.last_probe_decision["matched_bank"] = bool(matched)

                if self.material_subgoal_cycles > 0:
                    self.material_subgoal_cycles -= 1

                if self.target_goal_pending is not None and self.material_subgoal_cycles <= 0:
                    self.active_goal = self.target_goal_pending
                    self.target_goal_pending = None
                    self.current_goal_mode = "released_target"
                else:
                    self.active_goal = None

        elif self.mode == "NAVIGATE" and self.active_goal is not None:
            g_key_t = torch.from_numpy(self.active_goal.key).float().to(self.device).unsqueeze(0)
            a_tensor = self.option_net.act(s, g_key_t, greedy=True)
            a_intended = int(a_tensor.item())

        elif self.mode == "WANDER":
            # WANDER: sample from softmax(beta_t * disagreement) with temperature 1.0
            d_all = self.epistemic_net.disagreement_all_actions(s).squeeze(0)  # [17]
            probs = F.softmax(self.beta_t * d_all, dim=-1)
            a_intended = int(torch.multinomial(probs, num_samples=1).item())

        else:
            # FALLBACK: beta_t * disagreement + prior_logits_weighted
            self.mode = "FALLBACK"
            prior_logits = self.backbone.prior_logits_weighted(s, mask_t).squeeze(0)  # [17]
            d_all = self.epistemic_net.disagreement_all_actions(s).squeeze(0)  # [17]
            logits = self.beta_t * d_all + prior_logits
            a_intended = int(torch.argmax(logits).item())

        # Universal epsilon random exploration tail across ALL modes
        if np.random.rand() < self.epsilon:
            a_intended = int(np.random.randint(0, NUM_ACTIONS))

        # 4) Safety override (§5.6)
        all_risks = self.safety_net.risk_all_actions(s).squeeze(0)  # [17]
        risk_intended = float(all_risks[a_intended].item())
        min_risk_val = float(torch.min(all_risks).item())

        override_fired = False
        a_final = a_intended

        if (
            risk_intended > self.cfg.safety.risk_threshold
            and min_risk_val < (risk_intended - self.cfg.safety.risk_margin)
        ):
            # Take safest action; break ties with q_mean
            min_mask = (all_risks <= min_risk_val + 1e-4)
            candidate_actions = torch.where(min_mask)[0]
            if len(candidate_actions) == 1:
                a_final = int(candidate_actions[0].item())
            else:
                q_vals = torch.tensor(
                    [float(self.backbone.q_mean(s, cand.unsqueeze(0)).item()) for cand in candidate_actions],
                    device=self.device,
                )
                best_cand_idx = int(torch.argmax(q_vals).item())
                a_final = int(candidate_actions[best_cand_idx].item())
            override_fired = True

        # Compute intrinsic curiosity signal
        a_final_t = torch.tensor([a_final], dtype=torch.int64, device=self.device)
        d_val = float(self.epistemic_net.disagreement(s, a_final_t).item())
        knn_val = knn_novelty(h_np, self.h_memory, k=4)
        r_int = d_val + self.cfg.epistemic.knn_beta * knn_val

        # Predict event probs for chosen action (used to update belief next step)
        chosen_event_probs = self.backbone.event_probs(s, a_final_t, mask_t)
        self.last_event_probs = chosen_event_probs.squeeze(0)

        # Update step state
        self.h_prev = h
        self.a_prev = a_final_t

        # Compute human-readable decision reason
        a_int_name = CRAFTER_ACTION_NAMES[a_intended] if a_intended < len(CRAFTER_ACTION_NAMES) else str(a_intended)
        a_fin_name = CRAFTER_ACTION_NAMES[a_final] if a_final < len(CRAFTER_ACTION_NAMES) else str(a_final)

        if override_fired:
            decision_reason = (
                f"SAFETY OVERRIDE: Intended '{a_int_name}' had high risk ({risk_intended:.2f} > {self.cfg.safety.risk_threshold:.2f}). "
                f"Safest action '{a_fin_name}' chosen instead (risk: {min_risk_val:.2f})."
            )
        elif self.mode == "NAVIGATE":
            goal_key_str = f"Goal key #{int(abs(hash(str(self.active_goal.key[:4])))) % 1000}" if self.active_goal else "Goal"
            decision_reason = (
                f"NAVIGATE: OptionNet chose '{a_fin_name}' pursuing {goal_key_str} "
                f"(arrival_cos: {arrival_cos:.2f}, goal_score: {self.goal_score:.2f}, risk: {risk_intended:.2f})."
            )
        elif self.mode == "PROBE":
            decision_reason = (
                f"PROBE (Curiosity): Chose '{a_fin_name}' exploring affordances for locked achievements "
                f"(disagreement: {d_val:.2f}, risk: {risk_intended:.2f})."
            )
        elif self.mode == "WANDER":
            decision_reason = (
                f"WANDER (Exploration on timeout): Stochastic curiosity chose '{a_fin_name}' "
                f"after goal stagnation ({self.wander_steps_remaining} steps remaining, risk: {risk_intended:.2f})."
            )
        else:
            decision_reason = (
                f"FALLBACK (Prior + Exploration): Prior policy chose '{a_fin_name}' "
                f"(disagreement: {d_val:.2f}, risk: {risk_intended:.2f})."
            )

        self.prev_mode = self.mode

        telemetry = {
            "mode": self.mode,
            "goal_score": float(self.goal_score),
            "arrival_cos": float(arrival_cos),
            "psi": float(psi_val),
            "risk_intended": float(risk_intended),
            "override_fired": bool(override_fired),
            "r_int": float(r_int),
            "reason": decision_reason,
            "a_intended": a_int_name,
            "a_final": a_fin_name,
            "goal_mode": self.current_goal_mode,
            "option_budget": int(self.current_option_budget),
            "abort_reason": self.last_abort_reason,
        }

        # Record step buffer for collection
        self.episode_buffer.append({
            "z": z_np,
            "h": h_np,
            "a": a_final,
            "event_probs": self.last_event_probs.detach().cpu().numpy(),
            "m_soft": self.belief.m_soft.copy(),
            "m_bin": self.belief.m_bin.copy(),
            "disagreement": d_val,
        })

        return a_final, telemetry

    def observe_result(
        self,
        obs_uint8: np.ndarray,
        reward: float,
        done: bool,
        discount: float,
    ) -> None:
        """Called after env.step to log outcomes into episode buffer."""
        if self.episode_buffer:
            self.episode_buffer[-1]["reward"] = float(reward)
            self.episode_buffer[-1]["done"] = bool(done)
            self.episode_buffer[-1]["discount"] = float(discount)
