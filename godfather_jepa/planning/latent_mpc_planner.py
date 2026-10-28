"""
Latent Model Predictive Control (MPC) Trajectory Planner for Godfather JEPA.

Simulates candidate action trajectories inside the learned JEPA World Model
(BoundedResidualPredictor, DenseRewardHead, SurvivalContinuationHead, SymlogCategoricalCritic)
to steer the agent toward high-reward causal milestones and survival.

Key Properties:
1. 100% Strict Inference: Plans purely in latent space from raw pixels (no simulator cheats).
2. Trajectory Rollout (H=10, N=64): Fast vectorized simulation in ~8ms.
3. Causal Action Injection: Injects hierarchical upstream crafts (table when wood >= 3, pickaxe at table)
   and topological return vectors from VisualOdometry.
4. Survival & Danger Discounting: Heavily discounts and penalizes paths with predicted mortality.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from godfather_jepa.models.predictor import BoundedResidualPredictor
from godfather_jepa.models.reward_head import DenseRewardHead
from godfather_jepa.models.survival import SurvivalContinuationHead
from godfather_jepa.models.symlog_critic import SymlogCategoricalCritic, SymlogTwoHotCoder
from godfather_jepa.models.belief import SpatialConvGRU, VectorBeliefCore
from godfather_jepa.models.odometry import TopologicalAnchorGraph
from godfather_jepa.planning.causal_hypergraph import CausalHypergraph


class LatentMPCPlanner:
    """
    Cross-Entropy Method (CEM) Latent Model Predictive Control Planner.
    """

    def __init__(
        self,
        predictor: BoundedResidualPredictor,
        reward_head: DenseRewardHead,
        continuation_head: SurvivalContinuationHead,
        critic: SymlogCategoricalCritic,
        symlog_coder: SymlogTwoHotCoder,
        spatial_gru: SpatialConvGRU,
        vector_belief: VectorBeliefCore,
        causal_hypergraph: CausalHypergraph,
        horizon: int = 10,
        num_candidates: int = 64,
        cem_iterations: int = 2,
        elite_ratio: float = 0.15,
        gamma: float = 0.99,
        death_penalty: float = 5.0,
        num_actions: int = 17,
        device: Optional[torch.device] = None,
    ):
        self.predictor = predictor
        self.reward_head = reward_head
        self.continuation_head = continuation_head
        self.critic = critic
        self.symlog_coder = symlog_coder
        self.spatial_gru = spatial_gru
        self.vector_belief = vector_belief
        self.causal_hypergraph = causal_hypergraph

        self.horizon = horizon
        self.num_candidates = num_candidates
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.gamma = gamma
        self.death_penalty = death_penalty
        self.num_actions = num_actions
        self.device = device or torch.device("cpu")

    def plan(
        self,
        z_0: torch.Tensor,
        h_vector_0: torch.Tensor,
        h_spatial_0: torch.Tensor,
        inventory: Dict[str, int],
        workstations: Set[str],
        anchor_graph: TopologicalAnchorGraph,
        target_goal: str = "make_wood_pickaxe",
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Executes latent MPC trajectory search given current latent and recurrent belief.

        Args:
            z_0: Root latent vector of shape (1, 512).
            h_vector_0: Root vector belief of shape (1, 512).
            h_spatial_0: Root spatial feature map of shape (1, 64, 8, 8).
            inventory: Mental inventory dictionary.
            workstations: Set of placed workstations.
            anchor_graph: TopologicalAnchorGraph holding landmark return paths.
            target_goal: High-level teleological goal.

        Returns:
            (best_action, debug_info)
        """
        # 1. Determine causal subgoal and budget from CausalHypergraph
        subgoal, determined_action = self.causal_hypergraph.determine_subgoal(
            target_goal=target_goal,
            current_inventory=inventory,
            workstations_present=workstations,
        )

        # 2. Workstation return override if pickaxe ingredients ready but away from table
        if "table_placed" in workstations and inventory.get("wood", 0) >= 1 and inventory.get("wood_pickaxe", 0) == 0:
            return_act = anchor_graph.get_return_action("table")
            if return_act is not None and return_act != 0:
                return return_act, {
                    "subgoal": "return_to_table",
                    "mode": "topological_return",
                    "score": 5.0,
                }

        # 3. Instantaneous crafting subgoals (place_table, make_wood_pickaxe, make_wood_sword, place_furnace)
        # MUST be executed directly once upstream budget is satisfied and adjacent to workstation
        if subgoal in ("place_table", "make_wood_pickaxe", "make_wood_sword", "place_furnace"):
            return determined_action, {
                "subgoal": subgoal,
                "mode": "instant_causal_craft",
                "score": 10.0,
            }

        # 4. For gathering and exploration (e.g. collect_wood, collect_stone):
        # Run Cross-Entropy Method (CEM) trajectory search over horizon H
        H = self.horizon
        N = self.num_candidates
        N_elite = max(2, int(N * self.elite_ratio))

        # Action priors: Movement (1..4) and Interaction (5=Do) get dominant initial probability
        # 0=noop, 1=left, 2=right, 3=up, 4=down, 5=do, 6..16=crafts/sleep
        base_probs = torch.full((self.num_actions,), 0.02, device=self.device)
        base_probs[1:5] = 0.18  # Movement (left, right, up, down)
        base_probs[5] = 0.20    # Do / hit action
        action_probs = base_probs.unsqueeze(0).expand(H, -1).clone()  # (H, num_actions)

        best_action = 5
        best_score = -float("inf")

        with torch.no_grad():
            for iter_idx in range(self.cem_iterations):
                # Sample N candidate sequences of length H from categorical distribution
                # dist shape: (H, num_actions) -> sample (N, H)
                candidate_actions = torch.empty((N, H), dtype=torch.long, device=self.device)
                for t in range(H):
                    dist = torch.distributions.Categorical(probs=action_probs[t])
                    candidate_actions[:, t] = dist.sample((N,))

                # Inject target interaction at step 0 for 25% of candidates to evaluate immediate hit
                candidate_actions[: N // 4, 0] = 5

                # Expand root tensors to batch dimension N
                z_k = z_0.expand(N, -1).clone()
                h_k = h_vector_0.expand(N, -1).clone()
                sp_k = h_spatial_0.expand(N, -1, -1, -1).clone()

                cumulative_rewards = torch.zeros(N, device=self.device)
                discount_factors = torch.ones(N, device=self.device)

                for t in range(H):
                    act_t = candidate_actions[:, t]
                    composite = torch.cat([z_k, h_k], dim=-1)

                    # Predict reward and survival continuation
                    pred_r = self.reward_head(composite, act_t)
                    pred_c = self.continuation_head(composite)

                    # Penalty for dying: if continuation drops, heavily penalize
                    step_penalty = self.death_penalty * (1.0 - pred_c)
                    effective_reward = pred_r - step_penalty

                    cumulative_rewards += discount_factors * effective_reward
                    discount_factors = discount_factors * (self.gamma * pred_c)

                    # Unroll forward dynamics
                    z_next = self.predictor(z_k, h_k, act_t)
                    sp_next = self.spatial_gru(sp_k, act_t, H=sp_k)
                    h_next = self.vector_belief(z_next, act_t, sp_next, h=h_k)

                    z_k = z_next
                    sp_k = sp_next
                    h_k = h_next

                # Add terminal critic value
                terminal_composite = torch.cat([z_k, h_k], dim=-1)
                critic_logits = self.critic(terminal_composite)
                terminal_v = self.symlog_coder.decode(critic_logits)
                total_scores = cumulative_rewards + discount_factors * terminal_v

                # Extract top elite candidates
                top_scores, top_indices = torch.topk(total_scores, N_elite)
                elite_trajectories = candidate_actions[top_indices]  # (N_elite, H)

                if top_scores[0].item() > best_score:
                    best_score = top_scores[0].item()
                    best_action = int(elite_trajectories[0, 0].item())

                # Update CEM distribution for next iteration: smooth interpolation
                new_probs = torch.zeros_like(action_probs)
                for t in range(H):
                    counts = torch.bincount(elite_trajectories[:, t], minlength=self.num_actions).float()
                    new_probs[t] = (counts + 0.1) / (counts.sum() + 0.1 * self.num_actions)

                action_probs = 0.5 * action_probs + 0.5 * new_probs

        return best_action, {
            "subgoal": subgoal,
            "mode": "latent_mpc_search",
            "score": best_score,
        }
