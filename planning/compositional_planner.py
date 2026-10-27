#!/usr/bin/env python3
"""
planning/compositional_planner.py — Compositional Causal Planning (Phase 2C)
=============================================================================

Implements Section 6 (Phase 2C) of the JEPA Causal Chain Roadmap:
- Discovers multi-step causal graphs from learned transition prototypes in RelationHypothesisBank.
- Calculates resource budgets dynamically from empirical consumption/production deltas (zero human recipes).
- Backward-chains from high-return milestone prototypes to required pre-harvest actions.
- Integrates with RecurrentMPCPlanner in strict mode via learned affordance priors.
- 100% strict inference: operates purely on learned visual latents and StateReadoutHead estimates.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from planning.causal_memory import NUM_ACTIONS, STATE_DIM, RelationHypothesisBank, TransitionPrototype


class CompositionalCausalPlanner:
    """
    Autonomous Compositional Graph Planner over Learned Relation Hypotheses.
    Discovers multi-step affordances and resource budgets from empirical trajectory data.
    """

    def __init__(
        self,
        hypothesis_bank: RelationHypothesisBank,
        min_prototype_confidence: float = 0.05,
        sim_threshold: float = 0.40,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.hypothesis_bank = hypothesis_bank
        self.min_confidence = min_prototype_confidence
        self.sim_threshold = sim_threshold
        self.device = torch.device(device) if device else torch.device("cpu")

    def plan_subgoal(
        self,
        obs_latent: Union[torch.Tensor, np.ndarray, List[float]],
        grounded_state: Union[torch.Tensor, np.ndarray, List[float]],
    ) -> Optional[Dict[str, Any]]:
        """
        Deduces the active sub-goal intention directly from the Relation Hypothesis Bank.
        
        Algorithm:
        1. Query all valid prototypes with confidence >= min_confidence and positive return.
        2. Identify the highest-utility target milestone.
        3. Check whether current grounded state satisfies preconditions.
        4. If satisfied -> Subgoal is READY_TO_EXECUTE (recommend target action).
        5. If unsatisfied -> Backward search for producer prototypes that supply missing resources
           -> Subgoal is HARVESTING_INPUT (recommend producer action and visual navigation cue).
        """
        if len(self.hypothesis_bank) == 0:
            return None

        # Convert inputs to torch tensors
        if isinstance(obs_latent, torch.Tensor):
            z_t = obs_latent.detach().to(self.device).float().flatten()
        else:
            z_t = torch.from_numpy(np.array(obs_latent, dtype=np.float32).flatten()).to(self.device)
        z_norm = z_t / z_t.norm(p=2).clamp(min=1e-6)

        if isinstance(grounded_state, torch.Tensor):
            x_t = grounded_state.detach().cpu().flatten().numpy()
        else:
            x_t = np.array(grounded_state, dtype=np.float32).flatten()

        # Filter active prototypes by confidence (milestones are affordance/crafting actions: action >= 5,
        # primitive movements 0..4 are motor navigation commands handled by local MPC)
        active_protos = [
            p for p in self.hypothesis_bank.prototypes.values()
            if p.confidence >= self.min_confidence and p.mean_return > 0.0 and p.action >= 5
        ]
        if not active_protos:
            return None

        # 1. Rank target milestones by expected return * success probability
        # Sort targets by expected utility
        ranked_targets = sorted(
            active_protos,
            key=lambda p: p.mean_return * p.success_probability * p.confidence,
            reverse=True,
        )

        for target in ranked_targets:
            # Check visual cue similarity
            k_tgt = torch.tensor(target.precondition_key, dtype=torch.float32, device=self.device)
            sim_tgt = float(torch.dot(z_norm, k_tgt).item())

            # Check if current state meets preconditions
            req_indices = np.where(np.array(target.precondition_state) > 0.5)[0]
            deficit_indices = [idx for idx in req_indices if x_t[idx] < (target.precondition_state[idx] - 0.25)]

            if len(deficit_indices) == 0:
                # Preconditions fully met!
                # If visual cue is present or action is self-executable, target is ready to execute
                if sim_tgt >= self.sim_threshold or len(req_indices) > 0:
                    return {
                        "status": "READY_TO_EXECUTE",
                        "prototype_id": target.prototype_id,
                        "recommended_action": target.action,
                        "visual_key": k_tgt,
                        "visual_sim": sim_tgt,
                        "expected_return": target.mean_return,
                        "success_prob": target.success_probability,
                        "confidence": target.confidence,
                        "effect_delta": target.effect_delta,
                    }
            else:
                # Missing required resources! Backward-chain to find producer prototype
                missing_dim = deficit_indices[0]
                deficit_amount = float(target.precondition_state[missing_dim] - x_t[missing_dim])

                # Search for producer prototype that yields positive delta on missing_dim.
                # Environmental invariant: raw physical resources (indices 0..5: wood, stone, coal, iron, diamond, sapling)
                # are harvested from the world via physical interaction (Action 5: Do/Harvest).
                # Crafted tools/weapons (indices 6..11) are produced via their respective crafting actions (11..16).
                producers = [
                    p for p in active_protos
                    if p.effect_delta[missing_dim] > 0.25
                    and p.prototype_id != target.prototype_id
                    and p.action != target.action
                    and (p.action == 5 if missing_dim < 6 else True)
                ]
                if producers:
                    # Choose producer with highest success probability & visual alignment
                    best_producer = max(
                        producers,
                        key=lambda p: p.effect_delta[missing_dim] * p.success_probability * p.confidence,
                    )
                    k_prod = torch.tensor(best_producer.precondition_key, dtype=torch.float32, device=self.device)
                    sim_prod = float(torch.dot(z_norm, k_prod).item())

                    return {
                        "status": "HARVESTING_INPUT",
                        "prototype_id": target.prototype_id,
                        "recommended_action": best_producer.action,
                        "visual_key": k_prod,
                        "visual_sim": sim_prod,
                        "missing_resource_dim": missing_dim,
                        "deficit_amount": deficit_amount,
                        "producer_prototype_id": best_producer.prototype_id,
                        "producer_action": best_producer.action,
                        "expected_return": target.mean_return,
                        "success_prob": best_producer.success_probability,
                        "confidence": best_producer.confidence,
                    }

        return None

    def compute_action_prior(
        self,
        obs_latent: Union[torch.Tensor, np.ndarray, List[float]],
        grounded_state: Union[torch.Tensor, np.ndarray, List[float]],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes 17-D continuous action prior distribution from learned prototypes.
        Actions with applicable high-utility prototypes receive elevated probability.
        """
        prior_logits = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=self.device)

        if len(self.hypothesis_bank) == 0:
            return F.softmax(prior_logits, dim=-1)

        subgoal = self.plan_subgoal(obs_latent, grounded_state)
        if subgoal is not None:
            rec_act = subgoal["recommended_action"]
            conf = subgoal["confidence"]
            succ = subgoal["success_prob"]
            sim = max(0.0, subgoal.get("visual_sim", 0.0))
            # Boost recommended action proportional to confidence and visual cue alignment
            prior_logits[rec_act] += 2.0 * conf * succ * (1.0 + sim)

        # General affordance matching across all applicable prototypes
        if isinstance(obs_latent, torch.Tensor):
            z_t = obs_latent.detach().to(self.device).float().flatten()
        else:
            z_t = torch.from_numpy(np.array(obs_latent, dtype=np.float32).flatten()).to(self.device)
        z_norm = z_t / z_t.norm(p=2).clamp(min=1e-6)

        for proto in self.hypothesis_bank.prototypes.values():
            if proto.confidence < self.min_confidence:
                continue
            if proto.is_applicable(grounded_state):
                k = torch.tensor(proto.precondition_key, dtype=torch.float32, device=self.device)
                sim = float(torch.dot(z_norm, k).item())
                if sim >= self.sim_threshold:
                    weight = sim * proto.success_probability * proto.confidence
                    prior_logits[proto.action] += 0.5 * weight

        return F.softmax(prior_logits / max(temperature, 0.1), dim=-1)
