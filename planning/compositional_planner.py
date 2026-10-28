#!/usr/bin/env python3
"""
planning/compositional_planner.py — Compositional Causal Planning (Phase 2C, redesign)
=======================================================================================

Implements Stage 5 of docs/PHASE2_MEMORY_REDESIGN_PLAN.md:
- Top-level targets are EVENT-GROUNDED cards whose achievement is still locked in the
  strictly-learned belief mask m_bin (marginal value gating — one-time achievements
  cannot be farmed). Event-free resource cards are reachable ONLY as chain steps.
- Ranking: trust (Wilson lower bound) + UCB exploration bonus — low-evidence
  high-value cards get re-tested instead of being punished (the old confidence
  multiplier is gone; mean_return is never read).
- Multi-step backward chaining (depth-limited DFS) over the causal graph with cost
  summation: enablers first (an enabling edge must be satisfied in m_bin before its
  successor), then resource deficits via producers — "3 wood = 2 for table + 1 for
  pickaxe" emerges from the ledger, not from recipes.
- 100% strict inference: operates on learned visual latents, StateReadoutHead
  estimates, and the belief's achievement mask. Zero access to engine state.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from planning.causal_memory import (
    NUM_ACTIONS,
    STATE_DIM,
    NUM_ACHIEVEMENTS,
    RelationEdge,
    RelationHypothesisBank,
    TransitionPrototype,
)


class CompositionalCausalPlanner:
    """
    Autonomous Compositional Graph Planner over learned causal cards.
    """

    def __init__(
        self,
        hypothesis_bank: RelationHypothesisBank,
        min_prototype_confidence: float = 0.05,
        sim_threshold: float = 0.40,
        device: Optional[Union[str, torch.device]] = None,
        ucb_weight: float = 0.3,
        ucb_cap: float = 0.5,
        chain_depth: int = 3,
        top_k_targets: int = 3,
        chain_cost_penalty: float = 0.15,
        hypothesis_trust_floor: float = 0.2,
    ):
        self.hypothesis_bank = hypothesis_bank
        self.min_confidence = min_prototype_confidence
        self.sim_threshold = sim_threshold
        self.device = torch.device(device) if device else torch.device("cpu")
        self.ucb_weight = ucb_weight
        self.ucb_cap = ucb_cap
        self.chain_depth = chain_depth
        self.top_k_targets = top_k_targets
        self.chain_cost_penalty = chain_cost_penalty
        self.hypothesis_trust_floor = hypothesis_trust_floor

    # ------------------------------------------------------------------ trust

    def _trust(self, card: TransitionPrototype) -> float:
        """Ranking trust: Wilson lower bound, floored for hypothesis cards so
        low-evidence high-value cards stay re-testable (exploration, not avoidance)."""
        trust = float(card.wilson_lb)
        if card.status == "hypothesis":
            trust = max(trust, self.hypothesis_trust_floor)
        return trust

    def _ucb_bonus(self, card: TransitionPrototype) -> float:
        """UCB exploration bonus over empirical attempts; decays ~1/sqrt(n).
        N_total is the bank's total ATTEMPT evidence (sum of card attempt counts),
        not the corpus step count — otherwise the bonus saturates at the cap and
        exploration cannot discriminate between cards (review finding, Stage 5)."""
        bank = self.hypothesis_bank
        if not hasattr(self, "_n_total_cache") or self._n_total_cache[0] != len(bank.prototypes):
            self._n_total_cache = (len(bank.prototypes), sum(p.attempt_count for p in bank.prototypes.values()))
        n_total = max(self._n_total_cache[1], 1)
        n_p = max(card.attempt_count, 0)
        return float(min(self.ucb_cap, self.ucb_weight * (2.0 * np.log(1.0 + n_total) / (1.0 + n_p)) ** 0.5))

    def _is_marginal_target(self, card: TransitionPrototype, m_bin: np.ndarray) -> bool:
        """Top-level targets: event-grounded AND still locked in the belief mask."""
        return (
            card.event_dim is not None
            and 0 <= int(card.event_dim) < NUM_ACHIEVEMENTS
            and m_bin[int(card.event_dim)] < 0.5
            and card.confidence >= self.min_confidence
        )

    # ------------------------------------------------------------- chaining

    def _deficits(self, card: TransitionPrototype, x_t: np.ndarray) -> List[Tuple[int, float]]:
        out = []
        for dim, amount in card.needs:
            d, a = int(dim), float(amount)
            if d < 0 or d >= STATE_DIM:
                continue
            if x_t[d] < a - 0.25:
                out.append((d, a))
        return out

    def _enabler_unsatisfied(self, edge: RelationEdge, m_bin: np.ndarray) -> bool:
        """An enabling edge must be 'done' before its successor executes: the
        enabler card's event must already be locked in m_bin."""
        src = self.hypothesis_bank.prototypes.get(edge.src_id)
        if src is None or src.event_dim is None:
            return False
        return m_bin[int(src.event_dim)] < 0.5

    def _producer_amount(self, card: TransitionPrototype, dim: int) -> float:
        for d, amt in card.produces:
            if int(d) == dim:
                return float(amt)
        return 0.0

    def _resolve_chain(
        self,
        card: TransitionPrototype,
        x_t: np.ndarray,
        m_bin: np.ndarray,
        depth: int,
        visited: set,
    ) -> Optional[List[Dict[str, Any]]]:
        """DFS: returns the ordered chain [{card, action, applications}] that makes
        `card` executable, or None if not groundable within the depth limit."""
        if depth > self.chain_depth or card.prototype_id in visited:
            return None
        visited = visited | {card.prototype_id}
        prefix: List[Dict[str, Any]] = []

        # 1. Enablers first: unsatisfied enabling predecessors must happen before us.
        for edge in sorted(
            self.hypothesis_bank.edges.get(card.prototype_id, []),
            key=lambda e: -float(e.lift),
        ):
            if edge.kind != "enabling_lift" or not self._enabler_unsatisfied(edge, m_bin):
                continue
            enabler = self.hypothesis_bank.prototypes.get(edge.src_id)
            if enabler is None:
                continue
            sub = self._resolve_chain(enabler, x_t, m_bin, depth + 1, visited)
            if sub is None:
                return None  # enabler ungroundable -> the whole chain is
            prefix.extend(sub)

        # 2. Resource deficits: best producer per deficit, recursed.
        for dim, amount in self._deficits(card, x_t):
            producers = [
                p
                for p in self.hypothesis_bank.producers_for_dim(dim)
                if p.prototype_id != card.prototype_id and p.action != card.action
            ]
            producers.sort(
                key=lambda p: (
                    p.status == "confirmed",
                    self._trust(p) * min(self._producer_amount(p, dim), 1.0),
                ),
                reverse=True,
            )
            chained = None
            for producer in producers:
                sub = self._resolve_chain(producer, x_t, m_bin, depth + 1, visited)
                if sub is not None:
                    chained = sub
                    break
            if chained is None:
                return None
            per_apply = max(self._producer_amount(chained[-1]["card"], dim), 1e-6)
            # Cap applications per deficit: protects the cost term from a producer with a
            # tiny learned delta demanding absurd repeat counts within one plan step.
            applications = int(min(10, max(1, int(np.ceil(amount / per_apply)))))
            for link in chained:
                link["applications"] = link.get("applications", 0)
            chained[-1]["applications"] = applications
            prefix.extend(chained)

        # 3. This card itself: one execution.
        prefix.append({"card": card, "card_id": card.prototype_id, "action": int(card.action), "applications": 1})
        return prefix

    def _chain_stats(self, chain: List[Dict[str, Any]], target: TransitionPrototype) -> Tuple[float, int]:
        trust_prod = 1.0
        total_cost = 0
        for link in chain[:-1]:  # producers/enablers; the target's own execution is implicit
            card = link["card"]
            trust_prod *= self._trust(card)
            total_cost += int(link.get("applications", 1))
        chain_value = self._trust(target) * trust_prod * (0.9 ** max(len(chain) - 1, 0))
        return chain_value, total_cost

    # ------------------------------------------------------------- main entry

    def plan_subgoal(
        self,
        obs_latent: Union[torch.Tensor, np.ndarray, List[float]],
        grounded_state: Union[torch.Tensor, np.ndarray, List[float]],
        m_bin: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Deduces the active sub-goal from the causal card bank.

        1. Candidate targets: event-grounded cards still locked in m_bin (marginal).
        2. Rank by trust + UCB; evaluate the top-K with multi-step backward chaining.
        3. Return the best groundable chain, or None when nothing is marginal/groundable
           (memory goes silent and the transition model drives).
        """
        bank = self.hypothesis_bank
        if len(bank) == 0:
            return None

        if isinstance(obs_latent, torch.Tensor):
            z_t = obs_latent.detach().to(self.device).float().flatten()
        else:
            z_t = torch.from_numpy(np.array(obs_latent, dtype=np.float32).flatten()).to(self.device)
        z_norm = z_t / z_t.norm(p=2).clamp(min=1e-6)

        if isinstance(grounded_state, torch.Tensor):
            x_t = grounded_state.detach().cpu().flatten().numpy()
        else:
            x_t = np.array(grounded_state, dtype=np.float32).flatten()

        if m_bin is None:
            m_arr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        elif isinstance(m_bin, torch.Tensor):
            m_arr = m_bin.detach().cpu().flatten().numpy()
        else:
            m_arr = np.array(m_bin, dtype=np.float32).flatten()

        targets = [p for p in bank.prototypes.values() if self._is_marginal_target(p, m_arr)]
        if not targets:
            return None

        # Confirmed Causal Precedence:
        # If an event dimension already has at least one confirmed recipe in the bank,
        # unverified hypothesis cards for that event dimension cannot hijack or preempt the confirmed recipe.
        # Hypothesis cards are only eligible for frontier events where NO confirmed recipe exists yet.
        confirmed_events = {
            int(p.event_dim)
            for p in bank.prototypes.values()
            if p.event_dim is not None and p.status == "confirmed"
        }
        eligible_targets = [
            p for p in targets
            if (int(p.event_dim) not in confirmed_events or p.status == "confirmed")
        ]
        if not eligible_targets:
            return None

        eligible_targets.sort(key=lambda p: -(self._trust(p) + self._ucb_bonus(p)))

        best: Optional[Dict[str, Any]] = None
        for target in eligible_targets[: self.top_k_targets]:
            chain = self._resolve_chain(target, x_t, m_arr, 0, set())
            if chain is None:
                continue
            chain_value, total_cost = self._chain_stats(chain, target)
            score = chain_value / (1.0 + self.chain_cost_penalty * total_cost) + self._ucb_bonus(target)
            if best is None or score > best["score"]:
                first = chain[0]
                active_card = first["card"]
                k_active = torch.tensor(active_card.precondition_key, dtype=torch.float32, device=self.device)
                sim_active = float(torch.dot(z_norm, k_active).item())
                # Active link ready gate:
                # 1. Active step's cue is visible (sim >= sim_threshold), OR
                # 2. Active step is visually invariant (dispersion < 0.70), OR
                # 3. First step is an intermediate producer/enabler that initiates search/harvesting
                ready = (
                    sim_active >= self.sim_threshold
                    or float(active_card.visual_dispersion) < 0.70
                    or (first["card_id"] != target.prototype_id)
                )
                if not ready:
                    continue
                k_tgt = torch.tensor(target.precondition_key, dtype=torch.float32, device=self.device)
                sim_tgt = float(torch.dot(z_norm, k_tgt).item())
                best = {
                    "score": score,
                    "status": "READY_TO_EXECUTE",
                    "prototype_id": target.prototype_id,
                    "recommended_action": int(target.action),
                    "visual_key": k_tgt,
                    "visual_sim": sim_tgt,
                    "expected_return": chain_value,
                    "marginal_value": self._trust(target),
                    "chain_value": chain_value,
                    "total_cost": total_cost,
                    "ucb_bonus": self._ucb_bonus(target),
                    "success_prob": float(target.success_probability),
                    "confidence": float(target.confidence),
                    "effect_delta": list(target.effect_delta),
                    "chain": [
                        {"card_id": link["card_id"], "action": int(link["action"]), "applications": int(link["applications"])}
                        for link in chain
                    ],
                }
                # If the first executable link is NOT the target, surface it as the
                # harvesting step (legacy MPC keys).
                first = chain[0]
                if first["card_id"] != target.prototype_id:
                    first_card = first["card"]
                    k_prod = torch.tensor(first_card.precondition_key, dtype=torch.float32, device=self.device)
                    best["status"] = "HARVESTING_INPUT"
                    best["recommended_action"] = int(first_card.action)
                    best["visual_key"] = k_prod
                    best["visual_sim"] = float(torch.dot(z_norm, k_prod).item())
                    best["producer_prototype_id"] = first_card.prototype_id
                    best["producer_action"] = int(first_card.action)
                    deficits = self._deficits(target, x_t)
                    if deficits:
                        best["missing_resource_dim"] = deficits[0][0]
                        best["deficit_amount"] = deficits[0][1]

        if best is None:
            return None
        best.pop("score", None)
        return best
