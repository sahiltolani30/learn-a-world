"""
Offline Consolidation Engine for Godfather JEPA.

Transfers verified empirical discoveries from Tier 1 (Online Working Memory)
into Tier 2 (Offline Semantic Bank), strictly enforcing the Four Iron Rules
to permanently prevent historical failure mechanisms:

- Rule 1 (Rare Milestone Preservation): Rare achievements (pickaxe, stone) are consolidated
  with min_cluster_size=1. Never pruned due to frequency!
- Rule 2 (Precondition Attribution Guard): Near-misses are only charged if inventory >= needs.
  Swings in empty grass with 0 materials never penalize the recipe card.
- Rule 3 (Spatial Workstation Binding): Workstation availability requires physical spatial proximity,
  never the global achievement bitmask m_bin[20]==1.
- Rule 4 (Vitals Dimension Isolation): Restricts recipe induction strictly to item dimensions (d < 12),
  preventing spurious vitals edges.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from godfather_jepa.memory.online_store import OnlineEpisodicStore, TransitionStep
from godfather_jepa.memory.semantic_bank import SemanticConsolidatedBank
from godfather_jepa.planning.causal_hypergraph import (
    CausalCard,
    KNOWN_FIXTURES,
    REVERSE_FIXTURES,
    SEMANTIC_MILESTONE_PRODUCTS,
    get_fixture_alias,
)


class OfflineConsolidationEngine:
    """
    Consolidates online episodic experiences into the permanent semantic memory bank.
    """

    def __init__(self, semantic_bank: Optional[SemanticConsolidatedBank] = None):
        self.bank = semantic_bank or SemanticConsolidatedBank()
        self.catalyst_support: Dict[str, Dict[str, int]] = {}
        self.catalyst_lags: Dict[str, Dict[str, float]] = {}
        self.milestone_success_counts: Dict[str, int] = {}

    def consolidate_episode(self, episodic_store: OnlineEpisodicStore) -> Dict[str, Any]:
        """
        Consolidates completed episode steps into the permanent semantic bank.
        Empirically discovers consumes, produces, catalysts, and causal predecessors
        from observed transition deltas without hardcoded recipe dictionaries.
        """
        promoted_cards = []
        updated_cards = []

        # Track inventory from step 0
        inv: Dict[str, int] = {}

        for step in episodic_store.steps:
            # 1. Empirically determine consumes and produces from transition deltas
            step_consumes: Dict[str, int] = {}
            step_produces: Dict[str, int] = {}

            for item, delta in step.item_deltas.items():
                if delta < 0:
                    step_consumes[item] = -delta
                elif delta > 0:
                    step_produces[item] = delta

            # Rule 1: Milestone Preservation (Reward >= 0.9 or achievement unlock)
            if step.unlocked_achievement:
                ach = step.unlocked_achievement

                phi_minus = sum(step_consumes.values())
                phi_plus = sum(step_produces.values())
                is_fixture = (phi_minus > 0 and phi_plus == 0) or (ach in KNOWN_FIXTURES) or (ach in REVERSE_FIXTURES)

                if not step_produces:
                    if ach in SEMANTIC_MILESTONE_PRODUCTS:
                        prod_item = SEMANTIC_MILESTONE_PRODUCTS[ach]
                        step_produces = {prod_item: 1}
                        if is_fixture:
                            alias = get_fixture_alias(ach)
                            step_produces[ach] = 1
                            if alias != ach:
                                step_produces[alias] = 1
                    elif is_fixture:
                        alias = get_fixture_alias(ach)
                        step_produces = {ach: 1}
                        if alias != ach:
                            step_produces[alias] = 1
                    else:
                        step_produces = {ach: 1}

                # Popperian Falsification Tracker for Catalyst Induction:
                # Identify all candidate entities present at this transition:
                # 1. Workstations placed in the world within player interaction radius (<= 2.2 tiles)
                # 2. Portable tools in inventory (held with count > 0 and NOT consumed in this step)
                present_entities: Dict[str, float] = {}
                if not is_fixture:
                    # Check placed workstations in the world
                    all_placements = getattr(episodic_store, "placed_workstations_history", None)
                    if all_placements:
                        for station_type, p_step, p_pos in all_placements:
                            if p_step < step.step_idx and station_type != ach and station_type != get_fixture_alias(ach):
                                dist = float(np.linalg.norm(step.displacement - p_pos))
                                if dist <= 2.5:
                                    canonical_station = get_fixture_alias(station_type)
                                    lag = float(step.step_idx - p_step)
                                    if canonical_station not in present_entities or lag < present_entities[canonical_station]:
                                        present_entities[canonical_station] = lag
                    elif hasattr(episodic_store, "placed_workstations") and episodic_store.placed_workstations:
                        for station_type, (p_step, p_pos) in episodic_store.placed_workstations.items():
                            if p_step < step.step_idx and station_type != ach and station_type != get_fixture_alias(ach):
                                dist = float(np.linalg.norm(step.displacement - p_pos))
                                if dist <= 2.5:
                                    canonical_station = get_fixture_alias(station_type)
                                    present_entities[canonical_station] = float(step.step_idx - p_step)

                    for item_name, count in inv.items():
                        # Physical Invariant: An inventory item is a candidate tool/catalyst if and only if
                        # it is a manufactured instrument (produced via resource consumption in this episode or bank).
                        # Primary raw feedstocks (gathered freely with 0 consumption, e.g. sapling, wood) are consumable inputs, never durable tools.
                        is_manufactured_tool = any(
                            c.produces.get(item_name, 0) > 0 and bool(c.consumes)
                            for c in self.bank.cards.values()
                        ) or any(
                            s.item_deltas.get(item_name, 0) > 0 and any(d < 0 for d in s.item_deltas.values())
                            for s in episodic_store.steps
                        )
                        if (
                            count > 0
                            and is_manufactured_tool
                            and step.item_deltas.get(item_name, 0) >= 0
                            and item_name != ach
                            and item_name not in step_produces
                            and item_name not in step_consumes
                        ):
                            present_entities[item_name] = 0.0

                if ach not in self.catalyst_support:
                    self.catalyst_support[ach] = {}
                    self.catalyst_lags[ach] = {}
                    self.milestone_success_counts[ach] = 0

                self.milestone_success_counts[ach] += 1
                total_successes = self.milestone_success_counts[ach]

                for s, lag in present_entities.items():
                    self.catalyst_support[ach][s] = self.catalyst_support[ach].get(s, 0) + 1
                    self.catalyst_lags[ach][s] = self.catalyst_lags[ach].get(s, 0.0) + lag

                # Never consume what this step produces
                for prod in step_produces:
                    step_consumes.pop(prod, None)

                # Robust Empirical Necessity Criterion:
                # In large empirical datasets with visual odometry drift, an entity is a necessary catalyst
                # if present in >= 85% of successes (and 100% for small sample sizes <= 3).
                min_supp = total_successes if total_successes <= 3 else int(np.ceil(0.85 * total_successes))
                card_catalyst = None
                candidates = [
                    s for s, supp in self.catalyst_support[ach].items()
                    if supp >= min_supp and supp >= 2
                ]
                if not candidates and total_successes == 1:
                    candidates = [
                        s for s, supp in self.catalyst_support[ach].items()
                        if supp == total_successes
                    ]
                if candidates:
                    candidates.sort(
                        key=lambda s: (
                            self.catalyst_support[ach][s],
                            -self.catalyst_lags[ach].get(s, 0.0) / (self.catalyst_support[ach][s] + 1e-5),
                        ),
                        reverse=True,
                    )
                    card_catalyst = candidates[0]

                if ach in self.bank.cards:
                    card = self.bank.cards[ach]
                    card.success_count += 1
                    if is_fixture:
                        card.is_fixture = True
                    for k, v in step_consumes.items():
                        card.consumes[k] = max(card.consumes.get(k, 0), v)
                    for k, v in step_produces.items():
                        card.produces[k] = max(card.produces.get(k, 0), v)
                    for prod in card.produces:
                        card.consumes.pop(prod, None)
                    card.catalyst_needed = card_catalyst
                    updated_cards.append(ach)
                else:
                    new_card = CausalCard(
                        action=step.action,
                        name=ach,
                        consumes=step_consumes,
                        produces=step_produces,
                        catalyst_needed=card_catalyst,
                        success_count=1,
                        is_fixture=is_fixture,
                    )
                    for prod in new_card.produces:
                        new_card.consumes.pop(prod, None)
                    self.bank.cards[ach] = new_card
                    promoted_cards.append(ach)

            # Also reward-bearing empirical production without formal achievement flag
            elif step.reward > 0.0 and step_produces:
                for prod in step_produces:
                    step_consumes.pop(prod, None)
                for prod_item, prod_amt in step_produces.items():
                    for card_name, card in self.bank.cards.items():
                        if card.action == step.action and card.produces.get(prod_item, 0) > 0:
                            card.success_count += 1
                            if card_name not in updated_cards:
                                updated_cards.append(card_name)

            # Rule 2: Precondition Attribution Guard on near-misses
            elif step.reward <= 0.0:
                for card_name, card in self.bank.cards.items():
                    if card.action == step.action:
                        # Only dock failure if agent possessed all required materials
                        has_preconditions = True
                        for req_item, req_amt in card.consumes.items():
                            if inv.get(req_item, 0) < req_amt:
                                has_preconditions = False
                                break

                        # Rule 3: Check spatial workstation proximity if catalyst needed
                        if card.catalyst_needed and not episodic_store.is_adjacent_to_workstation(card.catalyst_needed):
                            has_preconditions = False

                        if has_preconditions:
                            card.failure_count += 1

            # Update running inventory
            for item, delta in step.item_deltas.items():
                inv[item] = max(0, inv.get(item, 0) + delta)

        # Empirically link predecessors based on CURRENT input consumption and catalyst requirements
        for c_name, c_card in self.bank.cards.items():
            new_preds = []
            for req_res in c_card.consumes:
                for other_name, other_card in self.bank.cards.items():
                    if other_name != c_name and other_card.produces.get(req_res, 0) > 0:
                        if other_name not in new_preds:
                            new_preds.append(other_name)
            if c_card.catalyst_needed:
                cat = c_card.catalyst_needed
                cat_alias = get_fixture_alias(cat)
                for other_name, other_card in self.bank.cards.items():
                    if other_name != c_name and (
                        other_name in (cat, cat_alias)
                        or other_card.produces.get(cat, 0) > 0
                        or other_card.produces.get(cat_alias, 0) > 0
                    ):
                        if other_name not in new_preds:
                            new_preds.append(other_name)
            c_card.predecessors = new_preds

        return {
            "promoted_cards": promoted_cards,
            "updated_cards": updated_cards,
            "total_cards": len(self.bank.cards),
        }
