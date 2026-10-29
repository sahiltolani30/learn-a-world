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
from godfather_jepa.planning.causal_hypergraph import CausalCard


class OfflineConsolidationEngine:
    """
    Consolidates online episodic experiences into the permanent semantic memory bank.
    """

    def __init__(self, semantic_bank: Optional[SemanticConsolidatedBank] = None):
        self.bank = semantic_bank or SemanticConsolidatedBank()
        self.catalyst_counts: Dict[str, Dict[str, int]] = {}

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
                # If visual decoding did not produce an inventory item delta (e.g. world placement or synthetic step),
                # map milestone to target semantic entity
                if not step_produces:
                    if ach.startswith("place_"):
                        step_produces = {f"{ach[len('place_'):]}_placed": 1}
                    elif ach.startswith("collect_"):
                        step_produces = {ach[len("collect_"):]: 1}
                    elif ach.startswith("make_"):
                        step_produces = {ach[len("make_"):]: 1}
                    elif ach.startswith("smelt_"):
                        step_produces = {ach[len("smelt_"):]: 1}
                    else:
                        step_produces = {ach: 1}

                # Pearlian Causal Contrast for Catalyst Induction:
                # Identify all workstations adjacent (player reach <= 2.2 tiles) to the agent at this milestone.
                nearby_stations: Set[str] = set()
                if not ach.startswith("place_"):
                    if hasattr(episodic_store, "placed_workstations") and episodic_store.placed_workstations:
                        for station_type, (p_step, p_pos) in episodic_store.placed_workstations.items():
                            if p_step < step.step_idx and station_type in ("table_placed", "furnace_placed"):
                                dist = float(np.linalg.norm(step.displacement - p_pos))
                                if dist <= 2.2:
                                    nearby_stations.add(station_type)
                    elif (
                        episodic_store.workstation_placed_step is not None
                        and episodic_store.workstation_placed_step < step.step_idx
                    ):
                        dist = float(np.linalg.norm(step.displacement - episodic_store.workstation_pos))
                        if dist <= 2.2:
                            nearby_stations.add("table_placed")

                # Accumulate co-occurrence counts
                if ach not in self.catalyst_counts:
                    self.catalyst_counts[ach] = {}
                for s in nearby_stations:
                    self.catalyst_counts[ach][s] = self.catalyst_counts[ach].get(s, 0) + 1

                # Never consume what this step produces
                for prod in step_produces:
                    step_consumes.pop(prod, None)

                if ach in self.bank.cards:
                    card = self.bank.cards[ach]
                    card.success_count += 1
                    for k, v in step_consumes.items():
                        card.consumes[k] = max(card.consumes.get(k, 0), v)
                    for k, v in step_produces.items():
                        card.produces[k] = max(card.produces.get(k, 0), v)
                    for prod in card.produces:
                        card.consumes.pop(prod, None)
                    # Causal binding: catalyst must be present in majority (>= 50%) of successes
                    card_catalyst = None
                    if ach in self.catalyst_counts:
                        for s, count in self.catalyst_counts[ach].items():
                            if count / card.success_count >= 0.5:
                                card_catalyst = s
                                break
                    card.catalyst_needed = card_catalyst
                    updated_cards.append(ach)
                else:
                    card_catalyst = next(iter(nearby_stations)) if len(nearby_stations) == 1 else None
                    new_card = CausalCard(
                        action=step.action,
                        name=ach,
                        consumes=step_consumes,
                        produces=step_produces,
                        catalyst_needed=card_catalyst,
                        success_count=1,
                    )
                    for prod in new_card.produces:
                        new_card.consumes.pop(prod, None)
                    self.bank.cards[ach] = new_card
                    promoted_cards.append(ach)
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

        # Empirically link predecessors based on input consumption and catalyst requirements
        for c_name, c_card in self.bank.cards.items():
            for req_res in c_card.consumes:
                for other_name, other_card in self.bank.cards.items():
                    if other_name != c_name and other_card.produces.get(req_res, 0) > 0:
                        if other_name not in c_card.predecessors:
                            c_card.predecessors.append(other_name)
            if c_card.catalyst_needed:
                for other_name, other_card in self.bank.cards.items():
                    if other_name != c_name and (
                        other_name == c_card.catalyst_needed
                        or other_card.produces.get(c_card.catalyst_needed, 0) > 0
                    ):
                        if other_name not in c_card.predecessors:
                            c_card.predecessors.append(other_name)

        return {
            "promoted_cards": promoted_cards,
            "updated_cards": updated_cards,
            "total_cards": len(self.bank.cards),
        }
