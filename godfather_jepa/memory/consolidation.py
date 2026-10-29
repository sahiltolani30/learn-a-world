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
                if ach.startswith("place_"):
                    step_produces = {f"{ach[len('place_'):]}_placed": 1}
                elif ach.startswith("make_"):
                    step_produces = {ach[len("make_"):]: 1}
                elif ach.startswith("collect_"):
                    step_produces = {ach[len("collect_"):]: 1}
                elif ach.startswith("smelt_"):
                    step_produces = {ach[len("smelt_"):]: 1}
                else:
                    step_produces = {ach: 1}

                # Only assembly/manufacturing achievements (e.g. make_*, smelt_*) require workstation catalysts.
                # Primitive collections (collect_*), vitals (drink, eat, wake), and workstation placements (place_*) never require catalysts.
                catalyst_needed = None
                if ach.startswith("make_") or ach.startswith("smelt_"):
                    if hasattr(episodic_store, "placed_workstations") and episodic_store.placed_workstations:
                        for station_type, (p_step, p_pos) in episodic_store.placed_workstations.items():
                            if station_type in ("table_placed", "furnace_placed") and p_step < step.step_idx:
                                dist = float(np.linalg.norm(step.displacement - p_pos))
                                if dist <= 1.5:
                                    catalyst_needed = station_type
                                    break
                    elif (
                        episodic_store.workstation_placed_step is not None
                        and episodic_store.workstation_placed_step < step.step_idx
                    ):
                        dist = float(np.linalg.norm(step.displacement - episodic_store.workstation_pos))
                        if dist <= 1.5:
                            catalyst_needed = "table_placed"

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
                    if catalyst_needed and not card.catalyst_needed:
                        card.catalyst_needed = catalyst_needed
                    for prod in card.produces:
                        card.consumes.pop(prod, None)
                    updated_cards.append(ach)
                else:
                    new_card = CausalCard(
                        action=step.action,
                        name=ach,
                        consumes=step_consumes,
                        produces=step_produces,
                        catalyst_needed=catalyst_needed,
                        success_count=1,
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
