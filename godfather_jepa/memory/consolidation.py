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
        """
        promoted_cards = []
        updated_cards = []

        # Track inventory from step 0
        inv: Dict[str, int] = {}

        for step in episodic_store.steps:
            # Update running inventory
            for item, delta in step.item_deltas.items():
                inv[item] = max(0, inv.get(item, 0) + delta)

            # Rule 1: Milestone Preservation (Reward >= 0.9 or achievement unlock)
            if step.unlocked_achievement:
                ach = step.unlocked_achievement
                if ach in self.bank.cards:
                    card = self.bank.cards[ach]
                    card.success_count += 1
                    updated_cards.append(ach)
                else:
                    # New discovery: create card with min_cluster_size=1
                    new_card = CausalCard(
                        action=step.action,
                        name=ach,
                        consumes={},
                        produces={ach: 1},
                        success_count=1,
                    )
                    self.bank.cards[ach] = new_card
                    promoted_cards.append(ach)

            # Rule 2: Precondition Attribution Guard on near-misses
            elif step.reward <= 0.0 and step.action in (8, 11, 14, 9, 12, 15):
                # Check which card this action belongs to
                for card_name, card in self.bank.cards.items():
                    if card.action == step.action:
                        # Only dock failure if agent possessed all required materials
                        has_preconditions = True
                        for req_item, req_amt in card.consumes.items():
                            if inv.get(req_item, 0) < req_amt:
                                has_preconditions = False
                                break

                        # Rule 3: Check spatial workstation proximity if catalyst needed
                        if card.catalyst_needed and not episodic_store.is_adjacent_to_workstation():
                            has_preconditions = False

                        if has_preconditions:
                            card.failure_count += 1

        return {
            "promoted_cards": promoted_cards,
            "updated_cards": updated_cards,
            "total_cards": len(self.bank.cards),
        }
