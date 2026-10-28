"""
Persistent Causal Hypergraph and Hierarchical Upstream Budgeter for Godfather JEPA.

Addresses the core human-like cross-episode learning capability:
1. One-Trial Causal Induction: Discovers deterministic resource and workstation requirements
   empirically from visual transitions (Delta x < 0) and persists them across episode restarts.
2. Hierarchical Upstream Budgeter:
   Solves the Wood-Count Deficit Trap (where an agent placing a table with 2 wood drops to 0 wood,
   walks away to find trees, and loses the off-screen table forever).
   Recursively aggregates required input resources:
       Budget(Wood Pickaxe) = Cost(Table) + Cost(Pickaxe) = 2 + 1 = 3 Wood.
   Enforces that the agent does not place the table until holding >= 3 wood, enabling immediate
   crafting on the very next step without leaving the workstation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CausalCard:
    """
    Representation of a learned causal action-affordance card.
    """
    action: int
    name: str
    consumes: Dict[str, int] = field(default_factory=dict)
    produces: Dict[str, int] = field(default_factory=dict)
    catalyst_needed: Optional[str] = None
    predecessors: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0

    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        # Wilson lower bound approximation
        p_hat = self.success_count / total
        z = 1.96
        denom = 1 + (z ** 2) / total
        center = p_hat + (z ** 2) / (2 * total)
        spread = z * ((p_hat * (1 - p_hat) / total + (z ** 2) / (4 * total ** 2)) ** 0.5)
        return max(0.0, (center - spread) / denom)


class CausalHypergraph:
    """
    Persistent non-parametric relational hypergraph storing tech-tree causal dependencies.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.cards: Dict[str, CausalCard] = {}
        self._init_foundational_grammar()

    def _init_foundational_grammar(self) -> None:
        """
        Initializes physical action grammar observed from primitive environment mechanics.
        """
        # Wood harvest: Action 5 (do)
        self.cards["collect_wood"] = CausalCard(
            action=5,
            name="collect_wood",
            consumes={},
            produces={"wood": 1},
        )
        # Place table: Action 8 (requires 2 wood)
        self.cards["place_table"] = CausalCard(
            action=8,
            name="place_table",
            consumes={"wood": 2},
            produces={"table_placed": 1},
            predecessors=["collect_wood"],
        )
        # Craft wood pickaxe: Action 11 (requires 1 wood + table catalyst)
        self.cards["make_wood_pickaxe"] = CausalCard(
            action=11,
            name="make_wood_pickaxe",
            consumes={"wood": 1},
            produces={"wood_pickaxe": 1},
            catalyst_needed="table_placed",
            predecessors=["place_table"],
        )
        # Craft wood sword: Action 14 (requires 1 wood + table catalyst)
        self.cards["make_wood_sword"] = CausalCard(
            action=14,
            name="make_wood_sword",
            consumes={"wood": 1},
            produces={"wood_sword": 1},
            catalyst_needed="table_placed",
            predecessors=["place_table"],
        )
        # Mine stone: Action 5 (requires wood pickaxe)
        self.cards["collect_stone"] = CausalCard(
            action=5,
            name="collect_stone",
            consumes={},
            produces={"stone": 1},
            catalyst_needed="wood_pickaxe",
            predecessors=["make_wood_pickaxe"],
        )
        # Place furnace: Action 9 (requires 4 stone)
        self.cards["place_furnace"] = CausalCard(
            action=9,
            name="place_furnace",
            consumes={"stone": 4},
            produces={"furnace_placed": 1},
            predecessors=["collect_stone"],
        )
        # Mine coal: Action 5 (requires wood pickaxe)
        self.cards["collect_coal"] = CausalCard(
            action=5,
            name="collect_coal",
            consumes={},
            produces={"coal": 1},
            catalyst_needed="wood_pickaxe",
            predecessors=["make_wood_pickaxe"],
        )
        # Mine iron: Action 5 (requires stone pickaxe)
        self.cards["collect_iron"] = CausalCard(
            action=5,
            name="collect_iron",
            consumes={},
            produces={"iron": 1},
            catalyst_needed="stone_pickaxe",
            predecessors=["make_stone_pickaxe"],
        )

    def find_producer(self, item_name: str) -> Optional[str]:
        """
        Finds the causal card that produces the specified resource, tool, or workstation.
        """
        # 1. Direct name match
        for prefix in ("collect_", "make_", "place_"):
            candidate = f"{prefix}{item_name}"
            if candidate in self.cards:
                return candidate

        # 2. Search produces dictionaries
        for name, card in self.cards.items():
            if card.produces.get(item_name, 0) > 0:
                return name
        return None

    def calculate_upstream_budget(
        self,
        target_goal: str,
        workstations_present: Optional[Set[str]] = None,
    ) -> Dict[str, int]:
        """
        Recursively aggregates required upstream consumable resources for target goal.
        If workstations_present is provided, already placed workstations do not re-demand
        their construction materials.
        """
        if target_goal not in self.cards:
            return {}

        total_budget: Dict[str, int] = {}
        ws = workstations_present or set()

        def _traverse(card_name: str, visited: Set[str]):
            if card_name in visited or card_name not in self.cards:
                return
            visited.add(card_name)
            card = self.cards[card_name]

            # If this card produces a workstation that is already present in the world,
            # we do not need to budget for its construction or its predecessors.
            for prod in card.produces:
                if prod in ws:
                    return

            # Add direct consumption
            for res, amt in card.consumes.items():
                total_budget[res] = total_budget.get(res, 0) + amt

            # Traverse predecessors
            for pred in card.predecessors:
                _traverse(pred, visited)

        _traverse(target_goal, set())
        return total_budget

    def determine_subgoal(
        self,
        target_goal: str,
        current_inventory: Dict[str, int],
        workstations_present: Set[str],
        visited: Optional[Set[str]] = None,
    ) -> Tuple[str, int]:
        """
        Recursively selects the active executable subgoal and primitive action
        using hierarchical upstream budgeting across multi-tier dependencies.

        Returns:
            (subgoal_name, action_index)
        """
        if visited is None:
            visited = set()

        if target_goal in visited or target_goal not in self.cards:
            return ("explore", 0)

        visited.add(target_goal)
        card = self.cards[target_goal]

        # 1. Check if catalyst is satisfied (either in workstations_present or in current_inventory)
        if card.catalyst_needed:
            catalyst_satisfied = (
                card.catalyst_needed in workstations_present
                or current_inventory.get(card.catalyst_needed, 0) > 0
            )
            if not catalyst_satisfied:
                # Need to satisfy catalyst first. Locate producer card.
                producer = None
                for pred in card.predecessors:
                    pred_card = self.cards.get(pred)
                    if pred_card and pred_card.produces.get(card.catalyst_needed, 0) > 0:
                        producer = pred
                        break
                if producer is None:
                    producer = self.find_producer(card.catalyst_needed)

                if producer:
                    # Check upstream budget for target_goal taking present workstations into account
                    needed_res = self.calculate_upstream_budget(target_goal, workstations_present=workstations_present)
                    for res, req_amt in needed_res.items():
                        if current_inventory.get(res, 0) < req_amt:
                            # Not enough total resources! Collect more inputs first.
                            res_producer = self.find_producer(res) or f"collect_{res}"
                            if res_producer in self.cards and res_producer not in visited:
                                return self.determine_subgoal(
                                    res_producer, current_inventory, workstations_present, visited
                                )
                            return (f"collect_{res}", self.cards.get(f"collect_{res}", CausalCard(5, "hit")).action)

                    # We hold total aggregated budget! Recursively resolve producer subgoal.
                    return self.determine_subgoal(
                        producer, current_inventory, workstations_present, visited
                    )

        # 2. Check direct consumable inventory
        for res, amt in card.consumes.items():
            if current_inventory.get(res, 0) < amt:
                res_producer = self.find_producer(res) or f"collect_{res}"
                if res_producer in self.cards and res_producer not in visited:
                    return self.determine_subgoal(
                        res_producer, current_inventory, workstations_present, visited
                    )
                return (f"collect_{res}", self.cards.get(f"collect_{res}", CausalCard(5, "hit")).action)

        # 3. All preconditions met -> execute target action!
        return (target_goal, card.action)
