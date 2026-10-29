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
import numpy as np


KNOWN_FIXTURES: Dict[str, str] = {
    "place_table": "table_placed",
    "place_furnace": "furnace_placed",
    "place_plant": "plant_placed",
    "place_stone": "stone_placed",
}
REVERSE_FIXTURES: Dict[str, str] = {v: k for k, v in KNOWN_FIXTURES.items()}

SEMANTIC_MILESTONE_PRODUCTS: Dict[str, str] = {
    # World Fixtures
    "place_table": "table_placed",
    "place_furnace": "furnace_placed",
    "place_plant": "plant_placed",
    "place_stone": "stone_placed",
    # Tools & Weapons
    "make_wood_pickaxe": "wood_pickaxe",
    "make_stone_pickaxe": "stone_pickaxe",
    "make_iron_pickaxe": "iron_pickaxe",
    "make_wood_sword": "wood_sword",
    "make_stone_sword": "stone_sword",
    "make_iron_sword": "iron_sword",
    # Resources
    "collect_wood": "wood",
    "collect_stone": "stone",
    "collect_coal": "coal",
    "collect_iron": "iron",
    "collect_diamond": "diamond",
    "collect_drink": "drink",
    "collect_sapling": "sapling",
    "eat_cow": "food",
    "eat_plant": "food",
}


def get_fixture_alias(name: str) -> str:
    """Returns canonical alias for known spatial fixtures, or the name itself."""
    if name in KNOWN_FIXTURES:
        return KNOWN_FIXTURES[name]
    if name in REVERSE_FIXTURES:
        return REVERSE_FIXTURES[name]
    return name


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
    is_fixture: bool = False

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

    def find_producer(self, item_name: str) -> Optional[str]:
        """
        Finds the causal card that produces the specified resource, tool, or workstation.
        """
        if item_name in self.cards and self.cards[item_name].produces.get(item_name, 0) > 0:
            return item_name

        for name, card in self.cards.items():
            if card.produces.get(item_name, 0) > 0:
                return name
        return None

    def calculate_upstream_budget(
        self,
        target_goal: str,
        current_inventory: Optional[Dict[str, int]] = None,
        workstations_present: Optional[Set[str]] = None,
    ) -> Dict[str, int]:
        """
        Recursively aggregates required upstream consumable resources for target goal.
        If workstations_present or current_inventory satisfies a predecessor's products,
        we do not re-demand its construction materials.
        """
        if target_goal not in self.cards:
            return {}

        total_budget: Dict[str, int] = {}
        ws = workstations_present or set()
        inv = current_inventory or {}

        def _traverse(card_name: str, visited: Set[str]):
            if card_name in visited or card_name not in self.cards:
                return
            visited.add(card_name)
            card = self.cards[card_name]

            # If this is not the root target and its products are already satisfied
            # in current_inventory or workstations_present, stop traversing upstream
            if card_name != target_goal:
                for prod in card.produces:
                    if prod in ws or get_fixture_alias(prod) in ws or inv.get(prod, 0) > 0:
                        return

            # Add direct consumption
            for res, amt in card.consumes.items():
                total_budget[res] = total_budget.get(res, 0) + amt

            # Traverse predecessors
            for pred in card.predecessors:
                _traverse(pred, visited)

        _traverse(target_goal, set())
        return total_budget

    def get_card_depth(self, card_name: str, visited: Optional[Set[str]] = None) -> int:
        """
        Computes the topological tech-tree depth (longest causal dependency chain).
        Domain-agnostic DAG traversal: depth = 1 + max(depth(predecessors)).
        """
        if visited is None:
            visited = set()
        if card_name in visited or card_name not in self.cards:
            return 0
        visited.add(card_name)
        card = self.cards[card_name]

        preds = list(card.predecessors)
        if card.catalyst_needed:
            producer = self.find_producer(card.catalyst_needed)
            if producer and producer not in preds:
                preds.append(producer)

        if not preds:
            return 0
        return 1 + max(self.get_card_depth(p, visited.copy()) for p in preds)

    def select_affordance_frontier(
        self,
        current_inventory: Dict[str, int],
        workstations_present: Optional[Set[str]] = None,
    ) -> str:
        """
        Autonomous Affordance Frontier Selector (AAFS):
        Dynamically selects the optimal reachable milestone from the learned causal hypergraph.
        Scores candidate goals by reachability from current inventory, expected yield,
        epistemic curiosity, and topological tech-tree depth without any hardcoded goal strings.
        """
        if not self.cards:
            return "explore"

        ws = workstations_present or set()
        candidates: List[Tuple[float, str]] = []

        # Derived catalyst tools from hypergraph (items required as catalysts, not placed fixtures)
        catalyst_tools = set()
        for c in self.cards.values():
            if c.catalyst_needed:
                producer = self.find_producer(c.catalyst_needed)
                is_cat_fixture = (
                    c.catalyst_needed in ws
                    or (producer is not None and producer in self.cards and self.cards[producer].is_fixture)
                )
                if not is_cat_fixture:
                    catalyst_tools.add(c.catalyst_needed)

        for name, card in self.cards.items():
            # 1. Skip if card produces a catalyst tool that the agent already possesses
            if any(prod in catalyst_tools and current_inventory.get(prod, 0) > 0 for prod in card.produces):
                continue

            # 2. Skip if card only produces placed workstations that are already in the world
            produces_fixture = False
            fixture_already_placed = True
            for prod in card.produces:
                is_prod_fixture = (card.is_fixture or prod in ws)
                if is_prod_fixture:
                    produces_fixture = True
                    if prod not in ws:
                        fixture_already_placed = False
            if produces_fixture and fixture_already_placed:
                continue

            # 3. Zone of Proximal Development (ZPD) filter:
            # A candidate must not require a catalyst or predecessor tool whose own prerequisite catalyst is unmet.
            # E.g. place_stone requires collect_stone (needs pickaxe, which needs table). If table/pickaxe unmet, reject!
            zpd_unreachable = False
            if card.catalyst_needed:
                catalyst_satisfied = (
                    card.catalyst_needed in ws
                    or current_inventory.get(card.catalyst_needed, 0) > 0
                )
                if not catalyst_satisfied:
                    producer = self.find_producer(card.catalyst_needed)
                    if producer and producer in self.cards:
                        pred_card = self.cards[producer]
                        if pred_card.catalyst_needed:
                            pred_cat_satisfied = (
                                pred_card.catalyst_needed in ws
                                or current_inventory.get(pred_card.catalyst_needed, 0) > 0
                            )
                            if not pred_cat_satisfied:
                                zpd_unreachable = True

            if not zpd_unreachable:
                for pred in card.predecessors:
                    if pred in self.cards:
                        p_card = self.cards[pred]
                        if p_card.catalyst_needed:
                            p_cat_satisfied = (
                                p_card.catalyst_needed in ws
                                or current_inventory.get(p_card.catalyst_needed, 0) > 0
                            )
                            if not p_cat_satisfied:
                                p_producer = self.find_producer(p_card.catalyst_needed)
                                if p_producer and p_producer in self.cards:
                                    pp_card = self.cards[p_producer]
                                    if pp_card.catalyst_needed and (
                                        pp_card.catalyst_needed not in ws
                                        and current_inventory.get(pp_card.catalyst_needed, 0) == 0
                                    ):
                                        zpd_unreachable = True
                                        break
            if zpd_unreachable:
                continue

            # 4. Calculate upstream budget and resource deficit from current inventory
            budget = self.calculate_upstream_budget(
                name, current_inventory=current_inventory, workstations_present=ws
            )
            deficit = sum(max(0, req - current_inventory.get(res, 0)) for res, req in budget.items())

            # 5. Multi-objective scoring
            # Reachability: exp(-0.5 * deficit), in (0.0, 1.0]
            reachability = float(np.exp(-0.5 * deficit))

            # Expected yield from historical trials
            total_trials = card.success_count + card.failure_count
            p_success = (card.success_count / (total_trials + 1e-5)) if total_trials > 0 else 0.5

            # Epistemic curiosity: higher score for unmastered or newly discovered cards
            curiosity = 1.0 - card.confidence

            # Topological depth bonus: deeper tech cards in the causal DAG provide higher long-term utility
            depth = self.get_card_depth(name)
            depth_bonus = 0.5 * depth

            score = 2.0 * reachability + 1.0 * p_success + 0.5 * curiosity + depth_bonus
            candidates.append((score, name))

        if not candidates:
            return "explore"

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

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

        if not target_goal or target_goal == "explore" or target_goal in visited or target_goal not in self.cards:
            return ("explore", 0)

        visited.add(target_goal)
        card = self.cards[target_goal]

        cat_needed = card.catalyst_needed

        # 1. Check if catalyst is satisfied (either in workstations_present or in current_inventory)
        if cat_needed:
            catalyst_satisfied = (
                cat_needed in workstations_present
                or get_fixture_alias(cat_needed) in workstations_present
                or current_inventory.get(cat_needed, 0) > 0
                or current_inventory.get(get_fixture_alias(cat_needed), 0) > 0
            )
            if not catalyst_satisfied:
                # Need to satisfy catalyst first. Locate producer card.
                producer = None
                for pred in card.predecessors:
                    pred_card = self.cards.get(pred)
                    if pred_card and (
                        pred_card.produces.get(cat_needed, 0) > 0
                        or pred_card.produces.get(get_fixture_alias(cat_needed), 0) > 0
                    ):
                        producer = pred
                        break
                if producer is None:
                    producer = self.find_producer(cat_needed) or self.find_producer(get_fixture_alias(cat_needed))

                if producer:
                    # Check upstream budget for target_goal taking present workstations into account
                    needed_res = self.calculate_upstream_budget(target_goal, workstations_present=workstations_present)
                    for res, req_amt in needed_res.items():
                        if current_inventory.get(res, 0) < req_amt:
                            # Not enough total resources! Collect more inputs first.
                            res_producer = self.find_producer(res)
                            if res_producer and res_producer in self.cards and res_producer not in visited:
                                return self.determine_subgoal(
                                    res_producer, current_inventory, workstations_present, visited
                                )
                            elif res_producer and res_producer in self.cards:
                                return (res_producer, self.cards[res_producer].action)
                            return ("explore", 0)

                    # We hold total aggregated budget! Recursively resolve producer subgoal.
                    return self.determine_subgoal(
                        producer, current_inventory, workstations_present, visited
                    )

        # 2. Check direct consumable inventory
        for res, amt in card.consumes.items():
            if current_inventory.get(res, 0) < amt:
                res_producer = self.find_producer(res)
                if res_producer and res_producer in self.cards and res_producer not in visited:
                    return self.determine_subgoal(
                        res_producer, current_inventory, workstations_present, visited
                    )
                elif res_producer and res_producer in self.cards:
                    return (res_producer, self.cards[res_producer].action)
                return ("explore", 0)

        # 3. All preconditions met -> execute target action!
        return (target_goal, card.action)
