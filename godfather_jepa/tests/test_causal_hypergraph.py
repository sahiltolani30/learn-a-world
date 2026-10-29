"""
Adversarial Empirical Challenge Suite for Requirement R3:
Causal Hypergraph & Hierarchical Upstream Budgeter (godfather_jepa/planning/causal_hypergraph.py).

Verifies:
1. Exact upstream wood budgeting edge cases: 0, 1, 2, 3, 4 wood.
2. Strict blocking of table placement (action 8) when holding < 3 wood for pickaxe target.
3. Recursive tree resolution: Budget(Wood Pickaxe) = 3 Wood.
4. Causal card creation, confidence dynamics, and cross-episode persistence.
5. Adversarial edge cases: cycles, missing nodes, deep catalyst chains, and prerequisite bypasses.
"""
import copy
import pickle
import pytest
from typing import Dict, Set

from godfather_jepa.planning.causal_hypergraph import CausalHypergraph, CausalCard


class TestCausalCardProperties:
    """Adversarial stress testing of the CausalCard dataclass and confidence formula."""

    def test_default_initialization(self):
        card = CausalCard(action=5, name="test_action")
        assert card.action == 5
        assert card.name == "test_action"
        assert card.consumes == {}
        assert card.produces == {}
        assert card.catalyst_needed is None
        assert card.predecessors == []
        assert card.success_count == 0
        assert card.failure_count == 0
        assert card.confidence == 0.0

    def test_confidence_zero_trials(self):
        card = CausalCard(action=1, name="dummy", success_count=0, failure_count=0)
        assert card.confidence == 0.0

    def test_confidence_all_failures(self):
        card = CausalCard(action=1, name="dummy", success_count=0, failure_count=100)
        assert card.confidence == 0.0

    def test_confidence_monotonic_with_successes(self):
        confidences = []
        for successes in [1, 5, 10, 50, 100, 500]:
            card = CausalCard(action=1, name="dummy", success_count=successes, failure_count=0)
            confidences.append(card.confidence)
        # Strictly monotonically increasing
        for i in range(len(confidences) - 1):
            assert confidences[i] < confidences[i + 1]
            assert 0.0 < confidences[i] < 1.0

    def test_confidence_monotonic_decrease_with_failures(self):
        confidences = []
        for failures in [0, 2, 10, 50, 200]:
            card = CausalCard(action=1, name="dummy", success_count=20, failure_count=failures)
            confidences.append(card.confidence)
        # Strictly monotonically decreasing
        for i in range(len(confidences) - 1):
            assert confidences[i] > confidences[i + 1]


def _build_test_graph() -> CausalHypergraph:
    graph = CausalHypergraph()
    graph.cards["collect_wood"] = CausalCard(
        action=5, name="collect_wood", consumes={}, produces={"wood": 1}
    )
    graph.cards["place_table"] = CausalCard(
        action=8, name="place_table", consumes={"wood": 2}, produces={"table_placed": 1, "place_table": 1}, predecessors=["collect_wood"]
    )
    graph.cards["make_wood_pickaxe"] = CausalCard(
        action=11, name="make_wood_pickaxe", consumes={"wood": 1}, produces={"wood_pickaxe": 1}, catalyst_needed="table_placed", predecessors=["place_table"]
    )
    graph.cards["make_wood_sword"] = CausalCard(
        action=14, name="make_wood_sword", consumes={"wood": 1}, produces={"wood_sword": 1}, catalyst_needed="table_placed", predecessors=["place_table"]
    )
    graph.cards["collect_stone"] = CausalCard(
        action=5, name="collect_stone", consumes={}, produces={"stone": 1}, catalyst_needed="wood_pickaxe", predecessors=["make_wood_pickaxe"]
    )
    graph.cards["place_furnace"] = CausalCard(
        action=9, name="place_furnace", consumes={"stone": 4}, produces={"furnace_placed": 1}, predecessors=["collect_stone"]
    )
    return graph


class TestHierarchicalWoodBudgeting:
    """
    Empirically verifies the Hierarchical Upstream Budgeter.
    Solves the Wood-Count Deficit Trap: agent holding < 3 wood must NOT place table.
    """

    def test_tabula_rasa_init_clean(self):
        graph = CausalHypergraph()
        assert len(graph.cards) == 0

    def test_recursive_upstream_budget_wood_pickaxe(self):
        """Budget(Wood Pickaxe) = Cost(Table)[2 wood] + Cost(Pickaxe)[1 wood] = 3 Wood."""
        graph = _build_test_graph()
        budget = graph.calculate_upstream_budget("make_wood_pickaxe")
        assert budget == {"wood": 3}

    def test_recursive_upstream_budget_wood_sword(self):
        """Budget(Wood Sword) = Cost(Table)[2 wood] + Cost(Sword)[1 wood] = 3 Wood."""
        graph = _build_test_graph()
        budget = graph.calculate_upstream_budget("make_wood_sword")
        assert budget == {"wood": 3}

    def test_direct_table_budget(self):
        """Budget(Place Table) = 2 Wood."""
        graph = _build_test_graph()
        budget = graph.calculate_upstream_budget("place_table")
        assert budget == {"wood": 2}

    @pytest.mark.parametrize("wood_count,expected_subgoal,expected_action,is_table_blocked", [
        (-2, "collect_wood", 5, True),
        (-1, "collect_wood", 5, True),
        (0, "collect_wood", 5, True),
        (1, "collect_wood", 5, True),
        (2, "collect_wood", 5, True),   # THE CRUCIAL DEFICIT TRAP CHECK: 2 wood must NOT place table!
        (3, "place_table", 8, False),   # Budget satisfied: Table placement unlocked
        (4, "place_table", 8, False),   # Surplus wood: Table placement unlocked
        (99, "place_table", 8, False),
    ])
    def test_wood_budgeting_edge_cases(
        self, wood_count: int, expected_subgoal: str, expected_action: int, is_table_blocked: bool
    ):
        """
        Adversarially sweep wood count from negative through surplus (0, 1, 2, 3, 4).
        Assert that when holding < 3 wood, table placement (action 8) is strictly blocked.
        """
        graph = _build_test_graph()
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={"wood": wood_count},
            workstations_present=set(),
        )
        assert subgoal == expected_subgoal
        assert action == expected_action
        if is_table_blocked:
            assert action != 8, f"Table placement (action 8) was NOT blocked at wood={wood_count}!"
        else:
            assert action == 8, f"Table placement (action 8) was NOT allowed at wood={wood_count}!"

    def test_inventory_without_wood_key_treated_as_zero(self):
        """Empty inventory dictionary should safely default to 0 wood and block table placement."""
        graph = _build_test_graph()
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={},
            workstations_present=set(),
        )
        assert subgoal == "collect_wood"
        assert action == 5

    def test_post_table_placement_immediate_craft(self):
        """
        Once table is placed and inventory drops from 3 -> 1 wood,
        the agent must immediately execute Action 11 (make_wood_pickaxe).
        """
        graph = _build_test_graph()
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={"wood": 1},
            workstations_present={"table_placed"},
        )
        assert subgoal == "make_wood_pickaxe"
        assert action == 11

    def test_post_table_placement_zero_wood_forces_gathering(self):
        """
        If table is placed but wood is 0, the agent must gather 1 wood (Action 5).
        """
        graph = _build_test_graph()
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={"wood": 0},
            workstations_present={"table_placed"},
        )
        assert subgoal == "collect_wood"
        assert action == 5


class TestRecursiveHypergraphTraversal:
    """Stress testing recursive resolution, cycles, and multi-resource requirements."""

    def test_arbitrary_deep_linear_dependency(self):
        """Builds a 4-deep linear dependency chain and tests total cost aggregation."""
        graph = CausalHypergraph()
        # D -> C -> B -> A
        graph.cards["item_A"] = CausalCard(action=1, name="item_A", consumes={"raw": 3})
        graph.cards["item_B"] = CausalCard(action=2, name="item_B", consumes={"raw": 2}, predecessors=["item_A"])
        graph.cards["item_C"] = CausalCard(action=3, name="item_C", consumes={"raw": 1}, predecessors=["item_B"])
        graph.cards["item_D"] = CausalCard(action=4, name="item_D", consumes={"flux": 1}, predecessors=["item_C"])

        budget = graph.calculate_upstream_budget("item_D")
        assert budget == {"raw": 6, "flux": 1}

    def test_cyclic_dependency_does_not_infinite_loop(self):
        """Circular predecessor references must be pruned by the visited set."""
        graph = CausalHypergraph()
        graph.cards["loop_1"] = CausalCard(action=1, name="loop_1", consumes={"wood": 1}, predecessors=["loop_2"])
        graph.cards["loop_2"] = CausalCard(action=2, name="loop_2", consumes={"stone": 1}, predecessors=["loop_1"])

        budget = graph.calculate_upstream_budget("loop_1")
        assert budget == {"wood": 1, "stone": 1}

    def test_unregistered_goal_behavior(self):
        """Target goals not in hypergraph must return empty budget and default explore subgoal."""
        graph = CausalHypergraph()
        budget = graph.calculate_upstream_budget("nonexistent_ancient_debris")
        assert budget == {}

        subgoal, action = graph.determine_subgoal(
            target_goal="nonexistent_ancient_debris",
            current_inventory={"wood": 10},
            workstations_present=set(),
        )
        assert subgoal == "explore"
        assert action == 0


class TestCausalCardPersistenceAndLifecycle:
    """Verifies creation, step simulation, confidence evolution, and serialization."""

    def test_card_creation_and_registration(self):
        graph = CausalHypergraph()
        new_card = CausalCard(
            action=12,
            name="make_stone_pickaxe",
            consumes={"wood": 1, "stone": 1},
            produces={"stone_pickaxe": 1},
            catalyst_needed="table_placed",
            predecessors=["place_table", "collect_stone"],
        )
        graph.cards["make_stone_pickaxe"] = new_card
        assert "make_stone_pickaxe" in graph.cards
        retrieved = graph.cards.get("make_stone_pickaxe")
        assert retrieved is new_card
        assert retrieved.action == 12

    def test_simulated_step_lifecycle(self):
        """Simulates 10 steps of interactions tracking success/failure feedback."""
        graph = _build_test_graph()
        card = graph.cards["place_table"]
        assert card.confidence == 0.0

        # 5 consecutive successes
        for _ in range(5):
            card.success_count += 1
        conf_5 = card.confidence
        assert conf_5 > 0.0

        # 3 failures occur
        for _ in range(3):
            card.failure_count += 1
        conf_8 = card.confidence
        assert conf_8 < conf_5

    def test_serialization_roundtrip(self):
        """Verifies graph state (cards, counts, storage_path) can be saved and restored."""
        graph = _build_test_graph()
        graph.storage_path = "/tmp/causal_memory.json"
        graph.cards["place_table"].success_count = 42
        graph.cards["place_table"].failure_count = 3

        serialized = pickle.dumps(graph)
        restored: CausalHypergraph = pickle.loads(serialized)

        assert restored.storage_path == "/tmp/causal_memory.json"
        assert restored.cards["place_table"].success_count == 42
        assert restored.cards["place_table"].failure_count == 3
        assert restored.cards["place_table"].confidence == graph.cards["place_table"].confidence

    def test_storage_path_interface_gap_documented(self):
        """
        Adversarial Finding: CausalHypergraph accepts storage_path in __init__,
        but does not implement native save() / load() methods on the class.
        """
        graph = CausalHypergraph(storage_path="/tmp/test_storage.json")
        assert graph.storage_path == "/tmp/test_storage.json"
        # Empirically document absence of native save/load methods:
        assert not hasattr(graph, "save")
        assert not hasattr(graph, "load")


class TestAdversarialSubgoalVulnerabilities:
    """
    Exposes structural edge cases and priority leaps in determine_subgoal.
    """

    def test_adversarial_deep_catalyst_chain_bypass(self):
        """
        CRITICAL VULNERABILITY FINDING:
        When target_goal is 'collect_stone', its catalyst is 'wood_pickaxe'.
        If wood=3 and workstations_present=set() (no table placed!):
        determine_subgoal returns ('make_wood_pickaxe', 11), skipping table placement (action 8)!
        This happens because determine_subgoal returns (pred, pred_card.action) directly
        without recursively resolving if pred has its own unsatisfied catalyst ('table_placed').
        Now properly resolved: returns ('place_table', 8).
        """
        graph = _build_test_graph()
        subgoal, action = graph.determine_subgoal(
            target_goal="collect_stone",
            current_inventory={"wood": 3},
            workstations_present=set(),
        )
        assert subgoal == "place_table"
        assert action == 8

    def test_adversarial_furnace_without_pickaxe_leap(self):
        """
        When target_goal is 'place_furnace' with stone=0 and wood=0,
        recursive upstream budgeting correctly chains back to root input ('collect_wood', 5).
        """
        graph = _build_test_graph()
        subgoal, action = graph.determine_subgoal(
            target_goal="place_furnace",
            current_inventory={"stone": 0, "wood": 0},
            workstations_present=set(),
        )
        assert subgoal == "collect_wood"
        assert action == 5


class TestAutonomousAffordanceFrontierSelector:
    """Verifies that AAFS dynamically selects goals without hardcoded curricula."""

    def test_empty_graph_returns_explore(self):
        graph = CausalHypergraph()
        goal = graph.select_affordance_frontier(current_inventory={})
        assert goal == "explore"

    def test_frontier_selects_pickaxe_when_holding_budget(self):
        graph = _build_test_graph()
        # With 3 wood, pickaxe is reachable and preferred over basic wood gathering
        goal = graph.select_affordance_frontier(current_inventory={"wood": 3})
        assert goal in ("make_wood_pickaxe", "make_wood_sword")

    def test_frontier_advances_to_stone_once_pickaxe_held(self):
        graph = _build_test_graph()
        # Holding a wood pickaxe, the agent must NOT re-target pickaxe; it advances to collect_stone
        goal = graph.select_affordance_frontier(
            current_inventory={"wood_pickaxe": 1, "wood": 0},
            workstations_present={"table_placed"},
        )
        assert goal == "collect_stone"
