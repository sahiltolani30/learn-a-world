"""
Unit tests for the Human-Like Causal Learning and Upstream Budgeting subsystems in Godfather JEPA.

Tests:
1. DenseRewardHead forward and gradient flow.
2. CausalHypergraph recursive upstream budgeting (3 Wood = Table(2) + Pickaxe(1)).
3. TopologicalAnchorGraph landmark registration and closed-loop path reversal.
4. HomeostaticArbiter urgency evaluation and priority switching.
5. BenchmarkRunner stateful policy support with continuous recurrent belief.
"""
import numpy as np
import pytest
import torch

from godfather_jepa.eval.benchmark import BenchmarkRunner
from godfather_jepa.models.reward_head import DenseRewardHead
from godfather_jepa.models.odometry import VisualOdometryHead, TopologicalAnchorGraph
from godfather_jepa.planning.causal_hypergraph import CausalHypergraph, CausalCard
from godfather_jepa.planning.homeostatic_arbiter import HomeostaticArbiter


class TestDenseRewardHead:
    def test_forward_shape(self):
        head = DenseRewardHead(state_dim=1024, num_actions=17, act_dim=64, hidden_dim=128)
        state = torch.randn(4, 1024)
        action = torch.tensor([0, 5, 8, 11])
        out = head(state, action)
        assert out.shape == (4,)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        head = DenseRewardHead(state_dim=1024, num_actions=17)
        state = torch.randn(2, 1024, requires_grad=True)
        action = torch.tensor([1, 2])
        out = head(state, action)
        loss = out.sum()
        loss.backward()
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()


class TestCausalHypergraphAndBudgeting:
    def test_foundational_grammar_loaded(self):
        graph = CausalHypergraph()
        assert "collect_wood" in graph.cards
        assert "place_table" in graph.cards
        assert "make_wood_pickaxe" in graph.cards
        assert "collect_stone" in graph.cards

    def test_upstream_budget_wood_pickaxe(self):
        """
        Critical test: Verifies that make_wood_pickaxe aggregates:
        Cost(Table) [2 wood] + Cost(Pickaxe) [1 wood] = 3 Wood total.
        """
        graph = CausalHypergraph()
        budget = graph.calculate_upstream_budget("make_wood_pickaxe")
        assert budget.get("wood") == 3

    def test_upstream_budget_wood_sword(self):
        graph = CausalHypergraph()
        budget = graph.calculate_upstream_budget("make_wood_sword")
        assert budget.get("wood") == 3

    def test_wood_deficit_enforces_gathering_before_table(self):
        """
        Verifies the fix for the Wood-Count Deficit Trap:
        When the agent has 2 wood and wants a pickaxe, it must NOT place the table yet!
        It must gather 1 more wood first.
        """
        graph = CausalHypergraph()
        # Has 2 wood, table not placed
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={"wood": 2},
            workstations_present=set(),
        )
        assert subgoal == "collect_wood"
        assert action == 5  # hit tree

    def test_places_table_when_budget_satisfied(self):
        """
        When the agent has 3 wood, it is cleared to place the table.
        """
        graph = CausalHypergraph()
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={"wood": 3},
            workstations_present=set(),
        )
        assert subgoal == "place_table"
        assert action == 8  # place table

    def test_immediate_pickaxe_craft_after_table(self):
        """
        After table is placed and wood drops from 3 -> 1, immediately craft pickaxe!
        """
        graph = CausalHypergraph()
        subgoal, action = graph.determine_subgoal(
            target_goal="make_wood_pickaxe",
            current_inventory={"wood": 1},
            workstations_present={"table_placed"},
        )
        assert subgoal == "make_wood_pickaxe"
        assert action == 11  # craft pickaxe


class TestTopologicalAnchorGraph:
    def test_landmark_registration_and_return(self):
        anchor = TopologicalAnchorGraph()
        anchor.register_landmark("table")

        # Step 5 units to the right
        anchor.update_odometry(np.array([5.0, 0.0], dtype=np.float32))

        # Return vector should point left (-5.0, 0.0)
        vec = anchor.get_return_vector("table")
        assert vec is not None
        assert vec[0] == pytest.approx(-5.0)
        assert vec[1] == pytest.approx(0.0)

        # Return action should be Action 1 (Left)
        action = anchor.get_return_action("table")
        assert action == 1

    def test_odometry_head_forward(self):
        head = VisualOdometryHead(latent_dim=512, num_actions=17)
        z_t = torch.randn(4, 512)
        z_next = torch.randn(4, 512)
        action = torch.tensor([1, 2, 3, 4])
        disp = head(z_t, action, z_next)
        assert disp.shape == (4, 2)


class TestHomeostaticArbiter:
    def test_safe_condition_passes_teleological(self):
        arbiter = HomeostaticArbiter()
        action, mode = arbiter.arbitrate(
            vitals={"health": 9.0, "food": 9.0, "drink": 9.0},
            step=50,
            teleological_action=8,
            water_adjacent=False,
            continuation_prob=1.0,
        )
        assert action == 8
        assert mode == "teleological"

    def test_preemptive_drinking_override(self):
        arbiter = HomeostaticArbiter()
        # Thirst is depleted (1.0 <= 3.0 threshold), next to water
        action, mode = arbiter.arbitrate(
            vitals={"health": 8.0, "food": 8.0, "drink": 1.0},
            step=80,
            teleological_action=8,
            water_adjacent=True,
            continuation_prob=0.9,
        )
        assert action == 5  # drink reflex
        assert mode == "homeostatic_drink"

    def test_nightfall_danger_override(self):
        arbiter = HomeostaticArbiter()
        # Step 145 (dusk approaching) with low continuation probability
        action, mode = arbiter.arbitrate(
            vitals={"health": 7.0, "food": 7.0, "drink": 8.0},
            step=145,
            teleological_action=11,
            water_adjacent=False,
            continuation_prob=0.1,
        )
        assert action == 5  # combat / defensive reflex
        assert mode == "homeostatic_combat"


class TestBenchmarkStatefulSupport:
    def test_stateful_policy_preserves_recurrent_state(self):
        runner = BenchmarkRunner(seeds=[2050], max_steps_per_episode=10)

        steps_executed = 0
        state_history = []

        def stateful_policy(obs, state=None):
            nonlocal steps_executed
            steps_executed += 1
            cur_count = 0 if state is None else state
            next_state = cur_count + 1
            state_history.append(next_state)
            return 0, next_state

        results = runner.run(stateful_policy)
        assert steps_executed == 10
        # Check that state sequentially incremented: 1, 2, 3, ..., 10
        assert state_history == list(range(1, 11))
