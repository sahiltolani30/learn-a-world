"""
Unit and Invariant Tests for Latent MPC Planner in Godfather JEPA.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from godfather_jepa.models.predictor import BoundedResidualPredictor
from godfather_jepa.models.reward_head import DenseRewardHead
from godfather_jepa.models.survival import SurvivalContinuationHead
from godfather_jepa.models.symlog_critic import SymlogCategoricalCritic, SymlogTwoHotCoder
from godfather_jepa.models.belief import SpatialConvGRU, VectorBeliefCore
from godfather_jepa.models.odometry import TopologicalAnchorGraph
from godfather_jepa.planning.causal_hypergraph import CausalHypergraph
from godfather_jepa.planning.latent_mpc_planner import LatentMPCPlanner


class TestLatentMPCPlanner:
    @pytest.fixture
    def setup_planner(self):
        device = torch.device("cpu")
        predictor = BoundedResidualPredictor(latent_dim=512, belief_dim=512)
        reward_head = DenseRewardHead(state_dim=1024)
        continuation_head = SurvivalContinuationHead(state_dim=1024)
        critic = SymlogCategoricalCritic(state_dim=1024)
        symlog_coder = SymlogTwoHotCoder()
        spatial_gru = SpatialConvGRU(channels=64)
        vector_belief = VectorBeliefCore(latent_dim=512, spatial_channels=64)
        causal_hypergraph = CausalHypergraph()

        planner = LatentMPCPlanner(
            predictor=predictor,
            reward_head=reward_head,
            continuation_head=continuation_head,
            critic=critic,
            symlog_coder=symlog_coder,
            spatial_gru=spatial_gru,
            vector_belief=vector_belief,
            causal_hypergraph=causal_hypergraph,
            horizon=5,
            num_candidates=16,
            cem_iterations=2,
            device=device,
        )
        return planner

    def test_plan_output_contract(self, setup_planner):
        planner = setup_planner
        z_0 = torch.randn(1, 512)
        h_vec_0 = torch.randn(1, 512)
        h_sp_0 = torch.randn(1, 64, 8, 8)
        inv = {"wood": 0, "stone": 0}
        ws = set()
        anchor = TopologicalAnchorGraph()

        action, debug = planner.plan(
            z_0=z_0,
            h_vector_0=h_vec_0,
            h_spatial_0=h_sp_0,
            inventory=inv,
            workstations=ws,
            anchor_graph=anchor,
            target_goal="make_wood_pickaxe",
        )

        assert isinstance(action, int)
        assert 0 <= action <= 16
        assert "subgoal" in debug
        assert "mode" in debug

    def test_causal_action_injection_place_table_when_budget_satisfied(self, setup_planner):
        planner = setup_planner
        z_0 = torch.randn(1, 512)
        h_vec_0 = torch.randn(1, 512)
        h_sp_0 = torch.randn(1, 64, 8, 8)
        # When holding 3 wood (upstream budget satisfied!), it MUST place table (Action 8)
        inv = {"wood": 3, "stone": 0}
        ws = set()
        anchor = TopologicalAnchorGraph()

        action, debug = planner.plan(
            z_0=z_0,
            h_vector_0=h_vec_0,
            h_spatial_0=h_sp_0,
            inventory=inv,
            workstations=ws,
            anchor_graph=anchor,
            target_goal="make_wood_pickaxe",
        )

        assert action == 8
        assert debug["subgoal"] == "place_table"
        assert debug["mode"] == "instant_causal_craft"

    def test_causal_action_injection_make_wood_pickaxe_when_table_present(self, setup_planner):
        planner = setup_planner
        z_0 = torch.randn(1, 512)
        h_vec_0 = torch.randn(1, 512)
        h_sp_0 = torch.randn(1, 64, 8, 8)
        # When table is placed and holding 1 wood, it MUST craft wood pickaxe (Action 11)
        inv = {"wood": 1, "stone": 0}
        ws = {"table_placed"}
        anchor = TopologicalAnchorGraph()

        action, debug = planner.plan(
            z_0=z_0,
            h_vector_0=h_vec_0,
            h_spatial_0=h_sp_0,
            inventory=inv,
            workstations=ws,
            anchor_graph=anchor,
            target_goal="make_wood_pickaxe",
        )

        assert action == 11
        assert debug["subgoal"] == "make_wood_pickaxe"
        assert debug["mode"] == "instant_causal_craft"

    def test_topological_return_to_table_override(self, setup_planner):
        planner = setup_planner
        z_0 = torch.randn(1, 512)
        h_vec_0 = torch.randn(1, 512)
        h_sp_0 = torch.randn(1, 64, 8, 8)
        inv = {"wood": 1, "stone": 0, "wood_pickaxe": 0}
        ws = {"table_placed"}
        anchor = TopologicalAnchorGraph()
        # Register table at origin, then move away to (5.0, 0.0)
        anchor.register_landmark("table", latent_key=np.zeros(512, dtype=np.float32))
        anchor.current_displacement = np.array([5.0, 0.0], dtype=np.float32)

        action, debug = planner.plan(
            z_0=z_0,
            h_vector_0=h_vec_0,
            h_spatial_0=h_sp_0,
            inventory=inv,
            workstations=ws,
            anchor_graph=anchor,
            target_goal="make_wood_pickaxe",
        )

        # Return vector from (5.0, 0.0) back to table at (0, 0) requires moving left (Action 1)
        assert action == 1
        assert debug["subgoal"] == "return_to_table"
