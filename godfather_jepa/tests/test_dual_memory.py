"""
Unit tests for the Dual-Memory System in Godfather JEPA.

Tests:
1. OnlineEpisodicStore step recording, spatial proximity, and return vector.
2. SemanticConsolidatedBank upstream budgeting and serialization roundtrip.
3. OfflineConsolidationEngine 4 Iron Rules:
   - Rule 1: n=1 rare milestone preservation.
   - Rule 2: Precondition attribution guard (zero materials does not penalize card).
   - Rule 3: Spatial workstation proximity guard (away from table does not penalize pickaxe card).
"""
import os
import tempfile
import numpy as np
import pytest
import torch

from godfather_jepa.memory.online_store import OnlineEpisodicStore
from godfather_jepa.memory.semantic_bank import SemanticConsolidatedBank
from godfather_jepa.memory.consolidation import OfflineConsolidationEngine


class TestOnlineEpisodicStore:
    def test_step_recording_and_displacement(self):
        store = OnlineEpisodicStore()
        z0 = np.zeros(512, dtype=np.float32)
        z1 = np.ones(512, dtype=np.float32)

        # Step 1: Walk right
        store.record_step(
            step_idx=0,
            z_t=z0,
            action=2,
            reward=0.0,
            done=False,
            z_next=z1,
            delta_p=np.array([1.0, 0.0], dtype=np.float32),
        )
        assert store.current_displacement[0] == pytest.approx(1.0)
        assert len(store.steps) == 1

    def test_workstation_placement_and_proximity(self):
        store = OnlineEpisodicStore()
        z = np.zeros(512, dtype=np.float32)

        # Step 0: Place table at position (2.0, 0.0)
        store.record_step(
            step_idx=0,
            z_t=z,
            action=8,
            reward=1.0,
            done=False,
            z_next=z,
            unlocked_achievement="place_table",
            delta_p=np.array([2.0, 0.0], dtype=np.float32),
        )
        assert store.is_adjacent_to_workstation(threshold=1.5)

        # Step 1: Walk 4 tiles away into forest
        store.record_step(
            step_idx=1,
            z_t=z,
            action=2,
            reward=0.0,
            done=False,
            z_next=z,
            delta_p=np.array([4.0, 0.0], dtype=np.float32),
        )
        # Total pos is (6.0, 0.0), table is at (2.0, 0.0), dist is 4.0 > 1.5
        assert not store.is_adjacent_to_workstation(threshold=1.5)
        # Return vector points left (-4.0, 0.0)
        ret_vec = store.get_return_vector()
        assert ret_vec[0] == pytest.approx(-4.0)


class TestSemanticConsolidatedBank:
    def test_upstream_budget(self):
        bank = SemanticConsolidatedBank()
        budget = bank.calculate_upstream_budget("make_wood_pickaxe")
        assert budget.get("wood") == 3

    def test_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "test_bank.pt")
            bank = SemanticConsolidatedBank()
            bank.cards["collect_wood"].success_count = 42
            bank.save(pt_path)

            loaded_bank = SemanticConsolidatedBank()
            success = loaded_bank.load(pt_path)
            assert success
            assert loaded_bank.cards["collect_wood"].success_count == 42


class TestOfflineConsolidationEngine:
    def test_rule1_rare_milestone_preservation(self):
        """
        Rule 1: An achievement unlock with n=1 is consolidated and never pruned.
        """
        bank = SemanticConsolidatedBank()
        engine = OfflineConsolidationEngine(semantic_bank=bank)
        store = OnlineEpisodicStore()
        z = np.zeros(512, dtype=np.float32)

        store.record_step(
            step_idx=0,
            z_t=z,
            action=5,
            reward=1.0,
            done=False,
            z_next=z,
            unlocked_achievement="collect_stone",
        )

        res = engine.consolidate_episode(store)
        assert "collect_stone" in res["updated_cards"] or "collect_stone" in res["promoted_cards"]
        assert bank.cards["collect_stone"].success_count == 1

    def test_rule2_precondition_guard_prevents_card_penalty(self):
        """
        Rule 2: Swinging in grass with 0 materials does NOT penalize pickaxe card!
        """
        bank = SemanticConsolidatedBank()
        initial_fail_count = bank.cards["make_wood_pickaxe"].failure_count
        engine = OfflineConsolidationEngine(semantic_bank=bank)
        store = OnlineEpisodicStore()
        z = np.zeros(512, dtype=np.float32)

        # Agent fires Action 11 (make_wood_pickaxe) with 0 wood in inventory
        store.record_step(
            step_idx=0,
            z_t=z,
            action=11,
            reward=0.0,
            done=False,
            z_next=z,
            item_deltas={},  # 0 wood!
        )

        engine.consolidate_episode(store)
        # Failure count should NOT increase!
        assert bank.cards["make_wood_pickaxe"].failure_count == initial_fail_count

    def test_rule3_workstation_proximity_guard(self):
        """
        Rule 3: Swinging when away from table does NOT penalize pickaxe card!
        """
        bank = SemanticConsolidatedBank()
        initial_fail_count = bank.cards["make_wood_pickaxe"].failure_count
        engine = OfflineConsolidationEngine(semantic_bank=bank)
        store = OnlineEpisodicStore()
        z = np.zeros(512, dtype=np.float32)

        # Place table
        store.record_step(
            step_idx=0,
            z_t=z,
            action=8,
            reward=1.0,
            done=False,
            z_next=z,
            unlocked_achievement="place_table",
            delta_p=np.zeros(2, dtype=np.float32),
        )
        # Walk 5 tiles away into forest
        store.record_step(
            step_idx=1,
            z_t=z,
            action=2,
            reward=0.0,
            done=False,
            z_next=z,
            delta_p=np.array([5.0, 0.0], dtype=np.float32),
        )
        # Swing Action 11 in forest with 1 wood
        store.record_step(
            step_idx=2,
            z_t=z,
            action=11,
            reward=0.0,
            done=False,
            z_next=z,
            item_deltas={"wood": 1},
        )

        engine.consolidate_episode(store)
        # Since agent was 5 tiles away from table, card is protected from penalty!
        assert bank.cards["make_wood_pickaxe"].failure_count == initial_fail_count
