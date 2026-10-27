import numpy as np
import pytest
import torch
from pathlib import Path

from planning.causal_memory import (
    EpisodicEvidence,
    EpisodicEvidenceStore,
    RelationHypothesisBank,
    TransitionPrototype,
)


def test_evidence_store_record_and_commit():
    """Verify EpisodicEvidenceStore buffers steps, computes Monte Carlo returns, and stores transitions."""
    store = EpisodicEvidenceStore(max_capacity=100, gamma=0.9)

    # Simulate a 3-step episode where step 2 receives reward +1.0
    # Step 0: action 7 (chop wood), reward 0.0
    store.record_step(
        obs_latent=np.zeros(128),
        grounded_state=np.zeros(16),
        action=7,
        reward=0.0,
        done=False,
        next_obs_latent=np.zeros(128),
        next_grounded_state=np.zeros(16),
        episode_id=1,
        step_id=0,
    )
    # Step 1: action 7 (chop wood), reward 0.0, wood +1.0
    s1_next_x = np.zeros(16)
    s1_next_x[0] = 1.0  # wood = 1
    store.record_step(
        obs_latent=np.zeros(128),
        grounded_state=np.zeros(16),
        action=7,
        reward=0.0,
        done=False,
        next_obs_latent=np.zeros(128),
        next_grounded_state=s1_next_x,
        episode_id=1,
        step_id=1,
    )
    # Step 2: action 10 (place table), reward 1.0, done True
    store.record_step(
        obs_latent=np.zeros(128),
        grounded_state=s1_next_x,
        action=10,
        reward=1.0,
        done=True,
        next_obs_latent=np.zeros(128),
        next_grounded_state=s1_next_x,
        episode_id=1,
        step_id=2,
    )

    # When done=True, commit_episode should run automatically
    assert len(store) == 3

    # Check Monte Carlo discounted return G_t:
    # G_2 = 1.0
    # G_1 = 0.0 + 0.9 * 1.0 = 0.90
    # G_0 = 0.0 + 0.9 * 0.90 = 0.81
    assert store.entries[2].return_g == pytest.approx(1.0, abs=1e-3)
    assert store.entries[1].return_g == pytest.approx(0.90, abs=1e-3)
    assert store.entries[0].return_g == pytest.approx(0.81, abs=1e-3)


def test_relation_hypothesis_consolidation():
    """Verify RelationHypothesisBank consolidates breakthroughs into empirical prototypes."""
    store = EpisodicEvidenceStore()

    # Create 3 similar breakthrough transitions for action 11 (make wood pickaxe)
    for i in range(3):
        z_pre = np.ones(128) * 0.5
        x_pre = np.zeros(16)
        x_pre[0] = 1.0  # wood = 1 (precondition)
        x_post = np.zeros(16)
        x_post[6] = 1.0  # pickaxe = 1
        delta = x_post - x_pre  # wood -1, pickaxe +1

        store.entries.append(EpisodicEvidence(
            obs_latent=z_pre.tolist(),
            grounded_state=x_pre.tolist(),
            action=11,
            reward=1.0,
            done=False,
            next_obs_latent=z_pre.tolist(),
            next_grounded_state=x_post.tolist(),
            state_delta=delta.tolist(),
            return_g=1.0,
            episode_id=i,
            step_id=10,
        ))

    bank = RelationHypothesisBank()
    count = bank.consolidate_from_evidence(store, min_cluster_size=2)
    assert count >= 1
    assert len(bank) >= 1

    proto = list(bank.prototypes.values())[0]
    assert proto.action == 11
    # Check that discovered precondition captures wood requirement
    assert proto.precondition_state[0] == pytest.approx(1.0, abs=0.1)
    # Check that discovered effect delta captures pickaxe acquisition
    assert proto.effect_delta[6] == pytest.approx(1.0, abs=0.1)
    assert proto.success_count == 3
    assert proto.failure_count == 0
    assert proto.success_probability > 0.7


def test_hypothesis_revision_on_contradiction():
    """Verify Phase 2 Acceptance Criterion 3: Contradictory observations lower confidence."""
    bank = RelationHypothesisBank()
    proto = TransitionPrototype(
        prototype_id="proto_craft",
        action=11,
        precondition_key=[1.0] + [0.0] * 127,
        precondition_state=[1.0] + [0.0] * 15,
        effect_delta=[0.0] * 16,
        postcondition_key=[1.0] + [0.0] * 127,
        success_count=5,
        failure_count=0,
        mean_return=1.0,
        confidence=0.83,
    )
    bank.prototypes["proto_craft"] = proto

    initial_prob = proto.success_probability
    initial_conf = proto.confidence

    # Register 5 failed attempts (contradictory evidence)
    for _ in range(5):
        bank.register_interaction_outcome("proto_craft", succeeded=False)

    revised_prob = bank.prototypes["proto_craft"].success_probability
    revised_conf = bank.prototypes["proto_craft"].confidence

    # Confidence and success probability must decrease significantly
    assert revised_prob < initial_prob
    assert revised_conf < initial_conf
    assert bank.prototypes["proto_craft"].failure_count == 5


def test_memory_lifecycle_save_and_reload(tmp_path):
    """Verify Phase 2 memory persistence across process restarts."""
    store_file = tmp_path / "evidence.json"
    bank_file = tmp_path / "relations.json"

    # 1. Populate and save
    store = EpisodicEvidenceStore()
    store.entries.append(EpisodicEvidence(
        obs_latent=[0.1] * 128,
        grounded_state=[0.2] * 16,
        action=3,
        reward=0.5,
        done=False,
        next_obs_latent=[0.3] * 128,
        next_grounded_state=[0.4] * 16,
        state_delta=[0.2] * 16,
        return_g=1.5,
        episode_id=42,
        step_id=7,
    ))
    store.save(store_file)

    bank = RelationHypothesisBank()
    bank.prototypes["test_p"] = TransitionPrototype(
        prototype_id="test_p",
        action=3,
        precondition_key=[0.1] * 128,
        precondition_state=[0.2] * 16,
        effect_delta=[0.2] * 16,
        postcondition_key=[0.3] * 128,
        success_count=10,
        failure_count=2,
        mean_return=2.5,
        confidence=0.75,
    )
    bank.save(bank_file)

    # 2. Reload into fresh instances
    loaded_store = EpisodicEvidenceStore()
    loaded_store.load(store_file)
    assert len(loaded_store) == 1
    assert loaded_store.entries[0].episode_id == 42
    assert loaded_store.entries[0].return_g == 1.5

    loaded_bank = RelationHypothesisBank()
    loaded_bank.load(bank_file)
    assert len(loaded_bank) == 1
    assert "test_p" in loaded_bank.prototypes
    assert loaded_bank.prototypes["test_p"].success_count == 10
    assert loaded_bank.prototypes["test_p"].mean_return == 2.5
