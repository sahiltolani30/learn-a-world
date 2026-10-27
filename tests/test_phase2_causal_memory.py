import json
import numpy as np
import pytest
import torch
from pathlib import Path

from planning.causal_memory import (
    EpisodicEvidence,
    EpisodicEvidenceStore,
    RelationHypothesisBank,
    TransitionPrototype,
    wilson_interval,
)


def test_evidence_store_record_and_commit():
    """Verify EpisodicEvidenceStore buffers steps, computes Monte Carlo returns, and stores transitions."""
    store = EpisodicEvidenceStore(max_capacity=100, gamma=0.9)

    # Simulate a 3-step episode where step 2 receives reward +1.0
    # Step 0: action 7, reward 0.0 — carries belief-derived unlock labels (v2 fields)
    event_vec = [0] * 22
    event_vec[6] = 1  # first-time unlock of achievement dim 6 (belief-derived label)
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
        event_vec=event_vec,
        mask_before_vec=[0] * 22,
        source="offline",
    )
    # Step 1: action 7, reward 0.0, resource dim 0 +1.0
    s1_next_x = np.zeros(16)
    s1_next_x[0] = 1.0
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
    # Step 2: action 10, reward 1.0, done True
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

    # v2 fields: event-labelled entries carry their unlock decomposition; entries
    # without one default to None / "offline".
    assert store.entries[0].event_vec == event_vec
    assert store.entries[0].mask_before_vec == [0] * 22
    assert store.entries[0].source == "offline"
    assert store.entries[1].event_vec is None
    assert store.entries[1].source == "offline"


def test_relation_hypothesis_consolidation():
    """Verify RelationHypothesisBank consolidates event-grounded successes into causal cards."""
    store = EpisodicEvidenceStore()

    # Create 3 similar event-grounded breakthrough transitions for action 11.
    # Unlock label: achievement dim 15 fires for the first time (belief-derived).
    event_vec = [0] * 22
    event_vec[15] = 1
    for i in range(3):
        z_pre = np.ones(128) * 0.5
        x_pre = np.zeros(16)
        x_pre[0] = 1.0  # resource dim 0 = 1 (requirement for the craft)
        x_post = np.zeros(16)
        x_post[6] = 1.0  # state dim 6 acquired
        delta = x_post - x_pre  # dim 0 -1, dim 6 +1

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
            event_vec=list(event_vec),
            mask_before_vec=[0] * 22,
        ))

    # 10 matched near-miss counterexamples (same visual cue, no unlock, no
    # productive delta, reward <= 0, and lacking the required resource) so the
    # needs-enrichment test is statistically decidable for the requirement dim.
    for i in range(10):
        store.entries.append(EpisodicEvidence(
            obs_latent=(np.ones(128) * 0.5).tolist(),
            grounded_state=[0.0] * 16,  # missing the requirement entirely
            action=11,
            reward=0.0,
            done=False,
            next_obs_latent=(np.ones(128) * 0.5).tolist(),
            next_grounded_state=[0.0] * 16,
            state_delta=[0.0] * 16,
            return_g=0.0,
            episode_id=100 + i,
            step_id=3,
        ))

    bank = RelationHypothesisBank()
    count = bank.consolidate_from_evidence(store, min_cluster_size=2)
    assert count >= 1
    assert len(bank) >= 1

    proto = list(bank.prototypes.values())[0]
    assert proto.action == 11
    # Event-grounded card: grouped by the fired achievement dim
    assert proto.event_dim is not None
    # The discovered need keeps the required resource (enrichment-tested against
    # the matched near-misses)
    assert any(int(nd[0]) == 0 for nd in proto.needs)
    assert proto.precondition_state[0] == pytest.approx(1.0, abs=0.1)
    # Discovered effect delta captures the produced state dim
    assert proto.effect_delta[6] == pytest.approx(1.0, abs=0.1)
    assert proto.success_count == pytest.approx(3.0)
    assert proto.failure_count == pytest.approx(10.0)
    assert proto.success_probability == pytest.approx((3.0 + 1.0) / (3.0 + 10.0 + 2.0))
    assert proto.unlock_prob == pytest.approx((3.0 + 1.0) / (3.0 + 10.0 + 2.0))
    assert proto.status == "confirmed"
    assert proto.confidence == pytest.approx(max(0.05, wilson_interval(3.0, 13.0)[0]))


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
    """Verify Phase 2 memory persistence across process restarts, including v1 files."""
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
        event_vec=[1] + [0] * 21,
        mask_before_vec=[0] * 22,
        source="online",
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
        event_dim=13,
        needs=[[0.0, 1.5]],
        produces=[[0.0, 0.8]],
        consumes=[],
        unlock_prob=0.7,
        wilson_lb=0.6,
        visual_dispersion=0.95,
        status="confirmed",
        success_offline=8.0,
        success_online=2.0,
        failure_offline=2.0,
        failure_online=0.0,
        created_from="offline",
    )
    bank.save(bank_file)

    # 2. Reload into fresh instances
    loaded_store = EpisodicEvidenceStore()
    loaded_store.load(store_file)
    assert len(loaded_store) == 1
    assert loaded_store.entries[0].episode_id == 42
    assert loaded_store.entries[0].return_g == 1.5
    # v2 evidence fields round-trip
    assert loaded_store.entries[0].event_vec == [1] + [0] * 21
    assert loaded_store.entries[0].mask_before_vec == [0] * 22
    assert loaded_store.entries[0].source == "online"

    loaded_bank = RelationHypothesisBank()
    loaded_bank.load(bank_file)
    assert len(loaded_bank) == 1
    assert "test_p" in loaded_bank.prototypes
    card = loaded_bank.prototypes["test_p"]
    assert card.success_count == 10
    assert card.mean_return == 2.5
    # v2 card fields round-trip
    assert card.event_dim == 13
    assert card.needs == [[0.0, 1.5]]
    assert card.produces == [[0.0, 0.8]]
    assert card.unlock_prob == pytest.approx(0.7)
    assert card.wilson_lb == pytest.approx(0.6)
    assert card.visual_dispersion == pytest.approx(0.95)
    assert card.status == "confirmed"
    assert card.success_offline == pytest.approx(8.0)
    assert card.success_online == pytest.approx(2.0)
    assert card.failure_offline == pytest.approx(2.0)
    assert card.failure_online == pytest.approx(0.0)
    assert card.created_from == "offline"
    assert card.schema_version == "phase2_card_v2"

    # 3. A v1-format bank JSON (old field names only, no edges key) still loads.
    v1_bank = {
        "version": "phase2_relation_bank_v1",
        "prototypes": {
            "legacy_p": {
                "prototype_id": "legacy_p",
                "action": 3,
                "precondition_key": [0.1] * 128,
                "precondition_state": [0.2] * 16,
                "effect_delta": [0.2] * 16,
                "postcondition_key": [0.3] * 128,
                "success_count": 10,
                "failure_count": 2,
                "mean_return": 2.5,
                "mean_duration": 1.0,
                "confidence": 0.75,
            }
        },
    }
    v1_file = tmp_path / "relations_v1.json"
    v1_file.write_text(json.dumps(v1_bank))

    v1_loaded = RelationHypothesisBank()
    v1_loaded.load(v1_file)
    assert len(v1_loaded) == 1
    legacy = v1_loaded.prototypes["legacy_p"]
    # Missing v2 fields fall back to defaults
    assert legacy.event_dim is None
    assert legacy.needs == []
    assert legacy.produces == []
    assert legacy.consumes == []
    assert legacy.unlock_prob == 0.0
    assert legacy.success_offline == 0.0
    assert legacy.created_from == "offline"
    # Wilson stats/status are migrated from the raw v1 counts
    assert legacy.wilson_lb == pytest.approx(wilson_interval(10.0, 12.0)[0])
    assert legacy.confidence == pytest.approx(max(0.05, wilson_interval(10.0, 12.0)[0]))
    assert legacy.status == "confirmed"

    # 4. A v1-format evidence store JSON (entries without v2 keys) still loads.
    v1_store = {
        "version": "phase2_evidence_v1",
        "max_capacity": 100,
        "gamma": 0.9,
        "entries": [{
            "obs_latent": [0.1] * 128,
            "grounded_state": [0.2] * 16,
            "action": 3,
            "reward": 0.5,
            "done": False,
            "next_obs_latent": [0.3] * 128,
            "next_grounded_state": [0.4] * 16,
            "state_delta": [0.2] * 16,
            "return_g": 1.5,
            "episode_id": 42,
            "step_id": 7,
            "confidence": 1.0,
        }],
    }
    v1_store_file = tmp_path / "evidence_v1.json"
    v1_store_file.write_text(json.dumps(v1_store))

    v1_loaded_store = EpisodicEvidenceStore()
    v1_loaded_store.load(v1_store_file)
    assert len(v1_loaded_store) == 1
    assert v1_loaded_store.entries[0].event_vec is None
    assert v1_loaded_store.entries[0].mask_before_vec is None
    assert v1_loaded_store.entries[0].source == "offline"
