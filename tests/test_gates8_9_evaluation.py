#!/usr/bin/env python3
"""
Unit tests for Sections 8 & 9 (Gates 8 & 9): Multi-Horizon Rollouts and
Counterfactual Action Intervention Evaluation.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from training.sequence_buffer import EpisodeBuffer
from training.evaluate_multihorizon_interventions import (
    evaluate_multi_horizon,
    evaluate_counterfactual_interventions,
    run_evaluation,
)


@pytest.fixture
def synthetic_shards_buffer():
    """Create a temporary buffer with synthetic labeled transitions."""
    buffer = EpisodeBuffer(capacity_transitions=5000)
    n_eps = 6
    ep_len = 30
    total_t = n_eps * ep_len

    obs = np.random.randint(0, 256, (total_t, 64, 64, 3), dtype=np.uint8)
    actions = np.random.randint(0, 17, total_t, dtype=np.int64)
    rewards = np.zeros(total_t, dtype=np.float32)
    dones = np.zeros(total_t, dtype=bool)
    for e in range(1, n_eps + 1):
        dones[e * ep_len - 1] = True
    events = np.zeros((total_t, 22), dtype=np.uint8)
    mask_before = np.zeros((total_t, 22), dtype=np.uint8)

    # Some events produce non-zero delta
    state_delta = np.zeros((total_t, 16), dtype=np.float32)
    for t in range(0, total_t, 5):
        state_delta[t, 0] = 1.0  # wood delta
    state_delta_valid = np.ones(total_t, dtype=bool)

    offset = 0
    for e in range(n_eps):
        buffer.add_episode(
            obs=obs[offset : offset + ep_len],
            actions=actions[offset : offset + ep_len],
            rewards=rewards[offset : offset + ep_len],
            dones=dones[offset : offset + ep_len],
            events=events[offset : offset + ep_len],
            mask_before=mask_before[offset : offset + ep_len],
            state_delta=state_delta[offset : offset + ep_len],
            state_delta_valid=state_delta_valid[offset : offset + ep_len],
        )
        offset += ep_len
    return buffer


def test_evaluate_multi_horizon(synthetic_shards_buffer):
    """Verify multi-horizon free-running and teacher-forced evaluation over horizons H in {1, 5, 10}."""
    model = JEPAv3()
    model.eval()

    results = evaluate_multi_horizon(
        model=model,
        buffer=synthetic_shards_buffer,
        horizons=(1, 5, 10),
        batches=4,
        batch_size=4,
        labeled_ratio=1.0,
        device="cpu",
    )

    assert "gate_8_passed" in results
    assert "horizons" in results
    assert "horizon_1" in results["horizons"]
    assert "horizon_5" in results["horizons"]
    assert "horizon_10" in results["horizons"]

    h5 = results["horizons"]["horizon_5"]
    assert "freerun_mse" in h5
    assert "persistence_mse" in h5
    assert "mean_latent_l2_drift" in h5


def test_evaluate_counterfactual_interventions(synthetic_shards_buffer):
    """Verify counterfactual action intervention test checks action separation and intervention accuracy."""
    model = JEPAv3()
    model.eval()

    results = evaluate_counterfactual_interventions(
        model=model,
        buffer=synthetic_shards_buffer,
        batches=4,
        batch_size=4,
        device="cpu",
    )

    assert "gate_9_passed" in results
    assert "action_collapse_detected" in results
    assert "mean_pairwise_action_separation" in results
    assert "intervention_accuracy_pct" in results
    assert "mean_counterfactual_margin" in results


def test_end_to_end_gates8_9_report():
    """Verify run_evaluation generates a valid report JSON matching protocol schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        shard_path = tmp_p / "synthetic_shard_final.npz"
        ckpt_path = tmp_p / "test_model.pt"
        out_json = tmp_p / "phase1_5_intervention_eval.json"

        # Save dummy model
        model = JEPAv3()
        model.save_checkpoint(str(ckpt_path))

        # Save synthetic shard
        total_t = 100
        obs = np.random.randint(0, 256, (total_t, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randint(0, 17, total_t, dtype=np.int64)
        rewards = np.zeros(total_t, dtype=np.float32)
        dones = np.zeros(total_t, dtype=bool)
        dones[49] = True
        dones[99] = True
        events = np.zeros((total_t, 22), dtype=np.uint8)
        mask_before = np.zeros((total_t, 22), dtype=np.uint8)
        state_delta = np.zeros((total_t, 16), dtype=np.float32)
        state_delta[:, 0] = 0.5
        state_delta_valid = np.ones(total_t, dtype=bool)
        ep_lens = np.array([50, 50], dtype=np.int64)

        np.savez_compressed(
            shard_path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            events=events,
            mask_before=mask_before,
            state_delta=state_delta,
            state_delta_valid=state_delta_valid,
            episode_lengths=ep_lens,
        )

        report = run_evaluation(
            shards_dir=tmp_p,
            manifest_path=None,
            checkpoint_path=ckpt_path,
            output_json=out_json,
            val_ratio=0.5,
            split_seed=42,
            device="cpu",
        )

        assert out_json.exists()
        assert report["protocol"] == "strict_visual_jepa_clean_v2"
        assert "gate_8" in report
        assert "gate_9" in report
        assert "overall_status" in report
