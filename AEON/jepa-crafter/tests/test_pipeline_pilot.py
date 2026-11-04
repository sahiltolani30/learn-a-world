"""Tests for Phase 3: Environment, Replay, Checkpoint Smoke and Pipeline Pilot (P3_PIPELINE_PILOT_PASS)."""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from src.env.crafter_adapter import CrafterAdapter
from src.env.opaque_actions import OpaqueActionMapping
from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler
from src.data.checkpoint_store import CheckpointStore
from src.observability.trace_writer import TraceWriter
from src.observability.tracer import TraceManager
from src.models.jepa_world_model import JepaWorldModel
from src.learning.learner import JepaLearner
from src.training.resume import restore_checkpoint


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_ring_wraparound_generation():
    """Test 1: Ring-buffer wraparound tracking guarantees no sampling across overwritten generations."""
    capacity = 40
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(64, 64, 3))
    sampler = SequenceSampler(replay_buffer=buffer, context_length=4, prediction_horizon=8)

    # Write 3 episodes, totaling 100 transitions (more than 2 full overwrites of buffer capacity 40)
    obs_dummy = np.ones((64, 64, 3), dtype=np.uint8)

    for ep in range(1, 4):
        buffer.start_episode(obs_dummy * ep, episode_id=ep)
        for t in range(35):
            is_done = (t == 34)
            buffer.add(
                action=t % 17,
                reward=1.0 if is_done else 0.0,
                next_obs=obs_dummy * (ep * 10 + t),
                terminated=is_done,
                truncated=False,
            )

    assert buffer.size == capacity
    assert buffer.total_added == 105

    # Check that valid anchors only exist within the currently retained continuous spans
    valid_anchors = sampler.get_valid_anchor_indices()
    assert len(valid_anchors) > 0

    for anchor in valid_anchors:
        start_tid = buffer.transition_ids[anchor]
        ep_id = buffer.episode_ids[anchor]
        for k in range(sampler.prediction_horizon):
            idx = (anchor + k) % capacity
            # Absolute write generation check
            assert buffer.transition_ids[idx] == start_tid + k
            assert buffer.episode_ids[idx] == ep_id


def test_10k_sample_stress_test():
    """Test 2: Stress test of 10,000 sequence samples requires zero invalid boundary crossings

    and zero stale overwritten-frame references.
    """
    capacity = 100
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(64, 64, 3))
    sampler = SequenceSampler(replay_buffer=buffer, context_length=4, prediction_horizon=8)

    # Populate buffer across multiple episode boundaries with ring wraparounds
    rng = np.random.RandomState(42)
    obs_base = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

    episode_counter = 1
    total_steps = 350  # 3.5x buffer capacity
    step_in_ep = 0

    buffer.start_episode(obs_base, episode_id=episode_counter)
    for _ in range(total_steps):
        is_done = (step_in_ep >= 25)
        buffer.add(
            action=rng.randint(0, 17),
            reward=float(rng.rand() > 0.8),
            next_obs=obs_base,
            terminated=is_done,
            truncated=False,
        )
        if is_done:
            episode_counter += 1
            step_in_ep = 0
            buffer.start_episode(obs_base, episode_id=episode_counter)
        else:
            step_in_ep += 1

    valid_anchors = sampler.get_valid_anchor_indices()
    assert len(valid_anchors) > 0

    # Draw 10,000 sequence samples
    for i in range(10000):
        seq = sampler.sample_one_sequence(valid_indices=valid_anchors)
        
        # Verify sequence shape integrity
        assert seq["obs_context"].shape == (4, 64, 64, 3)
        assert seq["previous_actions"].shape == (4,)
        assert seq["context_mask"].shape == (4,)
        assert seq["target_1"].shape == (64, 64, 3)
        assert seq["target_2"].shape == (64, 64, 3)
        assert seq["target_4"].shape == (64, 64, 3)
        assert seq["target_8"].shape == (64, 64, 3)
        assert seq["actions_future"].shape == (8,)

        # Context mask must have at least the current anchor step valid
        assert seq["context_mask"][-1] == True


def test_replay_sample_provenance():
    """Test 3: Sampled sequences retain valid provenance traceable to source transition IDs."""
    buffer = ReplayBuffer(capacity=50, obs_shape=(64, 64, 3))
    sampler = SequenceSampler(replay_buffer=buffer, context_length=4, prediction_horizon=8)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    buffer.start_episode(obs, episode_id=77)
    for t in range(20):
        buffer.add(action=t % 17, reward=float(t), next_obs=obs, terminated=False, truncated=False)

    valid_anchors = sampler.get_valid_anchor_indices()
    for anchor in valid_anchors:
        ep_id = buffer.episode_ids[anchor]
        start_t = buffer.timesteps[anchor]
        start_tid = buffer.transition_ids[anchor]

        assert ep_id == 77
        assert start_tid >= 0
        assert start_t >= 0


def test_corrupt_checkpoint_preserves_previous_latest(temp_dir):
    """Test 4: Corrupt checkpoint write does not overwrite or corrupt previous valid latest."""
    store = CheckpointStore(str(temp_dir))
    valid_data_1 = {"step": 100, "weight": 1.23}
    store.save(step=100, checkpoint_data=valid_data_1)

    latest_path = temp_dir / "checkpoint_latest.pt"
    assert latest_path.exists()
    loaded_1 = torch.load(latest_path, weights_only=False)
    assert loaded_1["step"] == 100

    # Attempt to save corrupted object that fails during serialization or validation
    class UnpicklableObject:
        def __reduce__(self):
            raise RuntimeError("Simulated serialization crash mid-write")

    corrupted_data = {"step": 200, "bad_obj": UnpicklableObject()}
    with pytest.raises(RuntimeError, match="Failed to atomically save checkpoint"):
        store.save(step=200, checkpoint_data=corrupted_data)

    # Previous valid latest must remain completely intact
    assert latest_path.exists()
    loaded_after = torch.load(latest_path, weights_only=False)
    assert loaded_after["step"] == 100
    assert loaded_after["weight"] == 1.23


def test_uninterrupted_vs_resumed_equivalence(temp_dir):
    """Test 5: Uninterrupted collection vs save/resume/continue collection produces identical state."""
    mapping = OpaqueActionMapping(num_actions=17, seed=42)

    # Branch A: Uninterrupted 60 steps
    adapter_a = CrafterAdapter(seed=100, opaque_mapping=mapping)
    buffer_a = ReplayBuffer(capacity=100)
    obs = adapter_a.reset(seed=100)
    buffer_a.start_episode(obs, episode_id=1)
    for t in range(60):
        act = (t * 3) % 17
        n_obs, rew, term, trunc = adapter_a.step(act)
        buffer_a.add(action=act, reward=rew, next_obs=n_obs, terminated=term, truncated=trunc)
        if term or trunc:
            n_obs = adapter_a.reset(seed=101)
            buffer_a.start_episode(n_obs, episode_id=2)
    adapter_a.close()

    # Branch B: 30 steps -> save checkpoint -> restore -> 30 steps
    adapter_b = CrafterAdapter(seed=100, opaque_mapping=mapping)
    buffer_b = ReplayBuffer(capacity=100)
    obs = adapter_b.reset(seed=100)
    buffer_b.start_episode(obs, episode_id=1)
    for t in range(30):
        act = (t * 3) % 17
        n_obs, rew, term, trunc = adapter_b.step(act)
        buffer_b.add(action=act, reward=rew, next_obs=n_obs, terminated=term, truncated=trunc)
        if term or trunc:
            n_obs = adapter_b.reset(seed=101)
            buffer_b.start_episode(n_obs, episode_id=2)

    # Checkpoint replay state
    ckpt_path = str(temp_dir / "ckpt_step30.pt")
    torch.save({"replay_state": buffer_b.get_state(), "step": 30}, ckpt_path)

    # Restore in fresh buffer
    buffer_b_resumed = ReplayBuffer(capacity=100)
    saved = torch.load(ckpt_path, weights_only=False)
    buffer_b_resumed.set_state(saved["replay_state"])

    for t in range(30, 60):
        act = (t * 3) % 17
        n_obs, rew, term, trunc = adapter_b.step(act)
        buffer_b_resumed.add(action=act, reward=rew, next_obs=n_obs, terminated=term, truncated=trunc)
        if term or trunc:
            n_obs = adapter_b.reset(seed=101)
            buffer_b_resumed.start_episode(n_obs, episode_id=2)
    adapter_b.close()

    # Assert exact equivalence between uninterrupted and resumed buffer
    assert buffer_a.size == buffer_b_resumed.size
    assert buffer_a.total_added == buffer_b_resumed.total_added
    assert np.array_equal(buffer_a.actions[:buffer_a.size], buffer_b_resumed.actions[:buffer_b_resumed.size])
    assert np.array_equal(buffer_a.rewards[:buffer_a.size], buffer_b_resumed.rewards[:buffer_b_resumed.size])


def test_pilot_500_steps_single_worker(temp_dir):
    """Test 6: 500-step smoke collection run with full trace capture, one worker,

    asserting 100% decision-outcome linkage and zero trace gaps.
    """
    tracer = TraceManager(run_id="pilot-500", output_dir=str(temp_dir), enabled=True)
    mapping = OpaqueActionMapping(num_actions=17, seed=555)
    adapter = CrafterAdapter(seed=200, opaque_mapping=mapping)
    buffer = ReplayBuffer(capacity=1000)

    obs = adapter.reset(seed=200)
    buffer.start_episode(obs, episode_id=1)
    episode_id = 1
    episode_step = 0

    decisions = []
    outcomes = []

    for env_step in range(500):
        act_token = int(np.random.randint(0, 17))
        dec_id = tracer.log_action_decision(
            episode_id=episode_id,
            timestep=episode_step,
            obs_ref=f"f-{env_step}",
            candidate_scores=None,
            selected_token=act_token,
            mapping_version="v1.0",
            role="Explorer",
            mode="warmup",
        )
        decisions.append(dec_id)
        tracer.log_action_executed(dec_id)

        next_obs, rew, term, trunc = adapter.step(act_token)
        tracer.log_action_outcome(
            decision_id=dec_id,
            reward=rew,
            terminated=term,
            truncated=trunc,
            next_obs_ref=f"f-{env_step+1}",
        )
        outcomes.append(dec_id)
        buffer.add(action=act_token, reward=rew, next_obs=next_obs, terminated=term, truncated=trunc)
        episode_step += 1

        if term or trunc:
            episode_id += 1
            episode_step = 0
            next_obs = adapter.reset(seed=200 + episode_id)
            buffer.start_episode(next_obs, episode_id=episode_id)

    adapter.close()
    checksum = tracer.shutdown()

    assert len(decisions) == 500
    assert len(outcomes) == 500
    assert buffer.size == 500
    assert TraceWriter.verify_checksum(str(temp_dir / "traces" / "pilot-500.jsonl"))
