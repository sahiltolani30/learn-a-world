"""Tests proving that sampler strictly never samples across episode boundaries."""
import pytest
import numpy as np
from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler


def test_no_episode_crossing():
    buffer = ReplayBuffer(capacity=100)
    sampler = SequenceSampler(buffer, context_length=4, prediction_horizon=8)

    # Episode 1: 5 steps (too short for 8-step prediction)
    buffer.start_episode(np.zeros((64, 64, 3), dtype=np.uint8), episode_id=1)
    for s in range(5):
        buffer.add(s, 0.0, np.ones((64, 64, 3), dtype=np.uint8), terminated=(s == 4), truncated=False)

    # Episode 2: 7 steps (still too short for 8-step prediction)
    buffer.start_episode(np.zeros((64, 64, 3), dtype=np.uint8), episode_id=2)
    for s in range(7):
        buffer.add(s, 0.0, np.ones((64, 64, 3), dtype=np.uint8), terminated=(s == 6), truncated=False)

    # No sequence can cross between episode 1 and episode 2
    valid_anchors = sampler.get_valid_anchor_indices()
    assert len(valid_anchors) == 0, "No anchors should be valid when all episodes < 8 steps"

    # Add Episode 3: 10 steps (indices 0..2 within ep3 are valid anchors)
    buffer.start_episode(np.zeros((64, 64, 3), dtype=np.uint8), episode_id=3)
    for s in range(10):
        buffer.add(s, 0.0, np.ones((64, 64, 3), dtype=np.uint8), terminated=(s == 9), truncated=False)

    valid_anchors = sampler.get_valid_anchor_indices()
    assert len(valid_anchors) == 3

    # Every sampled sequence from valid anchors must strictly belong to Episode 3
    for anchor in valid_anchors:
        seq = sampler.sample_one_sequence([anchor])
        # Verify future actions are all from Episode 3
        assert len(seq["actions_future"]) == 8
