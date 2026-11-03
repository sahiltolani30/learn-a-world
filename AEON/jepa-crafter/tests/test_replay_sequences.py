"""Tests for replay buffer and sequence sampler alignment."""
import pytest
import numpy as np
import torch
from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler


def test_replay_sequence_sampling():
    buffer = ReplayBuffer(capacity=100)
    sampler = SequenceSampler(buffer, context_length=4, prediction_horizon=8)

    # Populate two episodes: episode 1 with 15 steps, episode 2 with 10 steps
    for ep in range(1, 3):
        init_obs = np.full((64, 64, 3), ep * 10, dtype=np.uint8)
        buffer.start_episode(init_obs, episode_id=ep)
        num_steps = 15 if ep == 1 else 10
        for s in range(num_steps):
            action = s % 17
            reward = float(s)
            next_obs = np.full((64, 64, 3), ep * 10 + s + 1, dtype=np.uint8)
            done = (s == num_steps - 1)
            buffer.add(action, reward, next_obs, terminated=done, truncated=False)

    valid_anchors = sampler.get_valid_anchor_indices()
    assert len(valid_anchors) > 0

    # Sample batch
    batch = sampler.sample_batch(batch_size=4)
    assert batch["obs_context"].shape == (4, 4, 3, 64, 64)
    assert batch["actions_future"].shape == (4, 8)
    assert batch["target_1"].shape == (4, 3, 64, 64)
    assert batch["target_2"].shape == (4, 3, 64, 64)
    assert batch["target_4"].shape == (4, 3, 64, 64)
    assert batch["target_8"].shape == (4, 3, 64, 64)
    assert batch["context_mask"].shape == (4, 4)
    assert batch["previous_actions"].shape == (4, 4)
    assert batch["reward_first"].shape == (4, 1)
    assert batch["done_first"].shape == (4, 1)


def test_context_left_padding():
    buffer = ReplayBuffer(capacity=50)
    sampler = SequenceSampler(buffer, context_length=4, prediction_horizon=8)

    # Episode of length 9: anchor 0 has no preceding context
    init_obs = np.full((64, 64, 3), 1, dtype=np.uint8)
    buffer.start_episode(init_obs, episode_id=1)
    for s in range(9):
        action = s % 17
        reward = 0.5
        next_obs = np.full((64, 64, 3), s + 2, dtype=np.uint8)
        done = (s == 8)
        buffer.add(action, reward, next_obs, terminated=done, truncated=False)

    assert sampler.is_valid_anchor(0)
    seq = sampler.sample_one_sequence([0])
    # For anchor 0: context slots 0, 1, 2 are unobserved (padded False), slot 3 is O[0] (True)
    assert seq["context_mask"][0] == False
    assert seq["context_mask"][1] == False
    assert seq["context_mask"][2] == False
    assert seq["context_mask"][3] == True
    assert (seq["obs_context"][3] == 1).all()
