"""Tests for parallel multi-environment collection (VectorCrafterAdapter & BatchedExperienceCollector)."""
import pytest
import numpy as np
import torch

from src.env.vector_adapter import VectorCrafterAdapter
from src.env.evaluator_tap import EvaluatorTap
from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler
from src.exploration.action_selector import ActionSelector
from src.training.collect import BatchedExperienceCollector


def test_vector_adapter_lockstep():
    tap = EvaluatorTap()
    num_envs = 4
    vec_adapter = VectorCrafterAdapter(num_envs=num_envs, evaluator_tap=tap, seed_start=500)
    
    obs = vec_adapter.reset()
    assert obs.shape == (num_envs, 64, 64, 3)
    assert obs.dtype == np.uint8

    # All envs must share the same opaque permutation
    perm0 = vec_adapter.adapters[0].action_mapping.permutation
    for i in range(1, num_envs):
        assert vec_adapter.adapters[i].action_mapping.permutation == perm0

    actions = [i % 17 for i in range(num_envs)]
    results = vec_adapter.step(actions)
    assert len(results) == num_envs
    for next_obs, rew, term, trunc in results:
        assert next_obs.shape == (64, 64, 3)
        assert isinstance(rew, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    vec_adapter.close()


def test_batched_collector_contiguous_episodes():
    tap = EvaluatorTap()
    num_envs = 4
    vec_adapter = VectorCrafterAdapter(num_envs=num_envs, evaluator_tap=tap, seed_start=600, max_episode_steps=20)
    replay = ReplayBuffer(capacity=1000)
    action_selector = ActionSelector()

    collector = BatchedExperienceCollector(
        vector_adapter=vec_adapter,
        replay_buffer=replay,
        action_selector=action_selector,
    )

    # Step until at least one episode completes
    for _ in range(50):
        collector.step(mode="warmup")
        if collector.completed_episodes >= 2:
            break

    assert collector.completed_episodes >= 2
    assert replay.size > 0

    # Ensure all transitions added to replay buffer belong to contiguous valid episodes
    sampler = SequenceSampler(replay, context_length=4, prediction_horizon=8)
    # Check that any sampled sequences do not cross episode IDs
    valid_anchors = sampler.get_valid_anchor_indices()
    for anchor in valid_anchors:
        seq = sampler.sample_one_sequence([anchor])
        assert len(seq["actions_future"]) == 8

    vec_adapter.close()
