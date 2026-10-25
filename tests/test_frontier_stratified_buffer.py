"""
tests/test_frontier_stratified_buffer.py
Unit tests for Prioritized Experience Replay (PER) frontier-stratified sampling in EpisodeBuffer.
"""

import numpy as np
import pytest
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS
from training.sequence_buffer import EpisodeBuffer, NUM_ACHIEVEMENTS


def test_frontier_stratified_sampling():
    buffer = EpisodeBuffer(capacity_transitions=50_000)
    
    # Create 5 episodes:
    # 4 standard episodes with only common events (sapling, plant)
    # 1 frontier episode with table (20) and pickaxe (15)
    idx_pick = CRAFTER_ACHIEVEMENTS.index("make_wood_pickaxe")
    idx_table = CRAFTER_ACHIEVEMENTS.index("place_table")
    idx_sapling = CRAFTER_ACHIEVEMENTS.index("collect_sapling")
    
    for ep_i in range(5):
        T = 50
        obs = np.zeros((T, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randint(0, 17, size=T, dtype=np.int64)
        rewards = np.zeros(T, dtype=np.float32)
        dones = np.zeros(T, dtype=bool)
        events = np.zeros((T, NUM_ACHIEVEMENTS), dtype=np.uint8)
        mask_before = np.zeros((T, NUM_ACHIEVEMENTS), dtype=np.uint8)
        
        if ep_i == 0:
            # Frontier episode!
            events[10, idx_table] = 1
            events[15, idx_pick] = 1
            rewards[10] = 1.0
            rewards[15] = 1.0
        else:
            # Common episode
            events[5, idx_sapling] = 1
            rewards[5] = 1.0
            
        buffer.add_episode(obs, actions, rewards, dones, events, mask_before)
        
    assert buffer.num_episodes == 5
    assert buffer.total_transitions == 250
    
    # Sample with frontier_stratified mode
    batch = buffer.sample_sequences(batch_size=16, seq_len=16, mode="frontier_stratified")
    
    assert batch["obs"].shape == (16, 16, 3, 64, 64)
    assert batch["actions"].shape == (16, 16)
    assert batch["events"].shape == (16, 16, NUM_ACHIEVEMENTS)
    
    # In frontier_stratified mode, at least one sequence MUST contain frontier events
    frontier_events_sampled = (batch["events"][:, :, idx_pick].sum() + batch["events"][:, :, idx_table].sum()).item()
    assert frontier_events_sampled > 0, "Frontier stratified sampling must include frontier events"
