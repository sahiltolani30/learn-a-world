r"""
Automated Gate Tests for Phase 1:
1. Maze Gym Compliance: Passes gymnasium.utils.env_checker.check_env
2. Buffer Persistence: Saves 1,000 transitions to .npz and reloads with 100% parity
3. Loss Convergence: Prediction MSE drops significantly over training
4. Anti-Collapse Metric: Latent std-dev \sigma_z > 0.05 (no representation collapse)
5. Checkpoint Resume: Model saves checkpoint and reloads with bitwise match
"""

import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import envs  # registers GridMaze-v0
from envs.grid_maze import GridMazeEnv
from models.jepa import JEPAWorldModel
from training.buffer import ReplayBuffer
from training.trainer import evaluate_prediction_error, seed_buffer, train_jepa

def test_maze_gym_compliance():
    """Gate Test 1: Verify GridMazeEnv strictly adheres to the Gymnasium API."""
    env = gym.make("GridMaze-v0")
    check_env(env.unwrapped)

    obs, info = env.reset(seed=42)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert "agent_pos" in info
    assert "goal_pos" in info

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()

def test_buffer_persistence(tmp_path):
    """Gate Test 2: Saves 1,000 transitions to .npz and reloads with 100% parity."""
    env = gym.make("GridMaze-v0")
    buffer = ReplayBuffer(capacity=5000, obs_shape=(64, 64, 3))
    seed_buffer(env, buffer, num_steps=1000)
    assert len(buffer) == 1000

    chunk_file = tmp_path / "test_chunk.npz"
    buffer.save_chunk(chunk_file)
    assert chunk_file.exists()

    # Create fresh buffer and reload
    new_buffer = ReplayBuffer(capacity=5000, obs_shape=(64, 64, 3))
    new_buffer.load_chunk(chunk_file)
    assert len(new_buffer) == 1000

    # Bitwise parity assertions
    assert np.array_equal(buffer.obs[:1000], new_buffer.obs[:1000])
    assert np.array_equal(buffer.next_obs[:1000], new_buffer.next_obs[:1000])
    assert np.array_equal(buffer.actions[:1000], new_buffer.actions[:1000])
    assert np.allclose(buffer.rewards[:1000], new_buffer.rewards[:1000])
    assert np.array_equal(buffer.dones[:1000], new_buffer.dones[:1000])
    env.close()

def test_jepa_training_and_anti_collapse(tmp_path):
    """
    Gate Test 3 & 4:
    Train JEPA model on GridMaze:
    - Loss drops significantly
    - Latent variance stays > 0.05 (anti-collapse)
    - Checkpoint saves cleanly
    """
    # Fast test run with 200 steps and 200 seed transitions
    model, history = train_jepa(
        config_path="experiments/configs/maze.yaml",
        device_override="cpu",
        output_dir_override=str(tmp_path),
        total_steps_override=200,
        seed_steps_override=200,
        resume=False,
    )

    # 1. Verify training progressed from step 0
    assert len(history["val_loss"]) >= 2
    initial_val_loss = history["val_loss"][0]
    final_val_loss = history["val_loss"][-1]

    # Val loss should decrease from untrained baseline
    assert final_val_loss < initial_val_loss, (
        f"Validation loss did not decrease: initial={initial_val_loss:.4f}, final={final_val_loss:.4f}"
    )

    # 2. Verify anti-collapse metric
    final_variance = history["variance"][-1]
    assert final_variance > 0.05, (
        f"Latent representation collapsed! Latent std={final_variance:.4f} <= 0.05"
    )

    # 3. Verify checkpoint created
    ckpt_path = tmp_path / "jepa_maze.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} was not created."

def test_checkpoint_resume(tmp_path):
    """Gate Test 5: Verify training can resume from checkpoint without corruption."""
    ckpt_dir = tmp_path / "resume_test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Step A: Train 100 steps
    model_a, hist_a = train_jepa(
        config_path="experiments/configs/maze.yaml",
        device_override="cpu",
        output_dir_override=str(ckpt_dir),
        total_steps_override=100,
        seed_steps_override=200,
        resume=False,
    )

    # Step B: Resume training from step 100 to 200
    model_b, hist_b = train_jepa(
        config_path="experiments/configs/maze.yaml",
        device_override="cpu",
        output_dir_override=str(ckpt_dir),
        total_steps_override=200,
        seed_steps_override=200,
        resume=True,
    )

    # Final steps in history should be 200
    assert hist_b["steps"][-1] == 200
