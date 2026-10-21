"""Environments module for Learn-a-World."""

import gymnasium as gym
from envs.crafter_wrapper import CrafterEnv
from envs.grid_maze import GridMazeEnv

# Register custom GridMaze environments
gym.register(
    id="GridMaze-v0",
    entry_point="envs.grid_maze:GridMazeEnv",
    max_episode_steps=100,
)

__all__ = ["CrafterEnv", "GridMazeEnv"]
