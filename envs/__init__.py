"""Environments module for Learn-a-World."""

import gymnasium as gym
from envs.adaptation_wrapper import DynamicsPerturbationWrapper
from envs.crafter_wrapper import CrafterEnv
from envs.grid_maze import GridMazeEnv
from envs.push_t import PushTEnv


# Register environments
gym.register(
    id="GridMaze-v0",
    entry_point="envs.grid_maze:GridMazeEnv",
    max_episode_steps=100,
)

gym.register(
    id="Crafter-v0",
    entry_point="envs.crafter_wrapper:CrafterEnv",
    max_episode_steps=10000,
)

gym.register(
    id="PushT-v0",
    entry_point="envs.push_t:PushTEnv",
    max_episode_steps=100,
)

__all__ = ["CrafterEnv", "GridMazeEnv", "PushTEnv", "DynamicsPerturbationWrapper"]



