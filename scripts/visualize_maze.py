#!/usr/bin/env python3
"""
Visualize the Custom Grid Maze environment.
Saves:
1. Snapshot of initial maze state: `docs/assets/maze_agent_world.png`
2. Animated GIF of agent movement: `docs/assets/maze_gameplay.gif`
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image
import gymnasium as gym
import envs  # registers GridMaze-v0

def main():
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("--> Initializing Custom Grid Maze environment...")
    env = gym.make("GridMaze-v0")
    obs, info = env.reset(seed=42)

    scale = 8  # 64x64 -> 512x512
    img = Image.fromarray(obs)
    img_large = img.resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
    snapshot_path = assets_dir / "maze_agent_world.png"
    img_large.save(snapshot_path)
    print(f"    [SAVED] Maze snapshot saved to: {snapshot_path}")

    frames = [img_large]
    print("--> Stepping agent in the maze...")
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = Image.fromarray(obs)
        frame_large = frame.resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
        frames.append(frame_large)
        if terminated or truncated:
            obs, info = env.reset()

    gif_path = assets_dir / "maze_gameplay.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"    [SAVED] Maze animation saved to: {gif_path}")
    print("\nVisual files ready:")
    print(f"  - Snapshot: {snapshot_path.resolve()}")
    print(f"  - Animation: {gif_path.resolve()}")

if __name__ == "__main__":
    main()
