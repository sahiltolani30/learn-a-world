#!/usr/bin/env python3
"""
Visualize the Push-T environment and physics.
Saves:
1. Snapshot of target, T-block, and agent: `docs/assets/pusht_agent_world.png`
2. Animated GIF of agent pushing the block: `docs/assets/pusht_gameplay.gif`
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image
import gymnasium as gym
import envs  # registers PushT-v0

def main():
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("--> Initializing Push-T 2D physics environment...")
    env = gym.make("PushT-v0")
    obs, info = env.reset(seed=42)

    scale = 8  # 64x64 -> 512x512
    img = Image.fromarray(obs)
    img_large = img.resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
    snapshot_path = assets_dir / "pusht_agent_world.png"
    img_large.save(snapshot_path)
    print(f"    [SAVED] Push-T snapshot saved to: {snapshot_path}")

    frames = [img_large]
    print("--> Stepping agent in Push-T to record pushing animation...")
    # Directed push downward toward block
    for step in range(40):
        # Action 2 is DOWN, action 4 is RIGHT, etc.
        action = 2 if step < 15 else (4 if step < 25 else env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        frame = Image.fromarray(obs)
        frame_large = frame.resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
        frames.append(frame_large)
        if terminated or truncated:
            break

    gif_path = assets_dir / "pusht_gameplay.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    print(f"    [SAVED] Push-T animation saved to: {gif_path}")
    print("\nVisual files ready:")
    print(f"  - Snapshot: {snapshot_path.resolve()}")
    print(f"  - Animation: {gif_path.resolve()}")

if __name__ == "__main__":
    main()
