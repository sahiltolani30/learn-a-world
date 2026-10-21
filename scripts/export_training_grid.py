#!/usr/bin/env python3
"""
Exports visual training sequences directly showing the game frames used by the JEPA model.
Generates:
1. `docs/assets/maze_training_sequence.png`: 8 consecutive transition frames in the maze
2. `docs/assets/crafter_training_sequence.png`: 8 consecutive transition frames in Crafter
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym
import envs

def create_sequence_strip(env_name: str, out_path: Path, steps: int = 8, scale: int = 4):
    env = gym.make(env_name)
    obs, info = env.reset(seed=42)

    frames = []
    actions = []

    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, info = env.step(action)
        frames.append(obs)
        actions.append(action)
        obs = next_obs
        if term or trunc:
            obs, info = env.reset()

    env.close()

    # Create horizontal strip
    H, W, C = frames[0].shape
    cell_w, cell_h = W * scale, H * scale
    banner_h = 32
    total_w = cell_w * steps
    total_h = cell_h + banner_h

    strip = Image.new("RGB", (total_w, total_h), (25, 30, 40))
    draw = ImageDraw.Draw(strip)

    for i, (f, a) in enumerate(zip(frames, actions)):
        img = Image.fromarray(f).resize((cell_w, cell_h), Image.NEAREST)
        x = i * cell_w
        strip.paste(img, (x, banner_h))

        # Divider line
        if i > 0:
            draw.line([(x, 0), (x, total_h)], fill=(70, 80, 95), width=2)

        # Label text
        label = f"Step t={i} | Act: {a}"
        draw.text((x + 8, 8), label, fill=(230, 240, 255))

    strip.save(out_path)
    print(f"    [SAVED] Training strip saved to: {out_path}")

def main():
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("--> Generating training frame sequence strips...")
    create_sequence_strip("GridMaze-v0", assets_dir / "maze_training_sequence.png")
    create_sequence_strip("Crafter-v0", assets_dir / "crafter_training_sequence.png")
    create_sequence_strip("PushT-v0", assets_dir / "pusht_training_sequence.png")
    print("\nAll training sequence visualizations ready!")

if __name__ == "__main__":
    main()
