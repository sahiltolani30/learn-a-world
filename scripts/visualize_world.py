#!/usr/bin/env python3
"""
Visualize the Crafter agent and game world.
Saves:
1. A crisp high-resolution snapshot: `docs/assets/crafter_agent_world.png`
2. An animated gameplay GIF: `docs/assets/crafter_gameplay.gif`
"""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image
import crafter

def main():
    assets_dir = Path("docs/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    print("--> Initializing Crafter game world...")
    env = crafter.Env(seed=42)
    obs = env.reset()
    
    # Scale up for crisp pixel art display (e.g. 64x64 -> 512x512)
    scale = 8
    
    # 1. Save single snapshot
    img = Image.fromarray(obs)
    img_large = img.resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
    snapshot_path = assets_dir / "crafter_agent_world.png"
    img_large.save(snapshot_path)
    print(f"    [SAVED] Agent & world snapshot saved to: {snapshot_path}")
    
    # 2. Record 40 steps and save an animated GIF
    print("--> Stepping agent through the world to record animation...")
    frames = [img_large]
    
    for step in range(40):
        # sample an action (movement, chop, etc.)
        action = np.random.randint(0, env.action_space.n)
        obs, reward, done, info = env.step(action)
        
        frame = Image.fromarray(obs)
        frame_large = frame.resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
        frames.append(frame_large)
        if done:
            break
            
    gif_path = assets_dir / "crafter_gameplay.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0
    )
    print(f"    [SAVED] Gameplay animation saved to: {gif_path}")
    print("\nOpen these files to view the world and agent:")
    print(f"  - Snapshot: {snapshot_path.resolve()}")
    print(f"  - Animation: {gif_path.resolve()}")

if __name__ == "__main__":
    main()
