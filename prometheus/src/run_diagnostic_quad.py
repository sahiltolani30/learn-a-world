"""
Project Prometheus: Diagnostic Quad (Items A, B, C, D)
Comprehensive diagnostic runner requested for the senior forensic review:
  Item A: Hotfix-6 code status checks
  Item B: Scene-separation at Token vs Pooled vs Readout levels (Pooling-collapse test)
  Item C: Exact configuration dumped from state.pt
  Item D: Predictor action-sensitivity across all 17 Crafter actions
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import crafter

# Ensure prometheus dir is in sys.path
PROMETHEUS_DIR = Path(__file__).resolve().parent.parent
if str(PROMETHEUS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMETHEUS_DIR))

from src.agent.agent import JEPAAgent
from src.diagnostics import CRAFTER_ACTION_NAMES


def locate_checkpoint(custom_path: Optional[str] = None) -> Path:
    """Locates the latest state.pt or step checkpoint."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    candidates.extend([
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v3/state.pt"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v3/state.pt"),
        Path("checkpoints_v3/state.pt"),
        Path("../checkpoints_v3/state.pt"),
        PROMETHEUS_DIR / "checkpoints_v3/state.pt",
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v3/checkpoint_step_25000.pt"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v3/checkpoint_step_25000.pt"),
    ])

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Search in directory if exists
    for dir_path in [
        "/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v3",
        "/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v3",
        "checkpoints_v3",
        str(PROMETHEUS_DIR / "checkpoints_v3"),
    ]:
        p = Path(dir_path)
        if p.exists():
            pts = list(p.glob("*.pt"))
            if pts:
                pts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return pts[0]

    raise FileNotFoundError(f"Could not find state.pt or checkpoint in candidate paths: {[str(c) for c in candidates]}")


def run_item_a():
    print("## A. Hotfix-6 status (3 code checks — raw output)\n")
    buffer_py = PROMETHEUS_DIR / "src/memory/buffer.py"
    train_py = PROMETHEUS_DIR / "src/train.py"

    print("--- 1. Does the buffer store the memory state h? ---")
    if buffer_py.exists():
        with open(buffer_py, "r") as f:
            buf_lines = f.readlines()
        
        # grep -n "def add"
        add_matches = [(i + 1, line.rstrip()) for i, line in enumerate(buf_lines) if "def add" in line]
        for idx, line in add_matches[:2]:
            print(f"{idx}:{line}")
            
        # grep -n "t1_h\|'h'\|\"h\""
        h_matches = [(i + 1, line.rstrip()) for i, line in enumerate(buf_lines) if ("t1_h" in line or "'h'" in line or '"h"' in line)]
        for idx, line in h_matches[:8]:
            print(f"{idx}:{line}")
    else:
        print(f"Error: {buffer_py} not found")

    print("\n--- 2. What does actor training feed the actor? (lines 1000-1015 requested, plus actor lines 1110-1125) ---")
    if train_py.exists():
        with open(train_py, "r") as f:
            train_lines = f.readlines()
        print("sed -n '1000,1015p' src/train.py:")
        for line in train_lines[999:1015]:
            print(line.rstrip())
        print("\nsed -n '1112,1125p' src/train.py (Actor policy update):")
        for line in train_lines[1111:1125]:
            print(line.rstrip())
    else:
        print(f"Error: {train_py} not found")

    print("\n--- 3. What does collection pass? ---")
    if train_py.exists():
        readout_matches = [(i + 1, line.rstrip()) for i, line in enumerate(train_lines) if "get_readout" in line]
        for idx, line in readout_matches[:6]:
            print(f"{idx}:{line}")
    else:
        print(f"Error: {train_py} not found")
        
    print("\n" + "=" * 80 + "\n")


def build_canonical_scenes(device: torch.device) -> List[torch.Tensor]:
    stages = [
        ("spawn", 0, None, "grass"),
        ("wood2", 2, None, "grass"),
        ("table", 1, "table", "table"),
        ("stone", 0, None, "stone"),
    ]
    scenes = []
    for name, wood_cnt, facing_item, target_tile in stages:
        env = crafter.Env(seed=42)
        env.reset()
        p = env._player
        p.inventory["wood"] = wood_cnt
        if name == "stone":
            p.inventory["wood_pickaxe"] = 1
        px, py = p.pos
        dx, dy = p.facing
        env._world[px + dx, py + dy] = target_tile
        frame = env.render()
        t = torch.from_numpy(frame.transpose((2, 0, 1))).unsqueeze(0).to(device).float() / 255.0
        scenes.append(t)
    return scenes


def run_item_b(agent: JEPAAgent, scenes: List[torch.Tensor]):
    print("## B. Scene-separation at TOKEN level vs POOLED level (the pooling-collapse test — the decisive one)\n")
    agent.eval()
    with torch.no_grad():
        tokens, pooled = agent.encoder.forward_tokens(torch.cat(scenes, 0))   # (4,64,64), (4,256)
        readouts = torch.stack([agent.get_readout(pooled[i:i+1], None).squeeze(0) for i in range(4)])

    def cos(a, b): 
        return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

    names = ["spawn", "wood2", "table", "stone"]
    print("=== TOKEN-GRID cosines (flatten 64x64) ===")
    for i in range(4):
        for j in range(i+1, 4):
            print(f"{names[i]} vs {names[j]}: {cos(tokens[i], tokens[j]):.4f}")
    
    print("=== POOLED-256 cosines ===")
    for i in range(4):
        for j in range(i+1, 4):
            print(f"{names[i]} vs {names[j]}: {cos(pooled[i], pooled[j]):.4f}")
            
    print("=== READOUT cosines ===")
    for i in range(4):
        for j in range(i+1, 4):
            print(f"{names[i]} vs {names[j]}: {cos(readouts[i], readouts[j]):.4f}")
            
    tok_std = tokens.std(dim=0).flatten()   # per-position std across the 4 scenes
    top = torch.topk(tok_std, 8)
    print("Most scene-varying token positions:", top.indices.tolist())
    print("Their std values:", [round(v, 4) for v in top.values.tolist()])
    print("Min/mean token std:", round(tok_std.min().item(), 4), round(tok_std.mean().item(), 4))
    print("\n" + "=" * 80 + "\n")


def run_item_c(checkpoint_path: Path):
    print("## C. Config actually used in the run\n")
    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    metadata = ck.get("metadata", {})
    cfg = metadata.get("config", {})
    print(f"Checkpoint Loaded: {checkpoint_path}")
    print(f"Total Steps in Checkpoint: {metadata.get('step', 'unknown')}")
    keys = ['eps_base', 'eps_novelty_coef', 'eps_max', 'mask_ratio', 'token_dim', 'mem_hidden', 'si_weight', 'num_envs', 'batch_size', 'nstep']
    for k in keys:
        print(f"{k} = {cfg.get(k)}")
    print("\n" + "=" * 80 + "\n")


def run_item_d(agent: JEPAAgent, scene_b: torch.Tensor, device: torch.device):
    print("## D. Predictor action-sensitivity (1 scene, 17 numbers)\n")
    agent.eval()
    with torch.no_grad():
        tokens_b, pooled_b = agent.encoder.forward_tokens(scene_b)
        tokens_17 = tokens_b.repeat(17, 1, 1)
        actions = torch.arange(17, device=device)
        mean_preds, all_preds = agent.predictor.forward_tokens(tokens_17, actions, h=None)
        disc = agent.predictor.compute_epistemic_disagreement(all_preds)
        
        disc_vals = disc.cpu().numpy()
        print("Disagreement across 17 actions on Scene B (wood2, facing grass):")
        for act_idx, val in enumerate(disc_vals):
            act_name = CRAFTER_ACTION_NAMES[act_idx] if act_idx < len(CRAFTER_ACTION_NAMES) else str(act_idx)
            print(f"  Action {act_idx:2d} ({act_name:<16}): {val:.6f}")
            
        print("\nSummary Statistics:")
        print(f"  Min disagreement:    {disc_vals.min():.6f} ({CRAFTER_ACTION_NAMES[int(disc_vals.argmin())]})")
        print(f"  Max disagreement:    {disc_vals.max():.6f} ({CRAFTER_ACTION_NAMES[int(disc_vals.argmax())]})")
        print(f"  Spread (max - min):  {disc_vals.max() - disc_vals.min():.6f}")
        print(f"  Mean disagreement:   {disc_vals.mean():.6f}")
        print(f"  Std disagreement:    {disc_vals.std():.6f}")
    print("\n" + "=" * 80 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  PROJECT PROMETHEUS: FORENSIC AUDIT (ITEMS A, B, C, D) ON {device.type.upper()}")
    print("=" * 80 + "\n")

    # Item A: Code checks
    run_item_a()

    # Checkpoint resolution
    custom_ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = locate_checkpoint(custom_ckpt)

    # Item C: Config check
    run_item_c(ckpt_path)

    # Load agent
    print(f"Loading JEPAAgent weights from: {ckpt_path}...")
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    agent = JEPAAgent(latent_dim=256, num_actions=17, device=device)
    
    # Robust weights loading
    if "components" in ck and "agent" in ck["components"]:
        agent_comp = ck["components"]["agent"]
        if hasattr(agent_comp, "state_dict"):
            agent.load_state_dict(agent_comp.state_dict())
        elif isinstance(agent_comp, dict):
            agent.load_state_dict(agent_comp)
        else:
            agent.load_state_dict(agent_comp)
    elif "agent_state_dict" in ck:
        agent.load_state_dict(ck["agent_state_dict"])
    else:
        agent.load_state_dict(ck)
        
    agent.eval()
    print("Agent weights successfully loaded!\n")

    # Build canonical scenes
    scenes = build_canonical_scenes(device)

    # Item B: Scene separation
    run_item_b(agent, scenes)

    # Item D: Predictor action sensitivity on Scene 1 (wood2)
    run_item_d(agent, scenes[1], device)


if __name__ == "__main__":
    main()
