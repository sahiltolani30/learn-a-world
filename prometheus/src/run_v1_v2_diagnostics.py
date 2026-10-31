"""
V1 & V2 Verification Suite for Prometheus v5 Checkpoint.
V1: Corrected Table-Persistence Scan over Replay Buffer (de-interleaving 8 parallel envs).
V2: Scene-Cosine Probe on v5 Checkpoint (M1: Token-Grid & Pooled Cosines on 4 Canonical Scenes).
"""

import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import crafter

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent import JEPAAgent
from src.memory.storage import CheckpointManager
from src.diagnostics import CRAFTER_ACTION_NAMES


def find_v5_checkpoint() -> Path:
    """Finds state.pt or latest checkpoint in checkpoints_v5 or candidate paths."""
    candidates = [
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v5/state.pt"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v5/state.pt"),
        ROOT_DIR / "checkpoints_v5/state.pt",
        Path("./checkpoints_v5/state.pt"),
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v4/state.pt"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v4/state.pt"),
        ROOT_DIR / "checkpoints_v4/state.pt",
        Path("./checkpoints_v4/state.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
            
    for d in [
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v5"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v5"),
        ROOT_DIR / "checkpoints_v5",
        Path("./checkpoints_v5"),
    ]:
        if d.exists():
            pts = sorted(list(d.glob("checkpoint_step_*.pt")), key=lambda x: x.stat().st_mtime, reverse=True)
            if pts:
                return pts[0]
                
    raise FileNotFoundError("Could not find state.pt in checkpoints_v5.")


# ==============================================================================
# V1: Corrected Table-Persistence Scan (De-interleaved Episode Reconstruction)
# ==============================================================================
def run_v1_corrected_table_persistence(raw_ckpt: Dict[str, Any]):
    print("\n" + "=" * 85)
    print(" 🪑 V1 — CORRECTED TABLE-PERSISTENCE SCAN (De-interleaved Episode Reconstruction)")
    print("=" * 85)
    
    comp = raw_ckpt.get("components", {})
    buf = comp.get("buffer") or raw_ckpt.get("buffer_arrays")
    
    if buf is None:
        print("[!] No buffer arrays found in checkpoint components.")
        print("=" * 85)
        return
        
    t1_actions = buf.get("t1_actions")
    t1_episodes = buf.get("t1_episode_id")
    t1_steps = buf.get("t1_step_in_ep")
    t1_size = int(buf.get("t1_size", 0))
    t1_milestones = buf.get("t1_is_milestone")
    
    if t1_actions is None or t1_size == 0:
        print("[!] Buffer is empty or t1_actions missing.")
        print("=" * 85)
        return
        
    act_pt = 8     # place_table action id
    act_pick = 11  # make_wood_pickaxe action id
    act_sword = 14 # make_wood_sword action id
    act_do = 5     # do action id
    nav_actions = {1, 2, 3, 4} # move_left, move_right, move_up, move_down
    
    # 1. Build Episode Timeline Map
    print(f"Scanning Tier 1 Replay Buffer ({t1_size} transitions)...")
    ep_timeline = defaultdict(list)
    for idx in range(t1_size):
        ep_id = int(t1_episodes[idx]) if t1_episodes is not None else 0
        step_in_ep = int(t1_steps[idx]) if t1_steps is not None else idx
        act = int(t1_actions[idx])
        is_ms = bool(t1_milestones[idx]) if t1_milestones is not None else False
        ep_timeline[ep_id].append((step_in_ep, act, is_ms, idx))
        
    # Sort transitions within each episode by step_in_ep
    for ep_id in ep_timeline:
        ep_timeline[ep_id].sort(key=lambda x: x[0])
        
    # 2. Find all place_table occurrences
    table_events = []
    for idx in range(t1_size):
        if t1_actions[idx] == act_pt:
            ep_id = int(t1_episodes[idx]) if t1_episodes is not None else 0
            step_in_ep = int(t1_steps[idx]) if t1_steps is not None else idx
            table_events.append((ep_id, step_in_ep, idx))
            
    print(f"Total 'place_table' (Action 8) Transitions Recorded: {len(table_events)}")
    print(f"Total Unique Episodes Recorded in Buffer:           {len(ep_timeline)}")
    
    if len(table_events) == 0:
        print("[!] Zero 'place_table' actions found in Tier 1 buffer.")
        print("=" * 85)
        return
        
    # 3. Analyze the exact 30-step window following each table placement
    total_valid_windows = 0
    total_window_steps = 0
    
    act_counts_in_window = defaultdict(int)
    steps_until_first_move = []
    immediate_walk_away_count = 0  # move within <= 2 steps
    pickaxe_attempted_in_window = 0
    sword_attempted_in_window = 0
    do_attempted_in_window = 0
    
    for ep_id, t_step, buf_idx in table_events:
        ep_history = ep_timeline[ep_id]
        
        # Extract transitions in the same episode where: t_step < step <= t_step + 30
        following = [
            (s, a, ms, i) for (s, a, ms, i) in ep_history
            if 0 < (s - t_step) <= 30
        ]
        
        if len(following) == 0:
            continue
            
        total_valid_windows += 1
        total_window_steps += len(following)
        
        # Check first movement step
        first_move = None
        for rel_step, (s, a, ms, i) in enumerate(following, start=1):
            act_counts_in_window[a] += 1
            if a in nav_actions and first_move is None:
                first_move = rel_step
                
        if first_move is not None:
            steps_until_first_move.append(first_move)
            if first_move <= 2:
                immediate_walk_away_count += 1
        else:
            steps_until_first_move.append(len(following) + 1)
            
        acts_in_this_win = {a for (s, a, ms, i) in following}
        if act_pick in acts_in_this_win:
            pickaxe_attempted_in_window += 1
        if act_sword in acts_in_this_win:
            sword_attempted_in_window += 1
        if act_do in acts_in_this_win:
            do_attempted_in_window += 1
            
    print(f"\n--- Episode De-Interleaved Scan Results ---")
    print(f"Total place_table occurrences:           {len(table_events)}")
    print(f"Valid 30-Step Windows Analyzed:          {total_valid_windows} (vs N=4 in interleaved scan)")
    print(f"Total Post-Table Steps Tracked:          {total_window_steps}")
    
    nav_total = sum(act_counts_in_window[a] for a in nav_actions)
    do_total = act_counts_in_window[act_do]
    pick_total = act_counts_in_window[act_pick]
    sword_total = act_counts_in_window[act_sword]
    sleep_total = act_counts_in_window[6]
    other_total = total_window_steps - (nav_total + do_total + pick_total + sword_total + sleep_total)
    
    print("\n--- Action Breakdown Across All Post-Table Windows ---")
    print(f"  Navigation (move_left/right/up/down): {nav_total:6} ({nav_total/max(1,total_window_steps)*100:5.1f}%)")
    print(f"  Interaction ('do'):                   {do_total:6} ({do_total/max(1,total_window_steps)*100:5.1f}%)")
    print(f"  Craft Pickaxe ('make_wood_pickaxe'):  {pick_total:6} ({pick_total/max(1,total_window_steps)*100:5.1f}%)")
    print(f"  Craft Sword ('make_wood_sword'):      {sword_total:6} ({sword_total/max(1,total_window_steps)*100:5.1f}%)")
    print(f"  Sleep ('sleep'):                      {sleep_total:6} ({sleep_total/max(1,total_window_steps)*100:5.1f}%)")
    print(f"  Other actions:                        {other_total:6} ({other_total/max(1,total_window_steps)*100:5.1f}%)")
    
    mean_steps_before_move = float(np.mean(steps_until_first_move)) if steps_until_first_move else 0.0
    walk_off_rate = (immediate_walk_away_count / max(1, total_valid_windows)) * 100
    
    print("\n--- Workstation Persistence / Amnesia Evaluation ---")
    print(f"  Mean steps before first movement:     {mean_steps_before_move:.2f} steps")
    print(f"  Immediate walk-off rate (\u2264 2 steps):   {walk_off_rate:.1f}% ({immediate_walk_away_count}/{total_valid_windows} windows)")
    print(f"  Windows with make_wood_pickaxe trial: {pickaxe_attempted_in_window}/{total_valid_windows} ({pickaxe_attempted_in_window/max(1,total_valid_windows)*100:.2f}%)")
    print(f"  Windows with make_wood_sword trial:   {sword_attempted_in_window}/{total_valid_windows} ({sword_attempted_in_window/max(1,total_valid_windows)*100:.2f}%)")
    print(f"  Windows with 'do' action trial:       {do_attempted_in_window}/{total_valid_windows} ({do_attempted_in_window/max(1,total_valid_windows)*100:.2f}%)")
    
    print("\n>>> V1 EMPIRICAL VERDICT:")
    if walk_off_rate >= 80.0:
        print(f"    CONFIRMED BEHAVIORAL AMNESIA: Agent walks off immediately in {walk_off_rate:.1f}% of windows across {total_valid_windows} verified placements.")
    else:
        print(f"    SCANNING ARTIFACT RESOLVED: Actual walk-off rate is {walk_off_rate:.1f}%; agent remains near table for an average of {mean_steps_before_move:.2f} steps.")
    print("=" * 85)


# ==============================================================================
# V2: Scene-Cosine Probe on v5 Checkpoint (M1)
# ==============================================================================
def run_v2_scene_cosine_probe(agent: JEPAAgent, device: torch.device):
    print("\n" + "=" * 85)
    print(" 🔬 V2 — SCENE-COSINE PROBE ON V5 CHECKPOINT (The In-Vivo Representation Test)")
    print("=" * 85)
    
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
        
    agent.eval()
    with torch.no_grad():
        scenes_tensor = torch.cat(scenes, dim=0)  # (4, 3, 64, 64)
        tokens, pooled = agent.encoder.forward_tokens(scenes_tensor)  # (4, 64, 64), (4, 256)
        readouts = torch.stack([agent.get_readout(pooled[i:i+1], None).squeeze(0) for i in range(4)])
        
    def cos_sim(a, b):
        return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
        
    names = ["spawn", "wood2", "table", "stone"]
    
    print("\n[A] TOKEN-GRID COSINES (Flatten 64 tokens x 64 dim = 4096-d):")
    print(f"    {'Pair Comparison':<22} | {'Cosine':<10} | {'Status'}")
    print("    " + "-" * 50)
    tok_cosines = []
    for i in range(4):
        for j in range(i+1, 4):
            c = cos_sim(tokens[i], tokens[j])
            tok_cosines.append(c)
            status = "SEPARATED" if c < 0.90 else "COLLAPSED"
            print(f"    {names[i] + ' vs ' + names[j]:<22} | {c:<10.4f} | {status}")
            
    mean_tok_cos = float(np.mean(tok_cosines))
    
    print(f"\n    --> Mean Token-Grid Cosine: {mean_tok_cos:.4f}")
    print(f"        (Pre-W1a was 0.9980; Target is \u2264 0.90)")
    
    print("\n[B] POOLED-256 LATENT COSINES:")
    print(f"    {'Pair Comparison':<22} | {'Cosine':<10}")
    print("    " + "-" * 38)
    pool_cosines = []
    for i in range(4):
        for j in range(i+1, 4):
            c = cos_sim(pooled[i], pooled[j])
            pool_cosines.append(c)
            print(f"    {names[i] + ' vs ' + names[j]:<22} | {c:<10.4f}")
    mean_pool_cos = float(np.mean(pool_cosines))
    print(f"    --> Mean Pooled-256 Cosine: {mean_pool_cos:.4f}")
    
    print("\n[C] READOUT COSINES (Vision + Memory Projected to 256-d):")
    print(f"    {'Pair Comparison':<22} | {'Cosine':<10}")
    print("    " + "-" * 38)
    for i in range(4):
        for j in range(i+1, 4):
            c = cos_sim(readouts[i], readouts[j])
            print(f"    {names[i] + ' vs ' + names[j]:<22} | {c:<10.4f}")
            
    # Per-token variance across the 4 scenes
    tok_std = tokens.std(dim=0).flatten()
    mean_std = float(tok_std.mean().item())
    min_std = float(tok_std.min().item())
    
    print(f"\n[D] PER-TOKEN VARIANCE SPREAD ACROSS SCENES:")
    print(f"    Mean Token Std across 4 scenes: {mean_std:.4f} (Pre-W1a was 0.0024)")
    print(f"    Min Token Std across 4 scenes:  {min_std:.4f}")
    
    print("\n>>> V2 EMPIRICAL VERDICT:")
    table_vs_grass_tok = cos_sim(tokens[0], tokens[2])
    table_vs_grass_pool = cos_sim(pooled[0], pooled[2])
    print(f"    'spawn (grass)' vs 'table' Token Cosine:  {table_vs_grass_tok:.4f}")
    print(f"    'spawn (grass)' vs 'table' Pooled Cosine: {table_vs_grass_pool:.4f}")
    
    if table_vs_grass_tok <= 0.90:
        print("    [PASS] W1A ENCODER IS HEALTHY: The encoder distinctly discriminates 'at table' from 'at grass'.")
    else:
        print("    [FAIL] ENCODER PARTIALLY OVER-INVARIANT: Token cosine > 0.90; token variance floor or recon weight escalation needed.")
    print("=" * 85)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running V1 & V2 Verification Suite on device: {device}")
    
    ckpt_path = find_v5_checkpoint()
    print(f"Loading Checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    agent = JEPAAgent(latent_dim=256, num_actions=17, device=device)
    comp = raw_ckpt.get("components", {})
    if "agent" in comp:
        agent.load_state_dict(comp["agent"])
    elif "encoder" in comp:
        agent.encoder.load_state_dict(comp["encoder"])
    print("[+] Agent restored successfully.")
    
    # Run V1
    run_v1_corrected_table_persistence(raw_ckpt)
    
    # Run V2
    run_v2_scene_cosine_probe(agent, device)


if __name__ == "__main__":
    main()
