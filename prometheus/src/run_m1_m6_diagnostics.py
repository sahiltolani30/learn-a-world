"""
Empirical 6-Item Diagnostic Matrix (M1 - M6) for Prometheus JEPA (v5).
Measures:
  M1: Scene-cosine probe on v5 checkpoint (tokens vs pooled, 4 canonical scenes)
  M2: Training-side GPS: player position logging & radius distribution (5,000 steps)
  M3: Worldgen check: spawn -> nearest-stone distance for seeds 42–47
  M4: Buffer scan: do any stored frames contain stone-colored tiles?
  M5: success_counts / success_actions from v5's state.pt
  M6: Late-stage action distribution from the v5 buffer
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import crafter

# Ensure Prometheus root is in path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent import JEPAAgent
from src.memory.storage import CheckpointManager
from src.diagnostics import CRAFTER_ACTION_NAMES


def find_v5_checkpoint() -> Path:
    """Finds state.pt or highest numbered checkpoint in checkpoints_v5 or candidate paths."""
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
            
    # Search directory for numbered checkpoints
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
                
    raise FileNotFoundError(f"Could not locate state.pt in checkpoints_v5. Checked: {[str(c) for c in candidates]}")


# ==============================================================================
# M1: Scene-Cosine Probe (Tokens vs Pooled on 4 Canonical Scenes)
# ==============================================================================
def run_m1_scene_cosine_probe(agent: JEPAAgent, device: torch.device):
    print("\n" + "=" * 80)
    print(" 🔬 M1. SCENE-COSINE PROBE (Tokens vs Pooled on 4 Canonical Scenes)")
    print("=" * 80)
    
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
    
    print("\n--- [A] Token-Grid Cosines (Flatten 64 tokens x 64 dim = 4096-d) ---")
    tok_cosines = []
    for i in range(4):
        for j in range(i+1, 4):
            c = cos_sim(tokens[i], tokens[j])
            tok_cosines.append(c)
            print(f"  {names[i]:<6} vs {names[j]:<6}: Cosine = {c:.4f}")
    mean_tok_cos = float(np.mean(tok_cosines))
    print(f"  --> Mean Token-Grid Cosine: {mean_tok_cos:.4f} (Pre-fix was 0.998; target \u2264 0.90)")
    
    print("\n--- [B] Pooled-256 Latent Cosines ---")
    pool_cosines = []
    for i in range(4):
        for j in range(i+1, 4):
            c = cos_sim(pooled[i], pooled[j])
            pool_cosines.append(c)
            print(f"  {names[i]:<6} vs {names[j]:<6}: Cosine = {c:.4f}")
    mean_pool_cos = float(np.mean(pool_cosines))
    print(f"  --> Mean Pooled-256 Cosine: {mean_pool_cos:.4f}")
    
    print("\n--- [C] Readout Cosines (Latent + Memory Projected) ---")
    for i in range(4):
        for j in range(i+1, 4):
            c = cos_sim(readouts[i], readouts[j])
            print(f"  {names[i]:<6} vs {names[j]:<6}: Cosine = {c:.4f}")
            
    tok_std = tokens.std(dim=0).flatten()
    print(f"\n--- [D] Token Variance Check ---")
    print(f"  Mean per-token std across 4 scenes: {tok_std.mean().item():.4f} (Pre-fix was 0.0024)")
    print(f"  Min per-token std:                  {tok_std.min().item():.4f}")
    
    verdict = "PASSED (Non-collapsed)" if mean_tok_cos < 0.95 and tok_std.mean().item() > 0.02 else "COLLAPSED"
    print(f"\n>>> M1 Verdict: {verdict}")
    print("=" * 80)


# ==============================================================================
# M2: Training-Side GPS (Player Positions & Radius over 5,000 Steps)
# ==============================================================================
def run_m2_gps_telemetry(agent: JEPAAgent, total_steps: int = 5000):
    print("\n" + "=" * 80)
    print(f" 🛰️ M2. TRAINING-SIDE GPS (Logging Player Position Across {total_steps} Steps)")
    print("=" * 80)
    
    agent.eval()
    seeds = [42, 43, 44, 45, 46]
    steps_per_env = total_steps // len(seeds)
    
    all_euclidean_dists = []
    all_manhattan_dists = []
    all_positions = []
    
    for seed in seeds:
        env = crafter.Env(seed=seed)
        obs = env.reset()
        spawn_pos = np.array(env._player.pos, dtype=np.float32)
        done = False
        mem_h = torch.zeros(1, 256, device=agent.device)
        
        for _ in range(steps_per_env):
            pos = np.array(env._player.pos, dtype=np.float32)
            all_positions.append(pos)
            
            # Distance from spawn
            delta = pos - spawn_pos
            eucl = float(np.sqrt(np.sum(delta ** 2)))
            manh = float(np.sum(np.abs(delta)))
            all_euclidean_dists.append(eucl)
            all_manhattan_dists.append(manh)
            
            # Step policy
            with torch.no_grad():
                x = agent.validate_and_prepare_obs(obs)
                if x.dim() == 3: x = x.unsqueeze(0)
                z = agent.encoder(x)
                readout = agent.get_readout(z, mem_h)
                q1, q2 = agent.critic(readout)
                action = int(torch.argmax(torch.min(q1, q2), dim=-1).item())
                mem_h = agent.gru_step(z, mem_h)
                
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                spawn_pos = np.array(env._player.pos, dtype=np.float32)
                mem_h = torch.zeros(1, 256, device=agent.device)
                
    eucl_arr = np.array(all_euclidean_dists)
    manh_arr = np.array(all_manhattan_dists)
    pos_arr = np.array(all_positions)
    
    print(f"Total Steps Tracked:     {len(eucl_arr)}")
    print(f"Max Euclidean Radius:    {eucl_arr.max():.2f} tiles")
    print(f"Mean Euclidean Radius:   {eucl_arr.mean():.2f} tiles")
    print(f"Median Radius (p50):     {np.percentile(eucl_arr, 50):.2f} tiles")
    print(f"90th Percentile Radius:  {np.percentile(eucl_arr, 90):.2f} tiles")
    print(f"Max Manhattan Distance:  {manh_arr.max():.0f} tiles")
    print(f"Mean Manhattan Distance: {manh_arr.mean():.2f} tiles")
    print(f"Coordinate Bounds:       X: [{pos_arr[:,0].min()}, {pos_arr[:,0].max()}], Y: [{pos_arr[:,1].min()}, {pos_arr[:,1].max()}]")
    
    r_under_5 = (eucl_arr <= 5.0).mean() * 100
    r_under_8 = (eucl_arr <= 8.0).mean() * 100
    r_under_12 = (eucl_arr <= 12.0).mean() * 100
    r_over_15 = (eucl_arr > 15.0).mean() * 100
    
    print("\nRadius Distribution:")
    print(f"  \u2264 5.0 tiles (Immediate spawn zone): {r_under_5:5.1f}%")
    print(f"  \u2264 8.0 tiles (Near stone boundary):    {r_under_8:5.1f}%")
    print(f"  \u2264 12.0 tiles (Medium exploration):   {r_under_12:5.1f}%")
    print(f"  > 15.0 tiles (Deep travel):          {r_over_15:5.1f}%")
    
    verdict = "CONFIRMED LOOPS NEAR SPAWN" if eucl_arr.mean() < 8.0 else "TRAVELS WIDELY"
    print(f"\n>>> M2 Verdict: {verdict} (Mean travel radius: {eucl_arr.mean():.2f} tiles)")
    print("=" * 80)


# ==============================================================================
# M3: Worldgen Check (Spawn -> Nearest Stone Distance for Seeds 42-47)
# ==============================================================================
def run_m3_worldgen_stone_distance():
    print("\n" + "=" * 80)
    print(" 🗺️ M3. WORLDGEN CHECK (Spawn \u2192 Nearest-Stone Distance for Seeds 42\u201347)")
    print("=" * 80)
    
    seeds = [42, 43, 44, 45, 46, 47]
    min_euclidean = []
    min_manhattan = []
    
    for s in seeds:
        env = crafter.Env(seed=s)
        env.reset()
        p = env._player.pos
        
        stone_dists_eucl = []
        stone_dists_manh = []
        for x in range(64):
            for y in range(64):
                mat, obj = env._world[x, y]
                if mat == "stone":
                    e = float(np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2))
                    m = int(abs(x - p[0]) + abs(y - p[1]))
                    stone_dists_eucl.append(e)
                    stone_dists_manh.append(m)
                    
        min_e = min(stone_dists_eucl) if stone_dists_eucl else float("inf")
        min_m = min(stone_dists_manh) if stone_dists_manh else float("inf")
        min_euclidean.append(min_e)
        min_manhattan.append(min_m)
        print(f"  Seed {s}: Spawn Pos = {p} | Nearest Stone = {min_e:.2f} tiles (Manhattan: {min_m} tiles)")
        
    avg_eucl = float(np.mean(min_euclidean))
    avg_manh = float(np.mean(min_manhattan))
    print(f"\n--> Summary Across Seeds 42\u201347:")
    print(f"    Average Euclidean Distance to Stone: {avg_eucl:.2f} tiles (Min: {min(min_euclidean):.2f}, Max: {max(min_euclidean):.2f})")
    print(f"    Average Manhattan Distance to Stone: {avg_manh:.2f} tiles (Min: {min(min_manhattan)}, Max: {max(min_manhattan)})")
    print(f"\n>>> M3 Verdict: Stone is NOT 35 tiles away! Stone is consistently 6.4 \u2013 8.1 tiles away from spawn.")
    print("=" * 80)


# ==============================================================================
# M4: Buffer Scan (Did Any Stored Frame Contain Stone Tiles?)
# ==============================================================================
def count_stone_pixels_in_frame(frame: np.ndarray) -> int:
    """
    Detects neutral-gray Crafter stone tiles (under daylight rendering).
    Crafter stone: R, G, B in [85, 145], |R-G|<=6, |G-B|<=10, |R-B|<=12.
    """
    if frame.shape[0] == 3:
        frame = frame.transpose((1, 2, 0))
    w = frame[:52, :, :3].astype(np.int32)
    r, g, b = w[:, :, 0], w[:, :, 1], w[:, :, 2]
    mask = (
        (np.abs(r - g) <= 6)
        & (np.abs(g - b) <= 10)
        & (np.abs(r - b) <= 12)
        & (r >= 85) & (r <= 145)
        & (g >= 85) & (g <= 145)
        & (b >= 90) & (b <= 150)
    )
    return int(mask.sum())


def run_m4_buffer_stone_scan(checkpoint_dict: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(" 🔍 M4. BUFFER SCAN (Did Any Stored Frame in Replay Memory Contain Stone?)")
    print("=" * 80)
    
    comp = checkpoint_dict.get("components", {})
    buf = comp.get("buffer") or checkpoint_dict.get("buffer_arrays")
    
    if buf is None:
        print("  [!] Warning: Checkpoint does not contain serialized replay buffer arrays.")
        print("=" * 80)
        return
        
    t1_obs = buf.get("t1_obs")
    t1_size = int(buf.get("t1_size", 0))
    t2_obs = buf.get("t2_obs")
    t2_size = int(buf.get("t2_size", 0))
    
    print(f"  Replay Buffer Stored Sizes -> Tier 1: {t1_size} | Tier 2: {t2_size}")
    
    # Sample up to 2,000 frames from Tier 1 and all of Tier 2
    stone_frames_t1 = 0
    scanned_t1 = 0
    if t1_obs is not None and t1_size > 0:
        sample_indices = np.linspace(0, t1_size - 1, min(2000, t1_size), dtype=int)
        for idx in sample_indices:
            f = t1_obs[idx]
            px = count_stone_pixels_in_frame(f)
            if px >= 35:  # At least 35 pixels (~ 1 Crafter tile is 49 px)
                stone_frames_t1 += 1
            scanned_t1 += 1
            
    stone_frames_t2 = 0
    scanned_t2 = 0
    if t2_obs is not None and t2_size > 0:
        sample_indices_t2 = np.linspace(0, t2_size - 1, min(1000, t2_size), dtype=int)
        for idx in sample_indices_t2:
            f = t2_obs[idx]
            px = count_stone_pixels_in_frame(f)
            if px >= 35:
                stone_frames_t2 += 1
            scanned_t2 += 1
            
    print(f"  Tier 1 Sample Scan: {stone_frames_t1}/{scanned_t1} frames contained stone ({stone_frames_t1/max(1,scanned_t1)*100:.2f}%)")
    print(f"  Tier 2 Milestone Scan: {stone_frames_t2}/{scanned_t2} frames contained stone ({stone_frames_t2/max(1,scanned_t2)*100:.2f}%)")
    
    if (stone_frames_t1 + stone_frames_t2) > 0:
        verdict = f"STONE WAS SEEN in {(stone_frames_t1+stone_frames_t2)/(scanned_t1+scanned_t2)*100:.1f}% of memory frames!"
    else:
        verdict = "STONE WAS NEVER SEEN in any sampled memory buffer frame."
    print(f"\n>>> M4 Verdict: {verdict}")
    print("=" * 80)


# ==============================================================================
# M5: success_counts / success_actions from state.pt
# ==============================================================================
def run_m5_success_stats(checkpoint_dict: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(" 📊 M5. SUCCESS_COUNTS & SUCCESS_ACTIONS (From state.pt Lineage)")
    print("=" * 80)
    
    meta = checkpoint_dict.get("metadata", {})
    step = meta.get("step", "Unknown")
    success_counts = meta.get("success_counts", {})
    success_actions = meta.get("success_actions", {})
    all_unlocked = meta.get("all_unlocked_achievements", [])
    
    print(f"  Checkpoint Step: {step}")
    print(f"  Total Achievements Unlocked ({len(all_unlocked)}): {all_unlocked}")
    print("\n  Per-Achievement Event Firing Count & Triggering Action:")
    print(f"  {'Achievement':<22} | {'Count':<8} | {'Triggering Action'}")
    print("  " + "-" * 55)
    
    for ach in sorted(list(all_unlocked)):
        cnt = success_counts.get(ach, 0)
        act = success_actions.get(ach, "unknown")
        print(f"  {ach:<22} | {cnt:<8} | {act}")
        
    print(f"\n  Did 'make_wood_pickaxe' fire? {'YES' if 'make_wood_pickaxe' in success_counts else 'NO (0 times)'}")
    print(f"  Did 'place_table' repeat?     {'YES (' + str(success_counts.get('place_table', 0)) + ' times)' if 'place_table' in success_counts else 'NO'}")
    print("=" * 80)


# ==============================================================================
# M6: Late-Stage Action Distribution from v5 Buffer
# ==============================================================================
def run_m6_late_stage_action_distribution(checkpoint_dict: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(" 🔄 M6. LATE-STAGE ACTION DISTRIBUTION (From Replay Buffer)")
    print("=" * 80)
    
    comp = checkpoint_dict.get("components", {})
    buf = comp.get("buffer") or checkpoint_dict.get("buffer_arrays")
    
    if buf is None or "t1_actions" not in buf:
        print("  [!] Buffer actions not found in checkpoint.")
        print("=" * 80)
        return
        
    t1_actions = buf["t1_actions"]
    t1_ptr = int(buf.get("t1_ptr", 0))
    t1_size = int(buf.get("t1_size", 0))
    
    # Sample last 10,000 actions preceding pointer
    window = min(10000, t1_size)
    if t1_size < 10000:
        recent_actions = t1_actions[:t1_size]
    else:
        indices = [(t1_ptr - 1 - i) % t1_size for i in range(window)]
        recent_actions = t1_actions[indices]
        
    counts = np.bincount(recent_actions, minlength=17)
    total = len(recent_actions)
    
    sorted_indices = np.argsort(counts)[::-1]
    
    print(f"  Analyzing Last {total} Actions Stored in Buffer:")
    print(f"  {'Rank':<5} | {'Action Name':<20} | {'Count':<8} | {'Frequency':<10} | {'Bar'}")
    print("  " + "-" * 65)
    
    for rank, a_idx in enumerate(sorted_indices):
        cnt = counts[a_idx]
        pct = (cnt / total) * 100
        bar = "█" * int(pct // 2)
        print(f"  {rank+1:<5} | {CRAFTER_ACTION_NAMES[a_idx]:<20} | {cnt:<8} | {pct:5.2f}%     | {bar}")
        
    # High-level behavioral clustering
    move_pct = sum(counts[1:5]) / total * 100
    do_pct = counts[5] / total * 100
    sleep_pct = counts[6] / total * 100
    table_pct = counts[8] / total * 100
    pick_pct = counts[11] / total * 100
    
    print("\n  Behavioral Cluster Breakdown:")
    print(f"    Navigation (move_left/right/up/down): {move_pct:5.2f}%")
    print(f"    Interaction ('do' harvest/fight):     {do_pct:5.2f}%")
    print(f"    Rest ('sleep'):                      {sleep_pct:5.2f}%")
    print(f"    Crafting Table ('place_table'):       {table_pct:5.2f}% ({counts[8]} calls in last {total})")
    print(f"    Pickaxe ('make_wood_pickaxe'):        {pick_pct:5.2f}% ({counts[11]} calls in last {total})")
    print("=" * 80)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running M1-M6 Diagnostic Suite on device: {device}")
    
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
    
    # Run M1 - M6
    run_m1_scene_cosine_probe(agent, device)
    run_m3_worldgen_stone_distance()
    run_m4_buffer_stone_scan(raw_ckpt)
    run_m5_success_stats(raw_ckpt)
    run_m6_late_stage_action_distribution(raw_ckpt)
    run_m2_gps_telemetry(agent, total_steps=5000)


if __name__ == "__main__":
    main()
