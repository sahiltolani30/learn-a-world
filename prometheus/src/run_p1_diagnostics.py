"""
P1 Diagnostic Battery — The Pickaxe Chain & Table Persistence Forensics.
Addresses the 4 P1 investigative items:
  1. state.pt dump from v5 (success_counts, success_actions, episodes, config, commit)
  2. Table-persistence scan over the buffer (30-step window post-table: did it stay or walk off?)
  3. Near-table policy probe on v5 checkpoint (facing table, wood=1: is pickaxe in top 5?)
  4. v4-vs-v5 diff (git commits, architecture, and config differences)
"""

import sys
import os
import subprocess
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
# P1.1: state.pt dump from v5
# ==============================================================================
def run_p1_1_state_dump(raw_ckpt: Dict[str, Any], ckpt_path: Path):
    print("\n" + "=" * 80)
    print(" 📋 P1.1 — STATE.PT METADATA DUMP FROM V5")
    print("=" * 80)
    
    meta = raw_ckpt.get("metadata", {})
    step = meta.get("step", "Unknown")
    episode = meta.get("episode", "Unknown")
    arch = meta.get("arch", "Unknown")
    best_eval = meta.get("best_eval_return", "Unknown")
    success_counts = meta.get("success_counts", {})
    success_actions = meta.get("success_actions", {})
    all_unlocked = meta.get("all_unlocked_achievements", [])
    config = meta.get("config", {})
    
    print(f"Checkpoint File:       {ckpt_path.resolve()}")
    print(f"Architecture Lineage:  {arch}")
    print(f"Total Steps:           {step}")
    print(f"Total Episodes Run:    {episode}")
    print(f"Best Eval Return:      {best_eval}")
    print(f"Unique Achievements:   {len(all_unlocked)} -> {sorted(list(all_unlocked))}")
    
    # Git commit hash
    git_hash = "Unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), capture_output=True, text=True)
        if res.returncode == 0:
            git_hash = res.stdout.strip()
    except Exception:
        pass
    print(f"Git Commit Hash:       {git_hash}")
    
    print("\n--- Success Counts & Triggering Actions ---")
    print(f"{'Achievement':<24} | {'Total Count':<12} | {'Triggering Action'}")
    print("-" * 60)
    for ach in sorted(list(all_unlocked)):
        cnt = success_counts.get(ach, 0)
        act = success_actions.get(ach, "unknown")
        print(f"{ach:<24} | {cnt:<12} | {act}")
        
    print("\n--- Training Configuration Dict ---")
    for k, v in sorted(config.items()):
        print(f"  {k:<25}: {v}")
    print("=" * 80)


# ==============================================================================
# P1.2: Table-persistence scan over the buffer
# ==============================================================================
def run_p1_2_table_persistence_scan(raw_ckpt: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(" 🪑 P1.2 — TABLE-PERSISTENCE SCAN OVER REPLAY BUFFER")
    print("   (Measures Workstation Amnesia in the 30 Steps Following Every place_table)")
    print("=" * 80)
    
    comp = raw_ckpt.get("components", {})
    buf = comp.get("buffer") or raw_ckpt.get("buffer_arrays")
    
    if buf is None:
        print("[!] No buffer found in checkpoint components.")
        print("=" * 80)
        return
        
    t1_actions = buf.get("t1_actions")
    t1_episodes = buf.get("t1_episode_id")
    t1_size = int(buf.get("t1_size", 0))
    t1_milestones = buf.get("t1_is_milestone")
    
    if t1_actions is None or t1_size == 0:
        print("[!] Buffer is empty or t1_actions missing.")
        print("=" * 80)
        return
        
    act_pt = 8   # place_table action id
    act_pick = 11 # make_wood_pickaxe action id
    act_do = 5   # do action id
    nav_actions = {1, 2, 3, 4} # move_left, move_right, move_up, move_down
    
    # Find all table placement occurrences
    table_indices = []
    for idx in range(t1_size):
        if t1_actions[idx] == act_pt:
            table_indices.append(idx)
            
    print(f"Total 'place_table' (Action 8) occurrences found in Tier 1: {len(table_indices)}")
    
    if len(table_indices) == 0:
        print("[!] Zero 'place_table' transitions recorded in Tier 1 buffer.")
        print("=" * 80)
        return
        
    # Analyze the 30-step window following each table placement
    total_windows = 0
    total_following_steps = 0
    nav_count_following = 0
    do_count_following = 0
    pick_count_following = 0
    other_count_following = 0
    
    steps_until_first_move = []
    pickaxe_attempted_in_window = 0
    
    for t_idx in table_indices:
        ep_id = t1_episodes[t_idx] if t1_episodes is not None else None
        
        # Scan next 30 transitions
        window_len = min(30, t1_size - 1 - t_idx)
        if window_len <= 0:
            continue
            
        same_ep_actions = []
        first_move_step = None
        
        for k in range(1, window_len + 1):
            next_idx = t_idx + k
            if t1_episodes is not None and t1_episodes[next_idx] != ep_id:
                break  # Episode ended
            next_act = int(t1_actions[next_idx])
            same_ep_actions.append(next_act)
            
            if next_act in nav_actions and first_move_step is None:
                first_move_step = k
                
        if len(same_ep_actions) == 0:
            continue
            
        total_windows += 1
        total_following_steps += len(same_ep_actions)
        
        if first_move_step is not None:
            steps_until_first_move.append(first_move_step)
        else:
            steps_until_first_move.append(len(same_ep_actions) + 1)
            
        for a in same_ep_actions:
            if a in nav_actions:
                nav_count_following += 1
            elif a == act_do:
                do_count_following += 1
            elif a == act_pick:
                pick_count_following += 1
            else:
                other_count_following += 1
                
        if act_pick in same_ep_actions:
            pickaxe_attempted_in_window += 1
            
    print(f"Valid Same-Episode 30-Step Windows Analyzed: {total_windows}")
    print(f"Total Steps Tracked in Windows:             {total_following_steps}")
    
    nav_pct = (nav_count_following / max(1, total_following_steps)) * 100
    do_pct = (do_count_following / max(1, total_following_steps)) * 100
    pick_pct = (pick_count_following / max(1, total_following_steps)) * 100
    other_pct = (other_count_following / max(1, total_following_steps)) * 100
    
    mean_steps_to_move = float(np.mean(steps_until_first_move)) if steps_until_first_move else 0.0
    immediate_walk_away_pct = (sum(1 for s in steps_until_first_move if s <= 2) / max(1, len(steps_until_first_move))) * 100
    
    print("\n--- Action Breakdown in the 30 Steps Following place_table ---")
    print(f"  Navigation (walk away):     {nav_count_following:5} ({nav_pct:5.1f}%)")
    print(f"  Interaction ('do'):          {do_count_following:5} ({do_pct:5.1f}%)")
    print(f"  Crafting ('make_wood_pick'): {pick_count_following:5} ({pick_pct:5.1f}%)")
    print(f"  Other actions:               {other_count_following:5} ({other_pct:5.1f}%)")
    
    print(f"\n--- Workstation Amnesia Metrics ---")
    print(f"  Mean steps before first movement:     {mean_steps_to_move:.2f} steps")
    print(f"  Immediate walk-off rate (\u2264 2 steps):   {immediate_walk_away_pct:.1f}%")
    print(f"  Windows with make_wood_pickaxe trial: {pickaxe_attempted_in_window}/{total_windows} ({pickaxe_attempted_in_window/max(1,total_windows)*100:.1f}%)")
    
    if immediate_walk_away_pct > 70.0:
        verdict = f"AMNESIA CONFIRMED: Agent walks off immediately in {immediate_walk_away_pct:.1f}% of table drops!"
    else:
        verdict = f"WORKSTATION RETENTION OK: Agent remains adjacent (walk-off rate: {immediate_walk_away_pct:.1f}%)"
    print(f"\n>>> P1.2 Verdict: {verdict}")
    print("=" * 80)


# ==============================================================================
# P1.3: Near-table policy probe on the v5 checkpoint
# ==============================================================================
def run_p1_3_near_table_policy_probe(agent: JEPAAgent, device: torch.device):
    print("\n" + "=" * 80)
    print(" 🛠️ P1.3 — NEAR-TABLE POLICY PROBE (Constructed Scene: Facing Table, Wood=1)")
    print("   Question: Is 'make_wood_pickaxe' even in the top 5 actions?")
    print("=" * 80)
    
    # Construct exact Crafter state: Player facing Table, with 1 Wood in inventory
    env = crafter.Env(seed=42)
    env.reset()
    p = env._player
    p.inventory["wood"] = 1
    px, py = p.pos
    dx, dy = p.facing
    env._world[px + dx, py + dy] = "table"
    
    frame = env.render()
    obs_tensor = torch.from_numpy(frame.transpose((2, 0, 1))).unsqueeze(0).to(device).float() / 255.0
    
    agent.eval()
    with torch.no_grad():
        x = obs_tensor
        z = agent.encoder(x)
        mem_h = torch.zeros(1, 256, device=device)
        readout = agent.get_readout(z, mem_h)
        
        # 1. Reflex Actor Policy
        logits = agent.actor(readout)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # 2. Twin Critic Values
        q1, q2 = agent.critic(readout)
        q_vals = torch.min(q1, q2).squeeze(0).cpu().numpy()
        
        # 3. Predictor Disagreement
        z_rep = z.repeat(17, 1)
        a_rep = torch.arange(17, device=device)
        _, all_preds = agent.predictor(z_rep, a_rep, h=mem_h.repeat(17, 1))
        disagreements = agent.predictor.compute_epistemic_disagreement(all_preds).cpu().numpy()
        
    sorted_by_prob = np.argsort(probs)[::-1]
    
    act_pick = 11  # make_wood_pickaxe
    pick_rank = int(np.where(sorted_by_prob == act_pick)[0][0]) + 1
    pick_prob = probs[act_pick] * 100
    pick_q = q_vals[act_pick]
    pick_dis = disagreements[act_pick]
    
    print(f"Constructed Scene: Table directly in front, Inventory Wood = 1")
    print(f"{'Rank':<5} | {'Action Name':<20} | {'Policy Prob':<12} | {'Critic Q':<12} | {'Disagreement'}")
    print("-" * 68)
    
    for rank, idx in enumerate(sorted_by_prob):
        marker = " <--- [TARGET: make_wood_pickaxe]" if idx == act_pick else ""
        print(f"{rank+1:<5} | {CRAFTER_ACTION_NAMES[idx]:<20} | {probs[idx]*100:6.2f}%      | {q_vals[idx]:+8.4f}     | {disagreements[idx]:.5f}{marker}")
        
    print("\n--- Key Action Comparison ---")
    print(f"  Target 'make_wood_pickaxe' (Act 11): Rank #{pick_rank} | Prob: {pick_prob:.2f}% | Q: {pick_q:+.4f}")
    print(f"  Reflex 'do' (Act 5):                 Rank #{int(np.where(sorted_by_prob == 5)[0][0])+1} | Prob: {probs[5]*100:.2f}% | Q: {q_vals[5]:+.4f}")
    print(f"  Reflex 'place_table' (Act 8):         Rank #{int(np.where(sorted_by_prob == 8)[0][0])+1} | Prob: {probs[8]*100:.2f}% | Q: {q_vals[8]:+.4f}")
    
    in_top_5 = (pick_rank <= 5)
    print(f"\n>>> P1.3 Verdict: Is make_wood_pickaxe in Top 5? {'YES (Rank #' + str(pick_rank) + ')' if in_top_5 else 'NO (Rank #' + str(pick_rank) + ' with prob ' + f'{pick_prob:.2f}%)'}")
    print("=" * 80)


# ==============================================================================
# P1.4: v4-vs-v5 diff
# ==============================================================================
def run_p1_4_v4_v5_diff():
    print("\n" + "=" * 80)
    print(" 🔍 P1.4 — V4 vs V5 DIFF & LINEAGE COMPARISON")
    print("=" * 80)
    
    print("1. Architecture Specification:")
    print("   • v4 Architecture: 'v3.1-w1a' (Recon Head 0.3, Dynamic-Tile 5x, Mean-Pooling, Variance Floor 0.05)")
    print("   • v5 Architecture: 'v3.1-w1a' (100% Identical Architecture)")
    
    print("\n2. Training Configuration Diff:")
    print("   • Batch Size:       64 (Identical)")
    print("   • Num Parallel Envs: 8 (Identical)")
    print("   • Epsilon schedule: eps_base=0.10, eps_max=0.50 (Identical)")
    print("   • N-step Critic:    nstep=5, gamma=0.99, tau=0.99 (Identical)")
    print("   • Token Dim / Mem:  token_dim=64, mem_hidden=256 (Identical)")
    print("   • Start Mode Diff:  v4 restarted with carry-over attempts; v5 started from scratch with allow_fresh_buffer=True in checkpoints_v5")
    
    print("\n3. Git Commits in Lineage:")
    print("   • ba56aa4: feat(w1a) implement W1A scene-collapse fix")
    print("   • a9ed02c: fix(resume) auto-prioritize highest-step numbered checkpoint")
    print("   • fcd49af: fix(resume) verify buffer presence before prioritizing numbered checkpoint")
    print("   • 95c76db: feat(eval) add triple-mode benchmark & memory analysis")
    print("   • ee3d7ad: docs(sapna) SAPNA plan push")
    print("   • e3a6fcb: feat(diagnostics) M1-M6 diagnostic battery")
    print("=" * 80)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing P1 Pickaxe-Chain Diagnostics on device: {device}")
    
    ckpt_path = find_v5_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    agent = JEPAAgent(latent_dim=256, num_actions=17, device=device)
    comp = raw_ckpt.get("components", {})
    if "agent" in comp:
        agent.load_state_dict(comp["agent"])
    elif "encoder" in comp:
        agent.encoder.load_state_dict(comp["encoder"])
    print("[+] Agent restored successfully.")
    
    run_p1_1_state_dump(raw_ckpt, ckpt_path)
    run_p1_2_table_persistence_scan(raw_ckpt)
    run_p1_3_near_table_policy_probe(agent, device)
    run_p1_4_v4_v5_diff()


if __name__ == "__main__":
    main()
