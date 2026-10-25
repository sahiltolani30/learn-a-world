#!/usr/bin/env python3
"""
scripts/screen_grandmaster_nohier.py
====================================
50-Episode Screening of the Grand Master Champion Configuration with
Hierarchical Tech-Tree (Lever E) turned OFF (hierarchical_tte=False).

Isolates whether the Grand Master's performance comes from domain-agnostic
mechanisms (value denormalization, TTE compass, recurring vitals, momentum,
deep compute) or from the hand-coded tech-tree progression.

Exact Parameters (Grand_Master_O1_H15 with hierarchical_tte=False):
- denormalize_q: True
- tte_weight: 0.5
- recurring_vitals: True
- vital_reward_weight: 0.3
- action_persistence: 0.3
- hierarchical_tte: False (THE ONLY CHANGE)
- uncertainty_weight: 0.1
- novelty_weight: 0.005
- prior_mix: 0.65
- horizon: 15
- num_candidates: 128
- cem_iterations: 3

Screening Protocol:
- 50 episodes on untouched evaluation seeds: 2050 to 2099
- Max steps: 1000 per episode
- Checkpoint: checkpoints/jepa_crafter_v3_step1_stone.pt
- Output: results/screen_v3_step1_nohier_results.json (+ Drive mirror)
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # registers Crafter-v0
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner


def resolve_checkpoint(ckpt_path: str, drive_dir: Optional[str] = None) -> Path:
    p = Path(ckpt_path)
    if p.exists():
        return p
    candidates = [
        REPO_ROOT / ckpt_path,
        REPO_ROOT / "checkpoints" / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / ckpt_path,
            dp / "checkpoints" / p.name,
            Path("/content/drive/MyDrive/learn-a-world") / ckpt_path,
            Path("/content/drive/MyDrive/learn-a-world/checkpoints") / p.name,
            Path("/content") / "checkpoints" / p.name,
            Path("/content") / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint: {ckpt_path}")


def run_screening_nohier(
    checkpoint_path: str = "/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_step1_stone.pt",
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    output_json: str = "results/screen_v3_step1_nohier_results.json",
    drive_dir: Optional[str] = "/content/drive/MyDrive/learn-a-world",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    dev = torch.device(device)
    ckpt_file = resolve_checkpoint(checkpoint_path, drive_dir)

    print("=" * 80)
    print("      GRAND MASTER NO-HIERARCHICAL SCREENING (50 EPISODES: 2050-2099)")
    print(f"      Checkpoint       : {ckpt_file}")
    print(f"      Device           : {dev}")
    print(f"      Episodes         : {num_episodes} (Seeds {seed_start}..{seed_start + num_episodes - 1})")
    print("      Parameters:")
    print("        - horizon              : 15")
    print("        - num_candidates       : 128")
    print("        - cem_iterations       : 3")
    print("        - denormalize_q        : True")
    print("        - tte_weight           : 0.5")
    print("        - recurring_vitals     : True")
    print("        - vital_reward_weight  : 0.3")
    print("        - action_persistence   : 0.3")
    print("        - hierarchical_tte     : False  <-- (ISOLATION TEST)")
    print("        - uncertainty_weight   : 0.1")
    print("        - novelty_weight       : 0.005")
    print("        - prior_mix            : 0.65")
    print("=" * 80, flush=True)

    # 1. Load Model
    model = JEPAv3.from_checkpoint(str(ckpt_file), device=device)
    model.eval()

    # 2. Build Planner with EXACT Grand Master parameters, hierarchical_tte=False
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=15,
        num_candidates=128,
        cem_iterations=3,
        elite_ratio=0.15,
        gamma=0.99,
        uncertainty_weight=0.1,
        novelty_weight=0.005,
        prior_mix=0.65,
        prior_temperature=1.0,
        action_persistence=0.3,
        vital_reward_weight=0.3,
        num_actions=17,
        use_calibration=True,
        denormalize_q=True,
        tte_weight=0.5,
        recurring_vitals=True,
        hierarchical_tte=False,  # <-- The ONLY variable changed
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256)

    # 3. Environment Loop
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    episode_records: List[Dict[str, Any]] = []
    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    t_start = time.time()

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()

        ep_reward = 0.0
        steps = 0
        t_ep_start = time.time()

        for step in range(max_steps):
            steps += 1
            action, _ = planner.plan(obs, belief)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(dev)
                z_t = model.encode(obs_t, use_target=True)
                act_t = torch.tensor([action], device=dev, dtype=torch.long)
                h_prev = belief.get_h_tensor(device=dev).unsqueeze(0)
                s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
                mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
                event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)

            belief.update(event_probs, reward, next_h=next_h.squeeze(0))
            obs = next_obs

            if done:
                break

        ep_duration = time.time() - t_ep_start
        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        unlocked = [k for k, v in ach.items() if v > 0]

        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

        episode_records.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": steps,
            "reward": round(ep_reward, 2),
            "unlocked": unlocked,
            "duration_sec": round(ep_duration, 2),
        })

        print(
            f"Ep {ep_idx + 1:02d}/{num_episodes} (Seed {seed}) | "
            f"Steps: {steps:<4d} | Reward: {ep_reward:<5.1f} | "
            f"Achievements ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'None'} "
            f"({ep_duration:.1f}s)",
            flush=True,
        )

    env.close()
    total_time = time.time() - t_start
    score_stats = compute_crafter_score(achievement_records)

    summary = {
        "checkpoint": str(ckpt_file),
        "config_name": "Grand_Master_O1_H15_NoHier",
        "parameters": {
            "horizon": 15,
            "num_candidates": 128,
            "cem_iterations": 3,
            "denormalize_q": True,
            "tte_weight": 0.5,
            "recurring_vitals": True,
            "vital_reward_weight": 0.3,
            "action_persistence": 0.3,
            "hierarchical_tte": False,
            "uncertainty_weight": 0.1,
            "novelty_weight": 0.005,
            "prior_mix": 0.65,
        },
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "total_runtime_seconds": round(total_time, 2),
        "mean_survival_length": float(np.mean(survival_lengths)),
        "std_survival_length": float(np.std(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "crafter_score": score_stats["score"],
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "unlocked_achievement_count": score_stats["unlocked_count"],
        "unlock_percentages": score_stats["unlock_percentages"],
        "episodes": episode_records,
    }

    # Print Official Report Block
    pickaxe_rate = float(score_stats["unlock_percentages"].get("make_wood_pickaxe", 0.0))
    stone_rate = float(score_stats["unlock_percentages"].get("collect_stone", 0.0))
    drink_rate = float(score_stats["unlock_percentages"].get("collect_drink", 0.0))

    print("\n" + "=" * 80)
    print("                    SCREENING SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Time                     : {total_time:.1f}s | Mean Ep Time: {total_time / num_episodes:.1f}s")
    print(f"Official Crafter Score         : {summary['crafter_score']:.2f}%")
    print(f"Mean Survival Length           : {summary['mean_survival_length']:.1f} +/- {summary['std_survival_length']:.1f} steps")
    print(f"Mean Episode Reward            : {summary['mean_episode_reward']:.2f} +/- {summary['std_episode_reward']:.2f}")
    print(f"Make Wood Pickaxe Rate         : {pickaxe_rate:.1f}%")
    print(f"Collect Stone Rate             : {stone_rate:.1f}%")
    print(f"Collect Drink Rate             : {drink_rate:.1f}%")
    print(f"Mean Achievements per Episode  : {summary['mean_achievements_per_ep']:.2f}")
    print(f"Unique Achievements Unlocked   : {summary['unlocked_achievement_count']} / 22")
    print("=" * 80)

    print("\nPer-Achievement Unlock Rates:")
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = score_stats["unlock_percentages"].get(ach, 0.0)
        bar = "#" * int(round(pct / 5))
        print(f"  {ach:<22}: {pct:5.1f}% | {bar}")

    # Save Results JSON
    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--> Full screening results saved to: {out_p}")

    # Drive Mirror
    if drive_dir:
        dp = Path(drive_dir)
        if dp.exists():
            drive_out = dp / "results" / out_p.name
            drive_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_p), str(drive_out))
            print(f"[Drive] Mirrored screening results to: {drive_out}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Grand Master No-Hierarchical Screening (50 Episodes).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_step1_stone.pt",
        help="Path to checkpoint",
    )
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes (default 50)")
    parser.add_argument("--seed_start", type=int, default=2050, help="Start seed (default 2050)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode (default 1000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/screen_v3_step1_nohier_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_screening_nohier(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        output_json=args.output_json,
        drive_dir=args.drive_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
