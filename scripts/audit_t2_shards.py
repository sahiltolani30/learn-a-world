#!/usr/bin/env python3
"""
scripts/audit_t2_shards.py — Comprehensive Audit of T2 Curiosity Shards
======================================================================

Analyzes all collected T2 curiosity shards (from local directory or Google Drive):
1. Reads all shard_curiosity_v3_t02_*.npz files.
2. Computes complete official Crafter benchmark block:
   - mean_survival_length, std_survival_length
   - mean_episode_reward, std_episode_reward
   - crafter_score (geometric mean across 22 achievements)
   - mean_achievements_per_ep
   - unlocked_achievement_count (out of 22)
   - unlock_percentages (all 22 achievements)
   - raw_unlock_counts
3. Prints side-by-side comparative table against Phase 3 T1 baseline.
4. Outputs the complete standard JSON result block.
5. Mirrors results to Google Drive.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score


def audit_shards(
    shards_dir: Path,
    drive_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Scans shards_dir for shard_curiosity_v3_t02_*.npz files and computes full metrics.
    """
    if not shards_dir.exists():
        raise FileNotFoundError(f"Shards directory not found: {shards_dir}")

    shard_files = sorted(shards_dir.glob("shard_curiosity_v3_t02_*.npz"))
    if not shard_files:
        shard_files = sorted(shards_dir.glob("*curiosity*.npz"))

    print(f"[Audit] Scanning shards directory: {shards_dir}")
    print(f"[Audit] Found {len(shard_files)} shard files.")

    if len(shard_files) == 0:
        print("[Audit] Error: No shard files found to audit.")
        return {"error": "no_shards_found", "num_shards": 0}

    total_transitions = 0
    total_episodes = 0
    achievement_unlock_counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
    per_episode_achievements: List[Dict[str, int]] = []
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for s_idx, sf in enumerate(shard_files):
        try:
            data = np.load(str(sf), allow_pickle=True)
            obs = data["obs"]
            events = data["events"]
            rewards = data["rewards"]
            dones = data["dones"]

            n_trans = len(obs)
            total_transitions += n_trans

            # Group into episodes via dones
            ep_start = 0
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            for ep_end in ep_indices:
                ep_events = events[ep_start : ep_end + 1]  # [T, 22]
                ep_rews = rewards[ep_start : ep_end + 1]
                ep_len = len(ep_events)
                ep_start = ep_end + 1
                total_episodes += 1

                episode_lengths.append(int(ep_len))
                episode_rewards.append(float(np.sum(ep_rews)))

                # Episode-level unlock (did this episode achieve it at least once?)
                ep_unlock_mask = np.any(ep_events > 0, axis=0)  # [22]
                ep_record = {}
                for a_idx, ach in enumerate(CRAFTER_ACHIEVEMENTS):
                    if ep_unlock_mask[a_idx]:
                        achievement_unlock_counts[ach] += 1
                        ep_record[ach] = 1
                    else:
                        ep_record[ach] = 0
                per_episode_achievements.append(ep_record)

        except Exception as e:
            print(f"  Warning: could not process shard {sf.name}: {e}")

    # Compute statistics
    mean_survival = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    std_survival = float(np.std(episode_lengths)) if episode_lengths else 0.0
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0

    # Compute unlock percentages
    unlock_percentages = {}
    unlocked_categories = 0
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = (achievement_unlock_counts[ach] / max(1, total_episodes)) * 100.0
        unlock_percentages[ach] = round(pct, 2)
        if achievement_unlock_counts[ach] > 0:
            unlocked_categories += 1

    crafter_score_data = compute_crafter_score(per_episode_achievements)

    # Reference T1 comparison counts (from Phase 3 T1 160k dataset)
    t1_baseline = {
        "collect_coal": 0,
        "collect_diamond": 0,
        "collect_drink": 339,
        "collect_iron": 0,
        "collect_sapling": 303,
        "collect_stone": 0,
        "collect_wood": 366,
        "defeat_skeleton": 1,
        "defeat_zombie": 3,
        "eat_cow": 4,
        "eat_plant": 123,
        "make_iron_pickaxe": 0,
        "make_iron_sword": 0,
        "make_stone_pickaxe": 0,
        "make_stone_sword": 0,
        "make_wood_pickaxe": 0,
        "make_wood_sword": 0,
        "place_furnace": 0,
        "place_plant": 123,
        "place_stone": 0,
        "place_table": 1,
        "wake_up": 54,
    }

    # Print Table
    print("\n" + "=" * 80)
    print("PHASE 3 TRANCHE T2 CURIOSITY SHARD AUDIT REPORT")
    print("=" * 80)
    print(f"Total Shards Analyzed:         {len(shard_files)}")
    print(f"Total Transitions Collected:   {total_transitions:,}")
    print(f"Total Episodes Collected:      {total_episodes:,}")
    print(f"Mean Survival Length:          {mean_survival:.2f} ± {std_survival:.2f} steps")
    print(f"Mean Episode Reward:           {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Official Crafter Score:        {crafter_score_data['score']:.2f}%")
    print(f"Mean Achievements Per Ep:      {crafter_score_data['mean_achievements_per_ep']:.2f}")
    print(f"Unique Categories Unlocked:    {unlocked_categories} / 22")
    print("-" * 80)
    print(f"{'Achievement':<24} | {'T2 Count':<10} | {'T2 Unlock %':<12} | {'T1 Baseline':<12} | {'Status'}")
    print("-" * 80)

    for ach in CRAFTER_ACHIEVEMENTS:
        t2_cnt = achievement_unlock_counts[ach]
        t2_pct = unlock_percentages[ach]
        t1_cnt = t1_baseline.get(ach, 0)

        delta = t2_cnt - t1_cnt
        if t2_cnt > 0 and t1_cnt == 0:
            status = "🔥 NEW DISCOVERY!"
        elif delta > 0:
            status = f"▲ +{delta}"
        elif delta == 0 and t2_cnt > 0:
            status = "Steady"
        elif t2_cnt == 0:
            status = "Frontier (0)"
        else:
            status = f"▼ {delta}"

        print(f"{ach:<24} | {t2_cnt:<10} | {t2_pct:>10.2f}% | {t1_cnt:<12} | {status}")

    print("=" * 80)

    result_block = {
        "num_shards": len(shard_files),
        "total_transitions": total_transitions,
        "total_episodes": total_episodes,
        "mean_survival_length": round(mean_survival, 2),
        "std_survival_length": round(std_survival, 2),
        "mean_episode_reward": round(mean_reward, 2),
        "std_episode_reward": round(std_reward, 2),
        "crafter_score": round(crafter_score_data["score"], 3),
        "mean_achievements_per_ep": round(crafter_score_data["mean_achievements_per_ep"], 2),
        "unlocked_achievement_count": unlocked_categories,
        "unlock_percentages": unlock_percentages,
        "per_achievement_counts": achievement_unlock_counts,
    }

    # Print Full JSON Block (Exact format used in screening checks)
    print("\n--- COMPLETE AUDIT RESULT BLOCK (JSON) ---")
    print(json.dumps(result_block, indent=2))
    print("------------------------------------------\n")

    # Save to local results
    out_dir = REPO_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "t2_curiosity_audit.json"
    out_file.write_text(json.dumps(result_block, indent=2))
    print(f"[Audit] Saved local report to: {out_file}")

    # Mirror to Drive if available
    if drive_dir is not None and drive_dir.exists():
        try:
            drive_out_dir = drive_dir / "results" / "diagnostics"
            drive_out_dir.mkdir(parents=True, exist_ok=True)
            drive_out_file = drive_out_dir / "t2_curiosity_audit.json"
            drive_out_file.write_text(json.dumps(result_block, indent=2))
            print(f"[Audit] Mirrored to Drive: {drive_out_file}")
        except Exception as e:
            print(f"[Audit] Warning: could not mirror to Drive: {e}")

    return result_block


def main():
    parser = argparse.ArgumentParser(description="Audit T2 curiosity shards with complete screening-style output")
    parser.add_argument(
        "--drive-dir",
        type=str,
        default="/content/drive/MyDrive/learn-a-world",
        help="Google Drive root path",
    )
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    candidates = [
        drive_p / "data" / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]

    target_dir = None
    for c in candidates:
        if c.exists() and len(list(c.glob("*.npz"))) > 0:
            target_dir = c
            break

    if target_dir is None:
        target_dir = candidates[0]

    audit_shards(target_dir, drive_dir=drive_p if drive_p.exists() else None)


if __name__ == "__main__":
    main()
