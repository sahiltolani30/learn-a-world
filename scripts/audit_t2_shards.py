#!/usr/bin/env python3
"""
scripts/audit_t2_shards.py — Comprehensive Audit of T2 Curiosity Shards
======================================================================

Analyzes all collected T2 curiosity shards (from local directory or Google Drive):
1. Reads all shard_curiosity_v3_t02_*.npz files.
2. Extracts transition count, episode count, and achievement events.
3. Computes per-achievement unlock counts and episode unlock percentages (%) across all 22 Crafter achievements.
4. Compares against Phase 3 T1 baseline to quantify improvements in rare achievements.
5. Prints a formatted Markdown summary table and saves results to results/diagnostics/t2_curiosity_audit.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score


def audit_shards(
    shards_dir: Path,
    t1_reference_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Scans shards_dir for shard_curiosity_v3_t02_*.npz files and computes full metrics.
    """
    if not shards_dir.exists():
        raise FileNotFoundError(f"Shards directory not found: {shards_dir}")

    shard_files = sorted(shards_dir.glob("shard_curiosity_v3_t02_*.npz"))
    if not shard_files:
        # Fallback: check for all curiosity shards
        shard_files = sorted(shards_dir.glob("*curiosity*.npz"))

    print(f"[Audit] Scanning shards directory: {shards_dir}")
    print(f"[Audit] Found {len(shard_files)} shard files.")

    if len(shard_files) == 0:
        return {"error": "no_shards_found", "num_shards": 0}

    total_transitions = 0
    total_episodes = 0
    achievement_unlock_counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
    per_episode_achievements: List[Dict[str, int]] = []

    for s_idx, sf in enumerate(shard_files):
        try:
            data = np.load(str(sf), allow_pickle=True)
            obs = data["obs"]
            events = data["events"]
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
                ep_start = ep_end + 1
                total_episodes += 1

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

    # Compute unlock percentages
    unlock_percentages = {}
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = (achievement_unlock_counts[ach] / max(1, total_episodes)) * 100.0
        unlock_percentages[ach] = round(pct, 2)

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

    print("\n" + "=" * 80)
    print("PHASE 3 TRANCHE T2 CURIOSITY SHARD AUDIT REPORT")
    print("=" * 80)
    print(f"Total Shards Analyzed:     {len(shard_files)}")
    print(f"Total Transitions:         {total_transitions:,}")
    print(f"Total Episodes:            {total_episodes:,}")
    print(f"Curiosity Rollout Score:   {crafter_score_data['score']:.2f}%")
    print(f"Mean Unlocks Per Episode:  {crafter_score_data['mean_achievements_per_ep']:.2f}")
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

    return {
        "num_shards": len(shard_files),
        "total_transitions": total_transitions,
        "total_episodes": total_episodes,
        "crafter_score": round(crafter_score_data["score"], 3),
        "mean_achievements_per_ep": round(crafter_score_data["mean_achievements_per_ep"], 2),
        "unlock_counts": achievement_unlock_counts,
        "unlock_percentages": unlock_percentages,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit T2 curiosity shards")
    parser.add_argument(
        "--drive-dir",
        type=str,
        default="/content/drive/MyDrive/learn-a-world",
        help="Google Drive root path",
    )
    args = parser.parse_args()

    candidates = [
        Path(args.drive_dir) / "data" / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]

    target_dir = None
    for c in candidates:
        if c.exists() and len(list(c.glob("*.npz"))) > 0:
            target_dir = c
            break

    if target_dir is None:
        target_dir = candidates[0]

    report = audit_shards(target_dir)

    # Save report
    out_dir = REPO_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "t2_curiosity_audit.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\n[Audit] Saved structured audit results to: {out_file}")


if __name__ == "__main__":
    main()
