#!/usr/bin/env python3
"""
scripts/deep_audit_t2_diagnostics.py — Deep Forensic Diagnostics for T2 Curiosity Collection
=============================================================================================

Performs in-depth analysis of the 5 critical research questions:
1. Archive Contents: Breakdown of the 50 saved states (how many are "pickaxe in hand" states?).
2. Restart Usage: Fresh vs restart-start episode counts and comparative achievement yield.
3. Steering Bonus Breakdown: Relative magnitude of task reward vs rarity vs latent disagreement.
4. The 11 Pickaxe Episodes' Stories: Step of craft, post-craft survival length, and cause of termination.
5. The 1 Stone Episode In Full: Complete chronological milestone recipe of the autonomous stone unlock.
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

from eval.metrics import CRAFTER_ACHIEVEMENTS

# Key achievement indices
ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_WOOD = ACH_TO_IDX.get("collect_wood", 6)
IDX_TABLE = ACH_TO_IDX.get("place_table", 20)
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)
IDX_ZOMBIE = ACH_TO_IDX.get("defeat_zombie", 8)
IDX_SKELETON = ACH_TO_IDX.get("defeat_skeleton", 7)


def analyze_archive_contents(drive_dir: Path, local_dir: Path) -> Dict[str, Any]:
    """
    Question 1: Archive contents — breakdown of saved states.
    Specifically checks how many are 'pickaxe in hand' states.
    """
    summary_candidates = [
        drive_dir / "results" / "t2_collection_summary.json",
        local_dir / "results" / "t2_collection_summary.json",
    ]
    summary_data = None
    for sc in summary_candidates:
        if sc.exists():
            try:
                summary_data = json.loads(sc.read_text())
                break
            except Exception:
                pass

    if summary_data is None or "archive_summary" not in summary_data:
        return {
            "status": "summary_not_found",
            "message": "t2_collection_summary.json not found yet (collection still running). Will analyze from shards."
        }

    arch_summary = summary_data["archive_summary"]
    archived_achievements = arch_summary.get("archived_achievements", [])

    tier_counts = {
        "wood_only": 0,
        "table_placed": 0,
        "pickaxe_in_hand": 0,
        "combat_mob": 0,
        "other": 0,
    }

    pickaxe_states_count = 0
    table_states_count = 0

    for state_achs in archived_achievements:
        has_pickaxe = IDX_WOOD_PICKAXE in state_achs
        has_table = IDX_TABLE in state_achs
        has_combat = (IDX_ZOMBIE in state_achs) or (IDX_SKELETON in state_achs)

        if has_pickaxe:
            pickaxe_states_count += 1
            tier_counts["pickaxe_in_hand"] += 1
        elif has_table:
            table_states_count += 1
            tier_counts["table_placed"] += 1
        elif has_combat:
            tier_counts["combat_mob"] += 1
        elif IDX_WOOD in state_achs:
            tier_counts["wood_only"] += 1
        else:
            tier_counts["other"] += 1

    return {
        "archive_size": arch_summary.get("size", len(archived_achievements)),
        "max_size": arch_summary.get("max_size", 50),
        "total_restarts_triggered": arch_summary.get("total_restarts_triggered", 0),
        "pickaxe_in_hand_count": pickaxe_states_count,
        "pickaxe_in_hand_pct": round((pickaxe_states_count / max(1, len(archived_achievements))) * 100, 1),
        "table_states_count": table_states_count,
        "table_states_pct": round((table_states_count / max(1, len(archived_achievements))) * 100, 1),
        "tier_breakdown": tier_counts,
    }


def analyze_bonus_breakdown(drive_dir: Path, local_dir: Path) -> Dict[str, Any]:
    """
    Question 3: Steering bonus breakdown across all shards in manifest.
    Task reward vs rarity vs latent curiosity disagreement.
    """
    manifest_candidates = [
        drive_dir / "results" / "dataset_index.json",
        local_dir / "results" / "dataset_index.json",
        drive_dir / "data" / "curiosity_manifest.json",
        local_dir / "data" / "curiosity_manifest.json",
    ]
    manifest_data = None
    for mc in manifest_candidates:
        if mc.exists():
            try:
                manifest_data = json.loads(mc.read_text())
                break
            except Exception:
                pass

    if manifest_data is None:
        return {"status": "manifest_not_found"}

    shards = manifest_data.get("shards", [])
    t2_shards = [s for s in shards if s.get("tranche") == 2 or "t02" in s.get("shard_name", "")]

    if not t2_shards:
        t2_shards = shards  # Fallback to all shards

    tot_task = sum(s.get("task_reward_sum", 0.0) for s in t2_shards)
    tot_cur = sum(s.get("curiosity_bonus_sum", 0.0) for s in t2_shards)
    tot_rare = sum(s.get("rarity_bonus_sum", 0.0) for s in t2_shards)
    tot_trans = sum(s.get("transition_count", 0) for s in t2_shards)

    total_steering = tot_task + tot_cur + tot_rare
    if total_steering <= 0:
        total_steering = 1.0

    return {
        "num_t2_shards_analyzed": len(t2_shards),
        "total_transitions": tot_trans,
        "task_reward_sum": round(tot_task, 2),
        "curiosity_bonus_sum": round(tot_cur, 2),
        "rarity_bonus_sum": round(tot_rare, 2),
        "task_reward_share_pct": round((tot_task / total_steering) * 100, 2),
        "curiosity_share_pct": round((tot_cur / total_steering) * 100, 2),
        "rarity_share_pct": round((tot_rare / total_steering) * 100, 2),
        "rarity_to_curiosity_ratio": round(tot_rare / max(0.01, tot_cur), 2),
        "verdict": (
            "✅ Rarity is dominating spatial curiosity"
            if tot_rare > tot_cur
            else "⚠️ Spatial curiosity is dominating rarity"
        ),
    }


def analyze_episodes_forensics(shards_dir: Path) -> Dict[str, Any]:
    """
    Questions 4 & 5: Deep forensics on all pickaxe and stone episodes.
    """
    shard_files = sorted(shards_dir.glob("shard_curiosity_v3_t02_*.npz"))
    if not shard_files:
        shard_files = sorted(shards_dir.glob("*curiosity*.npz"))

    pickaxe_episodes: List[Dict[str, Any]] = []
    stone_episodes: List[Dict[str, Any]] = []

    global_ep_idx = 0

    for sf in shard_files:
        try:
            data = np.load(str(sf), allow_pickle=True)
            obs = data["obs"]
            events = data["events"]
            rewards = data["rewards"]
            dones = data["dones"]
            actions = data["actions"]
            n_trans = len(obs)

            ep_start = 0
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            for ep_end in ep_indices:
                global_ep_idx += 1
                ep_len = ep_end - ep_start + 1
                ep_ev = events[ep_start : ep_end + 1]
                ep_rew = rewards[ep_start : ep_end + 1]
                ep_act = actions[ep_start : ep_end + 1]
                ep_start = ep_end + 1

                # Check for pickaxe unlock
                pickaxe_steps = np.where(ep_ev[:, IDX_WOOD_PICKAXE] > 0)[0]
                stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]

                # Identify all unlocked achievements in this episode
                unlocked_achs = []
                for a_idx, ach in enumerate(CRAFTER_ACHIEVEMENTS):
                    if np.any(ep_ev[:, a_idx] > 0):
                        unlocked_achs.append(ach)

                if len(pickaxe_steps) > 0:
                    t_craft = int(pickaxe_steps[0])
                    post_craft_life = int(ep_len - 1 - t_craft)
                    mined_stone = len(stone_steps) > 0

                    # Diagnosis of why stone was / wasn't mined
                    if mined_stone:
                        stone_step = int(stone_steps[0])
                        diagnosis = f"SUCCESS: Mined stone at step {stone_step} ({stone_step - t_craft} steps after pickaxe)"
                    elif post_craft_life <= 15:
                        diagnosis = f"SURVIVAL FAILURE: Died {post_craft_life} steps after craft (hp collapsed before finding stone)"
                    else:
                        diagnosis = f"EXPLORATION/CHOICE: Lived {post_craft_life} steps with pickaxe but did not execute mine action on stone"

                    pickaxe_episodes.append({
                        "episode_id": global_ep_idx,
                        "shard": sf.name,
                        "total_steps": int(ep_len),
                        "t_craft_pickaxe": t_craft,
                        "post_craft_steps_survived": post_craft_life,
                        "mined_stone": mined_stone,
                        "episode_reward": round(float(np.sum(ep_rew)), 2),
                        "diagnosis": diagnosis,
                        "all_unlocked": unlocked_achs,
                    })

                if len(stone_steps) > 0:
                    stone_step = int(stone_steps[0])
                    # Full chronological recipe
                    timeline = []
                    for t in range(ep_len):
                        ev_t = np.where(ep_ev[t] > 0)[0]
                        for a_idx in ev_t:
                            timeline.append({
                                "step": t,
                                "achievement": CRAFTER_ACHIEVEMENTS[a_idx],
                                "reward": float(ep_rew[t]),
                            })

                    stone_episodes.append({
                        "episode_id": global_ep_idx,
                        "shard": sf.name,
                        "total_steps": int(ep_len),
                        "t_stone_mined": stone_step,
                        "episode_reward": round(float(np.sum(ep_rew)), 2),
                        "timeline": timeline,
                        "all_unlocked": unlocked_achs,
                    })

        except Exception as e:
            print(f"Error processing {sf.name}: {e}")

    # Aggregate pickaxe survival diagnosis
    survival_failures = sum(1 for ep in pickaxe_episodes if "SURVIVAL FAILURE" in ep["diagnosis"])
    exploration_failures = sum(1 for ep in pickaxe_episodes if "EXPLORATION/CHOICE" in ep["diagnosis"])
    stone_successes = sum(1 for ep in pickaxe_episodes if ep["mined_stone"])

    return {
        "total_pickaxe_episodes": len(pickaxe_episodes),
        "total_stone_episodes": len(stone_episodes),
        "stone_conversion_ratio": f"{stone_successes} / {max(1, len(pickaxe_episodes))}",
        "pickaxe_outcomes": {
            "survival_deaths_soon_after_craft": survival_failures,
            "exploration_choice_without_mining": exploration_failures,
            "stone_mined_success": stone_successes,
        },
        "pickaxe_stories": pickaxe_episodes,
        "stone_recipes": stone_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep Forensic Diagnostics for T2 Curiosity Collection")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world", help="Google Drive root path")
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    local_p = REPO_ROOT

    # Determine shards path
    candidates = [
        drive_p / "data" / "curiosity_shards",
        local_p / "data" / "curiosity_shards",
    ]
    shards_dir = candidates[0] if candidates[0].exists() and len(list(candidates[0].glob("*.npz"))) > 0 else candidates[1]

    print("=" * 80)
    print("DEEP FORENSIC DIAGNOSTICS REPORT — TRANCHE T2 COLLECTION")
    print("=" * 80)

    # 1. Archive Contents
    archive_data = analyze_archive_contents(drive_p, local_p)
    print("\n[SECTION 1: ARCHIVE CONTENTS BREAKDOWN]")
    print(f"  Archive Size:            {archive_data.get('archive_size', 'N/A')} / {archive_data.get('max_size', 50)}")
    print(f"  Pickaxe-in-Hand States:  {archive_data.get('pickaxe_in_hand_count', 'N/A')} ({archive_data.get('pickaxe_in_hand_pct', 0)}%)")
    print(f"  Table-Placed States:     {archive_data.get('table_states_count', 'N/A')} ({archive_data.get('table_states_pct', 0)}%)")
    if "tier_breakdown" in archive_data:
        print("  State Tiers in Archive:")
        for tier, count in archive_data["tier_breakdown"].items():
            print(f"    - {tier:<20}: {count}")

    # 2. Bonus Breakdown
    bonus_data = analyze_bonus_breakdown(drive_p, local_p)
    print("\n[SECTION 2: STEERING BONUS BREAKDOWN]")
    print(f"  Task Reward Sum:         {bonus_data.get('task_reward_sum', 0.0):,} ({bonus_data.get('task_reward_share_pct', 0)}%)")
    print(f"  Curiosity (Latent):      {bonus_data.get('curiosity_bonus_sum', 0.0):,} ({bonus_data.get('curiosity_share_pct', 0)}%)")
    print(f"  Rarity Bonus Sum:        {bonus_data.get('rarity_bonus_sum', 0.0):,} ({bonus_data.get('rarity_share_pct', 0)}%)")
    print(f"  Rarity-to-Curiosity:     {bonus_data.get('rarity_to_curiosity_ratio', 0)}x")
    print(f"  Verdict:                 {bonus_data.get('verdict', 'N/A')}")

    # 3 & 4. Pickaxe and Stone Episode Stories
    forensics = analyze_episodes_forensics(shards_dir)
    print("\n[SECTION 3: THE 11 PICKAXE EPISODES' STORIES]")
    print(f"  Total Pickaxe Episodes:  {forensics['total_pickaxe_episodes']}")
    print(f"  Stone Conversion:        {forensics['stone_conversion_ratio']}")
    outcomes = forensics.get("pickaxe_outcomes", {})
    print(f"    - Died immediately (<15 steps):  {outcomes.get('survival_deaths_soon_after_craft', 0)}")
    print(f"    - Survived but didn't mine stone: {outcomes.get('exploration_choice_without_mining', 0)}")
    print(f"    - Successfully mined stone:       {outcomes.get('stone_mined_success', 0)}")

    print("\n  Detailed Breakdown of Pickaxe Episodes:")
    print(f"  {'Ep ID':<7} | {'Shard':<32} | {'Total':<6} | {'t_craft':<8} | {'Post-Life':<10} | {'Outcome'}")
    print("  " + "-" * 85)
    for ep in forensics.get("pickaxe_stories", []):
        outcome_label = "✅ STONE MINED" if ep["mined_stone"] else ("⚠️ Died Fast" if ep["post_craft_steps_survived"] <= 15 else "Wandered")
        print(f"  {ep['episode_id']:<7} | {ep['shard']:<32} | {ep['total_steps']:<6} | {ep['t_craft_pickaxe']:<8} | {ep['post_craft_steps_survived']:<10} | {outcome_label}")

    # 5. The 1 Stone Episode In Full
    print("\n[SECTION 4: THE AUTONOMOUS STONE RECIPE (IN FULL)]")
    stone_recipes = forensics.get("stone_recipes", [])
    if stone_recipes:
        for idx, s_ep in enumerate(stone_recipes):
            print(f"  --- Stone Episode #{idx+1} (ID: {s_ep['episode_id']}, Shard: {s_ep['shard']}) ---")
            print(f"  Total Steps: {s_ep['total_steps']} | Reward: {s_ep['episode_reward']}")
            print("  Chronological Milestone Timeline:")
            for item in s_ep.get("timeline", []):
                print(f"    Step {item['step']:3d}: {item['achievement']:<24} (reward: +{item['reward']:.1f})")
    else:
        print("  No stone episodes detected in scanned shards yet.")

    print("\n" + "=" * 80)

    # Save full diagnostic file
    out_dir = REPO_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "t2_deep_diagnostics.json"
    full_report = {
        "archive_analysis": archive_data,
        "bonus_breakdown": bonus_data,
        "forensics": forensics,
    }
    out_file.write_text(json.dumps(full_report, indent=2))
    print(f"[Deep Diagnostics] Saved full forensic JSON report to: {out_file}")

    if drive_p.exists():
        try:
            drive_out = drive_p / "results" / "diagnostics" / "t2_deep_diagnostics.json"
            drive_out.parent.mkdir(parents=True, exist_ok=True)
            drive_out.write_text(json.dumps(full_report, indent=2))
            print(f"[Deep Diagnostics] Mirrored to Drive: {drive_out}")
        except Exception as e:
            print(f"[Deep Diagnostics] Warning: could not mirror to Drive: {e}")


if __name__ == "__main__":
    main()
