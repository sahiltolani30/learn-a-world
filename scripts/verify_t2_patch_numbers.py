#!/usr/bin/env python3
"""
scripts/verify_t2_patch_numbers.py — Exact 3-Point Forensic Verification for T2
================================================================================

Reports the 3 exact numbers requested by Sahil & Claude:
1. PATCH PROOF: Weighted steering shares for post-patch shards only.
2. RESTARTS + ARCHIVE: Restart fraction, archive size, mask histogram, pickaxe states & usage.
3. STONE ORIGINS: Detailed origin (fresh vs restart [seed, step]) for all 12 stone episodes.
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

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)
IDX_COAL = ACH_TO_IDX.get("collect_coal", 0)


def load_manifest(drive_p: Path, local_p: Path) -> List[Dict[str, Any]]:
    candidates = [
        drive_p / "data" / "curiosity_manifest.json",
        drive_p / "results" / "dataset_index.json",
        local_p / "data" / "curiosity_manifest.json",
        local_p / "results" / "dataset_index.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text())
                shards = data.get("shards", [])
                t2_shards = [s for s in shards if s.get("tranche") == 2 or "t02" in s.get("shard_name", "")]
                if t2_shards:
                    return t2_shards
            except Exception:
                pass
    return []


def load_archive_summary(drive_p: Path, local_p: Path) -> Dict[str, Any]:
    candidates = [
        drive_p / "results" / "archive_summary_t2.json",
        local_p / "results" / "archive_summary_t2.json",
        drive_p / "results" / "t2_collection_summary.json",
        local_p / "results" / "t2_collection_summary.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text())
                if "archive_summary" in data:
                    return data["archive_summary"]
                return data
            except Exception:
                pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Verify 3 critical patch numbers")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    local_p = REPO_ROOT

    # Determine shards path
    candidates = [
        drive_p / "data" / "curiosity_shards",
        local_p / "data" / "curiosity_shards",
    ]
    shards_dir = candidates[0] if candidates[0].exists() and len(list(candidates[0].glob("*.npz"))) > 0 else candidates[1]

    shard_files = sorted(shards_dir.glob("shard_curiosity_v3_t02_*.npz"))
    if not shard_files:
        shard_files = sorted(shards_dir.glob("*.npz"))

    manifest_shards = load_manifest(drive_p, local_p)
    archive_summary = load_archive_summary(drive_p, local_p)

    print("=" * 80)
    print("T2 COLLECTOR IMPROVEMENT PATCH VERIFICATION REPORT")
    print(f"Total Shards Found: {len(shard_files)}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. PATCH PROOF: Weighted Steering Shares (Post-Patch Shards Only)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. PATCH PROOF — WEIGHTED STEERING SHARES (POST-PATCH ONLY)")
    print("=" * 80)

    # Identify post-patch shards: shards containing weighted_shares or weighted_task_sum, or index >= 17
    post_patch_manifest = []
    pre_patch_manifest = []

    for s in manifest_shards:
        if "weighted_shares" in s or "weighted_task_sum" in s:
            post_patch_manifest.append(s)
        else:
            # Fallback by shard name index >= 17
            sname = s.get("shard_name", "")
            try:
                sidx = int(sname.split("_")[-1].replace(".npz", ""))
                if sidx >= 17:
                    post_patch_manifest.append(s)
                else:
                    pre_patch_manifest.append(s)
            except Exception:
                pre_patch_manifest.append(s)

    post_patch_names = [s.get("shard_name", "") for s in post_patch_manifest]
    pre_patch_names = [s.get("shard_name", "") for s in pre_patch_manifest]

    print(f"Pre-Patch Shards Count:   {len(pre_patch_manifest)} ({pre_patch_names[0] if pre_patch_names else 'none'} .. {pre_patch_names[-1] if pre_patch_names else 'none'})")
    print(f"Post-Patch Shards Count:  {len(post_patch_manifest)} ({post_patch_names[0] if post_patch_names else 'none'} .. {post_patch_names[-1] if post_patch_names else 'none'})")
    print(f"Post-Patch Shard List:    {', '.join([s.replace('shard_curiosity_v3_t02_', '').replace('.npz', '') for s in post_patch_names])}")

    sum_w_task = sum(float(s.get("weighted_task_sum", 0.0)) for s in post_patch_manifest)
    sum_w_cur = sum(float(s.get("weighted_cur_sum", 0.0)) for s in post_patch_manifest)
    sum_w_rare = sum(float(s.get("weighted_rare_sum", 0.0)) for s in post_patch_manifest)
    sum_w_unc = sum(float(s.get("weighted_unc_sum", 0.0)) for s in post_patch_manifest)

    w_total = abs(sum_w_task) + abs(sum_w_cur) + abs(sum_w_rare) + abs(sum_w_unc)
    denom = max(1e-8, w_total)

    pct_task = (abs(sum_w_task) / denom) * 100.0
    pct_cur = (abs(sum_w_cur) / denom) * 100.0
    pct_rare = (abs(sum_w_rare) / denom) * 100.0
    pct_unc = (abs(sum_w_unc) / denom) * 100.0

    print("\nPost-Patch Weighted Component Sums:")
    print(f"  weighted_task_sum:  {sum_w_task:>12.3f} ({pct_task:>5.1f}%)")
    print(f"  weighted_cur_sum:   {sum_w_cur:>12.3f} ({pct_cur:>5.1f}%)")
    print(f"  weighted_rare_sum:  {sum_w_rare:>12.3f} ({pct_rare:>5.1f}%)")
    print(f"  weighted_unc_sum:   {sum_w_unc:>12.3f} ({pct_unc:>5.1f}%)")
    print(f"  -----------------------------------------------")
    print(f"  WEIGHTED RARE SHARE: {pct_rare:.2f}%")
    print(f"  Steering Mix Status: {'✅ HEALTHY (Rarity >= 25% of steering)' if pct_rare >= 25.0 else '⚠️ Low Rarity Share'}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. RESTARTS + ARCHIVE
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. RESTARTS + ARCHIVE STATUS")
    print("=" * 80)

    # Post-patch restarts from manifests
    total_post_eps = sum(s.get("episode_count", 0) for s in post_patch_manifest)
    total_post_restarts = sum(s.get("restart_episodes_in_shard", 0) for s in post_patch_manifest)
    manifest_restart_frac = (total_post_restarts / max(1, total_post_eps)) * 100.0

    print(f"Post-Patch Episodes:      {total_post_eps}")
    print(f"Post-Patch Restart Eps:   {total_post_restarts}")
    print(f"Post-Patch Restart Rate:  {manifest_restart_frac:.1f}% (Spec target: 40%)")

    arch_size = archive_summary.get("size", "N/A")
    total_restarts_triggered = archive_summary.get("total_restarts", archive_summary.get("total_restarts_triggered", 0))
    random_restarts = archive_summary.get("total_random_restarts", 0)
    stagnation_restarts = archive_summary.get("total_stagnation_restarts", 0)

    print(f"Archive Size:             {arch_size} / {archive_summary.get('max_size', 50)}")
    print(f"Total Restarts in Arch:   {total_restarts_triggered} (Random: {random_restarts}, Stagnation: {stagnation_restarts})")

    # Mask histogram
    mask_hist = archive_summary.get("mask_histogram", {})
    if not mask_hist and post_patch_manifest:
        # Check last shard manifest entry
        mask_hist = post_patch_manifest[-1].get("archive_mask_histogram", {})
    print(f"Archive Mask Histogram:   {json.dumps(mask_hist)}")

    # Pickaxe entries in archive
    archived_achs = archive_summary.get("archived_achievements", [])
    pickaxe_entries = [i for i, achs in enumerate(archived_achs) if IDX_WOOD_PICKAXE in achs]
    print(f"Pickaxe-Mask Entries:     {len(pickaxe_entries)} states in archive")
    if pickaxe_entries:
        print(f"  Entry indices carrying pickaxe: {pickaxe_entries}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. STONE ORIGINS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. STONE ORIGINS FORENSICS (ALL STONE EPISODES)")
    print("=" * 80)

    stone_episodes = []
    global_ep_idx = 0

    for sf in shard_files:
        try:
            data = np.load(str(sf), allow_pickle=True)
            obs = data["obs"]
            events = data["events"]
            dones = data["dones"]
            rewards = data["rewards"]
            masks_before = data["mask_before"] if "mask_before" in data else None
            episode_origins = data["episode_origins"] if "episode_origins" in data else None

            n_trans = len(obs)
            ep_start = 0
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            ep_in_shard_idx = 0
            for ep_end in ep_indices:
                global_ep_idx += 1
                ep_ev = events[ep_start : ep_end + 1]
                ep_len = ep_end - ep_start + 1

                # Check if stone was collected in this episode
                stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]
                if len(stone_steps) > 0:
                    t_stone = int(stone_steps[0])

                    # Check pickaxe status:
                    # 1. Did initial step have pickaxe in mask_before?
                    init_mask = masks_before[ep_start] if masks_before is not None else None
                    had_pickaxe_at_start = bool(init_mask[IDX_WOOD_PICKAXE] > 0) if init_mask is not None else False

                    # 2. Was pickaxe crafted in this episode?
                    pickaxe_steps = np.where(ep_ev[:, IDX_WOOD_PICKAXE] > 0)[0]
                    t_craft_pickaxe = int(pickaxe_steps[0]) if len(pickaxe_steps) > 0 else -1

                    # 3. Origin from episode_origins array
                    origin_type = "Pre-patch (no tag)"
                    origin_seed = "N/A"
                    origin_step = "N/A"

                    if episode_origins is not None and ep_in_shard_idx < len(episode_origins):
                        orig = episode_origins[ep_in_shard_idx]
                        origin_seed = int(orig[0])
                        origin_step = int(orig[1])
                        if origin_step == 0:
                            origin_type = f"Fresh Episode (seed {origin_seed})"
                        else:
                            origin_type = f"Restart Episode (from seed {origin_seed}, step {origin_step})"

                    stone_episodes.append({
                        "stone_id": len(stone_episodes) + 1,
                        "global_ep": global_ep_idx,
                        "shard": sf.name,
                        "ep_len": int(ep_len),
                        "t_stone": t_stone,
                        "had_pickaxe_at_start": had_pickaxe_at_start,
                        "t_craft_pickaxe": t_craft_pickaxe,
                        "origin_type": origin_type,
                        "origin_seed": origin_seed,
                        "origin_step": origin_step,
                    })

                ep_start = ep_end + 1
                ep_in_shard_idx += 1

        except Exception as e:
            print(f"Error scanning {sf.name}: {e}")

    print(f"Total Stone Episodes Detected: {len(stone_episodes)}")
    print(f"\n{'#':<3} | {'Global Ep':<10} | {'Shard':<32} | {'t_stone':<8} | {'Pickaxe at Start':<18} | {'t_craft':<8} | {'Origin'}")
    print("-" * 115)
    for se in stone_episodes:
        pickaxe_str = "YES (restored)" if se["had_pickaxe_at_start"] else "No"
        craft_str = str(se["t_craft_pickaxe"]) if se["t_craft_pickaxe"] >= 0 else "None"
        print(f"{se['stone_id']:<3} | {se['global_ep']:<10} | {se['shard']:<32} | {se['t_stone']:<8} | {pickaxe_str:<18} | {craft_str:<8} | {se['origin_type']}")

    print("=" * 115)


if __name__ == "__main__":
    main()
