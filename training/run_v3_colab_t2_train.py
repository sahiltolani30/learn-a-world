#!/usr/bin/env python3
"""
training/run_v3_colab_t2_train.py — Phase 3 Tranche T2 Joint Retrain & Screening
================================================================================

Executes the full post-T2 training and screening protocol on Google Colab GPU:
1. Ground-Truth Data Verification (Zero-RAM Audit):
   - Locates base demonstration snapshots: structured_random_final, teacher_hybrid_final, planner_v2_final
   - Counts all 94 T2 curiosity exploration shards (~195k transitions)
   - Verifies total joint buffer >= 450k transitions
2. Joint Buffer Loading:
   - Merges base demonstrations + T2 exploration data into single EpisodeBuffer (~520k transitions)
3. Joint Training (15,000 steps):
   - Starts strictly from clean baseline checkpoint: jepa_crafter_v3_phase2.pt
   - Event-balanced sequence sampling
   - Trains visual encoder, recurrent core (GRU), event heads, and clipped Double-Q critic
   - Saves: checkpoints/jepa_crafter_v3_t2_joint.pt on Drive
4. Screening Evaluation (50 episodes, seeds 2050–2099):
   - Evaluates retrained model on untouched screening seeds
   - Computes official Crafter score (geometric mean)
   - Gate comparison: Pre-Phase-3 baseline (2.31%) vs v2 champion (2.74%) vs Target (3.0%+)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.run_step2_v3_build import train_v3_recurrent
from training.run_v3_colab_run3 import (
    find_drive_base_shards,
    verify_joint_shards,
    build_joint_buffer,
    _copy_to_drive,
    _ensure_local_checkpoint,
)


def run_t2_joint_pipeline(
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
    base_checkpoint: Optional[str] = None,
    num_steps: int = 15_000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_screening: bool = False,
    screen_episodes: int = 50,
    screen_seed_start: int = 2050,
) -> Dict[str, Any]:
    drive_path = Path(drive_dir)
    print("=" * 80)
    print("          LEARN-A-WORLD v3: TRANCHE T2 JOINT RETRAIN PIPELINE")
    print(f"  Google Drive Path:   {drive_path}")
    print(f"  Training Steps:      {num_steps:,} | Batch: {batch_size} | Seq: {seq_len} | LR: {lr}")
    print(f"  Device:              {device}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Verify & Locate Shards on Drive
    # ──────────────────────────────────────────────────────────────────────────
    base_shards = find_drive_base_shards(str(drive_path))
    curiosity_dir = drive_path / "data" / "curiosity_shards"

    composition = verify_joint_shards(
        structured_random_final=base_shards["structured_random_final"],
        teacher_hybrid_final=base_shards["teacher_hybrid_final"],
        planner_v2_final=base_shards["planner_v2_final"],
        curiosity_dir=str(curiosity_dir),
    )

    print(f"\n[Verified] Total Transitions in Joint Dataset: {composition['total_transitions']:,}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Load Joint Buffer
    # ──────────────────────────────────────────────────────────────────────────
    buffer, comp_loaded = build_joint_buffer(
        structured_random_final=base_shards["structured_random_final"],
        teacher_hybrid_final=base_shards["teacher_hybrid_final"],
        planner_v2_final=base_shards["planner_v2_final"],
        curiosity_dir=str(curiosity_dir),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Joint Training Execution
    # ──────────────────────────────────────────────────────────────────────────
    if base_checkpoint is None:
        base_checkpoint = str(drive_path / "checkpoints" / "jepa_crafter_v3_phase2.pt")
    local_ckpt = _ensure_local_checkpoint(base_checkpoint, drive_path)

    out_ckpt = drive_path / "checkpoints" / "jepa_crafter_v3_t2_joint.pt"
    out_json = drive_path / "results" / "step3_t2_joint_train_results.json"

    print("\n" + "=" * 80)
    print("EXECUTING 15,000-STEP T2 JOINT RETRAINING")
    print(f"  Starting Base:  {local_ckpt}")
    print(f"  Joint Buffer:   {buffer.total_transitions:,} transitions")
    print(f"  Target Model:   {out_ckpt}")
    print("=" * 80)

    train_results = train_v3_recurrent(
        checkpoint_base=str(local_ckpt),
        shards_dir="",
        total_steps=num_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        device_name=device,
        output_checkpoint=str(out_ckpt),
        output_json=str(out_json),
        buffer=buffer,
    )

    train_results["buffer_composition"] = comp_loaded
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(train_results, indent=2))
    print(f"\n[Training Complete] Checkpoint saved: {out_ckpt}")

    # Also mirror to local repo
    local_out_json = REPO_ROOT / "results" / "step3_t2_joint_train_results.json"
    local_out_json.parent.mkdir(parents=True, exist_ok=True)
    local_out_json.write_text(json.dumps(train_results, indent=2))

    if skip_screening:
        print("[Screening] Skipped as requested via --skip-screening.")
        return train_results

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: 50-Episode Screening (Seeds 2050–2099)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"EXECUTING 50-EPISODE SCREENING EVALUATION (SEEDS {screen_seed_start}..{screen_seed_start + screen_episodes - 1})")
    print("=" * 80)

    screen_out_json = drive_path / "results" / "screen_v3_t2_joint_results.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "screen_v3_50eps.py"),
        "--checkpoint", str(out_ckpt),
        "--num_episodes", str(screen_episodes),
        "--seed_start", str(screen_seed_start),
        "--output_json", str(screen_out_json),
        "--device", device,
    ]
    subprocess.run(cmd, check=True, text=True)

    screen_results = json.loads(screen_out_json.read_text())
    screen_results["buffer_composition"] = comp_loaded
    screen_out_json.write_text(json.dumps(screen_results, indent=2))

    # Mirror to local
    local_screen_json = REPO_ROOT / "results" / "screen_v3_t2_joint_results.json"
    local_screen_json.parent.mkdir(parents=True, exist_ok=True)
    local_screen_json.write_text(json.dumps(screen_results, indent=2))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Gate Check & Comparative Report
    # ──────────────────────────────────────────────────────────────────────────
    v3_base = 2.313
    v2_champ = 2.74
    t2_score = float(screen_results.get("crafter_score", 0.0))
    zombie_pct = float(screen_results.get("unlock_percentages", {}).get("defeat_zombie", 0.0))

    score_pass = t2_score > v3_base
    champ_pass = t2_score >= v2_champ
    gate_passed = score_pass and zombie_pct > 0.0

    print("\n" + "=" * 80)
    print("TRANCHE T2 JOINT SCREENING VERDICT REPORT")
    print("=" * 80)
    print(f"  Pre-Phase-3 Baseline Score (v3 Phase 2) : {v3_base:.2f}%")
    print(f"  v2 Champion Paired Target               : {v2_champ:.2f}%")
    print(f"  T2 Joint Measured Score                 : {t2_score:.2f}%")
    print(f"  defeat_zombie Unlock Rate               : {zombie_pct:.1f}%")
    print()
    print(f"  Beats Baseline (> 2.31%)                : {'✅ PASSED' if score_pass else '❌ FAILED'}")
    print(f"  Beats v2 Champion (>= 2.74%)            : {'🎯 CHAMPION BEATEN!' if champ_pass else '⚠️ Below v2'}")
    print(f"  Combat Gate (defeat_zombie > 0%)        : {'✅ PASSED' if zombie_pct > 0 else '❌ FAILED'}")
    print(f"  Overall Gate Verdict                    : {'✅ ADVANCE TO PHASE 4' if gate_passed else '❌ FAILED'}")
    print("=" * 80)

    print("\nUnlock Breakdown (Seeds 2050–2099):")
    for ach, pct in sorted(screen_results.get("unlock_percentages", {}).items(), key=lambda x: -x[1]):
        if pct > 0:
            bar = "#" * int(round(pct / 5))
            print(f"  {ach:<22}: {pct:5.1f}% | {bar}")

    return screen_results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Tranche T2 Joint Retraining & Screening Pipeline")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world", help="Google Drive path")
    parser.add_argument("--num-steps", type=int, default=15_000, help="Total joint training gradient steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of sequence chunks")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for recurrent BPTT")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name")
    parser.add_argument("--skip-screening", action="store_true", help="Skip 50-episode screening evaluation")
    parser.add_argument("--screen-episodes", type=int, default=50, help="Number of screening episodes")
    parser.add_argument("--screen-seed-start", type=int, default=2050, help="Start seed for screening (2050-2099)")
    args = parser.parse_args()

    run_t2_joint_pipeline(
        drive_dir=args.drive_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
        skip_screening=args.skip_screening,
        screen_episodes=args.screen_episodes,
        screen_seed_start=args.screen_seed_start,
    )


if __name__ == "__main__":
    main()
