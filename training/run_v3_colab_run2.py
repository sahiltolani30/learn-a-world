"""
Colab Run 2 Campaign Script — Phase 3 T1 (Plan2Explore Curiosity Collection + Incremental Retrain).

Stages:
  1. Collect T1: +160k self-collected transitions using Plan2Explore curiosity
     (seeds 5000+). Shards + manifest saved to Drive.
  2. Incremental Retrain: from jepa_crafter_v3_phase2.pt on ALL data
     (old 8k shards + new 160k curiosity shards).
  3. T1 Screening: screen best checkpoint on seeds 2050-2099 (50 eps).
  4. T1 Gate check: print pass/fail inline.

Designed to run cell-by-cell in a Colab notebook OR as a single script.
Saves full training state to Drive every 500 steps; keeps last 2 + best.

Usage (Colab cells):
    from training.run_v3_colab_run2 import collect_t1, retrain_t1, screen_t1, gate_check_t1

    manifest = collect_t1(...)
    ckpt_path = retrain_t1(...)
    results   = screen_t1(ckpt_path, ...)
    gate_check_t1(results)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.collector_v3 import CollectorV3
from training.run_step1_value_fix import load_buffer_from_shards
from training.run_step2_v3_build import evaluate_v3_metrics, train_v3_recurrent
from training.sequence_buffer import EpisodeBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: T1 Collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_t1(
    drive_dir: str,
    model_checkpoint: Optional[str] = None,
    curiosity_checkpoint: Optional[str] = None,
    target_new_transitions: int = 160_000,
    seed_start: int = 5000,
    lambda_task: float = 0.3,
    lambda_cur: float = 1.0,
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    max_steps_per_ep: int = 1000,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Stage 1: Collect T1 (+160k transitions) using Plan2Explore curiosity.

    Args:
        drive_dir: Google Drive path where shards + manifest will be saved.
                   Also used to find the base checkpoint if model_checkpoint is None.
        model_checkpoint: Path to jepa_crafter_v3_phase2.pt.
                          Defaults to drive_dir/checkpoints/jepa_crafter_v3_phase2.pt
        curiosity_checkpoint: Optional path to resume curiosity predictor weights.
        target_new_transitions: Number of new transitions to collect (160k for T1).
        seed_start: First collection seed (5000+ always).

    Returns:
        Manifest summary dict with new_transitions, total_transitions, event_counts.
    """
    drive_path = Path(drive_dir)
    drive_path.mkdir(parents=True, exist_ok=True)

    if model_checkpoint is None:
        model_checkpoint = str(drive_path / "checkpoints" / "jepa_crafter_v3_phase2.pt")

    # Copy checkpoint to local /content/ for faster loading if on Colab
    local_ckpt = _ensure_local_checkpoint(model_checkpoint, drive_path)

    print("=" * 70)
    print("PHASE 3 — T1 COLLECTION (Plan2Explore Curiosity)")
    print(f"  Model:  {local_ckpt}")
    print(f"  Drive:  {drive_dir}")
    print(f"  Target: {target_new_transitions:,} new transitions")
    print(f"  Seeds:  {seed_start}+")
    print(f"  Device: {device}")
    print("=" * 70)

    collector = CollectorV3(
        model_checkpoint=str(local_ckpt),
        curiosity_checkpoint=curiosity_checkpoint,
        drive_dir=str(drive_path / "data" / "curiosity_shards"),
        tranche_id=1,
        lambda_task=lambda_task,
        lambda_cur=lambda_cur,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=cem_iterations,
        max_steps_per_ep=max_steps_per_ep,
        device=device,
    )

    summary = collector.collect(
        target_new_transitions=target_new_transitions,
        seed_start=seed_start,
    )

    print("\nT1 Collection Summary:")
    print(json.dumps(summary, indent=2))
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Incremental Retrain on ALL data
# ──────────────────────────────────────────────────────────────────────────────

def retrain_t1(
    drive_dir: str,
    base_checkpoint: Optional[str] = None,
    num_steps: int = 5_000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    save_every: int = 500,
    device: str = "cuda",
    output_checkpoint: Optional[str] = None,
    output_json: Optional[str] = None,
) -> str:
    """
    Stage 2: Incremental retrain on ALL accumulated data (old shards + T1 curiosity shards).

    Loads existing dataset from:
      - data/sequences/ (original 8k shards)
      - drive_dir/data/curiosity_shards/ (T1 new shards)

    Saves full training state to Drive every save_every steps.
    Returns path to best checkpoint (by val loss).
    """
    drive_path = Path(drive_dir)

    if base_checkpoint is None:
        base_checkpoint = str(drive_path / "checkpoints" / "jepa_crafter_v3_phase2.pt")

    local_ckpt = _ensure_local_checkpoint(base_checkpoint, drive_path)

    if output_checkpoint is None:
        output_checkpoint = str(drive_path / "checkpoints" / "jepa_crafter_v3_t1.pt")

    if output_json is None:
        output_json = str(drive_path / "results" / "step3_t1_train_results.json")

    os.makedirs(Path(output_json).parent, exist_ok=True)

    repo_shards = drive_path / "data" / "sequences"
    if not repo_shards.exists():
        repo_shards = REPO_ROOT / "data" / "sequences"

    # Merge ALL data shards into a single combined shards dir for training
    combined_dir = _merge_all_shards(
        repo_shards_dir=str(repo_shards),
        curiosity_shards_dir=str(drive_path / "data" / "curiosity_shards"),
        combined_dir=str(drive_path / "data" / "combined_shards"),
    )

    print("=" * 70)
    print("PHASE 3 — T1 INCREMENTAL RETRAIN")
    print(f"  Base checkpoint: {local_ckpt}")
    print(f"  Combined data:   {combined_dir}")
    print(f"  Steps:           {num_steps}")
    print(f"  Output:          {output_checkpoint}")
    print("=" * 70)

    # Run training using existing train_v3_recurrent function
    results = train_v3_recurrent(
        checkpoint_base=str(local_ckpt),
        shards_dir=combined_dir,
        total_steps=num_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        device_name=device,
        output_checkpoint=output_checkpoint,
        output_json=output_json,
    )

    # Copy checkpoint to Drive
    _copy_to_drive(output_checkpoint, drive_path / "checkpoints")
    _copy_to_drive(output_json, drive_path / "results")

    print(f"\nT1 retrain complete. Checkpoint: {output_checkpoint}")
    print(f"10-step MSE: {results.get('10_step_rollout_mse', 'N/A')}")
    return output_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: T1 Screening (50 eps, seeds 2050-2099)
# ──────────────────────────────────────────────────────────────────────────────

def screen_t1(
    checkpoint_path: str,
    drive_dir: str,
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    device: str = "cuda",
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 3: Screen T1 checkpoint on seeds 2050-2099 (50 eps).

    Calls scripts/screen_v3_50eps.py with the T1 checkpoint.
    Saves results to results/screen_v3_t1_results.json and copies to Drive.

    Returns the results dict.
    """
    if output_json is None:
        output_json = str(REPO_ROOT / "results" / "screen_v3_t1_results.json")

    os.makedirs(Path(output_json).parent, exist_ok=True)

    print("=" * 70)
    print("PHASE 3 — T1 SCREENING (50 eps, seeds 2050-2099)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_json}")
    print("=" * 70)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "screen_v3_50eps.py"),
        "--checkpoint", checkpoint_path,
        "--num_episodes", str(num_episodes),
        "--seed_start", str(seed_start),
        "--output_json", output_json,
        "--device", device,
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=False)

    # Load and return results
    results = json.loads(Path(output_json).read_text())

    # Copy to Drive before any reporting
    drive_path = Path(drive_dir)
    _copy_to_drive(output_json, drive_path / "results")

    print(f"\nT1 Screening Results (committed BEFORE this print):")
    print(f"  Crafter Score: {results['crafter_score']:.2f}%")
    print(f"  Mean Reward:   {results['mean_episode_reward']:.3f}")
    print(f"  Unique Ach:    {results['unlocked_achievement_count']} / 22")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: T1 Gate Check
# ──────────────────────────────────────────────────────────────────────────────

def gate_check_t1(t1_results: Dict[str, Any], pre_phase3_json: Optional[str] = None) -> bool:
    """
    Prints T1 gate check.
    Gate: T1 screening score must beat pre-Phase-3 v3 score (2.31%).
    Returns True if gate passes, False if it fails.
    """
    if pre_phase3_json is None:
        pre_phase3_json = str(REPO_ROOT / "results" / "screen_v3_50eps_results.json")

    pre_score = 2.313  # default fallback
    if Path(pre_phase3_json).exists():
        pre_data = json.loads(Path(pre_phase3_json).read_text())
        pre_score = pre_data.get("crafter_score", 2.313)

    t1_score = t1_results["crafter_score"]
    passed = t1_score > pre_score

    print("\n" + "=" * 70)
    print("T1 GATE CHECK")
    print(f"  Pre-Phase-3 v3 baseline (seeds 2050-2099): {pre_score:.2f}%")
    print(f"  T1 checkpoint score    (seeds 2050-2099): {t1_score:.2f}%")
    print(f"  Gate: T1 > pre-Phase-3?  →  {'✅ PASSED' if passed else '❌ FAILED — DO NOT PROCEED TO PHASE 4'}")
    print("=" * 70)

    if not passed:
        print("\n[STOP] T1 gate failed. Write a diagnosis note before proceeding.")
        print("  Checklist:")
        print("  1. Did curiosity bonus collapse to near 0 during collection?")
        print("  2. Did retrain overfit to curiosity data (val loss diverged)?")
        print("  3. Is the frontier still empty (stone=0, coal=0)?")
        print("  Compare: dataset_index.json event counts vs pre-T1 8k buffer.")

    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: full run
# ──────────────────────────────────────────────────────────────────────────────

def run_colab_run2_t1(
    drive_dir: str,
    target_new_transitions: int = 160_000,
    seed_start: int = 5000,
    num_train_steps: int = 5_000,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Full Phase 3 T1 campaign: collect → retrain → screen → gate_check.
    Designed to be called from a single Colab cell.
    """
    # Stage 1
    manifest = collect_t1(
        drive_dir=drive_dir,
        target_new_transitions=target_new_transitions,
        seed_start=seed_start,
        device=device,
    )

    # Stage 2
    ckpt_path = retrain_t1(
        drive_dir=drive_dir,
        num_steps=num_train_steps,
        device=device,
    )

    # Stage 3
    results = screen_t1(
        checkpoint_path=ckpt_path,
        drive_dir=drive_dir,
        device=device,
    )

    # Stage 4
    gate_passed = gate_check_t1(results)

    return {
        "collection_summary": manifest,
        "checkpoint_path": ckpt_path,
        "screening_results": results,
        "t1_gate_passed": gate_passed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_local_checkpoint(ckpt_path: str, drive_path: Path) -> Path:
    """
    On Colab: if checkpoint is on Drive (/content/drive/...), copy to /content/
    for faster access. Otherwise return as-is.
    """
    src = Path(ckpt_path)
    if not src.exists():
        # Try drive path
        drive_ckpt = drive_path / "checkpoints" / src.name
        if drive_ckpt.exists():
            local = Path("/content") / src.name
            if not local.exists():
                print(f"[ColabRun2] Copying checkpoint from Drive: {drive_ckpt}")
                shutil.copy(str(drive_ckpt), str(local))
            return local
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return src


def _copy_to_drive(src: str, dst_dir: Path) -> None:
    """Copy a file to Drive directory, creating it if needed."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(src).name
    try:
        if Path(src).resolve() != dst.resolve():
            shutil.copy(src, str(dst))
            print(f"[ColabRun2] Copied to Drive: {dst}")
    except (shutil.SameFileError, ValueError):
        pass


def _merge_all_shards(
    repo_shards_dir: str,
    curiosity_shards_dir: str,
    combined_dir: str,
) -> str:
    """
    Create a combined shards directory containing symlinks/copies of all shards
    from both the original repo data and new curiosity shards.
    Returns path to the combined directory.
    """
    Path(combined_dir).mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_dir in [repo_shards_dir, curiosity_shards_dir]:
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"[ColabRun2] Shards dir not found (skipping): {src_dir}")
            continue
        for npz in src_path.glob("*.npz"):
            dst = Path(combined_dir) / npz.name
            if not dst.exists():
                shutil.copy(str(npz), str(dst))
                copied += 1

    # Copy dataset_index.json from repo shards (training only needs the npz files)
    repo_index = Path(repo_shards_dir) / "dataset_index.json"
    combined_index = Path(combined_dir) / "dataset_index.json"
    if repo_index.exists() and not combined_index.exists():
        shutil.copy(str(repo_index), str(combined_index))

    print(f"[ColabRun2] Combined shards dir: {combined_dir} ({copied} new files copied)")
    return combined_dir
