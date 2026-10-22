"""
Google Colab Campaign Runner for Learn-a-World v3: Run 1 (Steps 1 & 2).
Automates GPU execution on Google Colab with automatic Google Drive persistence:
1. Stage 1 (Step 1 Value Fix): Retrains Clipped Double-Q critic with 10-step Bellman targets
   and DreamerV3 percentile return normalization (2,000 steps on GPU).
2. Stage 2 (Step 2 v3 Recurrent Model): Trains RecurrentCore GRU dynamics + task-shaped
   end-to-end representation learning (2,000 steps on GPU).
3. Evaluates promotion gates: 10-step imagination MSE < 0.85, Wood PR-AUC >= 0.08, Pearson r > 0.15.
4. Persists checkpoints directly into Google Drive.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from experiments.utils import get_device
from training.run_step1_value_fix import train_value_fix
from training.run_step2_v3_build import train_v3_recurrent


def run_colab_run1(
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
    step1_steps: int = 10000,
    step2_steps: int = 5000,
    batch_size: int = 16,
    recurrent_dim: int = 256,
    device_override: str = None,
):
    print("=" * 80)
    print("        LEARN-A-WORLD v3: COLAB GPU RUN 1 (STEPS 1 & 2)")
    print("        Step 1: Value Critic Fix (10-Step Bellman + Return Normalization)")
    print("        Step 2: Recurrent JEPA v3 + Task-Shaped Latent Representation")
    print("=" * 80)

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Compute Device: {device}")
    if device == "cuda":
        print(f"--> GPU Device Name: {torch.cuda.get_device_name(0)}")

    drive_path = Path(drive_dir)
    if drive_path.exists():
        print(f"--> Google Drive Detected: {drive_path}")
        shards_dir = drive_path / "data" / "sequences"
        ckpt_dir = drive_path / "checkpoints"
        results_dir = drive_path / "results"
    else:
        print("--> Note: Google Drive not mounted at default path; falling back to workspace.")
        shards_dir = repo_root / "data" / "sequences"
        ckpt_dir = repo_root / "checkpoints"
        results_dir = repo_root / "results"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint = ckpt_dir / "jepa_crafter_v2_best.pt"
    if not base_checkpoint.exists():
        local_base = repo_root / "checkpoints" / "jepa_crafter_v2_best.pt"
        if local_base.exists():
            base_checkpoint = local_base

    step1_checkpoint = ckpt_dir / "jepa_crafter_v3_step1_value.pt"
    step1_json = results_dir / "colab_step1_results.json"

    step2_checkpoint = ckpt_dir / "jepa_crafter_v3_phase2.pt"
    step2_json = results_dir / "colab_step2_results.json"

    # =========================================================================
    # STAGE 1: STEP 1 VALUE FIX
    # =========================================================================
    print("\n" + "#" * 80)
    print("  STAGE 1: VALUE CRITIC RETRAINING (10-Step Bellman Targets)")
    print("#" * 80)

    t0_step1 = time.time()
    step1_results = train_value_fix(
        checkpoint_path=str(base_checkpoint),
        shards_dir=str(shards_dir),
        total_steps=step1_steps,
        batch_size=32 if device == "cuda" else 8,
        seq_len=16,
        n_step=10,
        gamma=0.99,
        lr=3e-4,
        device_name=device,
        output_checkpoint=str(step1_checkpoint),
        output_json=str(step1_json),
    )
    t_step1 = time.time() - t0_step1
    print(f"\n--> Stage 1 Completed in {t_step1:.1f}s")
    print(f"--> Final Pearson r: {step1_results['final_calibration']['pearson_r']:.4f}")
    print(f"--> Final Q std: {step1_results['final_calibration']['q_pred_std']:.4f}")
    print(f"--> Checkpoint saved to: {step1_checkpoint}")

    # =========================================================================
    # STAGE 2: STEP 2 RECURRENT JEPA v3 + TASK SHAPING
    # =========================================================================
    print("\n" + "#" * 80)
    print("  STAGE 2: RECURRENT JEPA v3 & TASK-SHAPED REPRESENTATION")
    print("#" * 80)

    t0_step2 = time.time()
    step2_results = train_v3_recurrent(
        checkpoint_base=str(step1_checkpoint),
        shards_dir=str(shards_dir),
        total_steps=step2_steps,
        batch_size=batch_size,
        seq_len=16,
        recurrent_dim=recurrent_dim,
        lr=2e-4,
        device_name=device,
        output_checkpoint=str(step2_checkpoint),
        output_json=str(step2_json),
    )
    t_step2 = time.time() - t0_step2
    print(f"\n--> Stage 2 Completed in {t_step2:.1f}s")
    print(f"--> Final 10-Step Rollout MSE: {step2_results['final_metrics']['10_step_rollout_mse']:.4f}")
    print(f"--> Final Wood PR-AUC: {step2_results['final_metrics']['wood_pr_auc']:.4f}")
    print(f"--> Checkpoint saved to: {step2_checkpoint}")

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    summary = {
        "device": device,
        "runtime_seconds": t_step1 + t_step2,
        "step1": {
            "checkpoint": str(step1_checkpoint),
            "final_pearson_r": step1_results["final_calibration"]["pearson_r"],
            "final_q_std": step1_results["final_calibration"]["q_pred_std"],
            "gate_passed": step1_results["final_calibration"]["pearson_r"] > 0.15,
        },
        "step2": {
            "checkpoint": str(step2_checkpoint),
            "final_10_step_mse": step2_results["final_metrics"]["10_step_rollout_mse"],
            "final_wood_pr_auc": step2_results["final_metrics"]["wood_pr_auc"],
            "final_macro_pr_auc": step2_results["final_metrics"]["macro_pr_auc"],
            "gate_passed": step2_results["gate_passed"],
        },
    }

    summary_file = results_dir / "colab_run1_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("                    COLAB RUN 1 CAMPAIGN COMPLETE")
    print("=" * 80)
    print(f"Total GPU Training Time: {(t_step1 + t_step2)/60:.1f} minutes")
    print(f"Stage 1 Gate (Pearson r > 0.15): {'PASSED' if summary['step1']['gate_passed'] else 'NEEDS MORE DATA'}")
    print(f"Stage 2 Gate (10-step MSE < 0.85 & Wood PR-AUC >= 0.08): {'PASSED' if summary['step2']['gate_passed'] else 'IN-PROGRESS'}")
    print(f"Summary JSON saved to: {summary_file}")
    print(f"Base Checkpoint for Phase 3: {step2_checkpoint}")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Learn-a-World v3 Colab Campaign Run 1.")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--step1_steps", type=int, default=10000)
    parser.add_argument("--step2_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--recurrent_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_colab_run1(
        drive_dir=args.drive_dir,
        step1_steps=args.step1_steps,
        step2_steps=args.step2_steps,
        batch_size=args.batch_size,
        recurrent_dim=args.recurrent_dim,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
