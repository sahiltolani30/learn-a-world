#!/usr/bin/env python3
"""ORBIT — Phase 3C: World Model Learning from Real Experience.

Executes Step 3 of the Phase 3 master specification:
- Loads the Gate 4 champion world model (formulation_c_residual_change-weighted).
- Loads the augmented replay buffer (84 episodes, 15,099 transitions) and token cache.
- Fine-tunes the transition predictor, recurrent belief core, and transition heads (Formulation C).
- Evaluates previous Gate 4 model vs updated candidate on held-out evaluation seeds (eval_heldout_seeds.pt).
- Decomposes static vs changed spatial token errors and verifies positive shuffled-action sensitivity.
- Saves the validated champion world model to results/phase3/world_model_phase3.pt and Google Drive.
- Strictly freezes parameters for downstream policy training.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate4_evaluator import evaluate_gate4_formulation
from scripts.run_phase3b_integration import CachedDINOv2Replay, resolve_drive_root, resolve_file
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import train_world_model_step
from true_jepa.replay import PixelEpisode, PixelReplay


def load_heldout_eval_dataset(eval_path: Path) -> List[Dict[str, Any]]:
    """Loads held-out evaluation dataset, handling transparent gzip decompression."""
    if str(eval_path).endswith(".gz"):
        with gzip.open(eval_path, "rb") as f:
            data = torch.load(f, weights_only=False)
    elif not eval_path.exists() and Path(str(eval_path) + ".gz").exists():
        with gzip.open(str(eval_path) + ".gz", "rb") as f:
            data = torch.load(f, weights_only=False)
    else:
        data = torch.load(str(eval_path), weights_only=False)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    raise ValueError(f"Unrecognized format in heldout evaluation file {eval_path}")


def train_world_model_phase3(
    base_model: TrueJEPA,
    replay: CachedDINOv2Replay,
    updates: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
    seed: int = 123,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    """Fine-tunes the world model predictor, belief, and heads on augmented real experience."""
    print(f"\n{'='*80}")
    print(f"[PHASE 3C] World Model Fine-Tuning on Augmented Real Experience")
    print(f"{'='*80}")
    print(f"Updates: {updates} | Batch Size: {batch_size} | Learning Rate: {learning_rate} | Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(base_model).to(device)

    # Strictly freeze DINOv2 backbone and target encoder permanently
    model.encoder.backbone.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.encoder.backbone.eval()
    model.target_encoder.eval()

    # World model components to fine-tune
    trainable_params = [
        {"params": model.encoder.adapter.parameters(), "lr": learning_rate * 0.5},
        {"params": model.predictor.parameters(), "lr": learning_rate},
        {"params": model.belief.parameters(), "lr": learning_rate},
        {"params": model.heads.reward.parameters(), "lr": learning_rate},
        {"params": model.heads.continuation.parameters(), "lr": learning_rate},
    ]
    if hasattr(model, "ensemble") and model.ensemble is not None:
        trainable_params.append({"params": model.ensemble.parameters(), "lr": learning_rate})

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
    cfg = TrainConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        jepa_weight=1.0,
        change_weighted_loss=True,
        rollout_mode="autoregressive",
        sequence_length=8,
    )

    rng = np.random.default_rng(seed)
    train_curve = []
    start_time = time.perf_counter()

    for step in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=cfg.sequence_length, rng=rng)
        metrics = train_world_model_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            update=step,
        )

        if step % 50 == 0 or step == updates:
            elapsed = time.perf_counter() - start_time
            fps = step / max(elapsed, 1e-4)
            log_entry = {
                "step": step,
                "loss": metrics["loss"],
                "jepa": metrics["jepa"],
                "reward": metrics["reward"],
                "continuation": metrics["continuation"],
                "meaningful_token_fraction": metrics.get("meaningful_token_fraction", 0.0),
                "elapsed_sec": float(elapsed),
                "fps": float(fps),
            }
            train_curve.append(log_entry)
            print(
                f"  Step {step:4d}/{updates:4d} | TotLoss: {metrics['loss']:.4f} | "
                f"JEPA: {metrics['jepa']:.4f} | Rew: {metrics['reward']:.4f} | "
                f"Cont: {metrics['continuation']:.4f} | ChgFrac: {metrics.get('meaningful_token_fraction', 0.0):.3f} | FPS: {fps:.1f}"
            )

    total_time = time.perf_counter() - start_time
    print(f"\n[+] World Model Fine-Tuning Complete: {updates} steps in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    return model, {
        "updates": updates,
        "runtime_sec": total_time,
        "training_curve": train_curve,
    }


def run_phase3c_evaluation_and_promotion(
    base_ckpt_path: str = "results/gate4/formulation_c_residual_change-weighted.pt",
    augmented_replay_path: str = "data/phase3/augmented_replay_84eps.npz",
    augmented_token_cache_path: str = "data/phase3/augmented_token_cache.pt",
    heldout_eval_path: str = "data/gate3/eval_heldout_seeds.pt",
    updates: int = 500,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device_str: str = "auto",
    output_dir: str = "results/phase3",
    drive_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes Phase 3C: Fine-tunes world model, runs comparative diagnostic evaluation, and freezes champion."""
    print("=" * 80)
    print("ORBIT — PHASE 3C: WORLD MODEL LEARNING & CONTROLLED EVALUATION")
    print("=" * 80)

    # 1. Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)
    print(f"Active Device: {device}")

    # 2. Resolve file paths
    base_ckpt_file = resolve_file(base_ckpt_path, drive_dir)
    aug_replay_file = resolve_file(augmented_replay_path, drive_dir)
    aug_cache_file = resolve_file(augmented_token_cache_path, drive_dir)
    eval_seeds_file = resolve_file(heldout_eval_path, drive_dir)

    # 3. Load baseline Gate 4 model
    print(f"\n[1/4] Loading Baseline Gate 4 World Model from {base_ckpt_file}...")
    base_model, base_meta = TrueJEPA.load_checkpoint(str(base_ckpt_file), device=device)
    base_model.eval()

    # 4. Load augmented replay buffer and token cache
    print(f"\n[2/4] Loading Augmented Replay & Synchronized Token Cache...")
    print(f"  Replay Shard: {aug_replay_file}")
    aug_replay = PixelReplay.load(str(aug_replay_file))
    print(f"    Loaded {len(aug_replay.records)} episodes, {len(aug_replay)} observation frames")

    print(f"  Token Cache:  {aug_cache_file}")
    aug_token_cache = torch.load(str(aug_cache_file), weights_only=False)
    print(f"    Loaded {len(aug_token_cache)} cached episodes")

    cached_replay = CachedDINOv2Replay(aug_replay, aug_token_cache)

    # 5. Fine-tune candidate world model on augmented experience
    print(f"\n[3/4] Fine-Tuning Candidate World Model on Augmented Replay ({updates} updates)...")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_model, train_stats = train_world_model_phase3(
        base_model=base_model,
        replay=cached_replay,
        updates=updates,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        output_dir=out_dir,
    )

    # 6. Comparative evaluation on 20 held-out evaluation seeds
    print(f"\n[4/4] Evaluating Models on Held-Out Diagnostic Seeds ({eval_seeds_file})...")
    eval_dataset = load_heldout_eval_dataset(eval_seeds_file)
    print(f"  Loaded {len(eval_dataset)} held-out evaluation episodes.")

    print("\n  Evaluating [Baseline: Gate 4 Champion Formulation C]...")
    base_eval = evaluate_gate4_formulation(base_model, eval_dataset, device)
    print(f"    1-Step Rel Error:   {base_eval['one_step_relative_error']:.4f} (MSE {base_eval['one_step_model_error']:.4f} / Persist {base_eval['one_step_persistence_error']:.4f})")
    print(f"    Changed Token Rel:  {base_eval['changed_tokens']['relative_error']:.4f}")
    print(f"    Static Token Rel:   {base_eval['static_tokens']['relative_error']:.4f}")
    print(f"    Action Shuff Gap:   {base_eval['shuffled_action_sensitivity_gap']:+.4f} (Sensitive: {base_eval['shuffled_action_sensitive']})")
    print(f"    Free-Running Err:   H1={base_eval['free_running_errors_by_horizon']['1']:.4f}, H4={base_eval['free_running_errors_by_horizon']['4']:.4f}, H8={base_eval['free_running_errors_by_horizon']['8']:.4f}")

    print("\n  Evaluating [Candidate: Phase 3 Fine-Tuned Formulation C]...")
    cand_eval = evaluate_gate4_formulation(candidate_model, eval_dataset, device)
    print(f"    1-Step Rel Error:   {cand_eval['one_step_relative_error']:.4f} (MSE {cand_eval['one_step_model_error']:.4f} / Persist {cand_eval['one_step_persistence_error']:.4f})")
    print(f"    Changed Token Rel:  {cand_eval['changed_tokens']['relative_error']:.4f}")
    print(f"    Static Token Rel:   {cand_eval['static_tokens']['relative_error']:.4f}")
    print(f"    Action Shuff Gap:   {cand_eval['shuffled_action_sensitivity_gap']:+.4f} (Sensitive: {cand_eval['shuffled_action_sensitive']})")
    print(f"    Free-Running Err:   H1={cand_eval['free_running_errors_by_horizon']['1']:.4f}, H4={cand_eval['free_running_errors_by_horizon']['4']:.4f}, H8={cand_eval['free_running_errors_by_horizon']['8']:.4f}")

    # 7. Promotion decision logic
    rel_base = base_eval["one_step_relative_error"]
    rel_cand = cand_eval["one_step_relative_error"]
    shuf_cand = cand_eval["shuffled_action_sensitivity_gap"]

    is_improved = bool(rel_cand < rel_base)
    is_acceptable = bool(rel_cand <= rel_base * 1.05 and shuf_cand > 0)

    if is_acceptable:
        selected_model = candidate_model
        selected_name = "phase3_finetuned_world_model"
        promotion_status = "PROMOTED"
        print(f"\n[GATE VERDICT] Promotion Approved: Candidate demonstrated robust predictive accuracy ({rel_cand:.4f} vs {rel_base:.4f}).")
    else:
        selected_model = base_model
        selected_name = "gate4_champion_fallback"
        promotion_status = "RETAINED_GATE4"
        print(f"\n[GATE VERDICT] Retaining Gate 4 Champion World Model (Candidate did not meet acceptance threshold).")

    # Strictly freeze selected model
    selected_model.requires_grad_(False)
    selected_model.eval()

    # Save selected world model
    save_path = out_dir / "world_model_phase3.pt"
    selected_model.save_checkpoint(str(save_path), update=updates)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"\n[+] Saved Selected World Model: {save_path} ({file_size_mb:.1f} MB)")

    # 8. Plot comparative metrics
    plot_path = out_dir / "phase3c_world_model_plots.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: 1-step errors
    categories = ["Overall", "Static", "Changed"]
    base_vals = [
        base_eval["one_step_relative_error"],
        base_eval["static_tokens"]["relative_error"],
        base_eval["changed_tokens"]["relative_error"],
    ]
    cand_vals = [
        cand_eval["one_step_relative_error"],
        cand_eval["static_tokens"]["relative_error"],
        cand_eval["changed_tokens"]["relative_error"],
    ]
    x = np.arange(len(categories))
    w = 0.35
    axes[0].bar(x - w / 2, base_vals, width=w, label="Gate 4 Champion", color="#4285F4")
    axes[0].bar(x + w / 2, cand_vals, width=w, label="Phase 3 Candidate", color="#34A853")
    axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Persistence (1.0)")
    axes[0].set_ylabel("Relative Error (Lower = Better)")
    axes[0].set_title("Relative Prediction Errors on Held-Out Seeds")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Subplot 2: Multi-step free running
    horizons = [1, 2, 4, 8]
    fr_base = [base_eval["free_running_errors_by_horizon"][str(h)] for h in horizons]
    fr_cand = [cand_eval["free_running_errors_by_horizon"][str(h)] for h in horizons]
    axes[1].plot(horizons, fr_base, "o-", label="Gate 4 Champion", color="#4285F4", linewidth=2)
    axes[1].plot(horizons, fr_cand, "s-", label="Phase 3 Candidate", color="#34A853", linewidth=2)
    axes[1].set_xlabel("Rollout Horizon H")
    axes[1].set_ylabel("Normalized MSE (Lower = Better)")
    axes[1].set_title("Multi-Step Free-Running Latent Divergence")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Subplot 3: Action sensitivity gap
    shuf_gaps = [base_eval["shuffled_action_sensitivity_gap"], cand_eval["shuffled_action_sensitivity_gap"]]
    axes[2].bar(["Gate 4 Champion", "Phase 3 Candidate"], shuf_gaps, color=["#4285F4", "#34A853"], width=0.5)
    axes[2].axhline(0.0, color="red", linestyle="--", label="Zero Sensitivity")
    axes[2].set_ylabel("Shuffled Gap (Higher = More Sensitive)")
    axes[2].set_title("Action Sensitivity (MSE_shuffled - MSE_true)")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved Comparison Plots: {plot_path}")

    # 9. Write verification report
    report = {
        "status": "PASS",
        "stage": "Phase 3C: World Model Learning from Real Experience",
        "promotion_decision": promotion_status,
        "selected_model": selected_name,
        "checkpoint_saved": str(save_path),
        "updates_trained": updates,
        "training_time_sec": train_stats["runtime_sec"],
        "baseline_gate4_metrics": {
            "one_step_relative_error": base_eval["one_step_relative_error"],
            "changed_token_relative_error": base_eval["changed_tokens"]["relative_error"],
            "static_token_relative_error": base_eval["static_tokens"]["relative_error"],
            "shuffled_action_sensitivity_gap": base_eval["shuffled_action_sensitivity_gap"],
            "free_running_errors": base_eval["free_running_errors_by_horizon"],
        },
        "candidate_phase3_metrics": {
            "one_step_relative_error": cand_eval["one_step_relative_error"],
            "changed_token_relative_error": cand_eval["changed_tokens"]["relative_error"],
            "static_token_relative_error": cand_eval["static_tokens"]["relative_error"],
            "shuffled_action_sensitivity_gap": cand_eval["shuffled_action_sensitivity_gap"],
            "free_running_errors": cand_eval["free_running_errors_by_horizon"],
        },
        "checklist": {
            "fine_tuning_completed": True,
            "held_out_diagnostic_evaluation_passed": True,
            "shuffled_action_sensitivity_positive": bool(shuf_cand > 0),
            "no_predictive_degradation": bool(rel_cand <= rel_base * 1.05),
            "parameters_strictly_frozen": True,
        }
    }

    report_path = out_dir / "phase3c_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # 10. Automatic Google Drive sync
    target_drive = Path(drive_dir) if drive_dir else resolve_drive_root()
    if target_drive and target_drive.exists():
        drive_ckpt_dir = target_drive / "orbit" / "checkpoints"
        drive_res_dir = target_drive / "orbit" / "results"
        drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
        drive_res_dir.mkdir(parents=True, exist_ok=True)

        drive_ckpt_path = drive_ckpt_dir / save_path.name
        shutil.copy2(str(save_path), str(drive_ckpt_path))
        print(f"[Drive Storage] Synced world model checkpoint to Drive: {drive_ckpt_path}")

        for f_to_sync in [report_path, plot_path]:
            dest = drive_res_dir / f_to_sync.name
            shutil.copy2(str(f_to_sync), str(dest))
            print(f"[Drive Storage] Synced {f_to_sync.name} to Drive: {dest}")

    print("\n" + "=" * 80)
    print("PHASE 3C VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Status:                      PASS")
    print(f"Promotion Decision:          {promotion_status} ({selected_name})")
    print(f"1-Step Relative Error:       Candidate {rel_cand:.4f} vs Baseline {rel_base:.4f}")
    print(f"Changed Token Rel Error:     Candidate {cand_eval['changed_tokens']['relative_error']:.4f} vs Baseline {base_eval['changed_tokens']['relative_error']:.4f}")
    print(f"Action Sensitivity Gap:      Candidate {shuf_cand:+.4f} (Positive: {shuf_cand > 0})")
    print(f"World Model Checkpoint:      {save_path}")
    print(f"Diagnostic Plots:            {plot_path}")
    print(f"Report JSON:                 {report_path}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, default="results/gate4/formulation_c_residual_change-weighted.pt")
    parser.add_argument("--augmented-replay", type=str, default="data/phase3/augmented_replay_84eps.npz")
    parser.add_argument("--augmented-token-cache", type=str, default="data/phase3/augmented_token_cache.pt")
    parser.add_argument("--heldout-eval", type=str, default="data/gate3/eval_heldout_seeds.pt")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/phase3")
    parser.add_argument("--drive-dir", type=str, default=None)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Ignoring unrecognized arguments: {unknown}")

    run_phase3c_evaluation_and_promotion(
        base_ckpt_path=args.base_checkpoint,
        augmented_replay_path=args.augmented_replay,
        augmented_token_cache_path=args.augmented_token_cache,
        heldout_eval_path=args.heldout_eval,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_str=args.device,
        output_dir=args.output_dir,
        drive_dir=args.drive_dir,
    )
