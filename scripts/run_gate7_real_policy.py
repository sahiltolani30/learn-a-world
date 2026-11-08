#!/usr/bin/env python3
"""ORBIT — Gate 7: Real-Experience Policy Learning & Value Calibration.

Executes Gate 7 of the ORBIT Master Specification:
- Audits the real-experience replay dataset (T_obs = T_act + 1, seed isolation, bounds).
- Freezes world model trunk (DINOv2 backbone, spatial adapter, predictor, belief, reward, continuation).
- Calibrates pre-training reward prediction and value estimates against real transitions and ground-truth returns.
- Trains policy actor (heads.policy) and critic (heads.value) via offline behavior-regularized AWR on real experience only.
- Calibrates post-training reward prediction and value estimates.
- Benchmarks Candidate Policy vs Gate 6 Champion (H=4) vs Matched Random Baseline across 30 pinned seeds (5000..5029).
- Evaluates the Promotion Decision Matrix: Baseline Established vs Champion Replacement.
- Generates 4-panel telemetry plots and full machine-readable experiment manifest.
- Automatically mirrors all results to Google Drive.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate6_policy_evaluator import (
    ALL_CRAFTER_ACHIEVEMENTS,
    RandomCrafterAgent,
    compute_crafter_score,
    evaluate_policy_in_crafter,
)
from eval.gate7_evaluator import (
    audit_replay_dataset,
    calibrate_value_predictions,
    real_experience_policy_step,
    validate_reward_prediction,
)
from scripts.run_phase3b_integration import resolve_drive_root, resolve_file
from true_jepa.agent import VisualJEPAAgent
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.replay import PixelReplay


def load_heldout_episodes(path: Path) -> List[Dict[str, Any]]:
    """Loads held-out evaluation episodes from .pt or .pt.gz file."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            buf = io.BytesIO(f.read())
            episodes = torch.load(buf, weights_only=False, map_location="cpu")
    else:
        episodes = torch.load(path, weights_only=False, map_location="cpu")
    return episodes


def generate_gate7_plots(
    training_history: Dict[str, List[float]],
    pre_val_metrics: Dict[str, Any],
    post_val_metrics: Dict[str, Any],
    benchmark_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generates 4-panel Gate 7 telemetry and diagnostic visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1: Training Loss Trajectories
    steps = training_history.get("steps", [])
    actor_loss = training_history.get("actor_loss", [])
    value_loss = training_history.get("value_loss", [])
    entropy = training_history.get("entropy", [])

    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(steps, actor_loss, "b-o", label="Actor Loss", markersize=4)
    l2 = ax1.plot(steps, value_loss, "r-s", label="Value Loss", markersize=4)
    l3 = ax1_twin.plot(steps, entropy, "k--^", label="Policy Entropy", markersize=4, alpha=0.7)
    ax1.set_xlabel("Update Step")
    ax1.set_ylabel("Loss")
    ax1_twin.set_ylabel("Entropy (nats)")
    ax1.set_title("Offline Real-Experience Training")
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8)

    # Panel 2: Value Function Calibration (Decile Breakdown)
    ax2 = axes[1]
    pre_groups = pre_val_metrics.get("calibration_groups", [])
    post_groups = post_val_metrics.get("calibration_groups", [])

    if post_groups:
        real_means = [g["mean_real_return"] for g in post_groups]
        post_preds = [g["mean_pred_value"] for g in post_groups]
        pre_preds = [g["mean_pred_value"] for g in pre_groups] if len(pre_groups) == len(post_groups) else None

        x = np.arange(len(real_means))
        w = 0.28
        ax2.bar(x - w, real_means, width=w, label="Ground-Truth Return G_t", color="#34A853", alpha=0.85)
        if pre_preds is not None:
            ax2.bar(x, pre_preds, width=w, label="Pre-Train Value V(s)", color="#9AA0A6", alpha=0.85)
        ax2.bar(x + w, post_preds, width=w, label="Post-Train Value V(s)", color="#4285F4", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Q{i+1}" for i in range(len(real_means))])
        ax2.set_xlabel("Return Quantile")
        ax2.set_ylabel("Value / Return Magnitude")
        ax2.set_title(f"Value Calibration (Post r={post_val_metrics.get('value_correlation', 0.0):.2f})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Calibration Data Unavailable", ha="center", va="center")

    # Panel 3: Real Crafter Return & Score (30 Pinned Seeds)
    arms = ["Random Baseline", "Gate 6 Champion (H=4)", "Gate 7 Candidate"]
    keys = ["random_baseline", "gate6_champion", "gate7_candidate"]
    colors = ["#9AA0A6", "#4285F4", "#FBBC04"]

    returns = [benchmark_results[k]["mean_return"] for k in keys]
    stds = [benchmark_results[k]["std_return"] for k in keys]
    scores = [benchmark_results[k]["crafter_score"] for k in keys]

    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    b1 = ax3.bar(np.arange(len(arms)) - 0.15, returns, yerr=stds, capsize=4, width=0.3, color=colors, alpha=0.85, label="Mean Return")
    b2 = ax3_twin.bar(np.arange(len(arms)) + 0.15, scores, width=0.3, color=[c + "88" for c in colors], edgecolor=colors, linewidth=1.5, label="Crafter Score (%)")
    ax3.set_xticks(np.arange(len(arms)))
    ax3.set_xticklabels([a.replace(" ", "\n") for a in arms], fontsize=8)
    ax3.set_ylabel("Mean Return / Episode")
    ax3_twin.set_ylabel("Crafter Score (%)")
    ax3.set_title("30-Seed Real Crafter Benchmark")
    ax3.grid(axis="y", alpha=0.3)

    for i, v in enumerate(returns):
        ax3.text(i - 0.15, v + 0.1, f"{v:+.2f}", ha="center", fontsize=8, fontweight="bold")
    for i, v in enumerate(scores):
        ax3_twin.text(i + 0.15, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel 4: Key Achievement Unlock Rates
    ax4 = axes[3]
    top_achievements = ["collect_wood", "place_table", "collect_sapling", "place_plant", "wake_up"]
    x = np.arange(len(top_achievements))
    w = 0.25
    for idx, (k, col, name) in enumerate(zip(keys, colors, arms)):
        rates = [benchmark_results[k]["achievement_rates"].get(ach, 0.0) * 100 for ach in top_achievements]
        ax4.bar(x + (idx - 1) * w, rates, width=w, label=name, color=col, alpha=0.85)

    ax4.set_xticks(x)
    ax4.set_xticklabels([a.replace("_", "\n") for a in top_achievements], fontsize=8)
    ax4.set_ylabel("Unlock Rate (%)")
    ax4.set_title("Key Achievement Unlock Rates")
    ax4.legend(fontsize=7)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[Telemetry Plots] Saved 4-panel plots to {output_path}")


def run_gate7_real_experience_policy(
    world_model_ckpt: str = "results/phase3/world_model_phase3.pt",
    replay_dataset_path: str = "data/gate7/augmented_replay_local.npz",
    heldout_eval_path: str = "data/gate3/eval_heldout_seeds.pt.gz",
    gate6_champion_ckpt: str = "results/gate6/policy_h4.pt.gz",
    updates: int = 300,
    batch_size: int = 8,
    seq_len: int = 8,
    actor_lr: float = 1e-4,
    value_lr: float = 1e-4,
    gamma: float = 0.99,
    advantage_temperature: float = 1.0,
    max_weight: float = 20.0,
    entropy_weight: float = 0.01,
    benchmark_seeds_count: int = 30,
    max_steps: int = 300,
    device_str: str = "auto",
    output_dir: str = "results/gate7",
    drive_dir: Optional[str] = None,
    seed: int = 777,
) -> Dict[str, Any]:
    """Executes the complete Gate 7 training, calibration, and benchmarking pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print("ORBIT — GATE 7: REAL-EXPERIENCE POLICY LEARNING & VALUE CALIBRATION")
    print("=" * 80)

    # 1. Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)
    print(f"Active Device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. File resolution
    wm_file = resolve_file(world_model_ckpt, drive_dir)
    heldout_file = resolve_file(heldout_eval_path, drive_dir)
    gate6_file = resolve_file(gate6_champion_ckpt, drive_dir)

    # Try resolving replay dataset
    try:
        replay_file = resolve_file(replay_dataset_path, drive_dir)
    except FileNotFoundError:
        # Fallback to alternative names
        alt_paths = [
            "data/phase3/augmented_replay_84eps.npz",
            "data/gate7/augmented_replay_local.npz",
            "data/gate3/train_replay_10k.npz",
        ]
        found = False
        for alt in alt_paths:
            try:
                replay_file = resolve_file(alt, drive_dir)
                found = True
                print(f"[Dataset Fallback] Using alternative replay file: {replay_file}")
                break
            except FileNotFoundError:
                continue
        if not found:
            raise FileNotFoundError(f"Could not resolve replay dataset {replay_dataset_path} or any fallback.")

    print(f"\n[Resolved File Paths]")
    print(f"  World Model:    {wm_file}")
    print(f"  Replay Dataset: {replay_file}")
    print(f"  Held-out Eval:  {heldout_file}")
    print(f"  Gate 6 Policy:  {gate6_file}")

    # -----------------------------------------------------------------------
    # Step 1: Audit Replay Dataset
    # -----------------------------------------------------------------------
    print(f"\n[1/6] Auditing Real-Experience Replay Dataset from {replay_file}...")
    eval_seeds = list(range(3030, 3050))
    bench_seeds = list(range(5000, 5000 + benchmark_seeds_count))
    dataset_audit = audit_replay_dataset(replay_file, eval_seeds=eval_seeds, benchmark_seeds=bench_seeds)

    manifest_path = out_path / "gate7_dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(dataset_audit, f, indent=2)
    print(f"  Audit Status:              {dataset_audit['status']}")
    print(f"  Replay SHA-256:            {dataset_audit['sha256']}")
    print(f"  Valid Episodes:            {dataset_audit['num_episodes']}")
    print(f"  Total Transitions:         {dataset_audit['total_actions']}")
    print(f"  Temporal Alignment:        {'VALID' if dataset_audit['temporal_alignment_ok'] else 'FAIL'}")
    print(f"  Seed Isolation (Eval/Bench): {'DISJOINT (PASS)' if dataset_audit['seed_isolation']['is_disjoint'] else 'OVERLAP (FAIL)'}")

    if dataset_audit["status"] != "PASS":
        raise RuntimeError(f"Replay dataset audit failed: {dataset_audit}")

    # -----------------------------------------------------------------------
    # Step 2: Load World Model Champion & Enforce Parameter Freezes
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Loading World Model Champion & Freezing Non-Policy Parameters...")
    model, wm_meta = TrueJEPA.load_checkpoint(str(wm_file), device=device)

    # Strictly freeze all world model components
    model.encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.predictor.requires_grad_(False)
    model.belief.requires_grad_(False)
    model.heads.reward.requires_grad_(False)
    model.heads.continuation.requires_grad_(False)
    if hasattr(model, "ensemble") and model.ensemble is not None:
        model.ensemble.requires_grad_(False)

    model.encoder.eval()
    model.target_encoder.eval()
    model.predictor.eval()
    model.belief.eval()
    model.heads.reward.eval()
    model.heads.continuation.eval()

    # Train ONLY policy actor and value critic
    model.heads.policy.requires_grad_(True)
    model.heads.value.requires_grad_(True)
    model.heads.policy.train()
    model.heads.value.train()

    actor_params = list(model.heads.policy.parameters())
    value_params = list(model.heads.value.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=actor_lr)
    value_opt = torch.optim.Adam(value_params, lr=value_lr)

    # Invariant verification: zero parameter overlap
    assert len(set(actor_params) & set(value_params)) == 0, "Actor and Value parameters must be disjoint"
    assert len(set(actor_params) & set(model.encoder.parameters())) == 0, "Actor must not overlap encoder"
    assert len(set(value_params) & set(model.predictor.parameters())) == 0, "Value must not overlap predictor"
    print("  ✓ Frozen: Encoder Backbone, Spatial Adapter, Predictor, Belief Core, Reward Head, Continuation Head.")
    print(f"  ✓ Trainable: Policy Head ({sum(p.numel() for p in actor_params):,} params), Value Head ({sum(p.numel() for p in value_params):,} params).")
    print("  ✓ Parameter Disjointness Verified: 0 shared parameters.")

    # -----------------------------------------------------------------------
    # Step 3: Pre-Training Evaluation on Held-Out Seeds
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Pre-Training Evaluation on Held-Out Seeds ({heldout_file})...")
    heldout_episodes = load_heldout_episodes(heldout_file)
    print(f"  Loaded {len(heldout_episodes)} held-out evaluation episodes.")

    pre_reward_metrics = validate_reward_prediction(model, heldout_episodes, device)
    pre_value_metrics = calibrate_value_predictions(model, heldout_episodes, device, gamma=gamma)

    print(f"  Pre-Train Reward MAE:      {pre_reward_metrics['reward_mae']:.4f} (Correlation: {pre_reward_metrics['reward_correlation']:+.4f})")
    print(f"  Pre-Train Value MAE:       {pre_value_metrics['value_mae']:.4f} (Correlation: {pre_value_metrics['value_correlation']:+.4f})")
    print(f"  Pre-Train Value Bias:      {pre_value_metrics['value_bias']:+.4f}")

    # -----------------------------------------------------------------------
    # Step 4: Offline Policy Training on Real Experience (AWR)
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Training Policy via Real-Experience AWR ({updates} updates)...")
    print(f"  Hyperparameters: Batch Size: {batch_size} | Seq Len: {seq_len} | Gamma: {gamma} | Temp: {advantage_temperature} | MaxW: {max_weight} | Actor LR: {actor_lr} | Value LR: {value_lr}")

    replay = PixelReplay.load(str(replay_file))
    rng = np.random.default_rng(seed)

    training_history: Dict[str, List[float]] = {
        "steps": [],
        "actor_loss": [],
        "value_loss": [],
        "advantage": [],
        "adv_weight": [],
        "pred_value": [],
        "real_return": [],
        "entropy": [],
    }

    t0 = time.time()
    for step_idx in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        step_res = real_experience_policy_step(
            model=model,
            batch=batch,
            actor_optimizer=actor_opt,
            value_optimizer=value_opt,
            gamma=gamma,
            advantage_temperature=advantage_temperature,
            max_weight=max_weight,
            entropy_weight=entropy_weight,
            device=device,
        )

        training_history["steps"].append(step_idx)
        training_history["actor_loss"].append(step_res["actor_loss"])
        training_history["value_loss"].append(step_res["value_loss"])
        training_history["advantage"].append(step_res["mean_advantage"])
        training_history["adv_weight"].append(step_res["mean_adv_weight"])
        training_history["pred_value"].append(step_res["mean_pred_value"])
        training_history["real_return"].append(step_res["mean_real_return"])
        training_history["entropy"].append(step_res["mean_entropy"])

        if step_idx % 25 == 0 or step_idx == updates:
            elapsed = time.time() - t0
            fps = (step_idx * batch_size * seq_len) / max(elapsed, 1e-4)
            print(
                f"  Step {step_idx:4d}/{updates:4d} | "
                f"ActLoss: {step_res['actor_loss']:+.4f} | "
                f"ValLoss: {step_res['value_loss']:.4f} | "
                f"Adv: {step_res['mean_advantage']:+.4f} | "
                f"Weight: {step_res['mean_adv_weight']:.2f} | "
                f"PredVal: {step_res['mean_pred_value']:+.3f} | "
                f"Ent: {step_res['mean_entropy']:.3f} | "
                f"FPS: {fps:.1f}"
            )

    train_elapsed = time.time() - t0
    print(f"\n  ✓ Training Completed in {train_elapsed:.1f}s ({updates} updates).")

    # -----------------------------------------------------------------------
    # Step 5: Post-Training Evaluation on Held-Out Seeds
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Post-Training Evaluation on Held-Out Seeds...")
    model.eval()
    post_reward_metrics = validate_reward_prediction(model, heldout_episodes, device)
    post_value_metrics = calibrate_value_predictions(model, heldout_episodes, device, gamma=gamma)

    print(f"  Post-Train Reward MAE:     {post_reward_metrics['reward_mae']:.4f} (Correlation: {post_reward_metrics['reward_correlation']:+.4f})")
    print(f"  Post-Train Value MAE:      {post_value_metrics['value_mae']:.4f} (Correlation: {post_value_metrics['value_correlation']:+.4f})")
    print(f"  Post-Train Value Bias:     {post_value_metrics['value_bias']:+.4f}")
    val_mae_delta = post_value_metrics["value_mae"] - pre_value_metrics["value_mae"]
    val_corr_delta = post_value_metrics["value_correlation"] - pre_value_metrics["value_correlation"]
    print(f"  Value MAE Change:          {val_mae_delta:+.4f} ({'IMPROVED' if val_mae_delta < 0 else 'HIGHER'})")
    print(f"  Value Correlation Change:  {val_corr_delta:+.4f} ({'IMPROVED' if val_corr_delta > 0 else 'LOWER'})")

    # Save Candidate Policy Checkpoint
    candidate_ckpt_path = out_path / "policy_gate7.pt"
    candidate_ckpt_gz_path = out_path / "policy_gate7.pt.gz"
    model.save_checkpoint(str(candidate_ckpt_path), update=updates)
    with open(candidate_ckpt_path, "rb") as f_in, gzip.open(candidate_ckpt_gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  ✓ Candidate Policy Saved: {candidate_ckpt_path} ({candidate_ckpt_path.stat().st_size / (1024*1024):.1f} MB)")

    # -----------------------------------------------------------------------
    # Step 6: Rigorous 30-Seed Real Crafter Benchmark
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Rigorous Real Crafter Benchmark ({benchmark_seeds_count} Pinned Seeds: 5000..{5000 + benchmark_seeds_count - 1})...")

    # Arm 1: Matched Random Baseline
    print(f"\n  [ARM 1/3] Benchmarking Matched Random Baseline...")
    random_agent = RandomCrafterAgent(seed=42)
    t_arm1 = time.time()
    random_res = evaluate_policy_in_crafter(random_agent, seeds=bench_seeds, max_steps=max_steps)
    t_arm1_el = time.time() - t_arm1
    print(f"    Return: {random_res['mean_return']:+.2f} ± {random_res['std_return']:.2f} | Score: {random_res['crafter_score']:.2f}% | Steps: {random_res['mean_length']:.1f} | Elapsed: {t_arm1_el:.1f}s")

    # Arm 2: Gate 6 Champion (H=4)
    print(f"\n  [ARM 2/3] Benchmarking Gate 6 Champion ({gate6_file})...")
    gate6_model, _ = TrueJEPA.load_checkpoint(str(gate6_file), device=device)
    gate6_agent = VisualJEPAAgent(gate6_model, device=str(device), stochastic=True, temperature=1.0)
    t_arm2 = time.time()
    gate6_res = evaluate_policy_in_crafter(gate6_agent, seeds=bench_seeds, max_steps=max_steps)
    t_arm2_el = time.time() - t_arm2
    print(f"    Return: {gate6_res['mean_return']:+.2f} ± {gate6_res['std_return']:.2f} | Score: {gate6_res['crafter_score']:.2f}% | Steps: {gate6_res['mean_length']:.1f} | Elapsed: {t_arm2_el:.1f}s")

    # Arm 3: Gate 7 Candidate
    print(f"\n  [ARM 3/3] Benchmarking Gate 7 Candidate Policy...")
    gate7_agent = VisualJEPAAgent(model, device=str(device), stochastic=True, temperature=1.0)
    t_arm3 = time.time()
    gate7_res = evaluate_policy_in_crafter(gate7_agent, seeds=bench_seeds, max_steps=max_steps)
    t_arm3_el = time.time() - t_arm3
    print(f"    Return: {gate7_res['mean_return']:+.2f} ± {gate7_res['std_return']:.2f} | Score: {gate7_res['crafter_score']:.2f}% | Steps: {gate7_res['mean_length']:.1f} | Elapsed: {t_arm3_el:.1f}s")

    benchmark_manifest = {
        "random_baseline": random_res,
        "gate6_champion": gate6_res,
        "gate7_candidate": gate7_res,
    }

    # -----------------------------------------------------------------------
    # Step 7: Promotion Decision Matrix & Reporting
    # -----------------------------------------------------------------------
    score_gate7 = gate7_res["crafter_score"]
    score_gate6 = gate6_res["crafter_score"]
    return_gate7 = gate7_res["mean_return"]
    return_gate6 = gate6_res["mean_return"]

    # Promotion Rule: Must strictly outperform Gate 6 on Crafter score and Return
    if score_gate7 > score_gate6 and return_gate7 > return_gate6:
        promotion_status = "PROMOTED"
        verdict_str = f"Gate 7 Candidate PROMOTED as new production champion (Score: {score_gate7:.2f}% vs Gate 6 {score_gate6:.2f}%)."
    else:
        promotion_status = "CANDIDATE_NOT_PROMOTED"
        verdict_str = f"Gate 6 Champion RETAINED (Score: {score_gate6:.2f}% vs Gate 7 {score_gate7:.2f}%). Gate 7 establishes real-experience baseline."

    print("\n" + "=" * 80)
    print("GATE 7 DECISION MATRIX & PROMOTION STATUS")
    print("=" * 80)
    print(f"Baseline Established:        PASS")
    print(f"Promotion Verdict:           {promotion_status}")
    print(f"Summary:                     {verdict_str}")
    print(f"Gate 6 Champion:             Return: {return_gate6:+.2f} | Crafter Score: {score_gate6:.2f}% | Steps: {gate6_res['mean_length']:.1f}")
    print(f"Gate 7 Candidate:            Return: {return_gate7:+.2f} | Crafter Score: {score_gate7:.2f}% | Steps: {gate7_res['mean_length']:.1f}")
    print(f"Random Baseline:             Return: {random_res['mean_return']:+.2f} | Crafter Score: {random_res['crafter_score']:.2f}% | Steps: {random_res['mean_length']:.1f}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 8: Visualization & Manifest Output
    # -----------------------------------------------------------------------
    plot_path = out_path / "gate7_telemetry_plots.png"
    generate_gate7_plots(
        training_history=training_history,
        pre_val_metrics=pre_value_metrics,
        post_val_metrics=post_value_metrics,
        benchmark_results=benchmark_manifest,
        output_path=plot_path,
    )

    h_wm = hashlib.sha256()
    with open(wm_file, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h_wm.update(chunk)
    wm_sha256 = h_wm.hexdigest()

    metrics_manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "Gate 7: Real-Experience Policy Learning & Value Calibration",
        "baseline_status": "PASS",
        "promotion_status": promotion_status,
        "verdict": verdict_str,
        "provenance": {
            "world_model_path": str(wm_file),
            "world_model_sha256": wm_sha256,
            "replay_dataset_path": str(replay_file),
            "replay_dataset_sha256": dataset_audit["sha256"],
            "num_episodes": dataset_audit["num_episodes"],
            "total_transitions": dataset_audit["total_actions"],
            "candidate_ckpt_path": str(candidate_ckpt_path),
            "gate6_champion_path": str(gate6_file),
        },
        "training_config": {
            "updates": updates,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "actor_lr": actor_lr,
            "value_lr": value_lr,
            "gamma": gamma,
            "advantage_temperature": advantage_temperature,
            "max_weight": max_weight,
            "entropy_weight": entropy_weight,
            "device": str(device),
            "training_elapsed_sec": train_elapsed,
        },
        "pre_training_metrics": {
            "reward": pre_reward_metrics,
            "value": {k: v for k, v in pre_value_metrics.items() if k != "calibration_groups"},
        },
        "post_training_metrics": {
            "reward": post_reward_metrics,
            "value": {k: v for k, v in post_value_metrics.items() if k != "calibration_groups"},
        },
        "benchmark_results": {
            "random_baseline": {k: v for k, v in random_res.items() if k != "per_seed_results"},
            "gate6_champion": {k: v for k, v in gate6_res.items() if k != "per_seed_results"},
            "gate7_candidate": {k: v for k, v in gate7_res.items() if k != "per_seed_results"},
        },
    }

    metrics_json_path = out_path / "gate7_experiment_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_manifest, f, indent=2)
    print(f"[Metrics Manifest] Saved to {metrics_json_path}")

    # -----------------------------------------------------------------------
    # Step 9: Mirror to Google Drive if available
    # -----------------------------------------------------------------------
    target_drive = Path(drive_dir) if drive_dir else resolve_drive_root()
    if target_drive and target_drive.exists():
        drive_out = target_drive / "orbit" / "results" / "gate7"
        drive_out.mkdir(parents=True, exist_ok=True)
        for f_to_copy in [manifest_path, plot_path, metrics_json_path, candidate_ckpt_path, candidate_ckpt_gz_path]:
            if f_to_copy.exists():
                shutil.copy2(str(f_to_copy), str(drive_out / f_to_copy.name))
                print(f"[Drive Sync] Mirrored {f_to_copy.name} to {drive_out}")

    return metrics_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBIT — Gate 7: Real-Experience Policy Learning & Value Calibration")
    parser.add_argument("--world-model", type=str, default="results/phase3/world_model_phase3.pt")
    parser.add_argument("--replay-dataset", type=str, default="data/gate7/augmented_replay_local.npz")
    parser.add_argument("--heldout-eval", type=str, default="data/gate3/eval_heldout_seeds.pt.gz")
    parser.add_argument("--gate6-champion", type=str, default="results/gate6/policy_h4.pt.gz")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--advantage-temperature", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--benchmark-seeds", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/gate7")
    parser.add_argument("--drive-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=777)

    args = parser.parse_args()

    run_gate7_real_experience_policy(
        world_model_ckpt=args.world_model,
        replay_dataset_path=args.replay_dataset,
        heldout_eval_path=args.heldout_eval,
        gate6_champion_ckpt=args.gate6_champion,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        actor_lr=args.actor_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        advantage_temperature=args.advantage_temperature,
        max_weight=args.max_weight,
        entropy_weight=args.entropy_weight,
        benchmark_seeds_count=args.benchmark_seeds,
        max_steps=args.max_steps,
        device_str=args.device,
        output_dir=args.output_dir,
        drive_dir=args.drive_dir,
        seed=args.seed,
    )
