"""ORBIT Gate 5 Experiment: Temporal Memory and Multi-Step World Learning.

Compares:
1. Multi-step Training Regimes:
   - Regime 1: One-Step-Only Training (H=1 teacher-forced)
   - Regime 2: Multi-Step Autoregressive Training (unrolled BPTT)
   - Regime 3: Multi-Step Mixed Scheduled Sampling (50% real / 50% predicted)
2. Temporal Memory Ablation:
   - Condition A: No Memory / Markovian (hidden = 0)
   - Condition B: Recurrent Memory (recent history L=16)
   - Condition C: Long-Sequence Memory (extended history L=48+)
3. Viewport Exit and Return Analysis:
   - Object permanence and spatial consistency when reversing actions or re-entering regions.
4. Multi-Step Rollout Horizons:
   - Horizons H in {1, 2, 4, 8, 16} under Teacher-Forced vs Free-Running vs Persistence.
5. Policy & Value Head Groundwork:
   - Reward calibration, continuation probability, value dispersion, policy entropy.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate4_evaluator import decompose_static_and_changed_errors
from eval.gate5_evaluator import (
    evaluate_gate5_multi_step,
    evaluate_policy_value_calibration,
    evaluate_temporal_memory_ablation,
    evaluate_viewport_return_probe,
)
from experiments.run_gate4_experiment import CachedDINOv2Replay
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import train_world_model_step
from true_jepa.replay import PixelReplay


def train_regime(
    regime_id: str,
    name: str,
    rollout_mode: str,
    mixed_prob: float,
    replay: CachedDINOv2Replay,
    eval_dataset: List[Dict[str, Any]],
    updates: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    print(f"\n{'='*70}\n[GATE 5] Training: Regime {regime_id} — {name}\n{'='*70}")
    print(f"Rollout Mode: {rollout_mode} | Mixed Prob: {mixed_prob} | Updates: {updates}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = JEPAConfig(predictor_residual=True)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="dinov2").to(device)

    train_cfg = TrainConfig(
        batch_size=batch_size,
        sequence_length=seq_len,
        learning_rate=3e-4,
        weight_decay=1e-6,
        change_weighted_loss=True,
        change_weight_cap=5.0,
        change_weight_bonus=2.0,
        rollout_mode=rollout_mode,
        mixed_rollout_prob=mixed_prob,
        amp=False,
    )

    world_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(world_params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    rng = np.random.default_rng(seed=seed)
    training_curve = []
    start_time = time.perf_counter()

    for step in range(1, updates + 1):
        model.train()
        batch = replay.sample_with_prefix(train_cfg.batch_size, train_cfg.sequence_length, rng)
        metrics = train_world_model_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            cfg=train_cfg,
            update=step,
        )

        if step % 50 == 0 or step == updates:
            elapsed = time.perf_counter() - start_time
            fps = step / max(elapsed, 1e-4)
            log_entry = {
                "step": step,
                "jepa_loss": float(metrics["jepa"]),
                "reward_loss": float(metrics["reward"]),
                "continue_loss": float(metrics["continuation"]),
                "total_loss": float(metrics["loss"]),
                "elapsed_sec": float(elapsed),
                "fps": float(fps),
            }
            if "meaningful_token_fraction" in metrics:
                log_entry["meaningful_token_fraction"] = float(metrics["meaningful_token_fraction"])
            training_curve.append(log_entry)
            print(f"Step {step:4d}/{updates:4d} | Total: {metrics['loss']:.4f} | JEPA: {metrics['jepa']:.4f} | FPS: {fps:.1f}")

    total_time = time.perf_counter() - start_time
    print(f"Training completed in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    # Save checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"regime_{regime_id.lower()}_{name.lower().replace(' ', '_')}.pt"
    model.save_checkpoint(str(ckpt_path), update=updates)
    print(f"Saved checkpoint to {ckpt_path}")

    # Evaluate Gate 5 metrics
    print(f"Evaluating Gate 5 multi-step, memory, and viewport metrics...")
    eval_metrics = evaluate_gate5_multi_step(model, eval_dataset, device, horizons=[1, 2, 4, 8, 16])
    mem_metrics = evaluate_temporal_memory_ablation(model, eval_dataset, device)
    vp_metrics = evaluate_viewport_return_probe(model, eval_dataset, device)
    pol_metrics = evaluate_policy_value_calibration(model, eval_dataset, device)

    eval_metrics["temporal_memory_ablation"] = mem_metrics
    eval_metrics["viewport_return_probe"] = vp_metrics
    eval_metrics["policy_value_calibration"] = pol_metrics
    eval_metrics["regime_id"] = regime_id
    eval_metrics["regime_name"] = name
    eval_metrics["rollout_mode"] = rollout_mode
    eval_metrics["training_telemetry"] = {
        "total_runtime_sec": total_time,
        "average_throughput_fps": updates / total_time,
        "final_jepa_loss": training_curve[-1]["jepa_loss"],
        "final_total_loss": training_curve[-1]["total_loss"],
    }
    eval_metrics["training_curve"] = training_curve

    return model, eval_metrics


def generate_gate5_plots(metrics: Dict[str, Any], output_path: Path) -> None:
    """Generates 4-panel comparison plots for Gate 5."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    horizons = [1, 2, 4, 8, 16]

    # Panel 1: Multi-Step Horizons Free-Running Error vs Persistence
    # Use Formulation C (promoted) and regimes
    colors = {"regime_1": "tab:purple", "regime_2": "tab:green", "regime_3": "tab:orange", "gate4_formulation_c": "tab:blue"}
    labels = {
        "regime_1": "Regime 1: One-Step Trained",
        "regime_2": "Regime 2: Autoregressive Trained",
        "regime_3": "Regime 3: Mixed Scheduled Sampling",
        "gate4_formulation_c": "Gate 4 Promoted (Formulation C, 1k up)",
    }

    # Plot persistence baseline
    ref_key = "gate4_formulation_c" if "gate4_formulation_c" in metrics else list(metrics.keys())[0]
    p_errs = [metrics[ref_key]["persistence_errors_by_horizon"][str(h)] for h in horizons]
    axs[0, 0].plot(horizons, p_errs, "k--", lw=2, label="Persistence Baseline")

    for k in metrics.keys():
        if "free_running_errors_by_horizon" in metrics[k]:
            fr_vals = [metrics[k]["free_running_errors_by_horizon"][str(h)] for h in horizons]
            axs[0, 0].plot(horizons, fr_vals, "o-", label=labels.get(k, k), color=colors.get(k, "tab:cyan"), lw=2)

    axs[0, 0].set_title("Free-Running Autoregressive Compounding (H=1..16)")
    axs[0, 0].set_xlabel("Horizon (Steps)")
    axs[0, 0].set_ylabel("Normalized Latent MSE")
    axs[0, 0].set_xticks(horizons)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # Panel 2: Relative Error Against Persistence by Horizon
    for k in metrics.keys():
        if "free_running_rel_error_by_horizon" in metrics[k]:
            rel_vals = [metrics[k]["free_running_rel_error_by_horizon"][str(h)] for h in horizons]
            axs[0, 1].plot(horizons, rel_vals, "s-", label=labels.get(k, k), color=colors.get(k, "tab:cyan"), lw=2)

    axs[0, 1].axhline(1.0, color="red", linestyle="--", lw=1.5, label="Persistence Threshold (1.0)")
    axs[0, 1].set_title("Relative Free-Running Error vs Persistence (H=1..16)")
    axs[0, 1].set_xlabel("Horizon (Steps)")
    axs[0, 1].set_ylabel("Relative Error (Model / Persistence)")
    axs[0, 1].set_xticks(horizons)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # Panel 3: Temporal Memory Ablation Comparison (No Memory vs Recurrent vs Long)
    conds = ["Condition A\n(No Memory)", "Condition B\n(Recurrent L=16)", "Condition C\n(Long-Seq L=48+)"]
    x = np.arange(len(conds))
    width = 0.25

    regime_keys = [k for k in ["regime_1", "regime_2", "regime_3", "gate4_formulation_c"] if k in metrics]
    for idx, k in enumerate(regime_keys):
        mem = metrics[k]["temporal_memory_ablation"]
        vals = [mem["condition_a_no_memory_mse"], mem["condition_b_standard_memory_mse"], mem["condition_c_long_memory_mse"]]
        axs[1, 0].bar(x + (idx - len(regime_keys) / 2 + 0.5) * width, vals, width, label=labels.get(k, k), color=colors.get(k, "tab:cyan"), alpha=0.85)

    axs[1, 0].set_title("Temporal Memory Ablation: Prediction Error")
    axs[1, 0].set_ylabel("Normalized Latent MSE")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(conds)
    axs[1, 0].grid(True, alpha=0.3, axis="y")
    axs[1, 0].legend()

    # Panel 4: Viewport Exit and Return Analysis
    vp_names = []
    vp_advs = []
    for k in regime_keys:
        vp = metrics[k]["viewport_return_probe"]
        vp_names.append(labels.get(k, k).split(":")[0])
        vp_advs.append(vp["return_event_memory_advantage_pct"])

    bar_colors = ["tab:green" if adv > 0 else "tab:red" for adv in vp_advs]
    axs[1, 1].bar(vp_names, vp_advs, color=bar_colors, alpha=0.85, width=0.5)
    axs[1, 1].axhline(0.0, color="black", linestyle="-", lw=1.0)
    axs[1, 1].set_title("Viewport Re-entry Probe: Memory Advantage (%)")
    axs[1, 1].set_ylabel("Memory Advantage (% Reduction in Return MSE)")
    axs[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generated Gate 5 comparison plots at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="results/gate5")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    print(f"Selected compute device: {device}")

    # Load canonical replay dataset and cached patch tokens
    print("Loading canonical replay dataset and cached patch tokens...")
    raw_replay = PixelReplay.load("data/gate3/train_replay_10k.npz")
    token_cache = torch.load("data/gate3/dinov2_token_cache.pt", weights_only=False)
    cached_replay = CachedDINOv2Replay(raw_replay, token_cache)

    # Load held-out evaluation dataset
    eval_dataset = torch.load("data/gate3/eval_heldout_seeds.pt", weights_only=False)
    print(f"Loaded held-out evaluation dataset: {len(eval_dataset)} episodes across seeds 3030..3049")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Any] = {}

    # 1. Comprehensive Gate 5 evaluation of promoted Gate 4 Formulation C champion (1,000 updates)
    gate4_c_path = Path("results/gate4/formulation_c_residual_change-weighted.pt")
    if gate4_c_path.exists():
        print("\nEvaluating promoted Gate 4 Formulation C (1,000 updates) on full Gate 5 protocol...")
        model_g4c, _ = TrueJEPA.load_checkpoint(str(gate4_c_path), device=device)
        eval_g4c = evaluate_gate5_multi_step(model_g4c, eval_dataset, device, horizons=[1, 2, 4, 8, 16])
        eval_g4c["temporal_memory_ablation"] = evaluate_temporal_memory_ablation(model_g4c, eval_dataset, device)
        eval_g4c["viewport_return_probe"] = evaluate_viewport_return_probe(model_g4c, eval_dataset, device)
        eval_g4c["policy_value_calibration"] = evaluate_policy_value_calibration(model_g4c, eval_dataset, device)
        eval_g4c["model_source"] = "Gate 4 Promoted Champion (1,000 updates)"
        all_metrics["gate4_formulation_c"] = eval_g4c

    # 2. Train and evaluate Regime 1: One-Step-Only Training (H=1)
    _, eval_r1 = train_regime(
        regime_id="1",
        name="One-Step Training",
        rollout_mode="one_step",
        mixed_prob=0.0,
        replay=cached_replay,
        eval_dataset=eval_dataset,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.sequence_length,
        device=device,
        output_dir=output_dir,
    )
    all_metrics["regime_1"] = eval_r1

    # 3. Train and evaluate Regime 2: Multi-Step Autoregressive Training
    _, eval_r2 = train_regime(
        regime_id="2",
        name="Autoregressive Training",
        rollout_mode="autoregressive",
        mixed_prob=0.0,
        replay=cached_replay,
        eval_dataset=eval_dataset,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.sequence_length,
        device=device,
        output_dir=output_dir,
    )
    all_metrics["regime_2"] = eval_r2

    # 4. Train and evaluate Regime 3: Multi-Step Mixed Scheduled Sampling
    _, eval_r3 = train_regime(
        regime_id="3",
        name="Mixed Scheduled Sampling",
        rollout_mode="mixed",
        mixed_prob=0.5,
        replay=cached_replay,
        eval_dataset=eval_dataset,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.sequence_length,
        device=device,
        output_dir=output_dir,
    )
    all_metrics["regime_3"] = eval_r3

    # Save metrics JSON
    metrics_path = output_dir / "gate5_experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved Gate 5 experiment metrics to {metrics_path}")

    # Generate comparison plots
    plot_path = output_dir / "gate5_comparison_plots.png"
    generate_gate5_plots(all_metrics, plot_path)

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Regime / Model':<32} | {'RelErr H=1':<10} | {'RelErr H=8':<10} | {'RelErr H=16':<11} | {'Mem Adv %':<10} | {'VP Adv %':<10}")
    print("-" * 105)
    for k in ["gate4_formulation_c", "regime_1", "regime_2", "regime_3"]:
        if k in all_metrics:
            m = all_metrics[k]
            name = m.get("regime_name", m.get("model_source", k))
            rel_1 = m["one_step_relative_error"]
            rel_8 = m["free_running_rel_error_by_horizon"].get("8", 0.0)
            rel_16 = m["free_running_rel_error_by_horizon"].get("16", 0.0)
            mem_adv = m["temporal_memory_ablation"]["standard_memory_advantage_pct"]
            vp_adv = m["viewport_return_probe"]["return_event_memory_advantage_pct"]
            print(f"{name:<32} | {rel_1:<10.4f} | {rel_8:<10.4f} | {rel_16:<11.4f} | {mem_adv:<10.2f} | {vp_adv:<10.2f}")
    print("=" * 105)


if __name__ == "__main__":
    main()
