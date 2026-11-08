"""ORBIT Gate 4 Controlled World Model Experiment Execution.

Evaluates and compares:
- Formulation A: Standard unweighted prediction loss (non-residual baseline)
- Formulation B: Residual prediction with unweighted loss
- Formulation C: Residual prediction with spatial change-weighted loss

All three formulations use the frozen Meta AI DINOv2 ViT-S/14 encoder with trainable adapter
and the recurrent belief architecture. Trained on identical replay data (10,000 transitions,
seeds 3000..3029) and evaluated on 20 held-out seeds (3030..3049).
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

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate4_evaluator import evaluate_gate4_formulation
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import train_world_model_step
from true_jepa.replay import PixelReplay


class CachedDINOv2Replay:
    """Bitwise-equivalent replay sampler using pre-extracted frozen DINOv2 patch tokens."""

    def __init__(self, raw_replay: PixelReplay, token_cache: List[np.ndarray]) -> None:
        self.raw_replay = raw_replay
        self.token_cache = token_cache
        self._episodes = raw_replay._episodes
        self._frames = raw_replay._frames

    def sample_with_prefix(
        self, batch_size: int, sequence_length: int, rng: np.random.Generator
    ) -> Dict[str, torch.Tensor]:
        if not self._episodes:
            raise RuntimeError("cannot sample an empty replay")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        token_shape = self.token_cache[0].shape[1:]
        obs = np.zeros((batch_size, sequence_length + 1) + token_shape, dtype=np.float32)
        actions = np.zeros((batch_size, sequence_length), dtype=np.int64)
        rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
        terminated = np.zeros((batch_size, sequence_length), dtype=bool)
        truncated = np.zeros((batch_size, sequence_length), dtype=bool)
        valid = np.zeros((batch_size, sequence_length), dtype=bool)

        sampled_indices = []
        starts = []

        for row in range(batch_size):
            ep_idx = int(rng.integers(0, len(self._episodes)))
            episode = self._episodes[ep_idx]
            ep_tokens = self.token_cache[ep_idx]
            length = int(episode.actions.shape[0])
            use = min(length, sequence_length)
            start = 0 if length <= sequence_length else int(rng.integers(0, length - sequence_length + 1))
            sampled_indices.append(ep_idx)
            starts.append(start)

            obs[row, : use + 1] = ep_tokens[start : start + use + 1]
            if use < sequence_length:
                obs[row, use + 1 :] = ep_tokens[-1]
            actions[row, :use] = episode.actions[start : start + use]
            rewards[row, :use] = episode.rewards[start : start + use]
            terminated[row, :use] = episode.terminated[start : start + use]
            truncated[row, :use] = episode.truncated[start : start + use]
            valid[row, :use] = True

        max_prefix = max(starts)
        prefix_obs = np.zeros((batch_size, max_prefix + 1) + token_shape, dtype=np.float32)
        prefix_actions = np.zeros((batch_size, max_prefix), dtype=np.int64)
        prefix_rewards = np.zeros((batch_size, max_prefix), dtype=np.float32)
        prefix_lengths = np.array(starts, dtype=np.int64)

        for row, (ep_idx, start) in enumerate(zip(sampled_indices, starts)):
            if start > 0:
                ep = self._episodes[ep_idx]
                ep_tokens = self.token_cache[ep_idx]
                prefix_obs[row, : start + 1] = ep_tokens[: start + 1]
                prefix_actions[row, :start] = ep.actions[:start]
                prefix_rewards[row, :start] = ep.rewards[:start]

        return {
            "obs": torch.from_numpy(obs),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "valid": torch.from_numpy(valid),
            "prefix_obs": torch.from_numpy(prefix_obs),
            "prefix_actions": torch.from_numpy(prefix_actions),
            "prefix_rewards": torch.from_numpy(prefix_rewards),
            "prefix_lengths": torch.from_numpy(prefix_lengths),
        }


def compute_drift_and_rank_diagnostics(
    model: TrueJEPA, eval_dataset: List[Dict[str, Any]], device: torch.device
) -> Dict[str, float]:
    """Computes online-target drift, active dimensions, and effective rank."""
    model.eval()
    all_tokens = []
    all_target_tokens = []
    drift_l2_list = []
    drift_cos_list = []

    with torch.no_grad():
        for ep in eval_dataset[:10]:
            obs = torch.from_numpy(ep["obs"]).to(device)
            T = ep["action"].shape[0]
            if T < 10:
                continue
            sample_idx = torch.linspace(0, T, min(32, T + 1), dtype=torch.long)
            online_f = model.encode(obs[sample_idx])
            target_f = model.target_encode(obs[sample_idx])

            all_tokens.append(online_f.tokens.cpu())
            all_target_tokens.append(target_f.tokens.cpu())

            o_tok = online_f.tokens
            t_tok = target_f.tokens
            l2_drift = ((o_tok - t_tok).norm(dim=-1) / (t_tok.norm(dim=-1).clamp_min(1e-6))).mean().item()
            cos_align = F.cosine_similarity(o_tok, t_tok, dim=-1).mean().item()
            drift_l2_list.append(l2_drift)
            drift_cos_list.append(cos_align)

    tokens_cat = torch.cat(all_tokens, dim=0).reshape(-1, 192).float()
    var_per_dim = tokens_cat.var(dim=0, unbiased=False)
    active_dims = int((var_per_dim > 1e-4).sum().item())

    centered = tokens_cat - tokens_cat.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = eigenvals / eigenvals.sum()
    entropy = -(p * torch.log(p)).sum().item()
    eff_rank = float(np.exp(entropy))

    return {
        "active_dimensions": active_dims,
        "effective_rank": eff_rank,
        "online_target_l2_drift": float(np.mean(drift_l2_list)),
        "online_target_cos_alignment": float(np.mean(drift_cos_list)),
    }


def train_formulation(
    formulation_id: str,
    name: str,
    residual: bool,
    change_weighted: bool,
    replay: CachedDINOv2Replay,
    eval_dataset: List[Dict[str, Any]],
    updates: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    print(f"\n{'='*70}\n[GATE 4] Starting Training: Formulation {formulation_id} — {name}\n{'='*70}")
    print(f"Residual Mode: {residual} | Change-Weighted Loss: {change_weighted} | Updates: {updates}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = JEPAConfig(predictor_residual=residual)
    model = TrueJEPA(cfg, num_actions=17, encoder_type="dinov2").to(device)

    train_cfg = TrainConfig(
        batch_size=batch_size,
        sequence_length=seq_len,
        learning_rate=3e-4,
        weight_decay=1e-6,
        change_weighted_loss=change_weighted,
        change_weight_cap=5.0,
        change_weight_bonus=2.0,
        amp=False,
    )

    world_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(world_params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    rng = np.random.default_rng(seed=seed)
    training_curve = []
    meaningful_history = []
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

        if "meaningful_token_fraction" in metrics:
            meaningful_history.append(metrics["meaningful_token_fraction"])

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

            extra_str = f" | DynFrac: {metrics['meaningful_token_fraction']:.1%}" if "meaningful_token_fraction" in metrics else ""
            print(f"Step {step:4d}/{updates:4d} | Total: {metrics['loss']:.4f} | JEPA: {metrics['jepa']:.4f}{extra_str} | FPS: {fps:.1f}")

    total_time = time.perf_counter() - start_time
    print(f"Training completed in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    # Save checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"formulation_{formulation_id.lower()}_{name.lower().replace(' ', '_')}.pt"
    model.save_checkpoint(str(ckpt_path), update=updates)
    print(f"Saved checkpoint to {ckpt_path}")

    # Evaluate held-out seeds
    print(f"Evaluating held-out transitions on 20 seeds...")
    eval_metrics = evaluate_gate4_formulation(model, eval_dataset, device, change_threshold=0.35)
    drift_metrics = compute_drift_and_rank_diagnostics(model, eval_dataset, device)
    eval_metrics.update(drift_metrics)

    eval_metrics["formulation_id"] = formulation_id
    eval_metrics["formulation_name"] = name
    eval_metrics["residual"] = residual
    eval_metrics["change_weighted"] = change_weighted
    eval_metrics["training_telemetry"] = {
        "total_runtime_sec": total_time,
        "average_throughput_fps": updates / total_time,
        "final_jepa_loss": training_curve[-1]["jepa_loss"],
        "final_total_loss": training_curve[-1]["total_loss"],
    }
    if meaningful_history:
        eval_metrics["training_telemetry"]["mean_meaningful_token_fraction"] = float(np.mean(meaningful_history))

    eval_metrics["training_curve"] = training_curve
    return model, eval_metrics


def generate_comparison_plots(metrics: Dict[str, Any], output_path: Path) -> None:
    """Generates 4-panel comparison plots across Formulations A, B, and C."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    labels = {
        "A": "Formulation A (Standard Non-Residual)",
        "B": "Formulation B (Residual Unweighted)",
        "C": "Formulation C (Residual Change-Weighted)",
    }

    # 1. JEPA Training Loss
    for f_id in ["A", "B", "C"]:
        curve = metrics[f"formulation_{f_id}"]["training_curve"]
        steps = [x["step"] for x in curve]
        losses = [x["jepa_loss"] for x in curve]
        axs[0, 0].plot(steps, losses, label=labels[f_id], color=colors[f_id], lw=2)
    axs[0, 0].set_title("JEPA Training Loss (1,000 Updates)")
    axs[0, 0].set_xlabel("Updates")
    axs[0, 0].set_ylabel("JEPA Loss")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. Static vs Changed Token Relative Error
    categories = ["Overall", "Static Tokens", "Changed Tokens"]
    x = np.arange(len(categories))
    width = 0.25

    for idx, f_id in enumerate(["A", "B", "C"]):
        m = metrics[f"formulation_{f_id}"]
        rel_overall = m["one_step_relative_error"]
        rel_static = m["static_tokens"]["relative_error"]
        rel_changed = m["changed_tokens"]["relative_error"]
        vals = [rel_overall, rel_static, rel_changed]
        axs[0, 1].bar(x + (idx - 1) * width, vals, width, label=f"Formulation {f_id}", color=colors[f_id], alpha=0.85)

    axs[0, 1].axhline(1.0, color="red", linestyle="--", lw=1.5, label="Persistence Baseline (1.0)")
    axs[0, 1].set_ylabel("Relative Error (Model / Persistence)")
    axs[0, 1].set_title("Held-Out 1-Step Relative Error Breakdown")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(categories)
    axs[0, 1].grid(True, alpha=0.3, axis="y")
    axs[0, 1].legend()

    # 3. Teacher-Forced Multi-Step Horizon Error
    horizons = [1, 2, 4, 8]
    for f_id in ["A", "B", "C"]:
        tf_dict = metrics[f"formulation_{f_id}"]["teacher_forced_errors_by_horizon"]
        tf_vals = [tf_dict[str(h)] for h in horizons]
        axs[1, 0].plot(horizons, tf_vals, "o-", label=labels[f_id], color=colors[f_id], lw=2)
    axs[1, 0].set_title("Teacher-Forced Multi-Step Horizon Error (H=1, 2, 4, 8)")
    axs[1, 0].set_xlabel("Rollout Horizon (Steps)")
    axs[1, 0].set_ylabel("Normalized Latent MSE")
    axs[1, 0].set_xticks(horizons)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. Free-Running Autoregressive Rollout Error
    for f_id in ["A", "B", "C"]:
        fr_dict = metrics[f"formulation_{f_id}"]["free_running_errors_by_horizon"]
        fr_vals = [fr_dict[str(h)] for h in horizons]
        axs[1, 1].plot(horizons, fr_vals, "s-", label=labels[f_id], color=colors[f_id], lw=2)
    axs[1, 1].set_title("Free-Running Autoregressive Compounding Error (H=1, 2, 4, 8)")
    axs[1, 1].set_xlabel("Rollout Horizon (Steps)")
    axs[1, 1].set_ylabel("Normalized Latent MSE")
    axs[1, 1].set_xticks(horizons)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generated comparison plots at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="results/gate4")
    parser.add_argument("--formulations", type=str, default="A,B,C", help="Comma-separated formulations to run")
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

    formulations_to_run = [f.strip().upper() for f in args.formulations.split(",")]
    all_metrics: Dict[str, Any] = {}

    # Formulation A
    if "A" in formulations_to_run:
        # Check if pre-trained Gate 3 Candidate B exists (which is exact Formulation A)
        gate3_candidate_b = Path("results/gate3/candidate_b_dinov2.pt")
        ckpt_a = output_dir / "formulation_a_standard.pt"
        if gate3_candidate_b.exists() and not ckpt_a.exists():
            print(f"Re-using verified 1,000-update Gate 3 Candidate B checkpoint for Formulation A baseline...")
            shutil.copy(gate3_candidate_b, ckpt_a)

        if ckpt_a.exists():
            print(f"Loading Formulation A from {ckpt_a} for full Gate 4 evaluation...")
            model_a, meta_a = TrueJEPA.load_checkpoint(str(ckpt_a), device=device)
            eval_a = evaluate_gate4_formulation(model_a, eval_dataset, device, change_threshold=0.35)
            eval_a.update(compute_drift_and_rank_diagnostics(model_a, eval_dataset, device))
            # Load training curve from gate3
            with open("results/gate3/gate3_representation_metrics.json") as f:
                g3_data = json.load(f)
            eval_a["training_curve"] = g3_data["candidate_b_dinov2"]["training_curve"]
            eval_a["training_telemetry"] = g3_data["candidate_b_dinov2"]["training_telemetry"]
            eval_a["formulation_id"] = "A"
            eval_a["formulation_name"] = "Standard Non-Residual"
            eval_a["residual"] = False
            eval_a["change_weighted"] = False
            all_metrics["formulation_A"] = eval_a
        else:
            _, eval_a = train_formulation(
                formulation_id="A",
                name="Standard Non-Residual",
                residual=False,
                change_weighted=False,
                replay=cached_replay,
                eval_dataset=eval_dataset,
                updates=args.updates,
                batch_size=args.batch_size,
                seq_len=args.sequence_length,
                device=device,
                output_dir=output_dir,
            )
            all_metrics["formulation_A"] = eval_a

    # Formulation B
    if "B" in formulations_to_run:
        _, eval_b = train_formulation(
            formulation_id="B",
            name="Residual Unweighted",
            residual=True,
            change_weighted=False,
            replay=cached_replay,
            eval_dataset=eval_dataset,
            updates=args.updates,
            batch_size=args.batch_size,
            seq_len=args.sequence_length,
            device=device,
            output_dir=output_dir,
        )
        all_metrics["formulation_B"] = eval_b

    # Formulation C
    if "C" in formulations_to_run:
        _, eval_c = train_formulation(
            formulation_id="C",
            name="Residual Change-Weighted",
            residual=True,
            change_weighted=True,
            replay=cached_replay,
            eval_dataset=eval_dataset,
            updates=args.updates,
            batch_size=args.batch_size,
            seq_len=args.sequence_length,
            device=device,
            output_dir=output_dir,
        )
        all_metrics["formulation_C"] = eval_c

    # Save metrics JSON
    metrics_path = output_dir / "gate4_experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved Gate 4 experiment metrics to {metrics_path}")

    # Generate comparison plots
    plot_path = output_dir / "gate4_comparison_plots.png"
    generate_comparison_plots(all_metrics, plot_path)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Formulation':<32} | {'RelPredErr':<10} | {'Static RelErr':<13} | {'Changed RelErr':<14} | {'Shuf Gap':<10}")
    print("-" * 90)
    for f_id in ["A", "B", "C"]:
        if f"formulation_{f_id}" in all_metrics:
            m = all_metrics[f"formulation_{f_id}"]
            name = m["formulation_name"]
            rel = m["one_step_relative_error"]
            stat_rel = m["static_tokens"]["relative_error"]
            chg_rel = m["changed_tokens"]["relative_error"]
            gap = m["shuffled_action_sensitivity_gap"]
            print(f"Formulation {f_id} ({name:<20}) | {rel:<10.4f} | {stat_rel:<13.4f} | {chg_rel:<14.4f} | {gap:<10.2e}")
    print("=" * 90)


if __name__ == "__main__":
    main()
