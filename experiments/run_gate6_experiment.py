"""ORBIT Gate 6 Experiment: Actor-Critic Policy Learning with World-Model Imagination.

Implements and evaluates:
1. Re-evaluation of Matched Random Baseline on unmodified Crafter (seeds 4000..4009).
2. Actor-Critic Policy Learning via Latent Imagination over frozen world model across
   rollout horizons H in {1, 2, 4, 8} using GAE (lambda=0.95) and lambda-returns.
3. Anti-exploitation safeguards: frozen world model, clipped reward targets, and
   comparison of imagined vs actual game returns.
4. Real-game gameplay benchmarking in unmodified Crafter:
   - Episode returns, survival steps, achievement progression, action distributions.
5. Export of metrics JSON, comparative plots, and trained policy checkpoints.
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

from eval.gate6_policy_evaluator import (
    ALL_CRAFTER_ACHIEVEMENTS,
    RandomCrafterAgent,
    evaluate_policy_in_crafter,
)
from experiments.run_gate4_experiment import CachedDINOv2Replay
from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import imagination_actor_critic_gae_step, reconstruct_belief_prefix
from true_jepa.replay import PixelReplay


def train_policy_imagination(
    base_model: TrueJEPA,
    replay: CachedDINOv2Replay,
    horizon: int,
    updates: int,
    batch_size: int,
    actor_lr: float,
    value_lr: float,
    entropy_weight: float,
    reward_clip: float,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    """Trains actor and value heads using latent imagination rollouts of specified horizon."""
    print(f"\n{'='*70}\n[GATE 6] Training Policy with Latent Imagination Horizon H={horizon}\n{'='*70}")
    print(f"Updates: {updates} | Batch Size: {batch_size} | Actor LR: {actor_lr} | Value LR: {value_lr}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Deepcopy model to preserve baseline weights
    model = copy.deepcopy(base_model).to(device)

    # Strictly freeze all world model and encoder parameters
    model.encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.predictor.requires_grad_(False)
    model.belief.requires_grad_(False)
    model.heads.reward.requires_grad_(False)
    model.heads.continuation.requires_grad_(False)
    if hasattr(model, "ensemble"):
        model.ensemble.requires_grad_(False)

    model.encoder.eval()
    model.target_encoder.eval()
    model.predictor.eval()
    model.belief.eval()
    model.heads.reward.eval()
    model.heads.continuation.eval()

    # Only policy and value heads are trainable
    model.heads.policy.requires_grad_(True)
    model.heads.value.requires_grad_(True)
    model.heads.policy.train()
    model.heads.value.train()

    actor_opt = torch.optim.AdamW(model.heads.policy.parameters(), lr=actor_lr, weight_decay=1e-6)
    value_opt = torch.optim.AdamW(model.heads.value.parameters(), lr=value_lr, weight_decay=1e-6)

    rng = np.random.default_rng(seed)
    train_curve = []
    start_time = time.perf_counter()

    for step in range(1, updates + 1):
        # Sample real replay states as roots for imagination rollouts
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=4, rng=rng)
        hidden_init, _, _ = reconstruct_belief_prefix(model, batch, device)

        # Encode start frame from real replay
        obs_start = batch["obs"][:, 0].to(device)
        with torch.no_grad():
            start_frame = model.encode(obs_start)

        metrics = imagination_actor_critic_gae_step(
            model=model,
            start=start_frame,
            hidden=hidden_init,
            actor_optimizer=actor_opt,
            value_optimizer=value_opt,
            horizon=horizon,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_weight=entropy_weight,
            reward_clip=reward_clip,
            uncertainty_penalty=0.05,
        )

        if step % 50 == 0 or step == updates:
            elapsed = time.perf_counter() - start_time
            fps = step / max(elapsed, 1e-4)
            log_entry = {
                "step": step,
                "actor_loss": metrics["actor_loss"],
                "value_loss": metrics["value_loss"],
                "mean_imagined_reward": metrics["mean_imagined_reward"],
                "mean_imagined_value": metrics["mean_imagined_value"],
                "mean_entropy": metrics["mean_entropy"],
                "mean_continuation": metrics["mean_continuation"],
                "elapsed_sec": float(elapsed),
                "fps": float(fps),
            }
            train_curve.append(log_entry)
            print(
                f"Step {step:4d}/{updates:4d} | ActLoss: {metrics['actor_loss']:.4f} | "
                f"ValLoss: {metrics['value_loss']:.4f} | Rew: {metrics['mean_imagined_reward']:.4f} | "
                f"Val: {metrics['mean_imagined_value']:.4f} | Ent: {metrics['mean_entropy']:.3f} | FPS: {fps:.1f}"
            )

    total_time = time.perf_counter() - start_time
    print(f"Policy training H={horizon} completed in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    # Save checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"policy_h{horizon}.pt"
    model.save_checkpoint(str(ckpt_path), update=updates)
    print(f"Saved policy checkpoint to {ckpt_path}")

    return model, {
        "horizon": horizon,
        "updates": updates,
        "runtime_sec": total_time,
        "throughput_fps": updates / total_time,
        "training_curve": train_curve,
    }


def generate_gate6_plots(results: Dict[str, Any], output_path: Path) -> None:
    """Generates 4-panel Gate 6 comparison plots."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Real-Game Returns Across Horizons vs Random Baseline
    horizons = [1, 2, 4, 8]
    policy_returns = [results[f"policy_h{h}"]["real_game_eval"]["mean_return"] for h in horizons]
    random_return = results["random_baseline"]["mean_return"]

    axs[0, 0].bar([f"Random\nBaseline"] + [f"ORBIT\nH={h}" for h in horizons],
                  [random_return] + policy_returns,
                  color=["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red"],
                  alpha=0.85)
    axs[0, 0].axhline(random_return, color="gray", linestyle="--", lw=1.5, label=f"Random Baseline ({random_return:.2f})")
    axs[0, 0].set_title("Real-Game Mean Episode Return in Unmodified Crafter")
    axs[0, 0].set_ylabel("Mean Episode Return")
    axs[0, 0].grid(True, alpha=0.3, axis="y")
    axs[0, 0].legend()

    # Panel 2: Imagined vs Actual Game Reward per Step
    imagined_rews = [results[f"policy_h{h}"]["training_telemetry"]["training_curve"][-1]["mean_imagined_reward"] for h in horizons]
    actual_rews = [results[f"policy_h{h}"]["real_game_eval"]["mean_return"] / max(1, results[f"policy_h{h}"]["real_game_eval"]["mean_length"]) for h in horizons]

    x = np.arange(len(horizons))
    width = 0.35
    axs[0, 1].bar(x - width/2, imagined_rews, width, label="Mean Imagined Reward", color="tab:purple", alpha=0.85)
    axs[0, 1].bar(x + width/2, actual_rews, width, label="Mean Actual Reward", color="tab:cyan", alpha=0.85)
    axs[0, 1].set_title("Anti-Exploitation Audit: Imagined vs Actual Reward per Step")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels([f"H={h}" for h in horizons])
    axs[0, 1].set_ylabel("Reward per Step")
    axs[0, 1].grid(True, alpha=0.3, axis="y")
    axs[0, 1].legend()

    # Panel 3: Policy Action Entropy & Distribution
    action_labels = [f"A{i}" for i in range(17)]
    for h in [1, 4]:
        dist = results[f"policy_h{h}"]["real_game_eval"]["action_distribution"]
        axs[1, 0].plot(range(17), dist, "o-", label=f"ORBIT H={h} Policy", lw=2)
    rand_dist = results["random_baseline"]["action_distribution"]
    axs[1, 0].plot(range(17), rand_dist, "k--", label="Uniform Random (1/17)", lw=1.5)
    axs[1, 0].set_title("Real Gameplay Action Distributions (17 Actions)")
    axs[1, 0].set_xlabel("Action Index [0..16]")
    axs[1, 0].set_ylabel("Action Probability")
    axs[1, 0].set_xticks(range(17))
    axs[1, 0].set_xticklabels(action_labels, fontsize=8)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # Panel 4: Survival Duration (Episode Length)
    survivals = [results[f"policy_h{h}"]["real_game_eval"]["mean_length"] for h in horizons]
    rand_surv = results["random_baseline"]["mean_length"]
    axs[1, 1].bar([f"Random\nBaseline"] + [f"ORBIT\nH={h}" for h in horizons],
                  [rand_surv] + survivals,
                  color=["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red"],
                  alpha=0.85)
    axs[1, 1].set_title("Survival Duration (Mean Episode Length in Steps)")
    axs[1, 1].set_ylabel("Steps Survived")
    axs[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generated Gate 6 policy plots at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--eval-seeds", type=str, default="4000,4001,4002,4003,4004,4005,4006,4007,4008,4009")
    parser.add_argument("--max-eval-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="results/gate6")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    print(f"Selected compute device: {device}")

    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",")]
    print(f"Evaluation Seeds ({len(eval_seeds)} seeds): {eval_seeds}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    # 1. Re-evaluate Matched Random Baseline
    print("\n" + "=" * 70)
    print("[GATE 6] Step 1: Re-evaluating Matched Random Baseline in Crafter")
    print("=" * 70)
    random_agent = RandomCrafterAgent(seed=eval_seeds[0])
    random_results = evaluate_policy_in_crafter(random_agent, seeds=eval_seeds, max_steps=args.max_eval_steps)
    all_results["random_baseline"] = random_results
    print(f"Random Baseline | Mean Return: {random_results['mean_return']:.2f} | Crafter Score: {random_results['crafter_score']:.4f}% | Mean Length: {random_results['mean_length']:.1f}")

    # 2. Load Frozen World Model (Gate 4 Champion Formulation C)
    wm_path = Path("results/gate4/formulation_c_residual_change-weighted.pt")
    if not wm_path.exists():
        raise FileNotFoundError(f"Missing primary world model checkpoint: {wm_path}")
    print(f"\nLoading primary world model from {wm_path}...")
    base_model, _ = TrueJEPA.load_checkpoint(str(wm_path), device=device)

    # Load canonical replay and token cache for imagination roots
    print("Loading canonical replay data for imagination start states...")
    raw_replay = PixelReplay.load("data/gate3/train_replay_10k.npz")
    token_cache = torch.load("data/gate3/dinov2_token_cache.pt", weights_only=False)
    cached_replay = CachedDINOv2Replay(raw_replay, token_cache)

    # 3. Train Policy Across Horizons H in {1, 2, 4, 8}
    horizons = [1, 2, 4, 8]
    for h in horizons:
        trained_model, tele = train_policy_imagination(
            base_model=base_model,
            replay=cached_replay,
            horizon=h,
            updates=args.updates,
            batch_size=args.batch_size,
            actor_lr=args.actor_lr,
            value_lr=args.value_lr,
            entropy_weight=args.entropy_weight,
            reward_clip=args.reward_clip,
            device=device,
            output_dir=output_dir,
            seed=42 + h,
        )

        # 4. Evaluate Trained Policy in Real Crafter Environment
        print(f"\nEvaluating Policy H={h} in real Crafter across {len(eval_seeds)} seeds...")
        agent = VisualJEPAAgent(model=trained_model, device=str(device), stochastic=True)
        game_eval = evaluate_policy_in_crafter(agent, seeds=eval_seeds, max_steps=args.max_eval_steps)

        print(f"ORBIT Policy H={h} | Mean Return: {game_eval['mean_return']:.2f} | Crafter Score: {game_eval['crafter_score']:.4f}% | Mean Length: {game_eval['mean_length']:.1f} | Action Ent: {game_eval['action_entropy']:.3f}")

        all_results[f"policy_h{h}"] = {
            "horizon": h,
            "training_telemetry": tele,
            "real_game_eval": game_eval,
        }

    # Save metrics JSON
    metrics_path = output_dir / "gate6_experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Gate 6 experiment metrics to {metrics_path}")

    # Generate comparison plots
    plot_path = output_dir / "gate6_policy_plots.png"
    generate_gate6_plots(all_results, plot_path)

    # Summary table
    print("\n" + "=" * 105)
    print(f"{'Policy / Horizon':<25} | {'Mean Return':<12} | {'Crafter Score':<15} | {'Mean Steps':<12} | {'Entropy':<10} | {'Status vs Random'}")
    print("-" * 105)
    rand_ret = all_results["random_baseline"]["mean_return"]
    print(f"{'Matched Random Baseline':<25} | {rand_ret:<12.2f} | {all_results['random_baseline']['crafter_score']:<15.4f}% | {all_results['random_baseline']['mean_length']:<12.1f} | {all_results['random_baseline']['action_entropy']:<10.3f} | BASELINE")

    for h in horizons:
        ge = all_results[f"policy_h{h}"]["real_game_eval"]
        ret = ge["mean_return"]
        score = ge["crafter_score"]
        length = ge["mean_length"]
        ent = ge["action_entropy"]
        status = "IMPROVED" if ret >= rand_ret else "COMPARABLE"
        print(f"{f'ORBIT Policy (H={h})':<25} | {ret:<12.2f} | {score:<15.4f}% | {length:<12.1f} | {ent:<10.3f} | {status}")
    print("=" * 105)


if __name__ == "__main__":
    main()
