"""ORBIT Phase 3: Iterative Self-Play and Online Replay Augmentation.

Controlled 6-Stage Execution Cycle:
Stage A: Collect real interaction trajectories in unmodified Crafter using current H=4 policy.
Stage B: Fine-tune the world model on augmented replay buffer (prior 10k replay + new self-play).
Stage C: Evaluate updated world model against Gate 4 champion on held-out evaluation seeds.
Stage D: Freeze selected world model.
Stage E: Train policy actor & value heads via latent imagination (H=4, GAE lambda=0.95, gamma=0.99).
Stage F: Comprehensive 30-seed gameplay benchmark in unmodified Crafter (Random vs Gate 6 vs Phase 3).
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import crafter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate4_evaluator import evaluate_gate4_formulation
from eval.gate6_policy_evaluator import (
    ALL_CRAFTER_ACHIEVEMENTS,
    RandomCrafterAgent,
    compute_crafter_score,
    evaluate_policy_in_crafter,
)
from experiments.run_gate4_experiment import CachedDINOv2Replay
from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.objectives import (
    imagination_actor_critic_gae_step,
    reconstruct_belief_prefix,
    train_world_model_step,
)
from true_jepa.replay import PixelEpisode, PixelReplay


# ==============================================================================
# STAGE A: Self-Play Data Collection and Augmented Token Caching
# ==============================================================================

def collect_selfplay_episodes(
    policy_model: TrueJEPA,
    num_episodes: int,
    start_seed: int,
    max_steps: int = 300,
    temperature: float = 1.2,
    device: str = "cpu",
) -> Tuple[List[PixelEpisode], Dict[str, Any]]:
    """Collects real trajectories using the specified policy in unmodified Crafter.

    Strictly satisfies:
    - Policy strictly receives (obs_uint8, reward_prev, step).
    - Episode records contain zero privileged simulator state.
    """
    print(f"\n{'='*70}\n[PHASE 3 — STAGE A] Self-Play Data Collection\n{'='*70}")
    print(f"Episodes: {num_episodes} | Start Seed: {start_seed} | Max Steps: {max_steps} | Temp: {temperature}")

    agent = VisualJEPAAgent(policy_model, device=device, stochastic=True, temperature=temperature)
    episodes: List[PixelEpisode] = []
    total_frames = 0
    total_reward = 0.0
    action_counts = np.zeros(17, dtype=np.int64)

    start_time = time.perf_counter()
    for ep_i in range(num_episodes):
        seed = start_seed + ep_i
        env = crafter.Env(seed=seed)
        obs = env.reset()
        agent.reset()

        obs_list = [obs]
        actions_list = []
        rewards_list = []
        terminated_list = []
        truncated_list = []

        ep_reward = 0.0
        step_count = 0
        reward_prev = 0.0
        done = False

        while not done and step_count < max_steps:
            action = agent.act(obs, reward_prev=reward_prev, step=step_count)
            action_int = int(action)
            action_counts[action_int] += 1

            next_obs, reward, done, info = env.step(action_int)
            reward_float = float(reward)
            ep_reward += reward_float

            obs_list.append(next_obs)
            actions_list.append(action_int)
            rewards_list.append(reward_float)
            terminated_list.append(bool(done and step_count < max_steps - 1))
            truncated_list.append(bool(step_count >= max_steps - 1))

            obs = next_obs
            reward_prev = reward_float
            step_count += 1

        episode = PixelEpisode(
            obs=np.array(obs_list, dtype=np.uint8),
            actions=np.array(actions_list, dtype=np.int64),
            rewards=np.array(rewards_list, dtype=np.float32),
            terminated=np.array(terminated_list, dtype=bool),
            truncated=np.array(truncated_list, dtype=bool),
            seed=seed,
            source="self_play_h4",
        )
        episodes.append(episode)
        total_frames += len(obs_list)
        total_reward += ep_reward

        print(f"  Collected Episode {ep_i+1:2d}/{num_episodes}: Seed={seed} | Steps={step_count:3d} | Return={ep_reward:+.1f}")

    elapsed = time.perf_counter() - start_time
    stats = {
        "num_episodes": num_episodes,
        "total_frames": total_frames,
        "mean_steps": total_frames / num_episodes,
        "mean_return": total_reward / num_episodes,
        "collection_sec": elapsed,
        "collection_fps": total_frames / elapsed,
        "action_distribution": (action_counts / action_counts.sum()).tolist(),
    }
    print(f"Self-play collection complete: {total_frames} frames collected in {elapsed:.2f}s ({total_frames/elapsed:.1f} FPS)")
    return episodes, stats


def build_augmented_replay_and_cache(
    base_replay_path: Path,
    base_token_cache_path: Path,
    new_episodes: List[PixelEpisode],
    model: TrueJEPA,
    device: torch.device,
    output_dir: Path,
) -> Tuple[CachedDINOv2Replay, Path, Path]:
    """Combines baseline replay with new self-play episodes and extracts DINOv2 tokens."""
    print(f"\n[PHASE 3 — STAGE A] Augmenting Replay Buffer & Extracting Patch Tokens...")
    raw_replay = PixelReplay.load(str(base_replay_path))
    raw_replay.capacity_frames = 500_000  # Expand capacity to prevent eviction desyncing token cache
    token_cache: List[np.ndarray] = torch.load(str(base_token_cache_path), weights_only=False)

    print(f"Baseline Replay: {len(raw_replay.records)} episodes, {len(raw_replay)} frames")

    # Extract frozen DINOv2 ViT-S/14 patch tokens for the new episodes
    model.encoder.eval()
    new_token_list: List[np.ndarray] = []
    
    with torch.no_grad():
        for ep_idx, ep in enumerate(new_episodes):
            raw_obs = torch.from_numpy(ep.obs).to(device)  # [T+1, 64, 64, 3]
            # Chunk through backbone in micro-batches if needed
            micro_batch = 64
            ep_tokens = []
            for b in range(0, len(raw_obs), micro_batch):
                batch_obs = raw_obs[b : b + micro_batch]
                norm_obs = model.encoder.preprocess(batch_obs)
                feats = model.encoder.backbone.forward_features(norm_obs)
                tokens = feats["x_norm_patchtokens"].cpu().numpy()  # [B, 256, 384]
                ep_tokens.append(tokens)
            ep_tokens_arr = np.concatenate(ep_tokens, axis=0)  # [T+1, 256, 384]
            new_token_list.append(ep_tokens_arr)
            raw_replay.add(ep)

    augmented_token_cache = token_cache + new_token_list
    assert len(raw_replay.records) == len(augmented_token_cache), (
        f"Replay episodes ({len(raw_replay.records)}) must match token cache ({len(augmented_token_cache)})"
    )
    for i, (rec, tok) in enumerate(zip(raw_replay.records, augmented_token_cache)):
        assert rec.obs.shape[0] == tok.shape[0], f"Frame mismatch at idx {i}: {rec.obs.shape[0]} vs {tok.shape[0]}"
    print(f"Augmented Replay: {len(raw_replay.records)} episodes, {len(raw_replay)} frames (all synchronized)")

    # Save self-play datasets and token cache (provenance preserved cleanly)
    aug_replay_dir = output_dir / "data"
    aug_replay_dir.mkdir(parents=True, exist_ok=True)
    selfplay_replay_path = aug_replay_dir / "selfplay_h4_replay.npz"
    selfplay_cache_path = aug_replay_dir / "selfplay_token_cache.pt"

    selfplay_replay = PixelReplay()
    for ep in new_episodes:
        selfplay_replay.add(ep)
    selfplay_replay.save(str(selfplay_replay_path))
    torch.save(new_token_list, str(selfplay_cache_path))
    print(f"Saved self-play replay ({len(new_episodes)} eps) to {selfplay_replay_path}")
    print(f"Saved self-play token cache to {selfplay_cache_path}")

    cached_replay = CachedDINOv2Replay(raw_replay, augmented_token_cache)
    return cached_replay, selfplay_replay_path, selfplay_cache_path


# ==============================================================================
# STAGE B: Fine-Tune World Model on Augmented Replay Buffer
# ==============================================================================

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
    """Fine-tunes the world model predictor and belief core on augmented replay data."""
    print(f"\n{'='*70}\n[PHASE 3 — STAGE B] World Model Fine-Tuning on Augmented Experience\n{'='*70}")
    print(f"Updates: {updates} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(base_model).to(device)

    # Freeze frozen DINOv2 backbone permanently
    model.encoder.backbone.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.encoder.backbone.eval()
    model.target_encoder.eval()

    # World model components to train: adapter, predictor, belief, heads
    trainable_params = [
        {"params": model.encoder.adapter.parameters(), "lr": learning_rate * 0.5},
        {"params": model.predictor.parameters(), "lr": learning_rate},
        {"params": model.belief.parameters(), "lr": learning_rate},
        {"params": model.heads.reward.parameters(), "lr": learning_rate},
        {"params": model.heads.continuation.parameters(), "lr": learning_rate},
        {"params": model.ensemble.parameters(), "lr": learning_rate},
    ]

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
                f"Step {step:4d}/{updates:4d} | TotLoss: {metrics['loss']:.4f} | "
                f"JEPA: {metrics['jepa']:.4f} | Rew: {metrics['reward']:.4f} | "
                f"ChgFrac: {metrics.get('meaningful_token_fraction', 0.0):.3f} | FPS: {fps:.1f}"
            )

    total_time = time.perf_counter() - start_time
    print(f"World model fine-tuning completed in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "world_model_phase3.pt"
    model.save_checkpoint(str(ckpt_path), update=updates)
    print(f"Saved Phase 3 fine-tuned world model to {ckpt_path}")

    return model, {
        "updates": updates,
        "runtime_sec": total_time,
        "training_curve": train_curve,
        "checkpoint_path": str(ckpt_path),
    }


# ==============================================================================
# STAGE C & D: Controlled World Model Evaluation & Freezing
# ==============================================================================

def evaluate_and_compare_world_models(
    prev_model: TrueJEPA,
    updated_model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    """Evaluates previous Gate 4 model vs Phase 3 fine-tuned model on held-out seeds."""
    print(f"\n{'='*70}\n[PHASE 3 — STAGE C] World Model Diagnostic Evaluation on Held-Out Seeds\n{'='*70}")

    print("Evaluating Previous Model (Gate 4 Champion Formulation C)...")
    prev_metrics = evaluate_gate4_formulation(prev_model, eval_dataset, device)
    print(
        f"  Previous Model | RelErr: {prev_metrics['one_step_relative_error']:.4f} | "
        f"StaticErr: {prev_metrics['static_tokens']['relative_error']:.4f} | "
        f"ChgErr: {prev_metrics['changed_tokens']['relative_error']:.4f} | "
        f"ShuffGap: {prev_metrics['shuffled_action_sensitivity_gap']:.4f}"
    )

    print("\nEvaluating Updated Model (Phase 3 Fine-Tuned Formulation C)...")
    updated_metrics = evaluate_gate4_formulation(updated_model, eval_dataset, device)
    print(
        f"  Updated Model  | RelErr: {updated_metrics['one_step_relative_error']:.4f} | "
        f"StaticErr: {updated_metrics['static_tokens']['relative_error']:.4f} | "
        f"ChgErr: {updated_metrics['changed_tokens']['relative_error']:.4f} | "
        f"ShuffGap: {updated_metrics['shuffled_action_sensitivity_gap']:.4f}"
    )

    # Decision rule: Select updated model if relative error does not degrade substantially
    rel_prev = prev_metrics["one_step_relative_error"]
    rel_upd = updated_metrics["one_step_relative_error"]

    if rel_upd <= rel_prev * 1.05:  # within 5% tolerance or improved
        print(f"\n[PHASE 3 — STAGE D] SELECTION: Promoting Phase 3 Updated World Model for Policy Learning.")
        selected_model = updated_model
        selected_name = "phase3_updated"
    else:
        print(f"\n[PHASE 3 — STAGE D] SELECTION: Retaining Gate 4 Champion World Model (degradation detected).")
        selected_model = prev_model
        selected_name = "gate4_champion"

    # Strictly freeze selected model
    selected_model.requires_grad_(False)
    selected_model.eval()

    comparison = {
        "selected_model": selected_name,
        "previous_gate4": prev_metrics,
        "updated_phase3": updated_metrics,
    }
    return selected_model, comparison


# ==============================================================================
# STAGE E: Actor-Critic Policy Learning via Latent Imagination
# ==============================================================================

def train_phase3_policy_imagination(
    frozen_model: TrueJEPA,
    replay: CachedDINOv2Replay,
    horizon: int = 4,
    updates: int = 300,
    batch_size: int = 8,
    actor_lr: float = 1e-4,
    value_lr: float = 1e-3,
    entropy_weight: float = 0.03,  # Enhanced from 0.01 to prevent action collapse
    reward_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("results/phase3"),
    seed: int = 777,
) -> Tuple[TrueJEPA, Dict[str, Any]]:
    """Trains policy & value heads via imagination over the frozen world model."""
    print(f"\n{'='*70}\n[PHASE 3 — STAGE E] Policy Learning with Imagination (Horizon H={horizon})\n{'='*70}")
    print(f"Updates: {updates} | Batch Size: {batch_size} | Entropy Weight: {entropy_weight} | Actor LR: {actor_lr}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(frozen_model).to(device)

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

    # Only actor & value heads are trainable
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
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=4, rng=rng)
        hidden_init, _, _ = reconstruct_belief_prefix(model, batch, device)

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
    print(f"Policy training completed in {total_time:.2f}s ({updates/total_time:.1f} steps/s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "policy_phase3.pt"
    model.save_checkpoint(str(ckpt_path), update=updates)
    print(f"Saved Phase 3 policy checkpoint to {ckpt_path}")

    return model, {
        "horizon": horizon,
        "updates": updates,
        "runtime_sec": total_time,
        "training_curve": train_curve,
        "checkpoint_path": str(ckpt_path),
    }


# ==============================================================================
# STAGE F: 30-Seed Gameplay Benchmark in Unmodified Crafter
# ==============================================================================

def generate_phase3_plots(
    results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generates 4-panel Phase 3 comparative visualizations."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    labels = ["Random\nBaseline", "Gate 6\nChampion (H=4)", "Phase 3\nUpdated (H=4)"]
    keys = ["random_baseline", "gate6_champion_h4", "phase3_policy"]
    colors = ["tab:gray", "tab:blue", "tab:green"]

    # Panel 1: Real-Game Return & Crafter Score Across 30 Seeds
    returns = [results[k]["mean_return"] for k in keys]
    scores = [results[k]["crafter_score"] for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    axs[0, 0].bar(x - width/2, returns, width, label="Mean Episode Return", color="tab:blue", alpha=0.85)
    axs[0, 0].bar(x + width/2, scores, width, label="Crafter Score (%)", color="tab:orange", alpha=0.85)
    axs[0, 0].set_title("Real Gameplay: Mean Return & Crafter Score (30 Pinned Seeds)")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].grid(True, alpha=0.3, axis="y")
    axs[0, 0].legend()

    # Panel 2: Key Achievement Unlocks Comparison
    ach_names = ["collect_sapling", "place_plant", "eat_cow", "collect_wood", "wake_up"]
    x_ach = np.arange(len(ach_names))
    width_ach = 0.25

    for i, (k, color, lbl) in enumerate(zip(keys, colors, labels)):
        rates = [results[k]["achievement_rates"].get(ach, 0.0) * 100 for ach in ach_names]
        axs[0, 1].bar(x_ach + (i - 1) * width_ach, rates, width_ach, label=lbl.replace("\n", " "), color=color, alpha=0.85)

    axs[0, 1].set_title("Affordance Discovery: Achievement Unlock Rates (%)")
    axs[0, 1].set_xticks(x_ach)
    axs[0, 1].set_xticklabels(ach_names, rotation=20, ha="right")
    axs[0, 1].set_ylabel("Success Rate (%)")
    axs[0, 1].grid(True, alpha=0.3, axis="y")
    axs[0, 1].legend()

    # Panel 3: Action Distributions Across 17 Crafter Actions
    action_names = [f"A{i}" for i in range(17)]
    for k, color, lbl in zip(keys, colors, labels):
        dist = results[k]["action_distribution"]
        ent = results[k]["action_entropy"]
        axs[1, 0].plot(range(17), dist, "o-", label=f"{lbl.replace(chr(10), ' ')} (H={ent:.2f})", color=color, lw=2)

    axs[1, 0].set_title("Action Distributions & Shannon Entropy (17 Actions)")
    axs[1, 0].set_xlabel("Action Index [0..16]")
    axs[1, 0].set_ylabel("Probability")
    axs[1, 0].set_xticks(range(17))
    axs[1, 0].set_xticklabels(action_names, fontsize=8)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # Panel 4: Survival Steps Distribution (Mean & Median)
    means = [results[k]["mean_length"] for k in keys]
    medians = [results[k]["median_length"] for k in keys]

    axs[1, 1].bar(x - width/2, means, width, label="Mean Survival Steps", color="tab:purple", alpha=0.85)
    axs[1, 1].bar(x + width/2, medians, width, label="Median Survival Steps", color="tab:cyan", alpha=0.85)
    axs[1, 1].set_title("Survival Duration in Unmodified Crafter (Steps)")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].grid(True, alpha=0.3, axis="y")
    axs[1, 1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generated Phase 3 comparison plots at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--selfplay-episodes", type=int, default=20)
    parser.add_argument("--selfplay-start-seed", type=int, default=1000)
    parser.add_argument("--wm-updates", type=int, default=300)
    parser.add_argument("--policy-updates", type=int, default=300)
    parser.add_argument("--eval-seeds-count", type=int, default=30)
    parser.add_argument("--eval-start-seed", type=int, default=5000)
    parser.add_argument("--max-eval-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="results/phase3")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    print(f"Active Compute Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Current Gate 6 Champion Policy Candidate (policy_h4.pt)
    gate6_ckpt = Path("results/gate6/policy_h4.pt")
    if not gate6_ckpt.exists():
        raise FileNotFoundError(f"Missing Gate 6 champion checkpoint: {gate6_ckpt}")
    print(f"\nLoading Gate 6 champion policy from {gate6_ckpt}...")
    gate6_policy_model, _ = TrueJEPA.load_checkpoint(str(gate6_ckpt), device=device)

    # 2. Stage A: Collect Real Trajectories in Crafter
    selfplay_episodes, selfplay_stats = collect_selfplay_episodes(
        policy_model=gate6_policy_model,
        num_episodes=args.selfplay_episodes,
        start_seed=args.selfplay_start_seed,
        max_steps=args.max_eval_steps,
        temperature=1.2,
        device=str(device),
    )

    # Build Augmented Replay Buffer & Token Cache
    cached_replay, aug_replay_path, aug_cache_path = build_augmented_replay_and_cache(
        base_replay_path=Path("data/gate3/train_replay_10k.npz"),
        base_token_cache_path=Path("data/gate3/dinov2_token_cache.pt"),
        new_episodes=selfplay_episodes,
        model=gate6_policy_model,
        device=device,
        output_dir=output_dir,
    )

    # 3. Stage B: Fine-Tune World Model on Augmented Replay
    gate4_wm_path = Path("results/gate4/formulation_c_residual_change-weighted.pt")
    if not gate4_wm_path.exists():
        raise FileNotFoundError(f"Missing Gate 4 world model: {gate4_wm_path}")
    print(f"\nLoading Gate 4 primary world model from {gate4_wm_path}...")
    gate4_wm, _ = TrueJEPA.load_checkpoint(str(gate4_wm_path), device=device)

    updated_wm, wm_train_telemetry = train_world_model_phase3(
        base_model=gate4_wm,
        replay=cached_replay,
        updates=args.wm_updates,
        batch_size=8,
        learning_rate=1e-4,
        device=device,
        output_dir=output_dir,
    )

    # 4. Stage C & D: Evaluate & Freeze Selected World Model
    print("Loading held-out evaluation dataset...")
    eval_dataset = torch.load("data/gate3/eval_heldout_seeds.pt", weights_only=False)
    selected_wm, wm_comparison = evaluate_and_compare_world_models(
        prev_model=gate4_wm,
        updated_model=updated_wm,
        eval_dataset=eval_dataset,
        device=device,
    )

    # 5. Stage E: Train Policy via World-Model Imagination (H=4, Entropy=0.03)
    phase3_policy_model, policy_telemetry = train_phase3_policy_imagination(
        frozen_model=selected_wm,
        replay=cached_replay,
        horizon=4,
        updates=args.policy_updates,
        batch_size=8,
        actor_lr=1e-4,
        value_lr=1e-3,
        entropy_weight=0.03,  # Calibrated to maintain exploration
        reward_clip=1.0,
        device=device,
        output_dir=output_dir,
    )

    # 6. Stage F: Comprehensive 30-Seed Gameplay Benchmark in Unmodified Crafter
    eval_seeds = [args.eval_start_seed + i for i in range(args.eval_seeds_count)]
    print(f"\n{'='*70}\n[PHASE 3 — STAGE F] Real Crafter 30-Seed Evaluation ({len(eval_seeds)} seeds)\n{'='*70}")
    print(f"Seeds: {eval_seeds[:5]} ... {eval_seeds[-5:]}")

    gameplay_results: Dict[str, Any] = {
        "evaluation_seeds": eval_seeds,
        "selfplay_collection_stats": selfplay_stats,
        "world_model_telemetry": wm_train_telemetry,
        "world_model_comparison": wm_comparison,
        "policy_training_telemetry": policy_telemetry,
    }

    # Evaluation 1: Matched Random Baseline
    print("\n--- Evaluating Matched Random Baseline across 30 seeds ---")
    random_agent = RandomCrafterAgent(seed=eval_seeds[0])
    rand_eval = evaluate_policy_in_crafter(random_agent, seeds=eval_seeds, max_steps=args.max_eval_steps)
    gameplay_results["random_baseline"] = rand_eval
    print(
        f"Random Baseline | Mean Return: {rand_eval['mean_return']:.2f} (std {rand_eval['std_return']:.2f}) | "
        f"Score: {rand_eval['crafter_score']:.4f}% | Mean Length: {rand_eval['mean_length']:.1f} | Ent: {rand_eval['action_entropy']:.3f}"
    )

    # Evaluation 2: Gate 6 Provisional Champion (policy_h4)
    print("\n--- Evaluating Existing Gate 6 Champion (H=4) across 30 seeds ---")
    gate6_agent = VisualJEPAAgent(gate6_policy_model, device=str(device), stochastic=True)
    gate6_eval = evaluate_policy_in_crafter(gate6_agent, seeds=eval_seeds, max_steps=args.max_eval_steps)
    gameplay_results["gate6_champion_h4"] = gate6_eval
    print(
        f"Gate 6 H=4      | Mean Return: {gate6_eval['mean_return']:.2f} (std {gate6_eval['std_return']:.2f}) | "
        f"Score: {gate6_eval['crafter_score']:.4f}% | Mean Length: {gate6_eval['mean_length']:.1f} | Ent: {gate6_eval['action_entropy']:.3f}"
    )

    # Evaluation 3: Phase 3 Newly Trained Policy (Updated H=4)
    print("\n--- Evaluating Phase 3 Newly Trained Policy (Updated H=4) across 30 seeds ---")
    phase3_agent = VisualJEPAAgent(phase3_policy_model, device=str(device), stochastic=True)
    phase3_eval = evaluate_policy_in_crafter(phase3_agent, seeds=eval_seeds, max_steps=args.max_eval_steps)
    gameplay_results["phase3_policy"] = phase3_eval
    print(
        f"Phase 3 Policy  | Mean Return: {phase3_eval['mean_return']:.2f} (std {phase3_eval['std_return']:.2f}) | "
        f"Score: {phase3_eval['crafter_score']:.4f}% | Mean Length: {phase3_eval['mean_length']:.1f} | Ent: {phase3_eval['action_entropy']:.3f}"
    )

    # Save Metrics JSON
    metrics_path = output_dir / "phase3_experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(gameplay_results, f, indent=2)
    print(f"\nSaved Phase 3 experiment metrics to {metrics_path}")

    # Generate Comparative Visualizations
    plot_path = output_dir / "phase3_comparison_plots.png"
    generate_phase3_plots(gameplay_results, plot_path)

    # Print Summary Table
    print("\n" + "=" * 115)
    print(f"{'Model / Policy':<28} | {'Mean Return':<13} | {'Median':<8} | {'Crafter Score':<15} | {'Mean Steps':<12} | {'Entropy':<9} | {'Status'}")
    print("-" * 115)
    print(
        f"{'Matched Random Baseline':<28} | "
        f"{rand_eval['mean_return']:<5.2f} ± {rand_eval['std_return']:<5.2f} | "
        f"{rand_eval['median_return']:<8.2f} | "
        f"{rand_eval['crafter_score']:<15.4f}% | "
        f"{rand_eval['mean_length']:<12.1f} | "
        f"{rand_eval['action_entropy']:<9.3f} | BASELINE"
    )
    print(
        f"{'Gate 6 Champion (H=4)':<28} | "
        f"{gate6_eval['mean_return']:<5.2f} ± {gate6_eval['std_return']:<5.2f} | "
        f"{gate6_eval['median_return']:<8.2f} | "
        f"{gate6_eval['crafter_score']:<15.4f}% | "
        f"{gate6_eval['mean_length']:<12.1f} | "
        f"{gate6_eval['action_entropy']:<9.3f} | PRIOR BEST"
    )
    p3_status = "PROMOTED" if phase3_eval["crafter_score"] > gate6_eval["crafter_score"] and phase3_eval["crafter_score"] > rand_eval["crafter_score"] else "EVALUATED"
    print(
        f"{'Phase 3 Policy (Updated H=4)':<28} | "
        f"{phase3_eval['mean_return']:<5.2f} ± {phase3_eval['std_return']:<5.2f} | "
        f"{phase3_eval['median_return']:<8.2f} | "
        f"{phase3_eval['crafter_score']:<15.4f}% | "
        f"{phase3_eval['mean_length']:<12.1f} | "
        f"{phase3_eval['action_entropy']:<9.3f} | {p3_status}"
    )
    print("=" * 115)


if __name__ == "__main__":
    main()
