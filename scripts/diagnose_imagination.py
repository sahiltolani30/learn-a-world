#!/usr/bin/env python3
"""
Diagnostic Script D4: Multi-Step Imagination Rollout Calibration
Evaluates the error compounding of JEPA v2's latent dynamics predictor
over a 10-step planning horizon on held-out trajectories.

Computes:
- Latent MSE at each step h in 1..10
- Latent Cosine Similarity at each step h in 1..10
- Latent Norm / Variance drift across horizon
- Reward prediction error compounding along imagined trajectories

Outputs:
- results/diagnostics/imagination_calibration.json
- results/diagnostics/imagination_calibration.png
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.jepa_v2 import JEPAv2
from models.jepa_v3 import JEPAv3


def load_episodes_for_rollout(shards_dir: str, min_length: int = 12) -> List[Dict[str, np.ndarray]]:
    """Loads episodes of sufficient length for 10-step imagination tests."""
    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No .npz shards found in {shards_dir}")

    episodes = []
    for path in shard_paths:
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        episode_lengths = data.get("episode_lengths", None)

        if episode_lengths is not None and len(episode_lengths) > 0:
            idx = 0
            for ep_len in episode_lengths:
                if ep_len >= min_length:
                    ep_slice = slice(idx, idx + ep_len)
                    episodes.append({
                        "obs": obs[ep_slice],
                        "actions": actions[ep_slice],
                        "rewards": rewards[ep_slice],
                    })
                idx += ep_len
        else:
            ep_start = 0
            for i, done in enumerate(dones):
                if done or i == len(dones) - 1:
                    ep_len = (i + 1) - ep_start
                    if ep_len >= min_length:
                        episodes.append({
                            "obs": obs[ep_start:i + 1],
                            "actions": actions[ep_start:i + 1],
                            "rewards": rewards[ep_start:i + 1],
                        })
                    ep_start = i + 1

    return episodes


def evaluate_imagination(
    model: JEPAv2,
    episodes: List[Dict[str, np.ndarray]],
    horizon: int = 10,
    device: str = "cpu",
    max_samples: int = 500,
) -> Dict[str, Any]:
    """Runs 10-step imagined rollouts and compares with ground-truth target representations."""
    model.eval()
    dev = torch.device(device)

    step_mse = [[] for _ in range(horizon)]
    step_cosine = [[] for _ in range(horizon)]
    step_norm_ratio = [[] for _ in range(horizon)]
    step_reward_mae = [[] for _ in range(horizon)]

    sample_count = 0

    with torch.no_grad():
        for ep in episodes:
            obs = ep["obs"]
            actions = ep["actions"]
            rewards = ep["rewards"]
            T = len(obs)

            # Sample segments of length horizon + 1
            for start_t in range(0, T - horizon, 5):
                if sample_count >= max_samples:
                    break

                seg_obs = obs[start_t:start_t + horizon + 1]  # [H+1, 64, 64, 3]
                seg_actions = actions[start_t:start_t + horizon]  # [H]
                seg_rewards = rewards[start_t:start_t + horizon]  # [H]

                # Encode all observations to ground-truth target latents [H+1, 128]
                t_obs = torch.from_numpy(seg_obs).permute(0, 3, 1, 2).float() / 255.0
                t_obs = t_obs.to(dev)
                gt_latents = model.encode(t_obs, use_target=True)

                if isinstance(model, JEPAv3):
                    z_hat = gt_latents[0:1]
                    h_hat = torch.zeros((1, model.recurrent_dim), device=dev)
                    m_hat = torch.zeros((1, 22), device=dev)

                    for h in range(horizon):
                        a_h = torch.tensor([seg_actions[h]], dtype=torch.long, device=dev)
                        next_z, next_h, next_s = model.predict_next_step(z_hat, h_hat, a_h)
                        gt_z = gt_latents[h + 1:h + 2]

                        mse = F.mse_loss(next_z, gt_z).item()
                        cos = F.cosine_similarity(next_z, gt_z).item()
                        norm_ratio = (torch.norm(next_z) / (torch.norm(gt_z) + 1e-6)).item()

                        r_hat, decomp = model.predict_reward(next_s, a_h, m_hat, return_decomposition=True)
                        r_mae = abs(float(r_hat.item()) - float(seg_rewards[h]))
                        p_event = decomp["event_probs"]
                        m_hat = torch.where(p_event > 0.5, torch.tensor(1.0, device=dev), m_hat)

                        step_mse[h].append(mse)
                        step_cosine[h].append(cos)
                        step_norm_ratio[h].append(norm_ratio)
                        step_reward_mae[h].append(r_mae)

                        z_hat = next_z
                        h_hat = next_h
                else:
                    # Imagined trajectory starting from z_0 for JEPAv2
                    z_hat = gt_latents[0:1]  # [1, 128]
                    m_hat = torch.zeros((1, 22), device=dev)

                    for h in range(horizon):
                        a_h = torch.tensor([seg_actions[h]], dtype=torch.long, device=dev)
                        # Next latent imagined
                        z_hat = model.predict_next(z_hat, a_h)
                        gt_z = gt_latents[h + 1:h + 2]

                        # Metrics
                        mse = F.mse_loss(z_hat, gt_z).item()
                        cos = F.cosine_similarity(z_hat, gt_z).item()
                        norm_ratio = (torch.norm(z_hat) / (torch.norm(gt_z) + 1e-6)).item()

                        # Reward prediction along imagined path
                        r_hat, decomp = model.predict_reward(z_hat, a_h, m_hat, return_decomposition=True)
                        r_mae = abs(float(r_hat.item()) - float(seg_rewards[h]))
                        p_event = decomp["event_probs"]
                        m_hat = torch.where(p_event > 0.5, torch.tensor(1.0, device=dev), m_hat)

                        step_mse[h].append(mse)
                        step_cosine[h].append(cos)
                        step_norm_ratio[h].append(norm_ratio)
                        step_reward_mae[h].append(r_mae)

                sample_count += 1
            if sample_count >= max_samples:
                break

    mean_mse_by_step = [float(np.mean(m)) for m in step_mse]
    mean_cosine_by_step = [float(np.mean(c)) for c in step_cosine]
    mean_norm_ratio_by_step = [float(np.mean(n)) for n in step_norm_ratio]
    mean_reward_mae_by_step = [float(np.mean(r)) for r in step_reward_mae]

    return {
        "num_sequences_evaluated": sample_count,
        "horizon": horizon,
        "mean_mse_by_step": mean_mse_by_step,
        "mean_cosine_by_step": mean_cosine_by_step,
        "mean_norm_ratio_by_step": mean_norm_ratio_by_step,
        "mean_reward_mae_by_step": mean_reward_mae_by_step,
        "step_1_mse": mean_mse_by_step[0],
        "step_5_mse": mean_mse_by_step[4],
        "step_10_mse": mean_mse_by_step[9],
        "step_1_cosine": mean_cosine_by_step[0],
        "step_5_cosine": mean_cosine_by_step[4],
        "step_10_cosine": mean_cosine_by_step[9],
        "error_growth_ratio_10_to_1": float(mean_mse_by_step[9] / max(mean_mse_by_step[0], 1e-6)),
    }


def plot_imagination(results: Dict[str, Any], output_png: str):
    """Plots compound error growth over 10-step horizon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    horizon = results["horizon"]
    steps = np.arange(1, horizon + 1)

    # Plot 1: Latent MSE
    ax1.plot(steps, results["mean_mse_by_step"], "o-", color="#d62728", linewidth=2.5, markersize=6)
    ax1.set_xlabel("Horizon Step (h)", fontsize=11)
    ax1.set_ylabel("Latent MSE ||ẑ_h - z*_h||²", fontsize=11)
    ax1.set_title(
        f"Compound Latent Prediction Error\n(Step 1 MSE: {results['step_1_mse']:.4f} → Step 10: {results['step_10_mse']:.4f})",
        fontsize=12,
        fontweight="bold"
    )
    ax1.set_xticks(steps)

    # Plot 2: Cosine Similarity
    ax2.plot(steps, results["mean_cosine_by_step"], "s-", color="#1f77b4", linewidth=2.5, markersize=6)
    ax2.set_xlabel("Horizon Step (h)", fontsize=11)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_ylim(-0.1, 1.05)
    ax2.set_title(
        f"Latent Alignment Decay\n(Step 1: {results['step_1_cosine']:.3f} → Step 10: {results['step_10_cosine']:.3f})",
        fontsize=12,
        fontweight="bold"
    )
    ax2.set_xticks(steps)

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"[Imagination] Saved plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-step imagination rollout calibration.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--shards_dir", type=str, default="data/sequences")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/imagination_calibration.json")
    parser.add_argument("--output_png", type=str, default="results/diagnostics/imagination_calibration.png")
    args = parser.parse_args()

    print(f"[Imagination] Loading model from {args.checkpoint}...")
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt_raw, dict) and ckpt_raw.get("version") == "v3":
        model = JEPAv3()
        model.load_checkpoint(args.checkpoint, device=args.device)
    else:
        model = JEPAv2.from_v1_checkpoint(args.checkpoint, device=args.device)

    print(f"[Imagination] Loading episodes from {args.shards_dir}...")
    episodes = load_episodes_for_rollout(args.shards_dir, min_length=args.horizon + 2)
    print(f"[Imagination] Found {len(episodes)} eligible episodes.")

    print(f"[Imagination] Evaluating {args.horizon}-step rollouts...")
    results = evaluate_imagination(
        model=model,
        episodes=episodes,
        horizon=args.horizon,
        device=args.device,
        max_samples=args.max_samples,
    )

    print("\n--- Multi-Step Imagination Diagnostics Summary ---")
    print(f"Sequences Evaluated: {results['num_sequences_evaluated']}")
    print(f"Step 1 Latent MSE: {results['step_1_mse']:.4f} | Cosine Sim: {results['step_1_cosine']:.4f}")
    print(f"Step 5 Latent MSE: {results['step_5_mse']:.4f} | Cosine Sim: {results['step_5_cosine']:.4f}")
    print(f"Step 10 Latent MSE: {results['step_10_mse']:.4f} | Cosine Sim: {results['step_10_cosine']:.4f}")
    print(f"Error Growth Ratio (Step 10 / Step 1): {results['error_growth_ratio_10_to_1']:.2f}x")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Imagination] Saved metrics to {args.output_json}")

    plot_imagination(results, args.output_png)


if __name__ == "__main__":
    main()
