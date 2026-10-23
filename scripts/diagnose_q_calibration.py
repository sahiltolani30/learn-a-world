#!/usr/bin/env python3
"""
Diagnostic Script D1: Q-Value Calibration Analysis
Loads trained JEPA v2 checkpoint and evaluates the correlation between
predicted Q-values and actual realized discounted returns on held-out trajectories.

Outputs:
- results/diagnostics/q_calibration.json
- results/diagnostics/q_calibration.png
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.jepa_v2 import JEPAv2
from models.jepa_v3 import JEPAv3


def compute_discounted_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Computes discounted return G_t = sum_{k=t}^{T-1} gamma^(k-t) * r_k."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def load_episodes_from_shards(shards_dir: str) -> List[Dict[str, np.ndarray]]:
    """Loads all episodes from .npz shards in shards_dir."""
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
                ep_slice = slice(idx, idx + ep_len)
                episodes.append({
                    "obs": obs[ep_slice],
                    "actions": actions[ep_slice],
                    "rewards": rewards[ep_slice],
                    "dones": dones[ep_slice],
                })
                idx += ep_len
        else:
            # Fallback split by dones
            ep_start = 0
            for i, done in enumerate(dones):
                if done or i == len(dones) - 1:
                    episodes.append({
                        "obs": obs[ep_start:i + 1],
                        "actions": actions[ep_start:i + 1],
                        "rewards": rewards[ep_start:i + 1],
                        "dones": dones[ep_start:i + 1],
                    })
                    ep_start = i + 1

    return episodes


def evaluate_q_calibration(
    model: JEPAv2,
    episodes: List[Dict[str, np.ndarray]],
    gamma: float = 0.99,
    device: str = "cpu",
    batch_size: int = 128,
) -> Dict[str, any]:
    """Evaluates Q-predicted vs realized return across all episode steps."""
    model.eval()
    dev = torch.device(device)

    all_q_pred = []
    all_q_std = []
    all_returns = []
    all_step_indices = []

    with torch.no_grad():
        for ep_idx, ep in enumerate(episodes):
            obs = ep["obs"]
            actions = ep["actions"]
            rewards = ep["rewards"]

            if len(rewards) == 0:
                continue

            returns = compute_discounted_returns(rewards, gamma=gamma)
            T = len(returns)

            if isinstance(model, JEPAv3):
                t_obs = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255.0
                t_obs = t_obs.to(dev)
                t_act = torch.from_numpy(actions).long().to(dev)

                z = model.encode(t_obs, use_target=True)
                h_seq = model.recurrent_core.unroll_sequence(z.unsqueeze(0), t_act.unsqueeze(0)).squeeze(0)
                s = torch.cat([z, h_seq], dim=-1)
                q_pred = model.q_mean(s, t_act)
                if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                    q_pred = model.return_normalizer.denormalize(q_pred)
                q_std = model.q_std(s, t_act)

                all_q_pred.extend(q_pred.cpu().numpy().tolist())
                all_q_std.extend(q_std.cpu().numpy().tolist())
                all_returns.extend(returns.tolist())
                all_step_indices.extend(list(range(T)))
            else:
                # Process in batches for JEPAv2
                for i in range(0, T, batch_size):
                    b_obs = obs[i:i + batch_size]
                    b_actions = actions[i:i + batch_size]

                    # Convert to tensor [B, C, H, W]
                    t_obs = torch.from_numpy(b_obs).permute(0, 3, 1, 2).float() / 255.0
                    t_obs = t_obs.to(dev)
                    t_act = torch.from_numpy(b_actions).long().to(dev)

                    z = model.encode(t_obs, use_target=True)
                    q_pred = model.q_mean(z, t_act).cpu().numpy()
                    q_std = model.q_std(z, t_act).cpu().numpy()

                    all_q_pred.extend(q_pred.tolist())
                    all_q_std.extend(q_std.tolist())
                    all_returns.extend(returns[i:i + batch_size].tolist())
                    all_step_indices.extend(list(range(i, min(i + batch_size, T))))

    q_pred_arr = np.array(all_q_pred, dtype=np.float32)
    q_std_arr = np.array(all_q_std, dtype=np.float32)
    returns_arr = np.array(all_returns, dtype=np.float32)

    # Compute correlation metrics
    p_corr, p_val = pearsonr(q_pred_arr, returns_arr)
    s_corr, s_val = spearmanr(q_pred_arr, returns_arr)
    mae = float(np.mean(np.abs(q_pred_arr - returns_arr)))
    rmse = float(np.sqrt(np.mean((q_pred_arr - returns_arr) ** 2)))

    metrics = {
        "num_episodes": len(episodes),
        "total_transitions": len(q_pred_arr),
        "pearson_r": float(p_corr),
        "pearson_p_value": float(p_val),
        "spearman_rho": float(s_corr),
        "spearman_p_value": float(s_val),
        "mae": mae,
        "rmse": rmse,
        "q_pred_mean": float(np.mean(q_pred_arr)),
        "q_pred_std": float(np.std(q_pred_arr)),
        "q_pred_min": float(np.min(q_pred_arr)),
        "q_pred_max": float(np.max(q_pred_arr)),
        "realized_return_mean": float(np.mean(returns_arr)),
        "realized_return_std": float(np.std(returns_arr)),
        "realized_return_min": float(np.min(returns_arr)),
        "realized_return_max": float(np.max(returns_arr)),
        "mean_ensemble_disagreement": float(np.mean(q_std_arr)),
        "max_ensemble_disagreement": float(np.max(q_std_arr)),
    }

    return metrics, q_pred_arr, returns_arr, q_std_arr


def plot_calibration(
    q_pred: np.ndarray,
    returns: np.ndarray,
    metrics: Dict[str, any],
    output_png: str,
):
    """Generates a professional scatter plot comparing Q_predicted vs Realized Return."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Downsample for cleaner scatter if large
    if len(q_pred) > 5000:
        idx = np.random.choice(len(q_pred), 5000, replace=False)
        x_plot = q_pred[idx]
        y_plot = returns[idx]
    else:
        x_plot = q_pred
        y_plot = returns

    plt.scatter(x_plot, y_plot, alpha=0.35, edgecolors="none", c="#1f77b4", s=20, label="Transitions")

    # Identity reference line
    lim_min = min(float(np.min(q_pred)), float(np.min(returns)))
    lim_max = max(float(np.max(q_pred)), float(np.max(returns)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.7, label="Ideal Calibration (y=x)")

    # Linear fit
    if np.std(x_plot) > 1e-6:
        m, b = np.polyfit(x_plot, y_plot, 1)
        plt.plot(x_plot, m * x_plot + b, "r-", linewidth=2, label=f"Fit (slope={m:.2f})")

    plt.title(
        f"Q-Value Calibration (Pearson r = {metrics['pearson_r']:.3f}, MAE = {metrics['mae']:.3f})\n"
        f"Mean Pred Q = {metrics['q_pred_mean']:.3f} | Mean Return = {metrics['realized_return_mean']:.3f}",
        fontsize=12,
        fontweight="bold"
    )
    plt.xlabel("Predicted Q-Value $\\hat{Q}(s_t, a_t)$", fontsize=11)
    plt.ylabel("Realized Return $G_t = \\sum \\gamma^k r_{t+k}$", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"[Q-Calibration] Saved scatter plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q-value calibration on held-out episodes.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--shards_dir", type=str, default="data/sequences")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/q_calibration.json")
    parser.add_argument("--output_png", type=str, default="results/diagnostics/q_calibration.png")
    args = parser.parse_args()

    print(f"[Q-Calibration] Loading model from {args.checkpoint}...")
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt_raw, dict) and ckpt_raw.get("version") == "v3":
        model = JEPAv3()
        model.load_checkpoint(args.checkpoint, device=args.device)
    else:
        model = JEPAv2.from_v1_checkpoint(args.checkpoint, device=args.device)
    model.to(args.device)

    print(f"[Q-Calibration] Loading episodes from {args.shards_dir}...")
    episodes = load_episodes_from_shards(args.shards_dir)
    print(f"[Q-Calibration] Loaded {len(episodes)} episodes.")

    print("[Q-Calibration] Running evaluation...")
    metrics, q_pred, returns, q_std = evaluate_q_calibration(
        model=model,
        episodes=episodes,
        gamma=args.gamma,
        device=args.device,
    )

    print("\n--- Q-Calibration Diagnostic Results ---")
    print(f"Total Transitions: {metrics['total_transitions']}")
    print(f"Pearson Correlation (r): {metrics['pearson_r']:.4f} (p={metrics['pearson_p_value']:.2e})")
    print(f"Spearman Correlation (rho): {metrics['spearman_rho']:.4f}")
    print(f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}")
    print(f"Q Predicted: Mean={metrics['q_pred_mean']:.4f}, Std={metrics['q_pred_std']:.4f}")
    print(f"Realized Return: Mean={metrics['realized_return_mean']:.4f}, Std={metrics['realized_return_std']:.4f}")
    print(f"Ensemble Disagreement (std): Mean={metrics['mean_ensemble_disagreement']:.6f}, Max={metrics['max_ensemble_disagreement']:.6f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Q-Calibration] Saved metrics to {args.output_json}")

    plot_calibration(q_pred, returns, metrics, args.output_png)


if __name__ == "__main__":
    main()
