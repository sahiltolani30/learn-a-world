#!/usr/bin/env python3
"""
Phase 1 Execution Script: Value Critic Fix (R1)
Re-trains the Q-ensemble using:
1. Multi-step n-step Bellman returns (n=10) with exact per-sequence done masking.
2. Clipped Double-Q (minimum across target ensemble heads to eliminate overestimation).
3. DreamerV3-style percentile return normalization.
4. Target network Polyak averaging (tau=0.005).
5. Continuous tracking of Q-vs-return calibration correlation r.

Outputs:
- checkpoints/jepa_crafter_v3_step1_value.pt
- results/step1_screening.json
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.heads import ReturnNormalizer
from models.jepa_v2 import JEPAv2
from training.sequence_buffer import EpisodeBuffer


def compute_n_step_bellman_targets(
    rewards: torch.Tensor,        # [B, T]
    dones: torch.Tensor,          # [B, T]
    target_q_vals: torch.Tensor,  # [B, T] (Clipped Double-Q min)
    gamma: float = 0.99,
    n_step: int = 10,
) -> torch.Tensor:
    """
    Computes rigorous n-step Bellman targets with vector-safe per-sequence termination:
    G_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * Q_target(s_{t+n}, pi(s_{t+n})) * alive_mask
    """
    B, T = rewards.shape
    targets = torch.zeros(B, T, device=rewards.device, dtype=torch.float32)

    for t in range(T):
        cum_reward = torch.zeros(B, device=rewards.device)
        alive_mask = torch.ones(B, device=rewards.device)
        discount = 1.0

        for k in range(n_step):
            idx = t + k
            if idx >= T:
                break
            cum_reward = cum_reward + discount * alive_mask * rewards[:, idx]
            alive_mask = alive_mask * (1.0 - dones[:, idx].float())
            discount *= gamma

        # Bootstrap terminal value if sequence is still active after n steps
        terminal_idx = t + n_step
        if terminal_idx < T:
            cum_reward = cum_reward + discount * alive_mask * target_q_vals[:, terminal_idx]

        targets[:, t] = cum_reward

    return targets


def load_buffer_from_shards(shards_dir: str, capacity: int = 200000) -> EpisodeBuffer:
    """Loads all .npz shards from shards_dir into an EpisodeBuffer."""
    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {shards_dir}")

    buffer = EpisodeBuffer(capacity_transitions=capacity)
    total_loaded = 0

    for path in shard_paths:
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        events = data["events"]
        mask_before = data["mask_before"]
        lengths = data.get("episode_lengths", None)

        if lengths is not None and len(lengths) > 0:
            idx = 0
            for ep_len in lengths:
                sl = slice(idx, idx + ep_len)
                buffer.add_episode(
                    obs=obs[sl],
                    actions=actions[sl],
                    rewards=rewards[sl],
                    dones=dones[sl],
                    events=events[sl],
                    mask_before=mask_before[sl],
                )
                idx += ep_len
                total_loaded += ep_len
        else:
            ep_start = 0
            for i, done in enumerate(dones):
                if done or i == len(dones) - 1:
                    sl = slice(ep_start, i + 1)
                    buffer.add_episode(
                        obs=obs[sl],
                        actions=actions[sl],
                        rewards=rewards[sl],
                        dones=dones[sl],
                        events=events[sl],
                        mask_before=mask_before[sl],
                    )
                    total_loaded += (i + 1 - ep_start)
                    ep_start = i + 1

    print(f"[Buffer] Loaded {buffer.num_episodes} episodes ({buffer.total_transitions} transitions) from {len(shard_paths)} shards.")
    return buffer


def evaluate_critic_calibration(
    model: JEPAv2,
    val_buffer: EpisodeBuffer,
    normalizer: Optional[ReturnNormalizer] = None,
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
    num_batches: int = 8,
    batch_size: int = 32,
    seq_len: int = 16,
) -> Dict[str, float]:
    """Computes empirical calibration between predicted Q and true realized returns."""
    model.eval()
    all_q_pred = []
    all_realized = []
    all_q_std = []

    with torch.no_grad():
        for _ in range(num_batches):
            batch = val_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len, device=device)
            obs = batch["obs"]          # [B, T, 3, 64, 64]
            actions = batch["actions"]  # [B, T]
            rewards = batch["rewards"]  # [B, T]
            B, T = actions.shape

            obs_flat = obs.view(B * T, 3, 64, 64)
            z_flat = model.encode(obs_flat, use_target=True)
            z = z_flat.view(B, T, -1)

            q_pred = model.q_mean(z, actions)  # [B, T]
            q_std = model.q_std(z, actions)    # [B, T]

            if normalizer is not None and normalizer.initialized:
                q_pred = normalizer.denormalize(q_pred)

            # Compute discounted realized returns along the sequence
            realized = torch.zeros_like(rewards)
            running = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                running = rewards[:, t] + gamma * running
                realized[:, t] = running

            all_q_pred.extend(q_pred.cpu().numpy().flatten().tolist())
            all_realized.extend(realized.cpu().numpy().flatten().tolist())
            all_q_std.extend(q_std.cpu().numpy().flatten().tolist())

    q_arr = np.array(all_q_pred, dtype=np.float32)
    ret_arr = np.array(all_realized, dtype=np.float32)
    std_arr = np.array(all_q_std, dtype=np.float32)

    corr, _ = pearsonr(q_arr, ret_arr)
    mae = float(np.mean(np.abs(q_arr - ret_arr)))

    return {
        "pearson_r": float(corr),
        "mae": mae,
        "q_pred_mean": float(np.mean(q_arr)),
        "q_pred_std": float(np.std(q_arr)),
        "realized_return_mean": float(np.mean(ret_arr)),
        "realized_return_std": float(np.std(ret_arr)),
        "mean_disagreement": float(np.mean(std_arr)),
    }


def train_value_fix(
    checkpoint_path: str = "checkpoints/jepa_crafter_v2_best.pt",
    shards_dir: str = "data/sequences",
    total_steps: int = 1500,
    batch_size: int = 32,
    seq_len: int = 16,
    n_step: int = 10,
    gamma: float = 0.99,
    lr: float = 3e-4,
    target_ema: float = 0.995,
    use_normalization: bool = True,
    device_name: str = "cpu",
    output_checkpoint: str = "checkpoints/jepa_crafter_v3_step1_value.pt",
    output_json: str = "results/step1_screening.json",
) -> Dict[str, Any]:
    """Executes Phase 1 Value Critic Retraining."""
    dev = torch.device(device_name)
    print(f"\n=======================================================")
    print(f"       PHASE 1: VALUE CRITIC RETRAINING (R1 FIX)       ")
    print(f"=======================================================")
    print(f"Checkpoint Base: {checkpoint_path}")
    print(f"Steps: {total_steps} | Batch: {batch_size} | SeqLen: {seq_len} | n-step: {n_step}")
    print(f"Device: {dev} | LR: {lr} | Target Polyak EMA: {target_ema}")

    # 1. Load buffer
    buffer = load_buffer_from_shards(shards_dir)
    train_buf, val_buf = buffer.split_train_val(val_ratio=0.15)
    if val_buf.num_episodes == 0:
        val_buf = train_buf

    # 2. Load model
    model = JEPAv2.from_v1_checkpoint(checkpoint_path, device=dev, freeze_base=True)
    model.to(dev)

    # 3. Setup optimizer for Q-ensemble and task heads
    critic_params = list(model.q_ensemble.parameters()) + list(model.event_head.parameters()) + list(model.health_head.parameters())
    optimizer = optim.AdamW(critic_params, lr=lr, weight_decay=1e-4)

    # 4. Return normalizer
    normalizer = ReturnNormalizer(low_percentile=5.0, high_percentile=95.0, target_max=12.0) if use_normalization else None

    # Initial calibration
    init_calib = evaluate_critic_calibration(model, val_buf, normalizer, gamma=gamma, device=dev)
    print(f"\n[Step 0 / Baseline] Pearson r: {init_calib['pearson_r']:.4f} | Q Std: {init_calib['q_pred_std']:.4f} | MAE: {init_calib['mae']:.4f}")

    history = {
        "step": [],
        "loss_q": [],
        "loss_event": [],
        "loss_health": [],
        "pearson_r": [],
        "q_pred_std": [],
        "mean_disagreement": [],
    }

    t0 = time.time()
    model.train()

    for step in range(1, total_steps + 1):
        batch = train_buf.sample_sequences(batch_size=batch_size, seq_len=seq_len, mode="event_balanced", device=dev)
        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]
        B, T = actions.shape

        with torch.no_grad():
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_flat = model.encode(obs_flat, use_target=True)
            z = z_flat.view(B, T, -1)

            # Sample bootstrap action from prior
            bootstrap_actions = model.sample_prior_action(z, mask, temperature=1.0)

            # Clipped Double-Q: MINIMUM across target ensemble heads
            target_q_min = model.q_min(z, bootstrap_actions, use_target=True)

            # Compute rigorous n-step Bellman targets
            n_step_targets = compute_n_step_bellman_targets(
                rewards=rewards,
                dones=dones,
                target_q_vals=target_q_min,
                gamma=gamma,
                n_step=n_step,
            )

            # Update normalizer with empirical Bellman targets
            if normalizer is not None:
                normalizer.update(n_step_targets)
                q_fit_targets = normalizer.normalize(n_step_targets)
            else:
                q_fit_targets = n_step_targets

        # Q-ensemble forward and loss
        q_logits = model.q_ensemble(z, actions)  # [K, B, T, 51]
        loss_q = model.q_ensemble.compute_loss(q_logits, q_fit_targets)

        # Auxiliary event and health losses to maintain head accuracy
        event_logits = model.event_head(z, actions, mask)
        loss_event = model.event_head.compute_loss(event_logits, events)

        event_counts = events.sum(dim=-1)
        health_tgt = (rewards - event_counts).clamp(-1.0, 1.0)
        health_pred = model.health_head(z, actions)
        loss_health = model.health_head.compute_loss(health_pred, health_tgt)

        total_loss = 1.0 * loss_q + 0.5 * loss_event + 0.2 * loss_health

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=1.0)
        optimizer.step()

        # Polyak update of target Q ensemble
        model.update_target_q(decay=target_ema)

        if step % 50 == 0 or step == total_steps:
            calib = evaluate_critic_calibration(model, val_buf, normalizer, gamma=gamma, device=dev)
            elapsed = time.time() - t0
            print(
                f"Step {step:4d}/{total_steps} ({elapsed:.1f}s) | "
                f"Loss Q: {loss_q.item():.4f} | "
                f"Pearson r: {calib['pearson_r']:.4f} | "
                f"Q Std: {calib['q_pred_std']:.4f} | "
                f"Disagreement: {calib['mean_disagreement']:.5f}"
            )

            history["step"].append(step)
            history["loss_q"].append(float(loss_q.item()))
            history["loss_event"].append(float(loss_event.item()))
            history["loss_health"].append(float(loss_health.item()))
            history["pearson_r"].append(calib["pearson_r"])
            history["q_pred_std"].append(calib["q_pred_std"])
            history["mean_disagreement"].append(calib["mean_disagreement"])

        # Periodic checkpoint save every 1,000 steps (Drive resilience)
        if step % 1000 == 0:
            Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(output_checkpoint, step=step)


    # Final calibration audit
    final_calib = evaluate_critic_calibration(model, val_buf, normalizer, gamma=gamma, device=dev, num_batches=16)

    # Save checkpoint
    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(output_checkpoint, step=total_steps)
    print(f"\n[Checkpoint] Saved value-fixed model to {output_checkpoint}")

    # Evaluate promotion gate
    passed_gate = bool(final_calib["pearson_r"] > 0.0 and final_calib["q_pred_std"] > 0.05)

    results = {
        "step": total_steps,
        "n_step": n_step,
        "gamma": gamma,
        "use_normalization": use_normalization,
        "initial_calibration": init_calib,
        "final_calibration": final_calib,
        "history": history,
        "passed_gate": passed_gate,
        "gate_conditions": {
            "pearson_r_positive": bool(final_calib["pearson_r"] > 0.0),
            "q_variance_healthy": bool(final_calib["q_pred_std"] > 0.05),
            "disagreement_active": bool(final_calib["mean_disagreement"] > 0.001),
        },
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved screening metrics to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Value Critic Retraining (R1 Fix).")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--shards_dir", type=str, default="data/sequences")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_step", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_ckpt", type=str, default="checkpoints/jepa_crafter_v3_step1_value.pt")
    parser.add_argument("--out_json", type=str, default="results/step1_screening.json")
    args = parser.parse_args()

    train_value_fix(
        checkpoint_path=args.checkpoint,
        shards_dir=args.shards_dir,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_step=args.n_step,
        gamma=args.gamma,
        lr=args.lr,
        device_name=args.device,
        output_checkpoint=args.out_ckpt,
        output_json=args.out_json,
    )


if __name__ == "__main__":
    main()
