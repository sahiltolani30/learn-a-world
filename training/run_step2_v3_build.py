#!/usr/bin/env python3
"""
Phase 2 Execution Script: v3 Recurrent Architecture & Task-Shaped Training (R2 + R5).
Trains JEPA v3 end-to-end with:
1. GRU RecurrentCore providing temporal memory across night cycles and occlusions.
2. Composite state s_t = [z_t, h_t] feeding all task heads and value critic.
3. Task-shaped joint backpropagation: task losses shape the ConvEncoder from step 1.
4. Multi-step recurrent imagination rollout loss.
5. Tracking 10-step latent prediction error and tier-1 head PR-AUCs.

Outputs:
- checkpoints/jepa_crafter_v3_phase2.pt
- results/step2_screening.json
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
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from training.run_step1_value_fix import compute_n_step_bellman_targets, load_buffer_from_shards
from training.sequence_buffer import EpisodeBuffer


def evaluate_v3_metrics(
    model: JEPAv3,
    val_buffer: EpisodeBuffer,
    horizon: int = 10,
    device: torch.device = torch.device("cpu"),
    num_batches: int = 6,
    batch_size: int = 16,
    seq_len: int = 16,
) -> Dict[str, Any]:
    """Computes 10-step rollout error and event head PR-AUCs for v3 validation."""
    model.eval()

    all_event_targets = []
    all_event_probs = []
    rollout_mses = []

    with torch.no_grad():
        for _ in range(num_batches):
            batch = val_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len, device=device)
            obs = batch["obs"]          # [B, T, 3, 64, 64]
            actions = batch["actions"]  # [B, T]
            events = batch["events"]    # [B, T, 22]
            mask = batch["mask_before"] # [B, T, 22]
            dones = batch["dones"]      # [B, T]
            B, T = actions.shape

            obs_flat = obs.view(B * T, 3, 64, 64)
            z_seq = model.encode(obs_flat, use_target=False).view(B, T, -1)
            z_target_seq = model.encode(obs_flat, use_target=True).view(B, T, -1)

            # Recurrent state unroll
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)

            # Event head probabilities
            probs = model.event_head.predict_proba(s_seq, actions, mask)
            all_event_targets.append(events.cpu().numpy().reshape(-1, 22))
            all_event_probs.append(probs.cpu().numpy().reshape(-1, 22))

            # 10-step imagined rollout test from step t=0
            if T >= horizon + 1:
                z_hat = z_seq[:, 0]
                h_hat = torch.zeros(B, model.recurrent_dim, device=device)
                for k in range(horizon):
                    a_k = actions[:, k]
                    z_hat, h_hat, _ = model.predict_next_step(z_hat, h_hat, a_k)
                mse_10 = F.mse_loss(z_hat, z_target_seq[:, horizon]).item()
                rollout_mses.append(mse_10)

    # Compute PR-AUCs
    y_true = np.concatenate(all_event_targets, axis=0)
    y_pred = np.concatenate(all_event_probs, axis=0)

    head_pr_aucs = {}
    for i, name in enumerate(CRAFTER_ACHIEVEMENTS):
        pos = int(np.sum(y_true[:, i]))
        if pos > 0:
            pr_auc = float(average_precision_score(y_true[:, i], y_pred[:, i]))
        else:
            pr_auc = 0.0
        head_pr_aucs[name] = {"pr_auc": pr_auc, "pos_count": pos}

    macro_pr_auc = float(np.mean([v["pr_auc"] for v in head_pr_aucs.values() if v["pos_count"] > 0]))
    mean_10step_mse = float(np.mean(rollout_mses)) if rollout_mses else 0.5

    return {
        "10_step_rollout_mse": mean_10step_mse,
        "macro_pr_auc": macro_pr_auc,
        "wood_pr_auc": head_pr_aucs["collect_wood"]["pr_auc"],
        "sapling_pr_auc": head_pr_aucs["collect_sapling"]["pr_auc"],
        "per_head": head_pr_aucs,
    }


def train_v3_recurrent(
    checkpoint_base: str = "checkpoints/jepa_crafter_v2_best.pt",
    shards_dir: str = "data/sequences",
    total_steps: int = 2000,
    batch_size: int = 16,
    seq_len: int = 16,
    recurrent_dim: int = 256,
    n_step: int = 10,
    gamma: float = 0.99,
    lr: float = 2e-4,
    device_name: str = "cpu",
    output_checkpoint: str = "checkpoints/jepa_crafter_v3_phase2.pt",
    output_json: str = "results/step2_screening.json",
) -> Dict[str, Any]:
    """Executes Phase 2 JEPA v3 Joint Recurrent Training."""
    dev = torch.device(device_name)
    print(f"\n=================================================================")
    print(f"       PHASE 2: JEPA v3 RECURRENT & TASK-SHAPED TRAINING       ")
    print(f"=================================================================")
    print(f"Base Checkpoint: {checkpoint_base}")
    print(f"Steps: {total_steps} | Batch: {batch_size} | SeqLen: {seq_len} | RecurrentDim: {recurrent_dim}")
    print(f"Device: {dev} | LR: {lr} (End-to-End Task Shaping)")

    # 1. Load buffer
    buffer = load_buffer_from_shards(shards_dir)
    train_buf, val_buf = buffer.split_train_val(val_ratio=0.15)
    if val_buf.num_episodes == 0:
        val_buf = train_buf

    # 2. Build JEPA v3 model
    model = JEPAv3(
        latent_dim=128,
        recurrent_dim=recurrent_dim,
        num_actions=17,
        action_emb_dim=32,
        hidden_dim=256,
        q_heads=3,
        q_bins=51,
        q_max=12.0,
    )
    model.to(dev)

    # Initialize encoder weights from previous checkpoint
    if Path(checkpoint_base).exists():
        model.load_checkpoint(checkpoint_base, device=dev)

    # 3. Setup joint optimizer: ALL parameters are active (task-shaped representation)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Initial baseline audit
    init_metrics = evaluate_v3_metrics(model, val_buf, horizon=10, device=dev)
    print(f"\n[Baseline / Step 0] 10-Step MSE: {init_metrics['10_step_rollout_mse']:.4f} | Wood PR-AUC: {init_metrics['wood_pr_auc']:.4f} | Macro PR-AUC: {init_metrics['macro_pr_auc']:.4f}")

    history = {
        "step": [],
        "total_loss": [],
        "loss_dyn": [],
        "loss_event": [],
        "loss_q": [],
        "10_step_rollout_mse": [],
        "wood_pr_auc": [],
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

        obs_flat = obs.view(B * T, 3, 64, 64)

        # 1. Online visual encoding (Gradients FLOW into ConvEncoder)
        z_seq = model.encoder(obs_flat).view(B, T, -1)

        # 2. Target visual encoding (stop-gradient EMA targets)
        with torch.no_grad():
            z_target_seq = model.target_encoder(obs_flat).view(B, T, -1)

        # 3. RecurrentCore temporal unroll
        h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
        s_seq = torch.cat([z_seq, h_seq], dim=-1)  # Composite state [B, T, 384]

        # 4. Multi-step Recurrent Latent Dynamics Loss
        K_unroll = min(4, T - 1)
        loss_dyn = torch.tensor(0.0, device=dev)
        z_hat = z_seq[:, 0]
        h_hat = torch.zeros(B, recurrent_dim, device=dev)

        for k in range(1, K_unroll + 1):
            a_prev = actions[:, k - 1]
            z_hat, h_hat, _ = model.predict_next_step(z_hat, h_hat, a_prev)
            loss_dyn = loss_dyn + F.mse_loss(z_hat, z_target_seq[:, k])
        loss_dyn = loss_dyn / K_unroll

        # Anti-collapse variance loss
        z_flat = z_seq.view(B * T, -1)
        std_z = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        loss_var = torch.mean(F.relu(1.0 - std_z))

        # 5. Task Losses on Composite State s_seq
        # Event Head
        event_logits = model.event_head(s_seq, actions, mask)
        loss_event = model.event_head.compute_loss(event_logits, events)

        # Health Head
        event_counts = events.sum(dim=-1)
        health_tgt = (rewards - event_counts).clamp(-1.0, 1.0)
        health_pred = model.health_head(s_seq, actions)
        loss_health = model.health_head.compute_loss(health_pred, health_tgt)

        # Policy Prior Head
        prior_logits = model.policy_prior(s_seq, mask)
        loss_prior = model.policy_prior.compute_loss(prior_logits, actions)

        # 6. Clipped Double-Q with n-step Bellman Targets
        with torch.no_grad():
            bootstrap_acts = model.sample_prior_action(s_seq, mask, temperature=1.0)
            target_q_min = model.q_min(s_seq, bootstrap_acts, use_target=True)
            n_step_targets = compute_n_step_bellman_targets(
                rewards=rewards,
                dones=dones,
                target_q_vals=target_q_min,
                gamma=gamma,
                n_step=n_step,
            )
            model.return_normalizer.update(n_step_targets)
            q_fit_targets = model.return_normalizer.normalize(n_step_targets)

        q_logits = model.q_ensemble(s_seq, actions)
        loss_q = model.q_ensemble.compute_loss(q_logits, q_fit_targets)

        # 7. Total Joint Task-Shaped Loss
        total_loss = (
            1.0 * loss_event +
            0.5 * loss_health +
            1.0 * loss_q +
            0.5 * loss_prior +
            0.5 * loss_dyn +
            0.1 * loss_var
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update EMA networks
        model.update_target_networks(decay_encoder=0.99, decay_q=0.995)

        if step % 50 == 0 or step == total_steps:
            val_metrics = evaluate_v3_metrics(model, val_buf, horizon=10, device=dev)
            elapsed = time.time() - t0
            print(
                f"Step {step:4d}/{total_steps} ({elapsed:.1f}s) | "
                f"Loss: {total_loss.item():.4f} (Event: {loss_event.item():.4f}, Q: {loss_q.item():.4f}, Dyn: {loss_dyn.item():.4f}) | "
                f"10-Step MSE: {val_metrics['10_step_rollout_mse']:.4f} | "
                f"Wood PR-AUC: {val_metrics['wood_pr_auc']:.4f}"
            )

            history["step"].append(step)
            history["total_loss"].append(float(total_loss.item()))
            history["loss_dyn"].append(float(loss_dyn.item()))
            history["loss_event"].append(float(loss_event.item()))
            history["loss_q"].append(float(loss_q.item()))
            history["10_step_rollout_mse"].append(val_metrics["10_step_rollout_mse"])
            history["wood_pr_auc"].append(val_metrics["wood_pr_auc"])

    # Final audit
    final_metrics = evaluate_v3_metrics(model, val_buf, horizon=10, device=dev, num_batches=12)

    # Save checkpoint
    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(output_checkpoint, step=total_steps)
    print(f"\n[Checkpoint] Saved v3 recurrent model to {output_checkpoint}")

    # Evaluate promotion gate
    gate_passed = bool(
        final_metrics["10_step_rollout_mse"] < 0.85 and
        final_metrics["wood_pr_auc"] >= 0.08
    )

    results = {
        "step": total_steps,
        "recurrent_dim": recurrent_dim,
        "initial_metrics": init_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "gate_passed": gate_passed,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved screening metrics to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2 JEPA v3 Joint Recurrent Training.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--shards_dir", type=str, default="data/sequences")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--recurrent_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_ckpt", type=str, default="checkpoints/jepa_crafter_v3_phase2.pt")
    parser.add_argument("--out_json", type=str, default="results/step2_screening.json")
    args = parser.parse_args()

    train_v3_recurrent(
        checkpoint_base=args.checkpoint,
        shards_dir=args.shards_dir,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        recurrent_dim=args.recurrent_dim,
        lr=args.lr,
        device_name=args.device,
        output_checkpoint=args.out_ckpt,
        output_json=args.out_json,
    )


if __name__ == "__main__":
    main()
