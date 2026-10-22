"""
Head-Only Diagnostic Trainer for Learn-a-World v2 (Phase 1).
Trains EventHead and HealthHead on frozen JEPA target-encoder latents:
- Loss = 1.0 * FocalBCE(EventHead, gamma=2.0) + 0.5 * Huber(HealthHead)
- Evaluates per-achievement PR-AUC and recall@0.5 on held-out episodes
- Generates results/v2_head_validation.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.optim as optim

from eval.metrics import CRAFTER_ACHIEVEMENTS
from experiments.utils import get_device, load_config
from models.heads import NUM_ACHIEVEMENTS
from models.jepa_v2 import JEPAv2
from training.sequence_buffer import EpisodeBuffer

def compute_binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes exact Area Under the Precision-Recall Curve (PR-AUC).
    Numerically integrates precision vs recall across all unique score thresholds.
    """
    y_true = (y_true > 0).astype(np.int32)
    positives = int(np.sum(y_true))
    if positives == 0:
        return 0.0
    if positives == len(y_true):
        return 1.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    recalls = tp / positives
    precisions = tp / (tp + fp)

    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([precisions[0]], precisions))

    # Numerical integration compatible with both NumPy 1.x (trapz) and 2.x (trapezoid)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    pr_auc = float(trapz_fn(precisions, recalls))
    return float(np.clip(pr_auc, 0.0, 1.0))

def evaluate_heads(
    model: JEPAv2,
    val_buffer: EpisodeBuffer,
    seq_len: int = 16,
    num_batches: int = 20,
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Evaluates diagnostic performance on held-out validation episodes:
    - Per-achievement PR-AUC and recall@0.5
    - Macro Tier-1 PR-AUC
    - Health prediction Mean Absolute Error (MAE)
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.eval()

    all_event_targets: List[np.ndarray] = []
    all_event_probs: List[np.ndarray] = []
    all_health_targets: List[np.ndarray] = []
    all_health_preds: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch = val_buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    mode="event_balanced",
                    device=dev,
                )
            except Exception:
                batch = val_buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    mode="uniform",
                    device=dev,
                )

            obs = batch["obs"]          # [B, T, 3, 64, 64]
            actions = batch["actions"]  # [B, T]
            rewards = batch["rewards"]  # [B, T]
            events = batch["events"]    # [B, T, 22]
            mask = batch["mask_before"] # [B, T, 22]

            B, T = actions.shape
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_flat = model.encode(obs_flat, use_target=True)
            z = z_flat.view(B, T, -1)

            probs = model.event_head.predict_proba(z, actions, mask)
            health_pred = model.health_head(z, actions)

            event_counts = events.sum(dim=-1)
            health_target = (rewards - event_counts).clamp(-1.0, 1.0)

            all_event_targets.append(events.view(-1, NUM_ACHIEVEMENTS).cpu().numpy())
            all_event_probs.append(probs.view(-1, NUM_ACHIEVEMENTS).cpu().numpy())
            all_health_targets.append(health_target.view(-1).cpu().numpy())
            all_health_preds.append(health_pred.view(-1).cpu().numpy())

    targets_arr = np.concatenate(all_event_targets, axis=0)
    probs_arr = np.concatenate(all_event_probs, axis=0)
    health_tgt_arr = np.concatenate(all_health_targets, axis=0)
    health_pred_arr = np.concatenate(all_health_preds, axis=0)

    # Per-achievement metrics
    per_ach_metrics: Dict[str, Dict[str, float]] = {}
    tier1_achs = ["collect_wood", "collect_drink", "collect_sapling", "eat_plant", "wake_up"]
    tier1_pr_aucs: List[float] = []

    for i, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
        y_true = targets_arr[:, i]
        y_pred = probs_arr[:, i]
        pos_count = int(np.sum(y_true))

        pr_auc = compute_binary_pr_auc(y_true, y_pred)
        if pos_count > 0:
            recall_at_half = float(np.mean(y_pred[y_true > 0] >= 0.5))
        else:
            recall_at_half = 0.0

        per_ach_metrics[ach_name] = {
            "pr_auc": float(round(pr_auc, 4)),
            "recall_at_0.5": float(round(recall_at_half, 4)),
            "positive_samples": pos_count,
        }

        if ach_name in tier1_achs and pos_count > 0:
            tier1_pr_aucs.append(pr_auc)

    macro_tier1_pr_auc = float(np.mean(tier1_pr_aucs)) if tier1_pr_aucs else 0.5
    health_mae = float(np.mean(np.abs(health_pred_arr - health_tgt_arr)))

    return {
        "macro_tier1_pr_auc": float(round(macro_tier1_pr_auc, 4)),
        "health_mae": float(round(health_mae, 4)),
        "per_achievement": per_ach_metrics,
        "representation_problem": bool(macro_tier1_pr_auc < 0.5),
    }

def train_heads(
    config_path: str = "experiments/configs/crafter_v2.yaml",
    checkpoint_path: str = "checkpoints/jepa_crafter.pt",
    buffer: Optional[EpisodeBuffer] = None,
    total_steps: int = 500,
    batch_size: int = 32,
    seq_len: int = 16,
    device_override: Optional[str] = None,
    out_json_path: str = "results/v2_head_validation.json",
    out_checkpoint_path: str = "checkpoints/jepa_crafter_v2.pt",
) -> Tuple[JEPAv2, Dict[str, Any]]:
    """
    Runs diagnostic training for EventHead and HealthHead.
    """
    device = get_device(device_override)
    cfg = load_config(config_path)

    loss_cfg = cfg.get("loss", {})
    t_cfg = cfg.get("training_v2", {})

    event_weight = loss_cfg.get("event_weight", 1.0)
    health_weight = loss_cfg.get("health_weight", 0.5)
    focal_gamma = loss_cfg.get("focal_gamma", 2.0)
    lr_heads = t_cfg.get("lr_heads", 3e-4)

    # Initialize JEPAv2 from v1 checkpoint
    print(f"--> Initializing JEPAv2 with base checkpoint: {checkpoint_path}")
    model = JEPAv2.from_v1_checkpoint(
        checkpoint_path=checkpoint_path,
        config=cfg,
        device=device,
        freeze_base=True,
    )

    optimizer = optim.AdamW(
        list(model.event_head.parameters()) + list(model.health_head.parameters()),
        lr=lr_heads,
        weight_decay=1e-4,
    )

    # Prepare data
    if buffer is None:
        raise ValueError("Buffer must be provided to train_heads.")

    train_buf, val_buf = buffer.split_train_val(val_ratio=0.10)
    if val_buf.num_episodes == 0:
        val_buf = train_buf  # Fallback if dataset is very small

    print(f"--> Beginning Head Training for {total_steps} steps (Batch: {batch_size}, SeqLen: {seq_len}, Device: {device})...")

    history = {"step": [], "loss": [], "loss_event": [], "loss_health": []}

    model.train()
    for step in range(1, total_steps + 1):
        batch = train_buf.sample_sequences(
            batch_size=batch_size,
            seq_len=seq_len,
            mode="event_balanced",
            device=device,
        )

        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]

        B, T = actions.shape

        with torch.no_grad():
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_flat = model.encode(obs_flat, use_target=True)
            z = z_flat.view(B, T, -1)

        event_logits = model.event_head(z, actions, mask)
        loss_event = model.event_head.compute_loss(event_logits, events, gamma=focal_gamma)

        event_counts = events.sum(dim=-1)
        health_target = (rewards - event_counts).clamp(-1.0, 1.0)
        health_pred = model.health_head(z, actions)
        loss_health = model.health_head.compute_loss(health_pred, health_target)

        total_loss = event_weight * loss_event + health_weight * loss_health

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.event_head.parameters()) + list(model.health_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        if step % 50 == 0 or step == total_steps:
            print(f"    Step {step:4d}/{total_steps} | Loss: {total_loss.item():.4f} (Event: {loss_event.item():.4f}, Health: {loss_health.item():.4f})")
            history["step"].append(step)
            history["loss"].append(float(total_loss.item()))
            history["loss_event"].append(float(loss_event.item()))
            history["loss_health"].append(float(loss_health.item()))

    print("--> Running validation diagnostic evaluation...")
    val_report = evaluate_heads(
        model=model,
        val_buffer=val_buf,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    val_report["training_steps"] = total_steps
    val_report["final_loss"] = float(total_loss.item())

    # Save validation report
    out_file = Path(out_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(val_report, f, indent=2)
    print(f"--> Saved validation report to: {out_file}")

    # Save checkpoint
    model.save_checkpoint(out_checkpoint_path, step=total_steps, optimizer_state=optimizer.state_dict())
    print(f"--> Saved trained v2 model checkpoint to: {out_checkpoint_path}")

    return model, val_report

def main():
    parser = argparse.ArgumentParser(description="Train Task Heads for JEPA v2")
    parser.add_argument("--config", type=str, default="experiments/configs/crafter_v2.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-json", type=str, default="results/v2_head_validation.json")
    args = parser.parse_args()

    # Collect mini dataset if not provided
    from training.collect_sequences import collect_episodes
    print("--> Collecting mini dataset for trainer smoke run...")
    buf = collect_episodes(
        policy_name="structured_random",
        n_transitions=1500,
        out_dir="data/sequences_smoke",
        chunk_size=1000,
        seed=100,
    )

    train_heads(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        buffer=buf,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device_override=args.device,
        out_json_path=args.out_json,
    )

if __name__ == "__main__":
    main()
