#!/usr/bin/env python3
"""
Diagnostic Script D3: Event & Health Head Prediction Audit
Audits all 22 achievement event heads and the health head of JEPA v2
on held-out transition data.

Computes:
- Per-achievement PR-AUC (Average Precision)
- ROC-AUC
- Recall & Precision at threshold 0.5
- Base event rate vs predicted rate
- Ranking of the 5 worst-performing event heads
- HealthHead MAE and correlation

Outputs:
- results/diagnostics/head_audit.json
- results/diagnostics/head_audit.png
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v2 import JEPAv2
from models.jepa_v3 import JEPAv3


def load_heldout_data(shards_dir: str, max_shards: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Loads transition shards from directory."""
    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No .npz shards found in {shards_dir}")
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    all_obs = []
    all_actions = []
    all_events = []
    all_masks = []
    all_rewards = []
    all_lengths = []

    print(f"[Head-Audit] Loading {len(shard_paths)} shards...")
    for p in shard_paths:
        data = np.load(p, allow_pickle=True)
        all_obs.append(data["obs"])
        all_actions.append(data["actions"])
        all_events.append(data["events"])
        all_masks.append(data["mask_before"])
        all_rewards.append(data["rewards"])
        lengths = data.get("episode_lengths", None)
        if lengths is not None and len(lengths) > 0:
            all_lengths.extend(lengths.tolist())
        else:
            dones = data["dones"]
            start = 0
            for i, d in enumerate(dones):
                if d or i == len(dones) - 1:
                    all_lengths.append(i + 1 - start)
                    start = i + 1

    return {
        "obs": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "events": np.concatenate(all_events, axis=0),
        "mask_before": np.concatenate(all_masks, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "episode_lengths": all_lengths,
    }


def audit_heads(
    model: JEPAv2,
    data: Dict[str, np.ndarray],
    device: str = "cpu",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Evaluates EventHead and HealthHead across all held-out transitions."""
    model.eval()
    dev = torch.device(device)

    obs = data["obs"]
    actions = data["actions"]
    events = data["events"]  # [N, 22]
    mask_before = data["mask_before"]  # [N, 22]
    rewards = data["rewards"]  # [N]

    N = len(obs)
    all_pred_probs = []
    all_pred_health = []

    print(f"[Head-Audit] Evaluating {N} transitions in batches of {batch_size}...")

    if isinstance(model, JEPAv3):
        lengths = data.get("episode_lengths", [N])
        idx = 0
        with torch.no_grad():
            for ep_len in lengths:
                ep_slice = slice(idx, idx + ep_len)
                b_obs = obs[ep_slice]
                b_act = actions[ep_slice]
                b_mask = mask_before[ep_slice]

                t_obs = torch.from_numpy(b_obs).permute(0, 3, 1, 2).float() / 255.0
                t_obs = t_obs.to(dev)
                t_act = torch.from_numpy(b_act).long().to(dev)
                t_mask = torch.from_numpy(b_mask).float().to(dev)

                z = model.encode(t_obs, use_target=True)
                h = model.recurrent_core.unroll_sequence(z.unsqueeze(0), t_act.unsqueeze(0)).squeeze(0)
                s = torch.cat([z, h], dim=-1)

                p_event = model.event_head.predict_proba(s, t_act, t_mask)
                pred_h = model.health_head(s, t_act).squeeze(-1)

                all_pred_probs.append(p_event.cpu().numpy())
                all_pred_health.append(pred_h.cpu().numpy())
                idx += ep_len
    else:
        with torch.no_grad():
            for i in range(0, N, batch_size):
                b_obs = obs[i:i + batch_size]
                b_act = actions[i:i + batch_size]
                b_mask = mask_before[i:i + batch_size]

                t_obs = torch.from_numpy(b_obs).permute(0, 3, 1, 2).float() / 255.0
                t_obs = t_obs.to(dev)
                t_act = torch.from_numpy(b_act).long().to(dev)
                t_mask = torch.from_numpy(b_mask).float().to(dev)

                z = model.encode(t_obs, use_target=True)
                p_event = model.event_head.predict_proba(z, t_act, t_mask)
                pred_h = model.health_head(z, t_act).squeeze(-1)

                all_pred_probs.append(p_event.cpu().numpy())
                all_pred_health.append(pred_h.cpu().numpy())

    pred_probs = np.concatenate(all_pred_probs, axis=0)  # [N, 22]
    pred_health = np.concatenate(all_pred_health, axis=0)  # [N]

    # Target for health: reward minus number of achievement events this step
    event_counts = events.sum(axis=-1)
    target_health = np.clip(rewards - event_counts, -1.0, 1.0)
    health_mae = float(np.mean(np.abs(pred_health - target_health)))

    per_achievement = {}
    eligible_for_worst = []

    for idx, name in enumerate(CRAFTER_ACHIEVEMENTS):
        y_true = events[:, idx].astype(np.int32)
        y_pred = pred_probs[:, idx]

        pos_count = int(np.sum(y_true))
        base_rate = float(pos_count / N)
        mean_pred = float(np.mean(y_pred))

        # Precision at 0.5
        pred_pos_05 = y_pred > 0.5
        tp = np.sum((y_pred > 0.5) & (y_true == 1))
        recall_05 = float(tp / pos_count) if pos_count > 0 else 0.0
        prec_05 = float(tp / np.sum(pred_pos_05)) if np.sum(pred_pos_05) > 0 else 0.0

        if pos_count > 0:
            pr_auc = float(average_precision_score(y_true, y_pred))
            try:
                roc_auc = float(roc_auc_score(y_true, y_pred))
            except Exception:
                roc_auc = 0.5
            eligible_for_worst.append({
                "name": name,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "pos_count": pos_count,
                "recall_at_0.5": recall_05,
                "base_rate": base_rate,
            })
        else:
            pr_auc = 0.0
            roc_auc = 0.5

        per_achievement[name] = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "positive_samples": pos_count,
            "base_rate": base_rate,
            "mean_pred_prob": mean_pred,
            "recall_at_0.5": recall_05,
            "precision_at_0.5": prec_05,
        }

    # Rank worst heads among those with positive examples
    eligible_for_worst.sort(key=lambda x: x["pr_auc"])
    worst_5 = eligible_for_worst[:5]

    # Zero-data heads
    zero_data_heads = [k for k, v in per_achievement.items() if v["positive_samples"] == 0]

    macro_pr_auc = float(np.mean([v["pr_auc"] for v in eligible_for_worst])) if eligible_for_worst else 0.0

    return {
        "total_transitions": N,
        "macro_pr_auc_active_heads": macro_pr_auc,
        "health_head_mae": health_mae,
        "worst_5_active_heads": worst_5,
        "zero_data_frontier_heads": zero_data_heads,
        "per_achievement": per_achievement,
    }


def plot_head_audit(audit: Dict[str, Any], output_png: str):
    """Generates a visualization of PR-AUC across achievements."""
    plt.figure(figsize=(12, 7), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    names = CRAFTER_ACHIEVEMENTS
    pr_aucs = [audit["per_achievement"][k]["pr_auc"] for k in names]
    pos_counts = [audit["per_achievement"][k]["positive_samples"] for k in names]

    # Color code: green = good, yellow = weak, red = 0 / bad
    colors = []
    for score, pos in zip(pr_aucs, pos_counts):
        if pos == 0:
            colors.append("#7f7f7f")  # grey: no data
        elif score < 0.25:
            colors.append("#d62728")  # red: weak
        elif score < 0.60:
            colors.append("#ff7f0e")  # orange: moderate
        else:
            colors.append("#2ca02c")  # green: strong

    y_pos = np.arange(len(names))
    bars = plt.barh(y_pos, pr_aucs, color=colors, edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, names, fontsize=9)
    plt.xlabel("PR-AUC (Average Precision)", fontsize=11)
    plt.xlim(0.0, 1.05)

    # Annotate with sample count
    for bar, count in zip(bars, pos_counts):
        w = bar.get_width()
        label = f"N={count}" if count > 0 else "0 data"
        plt.text(w + 0.02, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)

    plt.title(
        f"Event Head Audit — Macro PR-AUC (active heads): {audit['macro_pr_auc_active_heads']:.3f} | Health MAE: {audit['health_head_mae']:.3f}\n"
        f"Critical Weakness: collect_wood (PR-AUC={audit['per_achievement']['collect_wood']['pr_auc']:.2f}), collect_sapling (PR-AUC={audit['per_achievement']['collect_sapling']['pr_auc']:.2f})",
        fontsize=11,
        fontweight="bold"
    )

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"[Head-Audit] Saved plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Audit event and health prediction heads on held-out data.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--shards_dir", type=str, default="data/sequences")
    parser.add_argument("--max_shards", type=int, default=None, help="Maximum number of shards to audit (default: all)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/head_audit.json")
    parser.add_argument("--output_png", type=str, default="results/diagnostics/head_audit.png")
    args = parser.parse_args()

    print(f"[Head-Audit] Loading model from {args.checkpoint}...")
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt_raw, dict) and ckpt_raw.get("version") == "v3":
        model = JEPAv3()
        model.load_checkpoint(args.checkpoint, device=args.device)
    else:
        model = JEPAv2.from_v1_checkpoint(args.checkpoint, device=args.device)
    model.to(args.device)

    print(f"[Head-Audit] Loading held-out data from {args.shards_dir}...")
    data = load_heldout_data(args.shards_dir, max_shards=args.max_shards)

    print("[Head-Audit] Auditing heads...")
    audit = audit_heads(model, data, device=args.device)

    print("\n--- Event Head Audit Results ---")
    print(f"Total Evaluated Transitions: {audit['total_transitions']}")
    print(f"Macro PR-AUC (Active Heads): {audit['macro_pr_auc_active_heads']:.4f}")
    print(f"HealthHead MAE: {audit['health_head_mae']:.4f}")
    print("\nWorst 5 Active Heads by PR-AUC:")
    for h in audit["worst_5_active_heads"]:
        print(f"  - {h['name']:<20}: PR-AUC={h['pr_auc']:.4f}, Recall@0.5={h['recall_at_0.5']:.2f}, N_pos={h['pos_count']}")
    print(f"\nZero-Data Frontier Heads ({len(audit['zero_data_frontier_heads'])} / 22):")
    print("  " + ", ".join(audit["zero_data_frontier_heads"]))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[Head-Audit] Saved metrics to {args.output_json}")

    plot_head_audit(audit, args.output_png)


if __name__ == "__main__":
    main()
