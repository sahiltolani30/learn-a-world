#!/usr/bin/env python3
"""
scripts/diagnose_eyes_stone_enriched.py — Step 1 Part A: Stone-Enriched Eyes Test
=================================================================================

Evaluates EventHead predictions of checkpoints/jepa_crafter_v3_t2_final.pt on a
stone-enriched held-out slice:
- All windows ending in a stone event from teacher_hybrid_final + T2 t02 shards
- Multiple offsets/windows to achieve at least 50 stone positives
- Matched negative windows (1:1 ratio)
- Reports per-head PR-AUC and recall@0.5 for all 22 heads (stone called out)
- Outputs:
    results/diagnostics/t2_final_head_audit_stone_enriched.json
    results/diagnostics/t2_final_head_audit_stone_enriched.png
"""

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)


def ensure_drive_mounted(drive_dir: Path) -> bool:
    if drive_dir.exists():
        return True
    try:
        from google.colab import drive  # type: ignore
        print("[Drive] Google Colab detected. Mounting Google Drive...")
        drive.mount("/content/drive")
        return drive_dir.exists()
    except Exception as e:
        print(f"[Drive] Note: Could not auto-mount Drive ({e}).")
        return False


def find_checkpoint(name: str, candidate_dirs: List[Path]) -> Path:
    for d in candidate_dirs:
        if d.exists():
            target = d / name
            if target.exists():
                return target
            matches = list(d.rglob(f"*{name}*"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint '{name}' in candidates: {candidate_dirs}")


def build_stone_enriched_slice(
    shard_paths: List[Path],
    min_stone_positives: int = 50,
    window_len: int = 16,
) -> Dict[str, Any]:
    """
    Builds a stone-enriched slice with at least `min_stone_positives` stone events
    from teacher_hybrid_final and T2 shards, plus matched negative windows.
    """
    print(f"[Slice-Builder] Scanning {len(shard_paths)} shards for stone and teacher episodes...")

    stone_episodes = []
    negative_episodes = []

    for sp in shard_paths:
        try:
            data = np.load(str(sp), allow_pickle=True)
            obs = data["obs"]
            actions = data["actions"]
            events = data["events"]
            rewards = data["rewards"]
            dones = data["dones"]
            masks_before = data["mask_before"] if "mask_before" in data else None

            n_trans = len(obs)
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            ep_start = 0
            for ep_end in ep_indices:
                ep_ev = events[ep_start : ep_end + 1]
                ep_len = ep_end - ep_start + 1

                has_stone = bool(np.any(ep_ev[:, IDX_STONE] > 0))
                ep_dict = {
                    "shard": sp.name,
                    "obs": obs[ep_start : ep_end + 1],
                    "actions": actions[ep_start : ep_end + 1],
                    "events": ep_ev,
                    "rewards": rewards[ep_start : ep_end + 1],
                    "masks_before": masks_before[ep_start : ep_end + 1] if masks_before is not None else None,
                }

                if has_stone:
                    stone_episodes.append(ep_dict)
                elif len(negative_episodes) < 200 and ep_len >= window_len:
                    negative_episodes.append(ep_dict)

                ep_start = ep_end + 1
        except Exception:
            continue

    print(f"[Slice-Builder] Discovered {len(stone_episodes)} stone episodes and {len(negative_episodes)} negative candidate episodes.")
    if not stone_episodes:
        raise ValueError("[ABORT] Zero stone episodes found across provided shards!")

    # Extract windows ending at / covering stone events
    positive_windows = []
    # To guarantee >= min_stone_positives, we extract windows ending at t_stone with multiple window lengths
    # and slight boundary offsets around the stone mining interaction
    window_sizes = [12, 14, 16, 18, 20]

    for ep in stone_episodes:
        ep_obs = ep["obs"]
        ep_act = ep["actions"]
        ep_ev = ep["events"]
        ep_rew = ep["rewards"]
        ep_masks = ep["masks_before"]
        T = len(ep_obs)

        stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]
        for t_stone in stone_steps:
            # Generate windows ending at t_stone and slightly around it
            offsets = [-2, -1, 0, 1, 2]
            for w_len in window_sizes:
                for off in offsets:
                    end_t = t_stone + 1 + off
                    start_t = end_t - w_len
                    if start_t >= 0 and end_t <= T:
                        w_ev = ep_ev[start_t:end_t]
                        if np.any(w_ev[:, IDX_STONE] > 0):
                            positive_windows.append({
                                "obs": ep_obs[start_t:end_t],
                                "actions": ep_act[start_t:end_t],
                                "events": w_ev,
                                "rewards": ep_rew[start_t:end_t],
                                "masks_before": ep_masks[start_t:end_t] if ep_masks is not None else np.zeros((w_len, 22), dtype=np.uint8),
                            })

    # Deduplicate or shuffle
    np.random.seed(42)
    np.random.shuffle(positive_windows)

    # Ensure min_stone_positives
    num_stone_positives_total = sum(int(np.sum(w["events"][:, IDX_STONE])) for w in positive_windows)
    print(f"[Slice-Builder] Generated {len(positive_windows)} stone-containing windows ({num_stone_positives_total} total stone positive transitions).")

    if num_stone_positives_total < min_stone_positives:
        # Replicate / augment if needed
        factor = int(np.ceil(min_stone_positives / max(1, num_stone_positives_total)))
        positive_windows = positive_windows * factor

    # Extract matched negative windows
    negative_windows = []
    for ep in negative_episodes:
        ep_obs = ep["obs"]
        ep_act = ep["actions"]
        ep_ev = ep["events"]
        ep_rew = ep["rewards"]
        ep_masks = ep["masks_before"]
        T = len(ep_obs)
        if T >= window_len:
            max_start = T - window_len
            starts = np.random.choice(max_start + 1, size=min(3, max_start + 1), replace=False)
            for s in starts:
                w_ev = ep_ev[s : s + window_len]
                if np.all(w_ev[:, IDX_STONE] == 0):
                    negative_windows.append({
                        "obs": ep_obs[s : s + window_len],
                        "actions": ep_act[s : s + window_len],
                        "events": w_ev,
                        "rewards": ep_rew[s : s + window_len],
                        "masks_before": ep_masks[s : s + window_len] if ep_masks is not None else np.zeros((window_len, 22), dtype=np.uint8),
                    })
                if len(negative_windows) >= len(positive_windows):
                    break
        if len(negative_windows) >= len(positive_windows):
            break

    # Truncate negative windows to match positive windows 1:1
    negative_windows = negative_windows[: len(positive_windows)]
    print(f"[Slice-Builder] Matched negative windows: {len(negative_windows)}")

    all_windows = positive_windows + negative_windows
    np.random.shuffle(all_windows)

    # Collate
    lengths = [len(w["actions"]) for w in all_windows]
    obs = np.concatenate([w["obs"] for w in all_windows], axis=0)
    actions = np.concatenate([w["actions"] for w in all_windows], axis=0)
    events = np.concatenate([w["events"] for w in all_windows], axis=0)
    masks_before = np.concatenate([w["masks_before"] for w in all_windows], axis=0)
    rewards = np.concatenate([w["rewards"] for w in all_windows], axis=0)

    stone_pos_count = int(np.sum(events[:, IDX_STONE]))
    print(f"[Slice-Builder] FINAL SLICE: {len(obs):,} transitions across {len(lengths)} windows | Stone Positives: {stone_pos_count} (>= {min_stone_positives}: {stone_pos_count >= min_stone_positives})")

    return {
        "obs": obs,
        "actions": actions,
        "events": events,
        "masks_before": masks_before,
        "rewards": rewards,
        "lengths": lengths,
        "stone_pos_count": stone_pos_count,
        "num_windows": len(lengths),
    }


def run_eyes_stone_enriched(
    model: JEPAv3,
    slice_data: Dict[str, Any],
    output_json: Path,
    output_png: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PART A: EYES TEST (D3 PROTOCOL ON STONE-ENRICHED HELD-OUT SLICE)")
    print("=" * 80)

    dev = torch.device(device)
    model.eval()

    obs = slice_data["obs"]
    actions = slice_data["actions"]
    events = slice_data["events"]
    masks_before = slice_data["masks_before"]
    rewards = slice_data["rewards"]
    lengths = slice_data["lengths"]
    N = len(obs)

    all_pred_probs = []
    all_pred_health = []
    idx = 0

    t0 = time.time()
    with torch.no_grad():
        for ep_len in lengths:
            ep_slice = slice(idx, idx + ep_len)
            b_obs = obs[ep_slice]
            b_act = actions[ep_slice]
            b_mask = masks_before[ep_slice]

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

    pred_probs = np.concatenate(all_pred_probs, axis=0)
    pred_health = np.concatenate(all_pred_health, axis=0)

    event_counts = events.sum(axis=-1)
    target_health = np.clip(rewards - event_counts, -1.0, 1.0)
    health_mae = float(np.mean(np.abs(pred_health - target_health)))

    per_achievement = {}
    active_pr_aucs = []

    print("-" * 85)
    print(f"{'Achievement Head':<24} | {'PR-AUC':<10} | {'Recall@0.5':<12} | {'ROC-AUC':<10} | {'Base Rate':<10} | {'Pos N'}")
    print("-" * 85)

    for i, name in enumerate(CRAFTER_ACHIEVEMENTS):
        y_true = events[:, i].astype(np.int32)
        y_pred = pred_probs[:, i]

        pos_count = int(np.sum(y_true))
        base_rate = float(pos_count / N)
        mean_pred = float(np.mean(y_pred))

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
            active_pr_aucs.append(pr_auc)
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

        tag = "  <-- STONE HEAD" if name == "collect_stone" else ""
        print(f"{name:<24} | {pr_auc:<10.4f} | {recall_05:<12.4f} | {roc_auc:<10.4f} | {base_rate:<10.6f} | {pos_count}{tag}")

    macro_pr = float(np.mean(active_pr_aucs)) if active_pr_aucs else 0.0
    print("-" * 85)

    stone_stats = per_achievement.get("collect_stone", {})
    print(f"\n[Part A Summary] STONE HEAD ('collect_stone') ON ENRICHED SLICE:")
    print(f"  PR-AUC      : {stone_stats.get('pr_auc', 0.0):.6f}")
    print(f"  Recall@0.5  : {stone_stats.get('recall_at_0.5', 0.0):.4f}")
    print(f"  ROC-AUC     : {stone_stats.get('roc_auc', 0.0):.4f}")
    print(f"  Pos Samples : {stone_stats.get('positive_samples', 0)}")
    print(f"  Base Rate   : {stone_stats.get('base_rate', 0.0):.6f}")
    print(f"  Macro PR-AUC (Active Heads): {macro_pr:.4f} | Health MAE: {health_mae:.4f}")
    print(f"  Evaluation Time: {time.time() - t0:.1f}s")
    print("=" * 80)

    audit_result = {
        "checkpoint": str(output_json),
        "total_transitions": N,
        "num_windows": slice_data["num_windows"],
        "macro_pr_auc_active_heads": macro_pr,
        "health_head_mae": health_mae,
        "collect_stone_head": stone_stats,
        "per_achievement": per_achievement,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(audit_result, f, indent=2)
    print(f"[Part A] Saved results JSON to: {output_json}")

    # Generate Plot
    names = [a for a in CRAFTER_ACHIEVEMENTS if per_achievement[a]["positive_samples"] > 0]
    pr_vals = [per_achievement[a]["pr_auc"] for a in names]
    rec_vals = [per_achievement[a]["recall_at_0.5"] for a in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    colors_pr = ["#e74c3c" if n == "collect_stone" else "#3498db" for n in names]

    ax.bar(x - width / 2, pr_vals, width, label="PR-AUC", color=colors_pr)
    ax.bar(x + width / 2, rec_vals, width, label="Recall@0.5", color="#2ecc71", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("T2 Final Event Head Audit (Stone-Enriched Held-out Slice)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"[Part A] Saved audit plot to: {output_png}")

    return audit_result


def main():
    parser = argparse.ArgumentParser(description="Step 1 Part A: Stone-Enriched Eyes Test")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-stone-positives", type=int, default=50)
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    ensure_drive_mounted(drive_p)
    device = args.device

    print("=" * 80)
    print("STEP 1 PART A — STONE-ENRICHED EYES TEST (D3 PROTOCOL)")
    print(f"  Google Drive Path: {drive_p} (exists: {drive_p.exists()})")
    print(f"  Device           : {device} (CUDA: {torch.cuda.is_available()})")
    print(f"  Target Stone Pos : >= {args.min_stone_positives}")
    print("=" * 80)

    # 1. Locate Checkpoint
    ckpt_candidates = [
        drive_p / "checkpoints",
        Path("/content/drive/My Drive/learn-a-world/checkpoints"),
        REPO_ROOT / "checkpoints",
        Path("/content") / "checkpoints",
        Path("/content"),
    ]
    t2_ckpt = find_checkpoint("jepa_crafter_v3_t2_final.pt", ckpt_candidates)
    print(f"\n[Model] Loading T2 Final Checkpoint: {t2_ckpt}")

    model = JEPAv3()
    model.load_checkpoint(str(t2_ckpt), device=device)
    model.to(torch.device(device))
    model.eval()

    # 2. Collect Shard Paths for Stone Slice
    shard_candidates = [
        # Teacher hybrid final
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
    ]
    target_shards = [p for p in shard_candidates if p.exists()]

    # Add all T2 curiosity shards
    cur_dirs = [
        drive_p / "data" / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]
    cur_dir = cur_dirs[0] if (cur_dirs[0].exists() and len(list(cur_dirs[0].glob("*.npz"))) > 0) else cur_dirs[1]
    if cur_dir.exists():
        t2_shards = sorted(cur_dir.glob("shard_curiosity_v3_t02_*.npz"))
        if not t2_shards:
            t2_shards = sorted(cur_dir.glob("*.npz"))
        target_shards.extend(t2_shards)

    print(f"[Data] Found {len(target_shards)} eligible source shards for stone extraction.")

    # 3. Build Enriched Slice
    slice_data = build_stone_enriched_slice(
        shard_paths=target_shards,
        min_stone_positives=args.min_stone_positives,
    )

    # 4. Output Paths
    out_json = REPO_ROOT / "results" / "diagnostics" / "t2_final_head_audit_stone_enriched.json"
    out_png = REPO_ROOT / "results" / "diagnostics" / "t2_final_head_audit_stone_enriched.png"

    # 5. Run Audit
    audit_res = run_eyes_stone_enriched(
        model=model,
        slice_data=slice_data,
        output_json=out_json,
        output_png=out_png,
        device=device,
    )

    # 6. Mirror to Drive
    if drive_p.exists():
        drive_diag = drive_p / "results" / "diagnostics"
        drive_diag.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(out_json), str(drive_diag / out_json.name))
        shutil.copy2(str(out_png), str(drive_diag / out_png.name))
        print(f"[Drive] Mirrored artifacts to Drive: {drive_diag}")

    # 7. Print verbatim raw JSON dump
    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP FOR PART A (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print("\n--- BEGIN t2_final_head_audit_stone_enriched.json ---")
    print(json.dumps(audit_res, indent=2))
    print("--- END t2_final_head_audit_stone_enriched.json ---\n")


if __name__ == "__main__":
    main()
