#!/usr/bin/env python3
"""
Event Head Calibration Script for Learn-a-World v3.

Fits per-head temperature scaling / Platt scaling parameters on held-out validation episodes
(seeds disjoint from all training shards) and evaluates BEFORE/AFTER calibration metrics.

Outputs:
  - results/calibration/event_head_calibrator.json (calibrator weights)
  - results/diagnostics/calibration_t1_joint.json (full before/after diagnostic report)
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.calibration import EventHeadCalibrator, compute_binary_ece
from models.jepa_v3 import JEPAv3
from training.sequence_buffer import EpisodeBuffer


def load_heldout_data(shards_dir: str) -> Dict[str, np.ndarray]:
    """Loads held-out sequence episodes."""
    p = Path(shards_dir)
    if not p.exists():
        raise FileNotFoundError(f"Held-out directory not found: {shards_dir}")

    buffer = EpisodeBuffer(capacity_transitions=50_000)
    loaded = buffer.load_sharded_dataset(str(p))
    print(f"[Calibration] Loaded {buffer.num_episodes} held-out episodes ({buffer.total_transitions:,} transitions) from {shards_dir}")
    return buffer.get_all_transitions()


def extract_logits_and_labels(
    model: JEPAv3,
    data: Dict[str, np.ndarray],
    device: str = "cpu",
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extracts raw unscaled event logits [N, 22] and binary event targets [N, 22].
    Also returns episode lengths to allow episode-level train/val splitting.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    obs = data["obs"]
    actions = data["actions"]
    events = data["events"]  # [N, 22]
    mask_before = data["mask_before"]  # [N, 22]
    episode_lengths = data.get("episode_lengths", [len(obs)])

    N = len(obs)
    all_logits = []

    print(f"[Calibration] Extracting event logits for {N} transitions on {device}...")
    idx = 0
    with torch.no_grad():
        for ep_len in episode_lengths:
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

            # Raw uncalibrated logits [ep_len, 22]
            logits = model.event_head.forward(s, t_act, t_mask)
            all_logits.append(logits.cpu().numpy())
            idx += ep_len

    all_logits_arr = np.concatenate(all_logits, axis=0)
    return all_logits_arr, events.astype(np.int32), list(episode_lengths)


def split_by_episodes(
    logits: np.ndarray,
    labels: np.ndarray,
    episode_lengths: List[int],
    cal_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits transitions strictly by entire episodes to avoid leakage."""
    num_episodes = len(episode_lengths)
    n_cal_eps = max(1, int(round(num_episodes * cal_ratio)))

    cal_transitions = sum(episode_lengths[:n_cal_eps])
    print(f"[Calibration] Episode split: {n_cal_eps} calibration eps ({cal_transitions:,} transitions) | "
          f"{num_episodes - n_cal_eps} validation eps ({len(logits) - cal_transitions:,} transitions)")

    cal_logits = logits[:cal_transitions]
    cal_labels = labels[:cal_transitions]
    val_logits = logits[cal_transitions:]
    val_labels = labels[cal_transitions:]

    return cal_logits, cal_labels, val_logits, val_labels


def run_calibration_pipeline(
    checkpoint: str,
    shards_dir: str = "data/sequences",
    output_dir: str = "results/calibration",
    diag_dir: str = "results/diagnostics",
    drive_dir: Optional[str] = None,
    fit_bias: bool = True,
    cal_ratio: float = 0.8,
    device: str = "cuda",
) -> Dict[str, Any]:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    diag_p = Path(diag_dir)
    diag_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EVENT HEAD CALIBRATION PIPELINE (PART 1)")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Held-out:   {shards_dir}")
    print(f"  Fit Bias:   {fit_bias} (Platt Scaling)")
    print(f"  Device:     {device}")
    print("=" * 70 + "\n")

    # 1. Load model
    print(f"[Calibration] Loading model from {checkpoint}...")
    model = JEPAv3.load_checkpoint(checkpoint, device=device)
    model.to(torch.device(device))
    model.eval()

    # 2. Load held-out transitions
    data = load_heldout_data(shards_dir)

    # 3. Extract logits and labels
    logits, labels, episode_lengths = extract_logits_and_labels(model, data, device=device)

    # 4. Disjoint episode train/val split
    cal_logits, cal_labels, val_logits, val_labels = split_by_episodes(
        logits, labels, episode_lengths, cal_ratio=cal_ratio
    )

    # 5. Fit calibrator on calibration slice
    print(f"[Calibration] Fitting calibrator on calibration split ({len(cal_logits):,} transitions)...")
    calibrator = EventHeadCalibrator(num_heads=22, fit_bias=fit_bias)
    fit_details = calibrator.fit(cal_logits, cal_labels)

    # 6. Evaluate BEFORE and AFTER on validation slice
    print(f"\n[Calibration] Evaluating on held-out validation slice ({len(val_logits):,} transitions)...")
    raw_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30.0, 30.0)))
    cal_probs = calibrator.apply(val_logits)

    report_per_head = []
    active_ece_before = []
    active_ece_after = []
    recall05_before_cnt = 0
    recall05_after_cnt = 0

    print("\n" + "-" * 95)
    print(f"{'Idx':<4} {'Achievement':<22} {'Pos':<5} {'Temp':<7} {'Bias':<7} "
          f"{'PR-AUC':<8} {'ECE_Bef':<9} {'ECE_Aft':<9} {'Rec@0.5 Bef':<12} {'Rec@0.5 Aft':<12}")
    print("-" * 95)

    for k, name in enumerate(CRAFTER_ACHIEVEMENTS):
        y_val = val_labels[:, k]
        pos_cnt = int(np.sum(y_val))

        p_raw = raw_probs[:, k]
        p_cal = cal_probs[:, k]

        ece_before = compute_binary_ece(p_raw, y_val)
        ece_after = compute_binary_ece(p_cal, y_val)

        rec_before = float(np.sum((p_raw > 0.5) & (y_val == 1)) / pos_cnt) if pos_cnt > 0 else 0.0
        rec_after = float(np.sum((p_cal > 0.5) & (y_val == 1)) / pos_cnt) if pos_cnt > 0 else 0.0

        if pos_cnt > 0:
            pr_auc = float(average_precision_score(y_val, p_cal))
            active_ece_before.append(ece_before)
            active_ece_after.append(ece_after)
            if rec_before > 0:
                recall05_before_cnt += 1
            if rec_after > 0:
                recall05_after_cnt += 1
        else:
            pr_auc = 0.0

        T_val = float(calibrator.temperatures[k])
        b_val = float(calibrator.biases[k])

        print(f"{k:<4} {name:<22} {pos_cnt:<5} {T_val:<7.3f} {b_val:<7.3f} "
              f"{pr_auc:<8.4f} {ece_before:<9.4f} {ece_after:<9.4f} {rec_before:<12.2f} {rec_after:<12.2f}")

        report_per_head.append({
            "head_idx": k,
            "achievement": name,
            "pos_count": pos_cnt,
            "temperature": T_val,
            "bias": b_val,
            "pr_auc": pr_auc,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "recall_05_before": rec_before,
            "recall_05_after": rec_after,
        })

    macro_ece_before = float(np.mean(active_ece_before)) if active_ece_before else 0.0
    macro_ece_after = float(np.mean(active_ece_after)) if active_ece_after else 0.0

    print("-" * 95)
    print(f"Active Heads Macro ECE:     Before={macro_ece_before:.4f} | After={macro_ece_after:.4f} "
          f"({(1.0 - macro_ece_after / (macro_ece_before + 1e-9)) * 100:.1f}% reduction)")
    print(f"Active Heads with Recall@0.5 > 0: Before={recall05_before_cnt} | After={recall05_after_cnt}")
    print("=" * 95 + "\n")

    # 7. Save calibrator weights and diagnostic JSON
    calibrator_file = out_p / "event_head_calibrator.json"
    calibrator.save(calibrator_file)
    print(f"[Calibration] Saved calibrator parameters to {calibrator_file}")

    diag_file = diag_p / "calibration_t1_joint.json"
    summary_report = {
        "checkpoint": checkpoint,
        "shards_dir": shards_dir,
        "fit_bias": fit_bias,
        "cal_transitions": len(cal_logits),
        "val_transitions": len(val_logits),
        "macro_ece_before": macro_ece_before,
        "macro_ece_after": macro_ece_after,
        "recall_05_active_heads_before": recall05_before_cnt,
        "recall_05_active_heads_after": recall05_after_cnt,
        "temperatures": [float(t) for t in calibrator.temperatures],
        "biases": [float(b) for b in calibrator.biases],
        "per_head": report_per_head,
    }
    with open(diag_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Calibration] Saved calibration diagnostic summary to {diag_file}")

    # 8. Copy to Drive if specified
    if drive_dir:
        drive_p = Path(drive_dir)
        if drive_p.exists():
            drive_cal = drive_p / "results" / "calibration"
            drive_diag = drive_p / "results" / "diagnostics"
            drive_cal.mkdir(parents=True, exist_ok=True)
            drive_diag.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(calibrator_file), str(drive_cal / calibrator_file.name))
            shutil.copy2(str(diag_file), str(drive_diag / diag_file.name))
            print(f"[Drive] Copied calibrator and diagnostic results to {drive_dir}")

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate event head calibration.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_t1_joint.pt")
    parser.add_argument("--shards_dir", type=str, default="data/sequences")
    parser.add_argument("--output_dir", type=str, default="results/calibration")
    parser.add_argument("--diag_dir", type=str, default="results/diagnostics")
    parser.add_argument("--drive_dir", type=str, default=None)
    parser.add_argument("--no_bias", action="store_true", help="Fit pure temperature scalar without bias")
    parser.add_argument("--cal_ratio", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    run_calibration_pipeline(
        checkpoint=args.checkpoint,
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        diag_dir=args.diag_dir,
        drive_dir=args.drive_dir,
        fit_bias=not args.no_bias,
        cal_ratio=args.cal_ratio,
        device=args.device,
    )


if __name__ == "__main__":
    main()
