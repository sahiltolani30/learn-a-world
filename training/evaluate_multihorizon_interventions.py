#!/usr/bin/env python3
"""
Sections 8 & 9 (Gates 8 & 9): Multi-Horizon Free-Running Validation &
Counterfactual Action Intervention Evaluation.

Evaluates:
- Gate 8: Multi-horizon state delta accumulation across horizons H in {1, 5, 10, 25}.
  Compares teacher-forced vs free-running recurrent unroll against zero-change persistence.
- Gate 9: Counterfactual action intervention sensitivity and consequence matching.
  Tests pairwise action separation and whether actual action predictions beat alternative actions.

Saves report to results/diagnostics/phase1_5_intervention_eval.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.jepa_v3 import JEPAv3
from training.phase1_5_data import discover_active_shards, selected_paths_from_manifest
from training.sequence_buffer import EpisodeBuffer


def load_validation_buffer(
    shards_dir: Union[str, Path, Sequence[Union[str, Path]]],
    manifest_path: Optional[Union[str, Path]] = None,
    val_ratio: float = 0.10,
    split_seed: int = 42,
    capacity_transitions: int = 1_000_000,
) -> EpisodeBuffer:
    """Load deduplicated dataset and extract the exact held-out validation buffer."""
    if manifest_path is not None:
        shards = selected_paths_from_manifest(manifest_path)
    else:
        roots = [shards_dir] if isinstance(shards_dir, (str, Path)) else list(shards_dir)
        discovery = discover_active_shards(roots)
        shards = [Path(item["path"]) for item in discovery["selected_shards"]]
    if not shards:
        raise FileNotFoundError(f"No sequence shards found in {shards_dir}")

    full_buffer = EpisodeBuffer(capacity_transitions=capacity_transitions)
    for shard in shards:
        full_buffer.load_shard(shard)

    _, val_buffer = full_buffer.split_train_val(val_ratio=val_ratio, seed=split_seed)
    return val_buffer if len(val_buffer.episodes) > 0 else full_buffer


def evaluate_multi_horizon(
    model: JEPAv3,
    buffer: EpisodeBuffer,
    horizons: Sequence[int] = (1, 5, 10, 25),
    batches: int = 24,
    batch_size: int = 16,
    labeled_ratio: float = 0.8,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Evaluates multi-horizon state delta accumulation over horizons H in {1, 5, 10, 25}.
    Compares Free-Running unroll vs Teacher-Forced unroll vs Zero-Change Persistence.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.eval()

    max_h = max(horizons)
    seq_len = max_h + 1

    results_by_h: Dict[int, Dict[str, Any]] = {}
    for h in horizons:
        results_by_h[h] = {
            "valid_samples": 0,
            "freerun_se": 0.0,
            "teacher_forced_se": 0.0,
            "persistence_se": 0.0,
            "latent_drift_l2": 0.0,
        }

    with torch.no_grad():
        for _ in range(batches):
            try:
                batch = buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    labeled_ratio=labeled_ratio,
                    device=dev,
                )
            except Exception:
                break

            obs = batch["obs"]  # [B, seq_len, 3, 64, 64]
            actions = batch["actions"]  # [B, seq_len]
            dones = batch["dones"]  # [B, seq_len]
            state_deltas = batch["state_delta"]  # [B, seq_len, 16]
            valid_deltas = batch["state_delta_valid"]  # [B, seq_len]

            B, T = actions.shape

            # Encode online and target
            B_T_obs = obs.reshape(B * T, *obs.shape[2:])
            z_seq = model.encoder(B_T_obs).reshape(B, T, -1)
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)

            # Teacher-forced single-step predictions for all steps
            trans_all = model.predict_transition(s_seq, actions)
            delta_hat_tf = trans_all["state_delta"]  # [B, T, 16]

            # Evaluate each requested horizon starting from t=0
            for h in horizons:
                if h >= T:
                    continue

                # Valid mask: all steps up to horizon must have valid delta labels and no episode termination
                valid_mask = valid_deltas[:, :h].all(dim=1) & (~dones[:, :h - 1]).all(dim=1) if h > 1 else valid_deltas[:, 0]
                n_valid = int(valid_mask.sum().item())
                if n_valid == 0:
                    continue

                target_cum = state_deltas[:, :h].sum(dim=1)  # [B, 16]
                tf_cum = delta_hat_tf[:, :h].sum(dim=1)  # [B, 16]

                # Free-running recurrent unroll from t=0
                z_k = z_seq[:, 0]
                h_k = h_seq[:, 0]
                cum_delta_free = torch.zeros((B, 16), dtype=torch.float32, device=dev)

                for step_idx in range(h):
                    s_k = torch.cat([z_k, h_k], dim=-1)
                    a_k = actions[:, step_idx]
                    trans_k = model.predict_transition(s_k, a_k)
                    cum_delta_free = cum_delta_free + trans_k["state_delta"]

                    if step_idx < h - 1:
                        z_k, h_k, _ = model.predict_next_step(z_k, h_k, a_k)

                # Errors on valid samples
                diff_free = (cum_delta_free - target_cum)[valid_mask].cpu().numpy()
                diff_tf = (tf_cum - target_cum)[valid_mask].cpu().numpy()
                diff_persist = target_cum[valid_mask].cpu().numpy()  # Zero change prediction

                # Latent drift at horizon h
                z_target_h = z_seq[:, h - 1][valid_mask]
                latent_drift = (z_k[valid_mask] - z_target_h).norm(p=2, dim=-1).cpu().numpy()

                results_by_h[h]["valid_samples"] += n_valid
                results_by_h[h]["freerun_se"] += float((diff_free ** 2).sum())
                results_by_h[h]["teacher_forced_se"] += float((diff_tf ** 2).sum())
                results_by_h[h]["persistence_se"] += float((diff_persist ** 2).sum())
                results_by_h[h]["latent_drift_l2"] += float(latent_drift.sum())

    summary: Dict[str, Any] = {}
    for h, data in results_by_h.items():
        n = data["valid_samples"]
        if n == 0:
            summary[f"horizon_{h}"] = {"valid_samples": 0}
            continue
        slots = n * 16
        mse_free = data["freerun_se"] / slots
        mse_tf = data["teacher_forced_se"] / slots
        mse_persist = data["persistence_se"] / slots
        beats_persist = bool(mse_free < mse_persist)
        summary[f"horizon_{h}"] = {
            "horizon": h,
            "valid_samples": n,
            "freerun_mse": float(mse_free),
            "teacher_forced_mse": float(mse_tf),
            "persistence_mse": float(mse_persist),
            "freerun_beats_persistence": beats_persist,
            "relative_improvement_pct": float(100.0 * (mse_persist - mse_free) / max(1e-6, mse_persist)),
            "mean_latent_l2_drift": float(data["latent_drift_l2"] / n),
        }

    # Gate 8 verification: free-running at H=5 and H=10 must beat persistence
    gate_8_passed = True
    for h in [1, 5, 10]:
        h_key = f"horizon_{h}"
        if h_key in summary and summary[h_key].get("valid_samples", 0) > 0:
            if not summary[h_key].get("freerun_beats_persistence", False):
                gate_8_passed = False

    return {
        "gate_8_passed": gate_8_passed,
        "horizons": summary,
    }


def evaluate_counterfactual_interventions(
    model: JEPAv3,
    buffer: EpisodeBuffer,
    batches: int = 24,
    batch_size: int = 16,
    num_actions: int = 17,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Evaluates counterfactual action interventions from identical visual states.
    Checks:
    1. Pairwise Action Separation: Model predicts distinct consequences across candidate actions.
    2. Intervention Consequence Matching: Actual actions predict true consequence better than counterfactuals.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.eval()

    all_separations: List[float] = []
    total_events_checked = 0
    correct_intervention_rankings = 0
    actual_action_errors: List[float] = []
    cf_action_errors: List[float] = []
    margins: List[float] = []

    with torch.no_grad():
        for _ in range(batches):
            try:
                batch = buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=8,
                    labeled_ratio=1.0,
                    device=dev,
                )
            except Exception:
                break

            obs = batch["obs"]
            actions = batch["actions"]
            dones = batch["dones"]
            true_deltas = batch["state_delta"]  # [B, T, 16]
            valid_deltas = batch["state_delta_valid"]

            B, T = actions.shape
            z_seq = model.encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)  # [B, T, state_dim]

            for b in range(B):
                for t in range(T):
                    if not valid_deltas[b, t]:
                        continue

                    s_bt = s_seq[b, t].unsqueeze(0).expand(num_actions, -1)  # [17, state_dim]
                    all_acts = torch.arange(num_actions, dtype=torch.long, device=dev)

                    trans = model.predict_transition(s_bt, all_acts)
                    delta_candidates = trans["state_delta"]  # [17, 16]

                    # 1. Pairwise action separation
                    diffs = delta_candidates.unsqueeze(0) - delta_candidates.unsqueeze(1)  # [17, 17, 16]
                    dists = diffs.norm(p=2, dim=-1)  # [17, 17]
                    triu_mask = torch.triu(torch.ones(num_actions, num_actions, dtype=torch.bool, device=dev), diagonal=1)
                    pairwise_dists = dists[triu_mask]
                    all_separations.append(float(pairwise_dists.mean().item()))

                    # 2. Consequence matching on non-zero true deltas
                    true_delta = true_deltas[b, t]  # [16]
                    if true_delta.abs().sum() > 0.05:  # Non-trivial resource delta
                        actual_a = int(actions[b, t].item())
                        actual_pred = delta_candidates[actual_a]
                        actual_err = float((actual_pred - true_delta).norm(p=2).item())

                        cf_mask = torch.ones(num_actions, dtype=torch.bool, device=dev)
                        cf_mask[actual_a] = False
                        cf_preds = delta_candidates[cf_mask]  # [16, 16]
                        cf_errs = (cf_preds - true_delta.unsqueeze(0)).norm(p=2, dim=-1)
                        cf_mean_err = float(cf_errs.mean().item())

                        margin = cf_mean_err - actual_err
                        margins.append(margin)
                        actual_action_errors.append(actual_err)
                        cf_action_errors.append(cf_mean_err)
                        total_events_checked += 1

                        if actual_err < cf_mean_err:
                            correct_intervention_rankings += 1

    mean_separation = float(np.mean(all_separations)) if all_separations else 0.0
    action_collapse_detected = bool(mean_separation < 1e-4)

    intervention_accuracy_pct = float(100.0 * correct_intervention_rankings / max(1, total_events_checked))
    mean_margin = float(np.mean(margins)) if margins else 0.0

    # Gate 9 passed if action separation > 0 and intervention accuracy beats chance (> 50%)
    gate_9_passed = bool(not action_collapse_detected and intervention_accuracy_pct >= 50.0 and mean_margin >= 0.0)

    return {
        "gate_9_passed": gate_9_passed,
        "action_collapse_detected": action_collapse_detected,
        "mean_pairwise_action_separation": mean_separation,
        "total_event_transitions_evaluated": total_events_checked,
        "correct_intervention_rankings": correct_intervention_rankings,
        "intervention_accuracy_pct": intervention_accuracy_pct,
        "mean_counterfactual_margin": mean_margin,
        "mean_actual_action_error": float(np.mean(actual_action_errors)) if actual_action_errors else 0.0,
        "mean_counterfactual_action_error": float(np.mean(cf_action_errors)) if cf_action_errors else 0.0,
    }


def run_evaluation(
    shards_dir: Union[str, Path, Sequence[Union[str, Path]]],
    manifest_path: Optional[Union[str, Path]],
    checkpoint_path: Union[str, Path],
    output_json: Union[str, Path],
    val_ratio: float = 0.10,
    split_seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute both Gate 8 and Gate 9 evaluations and generate final diagnostic JSON."""
    start_time = time.time()
    dev = torch.device(device)

    print("=" * 80)
    print("LEARN-A-WORLD v3: MULTI-HORIZON & INTERVENTION EVALUATION (GATES 8 & 9)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_json}")
    print(f"Device:     {dev}")
    print("=" * 80)

    # 1. Load validation buffer
    val_buffer = load_validation_buffer(
        shards_dir=shards_dir,
        manifest_path=manifest_path,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    print(f"[Validation Buffer] Episodes: {len(val_buffer.episodes)} | Transitions: {val_buffer.total_transitions}")

    # 2. Load model
    model = JEPAv3()
    model.load_checkpoint(str(checkpoint_path), device=dev)
    model.to(dev)

    # 3. Gate 8: Multi-horizon evaluation
    print("\n[Gate 8 Evaluation] Evaluating multi-horizon free-running predictions (H in {1, 5, 10, 25})...")
    gate_8_results = evaluate_multi_horizon(
        model,
        val_buffer,
        horizons=(1, 5, 10, 25),
        batches=24,
        batch_size=16,
        labeled_ratio=0.8,
        device=dev,
    )
    print(f"  Gate 8 Passed: {gate_8_results['gate_8_passed']}")
    for h_name, h_stats in gate_8_results["horizons"].items():
        if h_stats.get("valid_samples", 0) > 0:
            print(
                f"    {h_name:<10}: Free-Run MSE = {h_stats['freerun_mse']:.6f} | "
                f"Persist MSE = {h_stats['persistence_mse']:.6f} | "
                f"Beats Persist = {h_stats['freerun_beats_persistence']} ({h_stats['relative_improvement_pct']:+.1f}%)"
            )

    # 4. Gate 9: Counterfactual intervention evaluation
    print("\n[Gate 9 Evaluation] Evaluating counterfactual action interventions...")
    gate_9_results = evaluate_counterfactual_interventions(
        model,
        val_buffer,
        batches=24,
        batch_size=16,
        device=dev,
    )
    print(f"  Gate 9 Passed: {gate_9_results['gate_9_passed']}")
    print(f"  Mean Action Separation:  {gate_9_results['mean_pairwise_action_separation']:.6f}")
    print(f"  Intervention Accuracy:   {gate_9_results['intervention_accuracy_pct']:.2f}%")
    print(f"  Counterfactual Margin:   {gate_9_results['mean_counterfactual_margin']:+.6f}")

    elapsed_sec = round(time.time() - start_time, 2)

    report = {
        "protocol": "strict_visual_jepa_clean_v2",
        "checkpoint": str(checkpoint_path),
        "dataset_split": {
            "validation_episodes": len(val_buffer.episodes),
            "validation_transitions": val_buffer.total_transitions,
            "split_seed": split_seed,
            "val_ratio": val_ratio,
        },
        "gate_8": gate_8_results,
        "gate_9": gate_9_results,
        "overall_status": {
            "gate_8_passed": gate_8_results["gate_8_passed"],
            "gate_9_passed": gate_9_results["gate_9_passed"],
            "both_passed": gate_8_results["gate_8_passed"] and gate_9_results["gate_9_passed"],
        },
        "elapsed_seconds": elapsed_sec,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n[Saved Diagnostic Report] -> {out_p}")
    print("=" * 80)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Multi-Horizon & Action Interventions (Gates 8 & 9)")
    parser.add_argument("--data_root", nargs="+", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", default="results/diagnostics/phase1_5_intervention_eval.json")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.data_root and args.manifest is None:
        parser.error("Must provide --data_root or --manifest")

    run_evaluation(
        shards_dir=args.data_root,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
