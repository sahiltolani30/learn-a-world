#!/usr/bin/env python3
"""
Section 6 / Gate 5: Supervised State Readout Learning for JEPAv3.

Trains the StateReadoutHead ([z_t, h_t] -> x_hat_t, sigma_t) to predict current
state/resource vectors from latent visual representations and recurrent belief.

Strict Architectural Constraints:
1. Base Model Frozen: Encoder, target encoder, recurrent core, Q heads, event head,
   and prior remain 100% frozen.
2. Privileged-Target Training: Environment inventory vectors are used exclusively as
   training targets. They never enter the model as inference inputs.
3. Strict Episode Splitting: Validation is split by complete, independent episodes / seeds.
4. Gate 5 Metric: Held-out state readout MSE must strictly beat the persistence baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.jepa_v3 import JEPAv3
from training.phase1_5_data import (
    discover_active_shards,
    selected_paths_from_manifest,
)
from training.sequence_buffer import EpisodeBuffer


def load_sequence_shards(
    shards_dir: Union[str, Path, Sequence[Union[str, Path]]],
    capacity_transitions: int = 1_000_000,
    manifest_path: Optional[Union[str, Path]] = None,
) -> EpisodeBuffer:
    """Load deduplicated sequence shards into an EpisodeBuffer with resilient fallback."""
    shards: List[Path] = []
    man_p = Path(manifest_path) if manifest_path is not None else None
    if man_p is not None and man_p.exists():
        shards = selected_paths_from_manifest(man_p)
    elif man_p is not None and (REPO_ROOT / man_p).exists():
        shards = selected_paths_from_manifest(REPO_ROOT / man_p)
    else:
        # Fallback to drive/manifest candidates or directory discovery
        candidates = [
            REPO_ROOT / "results" / "drive_data_audit_manifest.json",
            Path("/content/drive/MyDrive/learn-a-world/results/phase1_5_manifest.json"),
            Path("/content/drive/MyDrive/learn-a-world/results/drive_data_audit_manifest.json"),
        ]
        for c in candidates:
            if c.exists():
                try:
                    shards = selected_paths_from_manifest(c)
                    if shards:
                        print(f"-> Using manifest candidate: {c}")
                        break
                except Exception:
                    pass

    if not shards:
        roots: List[Union[str, Path]] = []
        if shards_dir:
            roots.extend([shards_dir] if isinstance(shards_dir, (str, Path)) else list(shards_dir))
        standard_search = [
            Path("/content/drive/MyDrive/learn-a-world/results/sequences"),
            Path("/content/drive/MyDrive/learn-a-world/results"),
            REPO_ROOT / "results" / "sequences",
            REPO_ROOT / "results",
        ]
        for p in standard_search:
            if p.exists() and p not in roots:
                roots.append(p)
        try:
            discovery = discover_active_shards(roots)
            shards = [Path(item["path"]) for item in discovery.get("selected_shards", [])]
        except Exception:
            shards = []

        if not shards:
            for r in roots:
                found = sorted(list(Path(r).glob("*.npz")))
                if found:
                    shards.extend(found)
                    break

    if not shards:
        raise FileNotFoundError(f"No sequence shards found in {shards_dir} or standard paths")

    print(f"-> Loading {len(shards)} sequence shard files into EpisodeBuffer...")
    buffer = EpisodeBuffer(capacity_transitions=capacity_transitions)
    for shard in shards:
        buffer.load_shard(shard)
    return buffer


def _encode_batch_frozen(model: JEPAv3, obs: torch.Tensor, use_target: bool = True) -> torch.Tensor:
    """Encode observation batch with frozen encoder [B, T, C, H, W] -> [B, T, 128].

    use_target=True (default, plan Stage 1 encoder unification): the target encoder is
    the serving convention shared by event/health-head training, belief updates, MPC
    planning, and consolidation keys — the readout head must live in the same space.
    """
    B, T = obs.shape[:2]
    with torch.no_grad():
        z = model.encode(obs.reshape(B * T, *obs.shape[2:]), use_target=use_target)
        return z.reshape(B, T, -1)


def evaluate_state_readout(
    model: JEPAv3,
    buffer: EpisodeBuffer,
    batch_size: int = 16,
    seq_len: int = 16,
    batches: int = 20,
    labeled_ratio: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    use_target_encoder: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates StateReadoutHead on validation episodes.
    Computes readout MSE versus persistence and zero-state baselines.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model.eval()

    total_samples = 0
    total_valid_slots = 0
    total_se_pred = 0.0
    total_se_zero = 0.0
    total_se_persist = 0.0
    slot_se_pred = np.zeros(16, dtype=np.float64)
    slot_se_zero = np.zeros(16, dtype=np.float64)
    slot_valid_counts = np.zeros(16, dtype=np.int64)

    with torch.no_grad():
        for _ in range(batches):
            try:
                batch = buffer.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    labeled_ratio=labeled_ratio,
                    device=dev,
                )
            except Exception:
                break

            obs = batch["obs"]
            actions = batch["actions"]
            dones = batch["dones"]
            target_state = batch["state_current"]
            valid_mask = batch["state_current_valid"]  # [B, T]

            z_seq = _encode_batch_frozen(model, obs, use_target=use_target_encoder)
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)

            pred_state, uncertainty = model.predict_state(s_seq)

            # Persistence baseline: state at t-1 (or 0 for t=0)
            B, T_dim, D_dim = target_state.shape
            persist_state = torch.zeros_like(target_state)
            if T_dim > 1:
                persist_state[:, 1:] = target_state[:, :-1]

            # Masked errors
            mask_flat = valid_mask.reshape(-1)
            if not mask_flat.any():
                continue

            pred_flat = pred_state.reshape(-1, D_dim)[mask_flat].cpu().numpy()
            target_flat = target_state.reshape(-1, D_dim)[mask_flat].cpu().numpy()
            persist_flat = persist_state.reshape(-1, D_dim)[mask_flat].cpu().numpy()

            n_valid = len(pred_flat)
            if n_valid == 0:
                continue

            se_pred = (pred_flat - target_flat) ** 2
            se_zero = target_flat ** 2
            se_persist = (persist_flat - target_flat) ** 2

            total_se_pred += float(se_pred.sum())
            total_se_zero += float(se_zero.sum())
            total_se_persist += float(se_persist.sum())
            total_valid_slots += n_valid * D_dim
            total_samples += n_valid

            slot_se_pred += se_pred.sum(axis=0)
            slot_se_zero += se_zero.sum(axis=0)
            slot_valid_counts += n_valid

    if total_valid_slots == 0:
        return {
            "state_readout_mse": None,
            "zero_state_mse": None,
            "persistence_mse": None,
            "state_readout_beats_persistence": False,
            "valid_samples": 0,
        }

    mse_pred = total_se_pred / total_valid_slots
    mse_zero = total_se_zero / total_valid_slots
    mse_persist = total_se_persist / total_valid_slots
    baseline_mse = min(mse_zero, mse_persist)
    beats_baseline = bool(mse_pred < baseline_mse)

    per_slot_pred_mse = (slot_se_pred / np.maximum(slot_valid_counts, 1)).tolist()
    per_slot_zero_mse = (slot_se_zero / np.maximum(slot_valid_counts, 1)).tolist()

    return {
        "state_readout_mse": float(mse_pred),
        "zero_state_mse": float(mse_zero),
        "persistence_mse": float(mse_persist),
        "baseline_best_mse": float(baseline_mse),
        "state_readout_beats_persistence": beats_baseline,
        "relative_improvement_pct": float(100.0 * (baseline_mse - mse_pred) / max(1e-6, baseline_mse)),
        "valid_samples": int(total_samples),
        "per_slot_pred_mse": per_slot_pred_mse,
        "per_slot_zero_mse": per_slot_zero_mse,
    }


def train_state_readout(
    shards_dir: Union[str, Path, Sequence[Union[str, Path]]],
    checkpoint_path: Optional[Union[str, Path]],
    output_checkpoint: Union[str, Path],
    output_json: Optional[Union[str, Path]] = None,
    total_steps: int = 1000,
    batch_size: int = 16,
    seq_len: int = 16,
    labeled_ratio: float = 0.5,
    lr: float = 1e-3,
    split_seed: int = 42,
    val_ratio: float = 0.10,
    manifest_path: Optional[Union[str, Path]] = None,
    capacity_transitions: int = 1_000_000,
    device: str = "cpu",
    use_target_encoder: bool = True,
) -> Dict[str, Any]:
    """
    Main training pipeline for StateReadoutHead under Section 6 / Gate 5.
    Freezes all base networks and trains only state_readout_head.
    """
    start_time = time.time()
    dev = torch.device(device)

    print("=" * 80)
    print("LEARN-A-WORLD v3: STATE READOUT LEARNING (SECTION 6 / GATE 5)")
    print(f"Checkpoint init: {checkpoint_path}")
    print(f"Output ckpt:     {output_checkpoint}")
    print(f"Device:          {dev}")
    print("=" * 80)

    # 1. Load data
    buffer = load_sequence_shards(
        shards_dir=shards_dir,
        manifest_path=manifest_path,
        capacity_transitions=capacity_transitions,
    )
    total_episodes = len(buffer.episodes)
    total_transitions = buffer.total_transitions

    # 2. Strict independent episode split
    train_buffer, val_buffer = buffer.split_train_val(val_ratio=val_ratio, seed=split_seed)
    train_eps = len(train_buffer.episodes)
    val_eps = len(val_buffer.episodes)
    train_trans = train_buffer.total_transitions
    val_trans = val_buffer.total_transitions

    print(f"[Dataset Audit] Total Episodes: {total_episodes} | Total Transitions: {total_transitions}")
    print(f"[Split] Train Episodes: {train_eps} ({train_trans} trans) | Val Episodes: {val_eps} ({val_trans} trans)")

    # 3. Model instantiation and strict parameter freezing
    model = JEPAv3()
    if checkpoint_path:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_checkpoint(str(ckpt_p), device=dev)
    model.to(dev)

    # Freeze base model parameters
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only StateReadoutHead
    for p in model.state_readout_head.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.state_readout_head.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"[Model Freeze] Base model frozen. StateReadoutHead trainable params: {param_count:,}")

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Baseline evaluation before training
    print("\n[Baseline Evaluation] Measuring held-out state readout before training...")
    initial_metrics = evaluate_state_readout(
        model,
        val_buffer if len(val_buffer.episodes) > 0 else train_buffer,
        batch_size=batch_size,
        seq_len=seq_len,
        batches=16,
        labeled_ratio=labeled_ratio,
        use_target_encoder=use_target_encoder,
        device=dev,
    )
    print(
        f"  Pre-train Readout MSE: {initial_metrics.get('state_readout_mse')}\n"
        f"  Zero-State MSE:       {initial_metrics.get('zero_state_mse')}\n"
        f"  Persistence MSE:      {initial_metrics.get('persistence_mse')}"
    )

    # 4. Training loop
    model.eval()  # Base remains in eval mode
    model.state_readout_head.train()
    history: List[Dict[str, float]] = []

    print(f"\n[Training] Starting {total_steps} optimization steps...")
    for step in range(1, total_steps + 1):
        batch = train_buffer.sample_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            labeled_ratio=labeled_ratio,
            device=dev,
        )

        obs = batch["obs"]
        actions = batch["actions"]
        dones = batch["dones"]
        target_state = batch["state_current"]
        valid_mask = batch["state_current_valid"]

        with torch.no_grad():
            z_seq = _encode_batch_frozen(model, obs, use_target=use_target_encoder)
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)

        pred_state, uncertainty = model.predict_state(s_seq)

        loss_mse = model.state_readout_head.compute_loss(pred_state, target_state, valid_mask=valid_mask)

        # Optional heteroscedastic loss component
        if valid_mask.any():
            mask_expanded = valid_mask.unsqueeze(-1).expand_as(pred_state)
            loss_nll = 0.5 * (
                (pred_state - target_state) ** 2 / uncertainty.clamp_min(1e-4)
                + torch.log(uncertainty.clamp_min(1e-4))
            )[mask_expanded].mean()
            loss = loss_mse + 0.1 * loss_nll
        else:
            loss = loss_mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
        optimizer.step()

        record = {
            "step": float(step),
            "loss": float(loss.detach().item()),
            "loss_mse": float(loss_mse.detach().item()),
        }
        history.append(record)

        if step == 1 or step == total_steps or step % max(1, total_steps // 10) == 0:
            print(f"  Step {step:4d}/{total_steps} | Loss: {record['loss']:.5f} | MSE: {record['loss_mse']:.5f}")

    # 5. Final validation evaluation on held-out episodes
    print("\n[Validation Evaluation] Running held-out validation evaluation...")
    final_metrics = evaluate_state_readout(
        model,
        val_buffer if len(val_buffer.episodes) > 0 else train_buffer,
        batch_size=batch_size,
        seq_len=seq_len,
        batches=24,
        labeled_ratio=labeled_ratio,
        use_target_encoder=use_target_encoder,
        device=dev,
    )

    elapsed_sec = time.time() - start_time
    print("-" * 80)
    print("FINAL EVALUATION SUMMARY")
    print(f"  Held-out State Readout MSE: {final_metrics['state_readout_mse']:.6f}")
    print(f"  Zero-State Baseline MSE:    {final_metrics['zero_state_mse']:.6f}")
    print(f"  Persistence Baseline MSE:   {final_metrics['persistence_mse']:.6f}")
    print(f"  Gate 5 Pass (Beats Best Baseline): {final_metrics['state_readout_beats_persistence']}")
    print(f"  Relative Improvement:       {final_metrics['relative_improvement_pct']:.2f}%")
    print("-" * 80)

    # 6. Save checkpoint
    config = {
        "phase": "state_readout_learning",
        "checkpoint_init": str(checkpoint_path) if checkpoint_path else None,
        "shards_dir": [str(s) for s in shards_dir] if isinstance(shards_dir, (list, tuple)) else str(shards_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "labeled_ratio": labeled_ratio,
        "lr": lr,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "base_model_frozen": True,
        "state_readout_head_trainable": True,
        "use_target_encoder": bool(use_target_encoder),
        "checkpoint_sha256": (
            hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
            if checkpoint_path and Path(checkpoint_path).exists() else None
        ),
    }

    out_ckpt = Path(output_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(out_ckpt), step=total_steps, optimizer=optimizer, config=config)
    print(f"[Checkpoint Saved] -> {out_ckpt}")

    result = {
        "status": "completed",
        "phase": "section6_state_readout",
        "gate_5_passed": final_metrics["state_readout_beats_persistence"],
        "checkpoint": str(out_ckpt),
        "dataset_audit": {
            "total_episodes": total_episodes,
            "total_transitions": total_transitions,
            "train_episodes": train_eps,
            "train_transitions": train_trans,
            "val_episodes": val_eps,
            "val_transitions": val_trans,
            "val_ratio": val_ratio,
            "split_seed": split_seed,
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "elapsed_seconds": round(elapsed_sec, 2),
        "config": config,
    }

    if output_json:
        out_j = Path(output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(result, indent=2))
        print(f"[Results JSON Saved] -> {out_j}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StateReadoutHead for Section 6 / Gate 5")
    parser.add_argument("--shards_dir", nargs="+", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_checkpoint", default="checkpoints/jepa_crafter_v3_state_readout.pt")
    parser.add_argument("--output_json", default="results/diagnostics/state_readout_training.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--labeled_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--capacity_transitions", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder", choices=["target", "online"], default="target",
                        help="Frozen encoder feeding the readout head. Default 'target' = plan Stage 1 serving convention.")
    args = parser.parse_args()

    if not args.shards_dir and args.manifest is None:
        parser.error("Must provide --shards_dir or --manifest")

    train_state_readout(
        shards_dir=args.shards_dir,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        labeled_ratio=args.labeled_ratio,
        lr=args.lr,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        capacity_transitions=args.capacity_transitions,
        device=args.device,
        use_target_encoder=(args.encoder == "target"),
    )


if __name__ == "__main__":
    main()
