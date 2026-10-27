#!/usr/bin/env python3
"""Phase 1 training for generic action-conditioned transition auxiliaries.

This module intentionally keeps the JEPA objective visible and active.  It can
resume from an existing v3 checkpoint, trains optional state-delta labels when
they exist in a shard, and always trains reward/termination/event objectives
available in the legacy sequence format.

Inventory/state vectors are targets only.  They are never passed to
``JEPAv3.predict_transition`` or to the planner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """Load deduplicated sequence shards from roots or a Phase 1.5 manifest."""
    if manifest_path is not None:
        shards = selected_paths_from_manifest(manifest_path)
    else:
        roots = [shards_dir] if isinstance(shards_dir, (str, Path)) else list(shards_dir)
        discovery = discover_active_shards(roots)
        shards = [Path(item["path"]) for item in discovery["selected_shards"]]
    if not shards:
        raise FileNotFoundError(f"No sequence shards found in {shards_dir}")
    buffer = EpisodeBuffer(capacity_transitions=capacity_transitions)
    for shard in shards:
        buffer.load_shard(shard)
    return buffer


def _make_model(
    checkpoint_path: Optional[Union[str, Path]],
    device: torch.device,
    transition_state_dim: int,
    allow_fresh_init: bool = False,
) -> JEPAv3:
    model = JEPAv3(transition_state_dim=transition_state_dim)
    if checkpoint_path:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Requested checkpoint does not exist: {checkpoint_path}. "
                "Failing immediately to prevent silent training from random initialization."
            )
        model.load_checkpoint(str(p), device=device)
    else:
        if not allow_fresh_init:
            raise ValueError(
                "No checkpoint_path provided. To train from scratch with random initialization, "
                "allow_fresh_init=True must be explicitly set."
            )
        model.to(device)
    return model


def _encode_batch(model: JEPAv3, obs: torch.Tensor) -> torch.Tensor:
    B, T = obs.shape[:2]
    return model.encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)


def _jepa_multistep_loss(
    model: JEPAv3,
    z_seq: torch.Tensor,
    z_target_seq: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    horizons: Sequence[int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute JEPA latent prediction at each requested valid horizon."""
    B, T = actions.shape
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}
    for horizon in horizons:
        if horizon >= T:
            continue
        z_hat = z_seq[:, 0]
        h_hat = torch.zeros(B, model.recurrent_dim, device=z_seq.device)
        for k in range(1, horizon + 1):
            z_hat, h_hat, _ = model.predict_next_step(z_hat, h_hat, actions[:, k - 1])
        # Do not compare across an episode boundary in a sampled window.
        valid = (~dones[:, :horizon]).all(dim=1).float()
        per_sample = F.mse_loss(z_hat, z_target_seq[:, horizon], reduction="none").mean(dim=-1)
        denom = valid.sum().clamp_min(1.0)
        loss_k = (per_sample * valid).sum() / denom
        losses.append(loss_k)
        metrics[f"latent_mse_{horizon}"] = float(loss_k.detach().item())
    if not losses:
        raise ValueError("Sequence length must be greater than at least one requested horizon")
    return torch.stack(losses).mean(), metrics


def evaluate_transition_heads(
    model: JEPAv3,
    buffer: EpisodeBuffer,
    *,
    batch_size: int = 16,
    seq_len: int = 16,
    batches: int = 16,
    labeled_ratio: float = 0.5,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Evaluate held-out auxiliary predictions and persistence baselines."""
    dev = torch.device(device)
    model.eval()
    delta_sq = 0.0
    persist_sq = 0.0
    delta_count = 0.0
    delta_valid_transitions = 0.0
    reward_abs = 0.0
    reward_count = 0
    done_correct = 0
    done_count = 0
    done_positives = 0
    done_pred_positives = 0
    done_true_positives = 0
    action_separation = 0.0

    latent_std_means: List[float] = []
    latent_std_mins: List[float] = []
    latent_std_maxs: List[float] = []

    with torch.no_grad():
        for _ in range(max(1, batches)):
            batch = buffer.sample_sequences(
                batch_size,
                seq_len,
                mode="uniform",
                labeled_ratio=labeled_ratio,
                device=dev,
            )
            obs, actions = batch["obs"], batch["actions"]
            B, T = actions.shape
            z = _encode_batch(model, obs)

            # Latent variance / collapse metrics
            z_flat = z.reshape(-1, z.shape[-1])
            dim_std = z_flat.std(dim=0)
            latent_std_means.append(float(dim_std.mean().item()))
            latent_std_mins.append(float(dim_std.min().item()))
            latent_std_maxs.append(float(dim_std.max().item()))

            h = model.recurrent_core.unroll_sequence(z, actions, reset_masks=batch["dones"])
            s = torch.cat([z, h], dim=-1)
            pred = model.predict_transition(s, actions)

            valid = batch["state_delta_valid"]
            target_delta = batch["state_delta"]
            if valid.any():
                err = (pred["state_delta"] - target_delta).square().sum(dim=-1)
                base = target_delta.square().sum(dim=-1)
                delta_sq += float(err[valid].sum().item())
                persist_sq += float(base[valid].sum().item())
                delta_count += float(valid.sum().item() * target_delta.shape[-1])
                delta_valid_transitions += float(valid.sum().item())

            reward_abs += float((pred["reward"] - batch["rewards"]).abs().sum().item())
            reward_count += int(batch["rewards"].numel())

            dones = batch["dones"]
            done_pred = pred["done_prob"] >= 0.5
            done_correct += int((done_pred == dones).sum().item())
            done_count += int(dones.numel())
            done_positives += int(dones.sum().item())
            done_pred_positives += int(done_pred.sum().item())
            done_true_positives += int((done_pred & dones).sum().item())

            # Intervention diagnostic: same latent state, different actions.
            s0 = s[:, 0].unsqueeze(1).expand(-1, model.num_actions, -1)
            all_actions = torch.arange(model.num_actions, device=dev).view(1, -1).expand(B, -1)
            all_delta = model.state_delta_head(s0, all_actions)
            action_separation += float(all_delta.var(dim=1).mean().item())

    has_delta = delta_count > 0.0
    state_delta_mse = (delta_sq / delta_count) if has_delta else None
    state_delta_persistence_mse = (persist_sq / delta_count) if has_delta else None
    beats_persist = (
        bool(state_delta_mse < state_delta_persistence_mse)
        if (state_delta_mse is not None and state_delta_persistence_mse is not None)
        else False
    )

    done_negatives = done_count - done_positives
    majority_baseline = max(done_negatives, done_positives) / max(done_count, 1)
    termination_recall = (
        float(done_true_positives / done_positives) if done_positives > 0 else 1.0
    )
    termination_precision = (
        float(done_true_positives / done_pred_positives)
        if done_pred_positives > 0
        else (1.0 if done_positives == 0 else 0.0)
    )

    lat_mean = float(np.mean(latent_std_means)) if latent_std_means else 0.0
    lat_min = float(np.min(latent_std_mins)) if latent_std_mins else 0.0
    lat_max = float(np.max(latent_std_maxs)) if latent_std_maxs else 0.0
    collapse = bool(lat_min < 0.01 or lat_mean < 0.05)

    return {
        "state_delta_mse": state_delta_mse,
        "state_delta_persistence_mse": state_delta_persistence_mse,
        "state_delta_beats_persistence": beats_persist,
        "state_delta_valid_transitions": delta_valid_transitions,
        "reward_mae": reward_abs / max(reward_count, 1),
        "termination_accuracy": done_correct / max(done_count, 1),
        "termination_majority_baseline": majority_baseline,
        "termination_recall": termination_recall,
        "termination_precision": termination_precision,
        "termination_positives": done_positives,
        "termination_true_positives": done_true_positives,
        "counterfactual_action_separation": action_separation / max(1, batches),
        "latent_std_mean": lat_mean,
        "latent_std_min": lat_min,
        "latent_std_max": lat_max,
        "latent_collapse_detected": collapse,
    }


def train_phase1_transitions(
    *,
    shards_dir: Union[str, Path, Sequence[Union[str, Path]]],
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_checkpoint: Union[str, Path] = "checkpoints/jepa_crafter_v3_phase1_transitions.pt",
    output_json: Optional[Union[str, Path]] = None,
    total_steps: int = 1000,
    batch_size: int = 16,
    seq_len: int = 16,
    horizons: Sequence[int] = (1, 5, 10),
    lr: float = 2e-4,
    state_delta_weight: float = 2.0,
    transition_weight: float = 1.0,
    consistency_weight: float = 0.5,
    event_weight: float = 0.3,
    health_weight: float = 0.2,
    variance_weight: float = 0.05,
    labeled_ratio: float = 0.5,
    allow_fresh_init: bool = False,
    device: Union[str, torch.device] = "cpu",
    val_ratio: float = 0.1,
    transition_state_dim: Optional[int] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    split_seed: int = 0,
    capacity_transitions: int = 1_000_000,
) -> Dict[str, Any]:
    """Fine-tune v3 with JEPA plus generic transition losses."""
    dev = torch.device(device)
    if seq_len <= max(horizons):
        raise ValueError(f"seq_len={seq_len} must be > max horizon={max(horizons)}")
    buffer = load_sequence_shards(
        shards_dir,
        capacity_transitions=capacity_transitions,
        manifest_path=manifest_path,
    )
    train_buffer, val_buffer = buffer.split_train_val(val_ratio=val_ratio, seed=split_seed)
    if train_buffer.num_episodes == 0:
        raise ValueError("Training split is empty")
    inferred_state_dim = int(train_buffer.episodes[0]["state_delta"].shape[-1])
    state_dim = int(transition_state_dim or inferred_state_dim)
    if state_dim != inferred_state_dim:
        raise ValueError(
            f"transition_state_dim={state_dim} does not match dataset width={inferred_state_dim}"
        )
    model = _make_model(
        checkpoint_path,
        dev,
        transition_state_dim=state_dim,
        allow_fresh_init=allow_fresh_init,
    )
    for p in model.parameters():
        p.requires_grad = True
    head_params = list(model.state_delta_head.parameters()) + list(model.transition_head.parameters())
    head_param_ids = {id(p) for p in head_params}
    base_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = optim.AdamW([
        {"params": base_params, "lr": lr},
        {"params": head_params, "lr": max(lr, 5e-4)},
    ], weight_decay=1e-4)
    history: List[Dict[str, float]] = []
    start = time.time()

    for step in range(1, total_steps + 1):
        batch = train_buffer.sample_sequences(
            batch_size,
            seq_len,
            mode="event_balanced",
            labeled_ratio=labeled_ratio,
            device=dev,
        )
        obs, actions = batch["obs"], batch["actions"]
        B, T = actions.shape
        z_seq = _encode_batch(model, obs)
        with torch.no_grad():
            z_target = model.target_encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
        h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=batch["dones"])
        s_seq = torch.cat([z_seq, h_seq], dim=-1)

        loss_jepa, latent_metrics = _jepa_multistep_loss(
            model, z_seq, z_target, actions, batch["dones"], horizons
        )
        z_std = torch.sqrt(z_seq.reshape(B * T, -1).var(dim=0) + 1e-4)
        loss_variance = F.relu(0.05 - z_std).mean()

        transition = model.predict_transition(s_seq, actions)
        loss_state = model.state_delta_head.compute_loss(
            transition["state_delta"], batch["state_delta"], batch["state_delta_valid"]
        )
        # Additive multi-step state consistency: predicted consequences should
        # compose over the same horizons used by latent JEPA unrolling.
        consistency_losses: List[torch.Tensor] = []
        for horizon in horizons:
            if horizon >= T:
                continue
            valid_h = batch["state_delta_valid"][:, :horizon].all(dim=1)
            valid_h = valid_h & (~batch["dones"][:, :horizon]).all(dim=1)
            pred_cumulative = transition["state_delta"][:, :horizon].sum(dim=1)
            target_cumulative = batch["state_delta"][:, :horizon].sum(dim=1)
            per_sample = F.mse_loss(
                pred_cumulative, target_cumulative, reduction="none"
            ).mean(dim=-1)
            consistency_losses.append(
                (per_sample * valid_h.float()).sum() / valid_h.float().sum().clamp_min(1.0)
            )
        loss_state_consistency = (
            torch.stack(consistency_losses).mean()
            if consistency_losses
            else torch.zeros((), device=dev)
        )
        loss_transition = model.transition_head.compute_loss(
            transition["reward"],
            transition["done_logits"],
            transition["uncertainty"],
            batch["rewards"],
            batch["dones"],
        )
        loss_event = model.event_head.compute_loss(
            model.event_head(s_seq, actions, batch["mask_before"]), batch["events"]
        )
        event_counts = batch["events"].sum(dim=-1)
        health_target = (batch["rewards"] - event_counts).clamp(-1.0, 1.0)
        loss_health = model.health_head.compute_loss(
            model.health_head(s_seq, actions), health_target
        )

        total = (
            loss_jepa
            + variance_weight * loss_variance
            + state_delta_weight * loss_state
            + consistency_weight * loss_state_consistency
            + transition_weight * loss_transition
            + event_weight * loss_event
            + health_weight * loss_health
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.update_target_networks(decay_encoder=0.99, decay_q=0.995)

        record = {
            "step": float(step),
            "total_loss": float(total.detach().item()),
            "loss_jepa": float(loss_jepa.detach().item()),
            "loss_state_delta": float(loss_state.detach().item()),
            "loss_state_consistency": float(loss_state_consistency.detach().item()),
            "loss_transition": float(loss_transition.detach().item()),
            "loss_event": float(loss_event.detach().item()),
            "loss_health": float(loss_health.detach().item()),
            "loss_variance": float(loss_variance.detach().item()),
        }
        record.update(latent_metrics)
        history.append(record)
        if step == 1 or step == total_steps or step % max(1, min(100, total_steps // 10)) == 0:
            print(
                f"[Phase 1] step {step}/{total_steps} total={record['total_loss']:.4f} "
                f"jepa={record['loss_jepa']:.4f} delta={record['loss_state_delta']:.4f}"
            )

    has_val_windows = any(len(ep["actions"]) >= seq_len for ep in val_buffer.episodes)
    eval_buffer = val_buffer if has_val_windows else train_buffer
    metrics = evaluate_transition_heads(
        model,
        eval_buffer,
        batch_size=batch_size,
        seq_len=seq_len,
        batches=16,
        labeled_ratio=labeled_ratio,
        device=dev,
    )
    config = {
        "phase": "phase1_transition_learning",
        "shards_dir": [str(s) for s in shards_dir] if isinstance(shards_dir, (list, tuple)) else str(shards_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "checkpoint_init": str(checkpoint_path) if checkpoint_path else None,
        "horizons": list(horizons),
        "seq_len": seq_len,
        "labeled_ratio": labeled_ratio,
        "transition_state_dim": state_dim,
        "consistency_weight": consistency_weight,
        "state_delta_labels_are_training_targets_only": True,
        "strict_inference_uses_inventory_input": False,
        "split_seed": split_seed,
    }
    model.save_checkpoint(str(output_checkpoint), step=total_steps, optimizer=optimizer, config=config)
    result = {
        "status": "completed",
        "phase": "phase1",
        "checkpoint": str(output_checkpoint),
        "dataset": train_buffer.get_dataset_statistics(),
        "validation": val_buffer.get_dataset_statistics(),
        "train_labeled_stats": train_buffer.get_labeled_statistics(),
        "val_labeled_stats": val_buffer.get_labeled_statistics(),
        "metrics": metrics,
        "history_tail": history[-10:],
        "elapsed_seconds": time.time() - start,
        "config": config,
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA-v3 generic transition heads")
    parser.add_argument("--shards_dir", nargs="+", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_checkpoint", default="checkpoints/jepa_crafter_v3_phase1_transitions.pt")
    parser.add_argument("--output_json", default="results/phase1_transition_training.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--labeled_ratio", type=float, default=0.5)
    parser.add_argument("--allow_fresh_init", action="store_true")
    parser.add_argument("--transition_state_dim", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--capacity_transitions", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.shards_dir and args.manifest is None:
        parser.error("provide --shards_dir or --manifest")
    result = train_phase1_transitions(
        shards_dir=args.shards_dir,
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        horizons=tuple(args.horizons),
        labeled_ratio=args.labeled_ratio,
        allow_fresh_init=args.allow_fresh_init,
        device=args.device,
        transition_state_dim=args.transition_state_dim,
        manifest_path=args.manifest,
        split_seed=args.split_seed,
        capacity_transitions=args.capacity_transitions,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
