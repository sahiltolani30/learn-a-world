#!/usr/bin/env python3
"""
training/run_q_longhorizon.py — Retrain Q-Ensemble on Full Monte-Carlo Returns
================================================================================

Retrains the value critic (Q-ensemble) on the full future reward using the existing
685k pool with EventCenteredSampler oversampling.

Key Protocol:
1. Starting Checkpoint: checkpoints/jepa_crafter_v3_t2_final.pt.
2. Complete Freezing: Freeze visual encoder, recurrent core, dynamics predictor,
   event head, health head, policy prior, and time-to-event head. Train ONLY Q-ensemble.
3. Targets: Full Monte-Carlo discounted returns to the end of each episode (gamma=0.99).
   No n-step bootstrapping truncation.
4. Loss: Two-hot cross-entropy regression on the return. No auxiliary losses.
5. Hyperparameters: 5,000 steps, batch size 32, frozen encoder.
   Saves: checkpoints/jepa_crafter_v3_q_longhorizon.pt.
6. Honest D1 Verification: Evaluates fresh self-play episodes (seeds 4000..4047)
   and reports overall correlation AND stone-subset correlation (gate: move 0.21 -> >= 0.45).
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.heads import QEnsemble, ReturnNormalizer
from models.jepa_v3 import JEPAv3
from scripts.diagnose_q_calibration_self import run_self_calibration_diagnostic
from training.run_v3_colab_run3 import (
    find_drive_base_shards,
    _load_single_shard,
    _ensure_local_checkpoint,
)
from training.sequence_buffer import EpisodeBuffer

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22


# ──────────────────────────────────────────────────────────────────────────────
# 1. Monte-Carlo Return Calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_monte_carlo_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Computes exact total discounted reward from each state t to the end of its episode:
    G_t = sum_{k=0}^{T - 1 - t} gamma^k * r_{t+k}
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G
        returns[t] = G
    return returns


# ──────────────────────────────────────────────────────────────────────────────
# 2. Buffer Loader for Full 685k Pool (Anti-Eviction Guarantee)
# ──────────────────────────────────────────────────────────────────────────────

def build_longhorizon_buffer(
    structured_random_final: str,
    teacher_hybrid_final: str,
    planner_v2_final: str,
    curiosity_dir: str,
    capacity_transitions: int = 1_000_000,
    gamma: float = 0.99,
) -> Tuple[EpisodeBuffer, Dict[str, Any]]:
    """
    Loads base demonstration snapshots and all curiosity shards (capacity 1,000,000).
    Precomputes full Monte-Carlo discounted returns on all episodes.
    Asserts zero silent evictions.
    """
    print("\n" + "=" * 80)
    print("BUFFER LOADER: FULL 685K POOL FOR LONG-HORIZON Q RETRAINING")
    print("=" * 80)
    print(f"Capacity Limit : {capacity_transitions:,} transitions")
    print(f"Base Snapshots : {Path(structured_random_final).name}, {Path(teacher_hybrid_final).name}, {Path(planner_v2_final).name}")
    print(f"Curiosity Pool : {curiosity_dir}")
    print(f"Discount Gamma : {gamma}")
    print("=" * 80)

    gc.collect()
    buffer = EpisodeBuffer(capacity_transitions=capacity_transitions)
    raw_loaded = 0
    per_source_counts: Dict[str, int] = {}

    # 1. Base Demonstrations
    n_struct = _load_single_shard(structured_random_final, "structured_random", buffer)
    raw_loaded += n_struct
    per_source_counts["structured_random"] = n_struct

    n_teacher = _load_single_shard(teacher_hybrid_final, "teacher_hybrid", buffer)
    raw_loaded += n_teacher
    per_source_counts["teacher_hybrid"] = n_teacher

    n_planner = _load_single_shard(planner_v2_final, "planner_v2", buffer)
    raw_loaded += n_planner
    per_source_counts["planner_v2"] = n_planner

    task_total = n_struct + n_teacher + n_planner

    # 2. Curiosity Shards
    shard_files = sorted(list(Path(curiosity_dir).glob("*.npz")))
    if not shard_files:
        # Check subdirectories
        shard_files = sorted(list(Path(curiosity_dir).glob("**/*.npz")))

    n_curiosity_total = 0
    for sf in shard_files:
        n_cur = _load_single_shard(str(sf), "curiosity", buffer)
        n_curiosity_total += n_cur
    raw_loaded += n_curiosity_total
    per_source_counts["curiosity"] = n_curiosity_total

    per_source_counts["total_transitions"] = raw_loaded
    per_source_counts["total_episodes"] = buffer.num_episodes

    print("-" * 80)
    print("LOADED TOTALS PER SOURCE:")
    print(f"  - Structured Random  : {n_struct:>9,} transitions")
    print(f"  - Teacher Hybrid     : {n_teacher:>9,} transitions")
    print(f"  - Planner v2         : {n_planner:>9,} transitions")
    print(f"  - Task Demonstrations: {task_total:>9,} transitions")
    print(f"  - Curiosity Shards   : {n_curiosity_total:>9,} transitions ({len(shard_files)} shards)")
    print(f"  - TOTAL IN BUFFER    : {buffer.total_transitions:>9,} transitions across {buffer.num_episodes:,} episodes")
    print("-" * 80)

    # Hard assert against silent eviction
    assert buffer.total_transitions == raw_loaded, (
        f"[FATAL AUDIT FAILURE] Silent eviction detected! Added {raw_loaded:,} transitions "
        f"but buffer holds {buffer.total_transitions:,}."
    )
    assert buffer.total_transitions <= capacity_transitions, (
        f"[FATAL AUDIT FAILURE] Total loaded transitions ({buffer.total_transitions:,}) "
        f"exceeds capacity ({capacity_transitions:,})!"
    )
    print("  ✅ BUFFER INTEGRITY ASSERTION PASSED: 100% data retained with zero eviction.")

    # Precompute Monte-Carlo returns for every episode
    print(f"\n[MC-Return] Precomputing Monte-Carlo returns across {len(buffer.episodes):,} episodes (gamma={gamma})...")
    t_mc = time.time()
    for ep in buffer.episodes:
        ep["mc_returns"] = compute_monte_carlo_returns(ep["rewards"], gamma=gamma)
    print(f"  ✅ Monte-Carlo returns precomputed in {time.time() - t_mc:.2f}s.")

    return buffer, per_source_counts


# ──────────────────────────────────────────────────────────────────────────────
# 3. Event-Centered Sampler with Inverse-Frequency Balancing
# ──────────────────────────────────────────────────────────────────────────────

class EventCenteredSampler:
    """
    Indexes achievement moments across all 22 achievements (0..21).
    Samples contiguous windows centered on achievement triggers with inverse-frequency
    balancing to oversample rare moments (stone, pickaxe, combat) without hardcoding.
    """

    def __init__(self, buffer: EpisodeBuffer, seq_len: int = 32):
        self.buffer = buffer
        self.seq_len = seq_len
        self.total_stone_sampled_transitions = 0
        self.total_sampled_transitions = 0
        self._build_event_windows()

    def _build_event_windows(self) -> None:
        print(f"\n[Sampler] Indexing event-centered windows (seq_len={self.seq_len})...")
        t0 = time.time()

        type_counts = np.zeros(NUM_ACHIEVEMENTS, dtype=np.int64)
        for ep in self.buffer.episodes:
            type_counts += ep["events"].sum(axis=0).astype(np.int64)

        self.type_counts = type_counts
        rare_weights = 1.0 / np.maximum(type_counts, 1.0)
        rare_weights /= np.sum(rare_weights)

        event_windows = []
        event_weights = []
        uniform_windows = []

        half_seq = self.seq_len // 2

        for ep_idx, ep in enumerate(self.buffer.episodes):
            T = len(ep["actions"])
            if T < self.seq_len:
                continue

            for start_t in range(T - self.seq_len + 1):
                uniform_windows.append((ep_idx, start_t))

            events = ep["events"]
            ach_steps, ach_types = np.where(events > 0)
            for step_t, ach_type in zip(ach_steps, ach_types):
                start_t = max(0, min(step_t - half_seq, T - self.seq_len))
                event_windows.append((ep_idx, start_t))
                event_weights.append(float(rare_weights[ach_type]))

        self.uniform_windows = uniform_windows
        self.event_windows = event_windows

        if event_weights:
            p_arr = np.array(event_weights, dtype=np.float64)
            p_arr /= np.sum(p_arr)
            self.event_probs = p_arr
        else:
            self.event_probs = None

        print(f"  Indexed {len(self.uniform_windows):,} uniform windows and {len(self.event_windows):,} event-centered windows ({time.time() - t0:.1f}s).")
        print(f"  Stone Events in Buffer: {type_counts[IDX_STONE]} | Wood Pickaxe: {type_counts[IDX_WOOD_PICKAXE]}")

    def sample_batch(
        self,
        batch_size: int = 32,
        event_ratio: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        n_event = int(round(batch_size * event_ratio)) if self.event_windows and self.event_probs is not None else 0
        n_uniform = batch_size - n_event

        chosen = []
        if n_event > 0:
            e_idx = np.random.choice(len(self.event_windows), size=n_event, replace=True, p=self.event_probs)
            for idx in e_idx:
                chosen.append(self.event_windows[idx])

        if n_uniform > 0:
            u_idx = np.random.choice(len(self.uniform_windows), size=n_uniform, replace=True)
            for idx in u_idx:
                chosen.append(self.uniform_windows[idx])

        b_obs = np.empty((batch_size, self.seq_len, 64, 64, 3), dtype=np.uint8)
        b_actions = np.empty((batch_size, self.seq_len), dtype=np.int64)
        b_dones = np.empty((batch_size, self.seq_len), dtype=np.bool_)
        b_returns = np.empty((batch_size, self.seq_len), dtype=np.float32)
        b_events = np.empty((batch_size, self.seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)

        for i, (ep_idx, start_t) in enumerate(chosen):
            ep = self.buffer.episodes[ep_idx]
            end_t = start_t + self.seq_len
            b_obs[i] = ep["obs"][start_t:end_t]
            b_actions[i] = ep["actions"][start_t:end_t]
            b_dones[i] = ep["dones"][start_t:end_t]
            b_returns[i] = ep["mc_returns"][start_t:end_t]
            b_events[i] = ep["events"][start_t:end_t]

        stone_count = int(np.sum(b_events[:, :, IDX_STONE] > 0))
        self.total_stone_sampled_transitions += stone_count
        self.total_sampled_transitions += (batch_size * self.seq_len)

        return {
            "obs": torch.from_numpy(b_obs).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device),
            "actions": torch.from_numpy(b_actions).long().to(device),
            "dones": torch.from_numpy(b_dones).bool().to(device),
            "mc_returns": torch.from_numpy(b_returns).float().to(device),
        }

    def get_effective_stone_sampling_rate(self) -> float:
        if self.total_sampled_transitions == 0:
            return 0.0
        return (self.total_stone_sampled_transitions / self.total_sampled_transitions) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Long-Horizon Q Retraining (Frozen Encoder & Dynamics, Q-Only Loss)
# ──────────────────────────────────────────────────────────────────────────────

def retrain_q_longhorizon(
    model: JEPAv3,
    sampler: EventCenteredSampler,
    total_steps: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Retrains ONLY the Q-ensemble on full Monte-Carlo returns for 5,000 steps.
    Encoder, recurrent core, dynamics, and all other heads are completely frozen.
    """
    dev = torch.device(device)
    model.to(dev)

    print("\n" + "=" * 80)
    print("Q-ENSEMBLE LONG-HORIZON RETRAINING PROTOCOL")
    print("=" * 80)
    print(f"Total Steps       : {total_steps:,}")
    print(f"Batch Size        : {batch_size}")
    print(f"Learning Rate     : {lr}")
    print(f"Target Type       : Full Monte-Carlo Returns (gamma=0.99, NO n-step bootstrapping)")
    print(f"Loss Function     : Two-hot Cross-Entropy on Q-Ensemble ONLY")
    print(f"Device            : {dev}")
    print("=" * 80)

    # 1. Put visual encoders in eval mode
    model.encoder.eval()
    model.target_encoder.eval()

    # 2. Freeze ALL parameters in the model
    for p in model.parameters():
        p.requires_grad = False

    # 3. Unfreeze ONLY QEnsemble
    for p in model.q_ensemble.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    q_params = [p for p in model.q_ensemble.parameters() if p.requires_grad]
    assert len(trainable_params) == len(q_params), "[FATAL AUDIT FAILURE] Non-Q parameters are unfrozen!"

    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable Parameters (Q-Ensemble Only): {n_trainable:,}")
    print(f"  Frozen Parameters (Encoder + Dyn + Heads): {n_frozen:,}")
    print("  ✅ PARAMETER FREEZING AUDIT PASSED: 100% isolated to Q-ensemble.")
    print("-" * 80)

    optimizer = optim.AdamW(model.q_ensemble.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    losses = []
    print_every = 250

    for step in range(1, total_steps + 1):
        batch = sampler.sample_batch(batch_size=batch_size, event_ratio=0.5, device=dev)
        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        mc_returns = batch["mc_returns"]  # [B, T]
        B, T = actions.shape

        # Compute latent state representations strictly with no grad
        with torch.no_grad():
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_seq = model.encoder(obs_flat).view(B, T, -1)
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)

            # Update ReturnNormalizer and normalize MC returns for two-hot binning
            if hasattr(model, "return_normalizer") and model.return_normalizer is not None:
                model.return_normalizer.update(mc_returns)
                q_fit_targets = model.return_normalizer.normalize(mc_returns)
            else:
                q_fit_targets = mc_returns

        # Forward Q-Ensemble
        q_logits = model.q_ensemble(s_seq, actions)  # [num_heads, B, T, num_bins]
        loss_q = model.q_ensemble.compute_loss(q_logits, q_fit_targets)

        optimizer.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(model.q_ensemble.parameters(), max_norm=1.0)
        optimizer.step()

        # Update target Q network via Polyak averaging
        model.update_target_networks(decay_encoder=1.0, decay_q=0.995)

        losses.append(float(loss_q.item()))

        if step % print_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            stone_rate = sampler.get_effective_stone_sampling_rate()
            mean_loss = float(np.mean(losses[-print_every:]))
            print(
                f"Step {step:5d}/{total_steps} ({elapsed:5.1f}s) | "
                f"Q Loss: {mean_loss:.4f} | "
                f"Stone Sampling Rate: {stone_rate:.2f}%"
            )

    train_duration = time.time() - t0
    print(f"\n--> Q Retraining completed in {train_duration:.1f}s ({train_duration / 60:.1f} min).")

    return {
        "final_q_loss": float(np.mean(losses[-100:])),
        "train_duration_sec": train_duration,
        "effective_stone_rate": sampler.get_effective_stone_sampling_rate(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main Execution Pipeline (Train + D1 Verification Gate)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain Q-Ensemble on full Monte-Carlo returns and verify D1.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_t2_final.pt",
        help="Starting checkpoint path",
    )
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        default="checkpoints/jepa_crafter_v3_q_longhorizon.pt",
        help="Where to save the retrained checkpoint",
    )
    parser.add_argument("--steps", type=int, default=5000, help="Training steps (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence window length (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run D1 verification only")
    args = parser.parse_args()

    dev = args.device

    # Resolve paths
    ckpt_path = _ensure_local_checkpoint(args.checkpoint, args.drive_dir)
    print(f"[Init] Starting checkpoint resolved to: {ckpt_path}")

    model = JEPAv3()
    model.load_checkpoint(ckpt_path, device=dev)
    model.to(torch.device(dev))

    # Resolve output checkpoint paths
    local_out_ckpt = REPO_ROOT / args.output_checkpoint
    local_out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    drive_out_ckpt = Path(args.drive_dir) / "checkpoints" / Path(args.output_checkpoint).name if args.drive_dir else None

    if not args.skip_train:
        # Resolve data paths for 685k pool
        drive_path = Path(args.drive_dir) if args.drive_dir and Path(args.drive_dir).exists() else REPO_ROOT
        base_shards = find_drive_base_shards(str(drive_path))

        curiosity_dir = drive_path / "data" / "sequences"
        if not curiosity_dir.exists():
            curiosity_dir = REPO_ROOT / "data" / "sequences"

        # Load buffer and precompute Monte-Carlo returns
        buffer, counts = build_longhorizon_buffer(
            structured_random_final=base_shards["structured_random_final"],
            teacher_hybrid_final=base_shards["teacher_hybrid_final"],
            planner_v2_final=base_shards["planner_v2_final"],
            curiosity_dir=str(curiosity_dir),
            capacity_transitions=1_000_000,
            gamma=args.gamma,
        )

        sampler = EventCenteredSampler(buffer=buffer, seq_len=args.seq_len)

        # Train Q-only
        train_stats = retrain_q_longhorizon(
            model=model,
            sampler=sampler,
            total_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=dev,
        )

        # Save checkpoint locally and to Google Drive
        model.save_checkpoint(str(local_out_ckpt))
        print(f"\n[Checkpoint] Saved retrained model to: {local_out_ckpt}")

        if drive_out_ckpt:
            try:
                drive_out_ckpt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(local_out_ckpt), str(drive_out_ckpt))
                print(f"[Checkpoint] Synced retrained model to Google Drive: {drive_out_ckpt}")
            except Exception as e:
                print(f"[Warning] Failed to sync checkpoint to Drive: {e}")

    eval_ckpt = str(local_out_ckpt if local_out_ckpt.exists() else ckpt_path)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Honest D1 Verification Gate (Fresh Self-Play Seeds 4000..4047)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: HONEST D1 VERIFICATION GATE (SEEDS 4000..4047)")
    print("=" * 80)
    print(f"Checkpoint to Evaluate : {eval_ckpt}")
    print(f"Evaluation Seeds       : 4000..4047 (48 fresh self-play episodes)")
    print(f"Gate Criterion         : Stone-Subset Pearson r must move from 0.21 -> >= 0.45")
    print("=" * 80)

    calibrator_path = REPO_ROOT / "results" / "calibration" / "event_head_calibrator.json"
    d1_json = REPO_ROOT / "results" / "diagnostics" / "q_longhorizon_d1.json"
    d1_png = REPO_ROOT / "results" / "diagnostics" / "q_longhorizon_d1.png"

    d1_metrics = run_self_calibration_diagnostic(
        checkpoint=eval_ckpt,
        calibrator_path=str(calibrator_path) if calibrator_path.exists() else None,
        num_episodes=48,
        seed_start=4000,
        device=dev,
        output_json=str(d1_json),
        output_png=str(d1_png),
        drive_dir=args.drive_dir,
        force=True,  # Strict guarantee: fresh self-play episodes generated on the new model
    )

    r_overall = d1_metrics.get("pearson_r", 0.0)
    stone_sub = d1_metrics.get("stone_subset", {})
    r_stone = stone_sub.get("pearson_r", 0.0)
    n_stone_trans = stone_sub.get("num_transitions", 0)

    print("\n" + "=" * 80)
    print("                    D1 VERIFICATION GATE REPORT")
    print("=" * 80)
    print(f"Overall Pearson r (Informational)  : {r_overall:.4f}")
    print(f"Stone-Subset Transitions           : {n_stone_trans}")
    print(f"Stone-Subset Pearson r (THE GATE)  : {r_stone:.4f}  (Baseline was 0.2127)")
    print("-" * 80)

    if r_stone >= 0.45:
        print("  🟢 GATE STATUS: PASSED! (r_stone >= 0.45)")
        print("  --> Stone progression chain learned! Proceed to screening.")
    elif r_stone > 0.30:
        print(f"  🟡 GATE STATUS: PARTIAL PROGRESS (r_stone = {r_stone:.4f} > 0.21, but < 0.45)")
        print("  --> Significant upward movement on stone subset.")
    else:
        print(f"  🔴 GATE STATUS: FAILED TO MOVE (r_stone = {r_stone:.4f} <= 0.21)")
        print("  --> Data too thin for stone chain. Proceed to T3 data collection.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
