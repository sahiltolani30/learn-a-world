#!/usr/bin/env python3
"""
training/run_v3_colab_step1_stone.py — Phase 3 Step 1: Stone Progression Fine-Tune
===================================================================================

Implements the complete Step 1 architectural fine-tuning protocol:
1. Starting Checkpoint: checkpoints/jepa_crafter_v3_t2_final.pt.
2. 1,000,000-Transition Capacity Loader:
   - Explicit hard assert: total_transitions == raw_loaded <= 1,000,000.
   - Silent eviction is strictly banned.
   - Reports loaded transitions per source family.
3. Event-Centered Balanced Sampling:
   - Contiguous windows centered on achievement moments across all 22 achievements.
   - Generic inverse-frequency weighting across achievement indices (no hardcoded names).
   - Real-time logging of effective stone-transition sampling rate in batches.
4. Longer-Horizon Value Targets:
   - n-step Bellman targets with n=20 (longer than current 10) to bridge pickaxe-to-stone horizon.
5. Generic Time-To-Event Head:
   - Predicts discounted time-to-event proximity tau in [0, 1] for all 22 achievements.
   - Operates strictly by index 0..21 (zero object-specific code).
6. Two-Phase Training:
   - Phase A: Frozen-encoder value & time-to-event warmup (5,000 steps).
   - Self-slice Q-calibration gate check (target r > 0.30).
   - Phase B: Full joint fine-tuning (10,000 steps).

DO NOT START TRAINING UNTIL CODE IS AUDITED AND PART A RESULTS ARE REVIEWED.
"""

import argparse
import copy
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

from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.heads import ReturnNormalizer, TimeToEventHead
from models.jepa_v3 import JEPAv3
from training.sequence_buffer import EpisodeBuffer
from training.run_step1_value_fix import compute_n_step_bellman_targets
from training.run_v3_colab_run3 import (
    find_drive_base_shards,
    verify_joint_shards,
    _load_single_shard,
    _ensure_local_checkpoint,
)

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22


# ──────────────────────────────────────────────────────────────────────────────
# 1. Capacity-Guaranteed Buffer Loader (1,000,000 Capacity, Anti-Eviction)
# ──────────────────────────────────────────────────────────────────────────────

def build_step1_buffer(
    structured_random_final: str,
    teacher_hybrid_final: str,
    planner_v2_final: str,
    curiosity_dir: str,
    capacity_transitions: int = 1_000_000,
) -> Tuple[EpisodeBuffer, Dict[str, Any]]:
    """
    Loads base demonstration snapshots and all curiosity shards with a 1,000,000 capacity.
    Hard-asserts that zero silent evictions occur during loading.
    """
    print("\n" + "=" * 80)
    print("STEP 1 BUFFER LOADER (1,000,000 CAPACITY | ZERO-SILENT-EVICTION GUARANTEE)")
    print("=" * 80)
    print(f"Target Capacity : {capacity_transitions:,} transitions")
    print(f"Base Snapshots  : {Path(structured_random_final).name}, {Path(teacher_hybrid_final).name}, {Path(planner_v2_final).name}")
    print(f"Curiosity Pool  : {curiosity_dir}")
    print("=" * 80)

    gc.collect()
    buffer = EpisodeBuffer(capacity_transitions=capacity_transitions)
    raw_loaded = 0
    per_source_counts: Dict[str, int] = {}

    # 1. Base Demonstrations
    n_struct = _load_single_shard(structured_random_final, "structured_random", buffer)
    raw_loaded += n_struct
    per_source_counts["structured_random"] = n_struct
    print(f"  [Loader] structured_random : {n_struct:>9,} transitions")

    n_teacher = _load_single_shard(teacher_hybrid_final, "teacher_hybrid", buffer)
    raw_loaded += n_teacher
    per_source_counts["teacher_hybrid"] = n_teacher
    print(f"  [Loader] teacher_hybrid    : {n_teacher:>9,} transitions")

    n_planner = _load_single_shard(planner_v2_final, "planner_v2", buffer)
    raw_loaded += n_planner
    per_source_counts["planner_v2"] = n_planner
    print(f"  [Loader] planner_v2        : {n_planner:>9,} transitions")

    # 2. Curiosity Exploration Shards
    cur_p = Path(curiosity_dir)
    shard_files = sorted(cur_p.glob("*.npz"))
    print(f"  [Loader] Loading {len(shard_files)} curiosity shards...")

    n_curiosity_total = 0
    t0 = time.time()
    for idx, sf in enumerate(shard_files):
        n_shard = _load_single_shard(str(sf), "curiosity", buffer)
        raw_loaded += n_shard
        n_curiosity_total += n_shard
        if (idx + 1) % 40 == 0 or (idx + 1) == len(shard_files):
            print(f"    Shards {idx + 1:3d}/{len(shard_files)} loaded ({time.time() - t0:.1f}s) -> {n_curiosity_total:>9,} curiosity transitions")

    per_source_counts["curiosity"] = n_curiosity_total
    task_total = n_struct + n_teacher + n_planner
    per_source_counts["task_total"] = task_total
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

    # HARD ASSERTS: Strict Anti-Eviction and Capacity Integrity
    assert buffer.total_transitions == raw_loaded, (
        f"[FATAL AUDIT FAILURE] Silent eviction detected! Added {raw_loaded:,} transitions "
        f"but buffer holds {buffer.total_transitions:,}. Difference of {raw_loaded - buffer.total_transitions:,} "
        f"transitions were silently dropped! Silent eviction is strictly banned."
    )
    assert buffer.total_transitions <= capacity_transitions, (
        f"[FATAL AUDIT FAILURE] Total loaded transitions ({buffer.total_transitions:,}) "
        f"exceeds buffer capacity ({capacity_transitions:,})!"
    )
    print("  ✅ BUFFER INTEGRITY ASSERTION PASSED: 100% data retained with zero silent eviction.")

    return buffer, per_source_counts


# ──────────────────────────────────────────────────────────────────────────────
# 2. Event-Centered Balanced Sampler (Achievement Moments & Effective Stone Rate)
# ──────────────────────────────────────────────────────────────────────────────

class EventCenteredSampler:
    """
    Indexes achievement moments across all 22 achievements (by index 0..21).
    Samples contiguous windows centered on achievement triggers with inverse-frequency
    balancing to oversample rare moments (stone, pickaxe, combat) without hardcoded names.
    Tracks and logs effective stone-transition sampling rates in training batches.
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

        # Count events per type across buffer
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

            # Standard uniform windows
            for start_t in range(T - self.seq_len + 1):
                uniform_windows.append((ep_idx, start_t))

            # Achievement-centered windows
            events = ep["events"]
            ach_steps, ach_types = np.where(events > 0)

            for step_t, ach_type in zip(ach_steps, ach_types):
                # Center window on the achievement trigger
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
        batch_size: int = 16,
        event_ratio: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Samples batch with event_ratio fraction centered on achievement moments.
        Tracks effective stone-transition sampling rate.
        """
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

        # Collate
        b_obs = np.empty((batch_size, self.seq_len, 64, 64, 3), dtype=np.uint8)
        b_actions = np.empty((batch_size, self.seq_len), dtype=np.int64)
        b_rewards = np.empty((batch_size, self.seq_len), dtype=np.float32)
        b_dones = np.empty((batch_size, self.seq_len), dtype=np.bool_)
        b_events = np.empty((batch_size, self.seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)
        b_mask = np.empty((batch_size, self.seq_len, NUM_ACHIEVEMENTS), dtype=np.uint8)

        # Time-to-event targets: [B, T, 22] discounted proximity gamma^Delta_t
        b_tte_targets = np.zeros((batch_size, self.seq_len, NUM_ACHIEVEMENTS), dtype=np.float32)
        gamma_tte = 0.95

        for i, (ep_idx, start_t) in enumerate(chosen):
            ep = self.buffer.episodes[ep_idx]
            end_t = start_t + self.seq_len
            b_obs[i] = ep["obs"][start_t:end_t]
            b_actions[i] = ep["actions"][start_t:end_t]
            b_rewards[i] = ep["rewards"][start_t:end_t]
            b_dones[i] = ep["dones"][start_t:end_t]
            b_events[i] = ep["events"][start_t:end_t]
            b_mask[i] = ep["mask_before"][start_t:end_t]

            # Compute forward time-to-event for each achievement index 0..21
            ep_ev = ep["events"]  # [T_ep, 22]
            T_ep = len(ep_ev)
            for ach_idx in range(NUM_ACHIEVEMENTS):
                future_occurrences = np.where(ep_ev[start_t:T_ep, ach_idx] > 0)[0]
                if len(future_occurrences) > 0:
                    for t_local in range(self.seq_len):
                        t_global = start_t + t_local
                        next_steps = np.where(ep_ev[t_global:T_ep, ach_idx] > 0)[0]
                        if len(next_steps) > 0:
                            delta_t = next_steps[0]
                            b_tte_targets[i, t_local, ach_idx] = float(gamma_tte ** delta_t)

        # Track stone sampling stats
        stone_count = int(np.sum(b_events[:, :, IDX_STONE] > 0))
        self.total_stone_sampled_transitions += stone_count
        self.total_sampled_transitions += (batch_size * self.seq_len)

        return {
            "obs": torch.from_numpy(b_obs).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device),
            "actions": torch.from_numpy(b_actions).long().to(device),
            "rewards": torch.from_numpy(b_rewards).float().to(device),
            "dones": torch.from_numpy(b_dones).bool().to(device),
            "events": torch.from_numpy(b_events).float().to(device),
            "mask_before": torch.from_numpy(b_mask).float().to(device),
            "tte_targets": torch.from_numpy(b_tte_targets).float().to(device),
        }

    def get_effective_stone_sampling_rate(self) -> float:
        """Returns overall percentage of sampled transitions containing stone events."""
        if self.total_sampled_transitions == 0:
            return 0.0
        return (self.total_stone_sampled_transitions / self.total_sampled_transitions) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Step 1 Training Execution Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_step1_value_phase(
    model: JEPAv3,
    sampler: EventCenteredSampler,
    total_steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 32,
    lr: float = 1e-4,
    n_step: int = 20,
    gamma: float = 0.99,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Phase A: Frozen visual encoder value & time-to-event training (5,000 steps).
    Uses n-step=20 Bellman targets and event-centered balanced windows.
    """
    print("\n" + "=" * 80)
    print(f"STEP 1 PHASE A — FROZEN-ENCODER VALUE & TTE TRAINING ({total_steps:,} STEPS)")
    print(f"  Value Targets Horizon: n-step = {n_step} (Extended from n=10)")
    print(f"  Sequence Length      : {seq_len} steps")
    print(f"  Encoder State        : FROZEN")
    print(f"  Time-To-Event Head   : ACTIVE (all 22 achievements by index)")
    print("=" * 80)

    dev = torch.device(device)
    model.to(dev)

    # Freeze online visual encoder
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Trainable parameters: QEnsemble, TimeToEventHead, HealthHead
    trainable_params = list(model.q_ensemble.parameters()) + list(model.time_to_event_head.parameters()) + list(model.health_head.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    t0 = time.time()
    losses = []

    for step in range(1, total_steps + 1):
        batch = sampler.sample_batch(batch_size=batch_size, event_ratio=0.5, device=dev)
        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        tte_targets = batch["tte_targets"]  # [B, T, 22]
        B, T = actions.shape

        with torch.no_grad():
            obs_flat = obs.view(B * T, 3, 64, 64)
            z_seq = model.encoder(obs_flat).view(B, T, -1)
            h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
            s_seq = torch.cat([z_seq, h_seq], dim=-1)

            # Target Q-values for n-step=20 Bellman targets
            target_q_min = model.q_target_ensemble.q_min(s_seq, actions)
            bellman_targets = compute_n_step_bellman_targets(
                rewards=rewards,
                dones=dones,
                target_q_vals=target_q_min,
                gamma=gamma,
                n_step=n_step,
            )

        # 1. Critic Loss
        loss_q = model.q_ensemble.compute_loss(s_seq, actions, bellman_targets)

        # 2. Time-To-Event Loss
        pred_tte = model.time_to_event_head(s_seq)
        loss_tte = model.time_to_event_head.compute_loss(pred_tte, tte_targets)

        total_loss = 2.0 * loss_q + 1.0 * loss_tte

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        # Polyak target network update
        model.update_target_networks(decay_encoder=1.0, decay_q=0.995)

        losses.append(float(total_loss.item()))

        if step % 500 == 0 or step == total_steps:
            elapsed = time.time() - t0
            stone_rate = sampler.get_effective_stone_sampling_rate()
            print(
                f"Step {step:5d}/{total_steps} ({elapsed:.1f}s) | "
                f"Loss: {total_loss.item():.4f} (Q: {loss_q.item():.4f}, TTE: {loss_tte.item():.4f}) | "
                f"Effective Stone Sampling Rate: {stone_rate:.2f}%"
            )

    return {
        "final_loss": float(np.mean(losses[-100:])),
        "effective_stone_rate": sampler.get_effective_stone_sampling_rate(),
    }


def train_step1_joint_phase(
    model: JEPAv3,
    sampler: EventCenteredSampler,
    total_steps: int = 10000,
    batch_size: int = 16,
    seq_len: int = 32,
    lr: float = 1e-4,
    n_step: int = 20,
    gamma: float = 0.99,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Phase B: Full joint fine-tuning (10,000 steps).
    Unfreezes all parameters with joint loss (JEPA + Dyn + Event + Value + TTE).
    """
    print("\n" + "=" * 80)
    print(f"STEP 1 PHASE B — FULL JOINT FINE-TUNING ({total_steps:,} STEPS)")
    print(f"  All Parameters       : UNFROZEN (End-to-End Task Shaping)")
    print(f"  Value Targets Horizon: n-step = {n_step}")
    print(f"  Event-Centered Ratio : 50% achievement-focused windows")
    print("=" * 80)

    dev = torch.device(device)
    model.to(dev)

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time()
    losses = []

    for step in range(1, total_steps + 1):
        batch = sampler.sample_batch(batch_size=batch_size, event_ratio=0.5, device=dev)
        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        events = batch["events"]    # [B, T, 22]
        masks = batch["mask_before"] # [B, T, 22]
        tte_targets = batch["tte_targets"]
        B, T = actions.shape

        obs_flat = obs.view(B * T, 3, 64, 64)
        z_seq = model.encoder(obs_flat).view(B, T, -1)
        h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
        s_seq = torch.cat([z_seq, h_seq], dim=-1)

        # 1. Event Loss (Focal BCE)
        loss_event = model.event_head.compute_loss(s_seq, actions, masks, events)

        # 2. Time-To-Event Loss
        pred_tte = model.time_to_event_head(s_seq)
        loss_tte = model.time_to_event_head.compute_loss(pred_tte, tte_targets)

        # 3. Value Loss with n-step=20 targets
        with torch.no_grad():
            target_q_min = model.q_target_ensemble.q_min(s_seq, actions)
            bellman_targets = compute_n_step_bellman_targets(
                rewards=rewards,
                dones=dones,
                target_q_vals=target_q_min,
                gamma=gamma,
                n_step=n_step,
            )
        loss_q = model.q_ensemble.compute_loss(s_seq, actions, bellman_targets)

        # 4. Latent Dynamics Loss
        with torch.no_grad():
            z_target = model.target_encoder(obs_flat).view(B, T, -1)
        z_pred = model.predictor(z_seq[:, :-1], h_seq[:, :-1], actions[:, :-1])
        loss_dyn = F.mse_loss(z_pred, z_target[:, 1:])

        total_loss = (
            0.5 * loss_dyn +
            0.3 * loss_event +
            1.5 * loss_q +
            0.8 * loss_tte
        )

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.update_target_networks(decay_encoder=0.99, decay_q=0.995)
        losses.append(float(total_loss.item()))

        if step % 500 == 0 or step == total_steps:
            elapsed = time.time() - t0
            stone_rate = sampler.get_effective_stone_sampling_rate()
            print(
                f"Step {step:5d}/{total_steps} ({elapsed:.1f}s) | "
                f"Loss: {total_loss.item():.4f} (Q: {loss_q.item():.4f}, TTE: {loss_tte.item():.4f}, Ev: {loss_event.item():.4f}) | "
                f"Effective Stone Sampling Rate: {stone_rate:.2f}%"
            )

    return {
        "final_loss": float(np.mean(losses[-100:])),
        "effective_stone_rate": sampler.get_effective_stone_sampling_rate(),
    }


def evaluate_self_slice_calibration(
    model: JEPAv3,
    sampler: EventCenteredSampler,
    num_batches: int = 25,
    batch_size: int = 16,
    gamma: float = 0.99,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluates Pearson r and MAE between predicted Q-values and realized returns
    on held-out / sampled windows from the active buffer.
    """
    dev = torch.device(device)
    model.eval()
    all_q_pred = []
    all_realized = []

    with torch.no_grad():
        for _ in range(num_batches):
            batch = sampler.sample_batch(batch_size=batch_size, event_ratio=0.5, device=dev)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            B, T = actions.shape

            obs_flat = obs.view(B * T, 3, 64, 64)
            z = model.encode(obs_flat, use_target=True).view(B, T, -1)
            h_seq = model.recurrent_core.unroll_sequence(z, actions, reset_masks=dones)
            s_seq = torch.cat([z, h_seq], dim=-1)

            q_pred = model.q_mean(s_seq, actions)
            if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                q_pred = model.return_normalizer.denormalize(q_pred)

            realized = torch.zeros_like(rewards)
            running = torch.zeros(B, device=dev)
            for t in reversed(range(T)):
                running = rewards[:, t] + gamma * (1.0 - dones[:, t].float()) * running
                realized[:, t] = running

            all_q_pred.extend(q_pred.cpu().numpy().flatten().tolist())
            all_realized.extend(realized.cpu().numpy().flatten().tolist())

    q_arr = np.array(all_q_pred, dtype=np.float32)
    ret_arr = np.array(all_realized, dtype=np.float32)
    corr, _ = stats.pearsonr(q_arr, ret_arr)
    mae = float(np.mean(np.abs(q_arr - ret_arr)))

    return {
        "pearson_r": float(corr),
        "mae": mae,
        "q_pred_mean": float(np.mean(q_arr)),
        "realized_return_mean": float(np.mean(ret_arr)),
    }


def ensure_drive_mounted(drive_dir: Union[str, Path]) -> bool:
    drive_p = Path(drive_dir)
    if drive_p.exists():
        return True
    try:
        from google.colab import drive  # type: ignore
        print("[Drive] Google Colab detected. Mounting Google Drive...")
        drive.mount("/content/drive")
        return drive_p.exists()
    except Exception as e:
        print(f"[Drive] Note: Could not auto-mount Drive ({e}).")
        return False


def resolve_curiosity_dir(drive_dir: Path) -> Path:
    candidates = [
        drive_dir / "data" / "curiosity_shards",
        drive_dir / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*.npz"))) > 0:
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Step 1: Stone Progression Fine-Tune (REVIEW-ONLY)")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--execute-training", action="store_true", help="Explicit gate to execute training")
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)

    print("=" * 80)
    print("PHASE 3 STEP 1 — ARCHITECTURAL CODE REVIEW & PRE-FLIGHT AUDIT")
    print(f"  Starting Base Checkpoint : checkpoints/jepa_crafter_v3_t2_final.pt")
    print(f"  Target Buffer Capacity   : 1,000,000 transitions (Anti-Eviction Assert Active)")
    print(f"  Event-Centered Windows   : Active (Balanced Sampling over all 22 achievements)")
    print(f"  Extended Value Targets   : n-step = 20 (Extended Horizon)")
    print(f"  Time-To-Event Head       : Initialized (22 heads by index, zero object code)")
    print("=" * 80)

    if not args.execute_training:
        print("\n[SAFETY GATE ACTIVE] --execute-training flag NOT provided.")
        print("Per directive: Code is pushed for review. Training will NOT start until")
        print("the code is fully audited and the Part A stone-enriched result is in.")
        return

    print("\n[EXECUTION AUTHORIZED] --execute-training flag detected.")
    print("Beginning Step 1 Data Loading & Training Pipeline...")

    # 1. Mount Drive & Resolve Shard Paths
    ensure_drive_mounted(drive_p)
    base_shards = find_drive_base_shards(args.drive_dir)
    curiosity_dir = str(resolve_curiosity_dir(drive_p))

    # 2. Build 1,000,000-capacity buffer with anti-eviction assert
    buffer, per_source_counts = build_step1_buffer(
        structured_random_final=base_shards["structured_random_final"],
        teacher_hybrid_final=base_shards["teacher_hybrid_final"],
        planner_v2_final=base_shards["planner_v2_final"],
        curiosity_dir=curiosity_dir,
        capacity_transitions=1_000_000,
    )

    # 3. Build Event-Centered Sampler
    sampler = EventCenteredSampler(buffer=buffer, seq_len=32)

    # 4. Load Starting Checkpoint (T2 Final)
    base_ckpt = drive_p / "checkpoints" / "jepa_crafter_v3_t2_final.pt"
    local_ckpt = _ensure_local_checkpoint(str(base_ckpt), drive_p)
    model = JEPAv3.from_checkpoint(str(local_ckpt), device=args.device)
    print(f"[Model] Successfully initialized JEPAv3 from {local_ckpt}")

    # 5. Execute Phase A: Frozen-encoder value & TTE warmup (5,000 steps)
    phase_a_results = train_step1_value_phase(
        model=model,
        sampler=sampler,
        total_steps=5000,
        batch_size=16,
        seq_len=32,
        lr=1e-4,
        n_step=20,
        gamma=0.99,
        device=args.device,
    )

    # 6. Gate Check: Self-Slice Q-Calibration
    calib_phase_a = evaluate_self_slice_calibration(model, sampler, device=args.device)
    r_val = calib_phase_a["pearson_r"]
    print("\n" + "=" * 80)
    print("STEP 1 PHASE A GATE AUDIT:")
    print(f"  Self-Slice Pearson r : {r_val:.4f} (Target: > 0.30)")
    print(f"  Self-Slice MAE       : {calib_phase_a['mae']:.4f}")
    print(f"  Phase A Gate Status  : {'✅ PASSED' if r_val > 0.30 else '⚠️ WARNING: r below 0.30 threshold'}")
    print("=" * 80)

    # Save Phase A checkpoint
    ckpt_phase_a_local = REPO_ROOT / "checkpoints" / "jepa_crafter_v3_step1_phase_a.pt"
    ckpt_phase_a_local.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(ckpt_phase_a_local))
    if drive_p.exists():
        drive_ckpt_a = drive_p / "checkpoints" / "jepa_crafter_v3_step1_phase_a.pt"
        drive_ckpt_a.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ckpt_phase_a_local), str(drive_ckpt_a))
        print(f"[Drive] Mirrored Phase A checkpoint to: {drive_ckpt_a}")

    # 7. Execute Phase B: Full Joint Fine-Tuning (10,000 steps)
    phase_b_results = train_step1_joint_phase(
        model=model,
        sampler=sampler,
        total_steps=10000,
        batch_size=16,
        seq_len=32,
        lr=1e-4,
        n_step=20,
        gamma=0.99,
        device=args.device,
    )

    # 8. Post-Phase B Gate Audit
    calib_phase_b = evaluate_self_slice_calibration(model, sampler, device=args.device)
    print("\n" + "=" * 80)
    print("STEP 1 FINAL GATE AUDIT:")
    print(f"  Final Pearson r : {calib_phase_b['pearson_r']:.4f}")
    print(f"  Final MAE       : {calib_phase_b['mae']:.4f}")
    print("=" * 80)

    # Save Final Step 1 Checkpoint
    ckpt_step1_local = REPO_ROOT / "checkpoints" / "jepa_crafter_v3_step1_stone.pt"
    model.save_checkpoint(str(ckpt_step1_local))
    print(f"[Model] Saved Step 1 final checkpoint to: {ckpt_step1_local}")
    if drive_p.exists():
        drive_ckpt_final = drive_p / "checkpoints" / "jepa_crafter_v3_step1_stone.pt"
        shutil.copy2(str(ckpt_step1_local), str(drive_ckpt_final))
        print(f"[Drive] Mirrored Step 1 final checkpoint to: {drive_ckpt_final}")

    # Save Results JSON
    results_payload = {
        "starting_checkpoint": str(local_ckpt),
        "buffer_composition": per_source_counts,
        "phase_a": {
            "steps": 5000,
            "final_loss": phase_a_results["final_loss"],
            "effective_stone_rate": phase_a_results["effective_stone_rate"],
            "calibration": calib_phase_a,
        },
        "phase_b": {
            "steps": 10000,
            "final_loss": phase_b_results["final_loss"],
            "effective_stone_rate": phase_b_results["effective_stone_rate"],
            "calibration": calib_phase_b,
        },
    }
    results_path = REPO_ROOT / "results" / "step1_stone_finetune_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[Results] Saved Step 1 results to: {results_path}")
    if drive_p.exists():
        drive_res = drive_p / "results" / "step1_stone_finetune_results.json"
        drive_res.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(results_path), str(drive_res))
        print(f"[Drive] Mirrored results to: {drive_res}")


if __name__ == "__main__":
    main()
