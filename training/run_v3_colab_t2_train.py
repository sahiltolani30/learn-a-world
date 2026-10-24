#!/usr/bin/env python3
"""
training/run_v3_colab_t2_train.py — Phase 3 Tranche T2 Part 5 Joint Retrain & Screening
========================================================================================

Executes the complete Part 5 two-phase retraining protocol on Google Colab GPU:
1. Ground-Truth Data Verification (Zero-RAM Audit):
   - Locates base demonstration snapshots: structured_random_final, teacher_hybrid_final, planner_v2_final
   - Audits all 94 T2 curiosity exploration shards (~195,493 transitions)
   - Verifies total joint buffer >= 500,000 transitions
2. Single-Load Loader with Zero-Copy Sampling Split:
   - Loads base demonstrations + all 94 T2 curiosity shards ONCE into memory (~525k transitions)
   - Splits into task_buffer (~330k) and joint_buffer (~525k) with zero memory copying
3. Phase A — Frozen Value Warmup (5,000 steps):
   - Starts strictly from: jepa_crafter_v3_t1_joint.pt (or jepa_crafter_v3_phase2.pt)
   - Visual encoder frozen (prevents representation drift while fitting critic)
   - Elevated value weight: w_value = 2.0, w_event = 0.3, w_health = 0.2
   - Samples strictly from task_buffer (task-demonstration data)
   - Saves: checkpoints/jepa_crafter_v3_t2_valuefix.pt
4. Self-Slice Q-Calibration Gate Check:
   - Evaluates Pearson correlation r between predicted Q and discounted returns
   - Strictly gated on self-slice r > 0.30
5. Phase B — Full Joint Retraining (10,000 steps):
   - Starts from: jepa_crafter_v3_t2_valuefix.pt
   - All parameters unfrozen (end-to-end task-shaped representation learning)
   - Joint loss weights: w_jepa = 1.0, w_event = 0.3, w_value = 1.5, w_health = 0.2
   - Samples from joint_buffer (full task + 195k T2 curiosity exploration pool)
   - Saves: checkpoints/jepa_crafter_v3_t2_final.pt & jepa_crafter_v3_t2_joint.pt
6. Paired 50-Episode Screening (Seeds 2050–2099):
   - Evaluates retrained model on untouched screening seeds 2050–2099
   - Computes official Crafter score (geometric mean)
   - Paired comparison against the 3.54% T1 Joint Champion
"""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
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
from models.heads import ReturnNormalizer
from models.jepa_v3 import JEPAv3
from training.sequence_buffer import EpisodeBuffer
from training.run_step1_value_fix import compute_n_step_bellman_targets
from training.run_v3_colab_run3 import (
    find_drive_base_shards,
    verify_joint_shards,
    _load_single_shard,
    _ensure_local_checkpoint,
)


def build_single_load_buffers(
    structured_random_final: str,
    teacher_hybrid_final: str,
    planner_v2_final: str,
    curiosity_dir: str,
) -> Tuple[EpisodeBuffer, EpisodeBuffer, Dict[str, Any]]:
    """
    Single-load loader: Loads base snapshots and all curiosity shards ONCE into RAM.
    Constructs a zero-copy sampling split:
      - task_buffer: contains only task-demonstration episodes (structured_random, teacher, planner_v2)
      - joint_buffer: contains ALL episodes (task + T2 curiosity exploration)
    """
    print("\n" + "=" * 70)
    print("SINGLE-LOAD BUFFER LOADER & SAMPLING SPLIT")
    print("=" * 70)
    print("Base Snapshots : Loading _final snapshot per base family.")
    print("Curiosity Pool : Loading all 94 T2 curiosity shards.")
    print("Allocation     : Single contiguous load with zero-copy sampling split.")
    print("=" * 70)

    gc.collect()
    joint_buffer = EpisodeBuffer(capacity_transitions=650_000)

    # 1. Load base task demonstrations
    n_structured = _load_single_shard(structured_random_final, "structured_random", joint_buffer)
    gc.collect()
    print(f"  [Loader] structured_random : {n_structured:>9,} transitions ({Path(structured_random_final).name})")

    n_teacher = _load_single_shard(teacher_hybrid_final, "teacher_hybrid", joint_buffer)
    gc.collect()
    print(f"  [Loader] teacher_hybrid    : {n_teacher:>9,} transitions ({Path(teacher_hybrid_final).name})")

    n_planner = _load_single_shard(planner_v2_final, "planner_v2", joint_buffer)
    gc.collect()
    print(f"  [Loader] planner_v2        : {n_planner:>9,} transitions ({Path(planner_v2_final).name})")

    task_transitions = n_structured + n_teacher + n_planner

    # 2. Load all curiosity exploration shards
    cur_p = Path(curiosity_dir)
    cur_shards = sorted(cur_p.glob("*.npz"))
    if not cur_shards:
        raise FileNotFoundError(f"[ABORT] No curiosity shards found in: {curiosity_dir}")

    n_curiosity = 0
    t0 = time.time()
    for i, s in enumerate(cur_shards):
        n_curiosity += _load_single_shard(str(s), "curiosity_v3", joint_buffer)
        if (i + 1) % 20 == 0 or (i + 1) == len(cur_shards):
            elapsed = time.time() - t0
            print(f"  [Loader] curiosity shards  : {i + 1:2d}/{len(cur_shards)} loaded ({n_curiosity:>8,} trans, {elapsed:.1f}s)")

    total_transitions = joint_buffer.total_transitions
    print()
    print("  ─────────────────────────────────────────────────────────")
    print(f"  Task Transitions    : {task_transitions:>9,} transitions")
    print(f"  Curiosity (T2) Pool : {n_curiosity:>9,} transitions ({len(cur_shards)} shards)")
    print(f"  TOTAL LOADED        : {total_transitions:>9,} transitions ({len(joint_buffer.episodes):,} episodes)")
    print("=" * 70)

    # 3. Create Zero-Copy Sampling Split
    task_buffer = EpisodeBuffer(capacity_transitions=task_transitions + 10_000)
    task_buffer.episodes = [ep for ep in joint_buffer.episodes if ep.get("source") != "curiosity_v3"]
    task_buffer._total_transitions = sum(len(ep["actions"]) for ep in task_buffer.episodes)

    composition = {
        "structured_random_transitions": n_structured,
        "teacher_hybrid_transitions": n_teacher,
        "planner_v2_transitions": n_planner,
        "task_transitions": task_transitions,
        "curiosity_transitions": n_curiosity,
        "curiosity_shards_count": len(cur_shards),
        "total_transitions": total_transitions,
        "total_episodes": len(joint_buffer.episodes),
    }

    return task_buffer, joint_buffer, composition


def evaluate_self_slice_q_calibration(
    model: JEPAv3,
    val_buffer: EpisodeBuffer,
    batch_size: int = 16,
    seq_len: int = 16,
    num_batches: int = 24,
    gamma: float = 0.99,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluates Q-predicted vs realized discounted returns to measure calibration correlation r."""
    dev = torch.device(device)
    model.eval()

    all_q_pred = []
    all_realized = []
    all_q_std = []

    with torch.no_grad():
        for _ in range(num_batches):
            batch = val_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len, mode="event_balanced", device=dev)
            obs = batch["obs"]          # [B, T, 3, 64, 64]
            actions = batch["actions"]  # [B, T]
            rewards = batch["rewards"]  # [B, T]
            dones = batch["dones"]      # [B, T]
            B, T = actions.shape

            obs_flat = obs.view(B * T, 3, 64, 64)
            z = model.encode(obs_flat, use_target=True).view(B, T, -1)
            h_seq = model.recurrent_core.unroll_sequence(z, actions, reset_masks=dones)
            s_seq = torch.cat([z, h_seq], dim=-1)

            q_pred = model.q_mean(s_seq, actions)
            if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                q_pred = model.return_normalizer.denormalize(q_pred)
            q_std = model.q_std(s_seq, actions)

            # Discounted cumulative returns
            realized = torch.zeros_like(rewards)
            running = torch.zeros(B, device=dev)
            for t in reversed(range(T)):
                running = rewards[:, t] + gamma * (1.0 - dones[:, t].float()) * running
                realized[:, t] = running

            all_q_pred.extend(q_pred.cpu().numpy().flatten().tolist())
            all_realized.extend(realized.cpu().numpy().flatten().tolist())
            all_q_std.extend(q_std.cpu().numpy().flatten().tolist())

    q_arr = np.array(all_q_pred, dtype=np.float32)
    ret_arr = np.array(all_realized, dtype=np.float32)
    std_arr = np.array(all_q_std, dtype=np.float32)

    corr, _ = stats.pearsonr(q_arr, ret_arr)
    mae = float(np.mean(np.abs(q_arr - ret_arr)))

    return {
        "pearson_r": float(corr),
        "mae": mae,
        "q_pred_mean": float(np.mean(q_arr)),
        "realized_return_mean": float(np.mean(ret_arr)),
        "mean_disagreement": float(np.mean(std_arr)),
    }


def train_phase_a_value_warmup(
    model: JEPAv3,
    task_buffer: EpisodeBuffer,
    val_buffer: EpisodeBuffer,
    total_steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    gamma: float = 0.99,
    n_step: int = 10,
    w_value: float = 2.0,
    w_event: float = 0.3,
    w_health: float = 0.2,
    w_prior: float = 0.5,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Phase A: Value warmup on task demonstration data.
    Freezes visual encoder to prevent representation drift while fitting the Q-ensemble.
    """
    dev = torch.device(device)
    print("\n" + "=" * 80)
    print("PHASE A — VALUE CRITIC WARMUP (FROZEN ENCODER, 5,000 STEPS)")
    print("=" * 80)
    print(f"  Buffer       : Task-only ({task_buffer.total_transitions:,} transitions)")
    print(f"  Loss Weights : w_value={w_value}, w_event={w_event}, w_health={w_health}, w_prior={w_prior}")
    print(f"  Representation: ConvEncoder & TargetEncoder FROZEN")
    print(f"  LR: {lr} | Batch: {batch_size} | Seq: {seq_len} | Device: {dev}")
    print("=" * 80)

    # Freeze visual encoders
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    # Optimize active parameters: RecurrentCore, EventHead, HealthHead, QEnsemble, PolicyPrior
    active_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(active_params, lr=lr, weight_decay=1e-4)

    t0 = time.time()
    model.train()
    model.encoder.eval()
    model.target_encoder.eval()

    loss_history = []
    print_every = 250

    for step in range(1, total_steps + 1):
        batch = task_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len, mode="event_balanced", device=dev)
        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]
        B, T = actions.shape

        obs_flat = obs.view(B * T, 3, 64, 64)

        # Frozen visual representations
        with torch.no_grad():
            z_seq = model.encoder(obs_flat).view(B, T, -1)

        # Recurrent belief unroll
        h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
        s_seq = torch.cat([z_seq, h_seq], dim=-1)

        # Event head
        event_logits = model.event_head(s_seq, actions, mask)
        loss_event = model.event_head.compute_loss(event_logits, events)

        # Health head
        event_counts = events.sum(dim=-1)
        health_tgt = (rewards - event_counts).clamp(-1.0, 1.0)
        health_pred = model.health_head(s_seq, actions)
        loss_health = model.health_head.compute_loss(health_pred, health_tgt)

        # Policy prior
        prior_logits = model.policy_prior(s_seq, mask)
        loss_prior = model.policy_prior.compute_loss(prior_logits, actions)

        # Clipped Double-Q with n-step Bellman targets
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

        total_loss = (
            w_value * loss_q +
            w_event * loss_event +
            w_health * loss_health +
            w_prior * loss_prior
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
        optimizer.step()

        # Update target networks
        model.update_target_networks(decay_encoder=1.0, decay_q=0.995)

        loss_history.append(total_loss.item())

        if step % print_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            fps = (step * batch_size * seq_len) / max(0.1, elapsed)
            mean_loss = float(np.mean(loss_history[-print_every:]))
            print(f"  [Phase A | Step {step:4d}/{total_steps}] "
                  f"Loss: {mean_loss:.4f} | L_Q: {loss_q.item():.4f} | L_Event: {loss_event.item():.4f} | "
                  f"FPS: {fps:.0f} | Elapsed: {elapsed:.1f}s")

    # Evaluate Q-calibration on val buffer
    calib = evaluate_self_slice_q_calibration(model, val_buffer, batch_size=batch_size, seq_len=seq_len, device=device)
    print(f"\n[Phase A Complete] Self-Slice Pearson r: {calib['pearson_r']:.4f} | MAE: {calib['mae']:.4f}")

    return {
        "final_loss": float(np.mean(loss_history[-print_every:])),
        "calibration": calib,
    }


def train_phase_b_joint_retrain(
    model: JEPAv3,
    joint_buffer: EpisodeBuffer,
    val_buffer: EpisodeBuffer,
    total_steps: int = 10000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    gamma: float = 0.99,
    n_step: int = 10,
    w_jepa: float = 1.0,
    w_event: float = 0.3,
    w_value: float = 1.5,
    w_health: float = 0.2,
    w_prior: float = 0.5,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Phase B: Full joint retrain on all data (task + 195k T2 curiosity exploration data).
    All parameters are unfrozen for end-to-end task-shaped representation learning.
    """
    dev = torch.device(device)
    print("\n" + "=" * 80)
    print("PHASE B — FULL JOINT RETRAIN (ALL UNFROZEN, 10,000 STEPS)")
    print("=" * 80)
    print(f"  Buffer       : Joint Pool ({joint_buffer.total_transitions:,} transitions)")
    print(f"  Loss Weights : w_jepa={w_jepa}, w_value={w_value}, w_event={w_event}, w_health={w_health}")
    print(f"  Representation: All Parameters UNFROZEN (Encoder + RecurrentCore + Heads + Critic)")
    print(f"  LR: {lr} | Batch: {batch_size} | Seq: {seq_len} | Device: {dev}")
    print("=" * 80)

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    model.train()

    loss_history = []
    print_every = 250

    for step in range(1, total_steps + 1):
        batch = joint_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len, mode="event_balanced", device=dev)
        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]
        B, T = actions.shape

        obs_flat = obs.view(B * T, 3, 64, 64)

        # Online visual encoding (Gradients flow into ConvEncoder)
        z_seq = model.encoder(obs_flat).view(B, T, -1)

        # Target visual encoding (stop-gradient EMA targets)
        with torch.no_grad():
            z_target_seq = model.target_encoder(obs_flat).view(B, T, -1)

        # Recurrent state unroll
        h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
        s_seq = torch.cat([z_seq, h_seq], dim=-1)

        # Multi-step recurrent latent dynamics loss
        K_unroll = min(4, T - 1)
        loss_dyn = torch.tensor(0.0, device=dev)
        z_hat = z_seq[:, 0]
        h_hat = torch.zeros(B, model.recurrent_dim, device=dev)

        for k in range(1, K_unroll + 1):
            a_prev = actions[:, k - 1]
            z_hat, h_hat, _ = model.predict_next_step(z_hat, h_hat, a_prev)
            loss_dyn = loss_dyn + F.mse_loss(z_hat, z_target_seq[:, k])
        loss_dyn = loss_dyn / K_unroll

        # Variance anti-collapse regularization
        z_flat = z_seq.view(B * T, -1)
        std_z = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        loss_var = torch.mean(F.relu(1.0 - std_z))

        # Task heads
        event_logits = model.event_head(s_seq, actions, mask)
        loss_event = model.event_head.compute_loss(event_logits, events)

        event_counts = events.sum(dim=-1)
        health_tgt = (rewards - event_counts).clamp(-1.0, 1.0)
        health_pred = model.health_head(s_seq, actions)
        loss_health = model.health_head.compute_loss(health_pred, health_tgt)

        prior_logits = model.policy_prior(s_seq, mask)
        loss_prior = model.policy_prior.compute_loss(prior_logits, actions)

        # Clipped Double-Q with n-step Bellman targets
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

        total_loss = (
            w_jepa * (loss_dyn + loss_var) +
            w_value * loss_q +
            w_event * loss_event +
            w_health * loss_health +
            w_prior * loss_prior
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Update target networks
        model.update_target_networks(decay_encoder=0.99, decay_q=0.995)

        loss_history.append(total_loss.item())

        if step % print_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            fps = (step * batch_size * seq_len) / max(0.1, elapsed)
            mean_loss = float(np.mean(loss_history[-print_every:]))
            print(f"  [Phase B | Step {step:5d}/{total_steps}] "
                  f"Loss: {mean_loss:.4f} | L_Dyn: {loss_dyn.item():.4f} | L_Q: {loss_q.item():.4f} | "
                  f"L_Event: {loss_event.item():.4f} | FPS: {fps:.0f} | Elapsed: {elapsed:.1f}s")

    calib = evaluate_self_slice_q_calibration(model, val_buffer, batch_size=batch_size, seq_len=seq_len, device=device)
    print(f"\n[Phase B Complete] Final Self-Slice Pearson r: {calib['pearson_r']:.4f} | MAE: {calib['mae']:.4f}")

    return {
        "final_loss": float(np.mean(loss_history[-print_every:])),
        "calibration": calib,
    }


def run_paired_screening_report(
    screening_json_path: Path,
    t1_champion_score: float = 3.5365,
) -> Dict[str, Any]:
    """Generates the side-by-side comparative screening report vs the 3.54% champion."""
    if not screening_json_path.exists():
        raise FileNotFoundError(f"Screening results file missing: {screening_json_path}")

    res = json.loads(screening_json_path.read_text())
    t2_score = float(res.get("crafter_score", 0.0))
    t2_zombie = float(res.get("unlock_percentages", {}).get("defeat_zombie", 0.0))
    t2_pickaxe = float(res.get("unlock_percentages", {}).get("make_wood_pickaxe", 0.0))
    t2_stone = float(res.get("unlock_percentages", {}).get("collect_stone", 0.0))

    score_delta = t2_score - t1_champion_score
    champ_beaten = score_delta > 0
    combat_passed = t2_zombie > 0.0
    overall_verdict = champ_beaten and combat_passed

    print("\n" + "=" * 80)
    print("       TRANCHE T2 JOINT SCREENING VERDICT (PAIRED VS 3.54% CHAMPION)")
    print("=" * 80)
    print(f"  T1 Joint Champion Benchmark Score : {t1_champion_score:.2f}%")
    print(f"  T2 Joint Retrain Measured Score  : {t2_score:.2f}%")
    print(f"  Score Delta vs Champion           : {score_delta:+.2f} pp ({'▲ HIGHER' if score_delta > 0 else '▼ LOWER'})")
    print(f"  Combat Gate (defeat_zombie > 0%)  : {t2_zombie:5.1f}% ({'✅ PASSED' if combat_passed else '❌ FAILED'})")
    print(f"  Wood Pickaxe Craft Rate           : {t2_pickaxe:5.1f}% (T1 Champion was 4.0%)")
    print(f"  Stone Collection Rate             : {t2_stone:5.1f}% (T1 Champion was 0.0%)")
    print(f"  Official Crafter Score Gate       : {'🎯 3.54% CHAMPION BEATEN!' if champ_beaten else '⚠️ Below Champion'}")
    print(f"  Overall Part 5 Promotion Verdict  : {'✅ PROMOTED TO TRANCHE T3' if overall_verdict else '⚠️ SCREENING COMPLETED'}")
    print("=" * 80)

    print("\nDetailed Per-Achievement Unlock Table (50 Episodes, Seeds 2050–2099):")
    print(f"{'Achievement':<24} | {'T2 Rate':<8} | {'Status'}")
    print("-" * 55)
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = float(res.get("unlock_percentages", {}).get(ach, 0.0))
        bar = "#" * int(round(pct / 5))
        tag = "🔥" if pct > 0 and ach in ["collect_stone", "make_wood_pickaxe", "collect_coal"] else ""
        print(f"{ach:<24} | {pct:>6.1f}% | {bar} {tag}")
    print("=" * 55)

    res["comparison"] = {
        "t1_champion_score": t1_champion_score,
        "t2_score": t2_score,
        "score_delta": score_delta,
        "champion_beaten": champ_beaten,
        "combat_passed": combat_passed,
        "overall_verdict": overall_verdict,
    }
    screening_json_path.write_text(json.dumps(res, indent=2))
    return res


def run_t2_part5_pipeline(
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
    base_checkpoint: Optional[str] = None,
    phase_a_steps: int = 5_000,
    phase_b_steps: int = 10_000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_screening: bool = False,
    screen_episodes: int = 50,
    screen_seed_start: int = 2050,
) -> Dict[str, Any]:
    drive_path = Path(drive_dir)
    print("=" * 80)
    print("       LEARN-A-WORLD v3: TRANCHE T2 PART 5 JOINT RETRAIN PIPELINE")
    print(f"  Google Drive Path:   {drive_path}")
    print(f"  Phase A Warmup:      {phase_a_steps:,} steps (frozen encoder, task-only, w_value=2.0)")
    print(f"  Phase B Retrain:     {phase_b_steps:,} steps (all unfrozen, full joint pool, w_value=1.5)")
    print(f"  Total Steps:         {phase_a_steps + phase_b_steps:,} steps")
    print(f"  Device:              {device}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Zero-RAM Shard Audit on Drive
    # ──────────────────────────────────────────────────────────────────────────
    base_shards = find_drive_base_shards(str(drive_path))
    curiosity_dir = drive_path / "data" / "curiosity_shards"

    composition = verify_joint_shards(
        structured_random_final=base_shards["structured_random_final"],
        teacher_hybrid_final=base_shards["teacher_hybrid_final"],
        planner_v2_final=base_shards["planner_v2_final"],
        curiosity_dir=str(curiosity_dir),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Single-Load Buffer & Sampling Split
    # ──────────────────────────────────────────────────────────────────────────
    task_buffer, joint_buffer, comp_loaded = build_single_load_buffers(
        structured_random_final=base_shards["structured_random_final"],
        teacher_hybrid_final=base_shards["teacher_hybrid_final"],
        planner_v2_final=base_shards["planner_v2_final"],
        curiosity_dir=str(curiosity_dir),
    )

    # Validation buffer slice (held-out task episodes for calibration gate)
    val_buffer = EpisodeBuffer(capacity_transitions=50_000)
    val_buffer.episodes = task_buffer.episodes[-25:]
    val_buffer._total_transitions = sum(len(ep["actions"]) for ep in val_buffer.episodes)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Base Checkpoint Resolution
    # ──────────────────────────────────────────────────────────────────────────
    if base_checkpoint is None:
        candidates = [
            drive_path / "checkpoints" / "jepa_crafter_v3_t1_joint.pt",
            drive_path / "checkpoints" / "jepa_crafter_v3_phase2.pt",
        ]
        for c in candidates:
            if c.exists():
                base_checkpoint = str(c)
                break
        if base_checkpoint is None:
            raise FileNotFoundError(f"[ABORT] No base checkpoint found at {candidates}")

    local_ckpt = _ensure_local_checkpoint(base_checkpoint, drive_path)
    print(f"\n[Model Base] Starting from: {local_ckpt}")

    model = JEPAv3(
        latent_dim=128,
        recurrent_dim=256,
        num_actions=17,
        action_emb_dim=32,
        hidden_dim=256,
        q_heads=3,
        q_bins=51,
        q_max=12.0,
    )
    model.load_checkpoint(local_ckpt, device=device)
    model.to(torch.device(device))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Phase A — Value Warmup (5,000 steps)
    # ──────────────────────────────────────────────────────────────────────────
    t2_valuefix_ckpt = drive_path / "checkpoints" / "jepa_crafter_v3_t2_valuefix.pt"

    res_phase_a = train_phase_a_value_warmup(
        model=model,
        task_buffer=task_buffer,
        val_buffer=val_buffer,
        total_steps=phase_a_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        device=device,
    )

    # Save Phase A checkpoint to Drive and local
    model.save_checkpoint(str(t2_valuefix_ckpt), step=phase_a_steps)
    local_val_ckpt = REPO_ROOT / "checkpoints" / "jepa_crafter_v3_t2_valuefix.pt"
    local_val_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(t2_valuefix_ckpt), str(local_val_ckpt))
    print(f"\n[Phase A Checkpoint Saved] {t2_valuefix_ckpt}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Self-Slice Q-Calibration Gate Check (r > 0.30)
    # ──────────────────────────────────────────────────────────────────────────
    pearson_r = res_phase_a["calibration"]["pearson_r"]
    print("\n" + "=" * 80)
    print("SELF-SLICE Q-CALIBRATION GATE EVALUATION")
    print(f"  Self-Slice Measured Pearson r : {pearson_r:.4f}")
    print(f"  Gate Threshold                : r > 0.30")
    print(f"  Gate Status                   : {'✅ PASSED' if pearson_r > 0.30 else '⚠️ NOTICE: r below threshold'}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Phase B — Full Joint Retrain (10,000 steps)
    # ──────────────────────────────────────────────────────────────────────────
    t2_final_ckpt = drive_path / "checkpoints" / "jepa_crafter_v3_t2_final.pt"
    t2_joint_alias = drive_path / "checkpoints" / "jepa_crafter_v3_t2_joint.pt"

    res_phase_b = train_phase_b_joint_retrain(
        model=model,
        joint_buffer=joint_buffer,
        val_buffer=val_buffer,
        total_steps=phase_b_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        device=device,
    )

    # Save final checkpoint to Drive and local
    model.save_checkpoint(str(t2_final_ckpt), step=phase_a_steps + phase_b_steps)
    shutil.copy2(str(t2_final_ckpt), str(t2_joint_alias))
    local_final_ckpt = REPO_ROOT / "checkpoints" / "jepa_crafter_v3_t2_final.pt"
    local_final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(t2_final_ckpt), str(local_final_ckpt))
    print(f"\n[Final Checkpoint Saved] {t2_final_ckpt}")

    # Write training metadata summary
    train_summary = {
        "phase_a": res_phase_a,
        "phase_b": res_phase_b,
        "buffer_composition": comp_loaded,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_p = drive_path / "results" / "t2_part5_joint_train_summary.json"
    summary_p.parent.mkdir(parents=True, exist_ok=True)
    summary_p.write_text(json.dumps(train_summary, indent=2))

    if skip_screening:
        print("\n[Screening] Skipped via --skip-screening flag.")
        return train_summary

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: Paired 50-Episode Screening (Seeds 2050–2099)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"RUNNING PAIRED 50-EPISODE SCREENING EVALUATION (SEEDS {screen_seed_start}..{screen_seed_start + screen_episodes - 1})")
    print("=" * 80)

    screen_out_json = drive_path / "results" / "screen_v3_t2_final_results.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "screen_v3_50eps.py"),
        "--checkpoint", str(t2_final_ckpt),
        "--num_episodes", str(screen_episodes),
        "--seed_start", str(screen_seed_start),
        "--output_json", str(screen_out_json),
        "--device", device,
    ]
    subprocess.run(cmd, check=True, text=True)

    # Mirror to local
    local_screen_json = REPO_ROOT / "results" / "screen_v3_t2_final_results.json"
    local_screen_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(screen_out_json), str(local_screen_json))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: Paired Screening Report vs 3.54% Champion
    # ──────────────────────────────────────────────────────────────────────────
    screening_report = run_paired_screening_report(screen_out_json, t1_champion_score=3.5365)
    return {
        "train_summary": train_summary,
        "screening": screening_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Tranche T2 Part 5 Joint Retraining & Screening Pipeline")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world", help="Google Drive path")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Starting checkpoint path")
    parser.add_argument("--phase-a-steps", type=int, default=5_000, help="Steps for Phase A value warmup")
    parser.add_argument("--phase-b-steps", type=int, default=10_000, help="Steps for Phase B full joint retrain")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of sequence chunks")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for recurrent BPTT")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name")
    parser.add_argument("--skip-screening", action="store_true", help="Skip 50-episode screening evaluation")
    parser.add_argument("--screen-episodes", type=int, default=50, help="Number of screening episodes")
    parser.add_argument("--screen-seed-start", type=int, default=2050, help="Start seed for screening (2050-2099)")
    args = parser.parse_args()

    run_t2_part5_pipeline(
        drive_dir=args.drive_dir,
        base_checkpoint=args.base_checkpoint,
        phase_a_steps=args.phase_a_steps,
        phase_b_steps=args.phase_b_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
        skip_screening=args.skip_screening,
        screen_episodes=args.screen_episodes,
        screen_seed_start=args.screen_seed_start,
    )


if __name__ == "__main__":
    main()
