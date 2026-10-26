#!/usr/bin/env python3
"""
training/run_v3_self_play_finetune.py — Autonomous Self-Play DAgger Fine-Tuning
==============================================================================

Implements 100% Pure JEPA Autonomous Self-Play Fine-Tuning:
1. Self-Play Rollout Collection (Phase A):
   - Uses the champion RecurrentMPCPlanner (persistence=0.5, frontier_threshold=0.25)
   - Executes autonomous rollouts on training seeds (e.g. 5000..5049)
   - Packages full trajectories (obs, actions, rewards, events, masks) into an EpisodeBuffer
   - Zero hardcoding, zero external demonstrations: 100% agent-discovered experience
2. Frontier-Stratified Buffer Assembly (Phase B):
   - Merges self-play episodes with base task demonstrations
   - Samples batches via 'frontier_stratified' mode (25% frontier, 25% milestone, 50% exploration)
3. Targeted Value & Policy Co-Training (Phase C):
   - Trains for 5,000 steps (~10-15 mins on Colab T4 GPU)
   - Two-Hot Q-ensemble loss against 10-step Bellman targets (learns tool/stone value)
   - DAgger PolicyPrior loss against MPC search decisions (learns to craft pickaxe/sword directly)
   - EventHead BCE loss & HealthHead Huber loss
   - Visual encoders frozen to protect representations from drift
4. Paired Evaluation Screening (Phase D):
   - Evaluates retrained checkpoint on fixed held-out seeds 2050..2099
   - Verifies official Crafter score progression toward 4.5%+
"""

import argparse
import copy
import gc
import json
import math
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
try:
    import gymnasium as gym
except ImportError:
    import gym

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # noqa: F401
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.heads import ReturnNormalizer
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from training.sequence_buffer import ACHIEVEMENT_TO_IDX, EpisodeBuffer, NUM_ACHIEVEMENTS
from planning.semantic_memory import inventory_dict_to_vector
from training.run_step1_value_fix import compute_n_step_bellman_targets


def collect_self_play_episodes(
    model: JEPAv3,
    num_episodes: int = 50,
    seed_start: int = 5000,
    max_steps: int = 1000,
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    action_persistence: float = 0.5,
    frontier_threshold: float = 0.25,
    common_threshold: float = 0.50,
    vital_weight: float = 0.5,
    death_risk_factor: float = 0.0,
    harvest_bonus: float = 0.15,
    device: str = "cuda",
) -> Tuple[EpisodeBuffer, Dict[str, Any]]:
    """
    Collects autonomous self-play episodes using the champion RecurrentMPCPlanner.
    Saves complete trajectories with per-step achievement events into an EpisodeBuffer.
    """
    dev = torch.device(device)
    print("\n" + "=" * 80)
    print("PHASE A: AUTONOMOUS SELF-PLAY ROLLOUT COLLECTION")
    print("=" * 80)
    print(f"Episodes: {num_episodes} | Seeds: {seed_start}..{seed_start + num_episodes - 1}")
    print(f"Planner : Recurrent MPC (H={horizon}, C={num_candidates}, Persistence={action_persistence})")
    print(f"Device  : {dev}")
    print("=" * 80)

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=cem_iterations,
        elite_ratio=0.15,
        gamma=0.99,
        uncertainty_weight=0.5,
        novelty_weight=0.02,
        prior_mix=0.5,
        action_persistence=action_persistence,
        num_actions=17,
        denormalize_q=False,
        tte_weight=0.0,
        standardize_scores=False,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
        vital_weight=vital_weight,
        death_risk_factor=death_risk_factor,
        harvest_bonus=harvest_bonus,
        device=dev,
    )
    belief = AgentBeliefV3(
        recurrent_dim=256,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
    )

    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)
    buffer = EpisodeBuffer(capacity_transitions=num_episodes * max_steps + 5000)

    stats_summary = {
        "num_episodes": num_episodes,
        "total_transitions": 0,
        "events_unlocked": {},
        "rewards": [],
        "durations": [],
    }

    t_start = time.time()
    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()

        ep_obs: List[np.ndarray] = []
        ep_actions: List[int] = []
        ep_rewards: List[float] = []
        ep_dones: List[bool] = []
        ep_events: List[np.ndarray] = []
        ep_mask: List[np.ndarray] = []
        ep_state_deltas: List[np.ndarray] = []
        ep_state_valid: List[bool] = []

        ach_prev = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        state_prev = inventory_dict_to_vector(info.get("inventory", {}) if isinstance(info, dict) else None)
        ep_reward = 0.0
        steps = 0
        t_ep_start = time.time()

        for step in range(max_steps):
            steps += 1
            action, _ = planner.plan(obs, belief)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Record event transitions
            ach_dict = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
            state_curr = inventory_dict_to_vector(info.get("inventory", {}) if isinstance(info, dict) else None)
            ach_curr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
            for a_name, a_val in ach_dict.items():
                if a_val > 0 and a_name in ACHIEVEMENT_TO_IDX:
                    ach_curr[ACHIEVEMENT_TO_IDX[a_name]] = 1
            ev_step = np.clip(ach_curr - ach_prev, 0, 1)
            mask_step = ach_prev.copy()
            ach_prev = ach_curr.copy()

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_events.append(ev_step)
            ep_mask.append(mask_step)
            ep_state_deltas.append(state_curr - state_prev)
            ep_state_valid.append(bool(isinstance(info, dict) and "inventory" in info))

            # Update belief state
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(dev)
                z_t = model.encode(obs_t, use_target=True)
                act_t = torch.tensor([action], device=dev, dtype=torch.long)
                h_prev = belief.get_h_tensor(device=dev).unsqueeze(0)
                s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
                mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
                event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)

            belief.update(event_probs, reward, next_h=next_h.squeeze(0))
            obs = next_obs
            state_prev = state_curr
            if done:
                break

        ep_duration = time.time() - t_ep_start
        ach_unlocked = [k for k, v in ach_dict.items() if v > 0]
        for a in ach_unlocked:
            stats_summary["events_unlocked"][a] = stats_summary["events_unlocked"].get(a, 0) + 1

        stats_summary["rewards"].append(ep_reward)
        stats_summary["durations"].append(ep_duration)
        stats_summary["total_transitions"] += steps

        buffer.add_episode(
            obs=np.array(ep_obs, dtype=np.uint8),
            actions=np.array(ep_actions, dtype=np.int64),
            rewards=np.array(ep_rewards, dtype=np.float32),
            dones=np.array(ep_dones, dtype=np.bool_),
            events=np.array(ep_events, dtype=np.uint8),
            mask_before=np.array(ep_mask, dtype=np.uint8),
            state_delta=np.array(ep_state_deltas, dtype=np.float32),
            state_delta_valid=np.array(ep_state_valid, dtype=np.bool_),
            source="self_play_mpc",
        )

        print(
            f"  Self-Play Ep {ep_idx + 1:02d}/{num_episodes} (Seed {seed}) | "
            f"Steps: {steps:<4d} | Reward: {ep_reward:<5.1f} | "
            f"Achievements ({len(ach_unlocked)}): {', '.join(ach_unlocked) if ach_unlocked else 'None'} "
            f"({ep_duration:.1f}s)"
        )

    env.close()
    total_elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Self-Play Collection Complete: {buffer.total_transitions:,} transitions in {total_elapsed:.1f}s")
    print(f"Discovered Milestones: {stats_summary['events_unlocked']}")
    print("=" * 80)
    return buffer, stats_summary


def load_joint_buffer(
    self_play_buffer: EpisodeBuffer,
    drive_dir: Optional[str] = None,
    capacity: int = 650_000,
) -> EpisodeBuffer:
    """
    Merges self-play episodes with base task demonstrations from Drive/local.
    """
    print("\n" + "=" * 80)
    print("PHASE B: FRONTIER-STRATIFIED BUFFER ASSEMBLY")
    print("=" * 80)

    joint_buffer = EpisodeBuffer(capacity_transitions=capacity)

    # 1. Add self-play episodes
    for ep in self_play_buffer.episodes:
        joint_buffer.add_episode(
            obs=ep["obs"],
            actions=ep["actions"],
            rewards=ep["rewards"],
            dones=ep["dones"],
            events=ep["events"],
            mask_before=ep["mask_before"],
            state_delta=ep.get("state_delta"),
            state_delta_valid=ep.get("state_delta_valid"),
            source="self_play_mpc",
        )
    print(f"  [Loader] Self-Play Trajectories : {self_play_buffer.total_transitions:,} transitions ({len(self_play_buffer.episodes)} eps)")

    # 2. Add base task demonstration shards if available
    base_dirs = []
    if drive_dir:
        base_dirs.extend([
            Path(drive_dir) / "data" / "sequences",
            Path(drive_dir) / "data" / "combined_shards",
        ])
    base_dirs.extend([
        REPO_ROOT / "data" / "sequences",
    ])

    loaded_shards = 0
    total_base_trans = 0
    loaded_names = set()
    for b_dir in base_dirs:
        if not b_dir.exists():
            continue
        all_npz = sorted(b_dir.glob("*.npz"))
        final_shards = [p for p in all_npz if "final" in p.name]
        target_shards = final_shards if final_shards else all_npz
        for shard_file in target_shards:
            if shard_file.name in loaded_names:
                continue
            try:
                init_trans = joint_buffer.total_transitions
                joint_buffer.load_shard(shard_file)
                added = joint_buffer.total_transitions - init_trans
                if added > 0:
                    loaded_shards += 1
                    total_base_trans += added
                    loaded_names.add(shard_file.name)
                    print(f"  [Loader] Base Shard: {shard_file.name} (+{added:,} trans)")
            except Exception as e:
                print(f"  [Loader] Warning: could not load {shard_file.name}: {e}")

    print(f"  [Loader] Total Base Transitions : {total_base_trans:,} ({loaded_shards} shards)")
    print(f"  [Loader] Total Combined Buffer  : {joint_buffer.total_transitions:,} transitions ({len(joint_buffer.episodes)} eps)")
    print("=" * 80)
    return joint_buffer


def co_train_self_play(
    model: JEPAv3,
    buffer: EpisodeBuffer,
    total_steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    gamma: float = 0.99,
    n_step: int = 10,
    w_value: float = 2.0,
    w_prior: float = 0.5,
    w_event: float = 0.3,
    w_health: float = 0.2,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Executes targeted value and policy co-training on the frontier-stratified buffer.
    Visual encoders remain frozen to prevent representation drift.
    """
    dev = torch.device(device)
    print("\n" + "=" * 80)
    print("PHASE C: TARGETED VALUE & POLICY CO-TRAINING (5,000 STEPS)")
    print("=" * 80)
    print(f"  Buffer Capacity : {buffer.total_transitions:,} transitions")
    print(f"  Sampling Mode   : frontier_stratified (25% frontier, 25% milestone, 50% explore)")
    print(f"  Weights         : w_value={w_value}, w_prior={w_prior}, w_event={w_event}, w_health={w_health}")
    print(f"  Encoders        : ConvEncoder & TargetEncoder FROZEN")
    print(f"  Optimizer       : AdamW (lr={lr}, weight_decay=1e-4)")
    print("=" * 80)

    # Freeze visual encoders
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    # Optimize active prediction and value circuits
    trainable_params = [
        {"params": model.recurrent_core.parameters(), "lr": lr * 0.5},
        {"params": model.q_ensemble.parameters(), "lr": lr},
        {"params": model.policy_prior.parameters(), "lr": lr},
        {"params": model.event_head.parameters(), "lr": lr * 0.5},
        {"params": model.health_head.parameters(), "lr": lr * 0.5},
    ]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    model.train()
    model.encoder.eval()
    model.target_encoder.eval()

    t_start = time.time()
    loss_history: List[float] = []

    for step in range(1, total_steps + 1):
        # Sample using frontier-stratified balanced window index
        try:
            batch = buffer.sample_sequences(
                batch_size=batch_size,
                seq_len=seq_len,
                mode="frontier_stratified",
                device=dev,
            )
        except Exception:
            batch = buffer.sample_sequences(
                batch_size=batch_size,
                seq_len=seq_len,
                mode="uniform",
                device=dev,
            )

        obs = batch["obs"]          # [B, T, 3, 64, 64]
        actions = batch["actions"]  # [B, T]
        rewards = batch["rewards"]  # [B, T]
        dones = batch["dones"]      # [B, T]
        events = batch["events"]    # [B, T, 22]
        mask = batch["mask_before"] # [B, T, 22]

        B, T = actions.shape
        obs_flat = obs.view(B * T, 3, 64, 64)

        with torch.no_grad():
            z = model.encode(obs_flat, use_target=True).view(B, T, -1)

        # Unroll recurrent core through the sequence
        h_seq = model.recurrent_core.unroll_sequence(z, actions, reset_masks=dones)
        s_seq = torch.cat([z, h_seq], dim=-1)

        # 1. Two-Hot Q-Ensemble Loss against n-step Bellman targets
        with torch.no_grad():
            bootstrap_acts = model.sample_prior_action(s_seq, mask, temperature=1.0)
            target_q = model.q_min(s_seq, bootstrap_acts, use_target=True)
            if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                target_q = model.return_normalizer.denormalize(target_q)
            target_returns = compute_n_step_bellman_targets(rewards, dones, target_q, gamma=gamma, n_step=n_step)
            if hasattr(model, "return_normalizer"):
                model.return_normalizer.update(target_returns)
                norm_targets = model.return_normalizer.normalize(target_returns)
            else:
                norm_targets = target_returns

        q_logits = model.q_ensemble(s_seq, actions)
        l_q = model.q_ensemble.compute_loss(q_logits, norm_targets)

        # 2. DAgger PolicyPrior Loss: Learn to imitate the MPC search decisions
        prior_logits = model.policy_prior(s_seq, mask)
        l_prior = model.policy_prior.compute_loss(prior_logits, actions)

        # 3. EventHead Loss
        event_logits = model.event_head(s_seq, actions, mask)
        l_event = model.event_head.compute_loss(event_logits, events)

        # 4. HealthHead Loss
        health_preds = model.health_head(s_seq, actions)
        event_counts = events.sum(dim=-1)
        health_tgt = (rewards - event_counts).clamp(-1.0, 1.0)
        l_health = model.health_head.compute_loss(health_preds, health_tgt)

        total_loss = w_value * l_q + w_prior * l_prior + w_event * l_event + w_health * l_health

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update target networks
        model.update_target_networks(decay_encoder=0.99, decay_q=0.995)

        loss_history.append(float(total_loss.item()))

        if step % 500 == 0 or step == total_steps:
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 1e-4)
            mean_loss = float(np.mean(loss_history[-500:]))
            print(
                f"  Step {step:5d}/{total_steps} | "
                f"Loss: {mean_loss:6.4f} | "
                f"Q: {float(l_q.item()):6.4f} | "
                f"Prior: {float(l_prior.item()):6.4f} | "
                f"Event: {float(l_event.item()):6.4f} | "
                f"({steps_per_sec:5.1f} steps/s)"
            )

    print("=" * 80)
    print(f"Co-Training Completed in {time.time() - t_start:.1f}s")
    print("=" * 80)
    return {"final_loss": float(np.mean(loss_history[-100:]))}


def run_pipeline(
    checkpoint: str,
    drive_dir: Optional[str] = None,
    collect_episodes: int = 50,
    seed_start: int = 5000,
    skip_collection: bool = False,
    self_play_shard: Optional[str] = None,
    train_steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    output_checkpoint: str = "checkpoints/jepa_crafter_v3_t3_selfplay.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_after: bool = True,
) -> None:
    """Complete end-to-end self-play collection, training, and evaluation pipeline."""
    dev = torch.device(device)

    # 1. Load starting champion model
    from scripts.screen_v3_50eps import _resolve_checkpoint
    resolved_ckpt = _resolve_checkpoint(checkpoint, drive_dir=drive_dir)
    print(f"\n[Pipeline] Loading base champion checkpoint: {resolved_ckpt}")
    model = JEPAv3(latent_dim=128, recurrent_dim=256, num_actions=17, action_emb_dim=32)
    model.load_checkpoint(resolved_ckpt, device=dev)
    model.to(dev)

    # Attach calibrated event probabilities if available
    model.load_default_calibrator()

    # Phase A: Collect autonomous self-play data (or reuse existing shard if requested)
    self_play_buffer = EpisodeBuffer(capacity_transitions=collect_episodes * 1000 + 10000)
    existing_shard = None

    if skip_collection:
        candidate_shards = []
        if self_play_shard:
            candidate_shards.append(Path(self_play_shard))
        default_name = f"shard_self_play_{seed_start}_{seed_start + collect_episodes - 1}.npz"
        candidate_shards.append(Path("data/self_play_shards") / default_name)
        if drive_dir:
            dp = Path(drive_dir) / "data" / "self_play_shards"
            candidate_shards.append(dp / default_name)
            if dp.exists():
                candidate_shards.extend(sorted(dp.glob("*.npz")))
        local_p = Path("data/self_play_shards")
        if local_p.exists():
            candidate_shards.extend(sorted(local_p.glob("*.npz")))

        for cs in candidate_shards:
            if cs.exists():
                existing_shard = cs
                break

    if skip_collection and existing_shard is not None:
        print("\n" + "=" * 80)
        print("PHASE A: REUSING SAVED AUTONOMOUS SELF-PLAY SHARD")
        print("=" * 80)
        print(f"  Shard File : {existing_shard}")
        self_play_buffer.load_shard(existing_shard)
        print(f"  Loaded     : {self_play_buffer.total_transitions:,} transitions ({len(self_play_buffer.episodes)} eps)")
        print("=" * 80)
    else:
        self_play_buffer, stats = collect_self_play_episodes(
            model=model,
            num_episodes=collect_episodes,
            seed_start=seed_start,
            device=device,
        )

        # Save self-play shard to disk & Drive
        shard_out = Path("data/self_play_shards") / f"shard_self_play_{seed_start}_{seed_start + collect_episodes - 1}.npz"
        shard_p = self_play_buffer.save_shard(shard_out)
        print(f"[Pipeline] Saved self-play shard to: {shard_p}")
        if drive_dir and Path(drive_dir).exists():
            drive_shard = Path(drive_dir) / "data" / "self_play_shards" / shard_p.name
            drive_shard.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(shard_p), str(drive_shard))
            print(f"[Drive] Mirrored self-play shard to: {drive_shard}")

    # Phase B: Merge into joint buffer
    joint_buffer = load_joint_buffer(self_play_buffer, drive_dir=drive_dir)

    # Phase C: Targeted Value & Policy Co-Training
    co_train_self_play(
        model=model,
        buffer=joint_buffer,
        total_steps=train_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        device=device,
    )

    # Save fine-tuned checkpoint
    out_p = Path(output_checkpoint)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(out_p))
    print(f"\n[Pipeline] Successfully saved fine-tuned checkpoint to: {out_p}")
    if drive_dir and Path(drive_dir).exists():
        drive_ckpt = Path(drive_dir) / "checkpoints" / out_p.name
        drive_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(out_p), str(drive_ckpt))
        print(f"[Drive] Mirrored fine-tuned checkpoint to: {drive_ckpt}")

    # Phase D: Paired 50-Episode Screening on held-out seeds (2050..2099)
    if eval_after:
        print("\n" + "=" * 80)
        print("PHASE D: HELD-OUT SCREENING EVALUATION (SEEDS 2050-2099)")
        print("=" * 80)
        from scripts.screen_v3_50eps import run_screening
        screen_json = "results/screen_v3_t3_selfplay_50ep.json"
        run_screening(
            checkpoint_path=str(out_p),
            num_episodes=50,
            seed_start=2050,
            action_persistence=0.5,
            denormalize_q=False,
            output_json=screen_json,
            drive_dir=drive_dir,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Self-Play Fine-Tuning Pipeline for JEPA v3.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_t2_final.pt")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--collect_episodes", type=int, default=50, help="Number of self-play episodes to collect.")
    parser.add_argument("--seed_start", type=int, default=5000, help="Starting seed for training self-play.")
    parser.add_argument("--skip_collection", action="store_true", help="Skip Phase A rollout and load existing self-play shard.")
    parser.add_argument("--self_play_shard", type=str, default=None, help="Explicit path to existing self-play shard.")
    parser.add_argument("--train_steps", type=int, default=5000, help="Number of co-training gradient steps.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/jepa_crafter_v3_t3_selfplay.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_eval", dest="eval_after", action="store_false", help="Skip post-training screening.")
    args = parser.parse_args()

    run_pipeline(
        checkpoint=args.checkpoint,
        drive_dir=args.drive_dir,
        collect_episodes=args.collect_episodes,
        seed_start=args.seed_start,
        skip_collection=args.skip_collection,
        self_play_shard=args.self_play_shard,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        output_checkpoint=args.output_checkpoint,
        device=args.device,
        eval_after=args.eval_after,
    )
