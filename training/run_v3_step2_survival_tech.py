#!/usr/bin/env python3
"""
training/run_v3_step2_survival_tech.py — Phase 3 Step 2: Armed Survival & Tech-Tree Fine-Tuning
=================================================================================================

Implements targeted Step 2 multi-objective backpropagation to solve the 175-step zombie death wall
and establish systematic pickaxe, weapon, and stone progression:

1. Starting Checkpoint:
   - checkpoints/jepa_crafter_v3_step1_stone.pt (verified Q-calibrated model).
2. Data Curriculum Integration:
   - Base demonstration buffer (structured_random, teacher_hybrid, planner_v2).
   - High-yield curriculum sequences (data/curriculum/shard_curriculum_weapon_stone.npz).
   - 50% curriculum/event-centered sampling ratio.
3. Loss Formulation:
   - Asymmetric Health Loss (10x penalty on negative vital deltas to instill fear of death).
   - PolicyPrior Supervision (cross-entropy on contiguous table -> pickaxe -> sword macros).
   - EventHead Focal BCE (multi-label unlocks including make_wood_sword and defeat_zombie).
   - Time-To-Event Supervision (potential field compass for survival and stone).
   - Clipped Double-Q Value Bellman regression with n=20 foresight.
4. Output:
   - Saves: checkpoints/jepa_crafter_v3_step2_armed_stone.pt.
   - Evaluates: 50-episode official Crafter screening on untouched seeds 2050–2099.
   - Outputs: results/step2_armed_stone_finetune_results.json.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.heads import ReturnNormalizer
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from training.run_step1_value_fix import compute_n_step_bellman_targets
from training.run_v3_colab_run3 import (
    find_drive_base_shards,
    _load_single_shard,
    _ensure_local_checkpoint,
)
from training.run_v3_colab_step1_stone import EventCenteredSampler
from training.sequence_buffer import EpisodeBuffer

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)


def resolve_checkpoint(ckpt_path: str, drive_dir: Optional[str] = None) -> Path:
    p = Path(ckpt_path)
    if p.exists():
        return p
    candidates = [
        REPO_ROOT / ckpt_path,
        REPO_ROOT / "checkpoints" / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / ckpt_path,
            dp / "checkpoints" / p.name,
            Path("/content") / "checkpoints" / p.name,
            Path("/content") / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint: {ckpt_path}")


def build_step2_buffer(
    curriculum_shard_path: str,
    teacher_hybrid_path: str,
    structured_random_path: str,
    capacity_transitions: int = 500_000,
) -> Tuple[EpisodeBuffer, Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STEP 2 BUFFER LOADER (CURRICULUM ENRICHMENT)")
    print("=" * 80)
    print(f"Curriculum Shard : {curriculum_shard_path}")
    print(f"Teacher Shard    : {teacher_hybrid_path}")
    print(f"Structured Shard : {structured_random_path}")
    print("=" * 80)

    gc.collect()
    buffer = EpisodeBuffer(capacity_transitions=capacity_transitions)
    counts = {}

    # 1. High-Yield Curriculum Shard (Highest priority)
    n_curric = _load_single_shard(curriculum_shard_path, "curriculum_weapon_stone", buffer)
    counts["curriculum_weapon_stone"] = n_curric
    print(f"  [Loader] Curriculum Data  : {n_curric:>8,d} transitions ({Path(curriculum_shard_path).name})")

    # 2. Base Teacher Hybrid
    n_teach = _load_single_shard(teacher_hybrid_path, "teacher_hybrid", buffer)
    counts["teacher_hybrid"] = n_teach
    print(f"  [Loader] Teacher Hybrid   : {n_teach:>8,d} transitions ({Path(teacher_hybrid_path).name})")

    # 3. Base Structured Random
    n_rand = _load_single_shard(structured_random_path, "structured_random", buffer)
    counts["structured_random"] = n_rand
    print(f"  [Loader] Structured Random: {n_rand:>8,d} transitions ({Path(structured_random_path).name})")

    print(f"\n[Buffer Total] {buffer.total_transitions:,} transitions across {buffer.total_episodes:,} episodes.")
    return buffer, counts


def train_step2_survival_tech(
    model: JEPAv3,
    sampler: EventCenteredSampler,
    total_steps: int = 2000,
    batch_size: int = 16,
    seq_len: int = 32,
    lr: float = 8e-5,
    n_step: int = 20,
    gamma: float = 0.99,
    device: str = "cuda",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"STEP 2 MULTI-OBJECTIVE FINE-TUNING ({total_steps:,} STEPS)")
    print("  Loss Components:")
    print("    - Asymmetric Health Loss (10x penalty on damage/death)")
    print("    - PolicyPrior Cross-Entropy (Macro Crafting & Combat Alignment)")
    print("    - EventHead Focal BCE (Multi-label Unlock Prediction)")
    print("    - Time-To-Event Loss (Potential Field Guidance)")
    print("    - Clipped Double-Q Value Loss (n-step=20 Horizon)")
    print("=" * 80)

    dev = torch.device(device)
    model.to(dev)

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
        masks = batch["mask_before"]
        tte_targets = batch["tte_targets"]  # [B, T, 22]
        B, T = actions.shape

        obs_flat = obs.view(B * T, 3, 64, 64)
        z_seq = model.encoder(obs_flat).view(B, T, -1)
        h_seq = model.recurrent_core.unroll_sequence(z_seq, actions, reset_masks=dones)
        s_seq = torch.cat([z_seq, h_seq], dim=-1)

        # 1. Asymmetric Health / Survival Loss
        # Target: non-achievement reward (health delta), clipped [-1, 1]
        ach_counts = events.sum(dim=-1).float()
        health_targets = (rewards - ach_counts).clamp(-1.0, 1.0)
        pred_health = model.health_head(s_seq, actions)
        health_err = F.smooth_l1_loss(pred_health, health_targets, reduction="none")
        # 10x penalty when taking damage / dying
        asym_weights = torch.where(health_targets < 0.0, 10.0, 1.0)
        loss_health = (asym_weights * health_err).mean()

        # 2. Policy Prior Cross-Entropy (Crafting & Combat Supervised Alignment)
        prior_logits = model.policy_prior(s_seq, masks)
        loss_prior = F.cross_entropy(prior_logits.view(-1, 17), actions.view(-1))

        # 3. Event Loss (Focal BCE)
        event_logits = model.event_head(s_seq, actions, masks)
        loss_event = model.event_head.compute_loss(event_logits, events)

        # 4. Time-To-Event Loss
        pred_tte = model.time_to_event_head(s_seq)
        loss_tte = model.time_to_event_head.compute_loss(pred_tte, tte_targets)

        # 5. Critic Loss with n-step=20 Bellman targets
        with torch.no_grad():
            bootstrap_acts = model.sample_prior_action(s_seq, masks, temperature=1.0)
            target_q_min = model.q_min(s_seq, bootstrap_acts, use_target=True)
            if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                target_q_min = model.return_normalizer.denormalize(target_q_min)
            bellman_targets = compute_n_step_bellman_targets(
                rewards=rewards,
                dones=dones,
                target_q_vals=target_q_min,
                gamma=gamma,
                n_step=n_step,
            )
            if hasattr(model, "return_normalizer"):
                model.return_normalizer.update(bellman_targets)
                q_fit_targets = model.return_normalizer.normalize(bellman_targets)
            else:
                q_fit_targets = bellman_targets

        q_logits = model.q_ensemble(s_seq, actions)
        loss_q = model.q_ensemble.compute_loss(q_logits, q_fit_targets)

        # 6. Latent Dynamics Loss
        with torch.no_grad():
            z_target = model.target_encoder(obs_flat).view(B, T, -1)
        z_pred = model.predictor(z_seq[:, :-1], h_seq[:, :-1], actions[:, :-1])
        loss_dyn = F.mse_loss(z_pred, z_target[:, 1:])

        total_loss = (
            1.0 * loss_q +
            0.6 * loss_tte +
            0.4 * loss_prior +
            0.4 * loss_event +
            0.4 * loss_health +
            0.2 * loss_dyn
        )

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.update_target_networks(decay_encoder=0.99, decay_q=0.995)
        losses.append(float(total_loss.item()))

        if step % 250 == 0 or step == total_steps:
            elapsed = time.time() - t0
            print(
                f"Step {step:5d}/{total_steps} ({elapsed:.1f}s) | "
                f"Loss: {total_loss.item():.4f} | "
                f"Q: {loss_q.item():.3f} | "
                f"TTE: {loss_tte.item():.3f} | "
                f"Prior: {loss_prior.item():.3f} | "
                f"Health: {loss_health.item():.3f} | "
                f"Ev: {loss_event.item():.3f}"
            )

    return {"final_loss": float(np.mean(losses[-100:])), "training_seconds": round(time.time() - t0, 1)}


def run_official_screening(
    model: JEPAv3,
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    device: str = "cuda",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"OFFICIAL 50-EPISODE SCREENING (SEEDS {seed_start}..{seed_start + num_episodes - 1})")
    print("=" * 80)

    import gymnasium as gym
    dev = torch.device(device)
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=15,
        num_candidates=128,
        cem_iterations=3,
        uncertainty_weight=0.1,
        novelty_weight=0.005,
        prior_mix=0.65,
        denormalize_q=True,
        tte_weight=0.5,
        recurring_vitals=True,
        vital_reward_weight=0.3,
        action_persistence=0.3,
        hierarchical_tte=True,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256)
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []
    t_start = time.time()

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()
        ep_reward = 0.0
        steps = 0

        for step in range(max_steps):
            steps += 1
            action, _ = planner.plan(obs, belief)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

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
            if done:
                break

        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

        if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == num_episodes:
            elapsed = time.time() - t_start
            print(f"  [Screening] {ep_idx + 1:2d}/{num_episodes} eps | Mean Steps: {np.mean(survival_lengths):.1f} | Elapsed: {elapsed:.1f}s")

    env.close()
    score_stats = compute_crafter_score(achievement_records)
    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print("                SCREENING SCORECARD")
    print("=" * 80)
    print(f"Official Crafter Score   : {score_stats['score']:.2f}% (Step 1 Baseline: 4.41%)")
    print(f"Mean Survival Length     : {np.mean(survival_lengths):.1f} steps (Old Baseline: 178.3)")
    print(f"Mean Episode Reward      : {np.mean(episode_rewards):.2f}")
    print(f"Make Wood Pickaxe Rate   : {score_stats['unlock_percentages'].get('make_wood_pickaxe', 0.0):.1f}%")
    print(f"Make Wood Sword Rate     : {score_stats['unlock_percentages'].get('make_wood_sword', 0.0):.1f}%")
    print(f"Collect Stone Rate       : {score_stats['unlock_percentages'].get('collect_stone', 0.0):.1f}%")
    print(f"Defeat Zombie Rate       : {score_stats['unlock_percentages'].get('defeat_zombie', 0.0):.1f}%")
    print(f"Total Screening Time     : {total_time:.1f}s")
    print("=" * 80)

    return {
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "crafter_score": score_stats["score"],
        "mean_survival_length": float(np.mean(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "unlock_percentages": score_stats["unlock_percentages"],
        "total_seconds": round(total_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 2: Armed Survival & Tech-Tree Fine-Tuning.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_step1_stone.pt")
    parser.add_argument("--curriculum_shard", type=str, default="data/curriculum/shard_curriculum_weapon_stone.npz")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--screen_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--output_ckpt", type=str, default="checkpoints/jepa_crafter_v3_step2_armed_stone.pt")
    parser.add_argument("--output_json", type=str, default="results/step2_armed_stone_finetune_results.json")
    args = parser.parse_args()

    ckpt_file = resolve_checkpoint(args.checkpoint, args.drive_dir)
    print("=" * 80)
    print("      STEP 2: ARMED SURVIVAL & TECH-TREE FINE-TUNING PIPELINE")
    print(f"      Base Checkpoint  : {ckpt_file}")
    print(f"      Device           : {args.device}")
    print(f"      Fine-Tune Steps  : {args.steps}")
    print("=" * 80)

    model = JEPAv3.from_checkpoint(str(ckpt_file), device=args.device)

    # 1. Locate Base Demonstration Shards
    drive_p = Path(args.drive_dir) if args.drive_dir else REPO_ROOT
    teacher_shards = [
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
    ]
    struct_shards = [
        drive_p / "data" / "sequences" / "shard_structured_random_final.npz",
        REPO_ROOT / "data" / "sequences" / "shard_structured_random_final.npz",
        drive_p / "data" / "sequences" / "shard_structured_random_0000.npz",
        REPO_ROOT / "data" / "sequences" / "shard_structured_random_0000.npz",
    ]
    curric_shards = [
        Path(args.curriculum_shard),
        REPO_ROOT / args.curriculum_shard,
        drive_p / args.curriculum_shard,
    ]

    teacher_path = next((s for s in teacher_shards if s.exists()), None)
    struct_path = next((s for s in struct_shards if s.exists()), None)
    curric_path = next((s for s in curric_shards if s.exists()), None)

    if not curric_path:
        # Collect curriculum dataset if not present
        print(f"[Warning] Curriculum shard not found at {args.curriculum_shard}. Generating now...")
        from scripts.collect_curriculum_weapon_stone import collect_curriculum_dataset
        curric_path = collect_curriculum_dataset(
            num_episodes=30,
            seed_start=1000,
            max_steps=500,
            out_dir=str(Path(args.curriculum_shard).parent),
        )

    buffer, counts = build_step2_buffer(
        curriculum_shard_path=str(curric_path),
        teacher_hybrid_path=str(teacher_path) if teacher_path else str(curric_path),
        structured_random_path=str(struct_path) if struct_path else str(curric_path),
    )

    sampler = EventCenteredSampler(buffer, seq_len=32)

    # 2. Multi-Objective Training
    train_res = train_step2_survival_tech(
        model=model,
        sampler=sampler,
        total_steps=args.steps,
        device=args.device,
    )

    # 3. Save Fine-Tuned Step 2 Model
    out_ckpt = Path(args.output_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(out_ckpt), step=args.steps)
    print(f"\n--> Successfully saved Step 2 checkpoint to: {out_ckpt}")

    if args.drive_dir:
        dp = Path(args.drive_dir)
        if dp.exists():
            drive_ckpt = dp / "checkpoints" / out_ckpt.name
            drive_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_ckpt), str(drive_ckpt))
            print(f"[Drive] Mirrored Step 2 checkpoint to: {drive_ckpt}")

    # 4. Official 50-Episode Screening
    screening_res = {}
    if args.screen_episodes > 0:
        screening_res = run_official_screening(
            model=model,
            num_episodes=args.screen_episodes,
            seed_start=2050,
            device=args.device,
        )

    summary = {
        "step": 2,
        "base_checkpoint": str(ckpt_file),
        "output_checkpoint": str(out_ckpt),
        "fine_tune_stats": train_res,
        "screening_results": screening_res,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"--> Saved Step 2 results to: {out_json}")

    if args.drive_dir:
        dp = Path(args.drive_dir)
        if dp.exists():
            drive_json = dp / "results" / out_json.name
            drive_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_json), str(drive_json))
            print(f"[Drive] Mirrored Step 2 results to: {drive_json}")

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print(f"--- BEGIN {out_json.name} ---")
    print(json.dumps(summary, indent=2))
    print(f"--- END {out_json.name} ---\n")


if __name__ == "__main__":
    main()
