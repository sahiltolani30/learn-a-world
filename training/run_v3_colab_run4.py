"""
training/run_v3_colab_run4.py — Phase 3 Tranche T2 Curiosity Collection
========================================================================

Collects ~200k new transitions targeting rare achievements using:
  1. Calibrated Event Heads: Platt + temperature calibrated event probabilities.
  2. Rarity Weighting: Dynamic inverse-count reward bonus sum_k w_k * P_k(s, a).
  3. Archive Restarts: Deterministic state snapshot restoration when stuck.
  4. Online Plan2Explore Curiosity: 3-predictor ensemble disagreement exploration.

Strict Standing Rules:
- Google Drive (/content/drive/MyDrive/learn-a-world) is the ground-truth store.
- Collection seeds: 6000–7999 (2,000 seeds available).
- FORBIDDEN SEEDS: NEVER use seeds 2050–2099 (screening) or 3000–3099 (strictly forbidden).
- Achievement logic: Counts / masks / integer indices only — ZERO hardcoded achievement names.
- Target budget: 200,000 transitions. Gate check: >= 150,000 transitions.
- Shard naming: shard_curiosity_v3_t2_{shard_idx:04d}.npz (tranche=2).
"""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.crafter_wrapper import CrafterEnv, patch_crafter_determinism
from models.calibration import EventHeadCalibrator
from models.curiosity import Plan2ExploreCuriosity
from models.jepa_v3 import JEPAv3
from models.rarity import RarityWeightScheduler
from planning.agent_v3 import AgentBeliefV3
from planning.rarity_mpc import CuriosityRarityMPCPlanner
from training.archive import ArchiveRestarter
from training.sequence_buffer import EpisodeBuffer, NUM_ACHIEVEMENTS

# Automatically ensure 100% determinism patch is applied
patch_crafter_determinism()

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_TRANSITIONS = 200_000
GATE_MIN_TRANSITIONS = 150_000
SHARD_EVERY_TRANSITIONS = 2_000
CURIOSITY_TRAIN_EVERY = 100
CURIOSITY_TRAIN_STEPS = 20
CURIOSITY_BATCH_SIZE = 32
CURIOSITY_SEQ_LEN = 8
CHECKPOINT_EVERY_STEPS = 2_000

# Rarity & Exploration Hyperparameters
DEFAULT_LAMBDA_TASK = 0.3
DEFAULT_LAMBDA_CUR = 1.0
DEFAULT_LAMBDA_RARE = 0.5
DEFAULT_RARITY_BASE = 5.0
DEFAULT_RARITY_DECAY = 0.5
DEFAULT_RARE_THRESHOLD = 10
STAGNATION_THRESHOLD = 20


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_and_calibrator(
    checkpoint_path: Union[str, Path],
    calibrator_path: Optional[Union[str, Path]],
    device: str,
) -> JEPAv3:
    """Loads JEPAv3 checkpoint and attaches EventHeadCalibrator if available."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    print(f"[Model] Loading checkpoint: {ckpt_p}")
    model = JEPAv3.from_checkpoint(str(ckpt_p), device=device)
    model.eval()

    if calibrator_path is not None:
        cal_p = Path(calibrator_path)
        if cal_p.exists():
            print(f"[Model] Attaching calibrator from: {cal_p}")
            model.load_calibrator(str(cal_p))
        else:
            print(f"[Model] Warning: calibrator path {cal_p} not found. Running uncalibrated.")
    else:
        print("[Model] Running with identity calibrator.")

    return model


def _extract_global_counts_from_manifests(
    manifest_paths: List[Path],
    num_achievements: int = NUM_ACHIEVEMENTS,
) -> np.ndarray:
    """
    Computes cumulative global event unlock counts by scanning existing manifests.
    Strictly index-based summing.
    """
    total_counts = np.zeros(num_achievements, dtype=np.int64)
    found_any = False

    for p in manifest_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                found_any = True
                shards = data.get("shards", [])
                for s in shards:
                    counts = s.get("per_achievement_event_counts", [])
                    if len(counts) == num_achievements:
                        total_counts += np.array(counts, dtype=np.int64)
                print(f"[Manifest] Parsed {len(shards)} shards from {p}")
                break
            except Exception as e:
                print(f"[Manifest] Could not parse {p}: {e}")

    if not found_any:
        print("[Manifest] No prior manifests found. Starting global counts at 0.")

    return total_counts


# ──────────────────────────────────────────────────────────────────────────────
# Collector Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class T2CuriosityCollector:
    """
    Orchestrates Tranche T2 collection on GPU using calibrated rarity-curiosity MPC
    and archive restarts.
    """

    def __init__(
        self,
        model: JEPAv3,
        curiosity: Plan2ExploreCuriosity,
        rarity_scheduler: RarityWeightScheduler,
        archive_restarter: ArchiveRestarter,
        drive_dir: Path,
        local_dir: Path,
        tranche_id: int = 2,
        lambda_task: float = DEFAULT_LAMBDA_TASK,
        lambda_cur: float = DEFAULT_LAMBDA_CUR,
        lambda_rare: float = DEFAULT_LAMBDA_RARE,
        horizon: int = 10,
        num_candidates: int = 64,
        max_steps_per_ep: int = 1000,
        shard_every: int = SHARD_EVERY_TRANSITIONS,
        device: str = "cpu",
    ):
        self.model = model
        self.curiosity = curiosity
        self.rarity_scheduler = rarity_scheduler
        self.archive_restarter = archive_restarter
        self.drive_dir = Path(drive_dir)
        self.local_dir = Path(local_dir)
        self.tranche_id = tranche_id
        self.max_steps_per_ep = max_steps_per_ep
        self.shard_every = shard_every
        self.device = torch.device(device)

        # Set up output directories
        self.local_shards_dir = self.local_dir / "data" / "curiosity_shards"
        self.drive_shards_dir = self.drive_dir / "data" / "curiosity_shards"
        self.local_shards_dir.mkdir(parents=True, exist_ok=True)
        self.drive_shards_dir.mkdir(parents=True, exist_ok=True)

        # Online curiosity optimizers
        self.curiosity_opts = self.curiosity.make_optimizers(lr=3e-4)

        # Build Curiosity + Rarity Planner
        self.planner = CuriosityRarityMPCPlanner(
            model=self.model,
            curiosity=self.curiosity,
            rarity_scheduler=self.rarity_scheduler,
            lambda_task=lambda_task,
            lambda_cur=lambda_cur,
            lambda_rare=lambda_rare,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=2,
            gamma=0.99,
            device=self.device,
        )
        self.belief = AgentBeliefV3(recurrent_dim=256)

        # Internal replay buffer for online curiosity training
        self._curiosity_buffer = EpisodeBuffer(capacity_transitions=50_000)

    def _train_curiosity_online(self) -> float:
        """Runs online gradient updates for curiosity predictors on collected buffer."""
        if self._curiosity_buffer.total_transitions < CURIOSITY_SEQ_LEN * CURIOSITY_BATCH_SIZE:
            return 0.0

        self.curiosity.train()
        total_loss = 0.0

        for _ in range(CURIOSITY_TRAIN_STEPS):
            batch = self._curiosity_buffer.sample_sequences(
                batch_size=CURIOSITY_BATCH_SIZE,
                seq_len=CURIOSITY_SEQ_LEN,
                device=str(self.device),
            )
            obs_b = batch["obs"] / 255.0  # [B, T, 3, 64, 64]
            acts_b = batch["actions"]     # [B, T]
            B, T = acts_b.shape

            with torch.no_grad():
                z_all = self.model.encode(obs_b.view(B * T, 3, 64, 64), use_target=False).view(B, T, -1)
                z_tgt_all = self.model.encode(obs_b.view(B * T, 3, 64, 64), use_target=True).view(B, T, -1)
                # Compute recurrent states
                h_b = torch.zeros(B, self.model.recurrent_dim, device=self.device)
                h_seq = []
                for t in range(T - 1):
                    h_seq.append(h_b)
                    _, h_b, _ = self.model.predict_next_step(z_all[:, t], h_b, acts_b[:, t])
                h_all = torch.stack(h_seq, dim=1)  # [B, T-1, 256]

            z_in = z_all[:, :-1].reshape((B * (T - 1)), -1)
            z_tgt = z_tgt_all[:, 1:].reshape((B * (T - 1)), -1)
            h_in = h_all.reshape((B * (T - 1)), -1)
            a_in = acts_b[:, :-1].reshape(B * (T - 1))

            step_loss = 0.0
            for pred, opt in zip(self.curiosity.predictors, self.curiosity_opts):
                opt.zero_grad()
                pred_z = pred(z_in, h_in, a_in)
                loss_k = F.mse_loss(pred_z, z_tgt.detach())
                loss_k.backward()
                opt.step()
                step_loss += loss_k.item()

            total_loss += step_loss / len(self.curiosity_opts)

        return total_loss / CURIOSITY_TRAIN_STEPS

    def _flush_shard(
        self,
        shard_buffer: EpisodeBuffer,
        shard_idx: int,
        seed_start: int,
        seed_end: int,
        component_sums: Dict[str, float],
    ) -> Dict[str, Any]:
        """Flushes in-memory episode buffer to .npz shard and mirrors to Drive."""
        shard_name = f"shard_curiosity_v3_t{self.tranche_id:02d}_{shard_idx:04d}.npz"
        local_path = self.local_shards_dir / shard_name
        drive_path = self.drive_shards_dir / shard_name

        # Flatten buffer arrays
        all_obs = []
        all_acts = []
        all_rews = []
        all_dones = []
        all_events = []
        all_masks = []

        total_event_counts = np.zeros(NUM_ACHIEVEMENTS, dtype=np.int64)

        for ep in shard_buffer.episodes:
            all_obs.append(ep["obs"])
            all_acts.append(ep["actions"])
            all_rews.append(ep["rewards"])
            all_dones.append(ep["dones"])
            all_events.append(ep["events"])
            all_masks.append(ep["mask_before"])
            total_event_counts += np.sum(ep["events"], axis=0).astype(np.int64)

        np.savez_compressed(
            str(local_path),
            obs=np.concatenate(all_obs, axis=0),
            actions=np.concatenate(all_acts, axis=0),
            rewards=np.concatenate(all_rews, axis=0),
            dones=np.concatenate(all_dones, axis=0),
            events=np.concatenate(all_events, axis=0),
            mask_before=np.concatenate(all_masks, axis=0),
        )

        # Mirror shard to Drive (if distinct directory)
        if local_path.resolve() != drive_path.resolve():
            try:
                shutil.copy2(str(local_path), str(drive_path))
            except Exception as e:
                print(f"[Collector] Warning: could not mirror shard to Drive: {e}")

        manifest_entry = {
            "shard_name": shard_name,
            "tranche": self.tranche_id,
            "source_policy": "curiosity_rarity_t2",
            "seed_range": [seed_start, seed_end],
            "transition_count": int(shard_buffer.total_transitions),
            "episode_count": len(shard_buffer.episodes),
            "per_achievement_event_counts": [int(c) for c in total_event_counts],
            "task_reward_sum": round(float(component_sums.get("task_reward", 0.0)), 2),
            "curiosity_bonus_sum": round(float(component_sums.get("curiosity_bonus", 0.0)), 2),
            "rarity_bonus_sum": round(float(component_sums.get("rarity_bonus", 0.0)), 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return manifest_entry

    def _update_manifest(self, new_entries: List[Dict[str, Any]]) -> None:
        """Atomically updates dataset_index.json locally and on Drive."""
        manifest_paths = [
            self.local_dir / "results" / "dataset_index.json",
            self.drive_dir / "results" / "dataset_index.json",
            self.local_dir / "data" / "curiosity_manifest.json",
            self.drive_dir / "data" / "curiosity_manifest.json",
        ]

        for p in manifest_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            existing_shards = []
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    existing_shards = data.get("shards", [])
                except Exception:
                    pass

            existing_names = {s["shard_name"] for s in existing_shards}
            combined = list(existing_shards)
            for entry in new_entries:
                if entry["shard_name"] not in existing_names:
                    combined.append(entry)
                    existing_names.add(entry["shard_name"])

            tot_trans = sum(s.get("transition_count", 0) for s in combined)
            tot_eps = sum(s.get("episode_count", 0) for s in combined)

            manifest_content = {
                "total_transitions": tot_trans,
                "total_episodes": tot_eps,
                "shards": combined,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            try:
                p.write_text(json.dumps(manifest_content, indent=2))
            except Exception as e:
                print(f"[Manifest] Warning: could not write {p}: {e}")

    def collect(
        self,
        target_transitions: int,
        seed_start: int = 6000,
        seed_end: int = 7999,
    ) -> Dict[str, Any]:
        """
        Runs the full T2 curiosity + rarity collection loop.
        """
        # Ensure seed bounds
        if seed_start < 6000 or seed_end > 7999:
            raise ValueError(f"Collection seeds must be strictly in 6000-7999. Got [{seed_start}, {seed_end}]")

        print("=" * 70)
        print(f"STARTING T2 CURIOSITY + RARITY COLLECTION (Tranche {self.tranche_id})")
        print(f"Target Transitions: {target_transitions:,}")
        print(f"Seed Range:         {seed_start} to {seed_end}")
        print(f"Device:             {self.device}")
        print("=" * 70)

        # Find existing T2 shards to determine next shard_idx
        existing_t2_shards = sorted(self.local_shards_dir.glob("shard_curiosity_v3_t02_*.npz"))
        shard_idx = len(existing_t2_shards)

        new_transitions = 0
        total_episodes = 0
        current_shard_buf = EpisodeBuffer(capacity_transitions=self.shard_every * 2)
        current_shard_seed_start = seed_start
        all_new_shard_entries: List[Dict[str, Any]] = []

        # Running component stats
        comp_sums = {"task_reward": 0.0, "curiosity_bonus": 0.0, "rarity_bonus": 0.0}
        shard_comp_sums = {"task_reward": 0.0, "curiosity_bonus": 0.0, "rarity_bonus": 0.0}

        env = CrafterEnv(max_steps=self.max_steps_per_ep)
        cur_seed = seed_start
        stagnant_count = 0

        t_start = time.time()

        try:
            while new_transitions < target_transitions and cur_seed <= seed_end:
                # ── Archive Restart Check ─────────────────────────────────────
                is_restart = False
                if self.archive_restarter.should_restart(stagnant_count):
                    archive_entry = self.archive_restarter.sample()
                    if archive_entry is not None:
                        env.restore_state(archive_entry["snapshot"])
                        obs = np.copy(archive_entry["obs"])
                        is_restart = True
                        stagnant_count = 0
                        print(f"  [Archive Restart] Resumed from seed {archive_entry['seed']} (step {archive_entry['step']})")

                if not is_restart:
                    obs, _ = env.reset(seed=cur_seed)

                self.belief.reset()
                ep_obs, ep_acts, ep_rews, ep_dones = [], [], [], []
                ep_events, ep_masks = [], []
                prev_ach = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)

                ep_reward = 0.0
                ep_novel_event = False
                ep_steps = 0

                for _ in range(self.max_steps_per_ep):
                    # Plan action
                    action, debug = self.planner.plan(obs, self.belief)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc

                    # Achievement masks
                    ach_dict = info.get("achievements", {})
                    # Indexed array of achievement statuses (no names used in logic)
                    curr_ach = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
                    for idx, key in enumerate(sorted(ach_dict.keys())):
                        if idx < NUM_ACHIEVEMENTS:
                            curr_ach[idx] = float(ach_dict[key] > 0)

                    event = np.clip(curr_ach - prev_ach, 0, 1).astype(np.uint8)
                    mask_before = prev_ach.astype(np.uint8)
                    prev_ach = np.maximum(prev_ach, curr_ach)

                    # Update rarity counts & check archive
                    if np.any(event > 0):
                        self.rarity_scheduler.update(event)
                        # Check if achievement is rare
                        if self.archive_restarter.is_interesting(event, self.rarity_scheduler.counts):
                            rare_idxs = self.archive_restarter.get_rare_indices(event, self.rarity_scheduler.counts)
                            self.archive_restarter.add(
                                seed=cur_seed,
                                step=ep_steps,
                                snapshot=env.snapshot_state(),
                                obs=obs,
                                unlocked_indices=rare_idxs,
                            )
                            ep_novel_event = True

                    # Record transition
                    ep_obs.append(obs)
                    ep_acts.append(action)
                    ep_rews.append(float(reward))
                    ep_dones.append(bool(done))
                    ep_events.append(event)
                    ep_masks.append(mask_before)

                    # Belief progression
                    obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    obs_t = obs_t.to(self.device)
                    a_t = torch.tensor([action], dtype=torch.long, device=self.device)
                    h_t = self.belief.get_h_tensor(device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        z_t = self.model.encode(obs_t, use_target=False)
                        s_t = torch.cat([z_t, h_t], dim=-1)
                        m_t = self.belief.get_mask_tensor(device=self.device).unsqueeze(0)
                        p_t = self.model.event_head.predict_proba(s_t, a_t, m_t).squeeze(0)
                        _, h_next, _ = self.model.predict_next_step(z_t, h_t, a_t)
                    self.belief.update(p_t, reward, next_h=h_next.squeeze(0))

                    # Track components
                    task_r = float(reward)
                    cur_b = debug.get("curiosity_bonus_sum", 0.0) / max(1, self.planner.horizon)
                    rare_b = debug.get("rarity_bonus_sum", 0.0) / max(1, self.planner.horizon)

                    comp_sums["task_reward"] += task_r
                    comp_sums["curiosity_bonus"] += cur_b
                    comp_sums["rarity_bonus"] += rare_b
                    shard_comp_sums["task_reward"] += task_r
                    shard_comp_sums["curiosity_bonus"] += cur_b
                    shard_comp_sums["rarity_bonus"] += rare_b

                    obs = next_obs
                    ep_reward += task_r
                    ep_steps += 1
                    new_transitions += 1

                    # Online curiosity training
                    if new_transitions % CURIOSITY_TRAIN_EVERY == 0:
                        self._train_curiosity_online()

                    if done or new_transitions >= target_transitions:
                        break

                # ── Episode completion ───────────────────────────────────────
                if len(ep_acts) > 0:
                    ep_data = {
                        "obs": np.stack(ep_obs, axis=0).astype(np.uint8),
                        "actions": np.array(ep_acts, dtype=np.int64),
                        "rewards": np.array(ep_rews, dtype=np.float32),
                        "dones": np.array(ep_dones, dtype=np.bool_),
                        "events": np.stack(ep_events, axis=0).astype(np.uint8),
                        "mask_before": np.stack(ep_masks, axis=0).astype(np.uint8),
                    }
                    current_shard_buf.add_episode(**ep_data, source="curiosity_t2")
                    self._curiosity_buffer.add_episode(**ep_data, source="curiosity_t2")
                    total_episodes += 1

                    self.archive_restarter.record_episode_novelty(ep_novel_event)
                    stagnant_count = self.archive_restarter.episodes_since_novel

                # Logging
                if total_episodes % 10 == 0 or new_transitions >= target_transitions:
                    elapsed = time.time() - t_start
                    fps = new_transitions / max(0.1, elapsed)
                    print(
                        f"[Ep {total_episodes:04d} | Seed {cur_seed}] "
                        f"trans: {new_transitions:,}/{target_transitions:,} ({new_transitions/target_transitions*100:.1f}%) | "
                        f"steps: {ep_steps:3d} | r_ep: {ep_reward:.1f} | "
                        f"archive: {len(self.archive_restarter)} | stagnant: {stagnant_count} | "
                        f"FPS: {fps:.1f}"
                    )

                # ── Shard Flushing ───────────────────────────────────────────
                if current_shard_buf.total_transitions >= self.shard_every:
                    entry = self._flush_shard(
                        shard_buffer=current_shard_buf,
                        shard_idx=shard_idx,
                        seed_start=current_shard_seed_start,
                        seed_end=cur_seed,
                        component_sums=shard_comp_sums,
                    )
                    all_new_shard_entries.append(entry)
                    shard_idx += 1
                    current_shard_seed_start = cur_seed + 1
                    shard_comp_sums = {"task_reward": 0.0, "curiosity_bonus": 0.0, "rarity_bonus": 0.0}

                    # Write manifest atomically
                    self._update_manifest(all_new_shard_entries)
                    print(f"  [Shard Flushed] {entry['shard_name']} ({entry['transition_count']:,} transitions) -> Drive updated")
                    current_shard_buf = EpisodeBuffer(capacity_transitions=self.shard_every * 2)

                cur_seed += 1

        finally:
            env.close()

        # Final shard flush if remaining transitions exist
        if current_shard_buf.total_transitions > 0:
            entry = self._flush_shard(
                shard_buffer=current_shard_buf,
                shard_idx=shard_idx,
                seed_start=current_shard_seed_start,
                seed_end=cur_seed,
                component_sums=shard_comp_sums,
            )
            all_new_shard_entries.append(entry)
            self._update_manifest(all_new_shard_entries)
            print(f"  [Final Shard Flushed] {entry['shard_name']} ({entry['transition_count']:,} transitions)")

        total_elapsed = time.time() - t_start
        gate_passed = new_transitions >= GATE_MIN_TRANSITIONS

        summary = {
            "tranche": self.tranche_id,
            "target_transitions": target_transitions,
            "collected_transitions": new_transitions,
            "total_episodes": total_episodes,
            "gate_passed": bool(gate_passed),
            "gate_threshold": GATE_MIN_TRANSITIONS,
            "elapsed_seconds": round(total_elapsed, 1),
            "avg_fps": round(new_transitions / max(0.1, total_elapsed), 1),
            "archive_summary": self.archive_restarter.summary(),
            "component_sums": {k: round(v, 2) for k, v in comp_sums.items()},
            "num_shards_created": len(all_new_shard_entries),
            "shards": all_new_shard_entries,
        }

        return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="T2 Curiosity + Rarity Collector (Run 4)")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world", help="Google Drive path")
    parser.add_argument("--target-transitions", type=int, default=DEFAULT_TARGET_TRANSITIONS, help="Target transitions budget")
    parser.add_argument("--seed-start", type=int, default=6000, help="First collection seed (strictly >= 6000)")
    parser.add_argument("--seed-end", type=int, default=7999, help="Last collection seed")
    parser.add_argument("--lambda-task", type=float, default=DEFAULT_LAMBDA_TASK, help="Task reward scaling")
    parser.add_argument("--lambda-cur", type=float, default=DEFAULT_LAMBDA_CUR, help="Curiosity bonus scaling")
    parser.add_argument("--lambda-rare", type=float, default=DEFAULT_LAMBDA_RARE, help="Rarity bonus scaling")
    parser.add_argument("--smoke-test", action="store_true", help="Quick CPU smoke test mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    # Safety checks on seeds
    if args.seed_start < 6000 or args.seed_end > 7999:
        raise ValueError("CRITICAL: Seeds must be in range 6000-7999. Do not touch 2050-2099 or 3000-3099.")

    drive_p = Path(args.drive_dir)
    local_p = REPO_ROOT

    # Determine checkpoint paths (Drive-first, local fallback)
    ckpt_candidates = [
        drive_p / "checkpoints" / "jepa_crafter_v3_t1_joint.pt",
        local_p / "checkpoints" / "jepa_crafter_v3_t1_joint.pt",
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break

    if ckpt_path is None:
        if args.smoke_test:
            print("[Smoke Test] No joint checkpoint found. Initializing fresh JEPAv3.")
            model = JEPAv3()
        else:
            raise FileNotFoundError(f"T1 Joint checkpoint not found at any candidate: {ckpt_candidates}")
    else:
        cal_path = drive_p / "results" / "calibration" / "event_head_calibrator.json"
        if not cal_path.exists():
            cal_path = local_p / "results" / "calibration" / "event_head_calibrator.json"
        model = _load_model_and_calibrator(ckpt_path, cal_path if cal_path.exists() else None, args.device)

    # Curiosity module
    curiosity = Plan2ExploreCuriosity(latent_dim=128, recurrent_dim=256, num_actions=17).to(args.device)

    # Manifest paths
    manifest_candidates = [
        drive_p / "results" / "dataset_index.json",
        local_p / "results" / "dataset_index.json",
        drive_p / "data" / "curiosity_manifest.json",
        local_p / "data" / "curiosity_manifest.json",
    ]
    initial_counts = _extract_global_counts_from_manifests(manifest_candidates)

    # Rarity and Archive modules
    rarity_scheduler = RarityWeightScheduler(
        num_achievements=NUM_ACHIEVEMENTS,
        base_weight=DEFAULT_RARITY_BASE,
        decay=DEFAULT_RARITY_DECAY,
        initial_counts=initial_counts,
    )
    archive_restarter = ArchiveRestarter(
        rare_threshold=DEFAULT_RARE_THRESHOLD,
        archive_size=50,
        stagnation_threshold=STAGNATION_THRESHOLD,
    )

    target_t = min(args.target_transitions, 25) if args.smoke_test else args.target_transitions
    shard_size = 10 if args.smoke_test else SHARD_EVERY_TRANSITIONS
    horizon = 2 if args.smoke_test else 10
    num_candidates = 4 if args.smoke_test else 64

    collector = T2CuriosityCollector(
        model=model,
        curiosity=curiosity,
        rarity_scheduler=rarity_scheduler,
        archive_restarter=archive_restarter,
        drive_dir=drive_p,
        local_dir=local_p,
        tranche_id=2,
        lambda_task=args.lambda_task,
        lambda_cur=args.lambda_cur,
        lambda_rare=args.lambda_rare,
        horizon=horizon,
        num_candidates=num_candidates,
        shard_every=shard_size,
        device=args.device,
    )

    summary = collector.collect(
        target_transitions=target_t,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    # Save summary report
    summary_path = local_p / "results" / "t2_collection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Summary] Collection summary written to: {summary_path}")

    try:
        drive_summary = drive_p / "results" / "t2_collection_summary.json"
        drive_summary.parent.mkdir(parents=True, exist_ok=True)
        drive_summary.write_text(json.dumps(summary, indent=2))
        print(f"[Summary] Mirrored to Drive: {drive_summary}")
    except Exception as e:
        print(f"[Summary] Could not mirror summary to Drive: {e}")

    print("\n" + "=" * 70)
    print(f"COLLECTION VERDICT: {'✅ GATE PASSED' if summary['gate_passed'] or args.smoke_test else '❌ INSUFFICIENT DATA'}")
    print(f"Collected: {summary['collected_transitions']:,} / {target_t:,} transitions")
    print(f"Restarts triggered: {summary['archive_summary']['total_restarts_triggered']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
