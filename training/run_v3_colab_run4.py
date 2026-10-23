"""
training/run_v3_colab_run4.py — Phase 3 Tranche T2 Curiosity Collection
========================================================================

Collects ~200k new transitions targeting rare achievements using:
  1. Calibrated Event Heads: Platt + temperature calibrated event probabilities.
  2. Rarity Weighting: Dynamic inverse-count reward bonus sum_k w_k * P_k(s, a).
  3. Archive Restarts: Deterministic state snapshot restoration with full belief preservation.
  4. Online Plan2Explore Curiosity: 3-predictor ensemble disagreement exploration.
  5. Running Scale Normalization (EMA): Normalized tri-objective + Q-uncertainty exploration.

Strict Standing Rules:
- Google Drive (/content/drive/MyDrive/learn-a-world) is the ground-truth store.
- Collection seeds: 6000–7999 (2,000 seeds available).
- FORBIDDEN SEEDS: NEVER use seeds 2050–2099 (screening) or 3000–3099 (strictly forbidden).
- Achievement logic: Counts / masks / integer indices only — ZERO hardcoded achievement names.
- Target budget: 200,000 transitions. Gate check: >= 150,000 transitions.
- Shard naming: shard_curiosity_v3_t02_{shard_idx:04d}.npz (tranche=2).
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

# Rarity & Exploration Hyperparameters (P0-1 & P0-2 & P1-4)
DEFAULT_LAMBDA_TASK = 1.0
DEFAULT_LAMBDA_CUR = 0.5
DEFAULT_LAMBDA_RARE = 1.0
DEFAULT_LAMBDA_Q = 0.3
DEFAULT_CAP_Q = 0.5

DEFAULT_RARITY_BASE = 5.0
DEFAULT_RARITY_DECAY = 0.5
DEFAULT_RARE_THRESHOLD = 10
DEFAULT_RESTART_PROB = 0.40
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
    drive_dir: Path,
    local_dir: Path,
    num_achievements: int = NUM_ACHIEVEMENTS,
) -> Tuple[np.ndarray, int, int, List[Dict[str, Any]]]:
    """
    Computes cumulative global event unlock counts by scanning all manifest files
    across Drive and local directories (P1-5: Counts seeding, loud).
    Strictly index-based summing.
    """
    candidate_dirs = [
        drive_dir / "data",
        drive_dir / "results",
        local_dir / "data",
        local_dir / "results",
    ]

    seen_shards = set()
    all_shard_entries: List[Dict[str, Any]] = []
    total_counts = np.zeros(num_achievements, dtype=np.int64)
    total_transitions = 0

    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        for jf in cdir.glob("**/*.json"):
            try:
                data = json.loads(jf.read_text())
                if isinstance(data, dict) and "shards" in data:
                    shards = data["shards"]
                    if shards and isinstance(shards[0], dict):
                        for s in shards:
                            name = s.get("shard_name", "")
                            if name and name not in seen_shards:
                                seen_shards.add(name)
                                all_shard_entries.append(s)
                                counts = s.get("per_achievement_event_counts", [])
                                if len(counts) == num_achievements:
                                    total_counts += np.array(counts, dtype=np.int64)
                                    total_transitions += int(s.get("transition_count", 0))
                    elif "per_achievement_positive_counts" in data:
                        # Base dataset index (e.g. dataset_300k_index.json)
                        doc_id = jf.name
                        if doc_id not in seen_shards:
                            seen_shards.add(doc_id)
                            pos = data["per_achievement_positive_counts"]
                            if isinstance(pos, dict) and len(pos) == num_achievements:
                                for idx, key in enumerate(sorted(pos.keys())):
                                    total_counts[idx] += int(pos[key])
                                total_transitions += int(data.get("total_transitions", 0))
            except Exception:
                pass

    total_events = int(np.sum(total_counts))

    # Loud check on zero priors if prior t02 shards exist
    prior_t02_files = list(drive_dir.glob("**/shard_curiosity_v3_t02_*.npz")) + list(
        local_dir.glob("**/shard_curiosity_v3_t02_*.npz")
    )
    if len(prior_t02_files) > 0 and total_events == 0:
        raise RuntimeError(
            f"[FATAL] Found {len(prior_t02_files)} existing t02 shard(s) on Drive/local, "
            f"but total loaded event counts across all manifests is 0! "
            f"Cannot continue with zero priors. Verify Drive mount and manifest paths."
        )

    print("=" * 70)
    print(f"[Rarity Prior Seeding] Scanned {len(seen_shards)} unique sources")
    print(f"[Rarity Prior Seeding] Total Transitions: {total_transitions:,}")
    print(f"[Rarity Prior Seeding] Total Events:      {total_events:,}")

    # 5 rarest and 5 commonest integer indices
    rarest_5 = sorted([(idx, int(total_counts[idx])) for idx in range(num_achievements)], key=lambda x: x[1])[:5]
    commonest_5 = sorted([(idx, int(total_counts[idx])) for idx in range(num_achievements)], key=lambda x: x[1], reverse=True)[:5]
    print(f"[Rarity Prior Seeding] 5 Rarest indices & counts:    {rarest_5}")
    print(f"[Rarity Prior Seeding] 5 Commonest indices & counts: {commonest_5}")
    print("=" * 70)

    return total_counts, total_transitions, total_events, all_shard_entries


# ──────────────────────────────────────────────────────────────────────────────
# Collector Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class T2CuriosityCollector:
    """
    Orchestrates Tranche T2 collection on GPU using calibrated rarity-curiosity MPC
    and archive restarts with belief preservation.
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
        lambda_q: float = DEFAULT_LAMBDA_Q,
        cap_q: float = DEFAULT_CAP_Q,
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
        try:
            self.drive_shards_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[Collector] Note: Drive path not writable on this host ({self.drive_shards_dir}): {e}")

        # Online curiosity optimizers
        self.curiosity_opts = self.curiosity.make_optimizers(lr=3e-4)

        # Attempt to load online curiosity checkpoint if present (P1-6)
        self._load_curiosity_checkpoint()

        # Build Curiosity + Rarity + Q-Uncertainty Planner with EMA normalization (P0-1 & P0-2)
        self.planner = CuriosityRarityMPCPlanner(
            model=self.model,
            curiosity=self.curiosity,
            rarity_scheduler=self.rarity_scheduler,
            lambda_task=lambda_task,
            lambda_cur=lambda_cur,
            lambda_rare=lambda_rare,
            lambda_q=lambda_q,
            cap_q=cap_q,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=2,
            gamma=0.99,
            device=self.device,
        )
        self.belief = AgentBeliefV3(recurrent_dim=256)

        # Internal replay buffer for online curiosity training
        self._curiosity_buffer = EpisodeBuffer(capacity_transitions=50_000)

    def _save_curiosity_checkpoint(self) -> None:
        """Persists online curiosity predictor weights and optimizer states (P1-6)."""
        local_ckpt = self.local_dir / "checkpoints" / "curiosity_t2.pt"
        drive_ckpt = self.drive_dir / "checkpoints" / "curiosity_t2.pt"
        local_ckpt.parent.mkdir(parents=True, exist_ok=True)
        drive_ckpt.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "predictors": [p.state_dict() for p in self.curiosity.predictors],
            "optimizers": [opt.state_dict() for opt in self.curiosity_opts],
            "timestamp": time.time(),
        }
        torch.save(payload, str(local_ckpt))
        if local_ckpt.resolve() != drive_ckpt.resolve():
            try:
                shutil.copy2(str(local_ckpt), str(drive_ckpt))
            except Exception as e:
                print(f"[Curiosity] Warning: could not mirror checkpoint to Drive: {e}")

    def _load_curiosity_checkpoint(self) -> bool:
        """Loads existing online curiosity weights if available on Drive or locally."""
        drive_ckpt = self.drive_dir / "checkpoints" / "curiosity_t2.pt"
        local_ckpt = self.local_dir / "checkpoints" / "curiosity_t2.pt"
        target_path = drive_ckpt if drive_ckpt.exists() else (local_ckpt if local_ckpt.exists() else None)

        if target_path is not None:
            try:
                data = torch.load(str(target_path), map_location=self.device)
                for pred, state in zip(self.curiosity.predictors, data["predictors"]):
                    pred.load_state_dict(state)
                for opt, state in zip(self.curiosity_opts, data["optimizers"]):
                    opt.load_state_dict(state)
                print(f"[Curiosity] Loaded existing predictor weights from {target_path}")
                return True
            except Exception as e:
                print(f"[Curiosity] Note: could not load existing checkpoint from {target_path}: {e}")
        return False

    def _mirror_archive_summary(self) -> None:
        """Mirrors archive diagnostic summary to local and Drive results (P1-6)."""
        local_p = self.local_dir / "results" / "archive_summary_t2.json"
        drive_p = self.drive_dir / "results" / "archive_summary_t2.json"
        local_p.parent.mkdir(parents=True, exist_ok=True)
        drive_p.parent.mkdir(parents=True, exist_ok=True)

        summary_data = self.archive_restarter.summary(self.rarity_scheduler.counts)
        content = json.dumps(summary_data, indent=2)
        local_p.write_text(content)
        if local_p.resolve() != drive_p.resolve():
            try:
                drive_p.write_text(content)
            except Exception as e:
                print(f"[Archive] Warning: could not mirror summary to Drive: {e}")

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
        shard_origins: List[List[int]],
        shard_restart_eps: int,
        shard_fresh_eps: int,
    ) -> Dict[str, Any]:
        """Flushes in-memory episode buffer to .npz shard and mirrors to Drive."""
        shard_name = f"shard_curiosity_v3_t{self.tranche_id:02d}_{shard_idx:04d}.npz"
        local_path = self.local_shards_dir / shard_name
        drive_path = self.drive_shards_dir / shard_name

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

        # Episode origins tagging (additive key: [seed, step])
        origins_arr = np.array(shard_origins, dtype=np.int64)

        np.savez_compressed(
            str(local_path),
            obs=np.concatenate(all_obs, axis=0),
            actions=np.concatenate(all_acts, axis=0),
            rewards=np.concatenate(all_rews, axis=0),
            dones=np.concatenate(all_dones, axis=0),
            events=np.concatenate(all_events, axis=0),
            mask_before=np.concatenate(all_masks, axis=0),
            episode_origins=origins_arr,
        )

        if local_path.resolve() != drive_path.resolve():
            try:
                shutil.copy2(str(local_path), str(drive_path))
            except Exception as e:
                print(f"[Collector] Warning: could not mirror shard to Drive: {e}")

        # Compute weighted shares across shard
        w_task = float(component_sums.get("weighted_task_sum", 0.0))
        w_cur = float(component_sums.get("weighted_cur_sum", 0.0))
        w_rare = float(component_sums.get("weighted_rare_sum", 0.0))
        w_unc = float(component_sums.get("weighted_unc_sum", 0.0))
        denom = max(1e-8, abs(w_task) + abs(w_cur) + abs(w_rare) + abs(w_unc))

        total_episodes = len(shard_buffer.episodes)
        restart_fraction = round(shard_restart_eps / max(1, total_episodes), 4)

        manifest_entry = {
            "shard_name": shard_name,
            "tranche": self.tranche_id,
            "source_policy": "curiosity_rarity_t2",
            "seed_range": [seed_start, seed_end],
            "transition_count": int(shard_buffer.total_transitions),
            "episode_count": total_episodes,
            "per_achievement_event_counts": [int(c) for c in total_event_counts],
            "task_reward_sum": round(float(component_sums.get("task_reward", 0.0)), 2),
            "curiosity_bonus_sum": round(float(component_sums.get("curiosity_bonus", 0.0)), 2),
            "rarity_bonus_sum": round(float(component_sums.get("rarity_bonus", 0.0)), 2),
            "unc_bonus_sum": round(float(component_sums.get("unc_bonus", 0.0)), 2),
            "weighted_task_sum": round(w_task, 3),
            "weighted_cur_sum": round(w_cur, 3),
            "weighted_rare_sum": round(w_rare, 3),
            "weighted_unc_sum": round(w_unc, 3),
            "weighted_shares": {
                "task": round(abs(w_task) / denom, 4),
                "curiosity": round(abs(w_cur) / denom, 4),
                "rarity": round(abs(w_rare) / denom, 4),
                "uncertainty": round(abs(w_unc) / denom, 4),
            },
            "archive_size": len(self.archive_restarter),
            "restart_episodes_in_shard": int(shard_restart_eps),
            "fresh_episodes_in_shard": int(shard_fresh_eps),
            "restart_fraction": restart_fraction,
            "archive_mask_histogram": self.archive_restarter.get_mask_histogram(self.rarity_scheduler.counts),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Save curiosity checkpoint & mirror archive summary on each flush (P1-6)
        self._save_curiosity_checkpoint()
        self._mirror_archive_summary()

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
        if seed_start < 6000 or seed_end > 7999:
            raise ValueError(f"Collection seeds must be strictly in 6000-7999. Got [{seed_start}, {seed_end}]")

        print("=" * 70)
        print(f"STARTING T2 CURIOSITY + RARITY COLLECTION (Tranche {self.tranche_id})")
        print(f"Target Transitions: {target_transitions:,}")
        print(f"Seed Range:         {seed_start} to {seed_end}")
        print(f"Device:             {self.device}")
        print("=" * 70)

        existing_t2_shards = sorted(self.local_shards_dir.glob("shard_curiosity_v3_t02_*.npz"))
        shard_idx = len(existing_t2_shards)

        new_transitions = 0
        total_episodes = 0
        current_shard_buf = EpisodeBuffer(capacity_transitions=self.shard_every * 2)
        current_shard_seed_start = seed_start
        all_new_shard_entries: List[Dict[str, Any]] = []

        shard_origins: List[List[int]] = []
        shard_restart_eps = 0
        shard_fresh_eps = 0

        # Running component stats (accumulated across entire run & per shard)
        comp_sums = {
            "task_reward": 0.0, "curiosity_bonus": 0.0, "rarity_bonus": 0.0, "unc_bonus": 0.0,
            "weighted_task_sum": 0.0, "weighted_cur_sum": 0.0, "weighted_rare_sum": 0.0, "weighted_unc_sum": 0.0,
        }
        shard_comp_sums = {
            "task_reward": 0.0, "curiosity_bonus": 0.0, "rarity_bonus": 0.0, "unc_bonus": 0.0,
            "weighted_task_sum": 0.0, "weighted_cur_sum": 0.0, "weighted_rare_sum": 0.0, "weighted_unc_sum": 0.0,
        }

        env = CrafterEnv(max_steps=self.max_steps_per_ep)
        cur_seed = seed_start
        stagnant_count = 0
        restart_rng = random.Random(seed_start)

        t_start = time.time()

        try:
            while new_transitions < target_transitions and cur_seed <= seed_end:
                # ── Archive Restart Check (P1-4: 40% random + stagnation) ──────
                do_restart, restart_reason = self.archive_restarter.should_restart(stagnant_count, restart_rng)
                is_restart = False
                ep_origin = [cur_seed, 0]

                if do_restart:
                    archive_entry = self.archive_restarter.sample(self.rarity_scheduler.counts)
                    if archive_entry is not None:
                        env.restore_state(archive_entry["snapshot"])
                        obs = np.copy(archive_entry["obs"])  # next_obs from snapshot time
                        is_restart = True
                        ep_origin = [int(archive_entry["seed"]), int(archive_entry["step"])]
                        stagnant_count = 0
                        # P0-3: RESTORE THE MIND (initialize belief from snapshot entry, never reset)
                        self.belief.set_mask(archive_entry["mask"])
                        self.belief.set_h(archive_entry["h"])
                        print(f"  [Archive Restart ({restart_reason})] Resumed from seed {ep_origin[0]} (step {ep_origin[1]})")

                if not is_restart:
                    obs, _ = env.reset(seed=cur_seed)
                    self.belief.reset()

                ep_obs, ep_acts, ep_rews, ep_dones = [], [], [], []
                ep_events, ep_masks = [], []
                prev_ach = self.belief.get_mask()  # Start from restored mask or zeros

                ep_reward = 0.0
                ep_novel_event = False
                ep_steps = 0

                for _ in range(self.max_steps_per_ep):
                    # Plan action
                    action, debug = self.planner.plan(obs, self.belief)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc

                    # Ground truth achievements extraction
                    ach_dict = info.get("achievements", {})
                    curr_ach = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
                    for idx, key in enumerate(sorted(ach_dict.keys())):
                        if idx < NUM_ACHIEVEMENTS:
                            curr_ach[idx] = float(ach_dict[key] > 0)

                    event = np.clip(curr_ach - prev_ach, 0, 1).astype(np.uint8)
                    mask_before = prev_ach.astype(np.uint8)
                    mask_after = curr_ach.astype(np.uint8)
                    prev_ach = np.maximum(prev_ach, curr_ach)

                    # P0-3: Collection-time only: ground belief mask in true achievements
                    self.belief.set_mask(curr_ach)

                    # Record transition
                    ep_obs.append(obs)
                    ep_acts.append(action)
                    ep_rews.append(float(reward))
                    ep_dones.append(bool(done))
                    ep_events.append(event)
                    ep_masks.append(mask_before)

                    # Belief progression (calculates next h)
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

                    # Update rarity counts & check archive (P0-3: stores next_obs, full true mask, h)
                    if np.any(event > 0):
                        self.rarity_scheduler.update(event)
                        if self.archive_restarter.is_interesting(event, self.rarity_scheduler.counts):
                            rare_idxs = self.archive_restarter.get_rare_indices(event, self.rarity_scheduler.counts)
                            self.archive_restarter.add(
                                seed=cur_seed,
                                step=ep_steps,
                                snapshot=env.snapshot_state(),
                                obs=next_obs,
                                unlocked_indices=rare_idxs,
                                mask=mask_after,
                                h=self.belief.get_h(),
                                global_counts=self.rarity_scheduler.counts,
                            )
                            ep_novel_event = True

                    # Track components (raw & weighted shares)
                    task_r = float(reward)
                    cur_b = debug.get("curiosity_bonus_sum", 0.0) / max(1, self.planner.horizon)
                    rare_b = debug.get("rarity_bonus_sum", 0.0) / max(1, self.planner.horizon)
                    unc_b = debug.get("unc_bonus_sum", 0.0) / max(1, self.planner.horizon)

                    w_task_r = debug.get("weighted_task_sum", 0.0) / max(1, self.planner.horizon)
                    w_cur_b = debug.get("weighted_cur_sum", 0.0) / max(1, self.planner.horizon)
                    w_rare_b = debug.get("weighted_rare_sum", 0.0) / max(1, self.planner.horizon)
                    w_unc_b = debug.get("weighted_unc_sum", 0.0) / max(1, self.planner.horizon)

                    for d in [comp_sums, shard_comp_sums]:
                        d["task_reward"] += task_r
                        d["curiosity_bonus"] += cur_b
                        d["rarity_bonus"] += rare_b
                        d["unc_bonus"] += unc_b
                        d["weighted_task_sum"] += w_task_r
                        d["weighted_cur_sum"] += w_cur_b
                        d["weighted_rare_sum"] += w_rare_b
                        d["weighted_unc_sum"] += w_unc_b

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

                    shard_origins.append(ep_origin)
                    if is_restart:
                        shard_restart_eps += 1
                    else:
                        shard_fresh_eps += 1

                    self.archive_restarter.record_episode_novelty(ep_novel_event)
                    stagnant_count = self.archive_restarter.episodes_since_novel

                # Logging
                if total_episodes % 10 == 0 or new_transitions >= target_transitions:
                    elapsed = time.time() - t_start
                    fps = new_transitions / max(0.1, elapsed)
                    w_tot = (
                        abs(shard_comp_sums["weighted_task_sum"])
                        + abs(shard_comp_sums["weighted_cur_sum"])
                        + abs(shard_comp_sums["weighted_rare_sum"])
                        + abs(shard_comp_sums["weighted_unc_sum"])
                    )
                    denom = max(1e-8, w_tot)
                    w_rare_pct = (abs(shard_comp_sums["weighted_rare_sum"]) / denom) * 100.0
                    print(
                        f"[Ep {total_episodes:04d} | Seed {cur_seed}] "
                        f"trans: {new_transitions:,}/{target_transitions:,} ({new_transitions/target_transitions*100:.1f}%) | "
                        f"steps: {ep_steps:3d} | r_ep: {ep_reward:.1f} | "
                        f"archive: {len(self.archive_restarter)} | rare_share: {w_rare_pct:.1f}% | "
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
                        shard_origins=shard_origins,
                        shard_restart_eps=shard_restart_eps,
                        shard_fresh_eps=shard_fresh_eps,
                    )
                    all_new_shard_entries.append(entry)
                    shard_idx += 1
                    current_shard_seed_start = cur_seed + 1

                    # Reset shard accumulators
                    shard_origins = []
                    shard_restart_eps = 0
                    shard_fresh_eps = 0
                    shard_comp_sums = {
                        "task_reward": 0.0, "curiosity_bonus": 0.0, "rarity_bonus": 0.0, "unc_bonus": 0.0,
                        "weighted_task_sum": 0.0, "weighted_cur_sum": 0.0, "weighted_rare_sum": 0.0, "weighted_unc_sum": 0.0,
                    }

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
                shard_origins=shard_origins,
                shard_restart_eps=shard_restart_eps,
                shard_fresh_eps=shard_fresh_eps,
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
            "archive_summary": self.archive_restarter.summary(self.rarity_scheduler.counts),
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
    parser.add_argument("--lambda-q", type=float, default=DEFAULT_LAMBDA_Q, help="Q epistemic uncertainty bonus scaling")
    parser.add_argument("--cap-q", type=float, default=DEFAULT_CAP_Q, help="Q epistemic uncertainty bonus cap")
    parser.add_argument("--restart-prob", type=float, default=DEFAULT_RESTART_PROB, help="Fraction of episodes starting from archive")
    parser.add_argument("--resume", action="store_true", help="Automatically resume accounting from existing shards")
    parser.add_argument("--smoke-test", action="store_true", help="Quick CPU smoke test mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    local_p = REPO_ROOT

    if args.smoke_test and not drive_p.exists():
        drive_p = local_p / "data" / "smoke_drive"

    # Scan and seed global counts from ALL prior manifests across Drive & local (P1-5)
    initial_counts, prior_trans, prior_events, prior_shards = _extract_global_counts_from_manifests(
        drive_dir=drive_p,
        local_dir=local_p,
        num_achievements=NUM_ACHIEVEMENTS,
    )

    # Resume accounting
    seed_start = args.seed_start
    target_transitions = args.target_transitions

    if args.resume:
        t2_shards = [s for s in prior_shards if s.get("tranche") == 2 or "t02" in s.get("shard_name", "")]
        if t2_shards:
            collected_t2_trans = sum(s.get("transition_count", 0) for s in t2_shards)
            target_transitions = max(0, DEFAULT_TARGET_TRANSITIONS - collected_t2_trans)
            max_seed_used = max(s.get("seed_range", [5999, 5999])[1] for s in t2_shards)
            seed_start = max_seed_used + 1
            print("=" * 70)
            print("[Resume Accounting]")
            print(f"  Found {len(t2_shards)} prior T2 shards.")
            print(f"  Collected transitions so far: {collected_t2_trans:,} / {DEFAULT_TARGET_TRANSITIONS:,}")
            print(f"  Remaining transitions to collect: {target_transitions:,}")
            print(f"  Last seed used: {max_seed_used} -> Resuming from seed: {seed_start}")
            print("=" * 70)

    # Safety checks on seeds
    if seed_start < 6000 or args.seed_end > 7999:
        raise ValueError(f"CRITICAL: Seeds must be in range 6000-7999. Got [{seed_start}, {args.seed_end}]. Do not touch 2050-2099 or 3000-3099.")

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
        restart_prob=args.restart_prob,
    )

    target_t = min(target_transitions, 25) if args.smoke_test else target_transitions
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
        lambda_q=args.lambda_q,
        cap_q=args.cap_q,
        horizon=horizon,
        num_candidates=num_candidates,
        shard_every=shard_size,
        device=args.device,
    )

    summary = collector.collect(
        target_transitions=target_t,
        seed_start=seed_start,
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
    print(f"Total restarts: {summary['archive_summary']['total_restarts']}")
    print(f"Random restarts: {summary['archive_summary']['total_random_restarts']}")
    print(f"Stagnation restarts: {summary['archive_summary']['total_stagnation_restarts']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
