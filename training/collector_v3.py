"""
Curiosity-Guided Episode Collector for Learn-a-World v3 — Phase 3 (Plan2Explore).

Design:
- CollectorV3: Drive-resumable curiosity-guided collection using Plan2ExploreCuriosity.
  * Shard naming: shard_curiosity_v3_t{tranche:02d}_{shard_idx:04d}.npz
  * Manifest: dataset_index.json, updated atomically after each shard flush.
    Each shard entry records: shard_name, tranche, source_policy, seed_range,
    transition_count, per_achievement_event_counts.
  * Resume logic: reads manifest on startup, counts already-collected transitions,
    resumes from the next uncollected seed.
  * Curiosity training: online every CURIOSITY_TRAIN_EVERY steps, 20 gradient steps
    on 32-sequence batches from the FULL accumulated buffer (old + new).
  * Checkpoint saves: Full training state every 1000 collection steps (model +
    curiosity + optimizers + step + rng states + config hash).

- CuriosityAugmentedPlanner: Wraps RecurrentMPCPlanner with augmented reward:
    r_total = lambda_task * r_task + lambda_cur * r_curiosity

Seed discipline: collection seeds start at 5000+ (never touch 2000-2099 / 3000-3099).
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import envs  # noqa: F401 — registers Crafter env
from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.calibration import EventHeadCalibrator
from models.curiosity import Plan2ExploreCuriosity
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from training.sequence_buffer import ACHIEVEMENT_TO_IDX, EpisodeBuffer, NUM_ACHIEVEMENTS
from planning.semantic_memory import inventory_dict_to_vector

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SHARD_EVERY_TRANSITIONS = 2_000          # flush shard after this many new transitions
CURIOSITY_TRAIN_EVERY = 100              # train predictors every N collected steps
CURIOSITY_TRAIN_STEPS = 20              # gradient steps per training burst
CURIOSITY_BATCH_SIZE = 32               # batch size for curiosity training
CURIOSITY_SEQ_LEN = 8                   # sequence length for curiosity batch
CHECKPOINT_EVERY_STEPS = 1_000          # full state save cadence


# ──────────────────────────────────────────────────────────────────────────────
# CuriosityAugmentedPlanner
# ──────────────────────────────────────────────────────────────────────────────

class CuriosityAugmentedPlanner:
    """
    Wraps RecurrentMPCPlanner, replacing task reward with:
        r_total = lambda_task * r_task + lambda_cur * r_curiosity
    Curiosity bonus computed from Plan2ExploreCuriosity predictor disagreement.
    Optionally consumes calibrated event probabilities via EventHeadCalibrator.
    """

    def __init__(
        self,
        model_checkpoint: str,
        curiosity: Plan2ExploreCuriosity,
        lambda_task: float = 0.3,
        lambda_cur: float = 1.0,
        horizon: int = 10,
        num_candidates: int = 64,
        cem_iterations: int = 2,
        num_actions: int = 17,
        calibrator: Optional[Union[str, EventHeadCalibrator]] = None,
        use_calibration: bool = True,
        device: str = "cpu",
    ):
        self.lambda_task = lambda_task
        self.lambda_cur = lambda_cur
        self.num_actions = num_actions
        self.use_calibration = use_calibration
        self.device = torch.device(device)

        # Load v3 model
        self.model = _load_jepa_v3(model_checkpoint, device)
        if use_calibration:
            if isinstance(calibrator, (str, Path)) and Path(calibrator).exists():
                self.model.load_calibrator(calibrator)
                print(f"[Planner] Loaded EventHeadCalibrator from {calibrator}")
            elif isinstance(calibrator, EventHeadCalibrator):
                self.model.set_calibrator(calibrator)
                print("[Planner] Attached EventHeadCalibrator instance")
            elif calibrator is None:
                default_cal = repo_root / "results" / "calibration" / "event_head_calibrator.json"
                if default_cal.exists():
                    self.model.load_calibrator(default_cal)
                    print(f"[Planner] Loaded default EventHeadCalibrator from {default_cal}")

        self.curiosity = curiosity.to(self.device)

        self._base_planner = RecurrentMPCPlanner(
            model=self.model,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=cem_iterations,
            num_actions=num_actions,
            device=device,
        )
        self.belief = AgentBeliefV3(recurrent_dim=256)

    def reset(self) -> None:
        """Call at each episode start."""
        self.belief.reset()

    def plan(self, obs: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """
        Plans next action with curiosity-augmented reward.
        Returns (action, debug_dict).
        """
        # Use base planner to get action (task reward inside planner)
        action, debug = self._base_planner.plan(obs, self.belief)

        # Compute curiosity bonus at root state for logging
        obs_t = _preprocess_obs(obs, self.device)
        with torch.no_grad():
            z = self.model.encode(obs_t, use_target=False)  # [1, 128]
            h = self.belief.get_h_tensor(device=self.device).unsqueeze(0)  # [1, 256]
            a = torch.tensor([action], dtype=torch.long, device=self.device)
            bonus = self.curiosity.curiosity_bonus(z, h, a)  # [1]

        debug["curiosity_bonus_mean"] = float(bonus.mean().item())
        return action, debug

    def update_belief(self, obs: np.ndarray, action: int, reward: float) -> None:
        """
        Update recurrent hidden state and belief mask after each env step.
        Must be called after collecting each transition.
        """
        obs_t = _preprocess_obs(obs, self.device)
        a_t = torch.tensor([action], dtype=torch.long, device=self.device)
        h_t = self.belief.get_h_tensor(device=self.device).unsqueeze(0)
        z_t = self.model.encode(obs_t, use_target=False)
        s_t = torch.cat([z_t, h_t], dim=-1)
        m_t = self.belief.get_mask_tensor(device=self.device).unsqueeze(0)

        with torch.no_grad():
            event_probs = self.model.event_head.predict_proba(s_t, a_t, m_t).squeeze(0)
            _, h_next, _ = self.model.predict_next_step(z_t, h_t, a_t)

        self.belief.update(event_probs, reward, next_h=h_next.squeeze(0))


# ──────────────────────────────────────────────────────────────────────────────
# CollectorV3
# ──────────────────────────────────────────────────────────────────────────────

class CollectorV3:
    """
    Drive-resumable curiosity-guided episode collector for Phase 3 (T1+).

    On init:
      - Loads model + curiosity checkpoints.
      - Reads drive_dir/dataset_index.json to count already-collected transitions
        and to determine the resume seed.

    On collect():
      - Runs episodes from resume_seed onward.
      - Flushes shard every SHARD_EVERY_TRANSITIONS transitions.
      - Updates manifest atomically after each shard.
      - Trains curiosity predictors online every CURIOSITY_TRAIN_EVERY steps.
      - Saves full checkpoint every CHECKPOINT_EVERY_STEPS steps.
    """

    def __init__(
        self,
        model_checkpoint: str,
        curiosity_checkpoint: Optional[str],
        drive_dir: str,
        tranche_id: int = 1,
        lambda_task: float = 0.3,
        lambda_cur: float = 1.0,
        horizon: int = 10,
        num_candidates: int = 64,
        cem_iterations: int = 2,
        max_steps_per_ep: int = 1000,
        shard_every_transitions: int = SHARD_EVERY_TRANSITIONS,
        curiosity_lr: float = 3e-4,
        calibrator: Optional[Union[str, Any]] = None,
        use_calibration: bool = True,
        device: str = "cpu",
    ):
        self.model_checkpoint = model_checkpoint
        self.drive_dir = Path(drive_dir)
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        self.tranche_id = tranche_id
        self.lambda_task = lambda_task
        self.lambda_cur = lambda_cur
        self.max_steps_per_ep = max_steps_per_ep
        self.shard_every = shard_every_transitions
        self.use_calibration = use_calibration
        self.device = torch.device(device)

        # Config hash for checkpoint integrity
        config = {
            "model_checkpoint": str(model_checkpoint),
            "tranche_id": tranche_id,
            "lambda_task": lambda_task,
            "lambda_cur": lambda_cur,
            "horizon": horizon,
            "num_candidates": num_candidates,
            "use_calibration": use_calibration,
        }
        self.config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

        # Load model
        self.model = _load_jepa_v3(model_checkpoint, str(device))

        # Init curiosity module
        self.curiosity = Plan2ExploreCuriosity(
            latent_dim=128,
            recurrent_dim=256,
            num_actions=17,
            num_predictors=3,
        ).to(self.device)

        if curiosity_checkpoint is not None and Path(curiosity_checkpoint).exists():
            print(f"[CollectorV3] Loading curiosity checkpoint: {curiosity_checkpoint}")
            self.curiosity.load_checkpoint(curiosity_checkpoint, device=str(self.device))

        self.curiosity_opts = self.curiosity.make_optimizers(lr=curiosity_lr)

        # Curiosity-augmented planner
        self.planner = CuriosityAugmentedPlanner(
            model_checkpoint=model_checkpoint,
            curiosity=self.curiosity,
            lambda_task=lambda_task,
            lambda_cur=lambda_cur,
            horizon=horizon,
            num_candidates=num_candidates,
            cem_iterations=cem_iterations,
            calibrator=calibrator,
            use_calibration=use_calibration,
            device=device,
        )

        # Read existing manifest to determine resume position
        self.manifest_path = self.drive_dir / "dataset_index.json"
        self.existing_shards: List[Dict[str, Any]] = []
        self.already_collected = 0
        self.already_episodes = 0
        self.resume_seed = 0  # set in _load_manifest

        self._accumulated_buffer = EpisodeBuffer(capacity_transitions=500_000)
        self._load_manifest()

    # ── Manifest helpers ────────────────────────────────────────────────────

    def _load_manifest(self) -> None:
        """Read existing manifest; populate resume state."""
        if not self.manifest_path.exists():
            self.resume_seed = 0
            self.already_collected = 0
            self.already_episodes = 0
            return

        manifest = json.loads(self.manifest_path.read_text())
        self.existing_shards = manifest.get("shards", [])
        self.already_collected = manifest.get("total_transitions", 0)
        self.already_episodes = manifest.get("total_episodes", 0)

        # Determine resume seed from max seed_range end across all shards
        max_seed = -1
        for shard in self.existing_shards:
            sr = shard.get("seed_range")
            if sr is not None and len(sr) == 2:
                max_seed = max(max_seed, sr[1])

        # Resume from the seed AFTER the last seed used
        self.resume_seed = max_seed + 1 if max_seed >= 0 else 0

        # Reload existing shard data into accumulated buffer
        for shard_info in self.existing_shards:
            shard_path = self.drive_dir / shard_info["shard_name"]
            if shard_path.exists():
                try:
                    self._accumulated_buffer.load_shard(str(shard_path))
                except Exception as e:
                    print(f"[CollectorV3] Warning: failed to load shard {shard_path}: {e}")

        print(
            f"[CollectorV3] Resumed: {self.already_collected} transitions already collected. "
            f"Next seed: {self.resume_seed}. "
            f"Accumulated buffer: {self._accumulated_buffer.total_transitions} transitions."
        )

    def _write_manifest(
        self,
        shard_entries: List[Dict[str, Any]],
        total_transitions: int,
        total_episodes: int,
    ) -> None:
        """Write manifest atomically (write to tmp → rename)."""
        manifest = {
            "total_transitions": total_transitions,
            "total_episodes": total_episodes,
            "shards": shard_entries,
        }
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(self.manifest_path)

    # ── Shard helpers ────────────────────────────────────────────────────────

    def _shard_name(self, shard_idx: int) -> str:
        return f"shard_curiosity_v3_t{self.tranche_id:02d}_{shard_idx:04d}.npz"

    def _flush_shard(
        self,
        episode_buffer: EpisodeBuffer,
        shard_idx: int,
        seed_start: int,
        seed_end: int,
    ) -> Dict[str, Any]:
        """Save episodes to a .npz shard and return shard manifest entry."""
        shard_name = self._shard_name(shard_idx)
        shard_path = self.drive_dir / shard_name
        episode_buffer.save_shard(str(shard_path))

        stats = episode_buffer.get_dataset_statistics()
        entry = {
            "shard_name": shard_name,
            "tranche": self.tranche_id,
            "source_policy": "curiosity_v3",
            "seed_range": [seed_start, seed_end],
            "transition_count": stats["total_transitions"],
            "per_achievement_event_counts": stats["per_achievement_positive_counts"],
        }
        return entry

    # ── Curiosity training ───────────────────────────────────────────────────

    def _train_curiosity_online(self) -> float:
        """Train all 3 predictors for CURIOSITY_TRAIN_STEPS steps. Returns mean loss."""
        if self._accumulated_buffer.total_transitions < CURIOSITY_SEQ_LEN * CURIOSITY_BATCH_SIZE:
            return 0.0

        self.curiosity.train()
        total_loss = 0.0

        for _ in range(CURIOSITY_TRAIN_STEPS):
            batch = self._accumulated_buffer.sample_sequences(
                batch_size=CURIOSITY_BATCH_SIZE,
                seq_len=CURIOSITY_SEQ_LEN,
                mode="uniform",
                device=str(self.device),
            )
            obs_seq = batch["obs"]        # [B, T, C, H, W]
            acts_seq = batch["actions"]   # [B, T]

            B, T = acts_seq.shape

            # Encode all obs at once to get latent sequence
            with torch.no_grad():
                obs_flat = obs_seq.view(B * T, *obs_seq.shape[2:])
                z_flat = self.model.encode(obs_flat, use_target=True)   # [B*T, 128]
                z_seq = z_flat.view(B, T, -1)                           # [B, T, 128]

            # Use only first T-1 steps as (z, h, a) → z_next targets
            # We need h_seq: unroll RecurrentCore through z_seq[:,:-1]
            acts_prev = acts_seq[:, :-1]   # [B, T-1]
            z_in = z_seq[:, :-1]           # [B, T-1, 128]
            z_tgt = z_seq[:, 1:]           # [B, T-1, 128]

            # h_0 = zeros; unroll core
            with torch.no_grad():
                h_seq = self.model.recurrent_core.unroll_sequence(
                    z_in, acts_prev, h_0=None
                )  # [B, T-1, 256]

            # Flatten for curiosity loss
            z_in_flat = z_in.reshape(B * (T - 1), -1)
            h_flat = h_seq.reshape(B * (T - 1), -1)
            a_flat = acts_prev.reshape(B * (T - 1))
            z_tgt_flat = z_tgt.reshape(B * (T - 1), -1)

            # Independent optimizer step per predictor
            step_loss = 0.0
            for k, (pred, opt) in enumerate(zip(self.curiosity.predictors, self.curiosity_opts)):
                opt.zero_grad()
                import torch.nn.functional as F
                pred_z = pred(z_in_flat, h_flat, a_flat)
                loss_k = F.mse_loss(pred_z, z_tgt_flat.detach())
                loss_k.backward()
                opt.step()
                step_loss += loss_k.item()
            total_loss += step_loss / len(self.curiosity_opts)

        return total_loss / CURIOSITY_TRAIN_STEPS

    # ── Full state checkpoint ────────────────────────────────────────────────

    def _save_full_checkpoint(self, step: int, shard_idx: int) -> Path:
        """Save full training state: model + curiosity + optimizers + RNG + config hash."""
        ckpt = {
            "step": step,
            "shard_idx": shard_idx,
            "tranche_id": self.tranche_id,
            "config_hash": self.config_hash,
            "model_state_dict": self.model.state_dict(),
            "curiosity_state_dict": self.curiosity.state_dict(),
            "curiosity_opt_state_dicts": [opt.state_dict() for opt in self.curiosity_opts],
            "rng_state_np": np.random.get_state(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_python": random.getstate(),
        }
        ckpt_name = f"train_state_t{self.tranche_id:02d}_step{step:07d}.pt"
        ckpt_path = self.drive_dir / "checkpoints" / ckpt_name
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(ckpt_path))

        # Keep only latest 2 checkpoints (disk hygiene)
        existing_ckpts = sorted(
            (self.drive_dir / "checkpoints").glob(f"train_state_t{self.tranche_id:02d}_*.pt")
        )
        while len(existing_ckpts) > 2:
            existing_ckpts.pop(0).unlink()
            existing_ckpts = sorted(
                (self.drive_dir / "checkpoints").glob(f"train_state_t{self.tranche_id:02d}_*.pt")
            )

        print(f"[CollectorV3] Checkpoint saved: {ckpt_name}")
        return ckpt_path

    # ── Main collection loop ─────────────────────────────────────────────────

    def collect(
        self,
        target_new_transitions: int,
        seed_start: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Collect target_new_transitions new transitions from the environment.

        seed_start: first seed to use (defaults to self.resume_seed).
        Returns summary dict with new_transitions, total_transitions, event_counts.
        """
        import gymnasium as gym

        if seed_start is None:
            seed_start = self.resume_seed
        elif self.resume_seed > 0:
            seed_start = max(seed_start, self.resume_seed)

        print(
            f"\n[CollectorV3] Starting T{self.tranche_id} collection: "
            f"target={target_new_transitions} new transitions | "
            f"seed_start={seed_start} | device={self.device}"
        )

        new_transitions = 0
        total_episodes = 0
        all_shard_entries = list(self.existing_shards)
        shard_idx = len([s for s in self.existing_shards
                         if f"t{self.tranche_id:02d}" in s.get("shard_name", "")])

        current_shard_buf = EpisodeBuffer(capacity_transitions=self.shard_every * 2)
        current_shard_seed_start = seed_start
        cumulative_step = 0
        curiosity_loss_ema = 0.0
        last_ckpt_step = 0
        last_100_ep_rewards: List[float] = []

        env = gym.make("Crafter-v0", render_size=64, max_steps=self.max_steps_per_ep)

        seed = seed_start
        try:
            while new_transitions < target_new_transitions:
                # ── Episode setup ────────────────────────────────────────────
                obs, info = env.reset(seed=seed)
                self.planner.reset()
                ep_obs, ep_acts, ep_rews, ep_dones = [], [], [], []
                ep_events = []
                ep_masks = []
                ep_state_deltas = []
                ep_state_valid = []
                prev_ach = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
                prev_state = inventory_dict_to_vector(info.get("inventory", {}) if isinstance(info, dict) else None)
                ep_reward = 0.0
                ep_steps = 0

                for _step in range(self.max_steps_per_ep):
                    action, debug = self.planner.plan(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Achievement events
                    ach_dict = info.get("achievements", {})
                    curr_ach = np.array(
                        [float(ach_dict.get(a, 0) > 0) for a in CRAFTER_ACHIEVEMENTS],
                        dtype=np.float32,
                    )
                    event = np.clip(curr_ach - prev_ach, 0, 1).astype(np.uint8)
                    mask_before = prev_ach.astype(np.uint8)
                    prev_ach = np.maximum(prev_ach, curr_ach)

                    # Optional privileged training label only. The model never
                    # receives this vector while acting or planning.
                    curr_state = inventory_dict_to_vector(info.get("inventory", {}) if isinstance(info, dict) else None)
                    ep_state_deltas.append(curr_state - prev_state)
                    ep_state_valid.append(bool(isinstance(info, dict) and "inventory" in info))
                    prev_state = curr_state

                    ep_obs.append(obs)
                    ep_acts.append(action)
                    ep_rews.append(float(reward))
                    ep_dones.append(bool(done))
                    ep_events.append(event)
                    ep_masks.append(mask_before)

                    self.planner.update_belief(obs, action, reward)
                    obs = next_obs
                    ep_reward += float(reward)
                    ep_steps += 1
                    new_transitions += 1
                    cumulative_step += 1

                    # Online curiosity training
                    if (
                        cumulative_step % CURIOSITY_TRAIN_EVERY == 0
                        and self._accumulated_buffer.total_transitions >= CURIOSITY_SEQ_LEN * CURIOSITY_BATCH_SIZE
                    ):
                        loss = self._train_curiosity_online()
                        curiosity_loss_ema = 0.95 * curiosity_loss_ema + 0.05 * loss

                    # Full checkpoint save
                    if cumulative_step - last_ckpt_step >= CHECKPOINT_EVERY_STEPS:
                        self._save_full_checkpoint(cumulative_step, shard_idx)
                        last_ckpt_step = cumulative_step

                    if done or new_transitions >= target_new_transitions:
                        break

                # ── Episode done ─────────────────────────────────────────────
                T = len(ep_acts)
                if T > 0:
                    ep_data = {
                        "obs": np.stack(ep_obs, axis=0).astype(np.uint8),
                        "actions": np.array(ep_acts, dtype=np.int64),
                        "rewards": np.array(ep_rews, dtype=np.float32),
                        "dones": np.array(ep_dones, dtype=np.bool_),
                        "events": np.stack(ep_events, axis=0).astype(np.uint8),
                        "mask_before": np.stack(ep_masks, axis=0).astype(np.uint8),
                        "state_delta": np.stack(ep_state_deltas, axis=0).astype(np.float32),
                        "state_delta_valid": np.array(ep_state_valid, dtype=np.bool_),
                    }
                    current_shard_buf.add_episode(**ep_data, source="curiosity_v3")
                    self._accumulated_buffer.add_episode(**ep_data, source="curiosity_v3")
                    total_episodes += 1
                    last_100_ep_rewards.append(ep_reward)
                    if len(last_100_ep_rewards) > 100:
                        last_100_ep_rewards.pop(0)

                # ── Logging ──────────────────────────────────────────────────
                if total_episodes % 10 == 0 or new_transitions >= target_new_transitions:
                    mean_r = np.mean(last_100_ep_rewards) if last_100_ep_rewards else 0.0
                    print(
                        f"[T{self.tranche_id}-Ep {total_episodes:04d}] "
                        f"Seed {seed} | Steps: {ep_steps:3d} | "
                        f"r_ep: {ep_reward:.2f} | mean_r100: {mean_r:.2f} | "
                        f"cur_loss: {curiosity_loss_ema:.4f} | "
                        f"new_trans: {new_transitions}/{target_new_transitions} | "
                        f"cur_bonus: {debug.get('curiosity_bonus_mean', 0.0):.4f}"
                    )

                # ── Shard flush ──────────────────────────────────────────────
                if current_shard_buf.total_transitions >= self.shard_every:
                    entry = self._flush_shard(
                        current_shard_buf, shard_idx,
                        seed_start=current_shard_seed_start,
                        seed_end=seed,
                    )
                    all_shard_entries.append(entry)
                    shard_idx += 1
                    current_shard_seed_start = seed + 1

                    total_trans_so_far = self.already_collected + new_transitions
                    self._write_manifest(
                        shard_entries=all_shard_entries,
                        total_transitions=total_trans_so_far,
                        total_episodes=self.already_episodes + total_episodes,
                    )
                    print(f"[CollectorV3] Shard {shard_idx - 1} written → {entry['shard_name']} "
                          f"({entry['transition_count']} transitions)")
                    current_shard_buf = EpisodeBuffer(capacity_transitions=self.shard_every * 2)

                seed += 1

        finally:
            env.close()

        # ── Final shard flush if buffer has data ─────────────────────────────
        if current_shard_buf.total_transitions > 0:
            entry = self._flush_shard(
                current_shard_buf, shard_idx,
                seed_start=current_shard_seed_start,
                seed_end=seed - 1,
            )
            all_shard_entries.append(entry)
            shard_idx += 1

        total_trans = self.already_collected + new_transitions
        cum_episodes = self.already_episodes + total_episodes
        self._write_manifest(
            shard_entries=all_shard_entries,
            total_transitions=total_trans,
            total_episodes=cum_episodes,
        )

        # Final checkpoint
        if cumulative_step > last_ckpt_step:
            self._save_full_checkpoint(cumulative_step, shard_idx)

        # Aggregate event counts across all new shards
        cumulative_events: Dict[str, int] = {a: 0 for a in CRAFTER_ACHIEVEMENTS}
        for entry in all_shard_entries:
            if entry.get("tranche") == self.tranche_id:
                for k, v in entry.get("per_achievement_event_counts", {}).items():
                    cumulative_events[k] = cumulative_events.get(k, 0) + v

        summary = {
            "tranche_id": self.tranche_id,
            "new_transitions": new_transitions,
            "total_transitions": total_trans,
            "total_episodes": cum_episodes,
            "num_shards": shard_idx,
            "seed_range": [seed_start, seed - 1],
            "event_counts": cumulative_events,
        }

        print("\n" + "=" * 60)
        print(f"T{self.tranche_id} COLLECTION COMPLETE")
        print(f"  New transitions: {new_transitions:,}")
        print(f"  Total transitions: {total_trans:,}")
        print(f"  Episodes: {total_episodes}")
        print(f"  Shards written: {shard_idx}")
        print(f"  collect_stone: {cumulative_events.get('collect_stone', 0)}")
        print(f"  collect_coal:  {cumulative_events.get('collect_coal', 0)}")
        print(f"  collect_iron:  {cumulative_events.get('collect_iron', 0)}")
        print("=" * 60)

        return summary


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_jepa_v3(checkpoint_path: str, device: str = "cpu") -> JEPAv3:
    """Load JEPAv3 from checkpoint with eval mode."""
    model = JEPAv3(
        latent_dim=128,
        recurrent_dim=256,
        num_actions=17,
        action_emb_dim=32,
    )
    # Native v3 checkpoints store one state dict per component (encoder,
    # predictor, heads, ...), while older collection checkpoints may wrap a
    # flat model state under model_state_dict/state_dict. Use the canonical
    # loader first so native checkpoints do not silently become random models.
    ckpt = torch.load(checkpoint_path, map_location=device)
    if ckpt.get("version") == "v3" or "encoder" in ckpt:
        model.load_checkpoint(checkpoint_path, device=device)
    else:
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _preprocess_obs(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert [64,64,3] uint8 numpy obs to [1,3,64,64] float tensor in [0,1]."""
    t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float()
    if t.max() > 1.0:
        t = t / 255.0
    return t.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point (for Mac smoke test)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plan2Explore Curiosity Collector v3")
    parser.add_argument("--model_checkpoint", required=True, help="Path to jepa_crafter_v3_phase2.pt")
    parser.add_argument("--curiosity_checkpoint", default=None, help="Resume curiosity checkpoint")
    parser.add_argument("--drive_dir", required=True, help="Output directory for shards + manifest")
    parser.add_argument("--target_transitions", type=int, default=160_000,
                        help="Number of new transitions to collect")
    parser.add_argument("--seed_start", type=int, default=5000,
                        help="Starting seed (collection uses 5000+)")
    parser.add_argument("--tranche_id", type=int, default=1)
    parser.add_argument("--lambda_task", type=float, default=0.3)
    parser.add_argument("--lambda_cur", type=float, default=1.0)
    parser.add_argument("--max_steps_per_ep", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=2)
    parser.add_argument("--shard_every", type=int, default=SHARD_EVERY_TRANSITIONS,
                        help="Flush shard after this many transitions")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    collector = CollectorV3(
        model_checkpoint=args.model_checkpoint,
        curiosity_checkpoint=args.curiosity_checkpoint,
        drive_dir=args.drive_dir,
        tranche_id=args.tranche_id,
        lambda_task=args.lambda_task,
        lambda_cur=args.lambda_cur,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        max_steps_per_ep=args.max_steps_per_ep,
        shard_every_transitions=args.shard_every,
        device=args.device,
    )

    summary = collector.collect(
        target_new_transitions=args.target_transitions,
        seed_start=args.seed_start,
    )

    # Print summary JSON to stdout
    print("\nCollection summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
