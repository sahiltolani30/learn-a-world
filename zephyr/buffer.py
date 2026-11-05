"""Zephyr replay: episode records, boundary-safe sequence sampling, shards.

Episodes are stored whole; sampled windows never cross episode boundaries
(short episodes are padded by repeating their final frame with valid=False).
Observations stay uint8 end to end; only the model converts to float.

Shard format ("zephyr_v1"): one .npz with concatenated arrays plus ``lengths``,
``seeds``, ``sources`` and a JSON ``meta`` string. Shards are the on-disk source
of truth; the in-memory buffer is a bounded FIFO view used for training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from zephyr.config import IMG_SIZE, NUM_ACHIEVEMENTS

_CHECKPOINT_FORMAT = "zephyr_v1"
_SHARD_SOURCE_LEN = 16

_INV_DIM = 16  # config.INVENTORY_KEYS length; kept literal to stay torch/config-light


@dataclass
class EpisodeRecord:
    """One finished episode with per-step privileged diagnostic labels."""

    obs: np.ndarray             # uint8 [T, 64, 64, 3]
    actions: np.ndarray         # int16 [T]
    rewards: np.ndarray         # float32 [T] (extrinsic env reward)
    rewards_int: np.ndarray     # float32 [T] (intrinsic; zeros for random data)
    done: np.ndarray            # bool [T]
    inventory: np.ndarray       # float32 [T, 16]
    achievements_cum: np.ndarray  # bool [T, 22]
    achievements_new: np.ndarray  # bool [T, 22]
    seed: int
    source: str                 # "random" | "actor" | "curriculum" | ...

    def __post_init__(self) -> None:
        t = self.obs.shape[0]
        if self.actions.shape != (t,):
            raise ValueError(f"actions shape {self.actions.shape} != ({t},)")
        if self.rewards.shape != (t,) or self.rewards_int.shape != (t,) or self.done.shape != (t,):
            raise ValueError("rewards/rewards_int/done must be [T]")
        if self.inventory.shape != (t, _INV_DIM):
            raise ValueError(f"inventory must be [T, {_INV_DIM}], got {self.inventory.shape}")
        if self.achievements_cum.shape != (t, NUM_ACHIEVEMENTS) or \
                self.achievements_new.shape != (t, NUM_ACHIEVEMENTS):
            raise ValueError("achievements arrays must be [T, 22]")


class ZephyrReplay:
    """In-memory FIFO of EpisodeRecords bounded by frame count."""

    def __init__(self, capacity_frames: Optional[int] = None) -> None:
        self.capacity_frames = capacity_frames
        self._episodes: List[EpisodeRecord] = []
        self._frames = 0

    # -- container -----------------------------------------------------------

    def add_episode(self, rec: EpisodeRecord) -> None:
        self._episodes.append(rec)
        self._frames += int(rec.obs.shape[0])
        if self.capacity_frames is not None:
            while self._frames > self.capacity_frames and len(self._episodes) > 1:
                dropped = self._episodes.pop(0)
                self._frames -= int(dropped.obs.shape[0])

    def __len__(self) -> int:
        return self._frames

    @property
    def episodes(self) -> int:
        return len(self._episodes)

    @property
    def records(self) -> List[EpisodeRecord]:
        return list(self._episodes)

    # -- sampling ------------------------------------------------------------

    def sample(
        self,
        batch: int,
        seq_len: int,
        rng: np.random.Generator,
        prioritize_achievement_frac: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        if not self._episodes:
            raise RuntimeError("sample from empty replay")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        frac = float(np.clip(prioritize_achievement_frac, 0.0, 1.0))
        pool = [i for i, ep in enumerate(self._episodes) if ep.achievements_new.any()]

        b_obs = np.zeros((batch, seq_len, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        b_obs_next = np.zeros_like(b_obs)
        b_act = np.zeros((batch, seq_len), dtype=np.int64)
        b_rew = np.zeros((batch, seq_len), dtype=np.float32)
        b_rew_int = np.zeros((batch, seq_len), dtype=np.float32)
        b_done = np.zeros((batch, seq_len), dtype=bool)
        b_valid = np.zeros((batch, seq_len), dtype=bool)
        b_inv = np.zeros((batch, seq_len, _INV_DIM), dtype=np.float32)
        b_inv_next = np.zeros((batch, seq_len, _INV_DIM), dtype=np.float32)
        b_ach = np.zeros((batch, seq_len, NUM_ACHIEVEMENTS), dtype=bool)

        for b in range(batch):
            use_pool = pool and (rng.random() < frac)
            idx = int(rng.choice(pool)) if use_pool else int(rng.integers(0, len(self._episodes)))
            ep = self._episodes[idx]
            t_total = ep.obs.shape[0]
            if t_total >= seq_len:
                start = int(rng.integers(0, t_total - seq_len + 1))
                length = seq_len
                valid = np.ones(seq_len, dtype=bool)
            else:
                start = 0
                length = t_total
                valid = np.arange(seq_len) < t_total
            sl = slice(start, start + length)
            pad = seq_len - length
            reaches_end = (start + length) >= ep.obs.shape[0]
            b_obs[b, :length] = ep.obs[sl]
            b_obs_next[b, : length - 1] = ep.obs[start + 1 : start + length]
            # window's last next-frame: true next frame mid-episode; the episode's
            # own final frame (bootstrap) when the window reaches the episode end
            b_obs_next[b, length - 1] = ep.obs[-1] if reaches_end else ep.obs[start + length]
            if pad:
                b_obs[b, length:] = ep.obs[-1]
                b_obs_next[b, length:] = ep.obs[-1]
            for key, buf, src in (
                ("act", b_act, ep.actions),
                ("rew", b_rew, ep.rewards),
                ("rew_int", b_rew_int, ep.rewards_int),
                ("done", b_done, ep.done),
                ("inv", b_inv, ep.inventory),
                ("inv_next", b_inv_next, ep.inventory),
                ("ach", b_ach, ep.achievements_new),
            ):
                if key == "inv_next":
                    buf[b, : length - 1] = ep.inventory[start + 1 : start + length]
                    buf[b, length - 1] = (
                        ep.inventory[-1] if reaches_end else ep.inventory[start + length]
                    )
                else:
                    buf[b, :length] = src[sl]
                if pad:
                    buf[b, length:] = src[-1]
            b_valid[b] = valid

        return {
            "obs": torch.from_numpy(b_obs),
            "obs_next": torch.from_numpy(b_obs_next),
            "actions": torch.from_numpy(b_act),
            "rewards": torch.from_numpy(b_rew),
            "rewards_int": torch.from_numpy(b_rew_int),
            "done": torch.from_numpy(b_done),
            "valid": torch.from_numpy(b_valid),
            "inventory": torch.from_numpy(b_inv),
            "inventory_next": torch.from_numpy(b_inv_next),
            "achievements_new": torch.from_numpy(b_ach),
        }

    # -- stats / split ---------------------------------------------------------

    def achievement_counts(self) -> Dict[str, int]:
        from zephyr.config import ACHIEVEMENTS

        counts = {name: 0 for name in ACHIEVEMENTS}
        for ep in self._episodes:
            for i, name in enumerate(ACHIEVEMENTS):
                if ep.achievements_new[:, i].any():
                    counts[name] += 1
        return counts

    def labeled_fraction(self) -> float:
        if not self._episodes:
            return 0.0
        total = 0
        labeled = 0
        for ep in self._episodes:
            total += ep.inventory.shape[0]
            labeled += int(np.isfinite(ep.inventory).all(axis=1).sum())
        return labeled / max(1, total)

    def split(self, val_ratio: float, seed: int = 4242) -> Tuple["ZephyrReplay", "ZephyrReplay"]:
        n = len(self._episodes)
        if n < 2:
            raise ValueError("need >= 2 episodes to split")
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        n_val = max(1, int(round(n * float(np.clip(val_ratio, 0.0, 0.9)))))
        val_idx = set(order[:n_val].tolist())
        train = ZephyrReplay(capacity_frames=None)
        val = ZephyrReplay(capacity_frames=None)
        for i, ep in enumerate(self._episodes):
            (val if i in val_idx else train).add_episode(ep)
        return train, val

    # -- persistence -----------------------------------------------------------

    def save_shard(self, path: str) -> None:
        if not self._episodes:
            raise RuntimeError("refusing to save an empty shard")
        lengths = np.asarray([ep.obs.shape[0] for ep in self._episodes], dtype=np.int64)
        seeds = np.asarray([ep.seed for ep in self._episodes], dtype=np.int64)
        sources = np.asarray(
            [ep.source[:_SHARD_SOURCE_LEN].ljust(_SHARD_SOURCE_LEN) for ep in self._episodes],
            dtype=f"<U{_SHARD_SOURCE_LEN}",
        )
        meta = json.dumps(
            {
                "format": _CHECKPOINT_FORMAT,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "episodes": int(lengths.size),
                "frames": int(lengths.sum()),
            },
            sort_keys=True,
        )
        np.savez_compressed(
            path,
            obs=np.concatenate([ep.obs for ep in self._episodes], axis=0),
            actions=np.concatenate([ep.actions for ep in self._episodes], axis=0).astype(np.int16),
            rewards=np.concatenate([ep.rewards for ep in self._episodes], axis=0),
            rewards_int=np.concatenate([ep.rewards_int for ep in self._episodes], axis=0),
            done=np.concatenate([ep.done for ep in self._episodes], axis=0),
            inventory=np.concatenate([ep.inventory for ep in self._episodes], axis=0),
            achievements_cum=np.concatenate([ep.achievements_cum for ep in self._episodes], axis=0),
            achievements_new=np.concatenate([ep.achievements_new for ep in self._episodes], axis=0),
            lengths=lengths,
            seeds=seeds,
            sources=sources,
            meta=np.asarray(meta),
        )

    @classmethod
    def load_shard(cls, path: str) -> "ZephyrReplay":
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            if meta.get("format") != _CHECKPOINT_FORMAT:
                raise ValueError(f"unexpected shard format {meta.get('format')!r} in {path}")
            obs = npz["obs"]
            actions = npz["actions"]
            rewards = npz["rewards"]
            rewards_int = npz["rewards_int"]
            done = npz["done"]
            inventory = npz["inventory"]
            ach_cum = npz["achievements_cum"]
            ach_new = npz["achievements_new"]
            lengths = npz["lengths"]
            seeds = npz["seeds"]
            sources = npz["sources"]
        replay = cls(capacity_frames=None)
        offset = 0
        for i, length in enumerate(lengths.tolist()):
            sl = slice(offset, offset + int(length))
            replay.add_episode(
                EpisodeRecord(
                    obs=obs[sl],
                    actions=actions[sl].astype(np.int16),
                    rewards=rewards[sl].astype(np.float32),
                    rewards_int=rewards_int[sl].astype(np.float32),
                    done=done[sl].astype(bool),
                    inventory=inventory[sl].astype(np.float32),
                    achievements_cum=ach_cum[sl].astype(bool),
                    achievements_new=ach_new[sl].astype(bool),
                    seed=int(seeds[i]),
                    source=str(sources[i]).strip(),
                )
            )
            offset += int(length)
        return replay
