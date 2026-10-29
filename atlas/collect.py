"""EpisodeCollector + ShardLoader: live play data gathering and offline shard ingestion.

Saves collected episodes into self-contained npz shards.
Loads both ATLAS live shards and legacy T2 sequence shards.
Strictness: ground-truth events and inventory are written to shards as TRAINING TARGETS only.
The agent itself NEVER reads env info during rollouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from atlas.arbiter import AtlasAgent
from atlas.config import (
    NUM_ACHIEVEMENTS,
    NUM_ACTIONS,
    STATE_VECTOR_DIM,
    CollectConfig,
)
# Safe pre-registration of eval.metrics to avoid pulling unused stable_baselines3 from eval/__init__.py
import importlib.util
import sys
if "eval.metrics" not in sys.modules:
    _metrics_p = Path(__file__).resolve().parent.parent / "eval" / "metrics.py"
    if _metrics_p.exists():
        _spec = importlib.util.spec_from_file_location("eval.metrics", str(_metrics_p))
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["eval.metrics"] = _mod
            _spec.loader.exec_module(_mod)

from envs.crafter_wrapper import CrafterEnv
from eval.metrics import CRAFTER_ACHIEVEMENTS

INVENTORY_ITEMS = [
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]
INVENTORY_VITALS = ["health", "food", "drink", "energy"]


def _extract_inventory_vector(inv_dict: Dict[str, Any]) -> np.ndarray:
    """Extracts 16-d target: 12 item counts (log1p) + 4 vitals (/16)."""
    vec = np.zeros(STATE_VECTOR_DIM, dtype=np.float32)
    for i, item in enumerate(INVENTORY_ITEMS):
        cnt = float(inv_dict.get(item, 0))
        vec[i] = np.log1p(cnt)
    for j, vital in enumerate(INVENTORY_VITALS):
        cnt = float(inv_dict.get(vital, 0))
        vec[12 + j] = cnt / 16.0
    return vec


def _extract_achievements_vector(ach_dict: Dict[str, Any]) -> np.ndarray:
    """Extracts 22-d cumulative achievement count vector."""
    vec = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
    for i, name in enumerate(CRAFTER_ACHIEVEMENTS):
        vec[i] = float(ach_dict.get(name, 0))
    return vec


class EpisodeCollector:
    """Collects live episodes with an AtlasAgent and stores them in npz shards."""

    def __init__(
        self,
        agent: AtlasAgent,
        out_dir: Optional[Union[str, Path]] = None,
        cfg: Optional[CollectConfig] = None,
    ):
        self.agent = agent
        self.cfg = cfg or CollectConfig()
        self.out_dir = Path(out_dir or self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def set_schedules(self, iter_idx: int) -> None:
        """Updates epsilon and beta exploration parameters according to linear decay."""
        frac = min(1.0, float(iter_idx) / max(1, self.cfg.epsilon_decay_iters))
        eps = (1.0 - frac) * self.cfg.epsilon_start + frac * self.cfg.epsilon_end
        beta = (1.0 - frac) * self.cfg.beta_start + frac * self.cfg.beta_end

        self.agent.epsilon = float(eps)
        self.agent.beta_t = float(beta)
        self.agent.current_iter = int(iter_idx)

    def run_episode(self, seed: int) -> Dict[str, Any]:
        """Runs a single episode in CrafterEnv and returns recorded transition arrays."""
        env = CrafterEnv(seed=seed)
        obs, reset_info = env.reset(seed=seed)

        self.agent.reset_episode(seed=seed)

        obs_list: List[np.ndarray] = []
        action_list: List[int] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        truncated_list: List[bool] = []
        discount_list: List[float] = []
        events_list: List[np.ndarray] = []
        mask_before_list: List[np.ndarray] = []
        inventory_list: List[np.ndarray] = []

        reward_prev = 0.0
        step = 0
        prev_ach_vec = _extract_achievements_vector(reset_info.get("achievements", {}))

        while step < self.cfg.max_steps:
            obs_uint8 = np.asarray(obs, dtype=np.uint8)
            action, telemetry = self.agent.act(obs_uint8, reward_prev, step)

            next_obs, reward, terminated, truncated, info = env.step(action)
            discount = float(info.get("discount", 0.0 if terminated else 1.0))

            # Ground truth targets extracted from env info (SCORING/TRAINING ONLY)
            curr_ach_vec = _extract_achievements_vector(info.get("achievements", {}))
            events_step = (curr_ach_vec > prev_ach_vec).astype(bool)
            mask_before_step = (prev_ach_vec > 0).astype(bool)
            inv_vec = _extract_inventory_vector(info.get("inventory", {}))

            self.agent.observe_result(next_obs, reward, terminated, discount)

            obs_list.append(obs_uint8)
            action_list.append(action)
            reward_list.append(float(reward))
            # dones: real death only
            real_death = bool(terminated and discount == 0.0)
            done_list.append(real_death)
            truncated_list.append(bool(truncated))
            discount_list.append(discount)
            events_list.append(events_step)
            mask_before_list.append(mask_before_step)
            inventory_list.append(inv_vec)

            prev_ach_vec = curr_ach_vec
            reward_prev = float(reward)
            obs = next_obs
            step += 1

            if terminated or truncated:
                break

        return {
            "obs": np.stack(obs_list, axis=0),
            "actions": np.array(action_list, dtype=np.int64),
            "rewards": np.array(reward_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=bool),
            "truncated": np.array(truncated_list, dtype=bool),
            "discounts": np.array(discount_list, dtype=np.float32),
            "events": np.stack(events_list, axis=0),
            "mask_before": np.stack(mask_before_list, axis=0),
            "inventory": np.stack(inventory_list, axis=0),
            "episode_length": step,
            "seed": seed,
        }

    def run_episodes(
        self,
        n: int,
        seed_base: int,
        iter_idx: int = 0,
    ) -> Path:
        """Runs n episodes, aggregates records, and writes to shard npz."""
        self.set_schedules(iter_idx)

        all_obs: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        all_rewards: List[np.ndarray] = []
        all_dones: List[np.ndarray] = []
        all_truncated: List[np.ndarray] = []
        all_discounts: List[np.ndarray] = []
        all_events: List[np.ndarray] = []
        all_mask_before: List[np.ndarray] = []
        all_inventory: List[np.ndarray] = []
        episode_lengths: List[int] = []
        seeds: List[int] = []

        for i in range(n):
            ep_seed = seed_base + i
            ep_data = self.run_episode(ep_seed)
            all_obs.append(ep_data["obs"])
            all_actions.append(ep_data["actions"])
            all_rewards.append(ep_data["rewards"])
            all_dones.append(ep_data["dones"])
            all_truncated.append(ep_data["truncated"])
            all_discounts.append(ep_data["discounts"])
            all_events.append(ep_data["events"])
            all_mask_before.append(ep_data["mask_before"])
            all_inventory.append(ep_data["inventory"])
            episode_lengths.append(ep_data["episode_length"])
            seeds.append(ep_seed)

        shard_path = self.out_dir / f"shard_{iter_idx}_{n}.npz"
        np.savez_compressed(
            shard_path,
            obs=np.concatenate(all_obs, axis=0),
            actions=np.concatenate(all_actions, axis=0),
            rewards=np.concatenate(all_rewards, axis=0),
            dones=np.concatenate(all_dones, axis=0),
            truncated=np.concatenate(all_truncated, axis=0),
            discounts=np.concatenate(all_discounts, axis=0),
            events=np.concatenate(all_events, axis=0),
            mask_before=np.concatenate(all_mask_before, axis=0),
            inventory=np.concatenate(all_inventory, axis=0),
            episode_lengths=np.array(episode_lengths, dtype=np.int64),
            seeds=np.array(seeds, dtype=np.int64),
            source="atlas_live",
        )
        return shard_path


class ShardLoader:
    """Loads and standardizes shards (ATLAS live or T2 sequences format)."""

    @staticmethod
    def load_shard(path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        data = np.load(p)
        keys = set(data.keys())

        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        events = data["events"]
        mask_before = data["mask_before"]
        episode_lengths = data["episode_lengths"]

        T = len(actions)
        if "truncated" in keys:
            truncated = data["truncated"]
        else:
            truncated = np.zeros(T, dtype=bool)

        if "discounts" in keys:
            discounts = data["discounts"]
        else:
            discounts = (1.0 - dones.astype(np.float32)).astype(np.float32)

        has_inventory = "inventory" in keys
        if has_inventory:
            inventory = data["inventory"]
        else:
            inventory = np.zeros((T, STATE_VECTOR_DIM), dtype=np.float32)

        seeds = data["seeds"] if "seeds" in keys else np.zeros(len(episode_lengths), dtype=np.int64)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "truncated": truncated,
            "discounts": discounts,
            "events": events,
            "mask_before": mask_before,
            "inventory": inventory,
            "has_inventory": has_inventory,
            "episode_lengths": episode_lengths,
            "seeds": seeds,
        }

    @staticmethod
    def split_episodes(shard_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits flat shard arrays into individual episode dicts."""
        ep_lengths = shard_dict["episode_lengths"]
        episodes: List[Dict[str, Any]] = []

        start = 0
        for ep_len in ep_lengths:
            end = start + ep_len
            episodes.append({
                "obs": shard_dict["obs"][start:end],
                "actions": shard_dict["actions"][start:end],
                "rewards": shard_dict["rewards"][start:end],
                "dones": shard_dict["dones"][start:end],
                "truncated": shard_dict["truncated"][start:end],
                "discounts": shard_dict["discounts"][start:end],
                "events": shard_dict["events"][start:end],
                "mask_before": shard_dict["mask_before"][start:end],
                "inventory": shard_dict["inventory"][start:end],
                "has_inventory": shard_dict.get("has_inventory", True),
            })
            start = end
        return episodes
