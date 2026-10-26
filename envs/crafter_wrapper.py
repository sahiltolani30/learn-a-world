"""
Gymnasium-compatible wrapper for the Crafter environment.
Provides standard Gymnasium spaces, reset/step return signatures, and seeding.
"""

from typing import Any, Dict, Optional, Tuple
import copy
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import crafter


def patch_crafter_determinism() -> None:
    """
    Patches crafter environment internals to guarantee 100% byte-identical determinism
    across runs with the same seed and exact state snapshot restoration.

    Root-cause fixed:
    In upstream Crafter, crafter.env.Env._balance_chunk iterates over an unordered Python
    `set` of Object pointers without a custom __hash__. This makes creature despawn order
    depend on CPython memory addresses, causing separate runs and deepcopy snapshots to
    diverge after ~60 steps. Sorting creatures by grid coordinates (int(pos[0]), int(pos[1]))
    and chunk iteration by chunk coordinates ensures complete determinism invariant to heap layout.
    """
    if getattr(crafter.env.Env, "_deterministic_patched", False):
        return

    # 1. Deterministic creature ordering in _balance_object
    def deterministic_balance_object(
        self, chunk, objs, cls, material, span_dist, despan_dist,
        spawn_prob, despawn_prob, ctor, target_fn
    ):
        xmin, xmax, ymin, ymax = chunk
        random_gen = self._world.random
        creatures = sorted(
            [obj for obj in objs if isinstance(obj, cls)],
            key=lambda x: (int(x.pos[0]), int(x.pos[1]))
        )
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random_gen.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
            xs, ys = xs[mask], ys[mask]
            i = random_gen.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            empty = self._world[pos][1] is None
            away = self._player.distance(pos) >= span_dist
            if empty and away:
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random_gen.uniform() < despawn_prob:
            obj = creatures[random_gen.randint(0, len(creatures))]
            away = self._player.distance(obj.pos) >= despan_dist
            if away:
                self._world.remove(obj)

    crafter.env.Env._balance_object = deterministic_balance_object

    # 2. Deterministic chunk iteration order
    orig_chunks = crafter.engine.World.chunks

    @property
    def deterministic_chunks(self):
        return {k: self._chunks[k] for k in sorted(self._chunks.keys())}

    crafter.engine.World.chunks = deterministic_chunks
    crafter.env.Env._deterministic_patched = True


# Automatically apply deterministic patch on module load
patch_crafter_determinism()


class CrafterEnv(gym.Env):
    """
    Gymnasium wrapper around crafter.Env.
    Observation: Box(0, 255, (64, 64, 3), uint8)
    Action: Discrete(17)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        seed: Optional[int] = None,
        render_size: int = 64,
        max_steps: int = 1000,
        **kwargs: Any,
    ):
        super().__init__()
        self.render_size = render_size
        self.max_steps = max_steps
        self._seed = seed
        self._env = crafter.Env(seed=seed)
        self.action_space = spaces.Discrete(self._env.action_space.n)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._env = crafter.Env(seed=seed)
        obs = self._env.reset()
        info: Dict[str, Any] = {
            "inventory": dict(self._env._player.inventory) if hasattr(self._env, "_player") and hasattr(self._env._player, "inventory") else {},
            "achievements": {},
            "discount": 1.0,
            "reward": 0.0,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = False
        info_copy = {
            "inventory": dict(info.get("inventory", {})),
            "achievements": dict(info.get("achievements", {})),
            "discount": float(info.get("discount", 1.0)),
            "semantic": np.copy(info["semantic"]) if "semantic" in info and info["semantic"] is not None else None,
            "player_pos": list(info.get("player_pos", [])) if "player_pos" in info and info["player_pos"] is not None else None,
            "reward": float(reward),
        }
        return obs, float(reward), terminated, truncated, info_copy

    def snapshot_state(self) -> Dict[str, Any]:
        """
        Takes a full, deepcopy snapshot of the environment state and random number generators.
        Enables exact state restoration for archive restarts and tree search.
        """
        return {
            "_env": copy.deepcopy(self._env),
            "_seed": self._seed,
            "np_random_state": np.random.get_state(),
            "py_random_state": random.getstate(),
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """
        Restores environment state and RNGs from a snapshot.
        """
        self._env = copy.deepcopy(snapshot["_env"])
        self._seed = snapshot.get("_seed", self._seed)
        np.random.set_state(snapshot["np_random_state"])
        random.setstate(snapshot["py_random_state"])

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self):
        return None

