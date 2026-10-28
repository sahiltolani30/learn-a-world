"""
Hermetic Gymnasium-compatible Crafter Environment Wrapper for Godfather JEPA.

Strict zero-touch isolation: does not import any legacy repository modules.
Provides deterministic execution via monkey-patching creature despawn & chunk order,
enforces Gymnasium (obs, reward, terminated, truncated, info) contract,
and guarantees strict pixel-only inference during policy execution.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Tuple, Union

import crafter
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..utils.seed_guard import assert_training_seed_pure


def patch_crafter_determinism() -> None:
    """
    Patches crafter internals to guarantee byte-identical determinism across runs.
    Sorts creature despawn and chunk iterations to remove CPython memory address dependencies.
    Idempotent: safe to invoke repeatedly across vectorized workers.
    """
    if getattr(crafter.env.Env, "_godfather_deterministic_patched", False):
        return

    # 1. Deterministic creature ordering in _balance_object
    def deterministic_balance_object(
        self: crafter.env.Env,
        chunk: Any,
        objs: Any,
        cls: Any,
        material: str,
        span_dist: float,
        despan_dist: float,
        spawn_prob: float,
        despawn_prob: float,
        ctor: Any,
        target_fn: Any,
    ) -> None:
        xmin, xmax, ymin, ymax = chunk
        random_gen = self._world.random
        creatures = sorted(
            [obj for obj in objs if isinstance(obj, cls)],
            key=lambda x: (int(x.pos[0]), int(x.pos[1]), getattr(x, "health", 0)),
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
    @property
    def deterministic_chunks(self: crafter.engine.World) -> Dict[Any, Any]:
        return {k: self._chunks[k] for k in sorted(self._chunks.keys())}

    crafter.engine.World.chunks = deterministic_chunks
    crafter.env.Env._godfather_deterministic_patched = True


# Apply determinism patch on module load
patch_crafter_determinism()


class GodfatherCrafterEnv(gym.Env):
    """
    Hermetic Gymnasium wrapper around crafter.Env.

    Observation space: Box(0, 255, (64, 64, 3), uint8)
    Action space: Discrete(17)
    Step returns: (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        seed: Optional[int] = None,
        render_size: int = 64,
        max_steps: int = 1000,
        training: bool = False,
    ) -> None:
        super().__init__()
        self.render_size = render_size
        self.max_steps = max_steps
        self.training = training
        self._seed = seed
        self._step_count = 0

        if self.training and self._seed is not None:
            assert_training_seed_pure(self._seed)

        self._env = crafter.Env(seed=seed, length=max_steps)
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            if self.training:
                assert_training_seed_pure(seed)
            self._seed = seed
            self._env._seed = seed
            self._env._episode = 0

        self._step_count = 0
        obs = self._env.reset()

        info: Dict[str, Any] = {
            "inventory": dict(self._env._player.inventory) if hasattr(self._env, "_player") else {},
            "achievements": {ach: 0 for ach in crafter.constants.achievements},
            "discount": 1.0,
            "reward": 0.0,
            "step": 0,
            "dead": False,
            "player_pos": list(self._env._player.pos) if hasattr(self._env, "_player") and hasattr(self._env._player, "pos") else None,
        }
        return obs, info

    def step(self, action: Union[int, np.integer]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise ValueError(f"Action {action!r} must be an integer, got {type(action).__name__}")

        act_int = int(action)
        if not (0 <= act_int < 17):
            raise ValueError(f"Action {action} out of bounds for Discrete(17)")

        self._step_count += 1
        obs, reward, done, raw_info = self._env.step(act_int)

        dead = bool(hasattr(self._env, "_player") and self._env._player.health <= 0)
        terminated = dead
        truncated = bool(self._step_count >= self.max_steps and not dead)

        info: Dict[str, Any] = {
            "inventory": dict(raw_info.get("inventory", {})),
            "achievements": dict(raw_info.get("achievements", {})),
            "discount": float(0.0 if dead else 1.0),
            "step": int(self._step_count),
            "dead": bool(dead),
            "player_pos": list(raw_info.get("player_pos", [])) if "player_pos" in raw_info else None,
            "reward": float(reward),
        }
        return obs, float(reward), terminated, truncated, info

    def snapshot_state(self) -> Dict[str, Any]:
        """Deepcopy state snapshot for deterministic branching and search."""
        return {
            "_env": copy.deepcopy(self._env),
            "_seed": self._seed,
            "_step_count": self._step_count,
            "np_random_state": np.random.get_state(),
            "py_random_state": random.getstate(),
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """Restores exact environment state and RNGs from snapshot."""
        self._env = copy.deepcopy(snapshot["_env"])
        self._seed = snapshot.get("_seed", self._seed)
        self._step_count = snapshot.get("_step_count", 0)
        np.random.set_state(snapshot["np_random_state"])
        random.setstate(snapshot["py_random_state"])

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self) -> None:
        pass


# Backward-compatible alias
CrafterEnvWrapper = GodfatherCrafterEnv

