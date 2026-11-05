"""Vectorized Crafter environment for Zephyr data collection.

Seeding: each worker ``r`` owns an episode counter ``k`` that starts at 0 on every
explicit ``reset()`` and increments on each auto-reset; episode ``k`` in worker
``r`` runs with seed ``base_seed + 1000*k + r``. Workers therefore observe
disjoint, reproducible seed streams.

Auto-reset convention: when an episode ends (death or ``max_steps`` truncation)
the worker immediately starts the next episode and returns the NEW episode's
first observation with ``done[i]=True``. The info dict of that step describes
the episode that just ENDED: terminal inventory, cumulative achievements, its
``episode_seed`` and whether it ended ``truncated``.

Torch-free by design: this module never imports torch, and ``zephyr.config``
(which transitively imports torch via ``eval.metrics``) is imported lazily in
the parent process only. Spawn workers therefore re-import crafter + numpy
alone; the privileged ``zephyr_*`` info fields are attached parent-side.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.crafter_wrapper import CrafterEnv  # read-only whitelist import

__all__ = ["ZephyrVecEnv", "achievements_new_from_cum"]


def achievements_new_from_cum(
    prev_cum: Optional[np.ndarray], cur_cum: np.ndarray
) -> np.ndarray:
    """Boolean [22] of achievements newly unlocked between two cumulative vectors.

    ``prev_cum=None`` (first step of an episode) returns a copy of ``cur_cum``.
    """
    cur = np.asarray(cur_cum, dtype=bool)
    if prev_cum is None:
        return cur.copy()
    prev = np.asarray(prev_cum, dtype=bool)
    if prev.shape != cur.shape:
        raise ValueError(f"cumulative shape mismatch: {prev.shape} vs {cur.shape}")
    return cur & ~prev


class _EpisodeRunner:
    """One CrafterEnv plus its seed counter.

    Shared by the subprocess body and the in-process mode, so both modes have
    identical episode/seed/truncation semantics. Deals in RAW env data only —
    the privileged zephyr_* augmentation happens in the parent process.
    """

    def __init__(self, rank: int, base_seed: int, size: int, max_steps: int) -> None:
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.episode_index = 0
        self.steps_in_episode = 0
        self.current_seed = self._seed_for(0)
        self._env = CrafterEnv(
            seed=None, render_size=self.size, max_steps=self.max_steps
        )

    def _seed_for(self, episode_index: int) -> int:
        """Episode k in worker r uses seed base_seed + 1000*k + r."""
        return self.base_seed + 1000 * episode_index + self.rank

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any], int, bool]:
        """Restart the seed counter and begin episode 0.

        Returns (obs, raw_info, episode_seed, truncated).
        """
        self.episode_index = 0
        self.steps_in_episode = 0
        self.current_seed = self._seed_for(0)
        obs, info = self._env.reset(seed=self.current_seed)
        return obs, info, self.current_seed, False

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], int, Optional[Dict[str, Any]]]:
        """Step the env; auto-reset on done.

        Returns (obs, reward, done, truncated, raw_info, episode_seed, next_info)
        where, on done, obs is the NEW episode's first frame,
        raw_info/seed/truncated describe the episode that just ended, and
        next_info is the new episode's initial raw info (None when not done).
        """
        obs, reward, terminated, _, info = self._env.step(int(action))
        self.steps_in_episode += 1
        hit_limit = self.steps_in_episode >= self.max_steps
        done = bool(terminated) or hit_limit
        truncated = hit_limit  # death (or crafter's own length) is not truncation
        seed = self.current_seed
        next_info: Optional[Dict[str, Any]] = None
        if done:
            self.episode_index += 1
            self.steps_in_episode = 0
            self.current_seed = self._seed_for(self.episode_index)
            obs, next_info = self._env.reset(seed=self.current_seed)
        return obs, float(reward), done, truncated, info, seed, next_info


def _worker(pipe: Connection, rank: int, base_seed: int, size: int, max_steps: int) -> None:
    """Subprocess body: one _EpisodeRunner replying with raw env data.

    Imports at child start: zephyr.vec_env (torch-free at module scope) and
    envs.crafter_wrapper — crafter + numpy only, never torch or zephyr.config.
    """
    try:
        runner = _EpisodeRunner(rank, base_seed, size, max_steps)
        pipe.send(("ready", None))
        while True:
            command, payload = pipe.recv()
            if command == "reset":
                pipe.send(("ok", runner.reset()))
            elif command == "step":
                pipe.send(("ok", runner.step(int(payload))))
            elif command == "close":
                pipe.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command: {command!r}")
    except Exception:
        try:
            pipe.send(("err", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _augment_info(
    info: Dict[str, Any],
    episode_seed: int,
    truncated: bool,
    next_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach the PRIVILEGED zephyr_* diagnostic keys (parent process only).

    ``zephyr_inventory`` is the fixed 16-d vector from
    ``zephyr.config.inventory_dict_to_vector`` (training target only, never an
    inference input); ``zephyr_achievements_cum`` maps the env's cumulative
    achievements dict by name into ``zephyr.config.ACHIEVEMENTS`` order. When
    ``next_info`` is given (auto-reset steps), ``next_inventory`` and
    ``next_achievements_cum`` describe the NEW episode's starting state.
    Imports zephyr.config lazily so this module (and spawn workers) stay
    torch-free at import time.
    """
    from zephyr.config import ACHIEVEMENTS, NUM_ACHIEVEMENTS, inventory_dict_to_vector

    def _augment_one(raw: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        unlocked = (raw or {}).get("achievements") or {}
        cum = np.zeros(NUM_ACHIEVEMENTS, dtype=bool)
        for i, name in enumerate(ACHIEVEMENTS):
            if unlocked.get(name, 0):
                cum[i] = True
        inv = inventory_dict_to_vector((raw or {}).get("inventory"))
        return inv, cum

    out = dict(info)
    inv, cum = _augment_one(info)
    out["zephyr_inventory"] = inv
    out["zephyr_achievements_cum"] = cum
    out["episode_seed"] = int(episode_seed)
    out["truncated"] = bool(truncated)
    if next_info is not None:
        n_inv, n_cum = _augment_one(next_info)
        out["next_inventory"] = n_inv
        out["next_achievements_cum"] = n_cum
    return out


class ZephyrVecEnv:
    """Synchronous vectorized ``envs.crafter_wrapper.CrafterEnv``.

    Uses 'spawn' multiprocessing with one worker process per env; the worker
    function is module-level (picklable) and stays crafter + numpy only.
    ``in_process=True`` skips multiprocessing and runs the envs sequentially in
    this process instead (macOS/pytest smoke tests, num_envs=1).

    step() info dicts carry at least: 'zephyr_inventory' (float32 [16],
    privileged), 'zephyr_achievements_cum' (bool [22], cumulative episode
    unlocks in config.ACHIEVEMENTS order), 'episode_seed' (int) and
    'truncated' (bool). On done, they describe the ended episode while the
    returned obs is the new episode's first frame; 'next_inventory' and
    'next_achievements_cum' then describe the new episode's starting state.
    """

    def __init__(
        self,
        num_envs: int,
        base_seed: int = 7000,
        size: int = 64,
        max_steps: int = 1000,
        in_process: bool = False,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.in_process = bool(in_process)
        self._closed = False
        self._runners: List[_EpisodeRunner] = []
        self._pipes: List[Connection] = []
        self._procs: List[BaseProcess] = []
        if self.in_process:
            self._runners = [
                _EpisodeRunner(r, self.base_seed, self.size, self.max_steps)
                for r in range(self.num_envs)
            ]
        else:
            ctx = mp.get_context("spawn")
            for r in range(self.num_envs):
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(
                    target=_worker,
                    args=(child_conn, r, self.base_seed, self.size, self.max_steps),
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                self._pipes.append(parent_conn)
                self._procs.append(proc)
            for r in range(self.num_envs):
                self._recv(r)  # ("ready", None); raises early if a worker died

    # -- public API ----------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all workers (seed counters restart at episode 0).

        Returns obs uint8 [N, size, size, 3] and per-env info dicts.
        """
        self._check_open()
        if self.in_process:
            raw = [runner.reset() for runner in self._runners]
        else:
            for pipe in self._pipes:
                pipe.send(("reset", None))
            raw = [self._recv(r) for r in range(self.num_envs)]
        infos = [_augment_info(info, seed, trunc, next_info=None)
                 for (_, info, seed, trunc) in raw]
        return self._stack_obs([obs for (obs, _, _, _) in raw]), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all envs with int actions [N] in 0..NUM_ACTIONS-1.

        Returns (obs uint8 [N,size,size,3], rewards float32 [N], done bool [N],
        info dicts). On done[i], obs[i] is the new episode's first frame and
        infos[i] describes the episode that just ended (auto-reset).
        """
        self._check_open()
        from zephyr.config import NUM_ACTIONS

        acts = np.asarray(actions, dtype=np.int64)
        if acts.shape != (self.num_envs,):
            raise ValueError(f"actions shape {acts.shape} != ({self.num_envs},)")
        if acts.size and (int(acts.min()) < 0 or int(acts.max()) >= NUM_ACTIONS):
            raise ValueError(f"actions must be in 0..{NUM_ACTIONS - 1}, got {acts}")
        if self.in_process:
            raw = [
                runner.step(int(action)) for runner, action in zip(self._runners, acts)
            ]
        else:
            for rank, action in enumerate(acts):
                self._pipes[rank].send(("step", int(action)))
            raw = [self._recv(r) for r in range(self.num_envs)]
        obs = self._stack_obs([entry[0] for entry in raw])
        rewards = np.asarray([entry[1] for entry in raw], dtype=np.float32)
        dones = np.asarray([entry[2] for entry in raw], dtype=np.bool_)
        infos = [
            _augment_info(entry[4], entry[5], entry[3], next_info=entry[6])
            for entry in raw
        ]
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Terminate workers cleanly; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self.in_process:
            self._runners = []
            return
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except (OSError, ValueError):
                pass
        for proc in self._procs:
            proc.join(timeout=5.0)
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs:
            proc.join(timeout=5.0)
        for pipe in self._pipes:
            try:
                pipe.close()
            except (OSError, ValueError):
                pass
        self._pipes = []
        self._procs = []

    # -- internals -----------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("ZephyrVecEnv is closed")

    def _recv(self, rank: int) -> Any:
        """Receive one worker reply; surface worker errors and deaths."""
        try:
            message, payload = self._pipes[rank].recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"zephyr worker {rank} died before replying (spawn child crash?)"
            ) from exc
        if message == "err":
            raise RuntimeError(f"zephyr worker {rank} raised:\n{payload}")
        return payload

    def _stack_obs(self, obs_list: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack([np.asarray(obs) for obs in obs_list], axis=0)
        if stacked.dtype != np.uint8:
            raise TypeError(f"expected uint8 observations, got {stacked.dtype}")
        expected = (self.num_envs, self.size, self.size, 3)
        if stacked.shape != expected:
            raise ValueError(f"obs shape {stacked.shape} != {expected}")
        return stacked
