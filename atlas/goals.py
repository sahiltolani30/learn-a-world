"""GoalBank + PredecessorLift: visual goal mining, scoring, and temporal dependency priors.

Mined strictly from the agent's own play:
1. event: keyframe (z_{t-1}, h_{t-1}) right before newly_unlocked fired
2. vital: keyframe (z_t, h_t) on health_head spike (Δr_vital >= 0.15) with no unlock
3. frontier: keyframe with top epistemic disagreement

Predecessor lift mines P(d before e | e) / P(d) over 22 event dimensions ONLY.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from atlas.config import (
    LATENT_DIM,
    NUM_ACHIEVEMENTS,
    NUM_ACTIONS,
    RECURRENT_DIM,
    GoalsConfig,
)


def _unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit_vector(a), _unit_vector(b)))


@dataclass
class GoalEntry:
    key: np.ndarray                    # shape (128,) visual key (z)
    context: np.ndarray                # shape (256,) belief (h)
    action_profile: np.ndarray         # shape (17, 22) calibrated event probs p(a, i)
    event_dims: List[int]              # unlocked event dimensions near this keyframe
    kind: str                          # "event" | "vital" | "frontier"
    n_try: int = 0
    n_succ: int = 0
    born_iter: int = 0
    last_used_iter: int = 0

    def __post_init__(self):
        self.key = np.asarray(self.key, dtype=np.float32).reshape(-1)
        self.context = np.asarray(self.context, dtype=np.float32).reshape(-1)
        self.action_profile = np.asarray(self.action_profile, dtype=np.float32).reshape(NUM_ACTIONS, NUM_ACHIEVEMENTS)
        self.event_dims = [int(x) for x in self.event_dims]
        self.kind = str(self.kind)


class PredecessorLift:
    """Mines temporal prerequisite lift P(d in window before e | e) / P(d)."""

    def __init__(self):
        self.lift: np.ndarray = np.ones((NUM_ACHIEVEMENTS, NUM_ACHIEVEMENTS), dtype=np.float32)
        self.ci_low: np.ndarray = np.zeros((NUM_ACHIEVEMENTS, NUM_ACHIEVEMENTS), dtype=np.float32)

    def mine(
        self,
        episodes_events: Sequence[np.ndarray],
        window: int = 30,
    ) -> None:
        """Computes empirical lift and Wilson CI lower bounds across episodes.

        Args:
            episodes_events: list of boolean event arrays, each shape [T, 22].
            window: temporal window in steps.
        """
        if not episodes_events:
            return

        total_steps = 0
        total_d_counts = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float64)
        target_counts = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float64)
        pair_counts = np.zeros((NUM_ACHIEVEMENTS, NUM_ACHIEVEMENTS), dtype=np.float64)

        for ep in episodes_events:
            ep_arr = np.asarray(ep, dtype=bool)
            T = ep_arr.shape[0]
            total_steps += T
            total_d_counts += np.sum(ep_arr, axis=0)

            # Find all firing timesteps per achievement
            fire_steps: Dict[int, List[int]] = {
                i: np.where(ep_arr[:, i])[0].tolist() for i in range(NUM_ACHIEVEMENTS)
            }

            for e in range(NUM_ACHIEVEMENTS):
                e_timesteps = fire_steps[e]
                target_counts[e] += len(e_timesteps)
                for t_e in e_timesteps:
                    t_start = max(0, t_e - window)
                    for d in range(NUM_ACHIEVEMENTS):
                        if d == e:
                            continue
                        if np.any(ep_arr[t_start:t_e, d]):
                            pair_counts[d, e] += 1

        if total_steps == 0:
            return

        base_p_d = np.maximum(total_d_counts / total_steps, 1e-4)

        z_crit = 1.96  # 95% Wilson score
        for e in range(NUM_ACHIEVEMENTS):
            n_e = target_counts[e]
            if n_e < 1:
                self.lift[:, e] = 1.0
                self.ci_low[:, e] = 0.0
                continue

            for d in range(NUM_ACHIEVEMENTS):
                if d == e:
                    self.lift[d, e] = 1.0
                    self.ci_low[d, e] = 0.0
                    continue

                k = pair_counts[d, e]
                p_cond = k / n_e
                self.lift[d, e] = float(np.clip(p_cond / base_p_d[d], 0.0, 50.0))

                # Wilson score interval lower bound
                denom = 1.0 + (z_crit ** 2) / n_e
                center = (p_cond + (z_crit ** 2) / (2.0 * n_e)) / denom
                margin = (
                    z_crit
                    * np.sqrt((p_cond * (1.0 - p_cond) + (z_crit ** 2) / (4.0 * n_e)) / n_e)
                    / denom
                )
                self.ci_low[d, e] = float(max(0.0, (center - margin) / base_p_d[d]))

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, lift=self.lift, ci_low=self.ci_low)

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = np.load(p)
        self.lift = data["lift"]
        self.ci_low = data["ci_low"]


class GoalBank:
    """Stores mined sub-goal keyframes and selects high-utility navigation targets."""

    def __init__(self, cfg: Optional[GoalsConfig] = None):
        self.cfg = cfg or GoalsConfig()
        self.entries: List[GoalEntry] = []

    def add(self, entry: GoalEntry) -> GoalEntry:
        """Adds a new goal entry. Merges into existing if cosine > 0.97."""
        entry_key = _unit_vector(entry.key)

        for existing in self.entries:
            sim = float(np.dot(entry_key, _unit_vector(existing.key)))
            if sim > 0.97:
                # Merge stats
                existing.n_try += entry.n_try
                existing.n_succ += entry.n_succ
                existing.last_used_iter = max(existing.last_used_iter, entry.last_used_iter)
                for dim in entry.event_dims:
                    if dim not in existing.event_dims:
                        existing.event_dims.append(dim)
                existing.action_profile = np.maximum(existing.action_profile, entry.action_profile)
                return existing

        if len(self.entries) >= self.cfg.max_goals:
            # Evict least used / lowest success entry
            self.entries.sort(key=lambda x: (x.n_succ / max(1, x.n_try), x.born_iter))
            self.entries.pop(0)

        self.entries.append(entry)
        return entry

    def mark_used(self, key: np.ndarray, success: bool, current_iter: int = 0) -> None:
        """Updates visit and success counts for the matched goal."""
        target = _unit_vector(key)
        best_sim = -1.0
        best_entry: Optional[GoalEntry] = None

        for entry in self.entries:
            sim = float(np.dot(target, _unit_vector(entry.key)))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= 0.93:
            best_entry.n_try += 1
            if success:
                best_entry.n_succ += 1
            best_entry.last_used_iter = current_iter

    def select(
        self,
        z: np.ndarray,
        h: np.ndarray,
        m_soft: np.ndarray,
        psi_fn: Callable[[np.ndarray, np.ndarray], float],
        danger_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        novelty_fn: Optional[Callable[[np.ndarray], float]] = None,
        lift: Optional[PredecessorLift] = None,
        current_iter: int = 0,
        beta_t: float = 1.0,
    ) -> Tuple[Optional[GoalEntry], float]:
        """Selects the best goal entry by composite utility score (§5.8)."""
        if not self.entries:
            return None, 0.0

        s = np.concatenate([z.reshape(-1), h.reshape(-1)], axis=-1)
        m_soft = np.asarray(m_soft, dtype=np.float32).reshape(-1)
        locked_dims = [i for i in range(NUM_ACHIEVEMENTS) if m_soft[i] < 0.5]

        best_score = -float("inf")
        best_goal: Optional[GoalEntry] = None

        for g in self.entries:
            # 1) Unlock potential
            if locked_dims:
                locked_profiles = g.action_profile[:, locked_dims]
                max_unlock = float(np.max(locked_profiles))
                # argmax achievement index among locked
                best_flat_idx = int(np.argmax(locked_profiles))
                best_locked_dim = locked_dims[best_flat_idx % len(locked_dims)]

                # Predecessor prior lift factor
                if lift is not None:
                    lift_sum = float(sum(lift.lift[d, best_locked_dim] for d in locked_dims if d != best_locked_dim))
                    predecessor_factor = 1.0 + 0.25 * min(lift_sum, 4.0)
                else:
                    predecessor_factor = 1.0
                score_unlock = self.cfg.w_unlock * max_unlock * predecessor_factor
            else:
                score_unlock = 0.0

            # 2) Reachability psi(s, g.key)
            psi_val = float(psi_fn(s, g.key))
            score_reach = self.cfg.w_reach * psi_val

            # 3) Exploration curiosity
            is_frontier = 1.0 if g.kind == "frontier" else 0.0
            score_explor = self.cfg.w_explor * is_frontier * beta_t

            # 4) Recency bonus
            age = max(0, current_iter - g.born_iter)
            score_recency = 0.1 * (1.0 / (1.0 + 0.1 * age))

            # 5) Danger penalty
            danger = float(danger_fn(s, g.key)) if danger_fn is not None else 0.0
            score_danger = self.cfg.w_danger * danger

            total_score = score_unlock + score_reach + score_explor + score_recency - score_danger
            if total_score > best_score:
                best_score = total_score
                best_goal = g

        return best_goal, best_score

    def save(self, path: Union[str, Path]) -> None:
        """Saves goal bank to npz + json metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not self.entries:
            keys = np.zeros((0, LATENT_DIM), dtype=np.float32)
            contexts = np.zeros((0, RECURRENT_DIM), dtype=np.float32)
            profiles = np.zeros((0, NUM_ACTIONS, NUM_ACHIEVEMENTS), dtype=np.float32)
        else:
            keys = np.stack([g.key for g in self.entries], axis=0)
            contexts = np.stack([g.context for g in self.entries], axis=0)
            profiles = np.stack([g.action_profile for g in self.entries], axis=0)

        meta = [
            {
                "event_dims": g.event_dims,
                "kind": g.kind,
                "n_try": g.n_try,
                "n_succ": g.n_succ,
                "born_iter": g.born_iter,
                "last_used_iter": g.last_used_iter,
            }
            for g in self.entries
        ]

        np.savez_compressed(str(p) + ".npz", keys=keys, contexts=contexts, profiles=profiles)
        with open(str(p) + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Loads goal bank from npz + json metadata."""
        npz_p = Path(str(path) + ".npz")
        json_p = Path(str(path) + ".json")
        if not npz_p.exists() or not json_p.exists():
            return

        data = np.load(npz_p)
        keys = data["keys"]
        contexts = data["contexts"]
        profiles = data["profiles"]
        with open(json_p, "r") as f:
            meta = json.load(f)

        self.entries.clear()
        for i in range(len(meta)):
            m = meta[i]
            entry = GoalEntry(
                key=keys[i],
                context=contexts[i],
                action_profile=profiles[i],
                event_dims=m["event_dims"],
                kind=m["kind"],
                n_try=m["n_try"],
                n_succ=m["n_succ"],
                born_iter=m["born_iter"],
                last_used_iter=m["last_used_iter"],
            )
            self.entries.append(entry)
