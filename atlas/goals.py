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


@dataclass(eq=False)
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
        norm = float(np.linalg.norm(self.key))
        self.key = self.key / max(norm, 1e-8)
        self.context = np.asarray(self.context, dtype=np.float32).reshape(-1)
        self.action_profile = np.asarray(self.action_profile, dtype=np.float32).reshape(NUM_ACTIONS, NUM_ACHIEVEMENTS)
        self.event_dims = [int(x) for x in self.event_dims]
        self.kind = str(self.kind)


class PredecessorLift:
    """Mines temporal prerequisite lift P(d in window before e | e) / P(d)."""

    def __init__(self, cfg: Optional[GoalsConfig] = None):
        self.cfg = cfg or GoalsConfig()
        self.lift: np.ndarray = np.ones((NUM_ACHIEVEMENTS, NUM_ACHIEVEMENTS), dtype=np.float32)
        self.ci_low: np.ndarray = np.zeros((NUM_ACHIEVEMENTS, NUM_ACHIEVEMENTS), dtype=np.float32)

    def assert_consistent(self) -> None:
        """Verifies that ci_low <= lift * 1.01 + 1e-6 elementwise."""
        inconsistent = self.ci_low > self.lift * 1.01 + 1e-6
        if np.any(inconsistent):
            max_viol = float(np.max(self.ci_low - self.lift * 1.01))
            raise RuntimeError(f"PredecessorLift inconsistent: ci_low > lift (max violation {max_viol}) — this is a bug, fix it")

    def mine(
        self,
        episodes_events: Sequence[np.ndarray],
        window: int = 30,
        cfg: Optional[GoalsConfig] = None,
    ) -> None:
        """Computes empirical lift and Wilson CI lower bounds across episodes.

        Args:
            episodes_events: list of boolean event arrays, each shape [T, 22].
            window: temporal window in steps.
            cfg: optional GoalsConfig overriding default.
        """
        if not episodes_events:
            self.assert_consistent()
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
            self.assert_consistent()
            return

        active_cfg = cfg or self.cfg
        lift_cap = active_cfg.lift_cap
        cfg_lift_min_base_count = active_cfg.lift_min_base_count
        cfg_lift_min_pair_support = active_cfg.lift_min_pair_support
        z_crit = 1.96  # 95% Wilson score

        for e in range(NUM_ACHIEVEMENTS):
            n_e = target_counts[e]
            for d in range(NUM_ACHIEVEMENTS):
                if d == e:
                    self.lift[d, e] = 1.0
                    self.ci_low[d, e] = 0.0
                    continue

                count_d = total_d_counts[d]
                k = pair_counts[d, e]
                p_d = count_d / total_steps

                if count_d < cfg_lift_min_base_count or k < cfg_lift_min_pair_support or n_e < 1:
                    self.lift[d, e] = 1.0
                    self.ci_low[d, e] = 0.0
                else:
                    if p_d <= 0.0:
                        self.lift[d, e] = 1.0
                        self.ci_low[d, e] = 0.0
                        continue
                    p_cond = k / n_e
                    self.lift[d, e] = float(min(lift_cap, p_cond / p_d))
                    denom = 1.0 + (z_crit ** 2) / n_e
                    center = (p_cond + (z_crit ** 2) / (2.0 * n_e)) / denom
                    margin = (
                        z_crit
                        * np.sqrt(p_cond * (1.0 - p_cond) / n_e + (z_crit ** 2) / (4.0 * (n_e ** 2)))
                        / denom
                    )
                    self.ci_low[d, e] = float(min(self.lift[d, e], max(0.0, (center - margin) / p_d)))
                    if self.ci_low[d, e] > self.lift[d, e] * 1.01 + 1e-6:
                        raise RuntimeError("PredecessorLift inconsistent: ci_low > lift — this is a bug, fix it")

        self.assert_consistent()

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
        self.rejected_same_kind: int = 0
        self.rejected_frontier_overflow: int = 0
        self._dim_counts: Dict[int, int] = {i: 0 for i in range(NUM_ACHIEVEMENTS)}
        self.last_selection_mode: str = "none"

    def _rebuild_dim_counts(self) -> None:
        self._dim_counts = {i: 0 for i in range(NUM_ACHIEVEMENTS)}
        for g in self.entries:
            if g.kind == "event":
                for d in g.event_dims:
                    self._dim_counts[d] = self._dim_counts.get(d, 0) + 1

    def add(self, entry: GoalEntry, current_iter: Optional[int] = None) -> Optional[GoalEntry]:
        """Adds a new goal entry with deduplication, per-dim caps, and frontier quotas."""
        entry.key = _unit_vector(entry.key)

        # Reject if any existing goal of the SAME kind has cosine > same_kind_dedupe (0.90)
        for existing in self.entries:
            if existing.kind == entry.kind:
                sim = float(np.dot(entry.key, existing.key))
                if sim > self.cfg.same_kind_dedupe:
                    self.rejected_same_kind += 1
                    return None

        # Count per-dim usage over "event" goals
        if entry.kind == "event":
            for d in entry.event_dims:
                if self._dim_counts.get(d, 0) + 1 > self.cfg.max_goals_per_dim:
                    return None

        # Enforce frontier quota: if kind == "event" and current frontier share < 0.20 and bank size > 50,
        # reject event goals until frontier share restored (frontier goals always accepted).
        bank_size = len(self.entries)
        if entry.kind == "event" and bank_size > 50:
            frontier_count = sum(1 for g in self.entries if g.kind == "frontier")
            if (frontier_count / bank_size) < self.cfg.frontier_min_share:
                return None

        # Eviction when full (bank_size >= max_goals): balanced eviction
        cur_iter = current_iter if current_iter is not None else entry.born_iter
        if bank_size >= self.cfg.max_goals:
            # 1. Compute current shares over the bank
            n_frontier = sum(1 for g in self.entries if g.kind == "frontier")
            n_vital = sum(1 for g in self.entries if g.kind == "vital")
            n_event = sum(1 for g in self.entries if g.kind == "event")
            s_frontier = n_frontier / bank_size
            s_vital = n_vital / bank_size
            s_event = n_event / bank_size

            t_frontier = self.cfg.frontier_target_share
            t_vital = self.cfg.vital_target_share
            t_event = max(0.0, 1.0 - t_frontier - t_vital)

            targets = {"frontier": t_frontier, "vital": t_vital, "event": t_event}
            shares = {"frontier": s_frontier, "vital": s_vital, "event": s_event}

            # 2. Over-represented kinds = those with share > target + 0.05
            over_rep = [k for k in ("frontier", "vital", "event") if shares[k] > targets[k] + 0.05]

            # 3. If any over-represented: lowest-evidence among MOST over-represented kind
            if over_rep:
                most_over_kind = max(over_rep, key=lambda k: shares[k] - targets[k])
                candidates = [g for g in self.entries if g.kind == most_over_kind]
            # 4. If none over-represented: lowest-evidence among event ∪ vital
            else:
                candidates = [g for g in self.entries if g.kind in ("event", "vital")]
                if not candidates:
                    candidates = list(self.entries)

            # 5. NEVER evict an event/vital entry to admit a frontier goal
            if entry.kind == "frontier":
                candidate_kinds = set(g.kind for g in candidates)
                if candidate_kinds.issubset({"event", "vital"}):
                    self.rejected_frontier_overflow += 1
                    return None

            def evidence_fn(g: GoalEntry) -> float:
                age = max(0, cur_iter - g.born_iter)
                return (g.n_succ / max(1, g.n_try)) * (1.0 / (1.0 + 0.1 * age))

            evict_target = min(candidates, key=evidence_fn)
            self.entries = [g for g in self.entries if g is not evict_target]
            if evict_target.kind == "event":
                for d in evict_target.event_dims:
                    self._dim_counts[d] = max(0, self._dim_counts.get(d, 0) - 1)

        self.entries.append(entry)
        if entry.kind == "event":
            for d in entry.event_dims:
                self._dim_counts[d] = self._dim_counts.get(d, 0) + 1

        return entry

    def mark_used(self, key: np.ndarray, success: bool, current_iter: int = 0) -> bool:
        """Updates visit and success counts for the matched goal; returns True if matched cos >= 0.93."""
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
            return True
        return False

    def _compute_goal_score(
        self,
        g: GoalEntry,
        s: np.ndarray,
        locked_dims: List[int],
        psi_fn: Callable[[np.ndarray, np.ndarray], float],
        danger_fn: Optional[Callable[[np.ndarray, np.ndarray], float]],
        novelty_fn: Optional[Callable[[np.ndarray], float]],
        lift: Optional[PredecessorLift],
        current_iter: int,
        beta_t: float,
    ) -> float:
        # 1) Unlock potential
        if locked_dims:
            locked_profiles = g.action_profile[:, locked_dims]
            max_unlock = float(np.max(locked_profiles))
            best_flat_idx = int(np.argmax(locked_profiles))
            best_locked_dim = locked_dims[best_flat_idx % len(locked_dims)]

            if lift is not None:
                pred_bonus = 0.25 * min(
                    sum(float(lift.ci_low[d, best_locked_dim]) for d in locked_dims if lift.ci_low[d, best_locked_dim] > 1.2),
                    4.0,
                )
                predecessor_factor = 1.0 + pred_bonus
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

        return float(score_unlock + score_reach + score_explor + score_recency - score_danger)

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
        """Selects a goal entry with statistical rotation to break selection monopoly (V2.2 F3)."""
        if not self.entries:
            self.last_selection_mode = "none"
            return None, 0.0

        s = np.concatenate([z.reshape(-1), h.reshape(-1)], axis=-1)
        m_soft = np.asarray(m_soft, dtype=np.float32).reshape(-1)
        locked_dims = [i for i in range(NUM_ACHIEVEMENTS) if m_soft[i] < 0.5]

        frontier_goals = [g for g in self.entries if g.kind == "frontier"]
        u = float(np.random.rand())

        frontier_share = getattr(self.cfg, "selection_frontier_share", 0.10)
        rotate_share = getattr(self.cfg, "selection_rotate_share", 0.25)

        if u < frontier_share and len(frontier_goals) > 0:
            mode = "frontier"
            chosen_g = min(frontier_goals, key=lambda g: (g.last_used_iter, g.n_try))
            chosen_score = self._compute_goal_score(
                chosen_g, s, locked_dims, psi_fn, danger_fn, novelty_fn, lift, current_iter, beta_t
            )
        elif u < (frontier_share + rotate_share):
            event_vital = [g for g in self.entries if g.kind in ("event", "vital")]
            if len(event_vital) > 0:
                mode = "rotate"
                scored_candidates = [
                    (
                        g,
                        self._compute_goal_score(
                            g, s, locked_dims, psi_fn, danger_fn, novelty_fn, lift, current_iter, beta_t
                        ),
                    )
                    for g in event_vital
                ]
                scored_candidates.sort(key=lambda item: item[1], reverse=True)
                top10 = scored_candidates[:10]
                chosen_g, chosen_score = min(top10, key=lambda item: (item[0].last_used_iter, item[0].n_try))
            else:
                mode = "argmax"
                scored_all = [
                    (
                        g,
                        self._compute_goal_score(
                            g, s, locked_dims, psi_fn, danger_fn, novelty_fn, lift, current_iter, beta_t
                        ),
                    )
                    for g in self.entries
                ]
                chosen_g, chosen_score = max(scored_all, key=lambda item: item[1])
        else:
            mode = "argmax"
            scored_all = [
                (
                    g,
                    self._compute_goal_score(
                        g, s, locked_dims, psi_fn, danger_fn, novelty_fn, lift, current_iter, beta_t
                    ),
                )
                for g in self.entries
            ]
            chosen_g, chosen_score = max(scored_all, key=lambda item: item[1])

        self.last_selection_mode = mode
        return chosen_g, chosen_score

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
        self._rebuild_dim_counts()
