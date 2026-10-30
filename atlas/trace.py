"""Decision tracing for Project ATLAS (T1 / Plan V4).

Exposes fine-grained step-level decisions, counterfactual scores, and mode reasoning.
Trace files are append-only JSON Lines: atlas_runs/<run_id>/trace/trace_it<iter>.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class TraceWriter:
    """Manages appending per-step decision records to disk with bounded folder size."""

    def __init__(self, run_dir: Union[str, Path], max_folder_mb: float = 500.0) -> None:
        self.run_dir = Path(run_dir)
        self.trace_dir = self.run_dir / "trace"
        self.max_bytes = int(max_folder_mb * 1024 * 1024)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.current_iter: int = 0
        self.current_ep_idx: int = 0
        self.current_seed: int = 0
        self.is_screen: bool = False

    def begin_episode(
        self,
        iter_idx: int,
        ep_idx: int,
        seed: int,
        screen: bool = False,
    ) -> None:
        """Opens trace file for appending at start of episode."""
        if self.current_file is not None and not self.current_file.closed:
            self.current_file.close()

        self.current_iter = iter_idx
        self.current_ep_idx = ep_idx
        self.current_seed = seed
        self.is_screen = screen or (iter_idx < 0)

        self._enforce_size_cap()
        filename = self.trace_dir / f"trace_it{iter_idx}.jsonl"
        self.current_file = open(filename, "a", encoding="utf-8")

    def write_step(self, agent: Any, step: int, reward_prev: float) -> None:
        """Writes one contractual JSON line. Reads ONLY agent attributes."""
        if self.current_file is None or self.current_file.closed:
            return

        # 1. Goal dictionary
        goal_dict = None
        active_goal = getattr(agent, "active_goal", None)
        if active_goal is not None:
            cos_val = round(float(getattr(agent, "last_arrival_cos", 0.0)), 4)
            psi_val = round(float(getattr(agent, "last_psi_val", 0.0)), 4)
            goal_dict = {
                "kind": str(getattr(active_goal, "kind", "event")),
                "dims": [int(d) for d in getattr(active_goal, "event_dims", [])],
                "cos": cos_val,
                "psi": psi_val,
                "budget": int(getattr(agent, "current_option_budget", 0)),
                "goal_mode": str(getattr(agent, "current_goal_mode", "none")),
                "n_try": int(getattr(active_goal, "n_try", 0)),
                "n_succ": int(getattr(active_goal, "n_succ", 0)),
            }

        # 2. Scores and sources
        scores_source = str(getattr(agent, "last_scores_source", "none"))
        scores_arr = getattr(agent, "last_scores_17", None)
        if scores_arr is not None:
            scores_list = [round(float(x), 4) for x in scores_arr]
        else:
            scores_list = [0.0] * 17

        # 3. Actions
        a_intended = int(getattr(agent, "last_a_intended", 0))
        a_final = int(getattr(agent, "last_a_final", 0))

        # 4. Exploration & Overrides
        epsilon_fired = bool(getattr(agent, "last_epsilon_fired", False))
        epsilon = round(float(getattr(agent, "epsilon", 0.0)), 4)
        beta_t = round(float(getattr(agent, "beta_t", 1.0)), 4)

        override_info = getattr(agent, "last_override_info", None)
        if override_info is None:
            override_info = {
                "fired": False,
                "risk_intended": 0.0,
                "min_risk": 0.0,
                "a_intended": a_intended,
                "a_final": a_final,
            }

        risks_arr = getattr(agent, "last_risks_17", None)
        if risks_arr is not None:
            risks_list = [round(float(x), 4) for x in risks_arr]
        else:
            risks_list = [0.0] * 17

        p_max_arr = getattr(agent, "last_event_p_max", None)
        if p_max_arr is not None:
            p_max_list = [round(float(x), 3) for x in p_max_arr]
        else:
            p_max_list = [0.0] * 22

        # 5. Belief and unlocks
        m_soft_arr = getattr(agent.belief, "m_soft", None)
        if m_soft_arr is not None:
            m_soft_list = [bool(x >= 0.5) for x in m_soft_arr]
        else:
            m_soft_list = [False] * 22

        newly_unlocked = [int(x) for x in getattr(agent, "probe_newly_unlocked", [])]
        probe_id = int(getattr(agent, "probe_id", 0))
        abort_reason = str(getattr(agent, "last_abort_reason", "none"))
        arrived = bool(getattr(agent, "last_arrived", False))

        # 6. Material gating & rewards
        suppressed_cnt = getattr(agent, "last_suppressed_count", None)
        material_gate = {"suppressed_count": int(suppressed_cnt)} if suppressed_cnt is not None else None

        r_prev = round(float(reward_prev), 4)
        r_int = round(float(getattr(agent, "last_r_int", 0.0)), 4)

        record: Dict[str, Any] = {
            "iter": self.current_iter,
            "ep_idx": self.current_ep_idx,
            "seed": self.current_seed,
            "t": step,
            "mode": getattr(agent, "mode", "NONE"),
            "prev_mode": getattr(agent, "prev_mode", "NONE"),
            "goal": goal_dict,
            "scores_source": scores_source,
            "scores": scores_list,
            "a_intended": a_intended,
            "a_final": a_final,
            "epsilon_fired": epsilon_fired,
            "epsilon": epsilon,
            "beta_t": beta_t,
            "override": override_info,
            "risks": risks_list,
            "event_p_max": p_max_list,
            "m_soft": m_soft_list,
            "newly_unlocked": newly_unlocked,
            "probe_id": probe_id,
            "abort_reason": abort_reason,
            "arrived": arrived,
            "material_gate": material_gate,
            "r_prev": r_prev,
            "r_int": r_int,
        }

        if self.is_screen:
            record["screen"] = True

        self.current_file.write(json.dumps(record) + "\n")

    def end_episode(self, length: int, total_reward: float, died: bool) -> None:
        """Closes file at end of episode."""
        if self.current_file is not None and not self.current_file.closed:
            self.current_file.flush()
            self.current_file.close()
            self.current_file = None

    def _enforce_size_cap(self) -> None:
        """Prunes oldest trace files when trace directory exceeds max_folder_mb."""
        files = sorted(self.trace_dir.glob("trace_it*.jsonl"), key=lambda p: p.stat().st_mtime)
        total_size = sum(f.stat().st_size for f in files)
        while total_size > self.max_bytes and files:
            oldest = files.pop(0)
            oldest_size = oldest.stat().st_size
            try:
                oldest.unlink()
                total_size -= oldest_size
            except OSError:
                pass
