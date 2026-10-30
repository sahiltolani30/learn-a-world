"""Runtime HealthMonitor for ATLAS: early warning alarms, auto-intervention, and report cards."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from atlas.config import HealthConfig, NUM_ACHIEVEMENTS, NUM_ACTIONS


class HealthMonitor:
    def __init__(self, cfg: Optional[HealthConfig] = None):
        self.cfg = cfg or HealthConfig()
        self.action_window: deque[int] = deque(maxlen=self.cfg.window_steps)
        self.recent_confidence: deque[float] = deque(maxlen=50)
        self.progress_history: List[tuple[int, float]] = []  # (iter_idx, ach_per_ep)
        self.alarms_so_far: List[Dict[str, Any]] = []

        self.last_best_iter: int = 0
        self.best_ach_per_ep: float = 0.0
        self.probe_fraction: float = 0.0

        self.boost_steps_left: int = 0
        self.epsilon_boost_active: bool = False

    def observe_action(self, action_idx: int) -> None:
        """Push an executed action into the recent window."""
        self.action_window.append(int(action_idx))

    def observe_confidence(self, mean_max_prob: float) -> None:
        """Push OptionNet validation confidence from val HER batch."""
        self.recent_confidence.append(float(mean_max_prob))

    def observe_progress(self, iter_idx: int, ach_per_ep: float) -> None:
        """Push dev screen achievements per episode."""
        ach = float(ach_per_ep)
        self.progress_history.append((int(iter_idx), ach))
        if len(self.progress_history) == 1:
            self.best_ach_per_ep = ach
            self.last_best_iter = int(iter_idx)
        else:
            if ach >= self.best_ach_per_ep * (1.0 + self.cfg.flatline_improvement) and ach > self.best_ach_per_ep:
                self.best_ach_per_ep = ach
                self.last_best_iter = int(iter_idx)

    def observe_probe_fraction(self, frac: float) -> None:
        """Passive metric updated from dev screen or train log."""
        self.probe_fraction = float(frac)

    def get_snapshot(self) -> Dict[str, Any]:
        """Captures health state snapshot for resume integrity (V2.2 F1)."""
        return {
            "best_ach_per_ep": float(self.best_ach_per_ep),
            "last_best_iter": int(self.last_best_iter),
            "progress_history": list(self.progress_history),
            "alarms_so_far": list(self.alarms_so_far),
        }

    def load_snapshot(self, snapshot: Optional[Dict[str, Any]], resume_iter: Optional[int] = None) -> None:
        """Restores health state snapshot on resume (V2.2 F1 / V3 C3)."""
        if not snapshot:
            return
        self.best_ach_per_ep = float(snapshot.get("best_ach_per_ep", 0.0))
        self.last_best_iter = int(snapshot.get("last_best_iter", 0))
        self.progress_history = [(int(it), float(val)) for it, val in snapshot.get("progress_history", [])]
        self.alarms_so_far = list(snapshot.get("alarms_so_far", []))

        # V3 C3: initialize monitor from latest dev-screen ach/ep if best_ach_per_ep == 0.0 but iter > 10
        cur_it = resume_iter if resume_iter is not None else (self.progress_history[-1][0] if self.progress_history else 0)
        if self.best_ach_per_ep == 0.0 and cur_it > 10:
            if self.progress_history:
                best_score = max(item[1] for item in self.progress_history)
                latest_best = max(item[0] for item in self.progress_history if item[1] >= best_score - 1e-6)
                self.best_ach_per_ep = float(best_score)
                self.last_best_iter = int(latest_best)
            else:
                self.last_best_iter = int(cur_it)

    def notify_step(self) -> None:
        """Called on each environment step during collection to decrement boost_steps_left."""
        if self.boost_steps_left > 0:
            self.boost_steps_left -= 1
            if self.boost_steps_left == 0:
                self.epsilon_boost_active = False

    def check(self, iter_idx: int, log_file: Optional[Path | str] = None) -> Dict[str, Any]:
        """Evaluates health conditions and triggers interventions."""
        active_alarms: List[str] = []
        beta_reset = False

        # Alarm A1: Action diversity
        # Top action share over window > action_share_alarm (0.60)
        top_action_share = 0.0
        if len(self.action_window) >= self.cfg.window_steps:
            actions_arr = np.array(self.action_window, dtype=np.int64)
            bincount = np.bincount(actions_arr, minlength=NUM_ACTIONS)
            top_action_share = float(np.max(bincount) / len(actions_arr))
            if top_action_share > self.cfg.action_share_alarm:
                active_alarms.append(
                    f"A1_ACTION_DIVERSITY: Top action share {top_action_share:.3f} > {self.cfg.action_share_alarm:.2f}"
                )
                self.epsilon_boost_active = True
                self.boost_steps_left = self.cfg.boost_steps

        # Alarm A2: OptionNet overconfidence
        # mean_max_prob > confidence_alarm (0.85)
        latest_conf = self.recent_confidence[-1] if self.recent_confidence else 0.0
        if latest_conf > self.cfg.confidence_alarm:
            active_alarms.append(
                f"A2_CONFIDENCE: OptionNet confidence {latest_conf:.3f} > {self.cfg.confidence_alarm:.2f}"
            )

        # Alarm A3: Progress flatline in iterations
        # best ach_per_ep not improved within last flatline_iters (15) iterations
        if len(self.progress_history) > 0 and (iter_idx - self.last_best_iter >= self.cfg.flatline_iters):
            active_alarms.append(
                f"A3_FLATLINE: Progress {self.best_ach_per_ep:.3f} flatlined for {iter_idx - self.last_best_iter} iters (>= {self.cfg.flatline_iters})"
            )
            beta_reset = True
            self.last_best_iter = iter_idx

        for alarm in active_alarms:
            alarm_record = {"iter": iter_idx, "alarm": alarm}
            self.alarms_so_far.append(alarm_record)
            print(f"!!! [HEALTH ALARM] {alarm}")
            if log_file:
                try:
                    with open(log_file, "a") as f:
                        f.write(json.dumps(alarm_record) + "\n")
                except Exception:
                    pass

        return {
            "alarms": active_alarms,
            "epsilon_boost_active": self.epsilon_boost_active,
            "boost_steps_left": self.boost_steps_left,
            "beta_reset": beta_reset,
            "top_action_share": top_action_share,
            "latest_confidence": latest_conf,
            "probe_fraction": self.probe_fraction,
        }

    def generate_report_card(
        self,
        iter_idx: int,
        agent: Any,
        val_her_batch: Optional[Dict[str, Any]] = None,
        out_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Generates report card dictionary and writes to results/atlas_health_it<N>.json."""
        # 1. Top action share
        if len(self.action_window) > 0:
            actions_arr = np.array(self.action_window, dtype=np.int64)
            bincount = np.bincount(actions_arr, minlength=NUM_ACTIONS)
            top_action_share = float(np.max(bincount) / len(actions_arr))
        else:
            top_action_share = 0.0

        # 2. OptionNet confidence on val HER positives
        if val_her_batch is not None and "s_pos" in val_her_batch and val_her_batch["s_pos"].shape[0] > 0:
            import torch
            with torch.no_grad():
                logits = agent.option_net.logits(val_her_batch["s_pos"], val_her_batch["g_pos"])
                probs = torch.softmax(logits, dim=-1)
                max_p = torch.max(probs, dim=-1).values
                optionnet_confidence = float(max_p.mean().item())
        elif self.recent_confidence:
            optionnet_confidence = float(self.recent_confidence[-1])
        else:
            optionnet_confidence = 0.0

        # 3. GoalBank variety
        bank_size = len(agent.goals.entries)
        per_dim_counts = {str(d): agent.goals._dim_counts.get(d, 0) for d in range(NUM_ACHIEVEMENTS)}
        if bank_size > 0:
            largest_dim_share = float(max(per_dim_counts.values()) / bank_size)
        else:
            largest_dim_share = 0.0

        # 4. Progress trend vs fixed iters 10-15 baseline window (V3 C4)
        base_scores = [score for it_num, score in self.progress_history if 10 <= it_num <= 15]
        if base_scores:
            baseline_val = float(np.mean(base_scores))
        elif self.progress_history:
            baseline_val = float(self.progress_history[0][1])
        else:
            baseline_val = 0.0

        latest_score = self.progress_history[-1][1] if self.progress_history else 0.0
        # Report raw difference: latest_score - baseline_val (never divide by zero)
        trend_raw = float(latest_score - baseline_val)
        trend_vs_it10 = trend_raw

        # 5. Lift statistics
        lift_edges_used = int(np.sum(agent.lift.ci_low > 1.2))
        max_ci_low = float(np.max(agent.lift.ci_low))

        # Check lift consistency
        try:
            agent.lift.assert_consistent()
            lift_consistent = True
        except Exception:
            lift_consistent = False

        pass_dict = {
            "diversity": bool(top_action_share <= 0.50),
            "humility": bool(optionnet_confidence <= 0.80),
            "album_variety": bool(largest_dim_share <= 0.40),
            "trend": bool(trend_raw >= 0.0 or iter_idx < 15),
            "lift": bool(lift_consistent),
        }

        report_card = {
            "top_action_share": top_action_share,
            "optionnet_confidence": optionnet_confidence,
            "goal_bank_largest_dim_share": largest_dim_share,
            "goal_bank_per_dim_counts": per_dim_counts,
            "ach_per_ep_trend_raw": trend_raw,
            "ach_per_ep_trend_vs_it10": trend_vs_it10,
            "lift_edges_used": lift_edges_used,
            "max_ci_low": max_ci_low,
            "alarms_so_far": [a["alarm"] if isinstance(a, dict) else str(a) for a in self.alarms_so_far],
            "pass": pass_dict,
        }

        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(report_card, f, indent=2)

        return report_card
