"""Diagnostics tracking MPC decisions and the imagination gap."""
from typing import Dict, Any, List
import numpy as np


class PlannerDiagnostics:
    """Tracks predicted vs realized outcomes and detects model exploitation."""

    def __init__(self):
        self.predicted_rewards: List[float] = []
        self.actual_rewards: List[float] = []
        self.predicted_terms: List[float] = []
        self.actual_terms: List[float] = []
        self.predicted_disagreements: List[float] = []
        self.scores: List[float] = []
        self.decision_latencies: List[float] = []

    def record_decision(
        self,
        predicted_reward: float,
        actual_reward: float,
        predicted_term: float,
        actual_term: float,
        disagreement: float,
        score: float,
        latency_sec: float = 0.0,
    ) -> None:
        self.predicted_rewards.append(predicted_reward)
        self.actual_rewards.append(actual_reward)
        self.predicted_terms.append(predicted_term)
        self.actual_terms.append(actual_term)
        self.predicted_disagreements.append(disagreement)
        self.scores.append(score)
        if latency_sec > 0:
            self.decision_latencies.append(latency_sec)

    def compute_imagination_gap(self) -> Dict[str, float]:
        """Compute average predicted outcome minus actual observed outcome."""
        if len(self.predicted_rewards) == 0:
            return {
                "imagination_gap_reward": 0.0,
                "reward_prediction_mae": 0.0,
                "term_prediction_brier": 0.0,
                "mean_latency_ms": 0.0,
                "num_decisions": 0,
            }

        p_r = np.array(self.predicted_rewards)
        a_r = np.array(self.actual_rewards)
        p_t = np.array(self.predicted_terms)
        a_t = np.array(self.actual_terms)

        gap = float(np.mean(p_r - a_r))
        mae = float(np.mean(np.abs(p_r - a_r)))
        brier = float(np.mean((p_t - a_t) ** 2))
        latency = float(np.mean(self.decision_latencies) * 1000.0) if self.decision_latencies else 0.0

        return {
            "imagination_gap_reward": gap,
            "reward_prediction_mae": mae,
            "term_prediction_brier": brier,
            "mean_latency_ms": latency,
            "num_decisions": len(self.predicted_rewards),
        }
