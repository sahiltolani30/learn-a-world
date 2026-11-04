"""Predictor ensemble disagreement estimator and learning progress tracking."""
from typing import List
import torch
import torch.nn.functional as F
from src.exploration.novelty import RunningNormalizer


class DisagreementEstimator:
    """Calculates running-normalized disagreement across ensemble member predictions."""

    def __init__(self, clip_val: float = 5.0):
        self.normalizer = RunningNormalizer(clip_val=clip_val)

    def evaluate_predictions(self, member_predictions: List[torch.Tensor], update_normalizer: bool = False) -> float:
        """Calculate pairwise cosine distance disagreement across member latents.
        
        Args:
            member_predictions: list of tensors of shape [1, 64, 128] or [64, 128]
            update_normalizer: if True, updates running statistics. Default is False.
        Returns:
            normalized_disagreement: float
        """
        M = len(member_predictions)
        if M < 2:
            return 0.0

        # Mean over spatial tokens to get pooled vector [128]
        normed = [
            F.normalize(p.reshape(-1, 128).mean(dim=0), dim=-1, eps=1e-6)
            for p in member_predictions
        ]

        distances = []
        for i in range(M):
            for j in range(i + 1, M):
                cos_sim = (normed[i] * normed[j]).sum().item()
                distances.append(1.0 - cos_sim)

        avg_dist = float(sum(distances) / len(distances))
        return self.normalizer.normalize(avg_dist, update=update_normalizer)

    def update_from_observation(self, member_predictions: List[torch.Tensor]) -> float:
        """Update normalizer from real observation predictions."""
        return self.evaluate_predictions(member_predictions, update_normalizer=True)

    def get_state(self) -> dict:
        return {"normalizer": self.normalizer.get_state()}

    def set_state(self, state: dict) -> None:
        self.normalizer.set_state(state["normalizer"])


class LearningProgressEstimator:
    """Estimates positive learning progress: error_long - error_short."""

    def __init__(self, clip_val: float = 5.0, alpha_short: float = 0.1, alpha_long: float = 0.01):
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.short_ma = 0.5
        self.long_ma = 0.5
        self.normalizer = RunningNormalizer(clip_val=clip_val)

    def update_from_residual(self, real_prediction_error: float) -> None:
        """Update long and short moving averages and running normalizer from actual measured transition error."""
        self.short_ma = (1.0 - self.alpha_short) * self.short_ma + self.alpha_short * real_prediction_error
        self.long_ma = (1.0 - self.alpha_long) * self.long_ma + self.alpha_long * real_prediction_error
        progress = max(0.0, self.long_ma - self.short_ma)
        self.normalizer.update(progress)

    def score(self) -> float:
        """Read-only score calculation without mutating normalizer or moving averages."""
        progress = max(0.0, self.long_ma - self.short_ma)
        return self.normalizer.normalize(progress, update=False)

    def update_and_score(self, current_prediction_error: float) -> float:
        self.update_from_residual(current_prediction_error)
        return self.score()

    def get_state(self) -> dict:
        return {
            "alpha_short": self.alpha_short,
            "alpha_long": self.alpha_long,
            "short_ma": self.short_ma,
            "long_ma": self.long_ma,
            "normalizer": self.normalizer.get_state(),
        }

    def set_state(self, state: dict) -> None:
        self.alpha_short = state["alpha_short"]
        self.alpha_long = state["alpha_long"]
        self.short_ma = state["short_ma"]
        self.long_ma = state["long_ma"]
        self.normalizer.set_state(state["normalizer"])
