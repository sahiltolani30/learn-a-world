"""Predictor ensemble disagreement estimator and learning progress tracking."""
from typing import List
import torch
import torch.nn.functional as F
from src.exploration.novelty import RunningNormalizer


class DisagreementEstimator:
    """Calculates running-normalized disagreement across ensemble member predictions."""

    def __init__(self, clip_val: float = 5.0):
        self.normalizer = RunningNormalizer(clip_val=clip_val)

    def evaluate_predictions(self, member_predictions: List[torch.Tensor]) -> float:
        """Calculate pairwise cosine distance disagreement across member latents.
        
        Args:
            member_predictions: list of tensors of shape [1, 64, 128] or [64, 128]
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
        return self.normalizer.normalize(avg_dist)


class LearningProgressEstimator:
    """Estimates positive learning progress: error_long - error_short."""

    def __init__(self, clip_val: float = 5.0, alpha_short: float = 0.1, alpha_long: float = 0.01):
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.short_ma = 0.5
        self.long_ma = 0.5
        self.normalizer = RunningNormalizer(clip_val=clip_val)

    def update_and_score(self, current_prediction_error: float) -> float:
        self.short_ma = (1.0 - self.alpha_short) * self.short_ma + self.alpha_short * current_prediction_error
        self.long_ma = (1.0 - self.alpha_long) * self.long_ma + self.alpha_long * current_prediction_error
        
        # Positive progress = error is decreasing (long moving average > short moving average)
        progress = max(0.0, self.long_ma - self.short_ma)
        return self.normalizer.normalize(progress)

    def score(self) -> float:
        progress = max(0.0, self.long_ma - self.short_ma)
        return self.normalizer.normalize(progress)
