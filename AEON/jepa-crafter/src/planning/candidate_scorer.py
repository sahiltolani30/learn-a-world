"""Candidate sequence scoring for MPC planning."""
from typing import Dict, Any, Optional
import torch
from src.exploration.role_definitions import RoleWeights
from src.exploration.novelty import NoveltyEstimator


class CandidateScorer:
    """Scores imagined latent rollouts using predicted rewards, intrinsic bonuses, and risk penalties."""

    def __init__(
        self,
        novelty_estimator: Optional[NoveltyEstimator] = None,
        discount: float = 0.99,
        max_intrinsic_bonus_per_step: float = 2.0,
    ):
        self.novelty = novelty_estimator if novelty_estimator is not None else NoveltyEstimator()
        self.discount = discount
        self.max_intrinsic_bonus_per_step = max_intrinsic_bonus_per_step

    def score_rollouts(
        self,
        rollout_dict: Dict[str, torch.Tensor],
        role_weights: RoleWeights,
    ) -> torch.Tensor:
        """Calculate total trajectory score for each candidate sequence.
        
        Args:
            rollout_dict: dict containing:
                'pooled_latents': [N, H, 128]
                'rewards': [N, H]
                'termination_probs': [N, H]
                'disagreements': [N, H]
            role_weights: RoleWeights containing novelty, disagreement, reward, and penalties
        Returns:
            scores: [N] tensor of candidate scores
        """
        pooled_latents = rollout_dict["pooled_latents"]  # [N, H, 128]
        rewards = rollout_dict["rewards"]                # [N, H]
        term_probs = rollout_dict["termination_probs"]  # [N, H]
        disagreements = rollout_dict["disagreements"]    # [N, H]

        N, H, _ = pooled_latents.shape
        device = pooled_latents.device
        scores = torch.zeros(N, device=device)

        # Discount weights gamma^k
        discounts = torch.tensor([self.discount ** k for k in range(H)], device=device)  # [H]

        for i in range(N):
            seq_score = 0.0
            for k in range(H):
                r_hat = rewards[i, k].item()
                # Novelty of imagined latent (strictly frozen normalizer & no memory mutation)
                nov_hat = self.novelty.evaluate(pooled_latents[i, k], update_memory=False, update_normalizer=False)
                # Disagreement
                dis_hat = disagreements[i, k].item()

                # Intrinsic bonus per step capped
                raw_intrinsic = role_weights.novelty * nov_hat + role_weights.disagreement * dis_hat
                intrinsic = min(self.max_intrinsic_bonus_per_step, max(-self.max_intrinsic_bonus_per_step, raw_intrinsic))

                step_value = role_weights.reward * r_hat + intrinsic
                seq_score += (self.discount ** k) * step_value

            # Risk penalty: maximum termination probability along rollout
            max_term = term_probs[i].max().item()
            term_penalty = role_weights.termination_penalty * max_term

            # Uncertainty penalty: average disagreement along rollout
            rollout_uncertainty = disagreements[i].mean().item()
            unc_penalty = role_weights.uncertainty_penalty * rollout_uncertainty

            total_seq_score = seq_score - term_penalty - unc_penalty
            scores[i] = total_seq_score

        return scores
