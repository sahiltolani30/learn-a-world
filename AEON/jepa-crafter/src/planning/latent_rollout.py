"""Parallel latent rollouts for candidate action sequences."""
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from src.models.jepa_world_model import JepaWorldModel


class LatentRollout:
    """Executes parallel multi-step latent rollouts for candidate action sequences."""

    def __init__(self, model: JepaWorldModel):
        self.model = model

    @torch.no_grad()
    def rollout_candidates(
        self,
        initial_latent: torch.Tensor,
        initial_context: torch.Tensor,
        candidate_sequences: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Roll out candidate action sequences in parallel under torch.no_grad().
        
        Args:
            initial_latent: [1, 64, 128]
            initial_context: [1, 256]
            candidate_sequences: [N, H] int64 candidate action tokens
        Returns:
            dict containing:
                'pooled_latents': [N, H, 128]
                'rewards': [N, H] predicted scalar rewards
                'termination_probs': [N, H] predicted termination probabilities
                'disagreements': [N, H] ensemble disagreement
        """
        N, H = candidate_sequences.shape
        device = candidate_sequences.device

        # Expand initial states across all N candidates
        z_t = initial_latent.expand(N, -1, -1)   # [N, 64, 128]
        h_t = initial_context.expand(N, -1)     # [N, 256]

        all_pooled = []
        all_rewards = []
        all_terms = []
        all_disagreements = []

        for step in range(H):
            action_step = candidate_sequences[:, step]  # [N]
            a_emb = self.model.action_embedding(action_step)  # [N, 64]

            # Use member 0 for rollout dynamics (or ensemble mean)
            z_next = self.model.predictor_ensemble.forward_member(0, z_t, h_t, a_emb)
            u_next = self.model.online_encoder.pool(z_next)  # [N, 128]
            h_next = self.model.temporal_context.step(u_next, a_emb, h_t)

            # Auxiliary outputs
            aux = self.model.auxiliary_heads(u_next, h_next)
            all_rewards.append(aux.reward.squeeze(-1))           # [N]
            all_terms.append(aux.done_probability.squeeze(-1))   # [N]
            all_pooled.append(u_next)

            # Compute disagreement across ensemble members at this step
            predictions = self.model.predictor_ensemble.forward_all(z_t, h_t, a_emb)
            disag = self.model.predictor_ensemble.compute_disagreement(predictions)
            all_disagreements.append(disag)

            z_t = z_next
            h_t = h_next

        return {
            "pooled_latents": torch.stack(all_pooled, dim=1),       # [N, H, 128]
            "rewards": torch.stack(all_rewards, dim=1),             # [N, H]
            "termination_probs": torch.stack(all_terms, dim=1),     # [N, H]
            "disagreements": torch.stack(all_disagreements, dim=1), # [N, H]
        }
