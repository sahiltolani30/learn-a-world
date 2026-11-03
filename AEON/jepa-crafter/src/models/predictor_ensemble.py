"""Ensemble of independently initialized latent predictors."""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.latent_predictor import LatentPredictorMember


class PredictorEnsemble(nn.Module):
    """Ensemble of 3 predictor members for dynamics forecasting and epistemic uncertainty."""

    def __init__(
        self,
        num_members: int = 3,
        latent_dim: int = 128,
        context_dim: int = 256,
        action_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.num_members = num_members
        self.members = nn.ModuleList([
            LatentPredictorMember(
                latent_dim=latent_dim,
                context_dim=context_dim,
                action_dim=action_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
            for _ in range(num_members)
        ])

    def forward_member(
        self,
        member_idx: int,
        current_tokens: torch.Tensor,
        context_state: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through a single ensemble member."""
        return self.members[member_idx](current_tokens, context_state, action_emb)

    def forward_all(
        self,
        current_tokens: torch.Tensor,
        context_state: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Forward pass through all ensemble members. Returns list of [B, 64, 128] tensors."""
        return [
            member(current_tokens, context_state, action_emb)
            for member in self.members
        ]

    def compute_disagreement(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Compute pairwise cosine distance disagreement across ensemble members.
        
        Args:
            predictions: list of num_members tensors of shape [B, 64, 128]
        Returns:
            disagreement: [B] scalar uncertainty estimate per batch item
        """
        # Global pooled representation per member: mean over 64 tokens -> [B, 128]
        normed_pooled = [F.normalize(p.mean(dim=1), dim=-1, eps=1e-6) for p in predictions]
        
        M = len(normed_pooled)
        if M < 2:
            return torch.zeros(predictions[0].size(0), device=predictions[0].device)
            
        distances = []
        for i in range(M):
            for j in range(i + 1, M):
                # 1 - cosine_similarity
                cos_sim = (normed_pooled[i] * normed_pooled[j]).sum(dim=-1)  # [B]
                distances.append(1.0 - cos_sim)
                
        # Average pairwise distance
        disagreement = torch.stack(distances, dim=0).mean(dim=0)  # [B]
        return disagreement
