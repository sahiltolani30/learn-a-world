"""Anonymous action embedding table."""
import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Learned embedding table for anonymous action tokens."""

    def __init__(self, num_actions: int = 17, embedding_dim: int = 64):
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        # Initialize with standard normal scaled
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_tokens: [B] or [B, T] int64 tensor with values in [0, num_actions-1]
        Returns:
            embedded: [B, 64] or [B, T, 64]
        """
        return self.embedding(action_tokens)
