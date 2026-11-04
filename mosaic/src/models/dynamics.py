"""Action-conditioned dynamics ensemble and uncertainty models for MOSAIC (Phase P2).

Predicts future belief states, immediate scalar reward, and episode continuation,
providing calibrated epistemic uncertainty across ensemble members.
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEmbedding(nn.Module):
    """Embeds opaque discrete action tokens into continuous representation vectors."""

    def __init__(self, num_actions: int = 17, embedding_dim: int = 64) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_actions, embedding_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: integer tensor [B] or [B, T] with values in [0, num_actions - 1].
        Returns:
            embedded: float tensor [B, embedding_dim] or [B, T, embedding_dim].
        """
        return self.embedding(actions)


class DynamicsPredictorMember(nn.Module):
    """Single dynamics ensemble member predicting next belief, scalar reward, and continuation."""

    def __init__(
        self,
        belief_dim: int,
        action_dim: int = 64,
        hidden_dim: int = 256,
        target_dim: int = 128,
    ) -> None:
        super().__init__()
        self.belief_dim = belief_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        # Core transition network: [belief; action] -> hidden -> next_belief_delta
        in_dim = belief_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Belief projection head: predicts delta to next belief
        self.belief_delta_head = nn.Linear(hidden_dim, belief_dim)

        # Target latent projection head (for JEPA prediction loss in latent space)
        self.latent_pred_head = nn.Linear(hidden_dim, target_dim)

        # Reward head: predicts scalar environment reward
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Continuation head: predicts continuation logit (logit of not-terminated)
        self.continue_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        belief: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            belief: [B, belief_dim] current predictive belief
            action_emb: [B, action_dim] candidate action embedding
        Returns:
            next_belief: [B, belief_dim] predicted next belief
            latent_pred: [B, target_dim] predicted visual latent for JEPA loss
            reward_pred: [B, 1] predicted immediate scalar reward
            continue_logit: [B, 1] predicted continuation logit
        """
        feat = torch.cat([belief, action_emb], dim=-1)
        h = self.net(feat)

        delta_b = self.belief_delta_head(h)
        next_belief = belief + delta_b
        latent_pred = self.latent_pred_head(h)
        reward_pred = self.reward_head(h)
        continue_logit = self.continue_head(h)

        return next_belief, latent_pred, reward_pred, continue_logit


class DynamicsEnsemble(nn.Module):
    """Ensemble of M dynamics predictors providing action-conditioned transitions and uncertainty."""

    def __init__(
        self,
        belief_dim: int,
        action_dim: int = 64,
        num_members: int = 3,
        hidden_dim: int = 256,
        target_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_members = int(num_members)
        self.belief_dim = belief_dim
        self.action_dim = action_dim

        self.members = nn.ModuleList([
            DynamicsPredictorMember(
                belief_dim=belief_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                target_dim=target_dim,
            )
            for _ in range(self.num_members)
        ])

    def forward(
        self,
        belief: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all ensemble members.

        Returns:
            dict containing:
                next_belief_mean: [B, belief_dim]
                next_belief_members: [M, B, belief_dim]
                latent_pred_mean: [B, target_dim]
                latent_pred_members: [M, B, target_dim]
                reward_mean: [B, 1]
                reward_members: [M, B, 1]
                continue_logit_mean: [B, 1]
                continue_logit_members: [M, B, 1]
                disagreement: [B] epistemic variance across member latent predictions
        """
        all_next_beliefs = []
        all_latent_preds = []
        all_rewards = []
        all_continues = []

        for member in self.members:
            nb, lp, rew, cont = member(belief, action_emb)
            all_next_beliefs.append(nb)
            all_latent_preds.append(lp)
            all_rewards.append(rew)
            all_continues.append(cont)

        # Stack across members: [M, B, ...]
        stacked_nb = torch.stack(all_next_beliefs, dim=0)
        stacked_lp = torch.stack(all_latent_preds, dim=0)
        stacked_rew = torch.stack(all_rewards, dim=0)
        stacked_cont = torch.stack(all_continues, dim=0)

        # Means across ensemble
        mean_nb = stacked_nb.mean(dim=0)
        mean_lp = stacked_lp.mean(dim=0)
        mean_rew = stacked_rew.mean(dim=0)
        mean_cont = stacked_cont.mean(dim=0)

        # Epistemic disagreement: mean squared deviation from ensemble center
        # D(b, a) = (1/M) sum_m || lp^(m) - bar_lp ||^2
        diff = stacked_lp - mean_lp.unsqueeze(0)  # [M, B, target_dim]
        disagreement = diff.pow(2).sum(dim=-1).mean(dim=0)  # [B]

        return {
            "next_belief_mean": mean_nb,
            "next_belief_members": stacked_nb,
            "latent_pred_mean": mean_lp,
            "latent_pred_members": stacked_lp,
            "reward_mean": mean_rew,
            "reward_members": stacked_rew,
            "continue_logit_mean": mean_cont,
            "continue_logit_members": stacked_cont,
            "disagreement": disagreement,
        }
