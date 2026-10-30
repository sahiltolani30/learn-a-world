"""
src/models/predictor.py

Action-Conditioned Latent Predictor Ensemble for Crafter & Atari JEPA World Model.

Fulfills requirements:
- R1: Pixel-free latent dynamics modeling via residual MLP delta prediction.
- R2: Epistemic predictor ensemble (K=4) for aleatoric noise rejection and intrinsic curiosity.
- R4: Plug-and-play discrete action embedding supporting Crafter (17) and Atari (variable).
- R5: Ultra-fast CPU execution (<0.5ms single-step latency, <600k parameter budget).
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEmbedding(nn.Module):
    """
    Learnable discrete action embedding module.
    
    Supports:
    - 1D integer action tensors: shape (B,)
    - 2D integer action tensors: shape (B, 1)
    - 2D one-hot float tensors: shape (B, num_actions) (e.g. from policy distribution)
    """

    def __init__(self, num_actions: int = 17, action_dim: int = 32):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.embedding = nn.Embedding(num_actions, action_dim)

        # Initialize embedding weights with standard normal distribution scaled by 0.1
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Embed discrete action into continuous space R^{action_dim}.
        
        Args:
            a: Tensor of action indices or one-hot vectors.
            
        Returns:
            a_emb: Tensor of shape (B, action_dim).
        """
        if a.dim() == 2 and a.size(-1) == 1:
            a = a.squeeze(-1)

        if a.dim() == 1:
            if a.dtype not in (torch.int64, torch.int32):
                a = a.long()
            if (a < 0).any() or (a >= self.num_actions).any():
                raise IndexError(
                    f"Action index out of range [0, {self.num_actions - 1}]. "
                    f"Min action: {a.min().item()}, Max action: {a.max().item()}"
                )
            return self.embedding(a)
        elif a.dim() == 2 and a.size(-1) == self.num_actions:
            # Differentiable one-hot or categorical probability projection
            return torch.matmul(a.float(), self.embedding.weight)
        else:
            raise ValueError(
                f"Unsupported action tensor shape {a.shape} for num_actions={self.num_actions}. "
                f"Expected (B,), (B, 1), or (B, {self.num_actions})."
            )


class ResidualMLPPredictorHead(nn.Module):
    """
    Individual Residual MLP Head predicting latent transition deltas.
    
    Architecture:
        Linear(in_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU()
        Linear(hidden_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU()
        Linear(hidden_dim, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 288,
        hidden_dim: int = 192,
        out_dim: int = 256,
        head_id: int = 0,
    ):
        super().__init__()
        self.head_id = head_id
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()

        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization on hidden layers for diverse representations
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        # Output projection initialized with small gain to anchor identity prior
        nn.init.orthogonal_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted latent delta.
        
        Args:
            x: Concatenated [z_t, e(a_t)] tensor of shape (B, in_dim).
            
        Returns:
            delta_z: Predicted delta tensor of shape (B, out_dim).
        """
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        delta_z = self.fc3(h)
        return delta_z


class ResidualMLPPredictor(nn.Module):
    """
    4-Head Residual MLP Predictor Ensemble for Action-Conditioned JEPA World Model.
    
    Given current latent state z_t and action a_t:
        1. Embed action e(a_t) in R^{action_dim}.
        2. Concatenate [z_t, e(a_t)].
        3. Predict transition deltas Delta_phi^{(k)} across all K=4 ensemble heads.
        4. Add residual connection: z_hat^{(k)}_{t+1} = z_t + Delta_phi^{(k)}.
        5. Return consensus mean z_bar_{t+1} and full stacked ensemble z_hat_{t+1}.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 17,
        action_dim: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 192,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.action_embedding = ActionEmbedding(num_actions=num_actions, action_dim=action_dim)
        in_dim = latent_dim + action_dim

        self.heads = nn.ModuleList([
            ResidualMLPPredictorHead(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                head_id=k,
            )
            for k in range(num_heads)
        ])

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ensemble predictor.
        
        Args:
            z: Latent state tensor of shape (B, latent_dim).
            a: Action tensor of shape (B,) or (B, 1) or (B, num_actions).
            
        Returns:
            mean_pred: Consensus ensemble prediction of shape (B, latent_dim).
            all_preds: Stacked predictions across heads of shape (num_heads, B, latent_dim).
        """
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent state z of shape (B, {self.latent_dim}), got {z.shape}"
            )

        # 1. Embed action: (B, action_dim)
        a_emb = self.action_embedding(a)

        # 2. Condition by concatenation: (B, latent_dim + action_dim)
        x = torch.cat([z, a_emb], dim=-1)

        # 3. Predict deltas for each head: (num_heads, B, latent_dim)
        deltas = torch.stack([head(x) for head in self.heads], dim=0)

        # 4. Residual addition: z_hat^{(k)} = z_t + Delta^{(k)}
        all_preds = z.unsqueeze(0) + deltas

        # 5. Consensus mean prediction
        mean_pred = all_preds.mean(dim=0)

        return mean_pred, all_preds

    def compute_epistemic_disagreement(self, all_preds: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic disagreement (variance across heads) as intrinsic curiosity.
        
        Formula:
            r_epistemic = (1/K) * sum_{k=1}^K || z_hat^{(k)} - z_bar ||_2^2
            
        Args:
            all_preds: Tensor of shape (num_heads, B, latent_dim).
            
        Returns:
            epistemic_variance: Non-negative tensor of shape (B,).
        """
        mean = all_preds.mean(dim=0, keepdim=True)  # (1, B, latent_dim)
        # Sum squared deviation over latent dimensions, then average across heads
        epistemic_var = ((all_preds - mean) ** 2).sum(dim=-1).mean(dim=0)  # (B,)
        return epistemic_var

    def predict_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Convenience method returning only the ensemble consensus prediction.
        
        Args:
            z: Latent state tensor of shape (B, latent_dim).
            a: Action tensor.
            
        Returns:
            mean_pred: Latent state tensor of shape (B, latent_dim).
        """
        mean_pred, _ = self.forward(z, a)
        return mean_pred

    def rollout(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        batch_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive multi-step latent rollout for Model Predictive Control (MPC/CEM).
        
        Args:
            z_0: Initial latent states of shape (B, latent_dim).
            actions: Action sequence tensor of shape (B, H) [if batch_first] or (H, B) [if not batch_first],
                or 1D tensor (B,) for H=1.
            batch_first: If True and actions is 2D with H == B, treats actions as (B, H).
                Default: True.
            
        Returns:
            traj_mean: Predicted mean trajectory of shape (H, B, latent_dim).
            traj_all: Full ensemble trajectory of shape (num_heads, H, B, latent_dim).
        """
        if actions.dim() == 1:
            # Single-step action (B,) -> shape (1, B)
            if actions.size(0) != z_0.size(0):
                raise ValueError(
                    f"Action batch size {actions.size(0)} does not match z_0 batch size {z_0.size(0)}"
                )
            action_seq = actions.unsqueeze(0)
            H = 1
        elif actions.dim() == 2:
            if actions.size(0) == actions.size(1) == z_0.size(0):
                # Disambiguate for H == B using batch_first flag:
                # If batch_first, treat shape as (B, H) and transpose to (H, B);
                # if not batch_first, treat as (H, B).
                if batch_first:
                    action_seq = actions.transpose(0, 1)
                else:
                    action_seq = actions
                H = action_seq.size(0)
            elif actions.size(1) == z_0.size(0):
                # Shape (H, B)
                action_seq = actions
                H = actions.size(0)
            elif actions.size(0) == z_0.size(0):
                # Shape (B, H)
                action_seq = actions.transpose(0, 1)
                H = action_seq.size(0)
            else:
                raise ValueError(
                    f"Unsupported action sequence shape {actions.shape} for batch size {z_0.size(0)}. "
                    f"Expected (H, B) or (B, H)."
                )
        else:
            raise ValueError(
                f"Unsupported action sequence shape {actions.shape}. Expected (H, B) or (B, H)."
            )

        traj_mean = []
        traj_all = []
        curr_z = z_0

        for t in range(H):
            m_pred, a_pred = self.forward(curr_z, action_seq[t])
            traj_mean.append(m_pred)
            traj_all.append(a_pred)
            curr_z = m_pred

        # Stack across horizon: (H, B, latent_dim)
        stacked_mean = torch.stack(traj_mean, dim=0)
        # Stack across horizon: (num_heads, H, B, latent_dim)
        stacked_all = torch.stack(traj_all, dim=1)

        return stacked_mean, stacked_all


# Interface alias matching PROJECT.md and task specification
LatentPredictor = ResidualMLPPredictor
