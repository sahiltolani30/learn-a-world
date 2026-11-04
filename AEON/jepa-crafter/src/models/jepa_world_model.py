"""JEPA World Model integrating encoder, context, predictor ensemble, and auxiliary heads."""
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.visual_encoder import VisualEncoder
from src.models.ema_encoder import EmaTargetEncoder
from src.models.action_embedding import ActionEmbedding
from src.models.temporal_context import TemporalContext
from src.models.predictor_ensemble import PredictorEnsemble
from src.models.auxiliary_heads import AuxiliaryHeads, AuxiliaryOutputs


class JepaWorldModel(nn.Module):
    """Complete JEPA Crafter World Model."""

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 64,
        context_dim: int = 256,
        num_actions: int = 17,
        num_predictor_members: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.num_predictor_members = num_predictor_members

        # Trainable online encoder
        self.online_encoder = VisualEncoder(latent_dim=latent_dim)
        
        # Stop-gradient EMA target encoder
        self.target_encoder = EmaTargetEncoder(self.online_encoder)

        # Action embedding table
        self.action_embedding = ActionEmbedding(num_actions=num_actions, embedding_dim=action_dim)

        # Temporal context GRU
        self.temporal_context = TemporalContext(visual_dim=latent_dim, action_dim=action_dim, hidden_dim=context_dim)

        # 3-member predictor ensemble
        self.predictor_ensemble = PredictorEnsemble(
            num_members=num_predictor_members,
            latent_dim=latent_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
        )

        # Auxiliary prediction heads
        self.auxiliary_heads = AuxiliaryHeads(visual_dim=latent_dim, context_dim=context_dim)

    def encode_actual_history(
        self,
        obs_context: torch.Tensor,
        previous_actions: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode context observations [B, T_ctx, 3, 64, 64] and action history into (z_current, h_t).
        
        Args:
            obs_context: [B, T_ctx, 3, 64, 64]
            previous_actions: [B, T_ctx]
            context_mask: [B, T_ctx] bool
        Returns:
            z_t: [B, 64, 128] spatial latent tokens at time t (last frame in context)
            h_t: [B, 256] context state at time t
        """
        B, T_ctx, C, H, W = obs_context.shape
        flat_obs = obs_context.reshape(B * T_ctx, C, H, W)
        all_tokens, all_pooled = self.online_encoder(flat_obs)
        
        tokens_seq = all_tokens.reshape(B, T_ctx, 64, self.latent_dim)
        pooled_seq = all_pooled.reshape(B, T_ctx, self.latent_dim)
        
        action_embs = self.action_embedding(previous_actions)  # [B, T_ctx, 64]
        h_t = self.temporal_context.forward_history(pooled_seq, action_embs, validity_mask=context_mask)

        # z_t is the last spatial grid in the context window
        z_t = tokens_seq[:, -1]
        return z_t, h_t

    def predict_one_step_ensemble(
        self,
        current_tokens: torch.Tensor,
        context_state: torch.Tensor,
        action_token: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Predict next step across all ensemble members.
        
        Returns:
            predictions: list of num_members tensors [B, 64, 128]
            mean_pooled: [B, 128] normalized mean pooled latent
            disagreement: [B] pairwise cosine disagreement
        """
        if action_token.dim() == 0:
            action_token = action_token.unsqueeze(0).expand(current_tokens.size(0))
        elif action_token.dim() == 1 and action_token.size(0) != current_tokens.size(0):
            action_token = action_token.expand(current_tokens.size(0))
            
        a_emb = self.action_embedding(action_token)
        predictions = self.predictor_ensemble.forward_all(current_tokens, context_state, a_emb)
        disagreement = self.predictor_ensemble.compute_disagreement(predictions)

        # Compute ensemble mean pooled latent
        pooled_members = [self.online_encoder.pool(p) for p in predictions]
        mean_pooled = torch.stack(pooled_members, dim=0).mean(dim=0)
        return predictions, mean_pooled, disagreement

    def autoregressive_rollout(
        self,
        initial_tokens: torch.Tensor,
        initial_context: torch.Tensor,
        future_action_tokens: Optional[torch.Tensor] = None,
        future_action_embeddings: Optional[torch.Tensor] = None,
        member_idx: int = 0,
    ) -> Dict[str, Any]:
        """Autoregressively roll out predictions over future action tokens or embeddings for a single member.
        
        Args:
            initial_tokens: [B, 64, 128]
            initial_context: [B, 256]
            future_action_tokens: [B, H] int tokens
            future_action_embeddings: [B, H, 64] float embeddings (for true zero/ablation controls)
            member_idx: which ensemble member to use
        Returns:
            dict with:
                'predictions_by_horizon': dict mapping horizon k -> z_hat [B, 64, 128]
                'pooled_by_horizon': dict mapping horizon k -> u_hat [B, 128]
                'auxiliary_first': AuxiliaryOutputs for step 1
        """
        if future_action_embeddings is not None:
            B, H, _ = future_action_embeddings.shape
        elif future_action_tokens is not None:
            B, H = future_action_tokens.shape
        else:
            raise ValueError("Must provide either future_action_tokens or future_action_embeddings")

        z_hat = initial_tokens
        h = initial_context
        
        predictions_by_horizon = {}
        pooled_by_horizon = {}
        aux_first = None

        for k in range(1, H + 1):
            if future_action_embeddings is not None:
                a_emb = future_action_embeddings[:, k - 1]
            else:
                a_idx = future_action_tokens[:, k - 1]
                a_emb = self.action_embedding(a_idx)

            z_hat = self.predictor_ensemble.forward_member(member_idx, z_hat, h, a_emb)
            u_hat = self.online_encoder.pool(z_hat)
            h = self.temporal_context.step(u_hat, a_emb, h)

            predictions_by_horizon[k] = z_hat
            pooled_by_horizon[k] = u_hat

            if k == 1:
                aux_first = self.auxiliary_heads(u_hat, h)

        return {
            "predictions_by_horizon": predictions_by_horizon,
            "pooled_by_horizon": pooled_by_horizon,
            "auxiliary_first": aux_first,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts for each module."""
        def num_params(mod):
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)

        return {
            "online_encoder": num_params(self.online_encoder),
            "action_embedding": num_params(self.action_embedding),
            "temporal_context": num_params(self.temporal_context),
            "predictor_ensemble": num_params(self.predictor_ensemble),
            "auxiliary_heads": num_params(self.auxiliary_heads),
            "total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "target_encoder_frozen": sum(p.numel() for p in self.target_encoder.parameters()),
        }
