"""MOSAIC World Model combining encoder, stochastic belief, dynamics ensemble, and anchor (Phase P2).

Enforces strict clean-room visual invariants: operates solely on raw pixels,
previous scalar rewards, and opaque action tokens.
"""

from __future__ import annotations

import copy
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VisualEncoder, AnchorDecoder
from .belief import RecurrentStochasticBelief, StochasticDistribution
from .dynamics import ActionEmbedding, DynamicsEnsemble


class MosaicWorldModel(nn.Module):
    """Integrated world model for MOSAIC Phase P2."""

    def __init__(
        self,
        num_actions: int = 17,
        latent_dim: int = 128,
        action_dim: int = 64,
        reward_dim: int = 16,
        hidden_dim: int = 256,
        stochastic_dim: int = 32,
        num_ensemble_members: int = 3,
        ema_decay: float = 0.99,
        use_anchor_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.num_ensemble_members = num_ensemble_members
        self.ema_decay = ema_decay
        self.use_anchor_decoder = use_anchor_decoder

        # 1. Online visual encoder
        self.encoder = VisualEncoder(latent_dim=latent_dim)

        # 2. Target visual encoder (EMA, frozen gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 3. Action embedding
        self.action_embedding = ActionEmbedding(num_actions=num_actions, embedding_dim=action_dim)

        # 4. Recurrent stochastic belief
        self.belief = RecurrentStochasticBelief(
            visual_dim=latent_dim,
            action_dim=action_dim,
            reward_dim=reward_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
        )

        # 5. Dynamics predictor ensemble
        self.dynamics = DynamicsEnsemble(
            belief_dim=self.belief.belief_dim,
            action_dim=action_dim,
            num_members=num_ensemble_members,
            hidden_dim=hidden_dim,
            target_dim=latent_dim,
        )

        # 6. Perceptual anchor decoder
        if self.use_anchor_decoder:
            self.anchor_decoder: Optional[AnchorDecoder] = AnchorDecoder(latent_dim=latent_dim)
        else:
            self.anchor_decoder = None

    @torch.no_grad()
    def update_target_network(self) -> None:
        """Update EMA target encoder: theta_target = decay * theta_target + (1 - decay) * theta_online."""
        for param_online, param_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.mul_(self.ema_decay).add_(param_online.data, alpha=1.0 - self.ema_decay)

    def initial_belief(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (h_0, z_0, b_0)."""
        h_0, z_0 = self.belief.initial_state(batch_size, device=device)
        b_0 = torch.cat([h_0, z_0], dim=-1)
        return h_0, z_0, b_0

    def step_belief(
        self,
        obs_uint8_or_float: torch.Tensor,
        action_token_prev: torch.Tensor,
        reward_prev: torch.Tensor,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        use_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, StochasticDistribution, StochasticDistribution, torch.Tensor]:
        """Perform one online belief update step.

        Returns:
            b_t: updated belief [B, belief_dim]
            h_t: updated context [B, hidden_dim]
            post_dist: posterior distribution
            prior_dist: prior distribution
            e_pooled: visual feature [B, latent_dim]
        """
        tokens, e_pooled = self.encoder(obs_uint8_or_float)
        a_prev_emb = self.action_embedding(action_token_prev)
        b_t, h_t, post_dist, prior_dist = self.belief.step(
            e_t=e_pooled,
            a_prev_emb=a_prev_emb,
            r_prev=reward_prev,
            h_prev=h_prev,
            z_prev=z_prev,
            use_posterior=use_posterior,
        )
        return b_t, h_t, post_dist, prior_dist, e_pooled

    def imagine_transition(
        self,
        belief: torch.Tensor,
        action_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Imagine an action-conditioned transition through the dynamics ensemble."""
        action_emb = self.action_embedding(action_token)
        return self.dynamics(belief, action_emb)

    def compute_losses(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        continue_seq: torch.Tensor,
        lambda_dyn: float = 1.0,
        lambda_anchor: float = 0.1,
        lambda_r: float = 1.0,
        lambda_c: float = 1.0,
        lambda_kl: float = 0.1,
        free_bits: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses on a batched trajectory sequence of length T.

        Args:
            obs_seq: [B, T, 3, 64, 64] uint8 or float [0, 1]
            action_seq: [B, T] int tokens
            reward_seq: [B, T] float scalar rewards
            continue_seq: [B, T] float continuation flags (1.0 = not terminated, 0.0 = done)
        """
        B, T = action_seq.shape
        device = obs_seq.device

        if obs_seq.dtype == torch.uint8:
            obs_norm = obs_seq.float() / 255.0
        else:
            obs_norm = obs_seq

        # Flatten sequence for parallel visual encoding
        flat_obs = obs_norm.view(B * T, 3, 64, 64)
        flat_tokens, flat_pooled = self.encoder(flat_obs)
        pooled_seq = flat_pooled.view(B, T, self.latent_dim)
        tokens_seq = flat_tokens.view(B, T, 64, self.latent_dim)

        # Target encoder forward pass without gradients
        with torch.no_grad():
            _, flat_target_pooled = self.target_encoder(flat_obs)
            target_pooled_seq = flat_target_pooled.view(B, T, self.latent_dim)

        # Embedded actions
        act_emb_seq = self.action_embedding(action_seq)

        # Sequential belief filtering
        h_t, z_t, b_t = self.initial_belief(B, device=device)
        dummy_action_emb = torch.zeros(B, self.action_dim, device=device)
        dummy_reward = torch.zeros(B, 1, device=device)

        beliefs = []
        post_dists = []
        prior_dists = []

        for t in range(T):
            a_prev = dummy_action_emb if t == 0 else act_emb_seq[:, t - 1]
            r_prev = dummy_reward if t == 0 else reward_seq[:, t - 1].unsqueeze(-1)
            b_t, h_t, post_dist, prior_dist = self.belief.step(
                e_t=pooled_seq[:, t],
                a_prev_emb=a_prev,
                r_prev=r_prev,
                h_prev=h_t,
                z_prev=z_t,
                use_posterior=True,
            )
            z_t = b_t[:, self.hidden_dim :]
            beliefs.append(b_t)
            post_dists.append(post_dist)
            prior_dists.append(prior_dist)

        stacked_beliefs = torch.stack(beliefs, dim=1)  # [B, T, belief_dim]

        # 1. Dynamics & Auxiliary loss over t = 0 ... T-2 (predicting t+1)
        # Flatten transitions: [B * (T - 1), ...]
        valid_T = T - 1
        if valid_T > 0:
            flat_curr_b = stacked_beliefs[:, :valid_T].reshape(B * valid_T, -1)
            flat_curr_a = act_emb_seq[:, :valid_T].reshape(B * valid_T, -1)
            target_latents = target_pooled_seq[:, 1:].reshape(B * valid_T, -1)  # [B * valid_T, latent_dim]
            target_rewards = reward_seq[:, :valid_T].reshape(B * valid_T, 1)
            target_continues = continue_seq[:, :valid_T].reshape(B * valid_T, 1)

            dyn_out = self.dynamics(flat_curr_b, flat_curr_a)
            # Members latent preds: [M, B * valid_T, latent_dim]
            member_latent_preds = dyn_out["latent_pred_members"]
            member_rewards = dyn_out["reward_members"]
            member_continues = dyn_out["continue_logit_members"]

            # Multi-member cosine latent prediction loss
            target_norm = F.normalize(target_latents, p=2, dim=-1).unsqueeze(0)  # [1, N, D]
            pred_norm = F.normalize(member_latent_preds, p=2, dim=-1)  # [M, N, D]
            cosine_sim = (pred_norm * target_norm).sum(dim=-1)  # [M, N]
            loss_dyn = (1.0 - cosine_sim).mean()

            # Reward loss (SmoothL1)
            target_r_exp = target_rewards.unsqueeze(0).expand(self.num_ensemble_members, -1, -1).contiguous()
            loss_r = F.smooth_l1_loss(member_rewards, target_r_exp)

            # Continuation loss (BCEWithLogits)
            target_c_exp = target_continues.unsqueeze(0).expand(self.num_ensemble_members, -1, -1).contiguous()
            loss_c = F.binary_cross_entropy_with_logits(member_continues, target_c_exp)
            disagreement = dyn_out["disagreement"].mean()
        else:
            loss_dyn = torch.tensor(0.0, device=device)
            loss_r = torch.tensor(0.0, device=device)
            loss_c = torch.tensor(0.0, device=device)
            disagreement = torch.tensor(0.0, device=device)

        # 2. Perceptual Anchor loss
        if self.use_anchor_decoder and self.anchor_decoder is not None:
            recon = self.anchor_decoder(flat_tokens)  # [B * T, 3, 64, 64]
            loss_anchor = F.mse_loss(recon, flat_obs)
        else:
            loss_anchor = torch.tensor(0.0, device=device)

        # 3. KL Divergence loss between posterior and prior
        kl_list = [post_dists[t].kl(prior_dists[t], free_bits=free_bits).mean() for t in range(T)]
        loss_kl = torch.stack(kl_list).mean()

        # Total combined objective
        total_loss = (
            lambda_dyn * loss_dyn
            + lambda_anchor * loss_anchor
            + lambda_r * loss_r
            + lambda_c * loss_c
            + lambda_kl * loss_kl
        )

        return {
            "total_loss": total_loss,
            "loss_dyn": loss_dyn,
            "loss_anchor": loss_anchor,
            "loss_r": loss_r,
            "loss_c": loss_c,
            "loss_kl": loss_kl,
            "disagreement": disagreement,
        }
