"""
JEPA v3 Architecture with Recurrent Memory and Task-Shaped Representation (R2 + R5).

Key Architectural Innovations:
1. RecurrentCore: A stateful GRU cell processing (z_t, a_{t-1}, h_{t-1}) -> h_t.
   Provides persistent belief memory across partial observability, dark night cycles, and occlusions.
2. Composite State: s_t = [z_t, h_t]. All task heads (EventHead, HealthHead, QEnsemble, PolicyPrior)
   evaluate on composite state s_t rather than a single memoryless frame.
3. RecurrentPredictor: Rolls forward both visual latent z_{t+1} and recurrent hidden state h_{t+1}
   in imagination during planning.
4. End-to-End Task Shaping: Gradient from task heads and Q flows into the ConvEncoder from step 1,
   preventing representation collapse on task-irrelevant background features.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ConvEncoder
from models.heads import (
    EventHead,
    HealthHead,
    PolicyPrior,
    QEnsemble,
    ReturnNormalizer,
    NUM_ACHIEVEMENTS,
)


class RecurrentCore(nn.Module):
    """
    GRU Recurrent Belief Core.
    Carries temporal context h_t alongside instant latent observation z_t.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_emb_dim: int = 32,
        recurrent_dim: int = 256,
        num_actions: int = 17,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_emb_dim = action_emb_dim
        self.recurrent_dim = recurrent_dim
        self.num_actions = num_actions

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + action_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, recurrent_dim),
            nn.LayerNorm(recurrent_dim),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(recurrent_dim, recurrent_dim)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step recurrence update:
        z: [B, latent_dim]
        action: [B] (int64) or [B, action_emb_dim]
        h_prev: [B, recurrent_dim]
        Returns: h_t: [B, recurrent_dim]
        """
        if action.dtype in (torch.int64, torch.int32):
            if action.ndim > 1:
                action = action.squeeze(-1)
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        za = torch.cat([z, a_emb], dim=-1)
        feat = self.input_proj(za)
        h_next = self.gru(feat, h_prev)
        return h_next

    def unroll_sequence(
        self,
        z_seq: torch.Tensor,
        action_seq: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        reset_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unrolls recurrence along a sequence of transitions:
        z_seq: [B, T, latent_dim]
        action_seq: [B, T]
        h_0: [B, recurrent_dim] (optional, defaults to zeros)
        reset_masks: [B, T] (optional binary mask where 1 = episode reset / boundary)
        Returns: h_seq: [B, T, recurrent_dim]
        """
        B, T, _ = z_seq.shape
        device = z_seq.device

        if h_0 is None:
            h = torch.zeros(B, self.recurrent_dim, device=device)
        else:
            h = h_0

        h_list = []
        for t in range(T):
            if reset_masks is not None:
                # Mask out hidden state on episode boundaries
                mask_t = (1.0 - reset_masks[:, t].float()).unsqueeze(-1)
                h = h * mask_t

            h = self.forward(z_seq[:, t], action_seq[:, t], h)
            h_list.append(h)

        return torch.stack(h_list, dim=1)


class RecurrentPredictor(nn.Module):
    """
    Action-Conditioned Recurrent Latent Predictor.
    Predicts next latent delta conditioned on current latent z, action a, and recurrent context h:
    hat{z}_{t+1} = z_t + Delta z(z_t, h_t, a_t).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_actions: int = 17,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.action_emb_dim = action_emb_dim

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + recurrent_dim + action_emb_dim
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: [..., latent_dim]
        h: [..., recurrent_dim]
        action: [...] int64
        Returns: next_z: [..., latent_dim]
        """
        if action.dtype in (torch.int64, torch.int32):
            if action.ndim > 1:
                action = action.squeeze(-1)
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        # Handle batch or sequence dims
        orig_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.latent_dim)
        h_flat = h.reshape(-1, self.recurrent_dim)
        a_flat = a_emb.reshape(-1, self.action_emb_dim)

        feat = torch.cat([z_flat, h_flat, a_flat], dim=-1)
        delta_z = self.net(feat)
        next_z = z_flat + delta_z
        return next_z.reshape(*orig_shape, self.latent_dim)


class JEPAv3(nn.Module):
    """
    Complete JEPA v3 Model with Recurrent Belief and Task-Shaped Representation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        q_heads: int = 3,
        q_bins: int = 51,
        q_max: float = 12.0,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.state_dim = latent_dim + recurrent_dim  # Composite state dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim
        self.ema_decay = ema_decay

        # 1. Observation Online Encoder
        self.encoder = ConvEncoder(in_channels=3, latent_dim=latent_dim)

        # 2. Observation Target Encoder (EMA consistency target)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 3. Recurrent Core & Predictor
        self.recurrent_core = RecurrentCore(
            latent_dim=latent_dim,
            action_emb_dim=action_emb_dim,
            recurrent_dim=recurrent_dim,
            num_actions=num_actions,
        )

        self.predictor = RecurrentPredictor(
            latent_dim=latent_dim,
            recurrent_dim=recurrent_dim,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )

        # 4. Task Heads operating on Composite State s_t = [z_t, h_t]
        self.event_head = EventHead(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            num_achievements=NUM_ACHIEVEMENTS,
            hidden_dim=hidden_dim,
        )

        self.health_head = HealthHead(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.q_ensemble = QEnsemble(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            num_heads=q_heads,
            num_bins=q_bins,
            v_min=0.0,
            v_max=q_max,
            hidden_dim=hidden_dim,
        )

        # Target Q Ensemble for Clipped Double-Q
        self.q_target_ensemble = copy.deepcopy(self.q_ensemble)
        for p in self.q_target_ensemble.parameters():
            p.requires_grad = False

        self.policy_prior = PolicyPrior(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )

        # Return normalizer for scale stability
        self.return_normalizer = ReturnNormalizer(
            low_percentile=5.0,
            high_percentile=95.0,
            target_max=q_max,
        )

    def encode(self, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Encodes [B, 3, 64, 64] image into latent z [B, latent_dim]."""
        if use_target:
            with torch.no_grad():
                return self.target_encoder(obs)
        return self.encoder(obs)

    def recurrent_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advances recurrent state by one step and returns composite state:
        Returns: (s_t, h_t) where s_t = [z, h_t]
        """
        h_t = self.recurrent_core(z, action, h_prev)
        s_t = torch.cat([z, h_t], dim=-1)
        return s_t, h_t

    def predict_next_step(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagines next state in latent space:
        Returns: (next_z, next_h, next_s)
        """
        next_z = self.predictor(z, h, action)
        next_h = self.recurrent_core(next_z, action, h)
        next_s = torch.cat([next_z, next_h], dim=-1)
        return next_z, next_h, next_s

    def predict_reward(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
        mask_before: torch.Tensor,
        return_decomposition: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Predicts expected reward from composite state s = [z, h]:
        r_hat = sum_i(p_i * (1 - m_i)) + h_hat
        """
        p = self.event_head.predict_proba(s, action, mask_before)
        h = self.health_head(s, action)

        unlocked_mask = mask_before.float()
        locked_weight = 1.0 - unlocked_mask
        achievement_reward = (p * locked_weight).sum(dim=-1)
        total_reward = achievement_reward + h

        if return_decomposition:
            return total_reward, {
                "event_probs": p,
                "achievement_reward": achievement_reward,
                "health_reward": h,
            }
        return total_reward

    def q_min(self, s: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Clipped Double-Q minimum return across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_min(s, action)

    def q_mean(self, s: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Mean expected return across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_mean(s, action)

    def q_std(self, s: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Epistemic uncertainty (std dev) across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_std(s, action)

    def sample_prior_action(
        self,
        s: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Samples categorical action indices guided by PolicyPrior."""
        return self.policy_prior.sample(s, mask, temperature=temperature)

    def update_target_networks(self, decay_encoder: float = 0.99, decay_q: float = 0.995) -> None:
        """Momentum update of target encoder and target Q ensemble."""
        with torch.no_grad():
            # Update target encoder
            for t_p, o_p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                t_p.data.mul_(decay_encoder).add_(o_p.data, alpha=1.0 - decay_encoder)
            # Update target Q ensemble
            for t_p, o_p in zip(self.q_target_ensemble.parameters(), self.q_ensemble.parameters()):
                t_p.data.mul_(decay_q).add_(o_p.data, alpha=1.0 - decay_q)

    def save_checkpoint(self, path: str, step: int = 0) -> None:
        """Saves complete v3 checkpoint."""
        ckpt = {
            "version": "v3",
            "step": step,
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "recurrent_core": self.recurrent_core.state_dict(),
            "predictor": self.predictor.state_dict(),
            "event_head": self.event_head.state_dict(),
            "health_head": self.health_head.state_dict(),
            "q_ensemble": self.q_ensemble.state_dict(),
            "q_target_ensemble": self.q_target_ensemble.state_dict(),
            "policy_prior": self.policy_prior.state_dict(),
            "normalizer": {
                "low": self.return_normalizer.low,
                "high": self.return_normalizer.high,
                "initialized": self.return_normalizer.initialized,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, device: Optional[Union[str, torch.device]] = None) -> None:
        """Loads v3 checkpoint or transfers weights from v2 base."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        ckpt = torch.load(path, map_location=dev)

        if ckpt.get("version") == "v3":
            self.encoder.load_state_dict(ckpt["encoder"])
            self.target_encoder.load_state_dict(ckpt["target_encoder"])
            self.recurrent_core.load_state_dict(ckpt["recurrent_core"])
            self.predictor.load_state_dict(ckpt["predictor"])
            self.event_head.load_state_dict(ckpt["event_head"])
            self.health_head.load_state_dict(ckpt["health_head"])
            self.q_ensemble.load_state_dict(ckpt["q_ensemble"])
            self.q_target_ensemble.load_state_dict(ckpt["q_target_ensemble"])
            self.policy_prior.load_state_dict(ckpt["policy_prior"])
            if "normalizer" in ckpt:
                self.return_normalizer.low = ckpt["normalizer"]["low"]
                self.return_normalizer.high = ckpt["normalizer"]["high"]
                self.return_normalizer.initialized = ckpt["normalizer"]["initialized"]
            print(f"[JEPAv3] Successfully loaded v3 checkpoint from {path}")
        else:
            # Transfer base encoder weights from v1/v2 checkpoint
            base_state = ckpt.get("base_model_state", ckpt.get("model_state", {}))
            encoder_dict = {
                k.replace("encoder.", ""): v
                for k, v in base_state.items()
                if k.startswith("encoder.")
            }
            if encoder_dict:
                self.encoder.load_state_dict(encoder_dict, strict=False)
                self.target_encoder.load_state_dict(encoder_dict, strict=False)
                print(f"[JEPAv3] Transferred {len(encoder_dict)} encoder weights from legacy checkpoint {path}")
            else:
                print(f"[JEPAv3] Initializing from base weights.")
