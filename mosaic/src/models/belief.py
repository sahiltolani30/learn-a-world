"""Recurrent stochastic belief model for MOSAIC (Phase P2).

Maintains a deterministic context h_t and a stochastic latent state z_t.
Exposes posterior q_phi(z_t | h_t, e_t) when current visual frame is observed,
and prior p_psi(z_t | h_t) when imagining rollouts without real visual frames.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDistribution:
    """Normal distribution helper supporting sampling and KL divergence with free-bits."""

    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean = mean
        self.logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.std)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: StochasticDistribution, free_bits: float = 0.1) -> torch.Tensor:
        """Analytical KL divergence D_KL(self || other) summed over latent dim, with free-bits thresholding."""
        var_self = torch.exp(self.logvar)
        var_other = torch.exp(other.logvar)

        # KL per dimension: 0.5 * ( (var_self + (mean_self - mean_other)^2) / var_other - 1 - (logvar_self - logvar_other) )
        kl_elem = 0.5 * (
            (var_self + (self.mean - other.mean).pow(2)) / var_other
            - 1.0
            - (self.logvar - other.logvar)
        )
        kl_dim = kl_elem.sum(dim=-1)  # [B]
        if free_bits > 0.0:
            kl_dim = torch.clamp(kl_dim, min=free_bits)
        return kl_dim


class RecurrentStochasticBelief(nn.Module):
    """Belief core combining GRU temporal context and Gaussian stochastic state.

    Input representations:
        e_t: pooled visual feature [B, visual_dim]
        a_prev: embedded previous action [B, action_dim]
        r_prev: previous scalar reward [B, 1]
    State:
        h_t: deterministic context [B, hidden_dim]
        z_t: stochastic sample [B, stochastic_dim]
        b_t = [h_t; z_t]: combined belief vector [B, hidden_dim + stochastic_dim]
    """

    def __init__(
        self,
        visual_dim: int = 128,
        action_dim: int = 64,
        reward_dim: int = 16,
        hidden_dim: int = 256,
        stochastic_dim: int = 32,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.belief_dim = hidden_dim + stochastic_dim

        # Reward embedder
        self.reward_encoder = nn.Sequential(
            nn.Linear(1, reward_dim),
            nn.GELU(),
            nn.Linear(reward_dim, reward_dim),
        )

        # Deterministic context GRU cell
        # Input: [visual_pooled; action_emb; reward_emb; stochastic_z_prev]
        gru_in_dim = visual_dim + action_dim + reward_dim + stochastic_dim
        self.gru_cell = nn.GRUCell(gru_in_dim, hidden_dim)

        # Posterior network q_phi(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim),
        )

        # Prior network p_psi(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim),
        )

        # Initial learned priors
        self.init_h = nn.Parameter(torch.zeros(1, hidden_dim))
        self.init_z = nn.Parameter(torch.zeros(1, stochastic_dim))

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device if device is not None else self.init_h.device
        h = self.init_h.expand(batch_size, -1).to(dev)
        z = self.init_z.expand(batch_size, -1).to(dev)
        return h, z

    def step(
        self,
        e_t: torch.Tensor,
        a_prev_emb: torch.Tensor,
        r_prev: torch.Tensor,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        use_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, StochasticDistribution, StochasticDistribution]:
        """Advance one recurrent step.

        Args:
            e_t: pooled visual feature [B, visual_dim]
            a_prev_emb: embedded action [B, action_dim]
            r_prev: previous scalar reward [B, 1]
            h_prev: previous deterministic state [B, hidden_dim]
            z_prev: previous stochastic state [B, stochastic_dim]
            use_posterior: if True, sample z_t from posterior q(z|h, e); else from prior p(z|h)

        Returns:
            b_t: combined belief [B, hidden_dim + stochastic_dim]
            h_t: new deterministic context [B, hidden_dim]
            post_dist: posterior distribution q_phi(z_t | h_t, e_t)
            prior_dist: prior distribution p_psi(z_t | h_t)
        """
        r_emb = self.reward_encoder(r_prev)
        gru_input = torch.cat([e_t, a_prev_emb, r_emb, z_prev], dim=-1)
        h_t = self.gru_cell(gru_input, h_prev)

        # Compute prior
        prior_out = self.prior_net(h_t)
        mu_p, logvar_p = prior_out.chunk(2, dim=-1)
        prior_dist = StochasticDistribution(mu_p, logvar_p)

        # Compute posterior
        post_inp = torch.cat([h_t, e_t], dim=-1)
        post_out = self.posterior_net(post_inp)
        mu_q, logvar_q = post_out.chunk(2, dim=-1)
        post_dist = StochasticDistribution(mu_q, logvar_q)

        dist_to_sample = post_dist if use_posterior else prior_dist
        z_t = dist_to_sample.sample()
        b_t = torch.cat([h_t, z_t], dim=-1)

        return b_t, h_t, post_dist, prior_dist
