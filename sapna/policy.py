"""
sapna/policy.py

Actor, Critic, SlowCritic, unimix, symlog, symexp, and SAPNAAgent facade per SAPNA_PLAN.md §2.2, §5.1, §9.
"""

from typing import Optional, Tuple, Union
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sapna.imagination import twohot_score, twohot_ce, twohot_encode


def symlog(x: Union[torch.Tensor, float, np.ndarray]) -> Union[torch.Tensor, float, np.ndarray]:
    """
    Symlog function (DreamerV3): symlog(x) = sign(x) * ln(1 + |x|).
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    elif isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    else:
        return (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)) * math.log1p(abs(x))


def symexp(y: Union[torch.Tensor, float, np.ndarray]) -> Union[torch.Tensor, float, np.ndarray]:
    """
    Inverse of symlog: symexp(y) = sign(y) * (exp(|y|) - 1).
    """
    if isinstance(y, torch.Tensor):
        return torch.sign(y) * torch.expm1(torch.abs(y))
    elif isinstance(y, np.ndarray):
        return np.sign(y) * np.expm1(np.abs(y))
    else:
        return (1.0 if y > 0 else (-1.0 if y < 0 else 0.0)) * math.expm1(abs(y))


def unimix(probs: torch.Tensor, unimix_ratio: float = 0.01) -> torch.Tensor:
    """
    Action unimixing (DreamerV3): probs = (1 - unimix) * probs + unimix / num_actions.
    """
    num_actions = probs.size(-1)
    return (1.0 - unimix_ratio) * probs + unimix_ratio / float(num_actions)


class Actor(nn.Module):
    """
    Actor policy network: Linear(512, 256) -> ELU -> Linear(256, 256) -> ELU -> Linear(256, 17).
    Orthogonal init, final layer gain 0.01.
    """

    def __init__(self, in_features: int = 512, hidden_features: int = 256, num_actions: int = 17):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act2 = nn.ELU()
        self.out = nn.Linear(hidden_features, num_actions)

        # Orthogonal initialization: hidden gain = sqrt(2), final layer gain = 0.01
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits of shape (B, num_actions)."""
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        return self.out(h)

    def get_dist(self, x: torch.Tensor, unimix_ratio: float = 0.01) -> Tuple[torch.Tensor, Categorical]:
        """Returns (unimixed_probs, Categorical distribution)."""
        logits = self.forward(x)
        raw_probs = torch.softmax(logits, dim=-1)
        probs = unimix(raw_probs, unimix_ratio=unimix_ratio)
        return probs, Categorical(probs=probs)


class Critic(nn.Module):
    """
    Distributional Critic network: Linear(512, 256) -> ELU -> Linear(256, 256) -> ELU -> Linear(256, 255).
    """

    def __init__(self, in_features: int = 512, hidden_features: int = 256, num_bins: int = 255):
        super().__init__()
        self.num_bins = num_bins
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act2 = nn.ELU()
        self.out = nn.Linear(hidden_features, num_bins)

        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.out.weight, gain=1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns two-hot distribution logits of shape (B, num_bins)."""
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        return self.out(h)


class SlowCritic(nn.Module):
    """
    Target SlowCritic with EMA update and permanent no_grad.
    """

    def __init__(self, critic: Critic, tau: float = 0.98):
        super().__init__()
        self.tau = float(tau)
        object.__setattr__(self, "online_critic", critic)

        self.critic = copy.deepcopy(critic)
        for param in self.critic.parameters():
            param.requires_grad = False
        self.critic.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.critic(x)

    def update(self, tau: Optional[float] = None) -> None:
        t = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for p_t, p_o in zip(self.critic.parameters(), self.online_critic.parameters()):
                p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)
            for b_t, b_o in zip(self.critic.buffers(), self.online_critic.buffers()):
                b_t.data.mul_(t).add_(b_o.data, alpha=1.0 - t)

    def ema_update(self, tau: Optional[float] = None) -> None:
        self.update(tau)

    def train(self, mode: bool = True):
        super().train(False)
        self.critic.eval()
        return self


class SAPNAAgent:
    """
    Unified agent facade bundling encoder, temporal core, policy, and action embedding.
    Single interface for evaluate() and benchmark.py per SAPNA_PLAN.md §9.
    """

    def __init__(
        self,
        encoder: nn.Module,
        temporal_core: nn.Module,
        action_embedding: nn.Module,
        actor: nn.Module,
        critic: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        unimix_ratio: float = 0.01,
    ):
        self.encoder = encoder
        self.temporal_core = temporal_core
        self.action_embedding = action_embedding
        self.actor = actor
        self.critic = critic
        self.device = device if device is not None else next(encoder.parameters()).device
        self.unimix_ratio = unimix_ratio

    def reset_h(self, batch_size: int = 1) -> torch.Tensor:
        """Reset hidden state to zeros (B, 256)."""
        return torch.zeros(batch_size, 256, device=self.device)

    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        h: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[Union[int, np.ndarray], torch.Tensor, torch.Tensor]:
        """
        Takes one action step given raw visual observation and current recurrent state h.
        Returns:
            action: int (or numpy array if batched)
            h_next: recurrent state after the transition
            z: pooled observation latent state (256-d)
        """
        unbatched = False
        if isinstance(obs, np.ndarray):
            if obs.ndim == 3:
                unbatched = True
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            else:
                obs_t = torch.from_numpy(obs).to(self.device)
        elif isinstance(obs, torch.Tensor):
            if obs.dim() == 3:
                unbatched = True
                obs_t = obs.unsqueeze(0).to(self.device)
            else:
                obs_t = obs.to(self.device)
        else:
            raise TypeError(f"Unsupported obs type: {type(obs)}")

        if obs_t.dtype == torch.uint8:
            obs_t = obs_t.float() / 255.0

        if h.dim() == 1:
            h_t = h.unsqueeze(0).to(self.device)
        else:
            h_t = h.to(self.device)

        with torch.no_grad():
            z = self.encoder(obs_t)  # (B, 256)
            state = torch.cat([z, h_t], dim=-1)  # (B, 512)
            logits = self.actor(state)
            raw_probs = torch.softmax(logits, dim=-1)
            probs = unimix(raw_probs, unimix_ratio=self.unimix_ratio)

            if greedy:
                action_t = probs.argmax(dim=-1)
            else:
                dist = Categorical(probs=probs)
                action_t = dist.sample()

            a_emb = self.action_embedding(action_t)
            h_next = self.temporal_core(z, h_t, a_emb=a_emb)

        if unbatched:
            action_out = int(action_t.item())
            h_next_out = h_next.squeeze(0)
            z_out = z.squeeze(0)
        else:
            action_out = action_t.cpu().numpy()
            h_next_out = h_next
            z_out = z

        return action_out, h_next_out, z_out
