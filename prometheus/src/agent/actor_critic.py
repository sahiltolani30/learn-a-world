"""
src/agent/actor_critic.py

Latent Actor-Critic Networks for Crafter & Atari JEPA World Model.
Fulfills requirements:
- R3: Zero-shortcut policy operating strictly in JEPA latent representation space z in R^{latent_dim}.
- R4: Plug-and-play discrete action distribution supporting Crafter (17) and variable Atari action sets.
- R5: Ultra-fast CPU execution (<0.05 ms latency per step).
"""

from typing import Tuple, Optional, Union, Dict, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorPrior(nn.Module):
    """
    MLP Policy Prior pi_psi(a | z) mapping latent state z in R^{latent_dim} to discrete action logits.

    Architecture:
        Linear(latent_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU()
        Linear(hidden_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU()
        Linear(hidden_dim, num_actions)

    Features:
        - LayerNorm and GELU activations for smooth latent policy geometry.
        - Orthogonal weight initialization.
        - Controllable temperature scaling.
        - Supports batched (B, D) and unbatched (D,) inputs.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 17,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {temperature}")

        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small gain on final layer to produce near-uniform initial exploration
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=0.01)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        z: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute discrete action logits from latent state z.

        Args:
            z: Latent state tensor of shape (B, latent_dim) or (latent_dim,).
            temperature: Optional override for temperature scaling.

        Returns:
            logits: Action logits of shape (B, num_actions) or (num_actions,).
        """
        was_1d = False
        if z.dim() == 1:
            if z.size(0) != self.latent_dim:
                raise ValueError(
                    f"Expected latent state dim {self.latent_dim}, got {z.size(0)}"
                )
            z = z.unsqueeze(0)
            was_1d = True
        elif z.dim() == 2:
            if z.size(1) != self.latent_dim:
                raise ValueError(
                    f"Expected latent state dim {self.latent_dim}, got {z.size(1)}"
                )
        else:
            raise ValueError(
                f"Expected 1D (D,) or 2D (B, D) latent tensor, got shape {tuple(z.shape)}"
            )

        logits = self.net(z)

        temp = self.temperature if temperature is None else float(temperature)
        temp = max(temp, 1e-6)
        scaled_logits = logits / temp

        if was_1d:
            return scaled_logits.squeeze(0)
        return scaled_logits

    def get_dist(
        self,
        z: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Categorical:
        """
        Construct a Categorical distribution over discrete actions.

        Args:
            z: Latent state tensor (B, latent_dim) or (latent_dim,).
            temperature: Optional temperature scaling override.

        Returns:
            dist: Categorical distribution instance.
        """
        logits = self.forward(z, temperature=temperature)
        return Categorical(logits=logits)

    def get_probs(
        self,
        z: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute action probability distribution over discrete actions.

        Args:
            z: Latent state tensor (B, latent_dim) or (latent_dim,).
            temperature: Optional temperature scaling override.

        Returns:
            probs: Probability tensor of shape (B, num_actions) or (num_actions,).
        """
        logits = self.forward(z, temperature=temperature)
        return F.softmax(logits, dim=-1)

    def sample(
        self,
        z: torch.Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample an action or select the greedy mode from the policy distribution.

        Args:
            z: Latent state tensor (B, latent_dim) or (latent_dim,).
            deterministic: If True, returns argmax(logits). If False, samples from distribution.
            temperature: Optional temperature scaling override.

        Returns:
            action: Action tensor of shape (B,) or scalar integer tensor.
        """
        if deterministic:
            logits = self.forward(z, temperature=temperature)
            return torch.argmax(logits, dim=-1)
        else:
            dist = self.get_dist(z, temperature=temperature)
            return dist.sample()

    def evaluate_actions(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Args:
            z: Latent state tensor (B, latent_dim).
            actions: Action tensor of shape (B,) or (B, 1).
            temperature: Optional temperature scaling override.

        Returns:
            log_prob: Tensor of shape (B,).
            entropy: Tensor of shape (B,).
        """
        dist = self.get_dist(z, temperature=temperature)
        if actions.dim() == 2 and actions.size(1) == 1:
            actions = actions.squeeze(1)
        log_prob = dist.log_prob(actions.long())
        entropy = dist.entropy()
        return log_prob, entropy


class CriticHead(nn.Module):
    """
    Individual Critic MLP Q(z, a) mapping latent state to action values.

    Architecture:
        Linear(latent_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU()
        Linear(hidden_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU()
        Linear(hidden_dim, num_actions)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 17,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small gain on final layer
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=0.1)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Q-value for given latent state and action.

        Args:
            z: Latent state tensor of shape (B, latent_dim) or (latent_dim,).
            a: Optional action tensor.
               - None: returns all Q-values of shape (B, num_actions) or (num_actions,)
               - 1D integer tensor (B,): returns scalar Q-value per item, shape (B,)
               - 2D integer tensor (B, 1): returns shape (B, 1)
               - 2D float probability / one-hot tensor (B, num_actions): returns expected Q, shape (B, 1)

        Returns:
            Q-value tensor matching input action shape.
        """
        was_1d = False
        if z.dim() == 1:
            if z.size(0) != self.latent_dim:
                raise ValueError(
                    f"Expected latent state dim {self.latent_dim}, got {z.size(0)}"
                )
            z = z.unsqueeze(0)
            was_1d = True
        elif z.dim() == 2:
            if z.size(1) != self.latent_dim:
                raise ValueError(
                    f"Expected latent state dim {self.latent_dim}, got {z.size(1)}"
                )
        else:
            raise ValueError(
                f"Expected 1D (D,) or 2D (B, D) latent tensor, got shape {tuple(z.shape)}"
            )

        q_all = self.net(z)  # (B, num_actions)

        if a is None:
            if was_1d:
                return q_all.squeeze(0)
            return q_all

        # Evaluate specific action
        if was_1d:
            if isinstance(a, (int, float)):
                a = torch.tensor([int(a)], device=z.device, dtype=torch.long)
            elif isinstance(a, torch.Tensor) and a.dim() == 0:
                a = a.unsqueeze(0).long()
            elif isinstance(a, torch.Tensor) and a.dim() == 1 and a.numel() == 1:
                a = a.long()

        if a.dim() == 1:
            # Shape (B,)
            if a.size(0) != z.size(0):
                raise ValueError(
                    f"Action batch size {a.size(0)} does not match latent batch size {z.size(0)}"
                )
            q_selected = q_all.gather(1, a.long().unsqueeze(1)).squeeze(1)
            if was_1d:
                return q_selected.squeeze(0)
            return q_selected
        elif a.dim() == 2 and a.size(1) == 1:
            # Shape (B, 1)
            if a.size(0) != z.size(0):
                raise ValueError(
                    f"Action batch size {a.size(0)} does not match latent batch size {z.size(0)}"
                )
            return q_all.gather(1, a.long())
        elif a.dim() == 2 and a.size(1) == self.num_actions:
            # Shape (B, num_actions) expectation: sum_a pi(a) Q(s, a)
            if a.size(0) != z.size(0):
                raise ValueError(
                    f"Action batch size {a.size(0)} does not match latent batch size {z.size(0)}"
                )
            expected_q = (q_all * a.float()).sum(dim=-1, keepdim=True)
            if was_1d:
                return expected_q.squeeze(0)
            return expected_q
        else:
            raise ValueError(
                f"Unsupported action tensor shape {a.shape} for num_actions={self.num_actions}. "
                f"Expected (B,), (B, 1), or (B, {self.num_actions})."
            )


class TwinCritic(nn.Module):
    """
    Clipped Double-Q Critics Q_{omega_1}(z, a) and Q_{omega_2}(z, a).
    Maintains online networks and Polyak EMA target networks (tau=0.99).

    Attributes:
        q1_net: Online critic head 1.
        q2_net: Online critic head 2.
        target_q1_net: Detached target critic head 1.
        target_q2_net: Detached target critic head 2.
        tau: Default Polyak momentum coefficient for EMA target updates.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 17,
        hidden_dim: int = 256,
        tau: float = 0.99,
    ):
        super().__init__()
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")

        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.tau = float(tau)

        # Online critics
        self.q1_net = CriticHead(latent_dim, num_actions, hidden_dim)
        self.q2_net = CriticHead(latent_dim, num_actions, hidden_dim)

        # Target critics (permanently in eval mode with stop-gradient)
        self.target_q1_net = copy.deepcopy(self.q1_net)
        self.target_q2_net = copy.deepcopy(self.q2_net)
        for p in self.target_q1_net.parameters():
            p.requires_grad = False
        for p in self.target_q2_net.parameters():
            p.requires_grad = False
        self.target_q1_net.eval()
        self.target_q2_net.eval()

    def forward(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of both online critic heads.

        Args:
            z: Latent state tensor (B, latent_dim).
            a: Optional action tensor.

        Returns:
            q1, q2: Q-values from both critic heads.
        """
        q1 = self.q1_net(z, a)
        q2 = self.q2_net(z, a)
        return q1, q2

    def q1(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate only online critic 1."""
        return self.q1_net(z, a)

    def q2(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate only online critic 2."""
        return self.q2_net(z, a)

    def q_min(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute clipped double-Q minimum: min(Q_1(z, a), Q_2(z, a)).
        """
        q1, q2 = self.forward(z, a)
        return torch.min(q1, q2)

    def target_forward(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of both target critic heads strictly under torch.no_grad().

        Args:
            z: Latent state tensor (B, latent_dim).
            a: Optional action tensor.

        Returns:
            target_q1, target_q2: Detached Q-values from target heads.
        """
        with torch.no_grad():
            t_q1 = self.target_q1_net(z, a).detach()
            t_q2 = self.target_q2_net(z, a).detach()
            return t_q1, t_q2

    def target_q_min(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute clipped target double-Q minimum strictly under torch.no_grad():
            min(Target_Q_1(z, a), Target_Q_2(z, a)).
        """
        t_q1, t_q2 = self.target_forward(z, a)
        return torch.min(t_q1, t_q2)

    def update_target(self, tau: Optional[float] = None) -> None:
        """
        Polyak EMA update:
            theta_target = tau * theta_target + (1 - tau) * theta_online

        Args:
            tau: Optional override for momentum coefficient. If None, uses self.tau.
        """
        t = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for p_t, p_o in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
                p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)
            for p_t, p_o in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
                p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)

            for b_t, b_o in zip(self.target_q1_net.buffers(), self.q1_net.buffers()):
                if b_t.is_floating_point():
                    b_t.data.mul_(t).add_(b_o.data, alpha=1.0 - t)
                else:
                    b_t.data.copy_(b_o.data)
            for b_t, b_o in zip(self.target_q2_net.buffers(), self.q2_net.buffers()):
                if b_t.is_floating_point():
                    b_t.data.mul_(t).add_(b_o.data, alpha=1.0 - t)
                else:
                    b_t.data.copy_(b_o.data)

    def hard_update(self) -> None:
        """
        Hard synchronization: theta_target = theta_online.
        Used at model initialization or checkpoint restoration.
        """
        with torch.no_grad():
            for p_t, p_o in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
                p_t.data.copy_(p_o.data)
            for p_t, p_o in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
                p_t.data.copy_(p_o.data)

            for b_t, b_o in zip(self.target_q1_net.buffers(), self.q1_net.buffers()):
                b_t.data.copy_(b_o.data)
            for b_t, b_o in zip(self.target_q2_net.buffers(), self.q2_net.buffers()):
                b_t.data.copy_(b_o.data)

    def train(self, mode: bool = True):
        """
        Override train mode to ensure target networks permanently stay in eval mode.
        """
        super().train(mode)
        self.target_q1_net.eval()
        self.target_q2_net.eval()
        return self

    def compute_loss(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        target_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MSE Bellman loss for both online critic heads.

        Args:
            z: Latent state tensor (B, latent_dim).
            a: Action tensor (B,) or (B, 1).
            target_q: Target Q-values (B,) or (B, 1).

        Returns:
            total_loss: Scalar loss tensor.
            loss_dict: Dictionary with individual head losses.
        """
        q1 = self.q1(z, a)
        q2 = self.q2(z, a)

        # Match shapes
        if target_q.dim() == 1 and q1.dim() == 2:
            target_q = target_q.unsqueeze(1)
        elif target_q.dim() == 2 and q1.dim() == 1:
            q1 = q1.unsqueeze(1)
            q2 = q2.unsqueeze(1)

        loss1 = F.mse_loss(q1, target_q.detach())
        loss2 = F.mse_loss(q2, target_q.detach())
        total_loss = loss1 + loss2

        return total_loss, {
            "loss_critic": total_loss,
            "loss_q1": loss1,
            "loss_q2": loss2,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
        }


class TargetCritic(nn.Module):
    """
    Dedicated Target Critic wrapper with Polyak EMA tracking and stop-gradient semantics.
    Provides identical interface to TargetEncoder for architectural symmetry.
    """

    def __init__(self, critic: TwinCritic, tau: float = 0.99):
        super().__init__()
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")

        object.__setattr__(self, "online_critic", critic)
        self.tau = float(tau)

        self.target_critic = copy.deepcopy(critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.target_critic.eval()

    def forward(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.target_critic(z, a)

    def q_min(
        self,
        z: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.target_critic.q_min(z, a)

    def update(self, tau: Optional[float] = None) -> None:
        t = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for p_t, p_o in zip(self.target_critic.parameters(), self.online_critic.parameters()):
                p_t.data.mul_(t).add_(p_o.data, alpha=1.0 - t)
            for b_t, b_o in zip(self.target_critic.buffers(), self.online_critic.buffers()):
                if b_t.is_floating_point():
                    b_t.data.mul_(t).add_(b_o.data, alpha=1.0 - t)
                else:
                    b_t.data.copy_(b_o.data)

    def hard_update(self) -> None:
        with torch.no_grad():
            for p_t, p_o in zip(self.target_critic.parameters(), self.online_critic.parameters()):
                p_t.data.copy_(p_o.data)
            for b_t, b_o in zip(self.target_critic.buffers(), self.online_critic.buffers()):
                b_t.data.copy_(b_o.data)

    def train(self, mode: bool = True):
        super().train(False)
        self.target_critic.eval()
        return self
