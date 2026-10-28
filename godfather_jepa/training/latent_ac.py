"""
Latent Actor-Critic with Multi-Step Lambda-Returns for Godfather JEPA.

Computes Generalized Advantage / lambda-returns (lambda=0.95, gamma=0.997)
over imagined latent rollouts to assign credit backward across 60-100 steps
from stone mining and tool crafting milestones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    lambd: float = 0.95,
) -> torch.Tensor:
    """
    Computes Generalized Advantage / multi-step lambda-returns over an [H, B] sequence:
        G_t^lambda = r_t + gamma_t * ((1 - lambda) * v_{t+1} + lambda * G_{t+1}^lambda)

    Args:
        rewards: Tensor of shape (H, B) containing immediate step rewards.
        values: Tensor of shape (H, B) or (H+1, B) containing estimated state values.
        discounts: Tensor of shape (H, B) containing state-dependent discount factors.
        lambd: Lambda weighting factor (default 0.95).

    Returns:
        returns: Tensor of shape (H, B) containing bootstrapped multi-step targets.
    """
    h, b = rewards.shape
    returns = torch.zeros_like(rewards)
    next_ret = values[-1]

    for t in reversed(range(h)):
        next_val = values[t + 1] if t + 1 < values.shape[0] else values[-1]
        target = (1.0 - lambd) * next_val + lambd * next_ret
        returns[t] = rewards[t] + discounts[t] * target
        next_ret = returns[t]

    return returns


class LatentActorCriticTrainer:
    """
    Coordinates policy gradient updates and symlog categorical value fitting
    using trajectories unrolled inside the learned JEPA world model.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        coder: Any,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        lambd: float = 0.95,
        base_gamma: float = 0.997,
    ):
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.coder = coder
        self.lambd = lambd
        self.base_gamma = base_gamma

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def train_step(
        self,
        imagined_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Executes one actor-critic training step over imagined rollout [H, B, D].
        """
        h, b, d = imagined_states.shape

        # 1. Compute target critic values under slow EMA target network
        with torch.no_grad():
            target_logits = self.target_critic(imagined_states)
            target_vals = self.coder.decode(target_logits)
            lambda_targets = compute_lambda_returns(rewards, target_vals, discounts, lambd=self.lambd)

        # 2. Update categorical critic
        critic_logits = self.critic(imagined_states)
        critic_loss = self.coder.loss(
            critic_logits.reshape(-1, self.coder.num_bins),
            lambda_targets.reshape(-1),
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # 3. Update policy actor via REINFORCE / Advantage
        with torch.no_grad():
            baseline_vals = self.coder.decode(self.critic(imagined_states))
            advantages = lambda_targets - baseline_vals

        log_probs = self.actor(imagined_states)  # (H, B, 17)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        actor_loss = -(action_log_probs * advantages).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return actor_loss.item(), critic_loss.item()
