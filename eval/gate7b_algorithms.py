"""Gate 7B Algorithmic Formulations: IQL, Discrete CQL, CRR & Distributional Value.

Strictly follows published literature references:
- IQL: Kostrikov, Nair and Levine, 2021 (Expectile V-learning, Bellman Q-learning, AWR actor extraction).
- Discrete CQL: Kumar et al., NeurIPS 2020 (Conservative Q-learning with logsumexp penalty).
- CRR: Wang et al., NeurIPS 2020 (Critic Regularized Regression with binary advantage filtering).
- Quantile Value: Dabney et al., 2018 (Quantile Huber regression).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from eval.gate7b_models import (
    DiscreteQCritic,
    DistributionalValueCritic,
    ExpectileValueCritic,
)
from true_jepa.model import LatentFrame, TrueJEPA


# ---------------------------------------------------------------------------
# 1. Pure Mathematical Loss Fixtures & Helpers
# ---------------------------------------------------------------------------

def expectile_loss(diff: Tensor, tau: float = 0.7) -> Tensor:
    """Asymmetric L2 expectile loss for IQL value learning.
    
    L2^tau(u) = |tau - I(u < 0)| * u^2
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return weight * (diff ** 2)


def cql_conservative_penalty(q_all: Tensor, actions: Tensor) -> Tensor:
    """Discrete CQL conservative regularizer.
    
    R_CQL(Q) = logsumexp(Q(s, :)) - Q(s, a_real)
    Pushes down unvisited action-values and pushes up real dataset actions.
    """
    logsumexp_q = torch.logsumexp(q_all, dim=-1)
    q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logsumexp_q - q_taken


def quantile_huber_loss(
    quantiles_pred: Tensor,
    targets: Tensor,
    tau_hats: Tensor,
    kappa: float = 1.0,
) -> Tensor:
    """Quantile Huber loss for distributional return learning.
    
    Args:
        quantiles_pred: [B, N] predicted quantiles
        targets: [B, 1] or [B] target returns
        tau_hats: [N] quantile midpoints
        kappa: Huber threshold
    """
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)  # [B, 1]
    
    # diff: [B, N] -> (targets - quantiles)
    diff = targets - quantiles_pred
    abs_diff = torch.abs(diff)
    huber = torch.where(abs_diff <= kappa, 0.5 * (diff ** 2), kappa * (abs_diff - 0.5 * kappa))
    
    weight = torch.abs(tau_hats.unsqueeze(0) - (diff < 0).float())
    return (weight * huber).sum(dim=-1).mean()


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    """Polyak averaging: target = (1 - tau) * target + tau * source."""
    with torch.no_grad():
        for target_p, source_p in zip(target_net.parameters(), source_net.parameters()):
            target_p.data.copy_((1.0 - tau) * target_p.data + tau * source_p.data)


# ---------------------------------------------------------------------------
# 2. Candidate B: Implicit Q-Learning (IQL) Step
# ---------------------------------------------------------------------------

def iql_training_step(
    model: TrueJEPA,
    q_critic: DiscreteQCritic,
    v_critic: ExpectileValueCritic,
    v_target: ExpectileValueCritic,
    actor_head: nn.Module,
    batch: Dict[str, Tensor],
    q_optimizer: torch.optim.Optimizer,
    v_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    expectile: float = 0.7,
    inv_temp: float = 3.0,
    max_weight: float = 20.0,
    entropy_weight: float = 0.01,
    polyak_tau: float = 0.005,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """One offline IQL training step.
    
    1. V-loss: Expectile regression on min(Q1, Q2) - V(s).
    2. Q-loss: Bellman regression to r + gamma * (1 - d) * V_target(s').
    3. Actor-loss: AWR advantage extraction: w = clamp(exp(beta * (Q - V)), 0, max_w).
    """
    q_optimizer.zero_grad(set_to_none=True)
    v_optimizer.zero_grad(set_to_none=True)
    actor_optimizer.zero_grad(set_to_none=True)

    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    terminated = batch["terminated"].to(device)
    valid = batch["valid"].to(device).float()

    B, T = actions.shape

    # Reconstruct states under frozen world model
    with torch.no_grad():
        if obs.ndim == 5:
            flat_obs = obs.reshape(-1, *obs.shape[2:])
            frames = model.encode(flat_obs)
            tokens_all = frames.tokens.reshape(B, T + 1, *frames.tokens.shape[1:])
            pooled_all = frames.pooled.reshape(B, T + 1, *frames.pooled.shape[1:])
        elif obs.ndim == 4:
            flat_tokens = obs.reshape(-1, *obs.shape[2:])
            frames = model.encoder.adapter(flat_tokens)
            tokens_all = frames[0].reshape(B, T + 1, *frames[0].shape[1:])
            pooled_all = frames[1].reshape(B, T + 1, *frames[1].shape[1:])
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

    # Unroll recurrent beliefs
    hidden = model.initial_belief(B, device)
    states_list = []
    with torch.no_grad():
        for t in range(T + 1):
            frame_t = LatentFrame(tokens=tokens_all[:, t], pooled=pooled_all[:, t])
            state_t = model.heads.state(frame_t, hidden)
            states_list.append(state_t)
            if t < T:
                hidden = model.belief_step(frame_t, actions[:, t], rewards[:, t], hidden)

    # Flatten valid transitions for batch optimization
    valid_mask = valid.bool()
    states_t = torch.stack(states_list[:-1], dim=1)[valid_mask]      # [N, 448]
    states_next = torch.stack(states_list[1:], dim=1)[valid_mask]    # [N, 448]
    act_t = actions[valid_mask].long()                               # [N]
    rew_t = rewards[valid_mask].float()                              # [N]
    term_t = terminated[valid_mask].float()                          # [N]

    # --- 1. Expectile Value Update ---
    with torch.no_grad():
        q1_target, q2_target = q_critic.evaluate_actions(states_t, act_t)
        target_q = torch.min(q1_target, q2_target)

    v_pred = v_critic(states_t)
    v_diff = target_q - v_pred
    v_loss = expectile_loss(v_diff, tau=expectile).mean()
    v_loss.backward()
    v_optimizer.step()

    # --- 2. Q-Critic Update ---
    with torch.no_grad():
        next_v = v_target(states_next)
        q_target = rew_t + gamma * (1.0 - term_t) * next_v
        q_target = torch.clamp(q_target, -10.0, 10.0)

    q1_pred, q2_pred = q_critic.evaluate_actions(states_t, act_t)
    q_loss = 0.5 * (F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target))
    q_loss.backward()
    q_optimizer.step()

    # --- 3. Actor Update (Advantage Extraction) ---
    with torch.no_grad():
        adv = target_q - v_pred
        adv_weights = torch.clamp(torch.exp(adv * inv_temp), 0.0, max_weight)

    logits = actor_head(states_t)
    dist = torch.distributions.Categorical(logits=logits)
    log_prob = dist.log_prob(act_t)
    entropy = dist.entropy().mean()

    actor_loss = - (adv_weights * log_prob).mean() - entropy_weight * entropy
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target value network
    soft_update(v_target, v_critic, tau=polyak_tau)

    return {
        "v_loss": float(v_loss.item()),
        "q_loss": float(q_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "mean_v": float(v_pred.mean().item()),
        "mean_q": float(target_q.mean().item()),
        "mean_adv": float(adv.mean().item()),
        "mean_adv_weight": float(adv_weights.mean().item()),
        "entropy": float(entropy.item()),
    }


# ---------------------------------------------------------------------------
# 3. Candidate C: Discrete Conservative Q-Learning (CQL) Step
# ---------------------------------------------------------------------------

def discrete_cql_training_step(
    model: TrueJEPA,
    q_critic: DiscreteQCritic,
    q_target: DiscreteQCritic,
    actor_head: nn.Module,
    batch: Dict[str, Tensor],
    q_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    cql_alpha: float = 1.0,
    entropy_weight: float = 0.01,
    polyak_tau: float = 0.005,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """One offline Discrete CQL training step.
    
    1. Q-loss: Bellman MSE + cql_alpha * (logsumexp(Q) - Q(a_real)).
    2. Actor-loss: Distillation to conservative softmax Q-distribution with entropy.
    """
    q_optimizer.zero_grad(set_to_none=True)
    actor_optimizer.zero_grad(set_to_none=True)

    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    terminated = batch["terminated"].to(device)
    valid = batch["valid"].to(device).float()

    B, T = actions.shape

    with torch.no_grad():
        if obs.ndim == 5:
            flat_obs = obs.reshape(-1, *obs.shape[2:])
            frames = model.encode(flat_obs)
            tokens_all = frames.tokens.reshape(B, T + 1, *frames.tokens.shape[1:])
            pooled_all = frames.pooled.reshape(B, T + 1, *frames.pooled.shape[1:])
        elif obs.ndim == 4:
            flat_tokens = obs.reshape(-1, *obs.shape[2:])
            frames = model.encoder.adapter(flat_tokens)
            tokens_all = frames[0].reshape(B, T + 1, *frames[0].shape[1:])
            pooled_all = frames[1].reshape(B, T + 1, *frames[1].shape[1:])
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

    hidden = model.initial_belief(B, device)
    states_list = []
    with torch.no_grad():
        for t in range(T + 1):
            frame_t = LatentFrame(tokens=tokens_all[:, t], pooled=pooled_all[:, t])
            state_t = model.heads.state(frame_t, hidden)
            states_list.append(state_t)
            if t < T:
                hidden = model.belief_step(frame_t, actions[:, t], rewards[:, t], hidden)

    valid_mask = valid.bool()
    states_t = torch.stack(states_list[:-1], dim=1)[valid_mask]
    states_next = torch.stack(states_list[1:], dim=1)[valid_mask]
    act_t = actions[valid_mask].long()
    rew_t = rewards[valid_mask].float()
    term_t = terminated[valid_mask].float()

    # --- 1. Q-Critic Bellman & Conservative Loss ---
    with torch.no_grad():
        q1_next_all, q2_next_all = q_target(states_next)
        min_q_next = torch.min(q1_next_all, q2_next_all)
        max_q_next = min_q_next.max(dim=-1)[0]
        y_target = rew_t + gamma * (1.0 - term_t) * max_q_next
        y_target = torch.clamp(y_target, -10.0, 10.0)

    q1_all, q2_all = q_critic(states_t)
    q1_a = q1_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)
    q2_a = q2_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)

    bellman_loss = 0.5 * (F.mse_loss(q1_a, y_target) + F.mse_loss(q2_a, y_target))
    cql_penalty = 0.5 * (
        cql_conservative_penalty(q1_all, act_t).mean() +
        cql_conservative_penalty(q2_all, act_t).mean()
    )
    total_q_loss = bellman_loss + cql_alpha * cql_penalty
    total_q_loss.backward()
    q_optimizer.step()

    # --- 2. Actor Policy Extraction ---
    with torch.no_grad():
        min_q = torch.min(q1_all.detach(), q2_all.detach())
        # Softmax Q-distribution target
        target_dist = F.softmax(min_q / 1.0, dim=-1)

    logits = actor_head(states_t)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_dist = torch.distributions.Categorical(logits=logits)
    entropy = policy_dist.entropy().mean()

    # Cross-entropy / KL to target softmax distribution
    policy_loss = - (target_dist * log_probs).sum(dim=-1).mean() - entropy_weight * entropy
    policy_loss.backward()
    actor_optimizer.step()

    soft_update(q_target, q_critic, tau=polyak_tau)

    return {
        "bellman_loss": float(bellman_loss.item()),
        "cql_penalty": float(cql_penalty.item()),
        "total_q_loss": float(total_q_loss.item()),
        "actor_loss": float(policy_loss.item()),
        "mean_q": float(0.5 * (q1_a.mean() + q2_a.mean()).item()),
        "entropy": float(entropy.item()),
    }


# ---------------------------------------------------------------------------
# 4. Candidate D: Critic Regularized Regression (CRR - Binary Filtered) Step
# ---------------------------------------------------------------------------

def crr_training_step(
    model: TrueJEPA,
    q_critic: DiscreteQCritic,
    q_target: DiscreteQCritic,
    actor_head: nn.Module,
    batch: Dict[str, Tensor],
    q_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    entropy_weight: float = 0.01,
    polyak_tau: float = 0.005,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """One offline CRR training step using Binary Indicator policy filtering.
    
    1. Critic: Bellman TD update using expected next-state value under policy.
    2. Advantage: A(s, a) = Q(s, a) - V^pi(s).
    3. Actor: Only update policy on transitions where A(s, a) > 0 (Indicator Filter).
    """
    q_optimizer.zero_grad(set_to_none=True)
    actor_optimizer.zero_grad(set_to_none=True)

    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    rewards = batch["rewards"].to(device)
    terminated = batch["terminated"].to(device)
    valid = batch["valid"].to(device).float()

    B, T = actions.shape

    with torch.no_grad():
        if obs.ndim == 5:
            flat_obs = obs.reshape(-1, *obs.shape[2:])
            frames = model.encode(flat_obs)
            tokens_all = frames.tokens.reshape(B, T + 1, *frames.tokens.shape[1:])
            pooled_all = frames.pooled.reshape(B, T + 1, *frames.pooled.shape[1:])
        elif obs.ndim == 4:
            flat_tokens = obs.reshape(-1, *obs.shape[2:])
            frames = model.encoder.adapter(flat_tokens)
            tokens_all = frames[0].reshape(B, T + 1, *frames[0].shape[1:])
            pooled_all = frames[1].reshape(B, T + 1, *frames[1].shape[1:])
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

    hidden = model.initial_belief(B, device)
    states_list = []
    with torch.no_grad():
        for t in range(T + 1):
            frame_t = LatentFrame(tokens=tokens_all[:, t], pooled=pooled_all[:, t])
            state_t = model.heads.state(frame_t, hidden)
            states_list.append(state_t)
            if t < T:
                hidden = model.belief_step(frame_t, actions[:, t], rewards[:, t], hidden)

    valid_mask = valid.bool()
    states_t = torch.stack(states_list[:-1], dim=1)[valid_mask]
    states_next = torch.stack(states_list[1:], dim=1)[valid_mask]
    act_t = actions[valid_mask].long()
    rew_t = rewards[valid_mask].float()
    term_t = terminated[valid_mask].float()

    # --- 1. Expected SARSA Critic Update ---
    with torch.no_grad():
        next_logits = actor_head(states_next)
        next_probs = F.softmax(next_logits, dim=-1)
        q1_next_all, q2_next_all = q_target(states_next)
        min_q_next = torch.min(q1_next_all, q2_next_all)
        v_next_pi = (next_probs * min_q_next).sum(dim=-1)
        y_target = rew_t + gamma * (1.0 - term_t) * v_next_pi
        y_target = torch.clamp(y_target, -10.0, 10.0)

    q1_a, q2_a = q_critic.evaluate_actions(states_t, act_t)
    q_loss = 0.5 * (F.mse_loss(q1_a, y_target) + F.mse_loss(q2_a, y_target))
    q_loss.backward()
    q_optimizer.step()

    # --- 2. CRR Binary Filtered Actor Update ---
    with torch.no_grad():
        q1_all, q2_all = q_critic(states_t)
        min_q = torch.min(q1_all, q2_all)
        cur_logits = actor_head(states_t)
        cur_probs = F.softmax(cur_logits, dim=-1)
        v_pi = (cur_probs * min_q).sum(dim=-1)
        q_taken = min_q.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)
        advantage = q_taken - v_pi
        # Binary indicator filter: w = 1.0 if A > 0, else 0.0
        binary_weights = (advantage > 0.0).float()

    logits = actor_head(states_t)
    dist = torch.distributions.Categorical(logits=logits)
    log_prob = dist.log_prob(act_t)
    entropy = dist.entropy().mean()

    weight_sum = binary_weights.sum().clamp_min(1.0)
    actor_loss = - (binary_weights * log_prob).sum() / weight_sum - entropy_weight * entropy
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(q_target, q_critic, tau=polyak_tau)

    return {
        "q_loss": float(q_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "mean_adv": float(advantage.mean().item()),
        "filter_pass_ratio": float(binary_weights.mean().item()),
        "entropy": float(entropy.item()),
    }
