"""JEPA and imagination objectives for the clean model."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from true_jepa.config import TrainConfig
from true_jepa.model import LatentFrame, TrueJEPA


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def variance_penalty(values: Tensor, floor: float = 1.0) -> Tensor:
    """VICReg-style anti-collapse term over [B,T,D] values."""

    flat = values.reshape(-1, values.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(floor - std).mean()


def covariance_penalty(values: Tensor) -> Tensor:
    flat = values.reshape(-1, values.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().mean()


def train_world_model_step(
    model: TrueJEPA,
    batch: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
    update: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    """Perform one closed-loop JEPA update on an observable replay batch."""

    obs = batch["obs"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    rewards = batch["rewards"].to(device, non_blocking=True)
    terminated = batch["terminated"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    batch_size, sequence_length = actions.shape
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)

    optimizer.zero_grad(set_to_none=True)
    with autocast:
        current = model.encode(obs[:, 0])
        hidden = model.initial_belief(batch_size, device)
        previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        previous_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
        jepa_terms = []
        reward_terms = []
        continue_terms = []
        ensemble_terms = []
        predicted_pooled = []

        for time in range(sequence_length):
            if time > 0:
                hidden = model.belief_step(current, previous_action, previous_reward, hidden)
            action = actions[:, time]
            reward_prediction, continuation_logit = model.transition_heads(current, hidden, action)
            predicted = model.predict(current, action, hidden)
            target = model.target_encode(obs[:, time + 1])
            step_mask = valid[:, time]
            jepa_terms.append(
                _masked_mean(
                    (predicted.tokens - target.tokens.detach()).square().mean(dim=(1, 2)),
                    step_mask,
                )
                + _masked_mean(
                    (predicted.pooled - target.pooled.detach()).square().mean(dim=-1),
                    step_mask,
                )
            )
            reward_terms.append(_masked_mean(F.smooth_l1_loss(reward_prediction, rewards[:, time], reduction="none"), step_mask))
            continue_terms.append(
                _masked_mean(
                    F.binary_cross_entropy_with_logits(
                        continuation_logit, (~terminated[:, time]).float(), reduction="none"
                    ),
                    step_mask,
                )
            )
            ensemble = model.ensemble(current, action, hidden)
            ensemble_target = target.pooled.detach().unsqueeze(0)
            ensemble_terms.append(
                _masked_mean((ensemble - ensemble_target).square().mean(dim=-1).mean(dim=0), step_mask)
            )
            predicted_pooled.append(predicted.pooled)
            current = predicted
            previous_action = action
            previous_reward = rewards[:, time]

        pooled = torch.stack(predicted_pooled, dim=1)
        jepa = torch.stack(jepa_terms).mean()
        reward_loss = torch.stack(reward_terms).mean()
        continue_loss = torch.stack(continue_terms).mean()
        ensemble_loss = torch.stack(ensemble_terms).mean()
        total = (
            cfg.jepa_weight * jepa
            + variance_penalty(pooled)
            + 0.02 * covariance_penalty(pooled)
            + cfg.reward_weight * reward_loss
            + cfg.continuation_weight * continue_loss
            + cfg.ensemble_weight * ensemble_loss
        )

    if scaler is not None and amp_enabled:
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    model.update_target(update)
    return {
        "loss": float(total.detach().float().cpu()),
        "jepa": float(jepa.detach().float().cpu()),
        "reward": float(reward_loss.detach().float().cpu()),
        "continuation": float(continue_loss.detach().float().cpu()),
        "ensemble": float(ensemble_loss.detach().float().cpu()),
        "latent_std": float(pooled.detach().float().std().cpu()),
    }


def imagination_policy_step(
    model: TrueJEPA,
    start: LatentFrame,
    hidden: Tensor,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    horizon: int,
    gamma: float,
    entropy_weight: float,
) -> Dict[str, float]:
    """Train policy/value heads using model-predicted consequences.

    The latent dynamics are detached during this update.  The actor learns from
    the score-function gradient through its sampled actions, while the world
    model remains a self-supervised JEPA rather than a policy-loss casualty.
    """

    actor_optimizer.zero_grad(set_to_none=True)
    value_optimizer.zero_grad(set_to_none=True)
    frame = LatentFrame(start.tokens.detach(), start.pooled.detach())
    belief = hidden.detach()
    log_probs = []
    entropies = []
    rewards = []
    values = []
    for _ in range(horizon):
        state = model.state(frame, belief)
        logits = model.policy_logits(frame, belief)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        values.append(model.heads.value(state).squeeze(-1))
        with torch.no_grad():
            reward, _continuation = model.transition_heads(frame, belief, action)
            next_frame = model.predict(frame, action, belief)
            next_belief = model.belief_step(next_frame, action, reward, belief)
        rewards.append(reward)
        frame = LatentFrame(next_frame.tokens.detach(), next_frame.pooled.detach())
        belief = next_belief.detach()
    reward_stack = torch.stack(rewards)
    returns = torch.zeros_like(reward_stack)
    running = torch.zeros_like(reward_stack[-1])
    for index in reversed(range(horizon)):
        running = reward_stack[index] + gamma * running
        returns[index] = running
    log_prob_stack = torch.stack(log_probs)
    entropy_stack = torch.stack(entropies)
    value_stack = torch.stack(values)
    actor_loss = -((returns.detach() - value_stack.detach()) * log_prob_stack).mean() - entropy_weight * entropy_stack.mean()
    value_loss = F.smooth_l1_loss(value_stack, returns.detach())
    actor_loss.backward()
    value_loss.backward()
    actor_optimizer.step()
    value_optimizer.step()
    return {"actor": float(actor_loss.detach()), "value": float(value_loss.detach())}
