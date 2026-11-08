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


@torch.no_grad()
def compute_spatial_change_weights(
    current_tokens: Tensor,
    target_tokens: Tensor,
    cap: float = 5.0,
    bonus: float = 2.0,
) -> Tuple[Tensor, float]:
    """Compute normalized and capped spatial change weights [B, K] from detached target transitions.

    Weights tokens by empirical target displacement ||z*(t+1) - z(t)||_2 so dynamic regions
    contribute higher loss gradients while static background is downweighted.

    Args:
        current_tokens: [B, K, D] current representation tokens
        target_tokens: [B, K, D] target representation tokens
        cap: Maximum relative displacement multiplier to prevent gradient explosion
        bonus: Multiplier on displacement ratio

    Returns:
        weights: [B, K] non-negative spatial weights normalized to mean 1.0 across tokens per sequence item.
        meaningful_fraction: float fraction of tokens receiving substantial change weight (>= 1.5).
    """
    diff = target_tokens.detach() - current_tokens.detach()
    d = torch.norm(diff, p=2, dim=-1)  # [B, K]
    mean_d = d.mean(dim=-1, keepdim=True)  # [B, 1]

    # Relative displacement ratio per token
    ratio = d / (mean_d + 1e-6)
    ratio_capped = torch.clamp(ratio, 0.0, float(cap))

    # Weight bonus proportional to displacement
    raw_weights = 1.0 + float(bonus) * ratio_capped

    # Normalize across the K spatial tokens so average weight per transition is exactly 1.0
    weights = raw_weights / raw_weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)

    # Diagnostic: fraction of tokens receiving meaningful weight (>= 1.5)
    meaningful_fraction = float((weights >= 1.5).float().mean().item())
    return weights.detach(), meaningful_fraction


@torch.no_grad()
def reconstruct_belief_prefix(
    model: TrueJEPA,
    batch: Dict[str, Tensor],
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Reconstruct recurrent belief state at window start using current model weights (R05).

    Unrolls the belief core over the sampled prefix [0..start] in no_grad under
    current parameters, ensuring the training window starts with an exact,
    history-consistent hidden state rather than a false reset (zero belief).

    Returns:
        hidden: [B, belief_dim] detached belief at window start
        previous_action: [B] previous action before window start
        previous_reward: [B] previous reward before window start
    """
    prefix_lengths = batch.get("prefix_lengths")
    obs = batch["obs"]
    batch_size = obs.shape[0]

    if prefix_lengths is None:
        return (
            model.initial_belief(batch_size, device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.float32, device=device),
        )

    prefix_lengths = prefix_lengths.to(device)
    max_prefix = int(prefix_lengths.max().item())
    if max_prefix == 0:
        return (
            model.initial_belief(batch_size, device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.float32, device=device),
        )

    prefix_obs = batch["prefix_obs"].to(device)
    prefix_actions = batch["prefix_actions"].to(device)
    prefix_rewards = batch["prefix_rewards"].to(device)

    hidden = model.initial_belief(batch_size, device)
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for step in range(1, max_prefix + 1):
        mask = step <= prefix_lengths
        if not mask.any():
            break
        frame = model.encode(prefix_obs[:, step])
        prev_a = prefix_actions[:, step - 1]
        prev_r = prefix_rewards[:, step - 1]
        new_hidden = model.belief_step(frame, prev_a, prev_r, hidden)
        hidden = torch.where(mask.unsqueeze(-1), new_hidden, hidden)

    for b in range(batch_size):
        p_len = int(prefix_lengths[b].item())
        if p_len > 0:
            previous_action[b] = prefix_actions[b, p_len - 1]
            previous_reward[b] = prefix_rewards[b, p_len - 1]

    return hidden.detach(), previous_action.detach(), previous_reward.detach()


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

    # Reconstruct belief at window start from prefix (R05)
    hidden_init, prev_act_init, prev_rew_init = reconstruct_belief_prefix(model, batch, device)

    optimizer.zero_grad(set_to_none=True)
    with autocast:
        current = model.encode(obs[:, 0])
        hidden = hidden_init
        previous_action = prev_act_init
        previous_reward = prev_rew_init
        jepa_terms = []
        reward_terms = []
        continue_terms = []
        ensemble_terms = []
        predicted_pooled = []
        meaningful_fractions = []

        for time in range(sequence_length):
            if time > 0:
                hidden = model.belief_step(current, previous_action, previous_reward, hidden)
            action = actions[:, time]
            reward_prediction, continuation_logit = model.transition_heads(current, hidden, action)
            predicted = model.predict(current, action, hidden)
            target = model.target_encode(obs[:, time + 1])
            step_mask = valid[:, time]

            # JEPA token prediction loss with optional spatial change weighting
            token_diff_sq = (predicted.tokens - target.tokens.detach()).square().mean(dim=-1)  # [B, K]
            if getattr(cfg, "change_weighted_loss", False):
                change_w, frac = compute_spatial_change_weights(
                    current.tokens,
                    target.tokens,
                    cap=getattr(cfg, "change_weight_cap", 5.0),
                    bonus=getattr(cfg, "change_weight_bonus", 2.0),
                )
                meaningful_fractions.append(frac)
                token_term = (change_w * token_diff_sq).mean(dim=-1)
            else:
                token_term = token_diff_sq.mean(dim=-1)

            pooled_term = (predicted.pooled - target.pooled.detach()).square().mean(dim=-1)
            jepa_terms.append(
                _masked_mean(token_term, step_mask) + _masked_mean(pooled_term, step_mask)
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

            rollout_mode = getattr(cfg, "rollout_mode", "autoregressive")
            if rollout_mode == "one_step":
                current = model.encode(obs[:, time + 1])
            elif rollout_mode == "mixed":
                if torch.rand(1).item() < getattr(cfg, "mixed_rollout_prob", 0.5):
                    current = predicted
                else:
                    current = model.encode(obs[:, time + 1])
            else:
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
    metrics = {
        "loss": float(total.detach().float().cpu()),
        "jepa": float(jepa.detach().float().cpu()),
        "reward": float(reward_loss.detach().float().cpu()),
        "continuation": float(continue_loss.detach().float().cpu()),
        "ensemble": float(ensemble_loss.detach().float().cpu()),
        "latent_std": float(pooled.detach().float().std().cpu()),
    }
    if meaningful_fractions:
        metrics["meaningful_token_fraction"] = float(sum(meaningful_fractions) / len(meaningful_fractions))
    return metrics


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


def imagination_actor_critic_gae_step(
    model: TrueJEPA,
    start: LatentFrame,
    hidden: Tensor,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    horizon: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_weight: float = 0.01,
    reward_clip: float = 1.0,
    uncertainty_penalty: float = 0.0,
) -> Dict[str, float]:
    """Train actor and value heads via latent imagination with GAE and lambda-returns.

    World model parameters are strictly frozen/detached. The actor learns from GAE advantages,
    and the value head regresses toward lambda-return targets.
    Continuation logits scale the discount factor dynamically to account for predicted death.
    """
    actor_optimizer.zero_grad(set_to_none=True)
    value_optimizer.zero_grad(set_to_none=True)

    frame = LatentFrame(start.tokens.detach(), start.pooled.detach())
    belief = hidden.detach()

    log_probs = []
    entropies = []
    rewards = []
    gammas = []
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
            reward, cont_logit = model.transition_heads(frame, belief, action)
            # Clip reward to prevent runaway exploitation
            if reward_clip > 0:
                reward = torch.clamp(reward, -reward_clip, reward_clip)

            # Optional uncertainty penalty via ensemble disagreement
            if uncertainty_penalty > 0 and hasattr(model, "ensemble"):
                ens = model.ensemble(frame, action, belief)
                disagreement = ens.std(dim=0).mean(dim=-1)
                reward = reward - uncertainty_penalty * disagreement

            # Continuation discount
            cont_prob = torch.sigmoid(cont_logit)
            step_gamma = gamma * cont_prob

            next_frame = model.predict(frame, action, belief)
            next_belief = model.belief_step(next_frame, action, reward, belief)

        rewards.append(reward)
        gammas.append(step_gamma)
        frame = LatentFrame(next_frame.tokens.detach(), next_frame.pooled.detach())
        belief = next_belief.detach()

    # Bootstrap final value V(s_H)
    with torch.no_grad():
        final_state = model.state(frame, belief)
        final_value = model.heads.value(final_state).squeeze(-1)

    values.append(final_value)

    # Compute GAE advantages and lambda-returns
    advantages = [torch.zeros_like(values[0]) for _ in range(horizon)]
    returns = [torch.zeros_like(values[0]) for _ in range(horizon)]
    running_gae = torch.zeros_like(values[0])

    for t in reversed(range(horizon)):
        delta = rewards[t] + gammas[t] * values[t + 1].detach() - values[t].detach()
        running_gae = delta + gammas[t] * gae_lambda * running_gae
        advantages[t] = running_gae
        returns[t] = running_gae + values[t].detach()

    adv_stack = torch.stack(advantages)
    ret_stack = torch.stack(returns)
    val_stack = torch.stack(values[:-1])
    log_prob_stack = torch.stack(log_probs)
    entropy_stack = torch.stack(entropies)

    # Normalize advantages per batch/horizon for stable gradients
    adv_mean = adv_stack.mean()
    adv_std = adv_stack.std().clamp_min(1e-6)
    adv_norm = (adv_stack - adv_mean) / adv_std

    # 1. Target return clamping to prevent runaway value divergence (DreamerV3 bounded targets)
    ret_clamped = torch.clamp(ret_stack.detach(), -10.0, 10.0)
    value_loss = F.smooth_l1_loss(val_stack, ret_clamped)

    # 2. Policy loss with entropy barrier to prevent catastrophic action collapse
    mean_ent = entropy_stack.mean()
    ent_barrier = torch.relu(1.2 - mean_ent).square() * 2.0
    actor_loss = -(adv_norm.detach() * log_prob_stack).mean() - entropy_weight * mean_ent + ent_barrier

    actor_loss.backward()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.heads.policy.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.heads.value.parameters(), max_norm=1.0)
    actor_optimizer.step()
    value_optimizer.step()

    return {
        "actor_loss": float(actor_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "mean_advantage": float(adv_mean.item()),
        "mean_imagined_reward": float(torch.stack(rewards).mean().item()),
        "mean_imagined_value": float(val_stack.mean().item()),
        "mean_entropy": float(entropy_stack.mean().item()),
        "mean_continuation": float(torch.stack(gammas).mean().item() / gamma),
    }
