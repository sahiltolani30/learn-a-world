"""Gate 8: Short-Horizon Imagination Policy & Mathematical Returns Module.

Implements:
1. Exact continuation-aware lambda-return formulation with value bootstrap (§8).
2. Episode prefix belief reconstruction for exact recurrent state burn-in (§6.3).
3. Free-running action-conditioned imagination rollouts (§6.2).
4. Prerequisite qualifications (closed-loop horizons, reward, continuation, value calibration).
5. Gate 8 imagination training steps with separate actor and critic optimizers (§9).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from true_jepa.model import TrueJEPA, JEPAConfig, LatentFrame


def compute_continuation_lambda_returns(
    rewards: Tensor,
    continuations: Tensor,
    values: Tensor,
    gamma: float = 0.97,
    lambda_param: float = 0.95,
) -> Tensor:
    """Computes continuation-aware lambda-returns across an imagined rollout.

    Args:
        rewards: [H, B] extrinsic rewards r_t for steps t = 0..H-1.
        continuations: [H, B] continuation probabilities c_t in [0, 1] for steps t = 0..H-1.
        values: [H + 1, B] state-values V(s_t) for steps t = 0..H.
        gamma: Environment discount factor in (0, 1].
        lambda_param: TD(lambda) return mixing factor in [0, 1].

    Returns:
        returns: [H, B] lambda-return targets G_t for steps t = 0..H-1.
    """
    H, B = rewards.shape
    if continuations.shape != (H, B):
        raise ValueError(f"continuations shape {continuations.shape} must match rewards shape {(H, B)}")
    if values.shape != (H + 1, B):
        raise ValueError(f"values shape {values.shape} must be (H + 1, B) = {(H + 1, B)}")

    # G_H = V(s_H) bootstrapped at artificial horizon
    returns = torch.zeros((H, B), dtype=rewards.dtype, device=rewards.device)
    running_return = values[H].clone()

    # Backward recursion: G_t = r_t + d_t * ((1 - lambda) * V(s_{t+1}) + lambda * G_{t+1})
    for t in reversed(range(H)):
        d_t = gamma * continuations[t]
        running_return = rewards[t] + d_t * (
            (1.0 - lambda_param) * values[t + 1] + lambda_param * running_return
        )
        returns[t] = running_return

    return returns


def reconstruct_belief_from_prefix(
    model: TrueJEPA,
    obs_prefix: Tensor,
    actions_prefix: Tensor,
    rewards_prefix: Tensor,
) -> Tuple[LatentFrame, Tensor]:
    """Reconstructs the recurrent belief core by burning in from true episode start.

    Args:
        model: TrueJEPA instance with .encoder and .belief.
        obs_prefix: [T, B, 64, 64, 3] or [B, T, 64, 64, 3] uint8 observations from t=0 to t_target.
        actions_prefix: [T - 1, B] actions executed prior to each step.
        rewards_prefix: [T - 1, B] scalar rewards received prior to each step.

    Returns:
        frame_target: LatentFrame encoded from the final observation at t_target.
        belief_target: [B, belief_dim] recurrent hidden state at t_target.
    """
    if obs_prefix.ndim == 5 and obs_prefix.shape[0] != actions_prefix.shape[0] + 1:
        # Check if batch dimension is first
        if obs_prefix.shape[1] == actions_prefix.shape[0] + 1:
            obs_prefix = obs_prefix.transpose(0, 1)

    T, B = obs_prefix.shape[0], obs_prefix.shape[1]
    device = next(model.parameters()).device
    hidden = model.initial_belief(B, device)

    # Encode first frame
    frame = model.encode(obs_prefix[0].to(device))

    # Burn in forward through prefix
    for step in range(T - 1):
        act = actions_prefix[step].to(device)
        rew = rewards_prefix[step].to(device)
        next_frame = model.encode(obs_prefix[step + 1].to(device))
        hidden = model.belief_step(frame, act, rew, hidden)
        frame = next_frame

    return frame, hidden


def generate_imagined_rollout(
    model: TrueJEPA,
    start_frame: LatentFrame,
    start_belief: Tensor,
    horizon: int,
    stochastic: bool = True,
    temperature: float = 1.0,
    reward_clip: float = 5.0,
    actor_head: Optional[nn.Module] = None,
    world_model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Simulates a free-running action-conditioned imagination rollout.

    Args:
        model: TrueJEPA model containing predictor, heads, and belief core.
        start_frame: Initial latent frame (pooled and tokens).
        start_belief: Initial recurrent hidden belief.
        horizon: Imagination steps H >= 1.
        stochastic: If True, samples actions from Categorical policy; if False, uses argmax.
        temperature: Action sampling temperature.
        reward_clip: Maximum absolute value for predicted reward clamping.
        actor_head: Optional policy head override (e.g. Residual MLP).
        world_model_override: Diagnostic ablation mode ('persistence', 'shuffled_action', None).

    Returns:
        Dictionary containing imagined tensors:
        - 'actions': [H, B]
        - 'rewards': [H, B]
        - 'continuations': [H, B]
        - 'values': [H + 1, B]
        - 'log_probs': [H, B]
        - 'entropies': [H, B]
    """
    B = start_belief.shape[0]
    device = start_belief.device

    frame = LatentFrame(start_frame.tokens.detach(), start_frame.pooled.detach())
    belief = start_belief.detach()

    actions_list = []
    log_probs_list = []
    entropies_list = []
    rewards_list = []
    conts_list = []
    values_list = []

    for step in range(horizon):
        # Current state representation: cat[pooled, hidden]
        state = model.state(frame, belief)
        val = model.heads.value(state).squeeze(-1)
        values_list.append(val)

        # Policy logits
        if actor_head is not None:
            logits = actor_head(state)
        else:
            logits = model.policy_logits(frame, belief)

        temp = max(temperature, 1e-4)
        dist = torch.distributions.Categorical(logits=logits / temp)
        if stochastic:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)

        actions_list.append(action)
        log_probs_list.append(dist.log_prob(action))
        entropies_list.append(dist.entropy())

        # Predicted transitions from world model
        with torch.no_grad():
            plan_action = action
            if world_model_override == "shuffled_action":
                # Shuffle actions along batch dimension to break action conditioning
                perm = torch.randperm(B, device=device)
                plan_action = action[perm]

            pred_reward, cont_logit = model.transition_heads(frame, belief, plan_action)
            if reward_clip > 0:
                pred_reward = torch.clamp(pred_reward, -reward_clip, reward_clip)

            cont_prob = torch.sigmoid(cont_logit)

            if world_model_override == "persistence":
                # Next frame is identical to current frame
                next_frame = frame
            else:
                next_frame = model.predict(frame, plan_action, belief)

            next_belief = model.belief_step(next_frame, plan_action, pred_reward, belief)

        rewards_list.append(pred_reward)
        conts_list.append(cont_prob)

        frame = LatentFrame(next_frame.tokens.detach(), next_frame.pooled.detach())
        belief = next_belief.detach()

    # Bootstrap terminal value V(s_H)
    with torch.no_grad():
        final_state = model.state(frame, belief)
        final_val = model.heads.value(final_state).squeeze(-1)
    values_list.append(final_val)

    return {
        "actions": torch.stack(actions_list, dim=0),
        "rewards": torch.stack(rewards_list, dim=0),
        "continuations": torch.stack(conts_list, dim=0),
        "values": torch.stack(values_list, dim=0),
        "log_probs": torch.stack(log_probs_list, dim=0),
        "entropies": torch.stack(entropies_list, dim=0),
    }


def gate8_imagination_training_step(
    model: TrueJEPA,
    real_batch: Dict[str, Tensor],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    horizon: int = 1,
    gamma: float = 0.97,
    lambda_param: float = 0.95,
    beta_temperature: float = 3.0,
    max_weight: float = 20.0,
    entropy_weight: float = 1e-3,
    real_critic_weight: float = 0.5,
    imagined_critic_weight: float = 0.5,
    world_model_override: Optional[str] = None,
) -> Dict[str, float]:
    """Executes one Gate 8 policy and value update combining real and imagined transitions.

    Enforces:
    1. Separate actor and critic optimizers.
    2. World model parameters strictly frozen (zero gradients).
    3. Real data value calibration preserved via real_critic_weight.
    4. Bounded AWR weights with clamped return targets.
    """
    device = next(model.parameters()).device
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)

    # 1. Real batch inputs
    real_states = real_batch["states"].to(device)  # [B, state_dim]
    real_actions = real_batch["actions"].to(device)  # [B]
    real_returns = real_batch["returns"].to(device)  # [B]
    start_frames = real_batch["start_frames"]  # LatentFrame of batch size B
    start_beliefs = real_batch["start_beliefs"].to(device)  # [B, belief_dim]

    # --- Real Critic Calibration ---
    real_pred_values = model.heads.value(real_states).squeeze(-1)
    real_ret_clamped = torch.clamp(real_returns, -10.0, 10.0)
    real_critic_loss = F.smooth_l1_loss(real_pred_values, real_ret_clamped)

    # --- Real Actor AWR ---
    with torch.no_grad():
        real_adv = real_returns - real_pred_values.detach()
        real_weights = torch.clamp(torch.exp(real_adv / beta_temperature), 0.0, max_weight)

    real_logits = model.heads.policy(real_states)
    real_dist = torch.distributions.Categorical(logits=real_logits)
    real_log_probs = real_dist.log_prob(real_actions)
    real_entropy = real_dist.entropy().mean()
    real_actor_loss = -(real_weights * real_log_probs).mean() - entropy_weight * real_entropy

    # --- Imagined Rollouts (Horizon H) ---
    imagined = generate_imagined_rollout(
        model=model,
        start_frame=start_frames,
        start_belief=start_beliefs,
        horizon=horizon,
        stochastic=True,
        temperature=1.0,
        world_model_override=world_model_override,
    )

    # Compute continuation-aware lambda-returns
    imag_returns = compute_continuation_lambda_returns(
        rewards=imagined["rewards"],
        continuations=imagined["continuations"],
        values=imagined["values"].detach(),
        gamma=gamma,
        lambda_param=lambda_param,
    )

    # Imagined Critic Loss (Smooth L1 regression toward lambda-returns)
    imag_val_preds = imagined["values"][:-1]  # [H, B]
    imag_ret_clamped = torch.clamp(imag_returns.detach(), -10.0, 10.0)
    imag_critic_loss = F.smooth_l1_loss(imag_val_preds, imag_ret_clamped)

    # Imagined Actor AWR Loss
    with torch.no_grad():
        imag_adv = imag_returns - imag_val_preds.detach()
        imag_weights = torch.clamp(torch.exp(imag_adv / beta_temperature), 0.0, max_weight)

    imag_log_probs = imagined["log_probs"]
    imag_entropy = imagined["entropies"].mean()
    imag_actor_loss = -(imag_weights * imag_log_probs).mean() - entropy_weight * imag_entropy

    # Combined Losses
    total_actor_loss = 0.5 * real_actor_loss + 0.5 * imag_actor_loss
    total_critic_loss = real_critic_weight * real_critic_loss + imagined_critic_weight * imag_critic_loss

    # Backward and optimize with gradient clipping
    total_actor_loss.backward()
    total_critic_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.heads.policy.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.heads.value.parameters(), max_norm=1.0)

    actor_optimizer.step()
    critic_optimizer.step()

    # Telemetry
    return {
        "total_actor_loss": float(total_actor_loss.item()),
        "real_actor_loss": float(real_actor_loss.item()),
        "imag_actor_loss": float(imag_actor_loss.item()),
        "total_critic_loss": float(total_critic_loss.item()),
        "real_critic_loss": float(real_critic_loss.item()),
        "imag_critic_loss": float(imag_critic_loss.item()),
        "mean_real_return": float(real_returns.mean().item()),
        "mean_imag_return": float(imag_returns.mean().item()),
        "mean_imag_reward": float(imagined["rewards"].mean().item()),
        "mean_imag_continuation": float(imagined["continuations"].mean().item()),
        "mean_entropy": float(0.5 * (real_entropy + imag_entropy).item()),
        "imag_vs_real_return_gap": float((imag_returns.mean() - real_returns.mean()).item()),
    }


# ---------------------------------------------------------------------------
# Prerequisite Qualifications (§6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_action_conditioned_prediction(
    model: TrueJEPA,
    eval_episodes: List[Dict[str, Any]],
    device: torch.device,
    horizons: List[int] = [1, 2, 4, 8],
    max_transitions_per_ep: int = 60,
    stride: int = 2,
) -> Dict[str, Any]:
    """Validates action-conditioned multi-step latent prediction against persistence and shuffled actions."""
    model.eval()
    max_H = max(horizons)

    errors_dict: Dict[int, Dict[str, List[float]]] = {
        h: {"true": [], "shuf": [], "persist": []} for h in horizons
    }

    for ep in eval_episodes:
        obs = ep["obs"][:max_transitions_per_ep + max_H]
        actions = ep["action"][:max_transitions_per_ep + max_H] if "action" in ep else ep["actions"][:max_transitions_per_ep + max_H]
        rewards = ep["reward"][:max_transitions_per_ep + max_H] if "reward" in ep else ep["rewards"][:max_transitions_per_ep + max_H]

        T = len(actions) - max_H
        if T <= 0:
            continue

        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
        frames = model.encode(obs_tensor)
        hidden = model.initial_belief(1, device)

        for t in range(0, T, stride):
            frame_t = LatentFrame(tokens=frames.tokens[t:t+1], pooled=frames.pooled[t:t+1])

            # Advance hidden state up to step t if needed
            for step_h in range(t):
                pass  # hidden is already maintained sequentially below

            for H in horizons:
                target_frame = LatentFrame(tokens=frames.tokens[t+H:t+H+1], pooled=frames.pooled[t+H:t+H+1])

                # Free-running closed loop rollout
                curr_f = frame_t
                curr_h = hidden
                shuf_f = frame_t
                shuf_h = hidden

                for step in range(H):
                    act_true = torch.tensor([int(actions[t + step])], device=device)
                    act_shuf = torch.randint(0, 17, (1,), device=device)

                    pred_r, _ = model.transition_heads(curr_f, curr_h, act_true)
                    pred_r_shuf, _ = model.transition_heads(shuf_f, shuf_h, act_shuf)

                    next_f = model.predict(curr_f, act_true, curr_h)
                    next_f_shuf = model.predict(shuf_f, act_shuf, shuf_h)

                    curr_h = model.belief_step(next_f, act_true, pred_r, curr_h)
                    shuf_h = model.belief_step(next_f_shuf, act_shuf, pred_r_shuf, shuf_h)

                    curr_f = next_f
                    shuf_f = next_f_shuf

                err_true = F.smooth_l1_loss(curr_f.pooled, target_frame.pooled).item()
                err_shuf = F.smooth_l1_loss(shuf_f.pooled, target_frame.pooled).item()
                err_persist = F.smooth_l1_loss(frame_t.pooled, target_frame.pooled).item()

                errors_dict[H]["true"].append(err_true)
                errors_dict[H]["shuf"].append(err_shuf)
                errors_dict[H]["persist"].append(err_persist)

            # Advance real belief by stride steps
            for s in range(stride):
                if t + s < T:
                    act_real = torch.tensor([int(actions[t + s])], device=device)
                    rew_real = torch.tensor([float(rewards[t + s])], device=device)
                    frame_s = LatentFrame(tokens=frames.tokens[t+s:t+s+1], pooled=frames.pooled[t+s:t+s+1])
                    hidden = model.belief_step(frame_s, act_real, rew_real, hidden)

    results: Dict[str, Any] = {}
    for H in horizons:
        mean_true = float(np.mean(errors_dict[H]["true"]))
        mean_shuf = float(np.mean(errors_dict[H]["shuf"]))
        mean_persist = float(np.mean(errors_dict[H]["persist"]))

        results[f"H_{H}"] = {
            "true_error": mean_true,
            "shuffled_error": mean_shuf,
            "persistence_error": mean_persist,
            "persistence_ratio": float(mean_true / max(1e-6, mean_persist)),
            "action_sensitivity_ratio": float(mean_shuf / max(1e-6, mean_true)),
            "beats_persistence": bool(mean_true < mean_persist),
            "action_conditioning_active": bool(mean_shuf > mean_true),
        }

    return results


@torch.no_grad()
def validate_reward_prediction(
    model: TrueJEPA,
    eval_episodes: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates world model reward head against real transitions and constant baseline."""
    model.eval()
    pred_rewards: List[float] = []
    real_rewards: List[float] = []

    for ep in eval_episodes:
        obs = ep["obs"]
        actions = ep["action"] if "action" in ep else ep["actions"]
        rewards = ep["reward"] if "reward" in ep else ep["rewards"]
        T = len(actions)
        if T == 0:
            continue

        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
        frames = model.encode(obs_tensor)
        hidden = model.initial_belief(1, device)

        for t in range(T):
            frame_t = LatentFrame(tokens=frames.tokens[t:t+1], pooled=frames.pooled[t:t+1])
            act_t = torch.tensor([int(actions[t])], device=device)
            pred_r, _ = model.transition_heads(frame_t, hidden, act_t)

            real_r = float(rewards[t])
            pred_rewards.append(float(pred_r.item()))
            real_rewards.append(real_r)

            hidden = model.belief_step(frame_t, act_t, torch.tensor([real_r], device=device), hidden)

    p_arr = np.array(pred_rewards, dtype=np.float64)
    r_arr = np.array(real_rewards, dtype=np.float64)

    errors = p_arr - r_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Constant mean baseline error
    const_pred = np.mean(r_arr)
    const_errors = const_pred - r_arr
    const_mae = float(np.mean(np.abs(const_errors)))
    const_rmse = float(np.sqrt(np.mean(const_errors ** 2)))

    p_std = np.std(p_arr)
    r_std = np.std(r_arr)
    corr = float(np.corrcoef(p_arr, r_arr)[0, 1]) if p_std > 1e-8 and r_std > 1e-8 else 0.0

    return {
        "reward_mae": mae,
        "reward_rmse": rmse,
        "reward_bias": bias,
        "reward_correlation": corr,
        "constant_baseline_mae": const_mae,
        "constant_baseline_rmse": const_rmse,
        "beats_constant_baseline": bool(mae < const_mae),
        "mean_real_reward": float(np.mean(r_arr)),
        "mean_pred_reward": float(np.mean(p_arr)),
        "num_eval_transitions": len(p_arr),
    }


@torch.no_grad()
def validate_continuation_prediction(
    model: TrueJEPA,
    eval_episodes: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates world model continuation head against termination outcomes and prevalence baseline."""
    model.eval()
    pred_conts: List[float] = []
    real_conts: List[float] = []

    for ep in eval_episodes:
        obs = ep["obs"]
        actions = ep["action"] if "action" in ep else ep["actions"]
        terminated = ep["terminated"] if "terminated" in ep else [False] * len(actions)
        rewards = ep["reward"] if "reward" in ep else ep["rewards"]

        T = len(actions)
        if T == 0:
            continue

        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
        frames = model.encode(obs_tensor)
        hidden = model.initial_belief(1, device)

        for t in range(T):
            frame_t = LatentFrame(tokens=frames.tokens[t:t+1], pooled=frames.pooled[t:t+1])
            act_t = torch.tensor([int(actions[t])], device=device)
            _, cont_logit = model.transition_heads(frame_t, hidden, act_t)

            pred_c = float(torch.sigmoid(cont_logit).item())
            real_c = 0.0 if bool(terminated[t]) else 1.0

            pred_conts.append(pred_c)
            real_conts.append(real_c)

            real_r = float(rewards[t])
            hidden = model.belief_step(frame_t, act_t, torch.tensor([real_r], device=device), hidden)

    p_arr = np.array(pred_conts, dtype=np.float64)
    y_arr = np.array(real_conts, dtype=np.float64)

    # Brier score: mean squared difference from binary outcome
    brier_score = float(np.mean((p_arr - y_arr) ** 2))

    # Prevalence baseline Brier score (predicting constant prevalence of survival)
    prevalence = float(np.mean(y_arr))
    prevalence_brier = float(np.mean((prevalence - y_arr) ** 2))

    # Binary accuracy with threshold 0.5
    preds_binary = (p_arr >= 0.5).astype(np.float64)
    accuracy = float(np.mean(preds_binary == y_arr))

    # True positives, False positives, etc. for death detection (event = 0)
    death_real = (y_arr == 0.0)
    death_pred = (p_arr < 0.5)
    tp = np.sum(death_pred & death_real)
    fp = np.sum(death_pred & ~death_real)
    fn = np.sum(~death_pred & death_real)
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))

    return {
        "brier_score": brier_score,
        "prevalence_brier": prevalence_brier,
        "beats_prevalence": bool(brier_score <= prevalence_brier),
        "accuracy": accuracy,
        "death_precision": precision,
        "death_recall": recall,
        "mean_pred_continuation": float(np.mean(p_arr)),
        "mean_real_continuation": prevalence,
        "num_eval_transitions": len(p_arr),
    }


@torch.no_grad()
def validate_value_critic_calibration(
    model: TrueJEPA,
    eval_episodes: List[Dict[str, Any]],
    device: torch.device,
    gamma: float = 0.97,
) -> Dict[str, Any]:
    """Evaluates scalar value critic against realized Monte Carlo discounted returns."""
    model.eval()
    pred_values: List[float] = []
    real_returns: List[float] = []

    for ep in eval_episodes:
        obs = ep["obs"]
        actions = ep["action"] if "action" in ep else ep["actions"]
        rewards = ep["reward"] if "reward" in ep else ep["rewards"]

        T = len(actions)
        if T == 0:
            continue

        # Ground-truth returns G_t = r_t + gamma * G_{t+1}
        returns_ep = np.zeros(T, dtype=np.float64)
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = float(rewards[t]) + gamma * running_g
            returns_ep[t] = running_g

        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
        frames = model.encode(obs_tensor)
        hidden = model.initial_belief(1, device)

        for t in range(T):
            frame_t = LatentFrame(tokens=frames.tokens[t:t+1], pooled=frames.pooled[t:t+1])
            state_t = model.state(frame_t, hidden)
            val = model.heads.value(state_t).squeeze(-1)

            pred_values.append(float(val.item()))
            real_returns.append(float(returns_ep[t]))

            act_t = torch.tensor([int(actions[t])], device=device)
            rew_t = torch.tensor([float(rewards[t])], device=device)
            hidden = model.belief_step(frame_t, act_t, rew_t, hidden)

    p_arr = np.array(pred_values, dtype=np.float64)
    g_arr = np.array(real_returns, dtype=np.float64)

    errors = p_arr - g_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Explained variance: 1 - Var(errors) / Var(returns)
    var_g = float(np.var(g_arr))
    var_err = float(np.var(errors))
    explained_variance = float(1.0 - (var_err / max(1e-8, var_g)))

    p_std = np.std(p_arr)
    g_std = np.std(g_arr)
    corr = float(np.corrcoef(p_arr, g_arr)[0, 1]) if p_std > 1e-8 and g_std > 1e-8 else 0.0

    return {
        "value_mae": mae,
        "value_rmse": rmse,
        "value_bias": bias,
        "value_correlation": corr,
        "explained_variance": explained_variance,
        "mean_real_return": float(np.mean(g_arr)),
        "mean_pred_value": float(np.mean(p_arr)),
        "std_real_return": float(g_std),
        "std_pred_value": float(p_std),
        "num_eval_states": len(p_arr),
    }

