"""Gate 7 Evaluator: Real-Experience Policy Learning, Value Calibration & Reward Validation.

Strictly adheres to Project ATLAS & Gate 7 invariants:
- Learner agent strictly receives (obs_uint8, reward_prev, step).
- Policy and value updates use real transitions from collected experience only (no imagined rollouts).
- Frozen DINOv2 backbone and target encoder are immutable.
- Evaluator-only metrics (achievements, info dicts) never enter policy decision loop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from true_jepa.model import LatentFrame, TrueJEPA


# ---------------------------------------------------------------------------
# 1. Dataset Audit & Provenance Verification
# ---------------------------------------------------------------------------

def audit_replay_dataset(
    path: Union[str, Path],
    eval_seeds: Optional[List[int]] = None,
    benchmark_seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Audits replay dataset schema, alignment, integrity, and seed isolation."""
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING", "path": str(p), "error": "File not found"}

    file_size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    sha256_hash = h.hexdigest()

    data = np.load(p, allow_pickle=True)
    required_keys = ["obs", "actions", "rewards", "terminated", "truncated", "obs_lengths", "action_lengths", "seeds"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        return {"status": "FAIL", "path": str(p), "error": f"Missing required keys: {missing_keys}"}

    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminated = data["terminated"]
    truncated = data["truncated"]
    obs_lengths = data["obs_lengths"]
    action_lengths = data["action_lengths"]
    seeds = [int(s) for s in data["seeds"]]

    num_episodes = len(obs_lengths)
    total_obs = int(sum(obs_lengths))
    total_actions = int(sum(action_lengths))

    # Temporal alignment: T_obs = T_act + 1 for each episode
    alignment_ok = True
    for idx, (o_len, a_len) in enumerate(zip(obs_lengths, action_lengths)):
        if o_len != a_len + 1:
            alignment_ok = False
            break

    # Numerical validity
    actions_valid = bool((actions.min() >= 0) and (actions.max() <= 16))
    rewards_valid = bool(np.isfinite(rewards).all())
    obs_valid = bool((obs.shape[1:] == (64, 64, 3)) and (obs.dtype == np.uint8))

    # Seed isolation check
    eval_seeds_set = set(eval_seeds) if eval_seeds is not None else set(range(3030, 3050))
    bench_seeds_set = set(benchmark_seeds) if benchmark_seeds is not None else set(range(5000, 5030))
    train_seeds_set = set(seeds)

    eval_overlap = list(train_seeds_set & eval_seeds_set)
    bench_overlap = list(train_seeds_set & bench_seeds_set)
    is_disjoint = (len(eval_overlap) == 0) and (len(bench_overlap) == 0)

    status = "PASS" if (alignment_ok and actions_valid and rewards_valid and obs_valid and is_disjoint) else "FAIL"

    return {
        "status": status,
        "path": str(p),
        "size_bytes": file_size,
        "sha256": sha256_hash,
        "num_episodes": num_episodes,
        "total_obs": total_obs,
        "total_actions": total_actions,
        "temporal_alignment_ok": alignment_ok,
        "actions_range": [int(actions.min()), int(actions.max())],
        "actions_valid": actions_valid,
        "rewards_valid": rewards_valid,
        "obs_shape_valid": obs_valid,
        "seeds": seeds,
        "unique_seeds_count": len(train_seeds_set),
        "seed_isolation": {
            "is_disjoint": is_disjoint,
            "eval_overlap": eval_overlap,
            "bench_overlap": bench_overlap,
        },
    }


# ---------------------------------------------------------------------------
# 2. Reward Prediction Validation against Real Observed Experience
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_reward_prediction(
    model: TrueJEPA,
    eval_episodes: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the world model's reward head on real observed transitions.

    Computes MAE, RMSE, Pearson correlation, and bias: (pred_reward - real_reward).
    """
    model.eval()
    pred_rewards: List[float] = []
    real_rewards: List[float] = []

    for ep in eval_episodes:
        obs_raw = ep["obs"]
        actions = ep["action"] if "action" in ep else ep["actions"]
        rewards = ep["reward"] if "reward" in ep else ep["rewards"]

        T = len(actions)
        if T == 0:
            continue

        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs_raw)).to(device)
        frames_all = model.encode(obs_tensor)

        hidden = model.initial_belief(1, device)
        for t in range(T):
            frame_t = LatentFrame(tokens=frames_all.tokens[t:t+1], pooled=frames_all.pooled[t:t+1])

            act_t = torch.tensor([int(actions[t])], device=device)
            pred_r, _ = model.transition_heads(frame_t, hidden, act_t)

            real_r = float(rewards[t])
            pred_rewards.append(float(pred_r.item()))
            real_rewards.append(real_r)

            # Update belief using real transition
            prev_act = act_t
            r_tensor = torch.tensor([real_r], device=device)
            hidden = model.belief_step(frame_t, prev_act, r_tensor, hidden)

    p_arr = np.array(pred_rewards, dtype=np.float64)
    r_arr = np.array(real_rewards, dtype=np.float64)

    errors = p_arr - r_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Pearson correlation
    p_std = np.std(p_arr)
    r_std = np.std(r_arr)
    if p_std > 1e-8 and r_std > 1e-8:
        corr = float(np.corrcoef(p_arr, r_arr)[0, 1])
    else:
        corr = 0.0

    return {
        "reward_mae": mae,
        "reward_rmse": rmse,
        "reward_bias": bias,
        "reward_correlation": corr,
        "num_eval_transitions": len(p_arr),
        "mean_real_reward": float(np.mean(r_arr)),
        "mean_pred_reward": float(np.mean(p_arr)),
    }


# ---------------------------------------------------------------------------
# 3. Value Calibration against Real Ground-Truth Discounted Returns
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_value_predictions(
    model: TrueJEPA,
    eval_episodes: List[Dict[str, Any]],
    device: torch.device,
    gamma: float = 0.99,
) -> Dict[str, Any]:
    """Calibrates value predictions against realized ground-truth returns.

    Computes MAE, RMSE, Pearson r, bias, and calibration across return deciles.
    """
    model.eval()
    pred_values: List[float] = []
    real_returns: List[float] = []

    for ep in eval_episodes:
        obs_raw = ep["obs"]
        actions = ep["action"] if "action" in ep else ep["actions"]
        rewards = ep["reward"] if "reward" in ep else ep["rewards"]

        T = len(actions)
        if T == 0:
            continue

        # Compute ground-truth Monte Carlo discounted returns: G_t = r_t + gamma * G_{t+1}
        returns_ep = np.zeros(T, dtype=np.float64)
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = float(rewards[t]) + gamma * running_g
            returns_ep[t] = running_g

        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs_raw)).to(device)
        frames_all = model.encode(obs_tensor)

        hidden = model.initial_belief(1, device)
        for t in range(T):
            frame_t = LatentFrame(tokens=frames_all.tokens[t:t+1], pooled=frames_all.pooled[t:t+1])

            state_t = model.heads.state(frame_t, hidden)
            val_t = model.heads.value(state_t).squeeze(-1)
            pred_values.append(float(val_t.item()))
            real_returns.append(float(returns_ep[t]))

            # Advance belief with real action and reward
            act_t = torch.tensor([int(actions[t])], device=device)
            r_tensor = torch.tensor([float(rewards[t])], device=device)
            hidden = model.belief_step(frame_t, act_t, r_tensor, hidden)

    p_arr = np.array(pred_values, dtype=np.float64)
    g_arr = np.array(real_returns, dtype=np.float64)

    errors = p_arr - g_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    p_std = np.std(p_arr)
    g_std = np.std(g_arr)
    if p_std > 1e-8 and g_std > 1e-8:
        corr = float(np.corrcoef(p_arr, g_arr)[0, 1])
    else:
        corr = 0.0

    # Decile calibration breakdown
    deciles = np.percentile(g_arr, np.linspace(0, 100, 6))
    calibration_groups = []
    for i in range(len(deciles) - 1):
        low, high = deciles[i], deciles[i + 1]
        mask = (g_arr >= low) & (g_arr <= high)
        if np.any(mask):
            calibration_groups.append({
                "return_range": [float(low), float(high)],
                "count": int(np.sum(mask)),
                "mean_real_return": float(np.mean(g_arr[mask])),
                "mean_pred_value": float(np.mean(p_arr[mask])),
                "group_mae": float(np.mean(np.abs(p_arr[mask] - g_arr[mask]))),
            })

    return {
        "value_mae": mae,
        "value_rmse": rmse,
        "value_bias": bias,
        "value_correlation": corr,
        "mean_real_return": float(np.mean(g_arr)),
        "mean_pred_value": float(np.mean(p_arr)),
        "min_pred_value": float(np.min(p_arr)),
        "max_pred_value": float(np.max(p_arr)),
        "num_eval_states": len(p_arr),
        "calibration_groups": calibration_groups,
    }


# ---------------------------------------------------------------------------
# 4. Offline Behavior-Regularized Policy Learning Step (AWR Objective)
# ---------------------------------------------------------------------------

def real_experience_policy_step(
    model: TrueJEPA,
    batch: Dict[str, Tensor],
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    advantage_temperature: float = 1.0,
    max_weight: float = 20.0,
    entropy_weight: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Executes one offline behavior-regularized policy & value update on real experience.

    Mathematical Objective:
    1. Ground-Truth Returns: G_t = r_t + gamma * (1 - terminated_t) * G_{t+1}.
    2. Value Objective: L_value = SmoothL1(V(s_t), G_t).
    3. Advantage: A_t = G_t - V(s_t).detach().
    4. Advantage Weighting: w_t = clamp(exp(A_t / tau), 0.0, max_weight).
    5. Actor Objective: L_actor = - w_t * log pi(a_t^real | s_t) - alpha_ent * H(pi).

    World model encoder and predictor parameters remain strictly frozen (0 gradients).
    """
    actor_optimizer.zero_grad(set_to_none=True)
    value_optimizer.zero_grad(set_to_none=True)

    obs = batch["obs"].to(device)                 # [B, T+1, 64, 64, 3] or tokens [B, T+1, 256, 384]
    actions = batch["actions"].to(device)         # [B, T]
    rewards = batch["rewards"].to(device)         # [B, T]
    terminated = batch["terminated"].to(device)   # [B, T]
    valid = batch["valid"].to(device)             # [B, T]

    B, T = actions.shape

    # 1. Compute ground-truth Monte Carlo / n-step returns backwards on real experience
    returns = torch.zeros((B, T), dtype=torch.float32, device=device)
    running_return = torch.zeros(B, dtype=torch.float32, device=device)

    for t in reversed(range(T)):
        cont_mask = (~terminated[:, t]).float()
        running_return = rewards[:, t] + gamma * cont_mask * running_return
        returns[:, t] = running_return

    # Clamping returns into realistic bounds [-10, 10]
    returns = torch.clamp(returns, -10.0, 10.0)

    # 2. Reconstruct beliefs and forward policy & value heads
    with torch.no_grad():
        if obs.ndim == 5:
            # Raw pixel frames: encode under no_grad
            flat_obs = obs.reshape(-1, *obs.shape[2:])
            frames = model.encode(flat_obs)
            tokens_all = frames.tokens.reshape(B, T + 1, *frames.tokens.shape[1:])
            pooled_all = frames.pooled.reshape(B, T + 1, *frames.pooled.shape[1:])
        elif obs.ndim == 4:
            # Pre-extracted DINOv2 tokens
            flat_tokens = obs.reshape(-1, *obs.shape[2:])
            frames = model.encoder.adapter(flat_tokens)
            tokens_all = frames[0].reshape(B, T + 1, *frames[0].shape[1:])
            pooled_all = frames[1].reshape(B, T + 1, *frames[1].shape[1:])
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

    # Unroll recurrent belief using real sequence transitions
    hidden = model.initial_belief(B, device)
    log_probs_list = []
    entropies_list = []
    values_list = []

    for t in range(T):
        frame_t = LatentFrame(tokens=tokens_all[:, t], pooled=pooled_all[:, t])
        state_t = model.heads.state(frame_t, hidden)

        # Policy logits and value estimate
        logits_t = model.heads.policy(state_t)
        val_t = model.heads.value(state_t).squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits_t)
        real_act_t = actions[:, t].long()

        log_probs_list.append(dist.log_prob(real_act_t))
        entropies_list.append(dist.entropy())
        values_list.append(val_t)

        # Advance belief with real transition under no_grad
        with torch.no_grad():
            act_tensor = actions[:, t]
            rew_tensor = rewards[:, t]
            hidden = model.belief_step(frame_t, act_tensor, rew_tensor, hidden)

    val_stack = torch.stack(values_list, dim=1)        # [B, T]
    log_prob_stack = torch.stack(log_probs_list, dim=1)# [B, T]
    entropy_stack = torch.stack(entropies_list, dim=1) # [B, T]

    valid_mask = valid.float()
    valid_count = valid_mask.sum().clamp_min(1.0)

    # 3. Value Loss on Real Returns
    value_diff = F.smooth_l1_loss(val_stack, returns, reduction="none")
    value_loss = (value_diff * valid_mask).sum() / valid_count

    # 4. Advantage-Weighted Behavior Regularization
    with torch.no_grad():
        advantages = returns - val_stack.detach()
        adv_weights = torch.clamp(torch.exp(advantages / max(advantage_temperature, 1e-4)), 0.0, max_weight)

    actor_nll = -(adv_weights * log_prob_stack * valid_mask).sum() / valid_count
    mean_entropy = (entropy_stack * valid_mask).sum() / valid_count
    actor_loss = actor_nll - entropy_weight * mean_entropy

    # Backward and optimize
    actor_loss.backward()
    value_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.heads.policy.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.heads.value.parameters(), max_norm=1.0)

    actor_optimizer.step()
    value_optimizer.step()

    return {
        "actor_loss": float(actor_loss.item()),
        "value_loss": float(value_loss.item()),
        "mean_advantage": float(((advantages * valid_mask).sum() / valid_count).item()),
        "mean_adv_weight": float(((adv_weights * valid_mask).sum() / valid_count).item()),
        "mean_pred_value": float(((val_stack * valid_mask).sum() / valid_count).item()),
        "mean_real_return": float(((returns * valid_mask).sum() / valid_count).item()),
        "mean_entropy": float(mean_entropy.item()),
    }
