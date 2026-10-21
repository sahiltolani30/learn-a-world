"""
Evaluation Metrics & Crafter Scoring for Learn-a-World.
Computes:
1. Official Crafter Achievement Score (geometric mean across 22 achievements).
2. Survival length and achievement unlock statistics.
3. Multi-step latent rollout prediction error & compounding drift.
"""

import math
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

# Standard 22 Crafter achievements (Hafner 2021)
CRAFTER_ACHIEVEMENTS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

def compute_crafter_score(achievement_records: List[Dict[str, int]]) -> Dict[str, Any]:
    """
    Computes standard Crafter benchmark score across evaluation episodes.
    score = exp( (1 / 22) * sum(ln(1 + percentage_i)) ) - 1
    where percentage_i is the percentage of episodes where achievement i > 0.
    """
    if not achievement_records:
        return {
            "score": 0.0,
            "unlocked_count": 0,
            "mean_achievements_per_ep": 0.0,
            "unlock_percentages": {k: 0.0 for k in CRAFTER_ACHIEVEMENTS},
        }

    num_episodes = len(achievement_records)
    unlock_counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}

    ep_unlocked_totals = []
    for rec in achievement_records:
        ep_count = 0
        for ach in CRAFTER_ACHIEVEMENTS:
            if rec.get(ach, 0) > 0:
                unlock_counts[ach] += 1
                ep_count += 1
        ep_unlocked_totals.append(ep_count)

    unlock_percentages = {
        ach: (count / num_episodes) * 100.0 for ach, count in unlock_counts.items()
    }

    # Official Crafter geometric mean formula
    log_sum = sum(math.log(1.0 + pct) for pct in unlock_percentages.values())
    crafter_score = float(math.exp(log_sum / len(CRAFTER_ACHIEVEMENTS)) - 1.0)
    unlocked_count = sum(1 for cnt in unlock_counts.values() if cnt > 0)
    mean_per_ep = float(np.mean(ep_unlocked_totals))

    return {
        "score": crafter_score,
        "unlocked_count": unlocked_count,
        "mean_achievements_per_ep": mean_per_ep,
        "unlock_percentages": unlock_percentages,
    }

def evaluate_crafter_agent(
    env: Any,
    action_fn: Callable[[np.ndarray, Dict[str, Any]], int],
    num_episodes: int = 20,
    max_steps: int = 1000,
    seed_offset: int = 4000,
) -> Dict[str, Any]:
    """
    Evaluates an agent on held-out procedural Crafter seeds.
    Tracks survival length, total reward, and achievement unlocks.
    """
    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        ep_reward = 0.0
        steps = 0

        for step in range(max_steps):
            steps += 1
            action = action_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

            if terminated or truncated:
                break

        achievements = dict(info.get("achievements", {}))
        achievement_records.append(achievements)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

    score_stats = compute_crafter_score(achievement_records)

    return {
        "crafter_score": score_stats["score"],
        "unlocked_count": score_stats["unlocked_count"],
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "mean_survival_steps": float(np.mean(survival_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "num_episodes": num_episodes,
        "unlock_percentages": score_stats["unlock_percentages"],
    }

def compute_multi_step_drift(
    model: Any,
    obs_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    next_obs_seq: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes compounding latent drift across multi-step imagination rollouts.
    obs_batch: [B, C, H, W]
    actions_batch: [B, Horizon]
    next_obs_seq: [B, Horizon, C, H, W]
    """
    model.eval()
    device = next(model.parameters()).device
    obs_batch = obs_batch.to(device)
    actions_batch = actions_batch.to(device)
    next_obs_seq = next_obs_seq.to(device)

    B, H = actions_batch.shape
    errors_per_step = []

    with torch.no_grad():
        z_pred = model.encode(obs_batch, use_target=False)
        for h in range(H):
            act_h = actions_batch[:, h]
            z_pred = model.predictor(z_pred, act_h)
            target_h = model.encode(next_obs_seq[:, h], use_target=True)
            err = torch.norm(z_pred - target_h, p=2, dim=-1).mean().item()
            errors_per_step.append(err)

    return {
        "mean_drift": float(np.mean(errors_per_step)),
        "final_step_drift": float(errors_per_step[-1]),
        "errors_per_step": errors_per_step,
    }
