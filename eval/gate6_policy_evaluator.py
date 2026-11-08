"""Gate 6 Policy Evaluator: Unmodified Crafter Real-Game Evaluation and Telemetry.

Strictly adheres to Project ATLAS & Gate 6 invariants:
- Learner agent strictly receives (obs_uint8, reward_prev, step).
- Evaluator-only metrics (achievements, episode lengths, health) are extracted
  strictly for post-hoc reporting and never enter the agent's input stream.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import crafter
import numpy as np
import torch

from true_jepa.agent import VisualJEPAAgent
from true_jepa.model import TrueJEPA


ALL_CRAFTER_ACHIEVEMENTS = [
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


def compute_crafter_score(achievement_rates: Dict[str, float]) -> float:
    """Computes official Crafter geometric score: exp(mean(ln(1 + rate))) - 1."""
    rates = [achievement_rates.get(ach, 0.0) for ach in ALL_CRAFTER_ACHIEVEMENTS]
    log_sum = sum(np.log(1.0 + r) for r in rates)
    return float(np.exp(log_sum / len(ALL_CRAFTER_ACHIEVEMENTS)) - 1.0) * 100.0


def evaluate_policy_in_crafter(
    agent: Any,
    seeds: List[int],
    max_steps: int = 500,
) -> Dict[str, Any]:
    """Evaluates an agent on unmodified Crafter across a specified list of seeds.

    Args:
        agent: Any agent implementing .act(obs_uint8, reward_prev, step) and optional .reset()
        seeds: List of integer seeds to evaluate
        max_steps: Maximum steps per episode (default 500)

    Returns:
        Comprehensive dictionary of return distributions, Crafter scores, achievement rates,
        action distributions, and survival statistics.
    """
    returns = []
    lengths = []
    achievements_unlocked_per_ep = {ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS}
    action_counts = np.zeros(17, dtype=np.int64)
    termination_causes = {"health_exhaustion": 0, "max_steps_reached": 0}
    per_seed_results = []

    for seed in seeds:
        env = crafter.Env(seed=seed)
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()

        ep_reward = 0.0
        step_count = 0
        reward_prev = 0.0
        done = False
        info = {}

        while not done and step_count < max_steps:
            action = agent.act(obs, reward_prev=reward_prev, step=step_count)
            action_int = int(action)
            action_counts[action_int] += 1

            obs, reward, done, info = env.step(action_int)
            ep_reward += float(reward)
            reward_prev = float(reward)
            step_count += 1

        returns.append(ep_reward)
        lengths.append(step_count)

        # Telemetry: Record unlocked achievements from info dict at episode end
        ep_achievements = info.get("achievements", {})
        for ach in ALL_CRAFTER_ACHIEVEMENTS:
            if ep_achievements.get(ach, 0) > 0:
                achievements_unlocked_per_ep[ach] += 1

        # Termination cause
        health = info.get("health", 0)
        if health <= 0 or done:
            termination_causes["health_exhaustion"] += 1
            cause = "health_exhaustion"
        else:
            termination_causes["max_steps_reached"] += 1
            cause = "max_steps_reached"

        per_seed_results.append({
            "seed": int(seed),
            "return": float(ep_reward),
            "length": int(step_count),
            "cause": cause,
            "achievements": [a for a, c in ep_achievements.items() if c > 0],
        })

    num_episodes = len(seeds)
    achievement_rates = {
        ach: float(count / num_episodes) for ach, count in achievements_unlocked_per_ep.items()
    }
    crafter_score = compute_crafter_score(achievement_rates)

    # Action distribution and empirical entropy
    total_actions = int(action_counts.sum())
    action_dist = (action_counts / max(1, total_actions)).tolist()
    probs = np.array(action_dist)
    nonzero_probs = probs[probs > 0]
    action_entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)))

    return {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "crafter_score": float(crafter_score),
        "achievement_rates": achievement_rates,
        "action_distribution": action_dist,
        "action_entropy": action_entropy,
        "termination_causes": termination_causes,
        "per_seed_results": per_seed_results,
    }


class RandomCrafterAgent:
    """Stochastic random baseline agent strictly respecting the pixel interface."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs_uint8: np.ndarray, reward_prev: float = 0.0, step: Optional[int] = None) -> int:
        return int(self.rng.integers(0, 17))
