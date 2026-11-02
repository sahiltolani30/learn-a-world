"""
sapna/benchmark.py

Crafter benchmarking and official scoring per SAPNA_PLAN.md §9.
Includes verbatim compute_crafter_score from Hafner (2021).
"""

from typing import Any, Dict, List, Optional, Union
import json
import os
from pathlib import Path
import numpy as np
import torch

from sapna.envs import make_crafter_env
from sapna.policy import SAPNAAgent


ALL_CRAFTER_ACHIEVEMENTS: List[str] = [
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

# Action indices in Crafter
MOVEMENT_ACTIONS = {1, 2, 3, 4}  # move_left, move_right, move_up, move_down


def compute_crafter_score(success_rates: Dict[str, float]) -> float:
    """
    Official Crafter Score formula (Hafner, 2021):
    Score = exp( (1/22) * sum_{i=1}^{22} ln(1 + S_i) ) - 1
    where S_i is in percent [0.0, 100.0].
    Copied verbatim from src/diagnostics.py:67-77.
    """
    log_sum = 0.0
    for ach in ALL_CRAFTER_ACHIEVEMENTS:
        s_i = success_rates.get(ach, 0.0)
        log_sum += np.log(1.0 + s_i)
    return float(np.exp(log_sum / float(len(ALL_CRAFTER_ACHIEVEMENTS))) - 1.0)


def run_benchmark(
    agent: SAPNAAgent,
    n: int = 50,
    policy: str = "greedy",
    max_steps: int = 10000,
    seed_base: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run N benchmark episodes with agent facade and compute official Crafter score.
    
    Args:
        agent: SAPNAAgent facade instance.
        n: Number of episodes to evaluate (default 50).
        policy: 'greedy' or 'stochastic'.
        max_steps: Maximum episode length (default 10000).
        seed_base: Base seed for deterministic evaluation.
        save_path: Optional path to save JSON results.
        
    Returns:
        dict containing official score, per-achievement success rates, returns, lengths, movement%.
    """
    greedy = (policy == "greedy")
    achievement_counts = {ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS}
    episode_returns = []
    episode_lengths = []
    unique_counts = []
    action_counts = np.zeros(17, dtype=np.int64)

    for ep_idx in range(n):
        env = make_crafter_env(seed=seed_base + ep_idx * 1000, length=max_steps)
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res

        # Hotfix-4 lesson: maintain eval_h per running episode!
        h = agent.reset_h(batch_size=1)
        ep_return = 0.0
        ep_len = 0
        ep_unlocked = set()

        done = False
        while not done and ep_len < max_steps:
            action, h, _ = agent.act(obs, h, greedy=greedy)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

            # Collect unlocked achievements from safe eval_info
            eval_info = info.get("eval_info", info)
            unlocked = eval_info.get("unlocked_achievements", [])
            for ach in unlocked:
                ep_unlocked.add(ach)

        env.close()

        for ach in ep_unlocked:
            achievement_counts[ach] += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        unique_counts.append(len(ep_unlocked))

    # Calculate percentage success rates [0, 100]
    success_rates = {
        ach: (achievement_counts[ach] / float(n)) * 100.0
        for ach in ALL_CRAFTER_ACHIEVEMENTS
    }
    score = compute_crafter_score(success_rates)

    total_actions = max(1, int(action_counts.sum()))
    movement_actions = sum(action_counts[a] for a in MOVEMENT_ACTIONS)
    movement_pct = float(movement_actions / total_actions) * 100.0

    report = {
        "score": score,
        "policy": policy,
        "episodes": n,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_unique_achievements": float(np.mean(unique_counts)),
        "movement_percent": movement_pct,
        "success_rates": success_rates,
        "action_histogram": action_counts.tolist(),
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
