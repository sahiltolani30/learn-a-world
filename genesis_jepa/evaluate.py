"""
genesis_jepa.evaluate — Cleanroom Autonomous Crafter Evaluation Harness
=======================================================================
Governing Laws: AGENTS.md & GEMINI.md

Strict Research Integrity Laws:
1. agent.act(obs_uint8, reward_prev, step) is the ONLY communication channel into the agent.
2. Zero simulator state dicts, coordinates, inventory mappings, or cheat recipe tables.
3. Episode metrics (achievements, survival steps, Crafter score) are tracked externally
   by the evaluation harness solely for verification and scientific reporting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import crafter
except ImportError:
    crafter = None

from genesis_jepa.agent import GenesisVisualAgent
from genesis_jepa.config import GenesisConfig


def evaluate_agent(
    agent: Optional[GenesisVisualAgent] = None,
    config: Optional[GenesisConfig] = None,
    num_episodes: int = 5,
    max_steps_per_episode: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates GenesisVisualAgent in pure Crafter environment tabula rasa.
    Strictly unprivileged evaluation loop.
    """
    if crafter is None:
        raise ImportError("Crafter is not installed. Install via `pip install crafter`.")

    config = config or GenesisConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if agent is None:
        agent = GenesisVisualAgent(config).to(device)
    else:
        agent = agent.to(device)

    env = crafter.Env(seed=seed)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    achievements_all: Dict[str, int] = {}
    nightfall_survivals: int = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        reward_prev = 0.0
        agent.reset()

        while not done and step_count < max_steps_per_episode:
            # Strictly unprivileged act() invocation
            action = agent.act(obs, reward_prev=reward_prev, step=step_count)

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            reward_prev = float(reward)
            step_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        # Track external evaluation metrics
        if step_count >= 220:
            nightfall_survivals += 1

        ep_achievements = info.get("achievements", {})
        for ach_name, count in ep_achievements.items():
            if count > 0:
                achievements_all[ach_name] = achievements_all.get(ach_name, 0) + 1

        if verbose:
            print(
                f"[Episode {ep + 1}/{num_episodes}] "
                f"Steps: {step_count} | Reward: {total_reward:.2f} | "
                f"Survived Night: {step_count >= 220} | "
                f"Milestones: {[k for k, v in ep_achievements.items() if v > 0]}"
            )

        if hasattr(env, "close"):
            env.close()

    # Compute achievement unlock rates (%)
    achievement_rates = {
        ach: (count / num_episodes) * 100.0 for ach, count in achievements_all.items()
    }

    # Crafter score definition: geometric mean of achievement rates:
    # score = exp(mean(ln(1 + rate))) - 1
    rates_array = np.array(list(achievement_rates.values()) if achievement_rates else [0.0])
    crafter_score = float(np.exp(np.mean(np.log(1.0 + rates_array))) - 1.0) if len(rates_array) > 0 else 0.0

    results = {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "nightfall_survival_rate": float(nightfall_survivals / num_episodes) * 100.0,
        "achievement_rates": achievement_rates,
        "crafter_score": crafter_score,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Genesis JEPA Agent on Crafter")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args, unknown = parser.parse_known_args()
    if unknown:
        clean_unknown = [u for u in unknown if u.strip() not in ("n", "\\n", "")]
        if clean_unknown:
            print(f"[Warning] Ignoring unrecognized CLI arguments: {clean_unknown}")

    print(f"=== Evaluating Genesis Visual JEPA Agent on Crafter ({args.episodes} episodes) ===")
    results = evaluate_agent(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        verbose=True,
    )

    print("\n=== Evaluation Results Summary ===")
    print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} steps")
    print(f"Nightfall Survival Rate: {results['nightfall_survival_rate']:.1f}%")
    print(f"Crafter Score: {results['crafter_score']:.2f}%")
    print("Achievement Unlock Rates:")
    for ach, rate in sorted(results["achievement_rates"].items(), key=lambda x: -x[1]):
        print(f"  - {ach}: {rate:.1f}%")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to {args.output_json}")


if __name__ == "__main__":
    main()
