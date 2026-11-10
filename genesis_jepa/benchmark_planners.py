"""
genesis_jepa.benchmark_planners — Empirical Permutation & Combination Benchmark
================================================================================
Compares decision architectures under identical Crafter seeds:
  1. reflex: Pure 2-layer MLP reactive policy (no lookahead)
  2. lookahead_1step: Pure 1-step JEPA mental simulation (Q = V(Predictor(s, a)))
  3. hybrid_mppi: Policy Actor Prior combined with JEPA 1-step lookahead
  4. multistep_rollout: Multi-step autoregressive horizon H=3 mental simulation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import crafter
except ImportError:
    crafter = None

from genesis_jepa.agent import GenesisVisualAgent
from genesis_jepa.config import GenesisConfig


def run_single_permutation(
    mode: str,
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 5,
    max_steps: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates a specific decision engine permutation on fixed Crafter seeds.
    """
    if crafter is None:
        raise ImportError("Crafter is not installed.")

    config = GenesisConfig(
        planning_mode=mode,
        plan_horizon=3,
        plan_temperature=1.0,
        plan_num_candidates=16,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = GenesisVisualAgent(config).to(dev)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=dev)
        agent.load_state_dict(ckpt["agent_state"])

    agent.eval()

    rewards: List[float] = []
    lengths: List[int] = []
    milestones: Dict[str, int] = {}
    latencies: List[float] = []
    night_survivals: int = 0

    for ep in range(num_episodes):
        env = crafter.Env(seed=seed + ep)
        obs = env.reset()
        done = False
        step = 0
        ep_reward = 0.0
        prev_rew = 0.0
        agent.reset()

        while not done and step < max_steps:
            t0 = time.time()
            action = agent.act(obs, reward_prev=prev_rew, step=step)
            latencies.append((time.time() - t0) * 1000.0)

            obs, rew, done, info = env.step(action)
            ep_reward += float(rew)
            prev_rew = float(rew)
            step += 1

        rewards.append(ep_reward)
        lengths.append(step)
        if step >= 220:
            night_survivals += 1

        for k, v in info.get("achievements", {}).items():
            if v > 0:
                milestones[k] = milestones.get(k, 0) + 1

        if hasattr(env, "close"):
            env.close()

    return {
        "mode": mode,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "night_survivals": night_survivals,
        "nightfall_rate": float(night_survivals / num_episodes) * 100.0,
        "mean_latency_ms": float(np.mean(latencies)),
        "milestones": milestones,
        "total_milestone_events": sum(milestones.values()),
    }


def run_full_permutation_benchmark(
    checkpoint_path: Optional[str] = None,
    num_episodes: int = 5,
    max_steps: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs permutations and combinations across all decision options.
    """
    modes = ["reflex", "lookahead_1step", "hybrid_mppi", "multistep_rollout"]
    results: Dict[str, Any] = {}

    print("\n" + "=" * 80)
    print("🔬 GENESIS JEPA: DECISION ARCHITECTURE PERMUTATION BENCHMARK")
    print(f"   Episodes per mode: {num_episodes} | Max steps: {max_steps} | Seed: {seed}")
    print(f"   Checkpoint: {checkpoint_path if checkpoint_path else 'Initialized (Zero-Shot)'}")
    print("=" * 80)

    for mode in modes:
        print(f"\n[Benchmarking Mode: {mode.upper()}] ...", flush=True)
        res = run_single_permutation(
            mode=mode,
            checkpoint_path=checkpoint_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        results[mode] = res
        print(
            f"  -> Reward: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f} | "
            f"Steps: {res['mean_length']:.1f} | "
            f"Latency: {res['mean_latency_ms']:.2f} ms/step | "
            f"Milestones: {list(res['milestones'].keys())}"
        )

    # Print Side-by-Side Comparative Table
    print("\n" + "=" * 80)
    print("📊 PERMUTATION BENCHMARK COMPARATIVE SUMMARY")
    print("=" * 80)
    print(f"{'Planning Mode':<20} | {'Mean Reward':<14} | {'Survival Steps':<14} | {'Nightfall':<10} | {'Latency':<12} | {'Milestone Count'}")
    print("-" * 80)
    for mode, r in results.items():
        print(
            f"{mode:<20} | "
            f"{r['mean_reward']:+.2f} +/- {r['std_reward']:.2f}  | "
            f"{r['mean_length']:<14.1f} | "
            f"{r['nightfall_rate']:<9.1f}% | "
            f"{r['mean_latency_ms']:<8.2f} ms   | "
            f"{r['total_milestone_events']} events ({list(r['milestones'].keys())})"
        )
    print("=" * 80 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Planner Permutations Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per mode")
    parser.add_argument("--max-steps", type=int, default=250, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON path")
    args, _ = parser.parse_known_args()

    results = run_full_permutation_benchmark(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
