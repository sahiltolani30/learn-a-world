#!/usr/bin/env python3
"""
50-Episode Gameplay Screening of JEPA v2 Champion on Untouched Seeds (2050-2099).
Provides the exact, matched head-to-head baseline against JEPA v3 on the identical screening seeds.

Features:
- Pure learned JEPA v2 model (checkpoints/jepa_crafter_v2_best.pt)
- RewardValuePlanner (H=10, C=64, CEM=2) matching v3 screening compute budget
- AgentBelief (achievement mask tracking)
- Strict zero-information leakage

Outputs:
- results/screen_v2_50eps_results.json
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # registers Crafter-v0
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.jepa import JEPAWorldModel
from models.jepa_v2 import JEPAv2
from planning.agent_v2 import AgentBelief
from planning.mpc_v2 import RewardValuePlanner


def run_v2_screening(
    checkpoint_path: str = "checkpoints/jepa_crafter_v2_best.pt",
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    prior_mix: float = 0.5,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.02,
    device: str = "cpu",
    output_json: str = "results/screen_v2_50eps_results.json",
) -> Dict[str, Any]:
    dev = torch.device(device)

    print("=" * 80)
    print("      LEARN-A-WORLD v2 CHAMPION: 50-EPISODE SCREENING (SEEDS 2050-2099)")
    print(f"      Checkpoint: {checkpoint_path}")
    print(f"      Episodes: {num_episodes} | Seeds: {seed_start}..{seed_start + num_episodes - 1}")
    print(f"      Planner: RewardValuePlanner (H={horizon}, C={num_candidates}, CEM={cem_iterations})")
    print(f"      Device: {dev}")
    print("=" * 80)

    # 1. Load v2 Model
    base_model = JEPAWorldModel(
        in_channels=3,
        latent_dim=128,
        num_actions=17,
        action_emb_dim=32,
        predictor_hidden_dim=256,
        predictor_layers=2,
    ).to(dev)

    model = JEPAv2(
        base_model=base_model,
        hidden_dim=256,
        action_emb_dim=32,
        q_heads=3,
        q_bins=51,
        q_max=12.0,
    ).to(dev)

    model.load_checkpoint(checkpoint_path, device=dev)
    model.eval()

    # 2. Build Planner & Belief
    planner = RewardValuePlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=cem_iterations,
        elite_ratio=0.15,
        gamma=0.99,
        uncertainty_weight=uncertainty_weight,
        novelty_weight=novelty_weight,
        prior_mix=prior_mix,
        num_actions=17,
        device=dev,
    )
    belief = AgentBelief()

    # 3. Environment
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    episode_records: List[Dict[str, Any]] = []
    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    t_start = time.time()

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()

        ep_reward = 0.0
        steps = 0
        t_ep_start = time.time()

        for step in range(max_steps):
            steps += 1

            action, debug = planner.plan(obs, belief)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Update belief using EventHead prediction
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(dev)
                z_t = model.encode(obs_t, use_target=True)
                act_t = torch.tensor([action], device=dev, dtype=torch.long)
                mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
                event_probs = model.event_head.predict_proba(z_t, act_t, mask_t).squeeze(0)

            belief.update(event_probs, reward)
            obs = next_obs

            if done:
                break

        ep_duration = time.time() - t_ep_start
        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        unlocked = [k for k, v in ach.items() if v > 0]

        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

        episode_records.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": steps,
            "reward": round(ep_reward, 2),
            "unlocked": unlocked,
            "duration_sec": round(ep_duration, 2),
        })

        print(
            f"Ep {ep_idx + 1:02d}/{num_episodes} (Seed {seed}) | "
            f"Steps: {steps:<4d} | Reward: {ep_reward:<5.1f} | "
            f"Achievements ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'None'} "
            f"({ep_duration:.1f}s)"
        )

    env.close()
    total_time = time.time() - t_start

    score_stats = compute_crafter_score(achievement_records)

    summary = {
        "checkpoint": checkpoint_path,
        "model": "v2_pure",
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "total_runtime_seconds": round(total_time, 2),
        "mean_survival_length": float(np.mean(survival_lengths)),
        "std_survival_length": float(np.std(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "crafter_score": score_stats["score"],
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "unlocked_achievement_count": score_stats["unlocked_count"],
        "unlock_percentages": score_stats["unlock_percentages"],
        "episodes": episode_records,
    }

    print("\n" + "=" * 80)
    print("               v2 CHAMPION SCREENING SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Time: {total_time:.1f}s | Mean Episode Time: {total_time / num_episodes:.1f}s")
    print(f"Mean Survival Length: {summary['mean_survival_length']:.1f} +/- {summary['std_survival_length']:.1f} steps")
    print(f"Mean Episode Reward: {summary['mean_episode_reward']:.2f} +/- {summary['std_episode_reward']:.2f}")
    print(f"Official Crafter Score: {summary['crafter_score']:.2f}%")
    print(f"Mean Achievements per Episode: {summary['mean_achievements_per_ep']:.2f}")
    print(f"Unique Achievements Unlocked: {summary['unlocked_achievement_count']} / 22")
    print("\nPer-Achievement Unlock Rates:")
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = score_stats["unlock_percentages"].get(ach, 0.0)
        bar = "#" * int(round(pct / 5))
        print(f"  {ach:<22}: {pct:5.1f}% | {bar}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--> Full v2 screening results saved to: {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-episode screening of JEPA v2 champion.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/screen_v2_50eps_results.json")
    args = parser.parse_args()

    run_v2_screening(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        device=args.device,
        output_json=args.output_json,
    )
