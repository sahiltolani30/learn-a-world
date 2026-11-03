"""Strict post-episode evaluation for a true-JEPA checkpoint.

The policy receives only pixels, previous reward, and step.  Crafter's info is
read only after each action for scoring and is never passed to the agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig
from true_jepa.model import TrueJEPA


def evaluate_checkpoint(
    checkpoint: str,
    episodes: int = 8,
    max_steps: int = 500,
    seed_start: int = 2050,
    device: str = "cpu",
) -> Dict[str, object]:
    """Run frozen visual evaluation and return a JSON-safe score report."""

    import crafter

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "true_jepa_run_v1":
        raise ValueError("expected a true_jepa_run_v1 checkpoint")
    model = TrueJEPA(
        JEPAConfig(**payload["config"]),
        num_actions=int(payload["num_actions"]),
    )
    model.load_state_dict(payload["model"])
    model.target_encoder.load_state_dict(payload["target_encoder"])
    agent = VisualJEPAAgent(model, device=device, stochastic=False)
    episode_logs: List[Dict[str, object]] = []

    for offset in range(int(episodes)):
        seed = int(seed_start + offset)
        env = crafter.Env(seed=seed)
        obs = np.asarray(env.reset(), dtype=np.uint8)
        agent.reset()
        previous_reward = 0.0
        unlocked = set()
        total_reward = 0.0
        steps = 0
        try:
            while steps < int(max_steps):
                action = agent.act(obs, previous_reward, steps)
                next_obs, reward, done, info = env.step(action)
                obs = np.asarray(next_obs, dtype=np.uint8)
                previous_reward = float(reward)
                total_reward += float(reward)
                steps += 1
                for name, value in (info.get("achievements", {}) or {}).items():
                    if value:
                        unlocked.add(str(name))
                if done:
                    break
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        episode_logs.append(
            {
                "seed": seed,
                "steps": steps,
                "reward": total_reward,
                "achievement_count": len(unlocked),
                "achievements": sorted(unlocked),
            }
        )

    counts = [int(log["achievement_count"]) for log in episode_logs]
    return {
        "protocol": "true_jepa_visual_only_post_episode_scoring_v1",
        "checkpoint": str(checkpoint),
        "episodes": len(episode_logs),
        "mean_reward": float(np.mean([log["reward"] for log in episode_logs])),
        "mean_achievements": float(np.mean(counts)),
        "max_achievements": int(max(counts, default=0)),
        "episode_logs": episode_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=2050)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        device=args.device,
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
