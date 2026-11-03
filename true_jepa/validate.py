"""Validation gates for the visual JEPA before longer self-play runs.

The latent/reward checks use only replay observations, actions, rewards, and
termination masks.  The behavior comparison reads Crafter achievement info
only after an episode for reporting; it is never passed to the agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig
from true_jepa.model import TrueJEPA
from true_jepa.replay import PixelReplay


def _load_model(checkpoint: str, device: torch.device) -> TrueJEPA:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "true_jepa_run_v1":
        raise ValueError("expected a true_jepa_run_v1 checkpoint")
    model = TrueJEPA(
        JEPAConfig(**payload["config"]),
        num_actions=int(payload["num_actions"]),
    )
    model.load_state_dict(payload["model"])
    model.target_encoder.load_state_dict(payload["target_encoder"])
    model.to(device).eval()
    return model


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def validate_replay(
    model: TrueJEPA,
    replay: PixelReplay,
    device: torch.device,
    batches: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> Dict[str, object]:
    """Compare real-action predictions and shuffled-action predictions."""

    rng = np.random.default_rng(seed)
    reward_values = np.concatenate(
        [episode.rewards.astype(np.float32) for episode in replay.records]
    )
    reward_baseline = float(reward_values.mean())
    real_latent_errors: List[float] = []
    shuffled_latent_errors: List[float] = []
    reward_errors: List[float] = []
    reward_baseline_errors: List[float] = []
    target_latents: List[torch.Tensor] = []
    valid_transitions = 0

    for _ in range(int(batches)):
        batch = replay.sample(batch_size, sequence_length, rng)
        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        valid = batch["valid"].to(device)
        hidden = model.initial_belief(batch_size, device)
        previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
        previous_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)

        for time in range(sequence_length):
            frame = model.encode(obs[:, time])
            if time > 0:
                hidden = model.belief_step(frame, previous_action, previous_reward, hidden)
            target = model.target_encode(obs[:, time + 1])
            real_prediction = model.predict(frame, actions[:, time], hidden)
            shuffled_actions = actions[:, time].roll(1)
            shuffled_prediction = model.predict(frame, shuffled_actions, hidden)
            reward_prediction, _ = model.transition_heads(frame, hidden, actions[:, time])
            mask = valid[:, time]
            if not bool(mask.any()):
                previous_action = actions[:, time]
                previous_reward = rewards[:, time]
                continue

            real_error = (real_prediction.pooled - target.pooled).square().mean(dim=-1)
            shuffled_error = (shuffled_prediction.pooled - target.pooled).square().mean(dim=-1)
            reward_error = (reward_prediction - rewards[:, time]).square()
            baseline_error = (torch.full_like(rewards[:, time], reward_baseline) - rewards[:, time]).square()
            real_latent_errors.extend(real_error[mask].float().cpu().tolist())
            shuffled_latent_errors.extend(shuffled_error[mask].float().cpu().tolist())
            reward_errors.extend(reward_error[mask].float().cpu().tolist())
            reward_baseline_errors.extend(baseline_error[mask].float().cpu().tolist())
            target_latents.append(target.pooled[mask].float().cpu())
            valid_transitions += int(mask.sum().item())
            previous_action = actions[:, time]
            previous_reward = rewards[:, time]

    real_mse = _mean(real_latent_errors)
    shuffled_mse = _mean(shuffled_latent_errors)
    reward_mse = _mean(reward_errors)
    reward_baseline_mse = _mean(reward_baseline_errors)
    target_std = float(torch.cat(target_latents, dim=0).std().item()) if target_latents else float("nan")
    return {
        "replay_episodes": replay.episodes,
        "replay_frames": len(replay),
        "valid_transitions": valid_transitions,
        "action_conditioning": {
            "real_latent_mse": real_mse,
            "shuffled_action_latent_mse": shuffled_mse,
            "shuffled_minus_real": shuffled_mse - real_mse,
            "passes": bool(shuffled_mse > real_mse),
        },
        "reward_prediction": {
            "model_mse": reward_mse,
            "constant_baseline_mse": reward_baseline_mse,
            "baseline_minus_model": reward_baseline_mse - reward_mse,
            "passes": bool(reward_baseline_mse > reward_mse),
        },
        "representation": {
            "target_latent_std": target_std,
            "passes_noncollapse_screen": bool(np.isfinite(target_std) and target_std > 0.1),
        },
    }


def _episode_score(env: object, action_fn, max_steps: int) -> Dict[str, object]:
    obs = np.asarray(env.reset(), dtype=np.uint8)
    previous_reward = 0.0
    total_reward = 0.0
    achievements = set()
    steps = 0
    while steps < int(max_steps):
        action = int(action_fn(obs, previous_reward, steps))
        next_obs, reward, done, info = env.step(action)
        obs = np.asarray(next_obs, dtype=np.uint8)
        previous_reward = float(reward)
        total_reward += float(reward)
        steps += 1
        for name, value in (info.get("achievements", {}) or {}).items():
            if value:
                achievements.add(str(name))
        if done:
            break
    return {
        "reward": total_reward,
        "achievement_count": len(achievements),
        "achievements": sorted(achievements),
        "steps": steps,
    }


def validate_behavior(
    model: TrueJEPA,
    episodes: int,
    max_steps: int,
    seed_start: int,
    device: torch.device,
) -> Dict[str, object]:
    """Compare a frozen visual policy with random actions on identical seeds."""

    import crafter

    agent = VisualJEPAAgent(model, device=str(device), stochastic=False)
    learned: List[Dict[str, object]] = []
    random: List[Dict[str, object]] = []
    for offset in range(int(episodes)):
        seed = int(seed_start + offset)
        learned_env = crafter.Env(seed=seed)
        random_env = crafter.Env(seed=seed)
        try:
            agent.reset()
            learned.append(_episode_score(learned_env, agent.act, max_steps))
            rng = np.random.default_rng(seed + 1_000_003)
            random.append(
                _episode_score(
                    random_env,
                    lambda _obs, _reward, _step: rng.integers(0, random_env.action_space.n),
                    max_steps,
                )
            )
        finally:
            for env in (learned_env, random_env):
                close = getattr(env, "close", None)
                if close is not None:
                    close()

    def summary(records: List[Dict[str, object]]) -> Dict[str, float]:
        return {
            "mean_reward": _mean([float(record["reward"]) for record in records]),
            "mean_achievements": _mean([float(record["achievement_count"]) for record in records]),
        }

    return {
        "episodes": int(episodes),
        "learned": summary(learned),
        "random": summary(random),
        "learned_episode_logs": learned,
        "random_episode_logs": random,
    }


def validate_checkpoint(
    checkpoint: str,
    replay_path: str,
    batches: int = 32,
    batch_size: int = 8,
    sequence_length: int = 16,
    episodes: int = 30,
    max_steps: int = 500,
    seed_start: int = 2050,
    device: str = "cpu",
) -> Dict[str, object]:
    torch_device = torch.device(device)
    model = _load_model(checkpoint, torch_device)
    replay = PixelReplay.load(replay_path)
    return {
        "protocol": "true_jepa_validation_v1",
        "checkpoint": str(checkpoint),
        "replay": validate_replay(
            model,
            replay,
            torch_device,
            batches,
            batch_size,
            sequence_length,
            seed_start,
        ),
        "behavior": validate_behavior(model, episodes, max_steps, seed_start, torch_device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=2050)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = validate_checkpoint(
        checkpoint=args.checkpoint,
        replay_path=args.replay,
        batches=args.batches,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        device=args.device,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
