"""P1 baseline training and evaluation runner.

The runner deliberately reports environment interaction metrics only. Crafter
achievement annotations remain outside the policy path and must be collected
by the isolated offline evaluator in a later benchmark step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .agent import RandomVisualPolicy, SB3VisualPolicy
from .checkpoint import save_atomic
from .env_boundary import CrafterPixelEnv, GymPixelEnv


@dataclass(frozen=True)
class BaselineConfig:
    algorithm: str = "ppo"
    train_steps: int = 10_000
    eval_episodes: int = 5
    eval_seed_start: int = 900_000
    train_seed: int = 12345
    max_episode_steps: int = 1000
    permutation_seed: int = 12345
    device: str = "auto"
    output_dir: str = "runs/mosaic/p1_baseline"
    num_envs: int = 1


def _require_sb3() -> Any:
    try:
        import stable_baselines3
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError(
            "P1 PPO requires stable-baselines3; install the repository requirements"
        ) from exc
    return stable_baselines3


def _make_vec_env(config: BaselineConfig):
    sb3 = _require_sb3()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    factories = []
    for offset in range(int(config.num_envs)):
        env_seed = int(config.train_seed) + offset

        def factory(seed=env_seed):
            return GymPixelEnv(
                seed=seed,
                max_episode_steps=config.max_episode_steps,
                permutation_seed=config.permutation_seed,
            )

        factories.append(factory)
    vector_env = DummyVecEnv(factories)
    if vector_env.observation_space.shape[-1] == 3:
        vector_env = VecTransposeImage(vector_env)
    return vector_env


def train_model(config: BaselineConfig):
    """Train the selected P1 PPO control and return the model and env."""

    if config.algorithm == "ppo":
        sb3 = _require_sb3()
        policy = "CnnPolicy"
        trainer = sb3.PPO
    elif config.algorithm == "recurrent_ppo":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "recurrent_ppo requires sb3-contrib; install mosaic/requirements.txt"
            ) from exc
        trainer = RecurrentPPO
        policy = "CnnLstmPolicy"
    else:
        raise ValueError(f"training is not defined for algorithm={config.algorithm!r}")

    env = _make_vec_env(config)
    model = trainer(
        policy,
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        seed=config.train_seed,
        device=config.device,
        verbose=0,
    )
    model.learn(total_timesteps=int(config.train_steps), progress_bar=False)
    return model, env


def evaluate_policy(
    policy: Any,
    config: BaselineConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate through the four-value pixel-only boundary."""

    rewards = []
    lengths = []
    terminated_count = 0
    mapping_manifest = None

    for episode_offset in range(int(config.eval_episodes)):
        seed = int(config.eval_seed_start) + episode_offset
        env = CrafterPixelEnv(
            seed=seed,
            max_episode_steps=config.max_episode_steps,
            permutation_seed=config.permutation_seed,
        )
        mapping_manifest = env.action_mapping.provenance()
        if hasattr(policy, "num_actions") and int(policy.num_actions) != env.num_actions:
            raise ValueError("policy action count does not match discovered environment action count")
        try:
            observation = env.reset(seed=seed)
            previous_reward = 0.0
            total_reward = 0.0
            step = 0
            while True:
                action = policy.act(observation, previous_reward, step)
                observation, reward, terminated, truncated = env.step(action)
                total_reward += float(reward)
                previous_reward = float(reward)
                step += 1
                if terminated or truncated:
                    terminated_count += int(terminated)
                    rewards.append(total_reward)
                    lengths.append(step)
                    break
        finally:
            env.close()

    metrics: Dict[str, Any] = {
        "episodes": int(config.eval_episodes),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "terminated_fraction": float(terminated_count / max(1, len(lengths))),
        "rewards": [float(value) for value in rewards],
        "lengths": [int(value) for value in lengths],
        "mapping": mapping_manifest,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_baseline(config: BaselineConfig) -> Dict[str, Any]:
    """Run a random or PPO baseline and write a resumable manifest."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_env = CrafterPixelEnv(
        seed=config.train_seed,
        max_episode_steps=config.max_episode_steps,
        permutation_seed=config.permutation_seed,
    )
    discovered_actions = probe_env.num_actions
    probe_env.close()
    if config.algorithm == "random":
        policy = RandomVisualPolicy(num_actions=discovered_actions, seed=config.train_seed)
        metrics = evaluate_policy(policy, config, output_dir)
        model = None
    else:
        model, vector_env = train_model(config)
        model_path = output_dir / "policy"
        model.save(str(model_path))
        metrics = evaluate_policy(
            SB3VisualPolicy(model, discovered_actions), config, output_dir
        )
        vector_env.close()

    result = {"format": "mosaic_p1_baseline_v1", "config": asdict(config), "metrics": metrics}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    save_atomic(
        output_dir / "runtime.pt",
        {"format": "mosaic_runtime_v1", "phase": "p1_baseline", "result": result},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MOSAIC P1 baseline ladder")
    parser.add_argument("--algorithm", choices=("random", "ppo", "recurrent_ppo"), default="ppo")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed-start", type=int, default=900_000)
    parser.add_argument("--train-seed", type=int, default=12345)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/mosaic/p1_baseline")
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    config = BaselineConfig(**vars(args))
    print(json.dumps(run_baseline(config), indent=2))


if __name__ == "__main__":
    main()
