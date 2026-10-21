"""
PPO Baseline Agent for Learn-a-World.
Uses Stable-Baselines3 PPO with CNN policy to provide a matched-budget RL benchmark.
"""

import argparse
from typing import Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import envs  # registers environments

def train_ppo(
    env_name: str = "GridMaze-v0",
    total_timesteps: int = 5000,
    seed: int = 42,
    device: str = "auto",
    num_eval_episodes: int = 20,
) -> Tuple[PPO, Dict[str, float]]:
    """
    Trains a PPO agent with CnnPolicy on the specified environment
    and evaluates success rate and average steps on fixed seeds.
    """
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        seed=seed,
        device=device,
        verbose=0,
    )

    print(f"--> Training PPO baseline on {env_name} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # Evaluate on fixed seeds
    successes = 0
    steps_list = []

    for ep in range(num_eval_episodes):
        obs, info = eval_env.reset(seed=1000 + ep)
        ep_success = False
        steps = 0

        for _ in range(100):
            steps += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            if terminated:
                ep_success = True
                break
            if truncated:
                break

        if ep_success:
            successes += 1
            steps_list.append(steps)
        else:
            steps_list.append(100)

    env.close()
    eval_env.close()

    metrics = {
        "success_rate": (successes / num_eval_episodes) * 100.0,
        "avg_steps": float(np.mean(steps_list)),
        "total_timesteps": total_timesteps,
    }
    print(f"    [PPO RESULT] Success: {metrics['success_rate']:.1f}% | Avg Steps: {metrics['avg_steps']:.1f}")
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Train PPO Baseline")
    parser.add_argument("--env", type=str, default="GridMaze-v0", help="Gym environment name")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total interaction timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_ppo(
        env_name=args.env,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
