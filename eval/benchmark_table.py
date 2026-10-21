"""
Phase 2 Benchmark Table Generator for Learn-a-World.
Compares:
1. Random Agent
2. World Model + MPC Planner
3. WM + MPC + Curiosity
4. PPO Baseline
Generates formatted Markdown comparison tables with measured metrics on fixed seeds.
"""

from pathlib import Path
from typing import Dict, List, Optional
import gymnasium as gym
import numpy as np
import torch

import envs  # registers environments
from baselines.ppo_baseline import train_ppo
from experiments.utils import get_device, load_config
from models.curiosity import CuriosityModule
from models.jepa import JEPAWorldModel
from planning.mpc import LatentMPCPlanner
from training.buffer import ReplayBuffer
from training.self_play import SelfPlayAgent, evaluate_agent
from training.trainer import train_jepa

def run_phase2_benchmarks(
    config_path: str = "experiments/configs/maze.yaml",
    checkpoint_path: str = "checkpoints/jepa_maze.pt",
    eval_episodes: int = 20,
    ppo_timesteps: int = 5000,
    device_override: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates all four agents and returns comparative performance metrics.
    """
    device = get_device(device_override)
    cfg = load_config(config_path)
    env_name = cfg["env"]["name"]
    env = gym.make(env_name)

    print("==================================================")
    print("      Running Phase 2 Benchmark Evaluations       ")
    print(f"      Environment: {env_name} | Device: {device} ")
    print("==================================================")

    results: Dict[str, Dict[str, float]] = {}

    # 1. Random Agent
    print("\n[1/4] Evaluating Random Baseline...")
    rand_res = evaluate_agent(env, planner=None, num_episodes=eval_episodes, random_policy=True)
    results["Random Agent"] = {
        "success_rate": rand_res["success_rate"],
        "avg_steps": rand_res["avg_steps"],
        "budget": 0,
    }
    print(f"      Random Success: {rand_res['success_rate']:.1f}% | Steps: {rand_res['avg_steps']:.1f}")

    # 2. World Model + MPC Planner
    print("\n[2/4] Evaluating World Model + MPC Planner...")
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print("      No checkpoint found, training Phase 1 model...")
        model, _ = train_jepa(config_path=config_path, device_override=str(device))
    else:
        print(f"      Loading trained model from {checkpoint_path}...")
        m_cfg = cfg["model"]
        model = JEPAWorldModel(
            in_channels=3,
            latent_dim=m_cfg.get("latent_dim", 128),
            num_actions=m_cfg.get("num_actions", 4),
            action_emb_dim=m_cfg.get("action_emb_dim", 32),
            predictor_hidden_dim=m_cfg.get("predictor_hidden_dim", 256),
            predictor_layers=m_cfg.get("predictor_layers", 2),
            ema_decay=m_cfg.get("ema_decay", 0.99),
        ).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    planner = LatentMPCPlanner(
        model=model,
        horizon=cfg["planning"].get("horizon", 5),
        num_candidates=cfg["planning"].get("num_candidates", 64),
        elite_ratio=cfg["planning"].get("elite_ratio", 0.1),
        device=device,
    )

    wm_res = evaluate_agent(env, planner=planner, num_episodes=eval_episodes, random_policy=False)
    results["World Model + MPC"] = {
        "success_rate": wm_res["success_rate"],
        "avg_steps": wm_res["avg_steps"],
        "budget": 2000,  # 2000 seed transitions used in training
    }
    print(f"      WM+MPC Success: {wm_res['success_rate']:.1f}% | Steps: {wm_res['avg_steps']:.1f}")

    # 3. WM + MPC + Curiosity Self-Play Exploration
    print("\n[3/4] Evaluating WM + MPC + Curiosity...")
    curiosity = CuriosityModule(model=model, eta=0.3, device=device)
    curiosity_buffer = ReplayBuffer(capacity=10000, obs_shape=(64, 64, 3), device=device)
    self_play_agent = SelfPlayAgent(
        env=env,
        model=model,
        buffer=curiosity_buffer,
        planner=planner,
        curiosity=curiosity,
        epsilon=0.1,
        device=device,
    )
    # Run 10 curiosity self-play exploration episodes
    for _ in range(10):
        self_play_agent.run_episode(max_steps=100)

    curiosity_res = evaluate_agent(env, planner=planner, num_episodes=eval_episodes)
    results["WM + MPC + Curiosity"] = {
        "success_rate": curiosity_res["success_rate"],
        "avg_steps": curiosity_res["avg_steps"],
        "budget": 2000 + len(curiosity_buffer),
    }
    print(f"      Curiosity Success: {curiosity_res['success_rate']:.1f}% | Steps: {curiosity_res['avg_steps']:.1f}")

    # 4. PPO Baseline
    print("\n[4/4] Training & Evaluating PPO Baseline...")
    _, ppo_res = train_ppo(
        env_name=env_name,
        total_timesteps=ppo_timesteps,
        num_eval_episodes=eval_episodes,
        device=str(device) if device.type != "mps" else "cpu",  # SB3 CNNPolicy on CPU/CUDA
    )
    results["PPO Baseline"] = {
        "success_rate": ppo_res["success_rate"],
        "avg_steps": ppo_res["avg_steps"],
        "budget": ppo_timesteps,
    }

    env.close()

    # Format Markdown Table
    md_table = format_markdown_table(results)
    print("\n" + md_table)

    # Save to assets
    out_file = Path("docs/assets/phase2_benchmark_table.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"\n[SAVED] Table saved to: {out_file.resolve()}")

    return results

def format_markdown_table(results: Dict[str, Dict[str, float]]) -> str:
    """Formats benchmark results dictionary into clean Markdown table."""
    lines = [
        "| Agent Method | Success Rate (%) | Avg Steps to Goal | Env Interaction Budget |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for method, metrics in results.items():
        lines.append(
            f"| **{method}** | **{metrics['success_rate']:.1f}%** | {metrics['avg_steps']:.1f} | {int(metrics['budget'])} steps |"
        )
    return "\n".join(lines)

if __name__ == "__main__":
    run_phase2_benchmarks()
