"""
Pure v2 Crafter Evaluator for Learn-a-World.
Features:
- Pure v2 Agent: No handcoded if-else reflexes, no info leaks.
- Uses Target Encoder + AgentBelief + RewardValuePlanner.
- Logs debug decomposition: {reward_sum, terminal_value, uncertainty_penalty}.
- Reports official Crafter geometric mean benchmark score and tier unlocks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import gymnasium as gym
import numpy as np
import torch

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import envs  # registers environments
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from experiments.utils import get_device, load_config
from models.jepa import JEPAWorldModel
from models.jepa_v2 import JEPAv2
from planning.agent_v2 import AgentBelief
from planning.mpc_v2 import RewardValuePlanner

def evaluate_v2(
    checkpoint_path: Optional[str] = None,
    config_path: str = "experiments/configs/crafter_v2.yaml",
    num_episodes: int = 20,
    seed_start: int = 1000,
    max_steps: int = 1000,
    device_override: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates pure v2 agent across episodes on held-out seeds.
    """
    device = get_device(device_override)
    cfg = load_config(config_path)

    env_cfg = cfg.get("env", {})
    env = gym.make(
        env_cfg.get("name", "Crafter-v0"),
        render_size=env_cfg.get("render_size", 64),
        max_steps=max_steps,
    )

    print("======================================================================")
    print("           Learn-a-World: Pure v2 Benchmark Evaluation                ")
    print(f"      Episodes: {num_episodes} | Seeds: {seed_start}..{seed_start + num_episodes - 1} | Device: {device} ")
    print("======================================================================")

    # Instantiate model
    m_cfg = cfg.get("model", {})
    h_cfg = cfg.get("heads", {})
    p_cfg = cfg.get("planner_v2", {})

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"--> Loading trained v2 model from: {checkpoint_path}")
        # Try loading as complete v2 checkpoint; fallback to v1 checkpoint
        try:
            base_model = JEPAWorldModel(
                in_channels=3,
                latent_dim=m_cfg.get("latent_dim", 128),
                num_actions=m_cfg.get("num_actions", 17),
                action_emb_dim=m_cfg.get("action_emb_dim", 32),
                predictor_hidden_dim=m_cfg.get("predictor_hidden_dim", 256),
                predictor_layers=m_cfg.get("predictor_layers", 2),
            ).to(device)
            model = JEPAv2(
                base_model=base_model,
                hidden_dim=h_cfg.get("hidden_dim", 256),
                action_emb_dim=m_cfg.get("action_emb_dim", 32),
                q_heads=h_cfg.get("q_ensemble", 3),
                q_bins=h_cfg.get("q_bins", 51),
                q_max=h_cfg.get("q_max", 12.0),
            ).to(device)
            model.load_checkpoint(checkpoint_path, device=device)
        except Exception:
            model = JEPAv2.from_v1_checkpoint(
                checkpoint_path=checkpoint_path,
                config=cfg,
                device=device,
                freeze_base=True,
            )
    else:
        print("--> Warning: No checkpoint found or provided. Initializing random v2 model for smoke eval.")
        base_model = JEPAWorldModel(
            in_channels=3,
            latent_dim=m_cfg.get("latent_dim", 128),
            num_actions=m_cfg.get("num_actions", 17),
            action_emb_dim=m_cfg.get("action_emb_dim", 32),
            predictor_hidden_dim=m_cfg.get("predictor_hidden_dim", 256),
            predictor_layers=m_cfg.get("predictor_layers", 2),
        ).to(device)
        model = JEPAv2(
            base_model=base_model,
            hidden_dim=h_cfg.get("hidden_dim", 256),
            action_emb_dim=m_cfg.get("action_emb_dim", 32),
            q_heads=h_cfg.get("q_ensemble", 3),
            q_bins=h_cfg.get("q_bins", 51),
            q_max=h_cfg.get("q_max", 12.0),
        ).to(device)

    model.eval()

    planner = RewardValuePlanner(
        model=model,
        horizon=p_cfg.get("horizon", 10),
        num_candidates=p_cfg.get("num_candidates", 256),
        cem_iterations=p_cfg.get("cem_iterations", 3),
        elite_ratio=p_cfg.get("elite_ratio", 0.15),
        gamma=cfg.get("loss", {}).get("gamma", 0.99),
        uncertainty_weight=p_cfg.get("uncertainty_weight", 0.5),
        novelty_weight=p_cfg.get("novelty_weight", 0.02),
        prior_mix=p_cfg.get("prior_mix", 0.5),
        num_actions=m_cfg.get("num_actions", 17),
        device=device,
    )

    belief = AgentBelief()

    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    all_reward_sums: List[float] = []
    all_terminal_vals: List[float] = []
    all_uncertainty_pens: List[float] = []
    all_scores: List[float] = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        belief.reset()

        ep_reward = 0.0
        steps = 0

        for step in range(max_steps):
            steps += 1
            # STRICT ZERO INFO LEAK: Planner receives only obs and belief
            action, debug = planner.plan(obs, belief)

            all_reward_sums.append(debug["reward_sum"])
            all_terminal_vals.append(debug["terminal_value"])
            all_uncertainty_pens.append(debug["uncertainty_penalty"])
            all_scores.append(debug["best_score"])

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Update belief using EventHead prediction on the transition taken
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                z_t = model.encode(obs_t.to(device), use_target=True)
                act_t = torch.tensor([action], device=device, dtype=torch.long)
                mask_t = belief.get_mask_tensor(device=device).unsqueeze(0)
                event_probs = model.event_head.predict_proba(z_t, act_t, mask_t).squeeze(0)

            belief.update(event_probs, reward)
            obs = next_obs

            if done:
                break

        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

    env.close()

    score_stats = compute_crafter_score(achievement_records)
    crafter_score = score_stats["score"]
    unlocked_count = score_stats["unlocked_count"]

    results = {
        "crafter_score": float(round(crafter_score, 4)),
        "unlocked_count": unlocked_count,
        "mean_survival_steps": float(round(float(np.mean(survival_lengths)), 2)),
        "mean_reward": float(round(float(np.mean(episode_rewards)), 2)),
        "num_episodes": num_episodes,
        "unlock_percentages": score_stats["unlock_percentages"],
        "debug_decomposition": {
            "mean_reward_sum": float(round(float(np.mean(all_reward_sums)), 4)),
            "mean_terminal_value": float(round(float(np.mean(all_terminal_vals)), 4)),
            "mean_uncertainty_penalty": float(round(float(np.mean(all_uncertainty_pens)), 4)),
            "mean_best_score": float(round(float(np.mean(all_scores)), 4)),
            "reward_sum_variance": float(round(float(np.var(all_reward_sums)), 6)),
        },
        "metadata": {
            "agent": "v2_pure",
            "checkpoint": checkpoint_path,
            "seed_start": seed_start,
            "num_episodes": num_episodes,
        },
    }

    print("\n" + "=" * 70)
    print("                     EVALUATION SCORECARD (v2 pure)")
    print("=" * 70)
    print(f"  Official Crafter Score:    {crafter_score:.2f}%")
    print(f"  Achievements Unlocked:     {unlocked_count}/22")
    print(f"  Mean Survival Steps:       {results['mean_survival_steps']:.1f}")
    print(f"  Mean Reward / Episode:     {results['mean_reward']:.2f}")
    print("  Debug Decomposition:")
    print(f"    Mean Reward Sum:         {results['debug_decomposition']['mean_reward_sum']:.4f}")
    print(f"    Mean Terminal Value:     {results['debug_decomposition']['mean_terminal_value']:.4f}")
    print(f"    Mean Uncertainty Pen:    {results['debug_decomposition']['mean_uncertainty_penalty']:.4f}")
    print(f"    Reward Sum Variance:     {results['debug_decomposition']['reward_sum_variance']:.6f}")
    print("=" * 70)

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"--> Saved evaluation results to: {p}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure v2 Crafter Agent")
    parser.add_argument("--checkpoint-v2", type=str, default="checkpoints/jepa_crafter_v2.pt",
                        help="Path to trained v2 checkpoint")
    parser.add_argument("--config", type=str, default="experiments/configs/crafter_v2.yaml",
                        help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed-start", type=int, default=1000,
                        help="Starting seed offset")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override")
    parser.add_argument("--out", type=str, default="results/v2_step2.json",
                        help="Output JSON path")
    args = parser.parse_args()

    evaluate_v2(
        checkpoint_path=args.checkpoint_v2,
        config_path=args.config,
        num_episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        device_override=args.device,
        out_path=args.out,
    )

if __name__ == "__main__":
    main()
