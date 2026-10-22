#!/usr/bin/env python3
"""
Diagnostic Script D2: Agent Death-Cause and Survival Analysis
Rolls out evaluation episodes using the trained JEPA v2 agent in Crafter
and tracks exact proximate causes of health loss and terminal state.

Categorizes deaths into:
- Starvation (food == 0)
- Dehydration (drink == 0)
- Zombie attack (zombie in melee range)
- Skeleton attack (hit by arrow / skeleton nearby)
- Night ambush (zombie/skeleton during night cycle)
- Other / Environmental damage
- Survival (reached max_steps without death)

Outputs:
- results/diagnostics/death_causes.json
- results/diagnostics/death_causes.png
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.crafter_wrapper import CrafterEnv
from models.jepa_v2 import JEPAv2
from planning.agent_v2 import AgentBelief
from planning.mpc_v2 import RewardValuePlanner


def determine_damage_cause(
    prev_inventory: Dict[str, int],
    curr_inventory: Dict[str, int],
    env_unwrapped: Any,
    step_num: int,
) -> str:
    """Attributes health loss at a transition to its underlying cause."""
    health_drop = prev_inventory.get("health", 9) - curr_inventory.get("health", 9)
    if health_drop <= 0:
        return "none"

    is_night = (step_num % 300) >= 150
    food = curr_inventory.get("food", 9)
    drink = curr_inventory.get("drink", 9)
    energy = curr_inventory.get("energy", 9)

    # Check proximity to monsters via Crafter world objects if available
    nearby_zombie = False
    nearby_skeleton = False
    nearby_arrow = False

    if hasattr(env_unwrapped, "_world") and hasattr(env_unwrapped._world, "objects"):
        player = getattr(env_unwrapped, "_player", None)
        if player is not None:
            p_pos = np.array(player.pos)
            for obj in env_unwrapped._world.objects:
                obj_name = obj.__class__.__name__.lower()
                dist = np.linalg.norm(np.array(obj.pos) - p_pos)
                if "zombie" in obj_name and dist <= 2.0:
                    nearby_zombie = True
                elif "skeleton" in obj_name and dist <= 5.0:
                    nearby_skeleton = True
                elif "arrow" in obj_name and dist <= 2.0:
                    nearby_arrow = True

    # Attribution hierarchy
    if nearby_zombie:
        return "zombie_night" if is_night else "zombie_day"
    if nearby_arrow or nearby_skeleton:
        return "skeleton_night" if is_night else "skeleton_day"
    if food <= 0 and drink <= 0:
        return "starvation_and_dehydration"
    if food <= 0:
        return "starvation"
    if drink <= 0:
        return "dehydration"
    if is_night:
        return "monster_night_unseen"
    return "environmental"


def run_death_diagnostics(
    model: JEPAv2,
    num_episodes: int = 25,
    seed_start: int = 2000,
    max_steps: int = 500,
    device: str = "cpu",
    horizon: int = 10,
    num_candidates: int = 64,
) -> Dict[str, Any]:
    """Runs evaluation rollouts and tracks death causes."""
    planner = RewardValuePlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=2,
        device=device,
    )

    cause_counts = {
        "starvation": 0,
        "dehydration": 0,
        "starvation_and_dehydration": 0,
        "zombie_night": 0,
        "zombie_day": 0,
        "skeleton_night": 0,
        "skeleton_day": 0,
        "monster_night_unseen": 0,
        "environmental": 0,
        "survived_max_steps": 0,
    }

    damage_event_counts = {k: 0 for k in cause_counts if k != "survived_max_steps"}
    episode_lengths = []
    death_steps = []
    night_deaths = 0
    day_deaths = 0

    episodes_summary = []

    print(f"[Death-Diagnostics] Evaluating {num_episodes} episodes (seeds {seed_start}..{seed_start + num_episodes - 1})...")

    for i in range(num_episodes):
        seed = seed_start + i
        env = CrafterEnv(seed=seed, max_steps=max_steps)
        obs, _ = env.reset(seed=seed)
        belief = AgentBelief()

        prev_inventory = {"health": 9, "food": 9, "drink": 9, "energy": 9}
        ep_damage_log = []
        fatal_cause = None

        step = 0
        done = False

        while not done and step < max_steps:
            action, _ = planner.plan(obs, belief)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            curr_inv = info.get("inventory", {})

            # Epistemic belief update
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(device)
                z = model.encode(obs_t, use_target=True)
                action_t = torch.tensor([action], device=device)
                mask_t = belief.get_mask_tensor(device=device).unsqueeze(0)
                p_event = model.event_head.predict_proba(z, action_t, mask_t).squeeze(0)
            belief.update(p_event, reward)

            # Check for damage
            dmg_cause = determine_damage_cause(
                prev_inventory=prev_inventory,
                curr_inventory=curr_inv,
                env_unwrapped=env._env,
                step_num=step,
            )

            if dmg_cause != "none":
                damage_event_counts[dmg_cause] = damage_event_counts.get(dmg_cause, 0) + 1
                ep_damage_log.append({"step": step, "cause": dmg_cause})

            if done and curr_inv.get("health", 0) <= 0:
                fatal_cause = dmg_cause if dmg_cause != "none" else (ep_damage_log[-1]["cause"] if ep_damage_log else "environmental")

            prev_inventory = curr_inv.copy()
            obs = next_obs
            step += 1

        episode_lengths.append(step)

        if not done or prev_inventory.get("health", 0) > 0:
            final_status = "survived_max_steps"
            cause_counts["survived_max_steps"] += 1
        else:
            final_status = fatal_cause or "environmental"
            cause_counts[final_status] = cause_counts.get(final_status, 0) + 1
            death_steps.append(step)
            is_night = (step % 300) >= 150
            if is_night:
                night_deaths += 1
            else:
                day_deaths += 1

        episodes_summary.append({
            "episode": i,
            "seed": seed,
            "length": step,
            "final_status": final_status,
            "damage_events": len(ep_damage_log),
        })

        if (i + 1) % 5 == 0 or (i + 1) == num_episodes:
            print(f"  Processed {i + 1}/{num_episodes} episodes | Avg Length: {np.mean(episode_lengths):.1f} steps")

    # Aggregate causes into higher-level buckets
    total_deaths = len(death_steps)
    summary_categories = {
        "Monsters (Zombies/Skeletons)": (
            cause_counts["zombie_night"] + cause_counts["zombie_day"] +
            cause_counts["skeleton_night"] + cause_counts["skeleton_day"] +
            cause_counts["monster_night_unseen"]
        ),
        "Starvation": cause_counts["starvation"],
        "Dehydration": cause_counts["dehydration"],
        "Starvation & Dehydration": cause_counts["starvation_and_dehydration"],
        "Environmental / Other": cause_counts["environmental"],
        "Survived Max Steps": cause_counts["survived_max_steps"],
    }

    results = {
        "num_episodes": num_episodes,
        "seed_range": f"{seed_start}-{seed_start + num_episodes - 1}",
        "max_steps": max_steps,
        "total_deaths": total_deaths,
        "survival_rate": float(cause_counts["survived_max_steps"] / num_episodes),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "mean_death_step": float(np.mean(death_steps)) if death_steps else None,
        "night_deaths": night_deaths,
        "day_deaths": day_deaths,
        "night_death_fraction": float(night_deaths / total_deaths) if total_deaths > 0 else 0.0,
        "cause_breakdown": cause_counts,
        "high_level_causes": summary_categories,
        "total_damage_events": damage_event_counts,
        "episodes": episodes_summary,
    }

    return results


def plot_death_causes(results: Dict[str, Any], output_png: str):
    """Generates charts for death causes and day/night survival distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Chart 1: High level causes
    causes = results["high_level_causes"]
    names = list(causes.keys())
    vals = list(causes.values())
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b", "#2ca02c"]

    bars = ax1.barh(names, vals, color=colors[:len(names)], edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Number of Episodes", fontsize=11)
    ax1.set_title(
        f"Terminal Fate Breakdown (N={results['num_episodes']})\n"
        f"Mean Survival: {results['mean_episode_length']:.1f} steps",
        fontsize=12,
        fontweight="bold"
    )
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax1.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({w/results['num_episodes']*100:.1f}%)", va="center")

    # Chart 2: Day vs Night Deaths
    if results["total_deaths"] > 0:
        day_night = [results["night_deaths"], results["day_deaths"]]
        labels = [f"Night ({results['night_deaths']})", f"Day ({results['day_deaths']})"]
        ax2.pie(
            day_night,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2b2b52", "#f39c12"],
            wedgeprops={"edgecolor": "black", "linewidth": 1.2},
            textprops={"fontsize": 11, "fontweight": "bold"}
        )
        ax2.set_title(
            f"Death Timing Distribution\n(Night Fraction: {results['night_death_fraction']*100:.1f}%)",
            fontsize=12,
            fontweight="bold"
        )
    else:
        ax2.text(0.5, 0.5, "No Deaths (100% Survival)", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"[Death-Diagnostics] Saved plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate death causes and survival bottlenecks.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/death_causes.json")
    parser.add_argument("--output_png", type=str, default="results/diagnostics/death_causes.png")
    args = parser.parse_args()

    print(f"[Death-Diagnostics] Loading model from {args.checkpoint}...")
    model = JEPAv2.from_v1_checkpoint(args.checkpoint, device=args.device)

    results = run_death_diagnostics(
        model=model,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        device=args.device,
    )

    print("\n--- Death & Survival Diagnostics Summary ---")
    print(f"Total Episodes: {results['num_episodes']} (Seeds {results['seed_range']})")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f} steps")
    print(f"Total Deaths: {results['total_deaths']} / {results['num_episodes']}")
    print(f"Night Deaths: {results['night_deaths']} ({results['night_death_fraction']*100:.1f}% of deaths)")
    print("High-Level Breakdown:")
    for k, v in results["high_level_causes"].items():
        print(f"  - {k}: {v} ({v / results['num_episodes'] * 100:.1f}%)")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Death-Diagnostics] Saved metrics to {args.output_json}")

    plot_death_causes(results, args.output_png)


if __name__ == "__main__":
    main()
