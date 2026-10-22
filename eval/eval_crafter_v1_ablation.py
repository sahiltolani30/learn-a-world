"""
Crafter Benchmark & Ablation Evaluator for Learn-a-World (v1 Freeze).
Based on eval/eval_crafter.py with ablation controls:
1. --reflexes {on, off} (default 'on'): Toggles all hardcoded if-else reflexes.
2. --do-bias FLOAT (default 0.35): Overrides the MPC fallback 'Do' rate.
3. --seed-offset INT (default 1000): Evaluates on specified reproducible seeds.
4. --out STR (default None): Saves evaluation results to a JSON file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
import torch

import envs  # registers environments
from eval.metrics import CRAFTER_ACHIEVEMENTS, evaluate_crafter_agent
from experiments.utils import get_device, load_config
from models.jepa import JEPAWorldModel
from planning.mpc import LatentMPCPlanner

# Achievement tiers for Crafter diagnosis
ACHIEVEMENT_TIERS = {
    "Tier 1 (Basic Gathering & Survival)": [
        "collect_wood", "collect_drink", "collect_sapling", "eat_plant", "wake_up"
    ],
    "Tier 2 (Early Crafting & Shelter)": [
        "place_table", "make_wood_pickaxe", "make_wood_sword", "place_plant"
    ],
    "Tier 3 (Mining & Stone Age)": [
        "collect_stone", "collect_coal", "place_stone", "place_furnace",
        "make_stone_pickaxe", "make_stone_sword"
    ],
    "Tier 4 (Advanced Metallurgy & Combat)": [
        "collect_iron", "collect_diamond", "make_iron_pickaxe", "make_iron_sword",
        "defeat_zombie", "defeat_skeleton", "eat_cow"
    ],
}

def evaluate_crafter_ablation(
    checkpoint_path: Optional[str] = None,
    config_path: str = "experiments/configs/crafter.yaml",
    num_episodes: int = 20,
    max_steps: int = 1000,
    device_override: Optional[str] = None,
    mode: str = "mpc",
    reflexes: str = "on",
    do_bias: float = 0.35,
    seed_offset: int = 1000,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs Crafter evaluation with configurable reflexes and do-bias ablations."""
    device = get_device(device_override)
    cfg = load_config(config_path)

    env_cfg = cfg["env"]
    env = gym.make(
        env_cfg["name"],
        render_size=env_cfg.get("render_size", 64),
        max_steps=max_steps,
    )

    use_reflexes = (reflexes.lower() == "on")

    print("======================================================================")
    print("   Learn-a-World: Crafter Benchmark Ablation Evaluation (v1 Freeze)   ")
    print(f"   Mode: {mode.upper()} | Reflexes: {reflexes.upper()} | Do-Bias: {do_bias:.2f} ")
    print(f"   Episodes: {num_episodes} | Seed Offset: {seed_offset} | Device: {device} ")
    print("======================================================================")

    planner = None
    if mode == "mpc" and checkpoint_path and Path(checkpoint_path).exists():
        print(f"--> Loading trained JEPA World Model from: {checkpoint_path}")
        m_cfg = cfg["model"]
        p_cfg = cfg.get("planning", {})
        model = JEPAWorldModel(
            in_channels=3,
            latent_dim=m_cfg.get("latent_dim", 128),
            num_actions=m_cfg.get("num_actions", 17),
            action_emb_dim=m_cfg.get("action_emb_dim", 32),
            predictor_hidden_dim=m_cfg.get("predictor_hidden_dim", 256),
            predictor_layers=m_cfg.get("predictor_layers", 2),
            ema_decay=m_cfg.get("ema_decay", 0.99),
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        planner = LatentMPCPlanner(
            model=model,
            horizon=p_cfg.get("horizon", 7),
            num_candidates=p_cfg.get("num_candidates", 128),
            elite_ratio=p_cfg.get("elite_ratio", 0.15),
            num_actions=17,
            device=device,
        )
        print(f"    Latent MPC Planner active (Horizon={p_cfg.get('horizon', 7)}, Candidates={p_cfg.get('num_candidates', 128)})")

    state = {
        "placed_table_step": -999,
        "plant_placed_step": -999,
        "step": 0,
    }

    def action_fn(obs: np.ndarray, info: dict) -> int:
        state["step"] += 1
        s = state["step"]

        inv = info.get("inventory", {}) if isinstance(info, dict) else {}
        ach = info.get("achievements", {}) if isinstance(info, dict) else {}

        wood = inv.get("wood", 0)
        stone = inv.get("stone", 0)
        drink = inv.get("drink", 0)
        food = inv.get("food", 0)
        energy = inv.get("energy", 9)
        sapling = inv.get("sapling", 0)

        has_table = ach.get("place_table", 0) > 0
        has_wood_pick = inv.get("wood_pickaxe", 0) > 0
        has_stone_pick = inv.get("stone_pickaxe", 0) > 0
        has_wood_sword = ach.get("make_wood_sword", 0) > 0
        did_place_stone = ach.get("place_stone", 0) > 0
        did_place_plant = ach.get("place_plant", 0) > 0

        # Per-episode state reset
        if not has_table:
            state["placed_table_step"] = -999

        if use_reflexes:
            # ── SURVIVAL PRIORITY (highest priority, prevents step-184 death) ──────
            if drink <= 2:
                return 5   # "Do" near water tile to drink
            if food <= 1:
                return 5   # "Do" to eat (cow, plant)
            if energy <= 2:
                return 6   # sleep to recover energy and wake_up

            # ── PLANT PLACEMENT ───────────────────────────────────────────────────
            if sapling >= 1 and not did_place_plant:
                return 10  # place_plant

            # ── PLACE STONE ACHIEVEMENT (Tier 3 free unlock) ──────────────────────
            # When stone >= 2, placing one unlocks place_stone while leaving >=1 for crafting
            if stone >= 2 and not did_place_stone:
                return 7   # place_stone

            # ── CRAFTING SEQUENCE (sequential reflexes) ───────────────────────────
            # 1. Wait for >= 4 wood (2 for table, 1 for pickaxe, 1 for sword)
            # Fallback at step > 80: if >= 3 wood, place table anyway
            can_place_table = (wood >= 4) or (wood >= 3 and s > 80)
            if can_place_table and not has_table:
                state["placed_table_step"] = s
                return 8   # place_table

            # 2. t+1: Craft wood pickaxe immediately while still adjacent to table
            if (s - state["placed_table_step"] == 1) and not has_wood_pick:
                return 11  # make_wood_pickaxe

            # 3. t+2: Craft wood sword immediately while still adjacent to table
            if (s - state["placed_table_step"] == 2) and wood >= 1 and not has_wood_sword:
                return 14  # make_wood_sword

            # 4. Near-table stone upgrades (within 30 steps of table placement)
            near_table = (s - state["placed_table_step"]) <= 30
            if near_table and has_wood_pick and stone >= 1 and wood >= 1 and not has_stone_pick:
                return 12  # make_stone_pickaxe
            if near_table and stone >= 4 and ach.get("place_furnace", 0) == 0:
                return 9   # place_furnace

            # ── POST-PICKAXE: HEAVY MINING MODE ──────────────────────────────────
            # Once the pickaxe is held, spam "Do" heavily (60%) to mine stone/coal
            if has_wood_pick and not has_stone_pick:
                if np.random.random() < 0.60:
                    return 5   # Mine (stone/coal require wood_pickaxe via "Do")
                return int(np.random.randint(1, 5))

        # ── PLANNER or DEFAULT EXPLORATION ───────────────────────────────────
        if planner is not None:
            if np.random.random() < do_bias:
                return 5   # "Do" bias: harvest, drink, eat
            return planner.plan(obs)
        elif mode == "curious":
            p = np.random.random()
            if p < 0.40:
                return 5
            elif p < 0.85:
                return int(np.random.randint(1, 5))
            else:
                return int(env.action_space.sample())
        else:
            return int(env.action_space.sample())

    print(f"--> Running {num_episodes} evaluation episodes (max {max_steps} steps/ep, seed_offset={seed_offset})...")
    results = evaluate_crafter_agent(
        env=env,
        action_fn=action_fn,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed_offset=seed_offset,
    )
    env.close()

    crafter_score = results["crafter_score"]
    unlocked_count = results["unlocked_count"]
    pcts = results["unlock_percentages"]

    print("\n" + "=" * 70)
    print("                     EVALUATION SCORECARD")
    print("=" * 70)
    print(f"  Official Crafter Score:    {crafter_score:.2f}%")
    print(f"  Achievements Unlocked:     {unlocked_count}/22")
    print(f"  Mean Survival Steps:       {results['mean_survival_steps']:.1f}")
    print(f"  Mean Reward / Episode:     {results['mean_reward']:.2f}")
    print("=" * 70)

    # Detailed tier diagnosis
    print("\n------------------- ACHIEVEMENT PROGRESSION BY TIER -------------------")
    stuck_by_tier = {}
    for tier_name, achs in ACHIEVEMENT_TIERS.items():
        print(f"\n{tier_name}:")
        stuck_in_tier = []
        for ach in achs:
            pct = pcts.get(ach, 0.0)
            status = "✅" if pct > 0 else "❌ STUCK"
            print(f"   [{status:7s}] {ach:22s} : {pct:5.1f}%")
            if pct == 0.0:
                stuck_in_tier.append(ach)
        stuck_by_tier[tier_name] = stuck_in_tier
    print("=" * 70 + "\n")

    # Attach run configuration metadata
    results["metadata"] = {
        "mode": mode,
        "reflexes": reflexes,
        "do_bias": do_bias,
        "num_episodes": num_episodes,
        "seed_offset": seed_offset,
        "checkpoint": checkpoint_path if mode == "mpc" else None,
    }

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"--> Saved evaluation results to: {out_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Crafter Agent Ablation (v1 Freeze)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="experiments/configs/crafter.yaml", help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--mode", type=str, default="mpc", choices=["mpc", "curious", "random"], help="Agent mode")
    parser.add_argument("--reflexes", type=str, default="on", choices=["on", "off"], help="Toggle hardcoded reflexes")
    parser.add_argument("--do-bias", type=float, default=0.35, help="Override MPC fallback Do rate")
    parser.add_argument("--seed-offset", type=int, default=1000, help="Evaluation seed offset")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_crafter_ablation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        device_override=args.device,
        mode=args.mode,
        reflexes=args.reflexes,
        do_bias=args.do_bias,
        seed_offset=args.seed_offset,
        out_path=args.out,
    )

if __name__ == "__main__":
    main()
