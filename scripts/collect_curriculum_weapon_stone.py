#!/usr/bin/env python3
"""
scripts/collect_curriculum_weapon_stone.py
=========================================
Autonomous High-Yield Curriculum Sequence Generator.
Collects expert demonstration episodes specifically covering the critical early-to-midgame tech chain:
  [Gather Wood -> Deploy Table -> Craft Wood Pickaxe -> Craft Wood Sword ->
   Mine Stone -> Craft Stone Sword / Pickaxe -> Deploy Furnace -> Combat & Survival]

Zero Cheating / Strict RL Principle:
This data is stored as visual observations obs [64, 64, 3], actions a in 0..16, rewards r,
events e in {0, 1}^22, and dones d for standard self-supervised imitation & value fine-tuning.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs
from crafter.objects import Zombie, Skeleton, Cow, Plant
from eval.metrics import CRAFTER_ACHIEVEMENTS
from training.sequence_buffer import EpisodeBuffer

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)


def find_nearest_material(player, material_name: str, max_dist: int = 25) -> Optional[Tuple[int, int]]:
    px, py = int(player.pos[0]), int(player.pos[1])
    best_dist = 999
    best_target = None
    for dx in range(-max_dist, max_dist + 1):
        for dy in range(-max_dist, max_dist + 1):
            tx, ty = px + dx, py + dy
            try:
                mat, obj = player.world[(tx, ty)]
                if mat == material_name:
                    d = abs(dx) + abs(dy)
                    if d < best_dist:
                        best_dist = d
                        best_target = (tx, ty)
            except Exception:
                continue
    return best_target


def step_toward(player, target: Tuple[int, int]) -> int:
    px, py = int(player.pos[0]), int(player.pos[1])
    tx, ty = target
    dx = tx - px
    dy = ty - py
    fx, fy = player.facing

    # 1. Adjacent check
    if abs(dx) + abs(dy) == 1:
        if (fx, fy) == (dx, dy):
            return 5  # "Do" (harvest, mine, strike, or interact)
        # Turn to face
        if dx == -1: return 1
        if dx == 1:  return 2
        if dy == -1: return 3
        if dy == 1:  return 4

    # 2. Pathfinding with obstacle avoidance
    if abs(dx) >= abs(dy) and dx != 0:
        cand_act = 1 if dx < 0 else 2
        cand_pos = (px + (-1 if dx < 0 else 1), py)
        if player.is_free(cand_pos):
            return cand_act
        elif dy != 0:
            return 3 if dy < 0 else 4
    elif dy != 0:
        cand_act = 3 if dy < 0 else 4
        cand_pos = (px, py + (-1 if dy < 0 else 1))
        if player.is_free(cand_pos):
            return cand_act
        elif dx != 0:
            return 1 if dx < 0 else 2

    return int(np.random.randint(1, 5))


def make_curriculum_expert(env: gym.Env):
    state = {
        "table_pos": None,
        "step": 0,
    }

    def policy(obs: np.ndarray, info: Dict[str, Any]) -> int:
        state["step"] += 1
        s = state["step"]

        inv = info.get("inventory", {})
        ach = info.get("achievements", {})

        wood = inv.get("wood", 0)
        stone = inv.get("stone", 0)
        drink = inv.get("drink", 9)
        food = inv.get("food", 9)
        energy = inv.get("energy", 9)
        health = inv.get("health", 9)

        has_table = ach.get("place_table", 0) > 0
        has_wood_pick = inv.get("wood_pickaxe", 0) > 0
        has_wood_sword = inv.get("wood_sword", 0) > 0
        has_stone_pick = inv.get("stone_pickaxe", 0) > 0
        has_stone_sword = inv.get("stone_sword", 0) > 0

        crafter_env = env.unwrapped._env
        player = crafter_env._player

        # 1. Combat Defense: engage monsters within distance <= 4 if armed
        nearby_objs, _ = player.world.nearby(player.pos, 4)
        hostiles = [obj for obj in nearby_objs if isinstance(obj, (Zombie, Skeleton))]
        if hostiles:
            closest = min(hostiles, key=lambda h: player.distance(h.pos))
            if has_wood_sword or has_stone_sword or health >= 6:
                return step_toward(player, (int(closest.pos[0]), int(closest.pos[1])))
            else:
                # Flee if unarmed and low health
                hx, hy = closest.pos
                dx = int(player.pos[0] - hx)
                dy = int(player.pos[1] - hy)
                if abs(dx) > abs(dy):
                    return 2 if dx > 0 else 1
                else:
                    return 4 if dy > 0 else 3

        # 2. Night Survival: sleep through darkness when tired or hurt
        is_night = (s % 300 > 130 and s % 300 < 270)
        if (energy <= 7 and is_night) or health <= 4:
            return 6  # sleep

        # 3. Essential Thirst & Hunger
        if drink <= 3:
            water = find_nearest_material(player, "water", max_dist=15)
            if water:
                return step_toward(player, water)
            return 5
        if food <= 2:
            plant = find_nearest_material(player, "plant", max_dist=12)
            if plant:
                return step_toward(player, plant)
            return 5

        # 4. Phase 1: Gather 4 Wood
        if wood < 4 and not has_table:
            tree = find_nearest_material(player, "tree", max_dist=20)
            if tree:
                return step_toward(player, tree)
            return int(np.random.randint(1, 5))

        # 5. Phase 2: Deploy Crafting Table
        if wood >= 4 and not has_table:
            state["table_pos"] = (int(player.pos[0] + player.facing[0]), int(player.pos[1] + player.facing[1]))
            return 8  # place table

        # 6. Phase 3: Craft Wood Pickaxe + Wood Sword
        if has_table and not state["table_pos"]:
            state["table_pos"] = find_nearest_material(player, "table", max_dist=15)

        if has_table and state["table_pos"]:
            tx, ty = state["table_pos"]
            d = abs(player.pos[0] - tx) + abs(player.pos[1] - ty)
            if d > 1:
                return step_toward(player, (tx, ty))
            if not has_wood_pick and wood >= 1:
                return 11  # make wood pickaxe
            if not has_wood_sword and wood >= 1:
                return 14  # make wood sword

        # 7. Phase 4: Mine Stone with Pickaxe
        if has_wood_pick and stone < 5:
            rock = find_nearest_material(player, "stone", max_dist=25)
            if rock:
                return step_toward(player, rock)
            return int(np.random.randint(1, 5))

        # 8. Phase 5: Stone Tech Upgrades at Table
        if stone >= 2 and state["table_pos"]:
            tx, ty = state["table_pos"]
            d = abs(player.pos[0] - tx) + abs(player.pos[1] - ty)
            if d > 1:
                return step_toward(player, (tx, ty))
            if not has_stone_sword and stone >= 1 and wood >= 1:
                return 15  # make stone sword
            if not has_stone_pick and stone >= 1 and wood >= 1:
                return 12  # make stone pickaxe
            if stone >= 4 and ach.get("place_furnace", 0) == 0:
                return 9   # place furnace

        # Exploration fallback
        return 5 if np.random.random() < 0.35 else int(np.random.randint(1, 5))

    return policy


def collect_curriculum_dataset(
    num_episodes: int = 30,
    seed_start: int = 1000,
    max_steps: int = 500,
    out_dir: str = "data/curriculum",
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "shard_curriculum_weapon_stone.npz"

    print("=" * 80)
    print("      AUTONOMOUS CURRICULUM SEQUENCE COLLECTOR")
    print(f"      Episodes   : {num_episodes}")
    print(f"      Seeds      : {seed_start}..{seed_start + num_episodes - 1}")
    print(f"      Output     : {out_file}")
    print("=" * 80)

    buffer = EpisodeBuffer(capacity_transitions=num_episodes * max_steps + 5000)
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    t0 = time.time()
    total_transitions = 0

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        policy = make_curriculum_expert(env)

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        ep_events = []
        ep_masks = []

        unlocked_mask = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        prev_ach = {}

        for step in range(max_steps):
            mask_before = unlocked_mask.copy()
            action = policy(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            curr_ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
            event_vec = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
            for ach_name, idx in ACH_TO_IDX.items():
                if curr_ach.get(ach_name, 0) > prev_ach.get(ach_name, 0):
                    event_vec[idx] = 1
                    unlocked_mask[idx] = 1

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(float(reward))
            ep_dones.append(done)
            ep_events.append(event_vec)
            ep_masks.append(mask_before)

            prev_ach = curr_ach
            obs = next_obs
            total_transitions += 1
            if done:
                break

        if len(ep_actions) > 0:
            buffer.add_episode(
                obs=np.array(ep_obs, dtype=np.uint8),
                actions=np.array(ep_actions, dtype=np.int64),
                rewards=np.array(ep_rewards, dtype=np.float32),
                dones=np.array(ep_dones, dtype=np.bool_),
                events=np.array(ep_events, dtype=np.uint8),
                mask_before=np.array(ep_masks, dtype=np.uint8),
                source="curriculum_weapon_stone",
            )

        if (ep_idx + 1) % 5 == 0 or (ep_idx + 1) == num_episodes:
            elapsed = time.time() - t0
            print(f"  [Progress] {ep_idx + 1:2d}/{num_episodes} episodes | {total_transitions:>6,d} transitions | ({elapsed:.1f}s)")

    env.close()
    buffer.save_shard(out_file)
    print(f"\n--> Successfully saved {buffer.total_transitions:,} curriculum transitions to: {out_file}")

    stats = buffer.get_dataset_statistics()
    print("\n[Curriculum Event Counts]:")
    for ach, cnt in stats["per_achievement_positive_counts"].items():
        if cnt > 0:
            print(f"   {ach:<24s} : {cnt:>4d} positive transitions")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Collect High-Yield Curriculum Episodes.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="data/curriculum")
    args = parser.parse_args()

    collect_curriculum_dataset(
        num_episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
