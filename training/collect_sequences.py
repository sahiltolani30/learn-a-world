"""
Sequence Data Collector for Learn-a-World v2.
Collects full episodes with per-step achievement events and unlocked masks.
Policies supported:
- 'structured_random': 60% movement, 25% 'Do', 15% random exploration.
- 'teacher_hybrid': cycle-4 sequential crafting reflex + MPC/Do fallback.
- 'planner_v2': pure v2 planner (used in Step 3 collect-retrain rounds).
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import gymnasium as gym
import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import envs  # registers environments
from eval.metrics import CRAFTER_ACHIEVEMENTS
from training.sequence_buffer import (
    ACHIEVEMENT_TO_IDX,
    NUM_ACHIEVEMENTS,
    EpisodeBuffer,
    save_achievement_index,
)

def build_policy(
    policy_name: str,
    env: gym.Env,
    checkpoint_path: Optional[str] = None,
) -> Callable[[np.ndarray, Dict[str, Any]], int]:
    """Instantiates the action selection function for the requested policy."""
    if policy_name == "structured_random":
        def structured_random_action(obs: np.ndarray, info: Dict[str, Any]) -> int:
            p = np.random.random()
            if p < 0.60:
                return int(np.random.randint(1, 5))  # Move Left, Right, Up, Down
            elif p < 0.85:
                return 5  # "Do" (harvest wood/stone, drink, eat, attack)
            else:
                return int(env.action_space.sample())
        return structured_random_action

    elif policy_name == "teacher_hybrid":
        # Imports the verified reflex logic from eval_crafter_v1_ablation
        state = {
            "placed_table_step": -999,
            "step": 0,
        }

        def teacher_hybrid_action(obs: np.ndarray, info: Dict[str, Any]) -> int:
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

            if not has_table:
                state["placed_table_step"] = -999

            # Survival reflexes
            if drink <= 2:
                return 5
            if food <= 1:
                return 5
            if energy <= 2:
                return 6

            # Plant & stone placement
            if sapling >= 1 and not did_place_plant:
                return 10
            if stone >= 2 and not did_place_stone:
                return 7

            # Sequential Crafting
            can_place_table = (wood >= 4) or (wood >= 3 and s > 80)
            if can_place_table and not has_table:
                state["placed_table_step"] = s
                return 8

            if (s - state["placed_table_step"] == 1) and not has_wood_pick:
                return 11
            if (s - state["placed_table_step"] == 2) and wood >= 1 and not has_wood_sword:
                return 14

            near_table = (s - state["placed_table_step"]) <= 30
            if near_table and has_wood_pick and stone >= 1 and wood >= 1 and not has_stone_pick:
                return 12
            if near_table and stone >= 4 and ach.get("place_furnace", 0) == 0:
                return 9

            if has_wood_pick and not has_stone_pick:
                if np.random.random() < 0.60:
                    return 5
                return int(np.random.randint(1, 5))

            # Exploration fallback
            p = np.random.random()
            if p < 0.35:
                return 5
            elif p < 0.75:
                return int(np.random.randint(1, 5))
            else:
                return int(env.action_space.sample())

        return teacher_hybrid_action

    elif policy_name == "planner_v2":
        from experiments.utils import get_device, load_config
        from models.jepa import JEPAWorldModel
        from models.jepa_v2 import JEPAv2
        from planning.agent_v2 import AgentBelief
        from planning.mpc_v2 import RewardValuePlanner

        device = get_device(None)
        cfg = load_config("experiments/configs/crafter_v2.yaml")
        m_cfg = cfg.get("model", {})
        h_cfg = cfg.get("heads", {})
        p_cfg = cfg.get("planner_v2", {})

        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                base_m = JEPAWorldModel(
                    in_channels=3,
                    latent_dim=m_cfg.get("latent_dim", 128),
                    num_actions=m_cfg.get("num_actions", 17),
                    action_emb_dim=m_cfg.get("action_emb_dim", 32),
                ).to(device)
                model = JEPAv2(base_model=base_m).to(device)
                model.load_checkpoint(checkpoint_path, device=device)
            except Exception:
                model = JEPAv2.from_v1_checkpoint(checkpoint_path, config=cfg, device=device)
        else:
            base_m = JEPAWorldModel(
                in_channels=3,
                latent_dim=m_cfg.get("latent_dim", 128),
                num_actions=m_cfg.get("num_actions", 17),
            ).to(device)
            model = JEPAv2(base_model=base_m).to(device)

        model.eval()
        planner = RewardValuePlanner(
            model=model,
            horizon=p_cfg.get("horizon", 10),
            num_candidates=p_cfg.get("num_candidates", 128),
            cem_iterations=p_cfg.get("cem_iterations", 2),
            device=device,
        )
        belief = AgentBelief()

        def planner_v2_action(obs: np.ndarray, info: Dict[str, Any]) -> int:
            # 10% structured_random mix for coverage
            if np.random.random() < 0.10:
                p = np.random.random()
                if p < 0.60:
                    return int(np.random.randint(1, 5))
                elif p < 0.85:
                    return 5
                return int(env.action_space.sample())

            action, _ = planner.plan(obs, belief)
            return action

        return planner_v2_action

    else:
        raise ValueError(f"Unknown collection policy: '{policy_name}'")

def collect_episodes(
    policy_name: str,
    n_transitions: int = 50_000,
    out_dir: str = "data/sequences",
    chunk_size: int = 5_000,
    seed: int = 42,
    max_steps_per_ep: int = 1000,
    checkpoint_path: Optional[str] = None,
) -> EpisodeBuffer:
    """
    Runs environment rollout loop and saves chunked sequence datasets.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_achievement_index(out_path)

    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps_per_ep)
    policy_fn = build_policy(policy_name, env, checkpoint_path=checkpoint_path)

    buffer = EpisodeBuffer(capacity_transitions=n_transitions + 5000)
    total_steps = 0
    ep_count = 0
    saved_shards: List[str] = []

    print(f"--> Collecting {n_transitions} transitions with policy='{policy_name}' (seed={seed})...")

    current_seed = seed
    while total_steps < n_transitions:
        obs, info = env.reset(seed=current_seed)
        current_seed += 1

        ep_obs: List[np.ndarray] = []
        ep_actions: List[int] = []
        ep_rewards: List[float] = []
        ep_dones: List[bool] = []
        ep_events: List[np.ndarray] = []
        ep_masks: List[np.ndarray] = []

        prev_ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}

        for step in range(max_steps_per_ep):
            mask_before = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
            for ach, idx in ACHIEVEMENT_TO_IDX.items():
                if prev_ach.get(ach, 0) > 0:
                    mask_before[idx] = 1

            action = policy_fn(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            curr_ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
            event_vec = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
            for ach, idx in ACHIEVEMENT_TO_IDX.items():
                if curr_ach.get(ach, 0) > prev_ach.get(ach, 0):
                    event_vec[idx] = 1

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(float(reward))
            ep_dones.append(done)
            ep_events.append(event_vec)
            ep_masks.append(mask_before)

            total_steps += 1
            prev_ach = curr_ach
            obs = next_obs

            if done or total_steps >= n_transitions:
                break

        if len(ep_actions) > 0:
            buffer.add_episode(
                obs=np.array(ep_obs, dtype=np.uint8),
                actions=np.array(ep_actions, dtype=np.int64),
                rewards=np.array(ep_rewards, dtype=np.float32),
                dones=np.array(ep_dones, dtype=np.bool_),
                events=np.array(ep_events, dtype=np.uint8),
                mask_before=np.array(ep_masks, dtype=np.uint8),
                source=policy_name,
            )
            ep_count += 1

            # Save shard incrementally if chunk threshold reached
            if buffer.total_transitions >= (len(saved_shards) + 1) * chunk_size:
                shard_name = f"shard_{policy_name}_{len(saved_shards):04d}.npz"
                shard_file = out_path / shard_name
                buffer.save_shard(shard_file)
                saved_shards.append(shard_name)
                buffer.save_dataset_index(out_path, saved_shards)
                print(f"    Saved shard {shard_name} (Total collected: {buffer.total_transitions}/{n_transitions})")

    # Save final shard if uncommitted transitions exist
    final_shard_name = f"shard_{policy_name}_final.npz"
    final_shard_file = out_path / final_shard_name
    buffer.save_shard(final_shard_file)
    if final_shard_name not in saved_shards:
        saved_shards.append(final_shard_name)
    buffer.save_dataset_index(out_path, saved_shards)
    env.close()

    stats = buffer.get_dataset_statistics()
    print("=" * 65)
    print(f"Collection complete: {stats['total_transitions']} transitions across {stats['total_episodes']} episodes.")
    top_events = {k: v for k, v in stats["per_achievement_positive_counts"].items() if v > 0}
    print(f"Unlocked events captured: {top_events}")
    print(f"Dataset index written to: {out_path / 'dataset_index.json'}")
    print("=" * 65)

    return buffer

def main():
    parser = argparse.ArgumentParser(description="Collect Crafter Sequence Episodes")
    parser.add_argument("--policy", type=str, default="structured_random",
                        choices=["structured_random", "teacher_hybrid", "planner_v2"],
                        help="Data collection policy")
    parser.add_argument("--n-transitions", type=int, default=50_000,
                        help="Target number of transitions to collect")
    parser.add_argument("--out-dir", type=str, default="data/sequences",
                        help="Output directory for sequence shards")
    parser.add_argument("--chunk-size", type=int, default=5_000,
                        help="Transitions per saved .npz shard")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v2.pt",
                        help="Checkpoint path for planner_v2 policy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    collect_episodes(
        policy_name=args.policy,
        n_transitions=args.n_transitions,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
