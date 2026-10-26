#!/usr/bin/env python3
"""
scripts/screen_v3_dual_memory.py — 50-Episode Screening with Dual-Memory Architecture
======================================================================================
Evaluates JEPA v3 with the complete Dual-Memory Cognitive System:
- System 1: ShortTermWorkingMemory (rolling sensory FIFO, landmark anchors, damage trends)
- System 2: LongTermSemanticMemory (precondition-gated visual cosine affordance bank)
- Online Learning: AutonomousMemoryWriter (captures milestone breakthroughs live)

Evaluates on untouched held-out seeds (2050-2099) with strict zero-leakage:
- Zero human recipes or rule tables.
- Pure visual observations (64x64x3 RGB) and scalar rewards.
- Outputs full performance metrics and table-to-tool conversion ratios.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # registers Crafter-v0
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from planning.semantic_memory import LongTermSemanticMemory, AutonomousMemoryWriter, DopaminePersonalBestTracker
from scripts.screen_v3_50eps import _resolve_checkpoint


def _resolve_memory_bank(bank_path: str, drive_dir: Optional[str] = None) -> str:
    p = Path(bank_path)
    if p.exists():
        return str(p)
    candidates = [
        REPO_ROOT / bank_path,
        REPO_ROOT / "checkpoints" / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / bank_path,
            dp / "checkpoints" / p.name,
            Path("/content") / "checkpoints" / p.name,
            Path("/content") / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(bank_path)


def run_dual_memory_screening(
    checkpoint_path: str = "checkpoints/jepa_crafter_v3_t2_final.pt",
    memory_bank_path: str = "checkpoints/semantic_memory_bank.pt",
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    prior_mix: float = 0.5,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.02,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: str = "results/screen_v3_dual_memory_50ep.json",
    drive_dir: Optional[str] = None,
    frontier_threshold: float = 0.25,
    common_threshold: float = 0.50,
    vital_weight: float = 0.5,
    harvest_bonus: float = 0.15,
    action_persistence: float = 0.5,
    memory_sim_threshold: float = 0.58,
    memory_inject_ratio: float = 0.50,
    empty_memory: bool = False,
    save_updated_memory: bool = True,
) -> Dict[str, Any]:
    dev = torch.device(device)
    resolved_ckpt = _resolve_checkpoint(checkpoint_path, drive_dir=drive_dir)
    resolved_bank = _resolve_memory_bank(memory_bank_path, drive_dir=drive_dir)

    print("=" * 80)
    print("      LEARN-A-WORLD v3: DUAL-MEMORY COGNITIVE SCREENING (SEEDS 2050-2099)")
    print(f"      Checkpoint  : {resolved_ckpt}")
    print(f"      Memory Bank : {resolved_bank}")
    print(f"      Episodes    : {num_episodes} | Seeds: {seed_start}..{seed_start + num_episodes - 1}")
    print(f"      Planner     : Recurrent MPC (H={horizon}, C={num_candidates}, CEM={cem_iterations})")
    print(f"      Affordance  : sim_thresh={memory_sim_threshold}, inject_ratio={memory_inject_ratio}")
    print(f"      Thresholds  : frontier={frontier_threshold}, common={common_threshold}")
    print(f"      Device      : {dev}")
    print("=" * 80)

    # 1. Load JEPA Model
    model = JEPAv3(
        latent_dim=128,
        recurrent_dim=256,
        num_actions=17,
        action_emb_dim=32,
        hidden_dim=256,
        q_heads=3,
        q_bins=51,
        q_max=12.0,
    )
    model.load_checkpoint(resolved_ckpt, device=dev)
    model.to(dev)
    model.eval()

    # 2. Load Long-Term Semantic Memory (System 2)
    memory_bank = LongTermSemanticMemory(latent_dim=128, device=dev)
    if empty_memory or (memory_bank_path and memory_bank_path.lower() in ("none", "empty")):
        print("[Memory Bank] Starting in TABULA RASA mode (0 entries). The agent will build its memory 100% live from scratch!")
        resolved_bank = "empty_memory (Tabula Rasa)"
    elif Path(resolved_bank).exists():
        memory_bank.load(resolved_bank)
        print(f"[Memory Bank] Successfully loaded {len(memory_bank)} affordance memories from {resolved_bank}")
    else:
        print(f"[Memory Bank] {resolved_bank} not found on disk. Auto-compiling from available shards...")
        from planning.build_semantic_memory import build_memory_bank
        memory_bank = build_memory_bank(
            checkpoint=resolved_ckpt,
            drive_dir=drive_dir,
            output_path=memory_bank_path,
            device=str(dev),
        )
        resolved_bank = memory_bank_path

    # 3. Dopamine Personal-Best Tracker & Autonomous Online Memory Writer
    dopamine_tracker = DopaminePersonalBestTracker(dopamine_scale=0.50)
    writer = AutonomousMemoryWriter(memory_bank=memory_bank, dopamine_tracker=dopamine_tracker)
    initial_bank_size = len(memory_bank)

    # 4. Build Planner & Stateful Belief (System 1 Working Memory is embedded in AgentBeliefV3)
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=cem_iterations,
        elite_ratio=0.15,
        gamma=0.99,
        uncertainty_weight=uncertainty_weight,
        novelty_weight=novelty_weight,
        prior_mix=prior_mix,
        action_persistence=action_persistence,
        num_actions=17,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
        vital_weight=vital_weight,
        harvest_bonus=harvest_bonus,
        semantic_memory=memory_bank,
        memory_sim_threshold=memory_sim_threshold,
        memory_inject_ratio=memory_inject_ratio,
        device=dev,
    )

    belief = AgentBeliefV3(
        recurrent_dim=256,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
    )

    # 5. Environment Execution Loop
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    episode_records: List[Dict[str, Any]] = []
    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    table_episodes = 0
    pickaxe_episodes = 0
    sword_episodes = 0
    stone_episodes = 0
    zombie_episodes = 0

    t_start = time.time()

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()
        writer.reset()

        ep_reward = 0.0
        steps = 0
        t_ep_start = time.time()
        ep_discoveries: List[str] = []

        for step in range(max_steps):
            steps += 1

            # Strict Zero-Leakage: agent plans solely using raw visual pixels + belief
            action, debug = planner.plan(obs, belief)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Update belief using RecurrentCore + EventHead
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(dev)
                z_t = model.encode(obs_t, use_target=True)
                act_t = torch.tensor([action], device=dev, dtype=torch.long)
                h_prev = belief.get_h_tensor(device=dev).unsqueeze(0)
                s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
                mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
                event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)

            # Online Autonomous Memory Writer detects milestone breakthroughs and triggers Dopamine surges
            newly_unlocked_event, dopamine_surge = writer.step(
                z_t=z_t.squeeze(0),
                action=action,
                reward=reward,
                info=info,
            )
            if newly_unlocked_event:
                dopamine_tag = f" (+{dopamine_surge:.1f} Dopamine PB!)" if dopamine_surge > 0 else ""
                ep_discoveries.append(f"{newly_unlocked_event}{dopamine_tag}")

            # System 1: Update belief masks, recurrent state, and short-term working memory
            belief.update(event_probs, reward, next_h=next_h.squeeze(0))
            belief.update_working_memory(
                z_t=z_t.squeeze(0),
                action=action,
                reward=reward,
                next_h=next_h.squeeze(0),
                unlocked_event=newly_unlocked_event,
            )

            obs = next_obs
            if done:
                break

        ep_duration = time.time() - t_ep_start
        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        unlocked = [k for k, v in ach.items() if v > 0]

        if ach.get("place_table", 0) > 0:
            table_episodes += 1
        if ach.get("make_wood_pickaxe", 0) > 0:
            pickaxe_episodes += 1
        if ach.get("make_wood_sword", 0) > 0:
            sword_episodes += 1
        if ach.get("collect_stone", 0) > 0:
            stone_episodes += 1
        if ach.get("defeat_zombie", 0) > 0:
            zombie_episodes += 1

        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

        discovery_str = f" [Discovered: {', '.join(ep_discoveries)}]" if ep_discoveries else ""
        print(
            f"Ep {ep_idx + 1:02d}/{num_episodes} (Seed {seed}) | "
            f"Steps: {steps:<4d} | Reward: {ep_reward:<5.1f} | "
            f"Achievements ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'None'}"
            f"{discovery_str} ({ep_duration:.1f}s)"
        )

        episode_records.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": steps,
            "reward": round(ep_reward, 2),
            "unlocked": unlocked,
            "discoveries": ep_discoveries,
            "duration_sec": round(ep_duration, 2),
        })

    env.close()
    total_time = time.time() - t_start

    score_stats = compute_crafter_score(achievement_records)
    new_memories_count = len(memory_bank) - initial_bank_size
    pickaxe_conversion = (pickaxe_episodes / table_episodes * 100.0) if table_episodes > 0 else 0.0

    summary = {
        "checkpoint": resolved_ckpt,
        "memory_bank": resolved_bank,
        "initial_memory_entries": initial_bank_size,
        "final_memory_entries": len(memory_bank),
        "newly_logged_memories": new_memories_count,
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "total_runtime_seconds": round(total_time, 2),
        "mean_survival_length": float(np.mean(survival_lengths)),
        "std_survival_length": float(np.std(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "crafter_score": score_stats["score"],
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "unlocked_achievement_count": score_stats["unlocked_count"],
        "unlock_percentages": score_stats["unlock_percentages"],
        "table_episodes": table_episodes,
        "pickaxe_episodes": pickaxe_episodes,
        "sword_episodes": sword_episodes,
        "stone_episodes": stone_episodes,
        "zombie_episodes": zombie_episodes,
        "pickaxe_conversion_percent": round(pickaxe_conversion, 1),
        "dopamine_lifetime_best_reward": dopamine_tracker.lifetime_best_reward,
        "dopamine_lifetime_best_milestones": dopamine_tracker.lifetime_best_milestones,
        "dopamine_lifetime_best_steps": dopamine_tracker.lifetime_best_steps,
        "episodes": episode_records,
    }

    # Print summary report
    print("\n" + "=" * 80)
    print("             DUAL-MEMORY COGNITIVE SCREENING SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Time: {total_time:.1f}s | Mean Episode Time: {total_time / num_episodes:.1f}s")
    print(f"Mean Survival Length: {summary['mean_survival_length']:.1f} +/- {summary['std_survival_length']:.1f} steps")
    print(f"Mean Episode Reward: {summary['mean_episode_reward']:.2f} +/- {summary['std_episode_reward']:.2f}")
    print(f"Official Crafter Score: {summary['crafter_score']:.2f}%")
    print(f"Mean Achievements per Episode: {summary['mean_achievements_per_ep']:.2f}")
    print(f"Unique Achievements Unlocked: {summary['unlocked_achievement_count']} / 22")
    print(f"Table Unlocks: {table_episodes} | Pickaxe Unlocks: {pickaxe_episodes} | Conversion: {pickaxe_conversion:.1f}%")
    print(f"Stone Harvests: {stone_episodes} | Sword Crafts: {sword_episodes} | Zombie Defeats: {zombie_episodes}")
    print(f"Dopamine Personal Bests: Max Reward={dopamine_tracker.lifetime_best_reward:.1f}, Max Milestones={dopamine_tracker.lifetime_best_milestones}, Max Steps={dopamine_tracker.lifetime_best_steps}")
    print(f"Semantic Memory: {initial_bank_size} initial -> {len(memory_bank)} entries (+{new_memories_count} learned)")

    print("\nPer-Achievement Unlock Rates:")
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = score_stats["unlock_percentages"].get(ach, 0.0)
        bar = "#" * int(round(pct / 5))
        print(f"  {ach:<22}: {pct:5.1f}% | {bar}")

    # Save results JSON
    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--> Full screening results saved to: {out_p}")

    # Mirror to Drive if requested
    if drive_dir and Path(drive_dir).exists():
        drive_out = Path(drive_dir) / "results" / out_p.name
        drive_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(out_p), str(drive_out))
        print(f"[Drive] Mirrored screening results to: {drive_out}")

    # Save updated memory bank
    if save_updated_memory and new_memories_count > 0:
        is_tabula_rasa = empty_memory or (memory_bank_path and memory_bank_path.lower() in ("none", "empty"))
        save_dest = "checkpoints/tabula_rasa_memory_bank.pt" if is_tabula_rasa else memory_bank_path
        saved_bank_p = memory_bank.save(save_dest)
        print(f"[Memory Bank] Successfully saved newly built memory bank with discoveries: {saved_bank_p}")
        if drive_dir and Path(drive_dir).exists():
            drive_bank_out = Path(drive_dir) / "checkpoints" / Path(save_dest).name
            shutil.copy2(str(saved_bank_p), str(drive_bank_out))
            print(f"[Drive] Mirrored updated memory bank to: {drive_bank_out}")

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print(f"--- BEGIN {out_p.name} ---")
    print(json.dumps(summary, indent=2))
    print(f"--- END {out_p.name} ---\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-episode dual-memory screening of JEPA v3.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_t2_final.pt")
    parser.add_argument("--memory_bank", type=str, default="checkpoints/semantic_memory_bank.pt")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=2)
    parser.add_argument("--prior_mix", type=float, default=0.5)
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--novelty_weight", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="results/screen_v3_dual_memory_50ep.json")
    parser.add_argument("--drive_dir", type=str, default=None)
    parser.add_argument("--frontier_threshold", type=float, default=0.25)
    parser.add_argument("--common_threshold", type=float, default=0.50)
    parser.add_argument("--vital_weight", type=float, default=0.5)
    parser.add_argument("--harvest_bonus", type=float, default=0.15)
    parser.add_argument("--action_persistence", type=float, default=0.5)
    parser.add_argument("--memory_sim_threshold", type=float, default=0.58)
    parser.add_argument("--memory_inject_ratio", type=float, default=0.50)
    parser.add_argument("--empty_memory", action="store_true", default=False, help="Start with 0 memories (Tabula Rasa).")
    parser.add_argument("--save_updated_memory", action="store_true", default=True)
    args = parser.parse_args()

    run_dual_memory_screening(
        checkpoint_path=args.checkpoint,
        memory_bank_path=args.memory_bank,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        prior_mix=args.prior_mix,
        uncertainty_weight=args.uncertainty_weight,
        novelty_weight=args.novelty_weight,
        device=args.device,
        output_json=args.output_json,
        drive_dir=args.drive_dir,
        frontier_threshold=args.frontier_threshold,
        common_threshold=args.common_threshold,
        vital_weight=args.vital_weight,
        harvest_bonus=args.harvest_bonus,
        action_persistence=args.action_persistence,
        memory_sim_threshold=args.memory_sim_threshold,
        memory_inject_ratio=args.memory_inject_ratio,
        empty_memory=args.empty_memory,
        save_updated_memory=args.save_updated_memory,
    )
