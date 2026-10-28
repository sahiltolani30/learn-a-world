#!/usr/bin/env python3
"""
scripts/inspect_episode_decisions.py — Forensic Subgoal Decision & Memory Inspector
==================================================================================

Parses the screening results JSON and online memory bank to show exactly:
1. How the agent made decisions step-by-step (subgoal traces).
2. Which causal memory cards the planner selected as targets.
3. How and when online memory cards were created, confirmed, or denied.
4. Survival lengths and achievements unlocked per game.

Usage:
    python3 scripts/inspect_episode_decisions.py
    python3 scripts/inspect_episode_decisions.py --results results/screen_v3_armC_dual_store_50ep.json
    python3 scripts/inspect_episode_decisions.py --episode 1
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def inspect_decisions(
    results_path: str = "results/screen_v3_armC_dual_store_50ep.json",
    online_bank_path: str = "results/dual_store_memory_bank_online.json",
    target_episode: Optional[int] = None,
) -> None:
    # 1. Resolve Results Path
    res_p = Path(results_path)
    if not res_p.exists():
        candidates = [
            REPO_ROOT / results_path,
            Path("/content/learn-a-world") / results_path,
            Path("/content/drive/MyDrive/learn-a-world") / results_path,
            Path("/content/drive/MyDrive/learn-a-world/results") / res_p.name,
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                res_p = c
                break

    if not res_p.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        return

    with open(res_p, "r") as f:
        data = json.load(f)

    episodes = data.get("episodes", data.get("episode_records", []))
    print("=" * 80)
    print("       DUAL-STORE AGENT: IN-GAME DECISION & MEMORY AUDIT")
    print(f"       Results File: {res_p}")
    print(f"       Total Episodes Logged: {len(episodes)}")
    print(f"       Official Score: {data.get('crafter_score', 0):.2f}% | Mean Reward: {data.get('mean_episode_reward', 0):.2f} | Mean Steps: {data.get('mean_survival_length', 0):.1f}")
    print("=" * 80)

    # 2. Iterate and Inspect Episodes
    for ep in episodes:
        ep_num = ep.get("episode", 0)
        if target_episode is not None and ep_num != target_episode:
            continue

        seed = ep.get("seed", 0)
        steps = ep.get("steps", 0)
        reward = ep.get("reward", 0.0)
        unlocked = ep.get("unlocked", [])
        online_stats = ep.get("online_stats", {})
        subgoal_trace = ep.get("subgoal_trace", [])

        print(f"\n🎮 EPISODE {ep_num:02d} (Seed {seed}) — Steps: {steps} | Reward: {reward} | Unlocks: {', '.join(unlocked) if unlocked else 'None'}")
        
        # Online Learning Summary for this episode
        if online_stats:
            c = online_stats.get("created", 0)
            conf = online_stats.get("confirmed", 0)
            d = online_stats.get("denied", 0)
            print(f"   🧠 Online Memory Delta: +{c} created, +{conf} confirmed, -{d} denied/pruned")

        # Decision & Subgoal Trace
        if subgoal_trace:
            print("   🧭 Subgoal Decision Trace (How Agent Planned Using Memory):")
            for tr in subgoal_trace:
                st = tr.get("step")
                status = tr.get("status")
                target = tr.get("target")
                
                # Human-readable meaning
                meaning = ""
                if "e20" in target:
                    meaning = "Goal: Crafting Table"
                elif "e15" in target:
                    meaning = "Goal: Wood Pickaxe"
                elif "e16" in target:
                    meaning = "Goal: Wood Sword"
                elif "e5" in target:
                    meaning = "Goal: Mine Stone"
                elif "e6" in target:
                    meaning = "Goal: Harvest Wood"
                elif "e18" in target:
                    meaning = "Goal: Place Plant"
                elif "e4" in target:
                    meaning = "Goal: Collect Sapling"
                elif "e2" in target:
                    meaning = "Goal: Drink Water"
                elif "e8" in target:
                    meaning = "Goal: Defeat Zombie"
                elif "e9" in target:
                    meaning = "Goal: Eat Cow"
                elif "e19" in target:
                    meaning = "Goal: Place Stone"
                elif "e21" in target:
                    meaning = "Goal: Wake Up"
                
                print(f"      • Step {st:3d}: Status = {status:<22s} | Target = {target:<18s} | {meaning}")
        else:
            print("   🧭 Subgoal Decision Trace: Direct reactive MPC (no high-level subgoal active)")

    print("\n" + "=" * 80)
    print("Audit Complete.")


def main():
    parser = argparse.ArgumentParser(description="Inspect Dual-Store agent in-game decisions and memory usage.")
    parser.add_argument("--results", type=str, default="results/screen_v3_armC_dual_store_50ep.json")
    parser.add_argument("--bank", type=str, default="results/dual_store_memory_bank_online.json")
    parser.add_argument("--episode", type=int, default=None, help="Inspect a specific episode number")
    args = parser.parse_args()

    inspect_decisions(args.results, args.bank, args.episode)


if __name__ == "__main__":
    main()
