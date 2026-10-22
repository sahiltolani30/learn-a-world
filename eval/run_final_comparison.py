"""
Phase 4: Locked Final Evaluation and Promotion Decision.
Evaluates four agents under identical conditions (100 episodes, seeds 1000..1099):
1. v1_hybrid: eval_crafter.py cycle-4 settings with reflexes on and do_bias 0.35
2. v1_pure:   eval_crafter_v1_ablation.py with reflexes off and do_bias 0.0
3. v2_pure:   eval_v2.py pure learned agent (RewardValuePlanner + AgentBelief, zero info-leak)
4. random:    pure random baseline

Promotion Rule (§6 Step 4):
v2 replaces v1 iff:
    v2_pure_score > v1_hybrid_score AND v2_pure_score >= v1_pure_score + 1.0

Outputs:
- results/final_comparison.json
- results/promotion_decision.md
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.eval_crafter_v1_ablation import evaluate_crafter_ablation
from eval.eval_v2 import evaluate_v2
from eval.metrics import CRAFTER_ACHIEVEMENTS

def run_evaluation_suite(
    v1_checkpoint: str = "checkpoints/jepa_crafter.pt",
    v2_checkpoint: str = "checkpoints/jepa_crafter_v2_best.pt",
    v1_config: str = "experiments/configs/crafter.yaml",
    v2_config: str = "experiments/configs/crafter_v2.yaml",
    episodes: int = 100,
    max_steps: int = 1000,
    seed_start: int = 1000,
    device_override: Optional[str] = None,
    agents: Optional[List[str]] = None,
    out_json: str = "results/final_comparison.json",
    out_md: str = "results/promotion_decision.md",
) -> Dict[str, Any]:
    """
    Executes the 4-agent comparison protocol and generates promotion decision.
    """
    selected_agents = agents or ["v1_hybrid", "v1_pure", "v2_pure", "random"]
    print("=" * 80)
    print("      PHASE 4: LOCKED FINAL EVALUATION & PROMOTION DECISION PROTOCOL")
    print(f"      Episodes: {episodes} | Seeds: {seed_start}..{seed_start + episodes - 1} | Max Steps: {max_steps}")
    print("=" * 80)

    # Check v2 checkpoint fallback if best doesn't exist yet
    v2_ckpt_path = Path(v2_checkpoint)
    if not v2_ckpt_path.exists():
        fallback = Path("checkpoints/jepa_crafter_v2.pt")
        if fallback.exists():
            print(f"Notice: '{v2_checkpoint}' not found; falling back to '{fallback}'.")
            v2_ckpt_path = fallback
        else:
            print(f"Notice: Neither '{v2_checkpoint}' nor '{fallback}' found; will use initial checkpoint if needed.")

    results: Dict[str, Any] = {}

    # 1. v1_hybrid
    if "v1_hybrid" in selected_agents:
        print("\n" + "-" * 70)
        print(">>> [Agent 1/4] Evaluating v1_hybrid (Reflexes ON, do_bias=0.35)...")
        print("-" * 70)
        res_hybrid = evaluate_crafter_ablation(
            checkpoint_path=v1_checkpoint,
            config_path=v1_config,
            num_episodes=episodes,
            max_steps=max_steps,
            device_override=device_override,
            mode="mpc",
            reflexes="on",
            do_bias=0.35,
            seed_offset=seed_start,
        )
        results["v1_hybrid"] = {
            "evaluator": "eval/eval_crafter_v1_ablation.py",
            "checkpoint": v1_checkpoint,
            "reflexes": "on",
            "do_bias": 0.35,
            "crafter_score": res_hybrid["crafter_score"],
            "unlocked_count": res_hybrid["unlocked_count"],
            "mean_survival_steps": res_hybrid["mean_survival_steps"],
            "mean_reward": res_hybrid["mean_reward"],
            "per_achievement_rates": res_hybrid.get("per_achievement_rates", res_hybrid.get("unlock_percentages", {})),
        }

    # 2. v1_pure
    if "v1_pure" in selected_agents:
        print("\n" + "-" * 70)
        print(">>> [Agent 2/4] Evaluating v1_pure (Reflexes OFF, do_bias=0.0)...")
        print("-" * 70)
        res_pure = evaluate_crafter_ablation(
            checkpoint_path=v1_checkpoint,
            config_path=v1_config,
            num_episodes=episodes,
            max_steps=max_steps,
            device_override=device_override,
            mode="mpc",
            reflexes="off",
            do_bias=0.0,
            seed_offset=seed_start,
        )
        results["v1_pure"] = {
            "evaluator": "eval/eval_crafter_v1_ablation.py",
            "checkpoint": v1_checkpoint,
            "reflexes": "off",
            "do_bias": 0.0,
            "crafter_score": res_pure["crafter_score"],
            "unlocked_count": res_pure["unlocked_count"],
            "mean_survival_steps": res_pure["mean_survival_steps"],
            "mean_reward": res_pure["mean_reward"],
            "per_achievement_rates": res_pure.get("per_achievement_rates", res_pure.get("unlock_percentages", {})),
        }

    # 3. v2_pure
    if "v2_pure" in selected_agents:
        print("\n" + "-" * 70)
        print(">>> [Agent 3/4] Evaluating v2_pure (Pure Learned RewardValuePlanner)...")
        print("-" * 70)
        res_v2 = evaluate_v2(
            checkpoint_path=str(v2_ckpt_path),
            config_path=v2_config,
            num_episodes=episodes,
            seed_start=seed_start,
            max_steps=max_steps,
            device_override=device_override,
        )
        results["v2_pure"] = {
            "evaluator": "eval/eval_v2.py",
            "checkpoint": str(v2_ckpt_path),
            "reflexes": "none (pure learned)",
            "do_bias": 0.0,
            "crafter_score": res_v2["crafter_score"],
            "unlocked_count": res_v2["unlocked_count"],
            "mean_survival_steps": res_v2["mean_survival_steps"],
            "mean_reward": res_v2["mean_reward"],
            "per_achievement_rates": res_v2.get("per_achievement_rates", res_v2.get("unlock_percentages", {})),
            "debug_decomposition": res_v2.get("debug_decomposition", {}),
        }

    # 4. random
    if "random" in selected_agents:
        print("\n" + "-" * 70)
        print(">>> [Agent 4/4] Evaluating random baseline...")
        print("-" * 70)
        res_rand = evaluate_crafter_ablation(
            checkpoint_path=None,
            config_path=v1_config,
            num_episodes=episodes,
            max_steps=max_steps,
            device_override=device_override,
            mode="random",
            seed_offset=seed_start,
        )
        results["random"] = {
            "evaluator": "eval/eval_crafter_v1_ablation.py",
            "checkpoint": "none",
            "crafter_score": res_rand["crafter_score"],
            "unlocked_count": res_rand["unlocked_count"],
            "mean_survival_steps": res_rand["mean_survival_steps"],
            "mean_reward": res_rand["mean_reward"],
            "per_achievement_rates": res_rand.get("per_achievement_rates", res_rand.get("unlock_percentages", {})),
        }

    # Promotion Rule Verification:
    # v2 replaces v1 iff:
    #   v2_pure_score > v1_hybrid_score AND v2_pure_score >= v1_pure_score + 1.0
    score_v2_pure = results.get("v2_pure", {}).get("crafter_score", 0.0)
    score_v1_hybrid = results.get("v1_hybrid", {}).get("crafter_score", 0.0)
    score_v1_pure = results.get("v1_pure", {}).get("crafter_score", 0.0)

    cond_beat_hybrid = bool(score_v2_pure > score_v1_hybrid)
    cond_beat_pure_plus_one = bool(score_v2_pure >= score_v1_pure + 1.0)
    is_promoted = bool(cond_beat_hybrid and cond_beat_pure_plus_one)

    decision_info = {
        "promoted": is_promoted,
        "rule": "v2_pure_score > v1_hybrid_score AND v2_pure_score >= v1_pure_score + 1.0",
        "v2_pure_score": score_v2_pure,
        "v1_hybrid_score": score_v1_hybrid,
        "v1_pure_score": score_v1_pure,
        "condition_beat_hybrid": cond_beat_hybrid,
        "condition_beat_pure_plus_one": cond_beat_pure_plus_one,
        "margin_over_hybrid": round(score_v2_pure - score_v1_hybrid, 2),
        "margin_over_pure": round(score_v2_pure - score_v1_pure, 2),
    }

    full_output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "protocol": {
            "episodes": episodes,
            "seed_start": seed_start,
            "seed_end": seed_start + episodes - 1,
            "max_steps": max_steps,
        },
        "promotion_decision": decision_info,
        "agents": results,
    }

    # Save results/final_comparison.json
    out_json_file = Path(out_json)
    out_json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_file, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n--> Saved comparison results JSON to: {out_json_file}")

    # Generate results/promotion_decision.md
    generate_markdown_report(full_output, out_md)
    print(f"--> Saved promotion decision Markdown to: {out_md}")

    # Print summary table to console
    print("\n" + "=" * 80)
    print(f"                 FINAL DECISION: {'✅ PROMOTED TO V2' if is_promoted else '❌ PROMOTION FAILED'}")
    print("=" * 80)
    print(f"  v2 Pure Score:   {score_v2_pure:.2f}%")
    print(f"  v1 Hybrid Score: {score_v1_hybrid:.2f}%  (Target: > {score_v1_hybrid:.2f}%) -> {'PASS' if cond_beat_hybrid else 'FAIL'}")
    print(f"  v1 Pure Score:   {score_v1_pure:.2f}%  (Target: >= {score_v1_pure + 1.0:.2f}%) -> {'PASS' if cond_beat_pure_plus_one else 'FAIL'}")
    print("=" * 80 + "\n")

    return full_output

def generate_markdown_report(data: Dict[str, Any], out_path: str) -> None:
    """Writes results/promotion_decision.md."""
    dec = data["promotion_decision"]
    proto = data["protocol"]
    agents = data["agents"]

    status_str = "PROMOTED TO V2" if dec["promoted"] else "PROMOTION FAILED"
    status_icon = "✅" if dec["promoted"] else "❌"

    lines = [
        f"# Learn-a-World v2: Promotion Decision Report",
        f"",
        f"**Status:** {status_icon} **{status_str}**  ",
        f"**Date:** {data['timestamp'][:19].replace('T', ' ')}  ",
        f"**Evaluation Protocol:** {proto['episodes']} episodes on fixed seeds {proto['seed_start']}..{proto['seed_end']} (max {proto['max_steps']} steps/ep)  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Comparison Table",
        f"",
        f"| Agent | Evaluator | Checkpoint | Reflexes | Crafter Score | Unlocked Ach. | Mean Survival | Mean Reward |",
        f"|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for name, res in agents.items():
        ck = Path(res['checkpoint']).name if res['checkpoint'] != 'none' else 'none'
        refl = res.get('reflexes', 'none')
        lines.append(
            f"| **{name}** | `{res['evaluator']}` | `{ck}` | {refl} | **{res['crafter_score']:.2f}%** | {res['unlocked_count']}/22 | {res['mean_survival_steps']:.1f} | {res['mean_reward']:.2f} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Promotion Criteria Audit",
        f"",
        f"The v2 promotion rule requires satisfaction of both clauses (§6 Step 4):",
        f"> **Rule:** `v2_pure_score > v1_hybrid_score` **AND** `v2_pure_score >= v1_pure_score + 1.0`",
        f"",
        f"- **Clause 1 (`v2_pure_score > v1_hybrid_score`):**  ",
        f"  `{dec['v2_pure_score']:.2f}% > {dec['v1_hybrid_score']:.2f}%` -> **{'PASS ✅' if dec['condition_beat_hybrid'] else 'FAIL ❌'}** (Margin: `{dec['margin_over_hybrid']:+.2f}%`)",
        f"- **Clause 2 (`v2_pure_score >= v1_pure_score + 1.0`):**  ",
        f"  `{dec['v2_pure_score']:.2f}% >= {dec['v1_pure_score'] + 1.0:.2f}%` -> **{'PASS ✅' if dec['condition_beat_pure_plus_one'] else 'FAIL ❌'}** (Margin over pure: `{dec['margin_over_pure']:+.2f}%`)",
        f"",
        f"**Overall Verdict:** {status_icon} **{status_str}**",
        f"",
    ])

    if not dec["promoted"]:
        lines.extend([
            f"> [!WARNING]",
            f"> Phase 4 promotion did not trigger. Per the v2 implementation plan, the next step is to activate **Phase 5: Hierarchical Achievement Layer** to mine the achievement dependency graph and guide the planner with explicit subgoal unlocking bonuses.",
            f"",
        ])
    else:
        lines.extend([
            f"> [!NOTE]",
            f"> Phase 4 promotion passed! The pure learned model without hardcoded heuristic reflexes successfully surpassed the hybrid baseline under the locked evaluation protocol.",
            f"",
        ])

    lines.extend([
        f"---",
        f"",
        f"## 3. Achievement Breakdown by Agent",
        f"",
        f"| Achievement | v1_hybrid (%) | v1_pure (%) | v2_pure (%) | random (%) |",
        f"|:---|:---:|:---:|:---:|:---:|",
    ])

    for ach in CRAFTER_ACHIEVEMENTS:
        r_hyb = agents.get("v1_hybrid", {}).get("per_achievement_rates", {}).get(ach, 0.0)
        r_pur = agents.get("v1_pure", {}).get("per_achievement_rates", {}).get(ach, 0.0)
        r_v2 = agents.get("v2_pure", {}).get("per_achievement_rates", {}).get(ach, 0.0)
        r_rnd = agents.get("random", {}).get("per_achievement_rates", {}).get(ach, 0.0)
        lines.append(f"| `{ach}` | {r_hyb:.1f}% | {r_pur:.1f}% | {r_v2:.1f}% | {r_rnd:.1f}% |")

    lines.append("")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Locked Final Comparison & Promotion Decision")
    parser.add_argument("--v1-checkpoint", type=str, default="checkpoints/jepa_crafter.pt")
    parser.add_argument("--v2-checkpoint", type=str, default="checkpoints/jepa_crafter_v2_best.pt")
    parser.add_argument("--v1-config", type=str, default="experiments/configs/crafter.yaml")
    parser.add_argument("--v2-config", type=str, default="experiments/configs/crafter_v2.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated list of agents to evaluate (default: all 4)")
    parser.add_argument("--out-json", type=str, default="results/final_comparison.json")
    parser.add_argument("--out-md", type=str, default="results/promotion_decision.md")
    args = parser.parse_args()

    agent_list = [a.strip() for a in args.agents.split(",")] if args.agents else None

    run_evaluation_suite(
        v1_checkpoint=args.v1_checkpoint,
        v2_checkpoint=args.v2_checkpoint,
        v1_config=args.v1_config,
        v2_config=args.v2_config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
        device_override=args.device,
        agents=agent_list,
        out_json=args.out_json,
        out_md=args.out_md,
    )

if __name__ == "__main__":
    main()
