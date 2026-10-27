#!/usr/bin/env python3
"""
Paired Statistical Analysis of Ablation C (Active Consequence Scoring)
versus Clean T2 Baseline (Ablation A) and Frozen Candidate (Ablation B)
across matched seeds 2050..2099 under Protocol v2.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats


def compute_crafter_score(unlock_rates):
    ALL_ACHIEVEMENTS = [
        "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
        "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
        "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
        "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
        "make_wood_pickaxe", "make_wood_sword", "place_furnace",
        "place_plant", "place_stone", "place_table", "wake_up"
    ]
    vals = [unlock_rates.get(k, 0.0) for k in ALL_ACHIEVEMENTS]
    log_terms = [np.log(1.0 + v) for v in vals]
    return float(np.exp(np.mean(log_terms)) - 1.0)


def bootstrap_paired_diff(arr_a, arr_b, num_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = arr_b - arr_a
    mean_diff = float(np.mean(diffs))
    n = len(diffs)
    indices = rng.integers(0, n, size=(num_bootstrap, n))
    boot_means = np.mean(diffs[indices], axis=1)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_diff, ci_lower, ci_upper


def bootstrap_crafter_score_diff(eps_a, eps_b, num_bootstrap=10000, seed=42):
    ALL_ACHIEVEMENTS = [
        "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
        "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
        "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
        "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
        "make_wood_pickaxe", "make_wood_sword", "place_furnace",
        "place_plant", "place_stone", "place_table", "wake_up"
    ]
    rng = np.random.default_rng(seed)
    n = len(eps_a)
    mat_a = np.zeros((n, len(ALL_ACHIEVEMENTS)), dtype=np.float32)
    mat_b = np.zeros((n, len(ALL_ACHIEVEMENTS)), dtype=np.float32)
    for i, ep in enumerate(eps_a):
        for ach in ep.get("unlocked", []):
            if ach in ALL_ACHIEVEMENTS:
                mat_a[i, ALL_ACHIEVEMENTS.index(ach)] = 1.0
    for i, ep in enumerate(eps_b):
        for ach in ep.get("unlocked", []):
            if ach in ALL_ACHIEVEMENTS:
                mat_b[i, ALL_ACHIEVEMENTS.index(ach)] = 1.0

    boot_diffs = []
    indices = rng.integers(0, n, size=(num_bootstrap, n))
    for idx_row in indices:
        rates_a = np.mean(mat_a[idx_row], axis=0) * 100.0
        rates_b = np.mean(mat_b[idx_row], axis=0) * 100.0
        score_a = np.exp(np.mean(np.log(1.0 + rates_a))) - 1.0
        score_b = np.exp(np.mean(np.log(1.0 + rates_b))) - 1.0
        boot_diffs.append(score_b - score_a)
    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return ci_lower, ci_upper


def main():
    root = Path(__file__).resolve().parent.parent
    clean_path = root / "results" / "screen_v3_t2_clean_protocol_v2_50ep.json"
    frozen_path = root / "results" / "screen_v3_phase1_5_frozen_protocol_v2_50ep.json"
    active_path = root / "results" / "screen_v3_ablation_c_active_50ep.json"
    diag_dir = root / "results" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_path = diag_dir / "ablation_c_active_comparison.json"

    with open(clean_path, "r") as f:
        data_clean = json.load(f)
    with open(frozen_path, "r") as f:
        data_frozen = json.load(f)
    with open(active_path, "r") as f:
        data_active = json.load(f)

    eps_clean = data_clean["episodes"]
    eps_frozen = data_frozen["episodes"]
    eps_active = data_active["episodes"]

    r_clean = np.array([ep["reward"] for ep in eps_clean])
    r_frozen = np.array([ep["reward"] for ep in eps_frozen])
    r_active = np.array([ep["reward"] for ep in eps_active])

    s_clean = np.array([ep["steps"] for ep in eps_clean])
    s_frozen = np.array([ep["steps"] for ep in eps_frozen])
    s_active = np.array([ep["steps"] for ep in eps_active])

    a_clean = np.array([len(ep["unlocked"]) for ep in eps_clean])
    a_frozen = np.array([len(ep["unlocked"]) for ep in eps_frozen])
    a_active = np.array([len(ep["unlocked"]) for ep in eps_active])

    # Comparisons
    # C vs Clean T2
    diff_r_c_clean, r_ci_low_clean, r_ci_high_clean = bootstrap_paired_diff(r_clean, r_active)
    diff_s_c_clean, s_ci_low_clean, s_ci_high_clean = bootstrap_paired_diff(s_clean, s_active)
    diff_a_c_clean, a_ci_low_clean, a_ci_high_clean = bootstrap_paired_diff(a_clean, a_active)
    p_r_clean = float(stats.ttest_rel(r_active, r_clean).pvalue)
    p_s_clean = float(stats.ttest_rel(s_active, s_clean).pvalue)
    p_a_clean = float(stats.ttest_rel(a_active, a_clean).pvalue)

    score_ci_low_clean, score_ci_high_clean = bootstrap_crafter_score_diff(eps_clean, eps_active)

    # C vs Frozen B
    diff_r_c_frozen, r_ci_low_frozen, r_ci_high_frozen = bootstrap_paired_diff(r_frozen, r_active)
    diff_s_c_frozen, s_ci_low_frozen, s_ci_high_frozen = bootstrap_paired_diff(s_frozen, s_active)
    diff_a_c_frozen, a_ci_low_frozen, a_ci_high_frozen = bootstrap_paired_diff(a_frozen, a_active)
    p_r_frozen = float(stats.ttest_rel(r_active, r_frozen).pvalue)
    p_s_frozen = float(stats.ttest_rel(s_active, s_frozen).pvalue)
    p_a_frozen = float(stats.ttest_rel(a_active, a_frozen).pvalue)

    score_ci_low_frozen, score_ci_high_frozen = bootstrap_crafter_score_diff(eps_frozen, eps_active)

    report = {
        "metadata": {
            "protocol_id": "strict_visual_jepa_clean_v2",
            "evaluation_seeds": [2050, 2099],
            "num_episodes": 50,
            "ablation_a": "Clean T2 Baseline",
            "ablation_b": "Frozen Phase 1.5 Candidate (Shadow)",
            "ablation_c": "Active Consequence Candidate (Transition Active)"
        },
        "scores": {
            "ablation_a_clean_t2": data_clean["crafter_score"],
            "ablation_b_frozen": data_frozen["crafter_score"],
            "ablation_c_active": data_active["crafter_score"],
            "gain_c_vs_a_pp": round(data_active["crafter_score"] - data_clean["crafter_score"], 2),
            "gain_c_vs_b_pp": round(data_active["crafter_score"] - data_frozen["crafter_score"], 2),
            "c_vs_a_score_95_ci": [round(score_ci_low_clean, 2), round(score_ci_high_clean, 2)],
            "c_vs_b_score_95_ci": [round(score_ci_low_frozen, 2), round(score_ci_high_frozen, 2)],
        },
        "rewards": {
            "ablation_a": float(np.mean(r_clean)),
            "ablation_b": float(np.mean(r_frozen)),
            "ablation_c": float(np.mean(r_active)),
            "gain_c_vs_a": round(diff_r_c_clean, 2),
            "gain_c_vs_a_pvalue": p_r_clean,
            "gain_c_vs_b": round(diff_r_c_frozen, 2),
            "gain_c_vs_b_pvalue": p_r_frozen,
        },
        "survival_steps": {
            "ablation_a": float(np.mean(s_clean)),
            "ablation_b": float(np.mean(s_frozen)),
            "ablation_c": float(np.mean(s_active)),
            "gain_c_vs_a": round(diff_s_c_clean, 1),
            "gain_c_vs_b": round(diff_s_c_frozen, 1),
        },
        "achievements_per_ep": {
            "ablation_a": float(np.mean(a_clean)),
            "ablation_b": float(np.mean(a_frozen)),
            "ablation_c": float(np.mean(a_active)),
            "gain_c_vs_a": round(diff_a_c_clean, 2),
            "gain_c_vs_b": round(diff_a_c_frozen, 2),
        },
        "key_achievement_unlock_rates": {
            "make_wood_pickaxe": {
                "ablation_a": data_clean["unlock_percentages"].get("make_wood_pickaxe", 0.0),
                "ablation_b": data_frozen["unlock_percentages"].get("make_wood_pickaxe", 0.0),
                "ablation_c": data_active["unlock_percentages"].get("make_wood_pickaxe", 0.0),
            },
            "make_wood_sword": {
                "ablation_a": data_clean["unlock_percentages"].get("make_wood_sword", 0.0),
                "ablation_b": data_frozen["unlock_percentages"].get("make_wood_sword", 0.0),
                "ablation_c": data_active["unlock_percentages"].get("make_wood_sword", 0.0),
            },
            "place_table": {
                "ablation_a": data_clean["unlock_percentages"].get("place_table", 0.0),
                "ablation_b": data_frozen["unlock_percentages"].get("place_table", 0.0),
                "ablation_c": data_active["unlock_percentages"].get("place_table", 0.0),
            },
            "collect_wood": {
                "ablation_a": data_clean["unlock_percentages"].get("collect_wood", 0.0),
                "ablation_b": data_frozen["unlock_percentages"].get("collect_wood", 0.0),
                "ablation_c": data_active["unlock_percentages"].get("collect_wood", 0.0),
            },
            "defeat_zombie": {
                "ablation_a": data_clean["unlock_percentages"].get("defeat_zombie", 0.0),
                "ablation_b": data_frozen["unlock_percentages"].get("defeat_zombie", 0.0),
                "ablation_c": data_active["unlock_percentages"].get("defeat_zombie", 0.0),
            }
        }
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("           ABLATION C (ACTIVE CONSEQUENCE) STATISTICAL REPORT")
    print("=" * 80)
    print(f"Official Crafter Score: {data_active['crafter_score']:.2f}% (Clean T2: {data_clean['crafter_score']:.2f}%, Frozen: {data_frozen['crafter_score']:.2f}%)")
    print(f"Score Delta vs T2:     +{data_active['crafter_score'] - data_clean['crafter_score']:.2f} pp [95% CI: {score_ci_low_clean:.2f}, {score_ci_high_clean:.2f}]")
    print(f"Mean Episode Reward:    {np.mean(r_active):.2f} +/- {np.std(r_active):.2f} (Clean T2: {np.mean(r_clean):.2f}, p={p_r_clean:.4e})")
    print(f"Mean Survival Steps:    {np.mean(s_active):.1f} steps (Clean T2: {np.mean(s_clean):.1f})")
    print(f"Mean Achievements / Ep: {np.mean(a_active):.2f} (Clean T2: {np.mean(a_clean):.2f}, p={p_a_clean:.4e})")
    print("\nKey Achievement Deltas:")
    print(f"  Wood Pickaxe: {data_clean['unlock_percentages'].get('make_wood_pickaxe', 0.0)}% -> {data_active['unlock_percentages'].get('make_wood_pickaxe', 0.0)}% (+{data_active['unlock_percentages'].get('make_wood_pickaxe', 0.0) - data_clean['unlock_percentages'].get('make_wood_pickaxe', 0.0)}%)")
    print(f"  Wood Sword:   {data_clean['unlock_percentages'].get('make_wood_sword', 0.0)}% -> {data_active['unlock_percentages'].get('make_wood_sword', 0.0)}% (+{data_active['unlock_percentages'].get('make_wood_sword', 0.0) - data_clean['unlock_percentages'].get('make_wood_sword', 0.0)}%)")
    print(f"  Place Table:  {data_clean['unlock_percentages'].get('place_table', 0.0)}% -> {data_active['unlock_percentages'].get('place_table', 0.0)}% (+{data_active['unlock_percentages'].get('place_table', 0.0) - data_clean['unlock_percentages'].get('place_table', 0.0)}%)")
    print(f"  Defeat Zombie:{data_clean['unlock_percentages'].get('defeat_zombie', 0.0)}% -> {data_active['unlock_percentages'].get('defeat_zombie', 0.0)}% (+{data_active['unlock_percentages'].get('defeat_zombie', 0.0) - data_clean['unlock_percentages'].get('defeat_zombie', 0.0)}%)")
    print("=" * 80)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
