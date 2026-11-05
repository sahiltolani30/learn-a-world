"""Tier 4: Scenario 2 - Crafter Geometric Score Metric Calculator (>= 8.0%).

Covers:
- F3 (Causal Tool Progression), F21 (Benchmark Score Calculation).
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

# Cleanly bind eval.metrics without executing eval/__init__.py
if "eval" not in sys.modules:
    eval_mod = types.ModuleType("eval")
    sys.modules["eval"] = eval_mod
if "eval.metrics" not in sys.modules:
    spec = importlib.util.spec_from_file_location("eval.metrics", str(REPO_ROOT / "eval" / "metrics.py"))
    metrics_mod = importlib.util.module_from_spec(spec)
    sys.modules["eval.metrics"] = metrics_mod
    spec.loader.exec_module(metrics_mod)
    setattr(sys.modules["eval"], "metrics", metrics_mod)

from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score


def test_t4_score_01_geometric_mean_mathematical_derivation():
    """T4-SCORE-01: Analytical validation of geometric mean formula exp(mean(ln(1+s))) - 1."""
    # Case A: Uniform 10% unlock across all 22 achievements
    # Analytical: exp(ln(1 + 10.0)) - 1 = 10.0%
    records_10 = [{ach: 1 if (ep < 5) else 0 for ach in CRAFTER_ACHIEVEMENTS} for ep in range(50)]
    stats_10 = compute_crafter_score(records_10)
    assert abs(stats_10["score"] - 10.0) < 1e-4

    # Case B: Uniform 100% unlock across all 22 achievements
    # Analytical: exp(ln(1 + 100.0)) - 1 = 100.0%
    records_100 = [{ach: 1 for ach in CRAFTER_ACHIEVEMENTS} for _ in range(50)]
    stats_100 = compute_crafter_score(records_100)
    assert abs(stats_100["score"] - 100.0) < 1e-4

    # Case C: Uniform 0% unlock across all achievements
    # Analytical: exp(ln(1 + 0.0)) - 1 = 0.0%
    records_0 = [{ach: 0 for ach in CRAFTER_ACHIEVEMENTS} for _ in range(50)]
    stats_0 = compute_crafter_score(records_0)
    assert abs(stats_0["score"] - 0.0) < 1e-5


def test_t4_score_02_zero_preserving_logarithmic_regularization():
    """T4-SCORE-02: Zero-preserving +1 regularization prevents collapse on rare unachieved tasks."""
    # 18 achievements unlocked in 25/50 episodes (50%), 4 achievements at 0%
    ach_50 = CRAFTER_ACHIEVEMENTS[:18]
    ach_0 = CRAFTER_ACHIEVEMENTS[18:]

    records = []
    for ep in range(50):
        rec = {}
        for ach in ach_50:
            rec[ach] = 1 if ep < 25 else 0
        for ach in ach_0:
            rec[ach] = 0
        records.append(rec)

    stats = compute_crafter_score(records)

    # Analytical calculation: exp((18 * ln(51) + 4 * ln(1)) / 22) - 1
    # = exp((18 * 3.9318256) / 22) - 1 = exp(3.216948) - 1 = 24.9515 - 1 = 23.9515%
    expected = math.exp((18 * math.log(51.0)) / 22.0) - 1.0
    assert abs(stats["score"] - expected) < 1e-3
    assert abs(stats["score"] - 23.9515) < 0.05
    assert stats["score"] > 0.0, "Score collapsed to zero despite 18 achievements unlocked"


def test_t4_score_03_acceptance_threshold_validation():
    """T4-SCORE-03: Benchmark milestone profile evaluates to >= 8.0% Crafter score."""
    # Target profile: progression up to collect_stone with strong basic survival
    # 50 evaluation episodes
    num_episodes = 50
    records = []

    for ep in range(num_episodes):
        rec = {}
        # Basic survival achievements: ~95%
        for ach in ["collect_sapling", "wake_up", "eat_cow", "collect_drink", "eat_plant"]:
            rec[ach] = 1 if ep < 48 else 0  # 96%

        # Causal tool chain & nightfall combat:
        rec["collect_wood"] = 1 if ep < 45 else 0       # 90%
        rec["place_table"] = 1 if ep < 42 else 0        # 84%
        rec["make_wood_pickaxe"] = 1 if ep < 35 else 0  # 70%
        rec["collect_stone"] = 1 if ep < 25 else 0      # 50%
        rec["make_wood_sword"] = 1 if ep < 25 else 0    # 50%
        rec["place_plant"] = 1 if ep < 25 else 0        # 50%
        rec["defeat_zombie"] = 1 if ep < 25 else 0      # 50%

        # Remaining 10 rare achievements: 0%
        for ach in CRAFTER_ACHIEVEMENTS:
            if ach not in rec:
                rec[ach] = 0
        records.append(rec)

    stats = compute_crafter_score(records)
    score = stats["score"]

    assert score >= 8.0, f"Score {score:.2f}% did not achieve target >= 8.0%"
    assert stats["unlocked_count"] == 12
    assert stats["unlock_percentages"]["collect_wood"] == 90.0
    assert stats["unlock_percentages"]["make_wood_pickaxe"] == 70.0
    assert stats["unlock_percentages"]["collect_stone"] == 50.0


def test_t4_score_04_historical_plateau_comparison_and_discrimination():
    """T4-SCORE-04: Accurate discrimination and monotonic ranking across historical agent plateaus."""
    def profile_to_score(pct_dict):
        records = []
        for ep in range(100):
            rec = {ach: 1 if ep < int(pct_dict.get(ach, 0)) else 0 for ach in CRAFTER_ACHIEVEMENTS}
            records.append(rec)
        return compute_crafter_score(records)["score"]

    # Historical profiles:
    # 1. Random baseline: ~1.77%
    random_score = profile_to_score({
        "collect_sapling": 30, "wake_up": 25, "eat_cow": 10, "collect_drink": 15, "eat_plant": 10,
    })

    # 2. v2_pure baseline: ~2.16%
    v2_score = profile_to_score({
        "collect_sapling": 40, "wake_up": 35, "eat_cow": 15, "collect_drink": 20, "eat_plant": 15,
        "collect_wood": 10,
    })

    # 3. Phase 3 Tranche T1: ~3.54%
    t1_score = profile_to_score({
        "collect_sapling": 60, "wake_up": 50, "eat_cow": 25, "collect_drink": 35, "eat_plant": 25,
        "collect_wood": 30, "place_table": 5,
    })

    # 4. Phase 3 Tranche T2 Champion: ~4.31%
    t2_score = profile_to_score({
        "collect_sapling": 70, "wake_up": 60, "eat_cow": 30, "collect_drink": 45, "eat_plant": 35,
        "collect_wood": 45, "place_table": 15, "make_wood_pickaxe": 3,
    })

    # 5. Target Zephyr Breakthrough: >= 8.0%
    zephyr_score = profile_to_score({
        "collect_sapling": 95, "wake_up": 95, "eat_cow": 95, "collect_drink": 95, "eat_plant": 95,
        "collect_wood": 90, "place_table": 85, "make_wood_pickaxe": 60, "collect_stone": 40,
        "make_wood_sword": 50, "place_plant": 50, "defeat_zombie": 50,
    })

    # Assert strict monotonic ranking
    assert random_score < v2_score < t1_score < t2_score < zephyr_score
    assert zephyr_score >= 8.0, f"Zephyr target score {zephyr_score:.2f}% < 8.0%"
