"""Independent benchmark score verification and provenance auditing tool.
Validates Crafter results artifacts per ORBIT v1.1 Section C.7 and Gate 0.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from eval.metrics import CRAFTER_ACHIEVEMENTS


def verify_result_artifact(path: str | Path) -> Dict[str, Any]:
    """Verify an evaluation result JSON file by independently recomputing metrics."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Result artifact not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    num_episodes = len(episodes)
    if num_episodes == 0:
        raise ValueError(f"No episodes found in {filepath}")

    # 1. Independently rebuild unlock counts from raw per-episode lists
    rebuilt_unlock_counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
    for ep in episodes:
        unlocked = set(ep.get("unlocked", []))
        for ach in CRAFTER_ACHIEVEMENTS:
            if ach in unlocked:
                rebuilt_unlock_counts[ach] += 1

    rebuilt_percentages = {
        ach: (count / num_episodes) * 100.0
        for ach, count in rebuilt_unlock_counts.items()
    }

    # 2. Recompute Crafter score using official formula
    log_sum = sum(math.log(1.0 + pct) for pct in rebuilt_percentages.values())
    recomputed_score = math.exp(log_sum / len(CRAFTER_ACHIEVEMENTS)) - 1.0

    reported_score = float(data.get("crafter_score", 0.0))
    reported_percentages = data.get("unlock_percentages", {})

    # Check percentages agreement
    pct_mismatches = {}
    for ach in CRAFTER_ACHIEVEMENTS:
        reported_pct = reported_percentages.get(ach)
        rebuilt_pct = rebuilt_percentages[ach]
        if reported_pct is None or abs(reported_pct - rebuilt_pct) > 1e-4:
            pct_mismatches[ach] = {
                "reported": reported_pct,
                "recomputed": rebuilt_pct,
            }

    score_match = abs(recomputed_score - reported_score) < 0.01

    return {
        "file": str(filepath),
        "checkpoint": data.get("checkpoint"),
        "num_episodes": num_episodes,
        "reported_score": reported_score,
        "recomputed_score": recomputed_score,
        "score_match": score_match,
        "percentage_mismatches": pct_mismatches,
        "rebuilt_percentages": rebuilt_percentages,
    }


def compare_artifacts(path_a: str | Path, path_b: str | Path) -> Dict[str, Any]:
    """Compare two evaluation artifacts declaring identical checkpoints (R01 audit)."""
    res_a = verify_result_artifact(path_a)
    res_b = verify_result_artifact(path_b)

    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    same_checkpoint = data_a.get("checkpoint") == data_b.get("checkpoint")
    same_seeds = data_a.get("seed_range") == data_b.get("seed_range")

    divergent_episodes = []
    episodes_a = {ep["seed"]: ep for ep in data_a.get("episodes", [])}
    episodes_b = {ep["seed"]: ep for ep in data_b.get("episodes", [])}

    shared_seeds = sorted(set(episodes_a.keys()).intersection(episodes_b.keys()))
    for seed in shared_seeds:
        ea = episodes_a[seed]
        eb = episodes_b[seed]
        if ea.get("steps") != eb.get("steps") or set(ea.get("unlocked", [])) != set(eb.get("unlocked", [])):
            divergent_episodes.append({
                "seed": seed,
                "steps_a": ea.get("steps"),
                "steps_b": eb.get("steps"),
                "unlocked_a": ea.get("unlocked"),
                "unlocked_b": eb.get("unlocked"),
            })

    provenance_status = "VERIFIED_IDENTICAL" if (not divergent_episodes and same_checkpoint and same_seeds) else "PROVENANCE_UNRESOLVED"

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "same_checkpoint_declared": same_checkpoint,
        "same_seed_range_declared": same_seeds,
        "score_a": res_a["recomputed_score"],
        "score_b": res_b["recomputed_score"],
        "divergent_episode_count": len(divergent_episodes),
        "divergent_episodes_sample": divergent_episodes[:5],
        "provenance_status": provenance_status,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and audit Crafter result JSON artifacts.")
    parser.add_argument("--artifact", help="Path to single result artifact")
    parser.add_argument("--compare", nargs=2, help="Paths to two artifacts to compare")
    args = parser.parse_args()

    if args.compare:
        report = compare_artifacts(args.compare[0], args.compare[1])
        print(json.dumps(report, indent=2))
    elif args.artifact:
        report = verify_result_artifact(args.artifact)
        print(json.dumps(report, indent=2))
    else:
        # Default: run R01 comparison on checked-in v3 files
        report = compare_artifacts(
            "results/screen_v3_champion_clean_50ep.json",
            "results/screen_v3_t2_final_results.json"
        )
        print(json.dumps(report, indent=2))
