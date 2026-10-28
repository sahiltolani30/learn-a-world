#!/usr/bin/env python3
"""
50-Episode Gameplay Screening of Godfather JEPA on Standard Held-Out Benchmark Seeds (2050..2099).

Strict zero-information-leakage policy:
- No handcoded heuristics or privileged simulator state leaks.
- Operates purely on raw RGB visual observations (64x64x3) and scalar rewards.
- Preserves recurrent spatial feature memory (SpatialConvGRU 8x8x64) and vector belief across steps.
- Uses Causal Hypergraph recursive upstream budgeting and Homeostatic Arbiter for vital preservation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from godfather_jepa.training.train import GodfatherJEPAPipeline
from godfather_jepa.metrics.crafter_score import compute_crafter_score, CRAFTER_ACHIEVEMENTS


def run_screening_50eps(
    checkpoint: str,
    causal_memory_bank: Optional[str] = None,
    num_episodes: int = 50,
    seed_start: int = 2050,
    device_str: str = "cuda",
    output_json: str = "results/screen_godfather_50ep.json",
    drive_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes official 50-episode screening on locked held-out seeds [2050..2099].
    """
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    print(f"=== Starting Godfather JEPA 50-Episode Screening on {device} ===")
    print(f"Model Checkpoint:     {checkpoint}")
    print(f"Causal Memory Bank:   {causal_memory_bank}")
    print(f"Benchmark Seed Range: [{seed_start} .. {seed_start + num_episodes - 1}] ({num_episodes} games)")

    # 1. Initialize Pipeline
    pipeline = GodfatherJEPAPipeline(device=device)

    # 2. Load Checkpoint
    if os.path.exists(checkpoint):
        pipeline.load_checkpoint(checkpoint)
    else:
        print(f"[Warning] Checkpoint {checkpoint} not found. Running with current weights.")

    # 3. Load Memory Bank if explicitly provided
    if causal_memory_bank and os.path.exists(causal_memory_bank):
        pipeline.semantic_bank.load(causal_memory_bank)
        print(f"[Memory Bank] Successfully loaded {len(pipeline.semantic_bank.cards)} causal cards from {causal_memory_bank}")

    # 4. Generate locked held-out benchmark seeds
    seeds = list(range(seed_start, seed_start + num_episodes))

    # 5. Execute Evaluation
    t0 = time.time()
    results = pipeline.evaluate(seeds=seeds)
    elapsed = time.time() - t0

    crafter_score = results.get("crafter_score", 0.0)
    mean_reward = results.get("mean_reward", results.get("mean_episode_reward", 0.0))
    mean_survival = results.get("mean_survival_length", 0.0)
    unlocks = results.get("achievements", results.get("unlock_percentages", {}))

    # 6. Format Official JSON Output
    report = {
        "model": "Godfather-JEPA-World-Model",
        "benchmark_protocol": "official_50ep_locked_seeds",
        "num_episodes": num_episodes,
        "seed_range": [seeds[0], seeds[-1]],
        "crafter_score": crafter_score,
        "mean_episode_reward": mean_reward,
        "mean_survival_length": mean_survival,
        "unlock_percentages": unlocks,
        "total_runtime_seconds": elapsed,
        "episodes": results.get("episodes", []),
    }

    # Save to local output path
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Results] Saved screening results to {out_path}")

    # Mirror to Drive if requested
    if drive_dir and os.path.exists(drive_dir):
        drive_out = Path(drive_dir) / "results" / out_path.name
        drive_out.parent.mkdir(parents=True, exist_ok=True)
        with open(drive_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Drive Sync] Mirrored results to Google Drive: {drive_out}")

    # 7. Print Terminal Report
    print("\n" + "=" * 60)
    print("      GODFATHER JEPA: 50-EPISODE BENCHMARK REPORT      ")
    print("=" * 60)
    print(f"Official Crafter Score: {crafter_score:.2f}%")
    print(f"Mean Episode Reward:    {mean_reward:.2f}")
    print(f"Mean Survival Length:   {mean_survival:.1f} steps")
    print(f"Total Runtime:          {elapsed:.1f}s ({elapsed / num_episodes:.2f}s / game)")
    print("\nAchievement Unlock Rates (50 Games):")
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = unlocks.get(ach, 0.0)
        print(f"  {ach:22s}: {pct:5.1f}%")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Godfather JEPA 50-Episode Screening")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/godfather_jepa_final.pt")
    parser.add_argument("--causal_memory_bank", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="results/screen_godfather_50ep.json")
    parser.add_argument("--drive_dir", type=str, default=None)
    args = parser.parse_args()

    run_screening_50eps(
        checkpoint=args.checkpoint,
        causal_memory_bank=args.causal_memory_bank,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        device_str=args.device,
        output_json=args.output_json,
        drive_dir=args.drive_dir,
    )


if __name__ == "__main__":
    main()
