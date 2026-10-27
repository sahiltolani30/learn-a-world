#!/usr/bin/env python3
"""CLI runner for Sections 8 & 9 (Gates 8 & 9): Multi-Horizon & Intervention Evaluation.

Example on Colab:
    python3 scripts/run_eval_gates8_9.py \
      --manifest /content/drive/MyDrive/learn-a-world/results/phase1_5_manifest.json \
      --checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_state_readout.pt \
      --output_json /content/drive/MyDrive/learn-a-world/results/diagnostics/phase1_5_intervention_eval.json \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.evaluate_multihorizon_interventions import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate 8 & 9 Evaluation")
    parser.add_argument("--data_root", nargs="+", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", default="results/diagnostics/phase1_5_intervention_eval.json")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.data_root and args.manifest is None:
        parser.error("Must provide --data_root or --manifest")

    report = run_evaluation(
        shards_dir=args.data_root,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_json=args.output_json,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        device=args.device,
    )

    print("\n" + "=" * 80)
    print("GATES 8 & 9 ACCEPTANCE SUMMARY")
    print("=" * 80)
    print(f"Gate 8 (Multi-Horizon Rollouts Passed): {report['overall_status']['gate_8_passed']}")
    print(f"Gate 9 (Counterfactual Interventions Passed): {report['overall_status']['gate_9_passed']}")
    print(f"Both Gates Passed: {report['overall_status']['both_passed']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
