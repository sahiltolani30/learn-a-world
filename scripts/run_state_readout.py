#!/usr/bin/env python3
"""CLI runner for Section 6 & Gate 5: StateReadoutHead training and validation.

Supports execution both on Google Colab (with GPU and Drive data) and locally (CPU).

Example usage on Colab:
    python3 scripts/run_state_readout.py \
      --data_root /content/drive/MyDrive/learn-a-world/data \
      --manifest /content/drive/MyDrive/learn-a-world/results/phase1_5_manifest.json \
      --checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_phase1_5_frozen.pt \
      --output_checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_state_readout.pt \
      --output_json /content/drive/MyDrive/learn-a-world/results/diagnostics/state_readout_training.json \
      --steps 1000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase1_5_data import discover_active_shards, load_manifest
from training.run_state_readout_learning import train_state_readout


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StateReadoutHead (Section 6 / Gate 5)")
    parser.add_argument("--data_root", nargs="+", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_checkpoint", default="checkpoints/jepa_crafter_v3_state_readout.pt")
    parser.add_argument("--output_json", default="results/diagnostics/state_readout_training.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--labeled_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--capacity_transitions", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.data_root and args.manifest is None:
        parser.error("Must provide --data_root or --manifest")

    manifest_path = None
    if args.manifest:
        manifest_p = Path(args.manifest)
        if manifest_p.exists():
            manifest_path = str(manifest_p)
            print(f"[State Readout] Using existing manifest: {manifest_p}")
        elif args.data_root:
            print(f"[State Readout] Generating manifest at: {manifest_p}")
            manifest = discover_active_shards(args.data_root, args.manifest)
            manifest_path = str(manifest_p)
    elif args.data_root:
        manifest_path = None

    result = train_state_readout(
        shards_dir=args.data_root if args.data_root else [],
        manifest_path=manifest_path,
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        labeled_ratio=args.labeled_ratio,
        lr=args.lr,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        capacity_transitions=args.capacity_transitions,
        device=args.device,
    )

    print("\n" + "=" * 80)
    print("GATE 5 ACCEPTANCE SUMMARY")
    print("=" * 80)
    fm = result["final_metrics"]
    print(f"Readout MSE:     {fm.get('state_readout_mse')}")
    print(f"Zero-State MSE:  {fm.get('zero_state_mse')}")
    print(f"Persistence MSE: {fm.get('persistence_mse')}")
    print(f"Best Baseline:   {fm.get('baseline_best_mse')}")
    print(f"Gate 5 Passed (Readout MSE < Baseline): {result['gate_5_passed']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
