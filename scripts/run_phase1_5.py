#!/usr/bin/env python3
"""Prepare a deduplicated Phase 1.5 manifest and optionally train Phase 1.

Example on Colab:

    python3 scripts/run_phase1_5.py \
      --data_root /content/drive/MyDrive/learn-a-world/data \
      --manifest /content/drive/MyDrive/learn-a-world/results/phase1_5_manifest.json \
      --checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_t2_final.pt \
      --output_checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_phase1_5.pt \
      --output_json /content/drive/MyDrive/learn-a-world/results/phase1_5_training.json \
      --steps 1000 --device cuda

Use ``--audit_only`` to create the manifest without starting training.
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

from training.phase1_5_data import discover_active_shards
from training.run_phase1_transition_learning import train_phase1_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the Phase 1.5 combined dataset")
    parser.add_argument("--data_root", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_checkpoint", default="checkpoints/jepa_crafter_v3_phase1_5.pt")
    parser.add_argument("--output_json", default="results/phase1_5_training.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--capacity_transitions", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()

    manifest = discover_active_shards(args.data_root, args.manifest)
    print(json.dumps({"phase": "phase1.5", "totals": manifest["totals"]}, indent=2))
    if args.audit_only:
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --audit_only is used")

    result = train_phase1_transitions(
        shards_dir=args.data_root,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        horizons=tuple(args.horizons),
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        capacity_transitions=args.capacity_transitions,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
