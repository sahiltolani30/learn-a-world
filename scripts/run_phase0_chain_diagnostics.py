#!/usr/bin/env python3
"""CLI for Phase 0 causal-chain instrumentation and ablations."""

import argparse
import json
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.phase0_trace import run_phase0_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/jepa_crafter_v3_phase2.pt")
    parser.add_argument("--memory_bank", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output_json", default="results/phase0_chain_diagnostics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_privileged", action="store_true")
    args = parser.parse_args()
    result = run_phase0_diagnostics(
        checkpoint_path=args.checkpoint,
        memory_bank_path=args.memory_bank,
        num_episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        output_json=args.output_json,
        device=args.device,
        include_privileged=args.include_privileged,
    )
    print(json.dumps({k: result[k] for k in ("phase", "status", "results")}, indent=2))


if __name__ == "__main__":
    main()
