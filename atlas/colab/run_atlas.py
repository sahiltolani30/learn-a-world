"""Colab entrypoint for Project ATLAS with dedicated Google Drive integration.

Storage Contract:
- Main Google Drive directory: /content/drive/MyDrive/learn-a-world (shared dataset & baselines)
- Dedicated ATLAS subfolder: /content/drive/MyDrive/learn-a-world/atlas/
    ├── shards/       (live collected shards & cached .latents.npz)
    ├── runs/<id>/    (atomic state.pt, train_log.jsonl, goal_bank, lift, topomap)
    ├── checkpoints/  (exported head models)
    └── results/      (protocol screening JSON reports)

Phases:
  selflearn: iterative collect -> train -> mine loop with wall-clock budget and state checkpointing
  screen: locked 50-episode strict screening (seeds 2050-2099)
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Safe pre-registration of eval.metrics to avoid pulling unused stable_baselines3 from eval/__init__.py
if "eval.metrics" not in sys.modules:
    _metrics_p = _REPO_ROOT / "eval" / "metrics.py"
    if _metrics_p.exists():
        _spec = importlib.util.spec_from_file_location("eval.metrics", str(_metrics_p))
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["eval.metrics"] = _mod
            _spec.loader.exec_module(_mod)

import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.collect import ShardLoader
from atlas.config import (
    REPO_ROOT,
    AtlasConfig,
    resolve_atlas_drive_dir,
    resolve_drive_root,
)
from atlas.screen import screen_agent
from atlas.selflearn import SelfLearnLoop, mine_goals_and_lift


def parse_args():
    parser = argparse.ArgumentParser(description="ATLAS Colab Runner")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["selflearn", "screen"],
        help="Execution phase (selflearn | screen)",
    )
    parser.add_argument(
        "--drive-dir",
        type=str,
        default=None,
        help="Main Google Drive directory (default: /content/drive/MyDrive/learn-a-world)",
    )
    parser.add_argument(
        "--atlas-dir",
        type=str,
        default=None,
        help="Dedicated ATLAS subfolder on Drive (default: <drive-dir>/atlas)",
    )
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations")
    parser.add_argument("--minutes", type=float, default=None, help="Wall-clock time budget in minutes")
    parser.add_argument("--run-id", type=str, default="atlas_v2_scratch", help="Unique run identifier")
    parser.add_argument("--tag", type=str, default="atlas_v2", help="Screening tag")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit backbone checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda | mps | cpu | auto)")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke verification pass")
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Skip load_state, skip re-mine, start with empty banks from iteration 0",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        default=False,
        help="Disable diagnostic audit pack exporter (V2.2 F8)",
    )
    return parser.parse_args()


def resolve_paths(args) -> dict:
    """Configures input and output directories adhering to the Drive layout contract."""
    drive_root = resolve_drive_root(args.drive_dir)

    if drive_root is not None and drive_root.exists():
        print(f"[Drive] Detected main Drive directory: {drive_root}")
        atlas_dir = Path(args.atlas_dir) if args.atlas_dir else (drive_root / "atlas")
        drive_ckpt = drive_root / "checkpoints" / "jepa_crafter_v3_state_readout_v2.pt"
    else:
        print("[Drive] Google Drive not mounted; falling back to local workspace.")
        atlas_dir = REPO_ROOT / "data" / "atlas"
        drive_ckpt = REPO_ROOT / "checkpoints" / "jepa_crafter_v3_phase2.pt"

    shards_dir = atlas_dir / "shards"
    runs_dir = atlas_dir / "runs" / args.run_id
    results_dir = atlas_dir / "results"
    checkpoints_dir = atlas_dir / "checkpoints"

    for d in [atlas_dir, shards_dir, runs_dir, results_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "drive_root": drive_root,
        "atlas_dir": atlas_dir,
        "shards_dir": shards_dir,
        "runs_dir": runs_dir,
        "results_dir": results_dir,
        "checkpoints_dir": checkpoints_dir,
        "drive_ckpt": drive_ckpt,
    }


def sync_results_to_repo(results_dir: Path, drive_root: Optional[Path] = None, runs_dir: Optional[Path] = None):
    """Copies screening JSON results and checkpoints back to local repo and main Drive directories."""
    repo_results = REPO_ROOT / "results"
    repo_results.mkdir(parents=True, exist_ok=True)
    if results_dir.exists() and results_dir != repo_results:
        for f in results_dir.glob("atlas_screen_*.json"):
            shutil.copy2(f, repo_results / f.name)

    # Mirror to main Drive results/ and checkpoints/ matching previous agents (Godfather JEPA, Phase 3)
    if drive_root is not None and drive_root.exists():
        main_results = drive_root / "results"
        main_results.mkdir(parents=True, exist_ok=True)
        if results_dir.exists():
            for f in results_dir.glob("atlas_screen_*.json"):
                shutil.copy2(f, main_results / f.name)
        if runs_dir is not None and runs_dir.exists():
            main_ckpts = drive_root / "checkpoints"
            main_ckpts.mkdir(parents=True, exist_ok=True)
            state_file = runs_dir / "state.pt"
            if state_file.exists():
                shutil.copy2(state_file, main_ckpts / "atlas_state.pt")
            journal_file = runs_dir / "game_journal.jsonl"
            if journal_file.exists():
                shutil.copy2(journal_file, main_results / "game_journal.jsonl")
                shutil.copy2(journal_file, repo_results / "game_journal.jsonl")


def main():
    print("[ATLAS V2] Training from scratch. Prior weight: 0.0. Bootstrap: NONE.")
    args = parse_args()
    paths = resolve_paths(args)

    if args.checkpoint:
        os.environ["ATLAS_CHECKPOINT"] = args.checkpoint
    elif paths["drive_ckpt"].exists():
        os.environ["ATLAS_CHECKPOINT"] = str(paths["drive_ckpt"])

    cfg = AtlasConfig()
    if args.device != "auto":
        cfg.backbone.device = args.device
        cfg.train.device = args.device

    if args.smoke:
        cfg.collect.max_steps = 20
        cfg.collect.episodes_per_iter = 1
        cfg.train.epochs_per_iter = 1
        cfg.screen.num_episodes = 2

    print(f"[ATLAS] Initializing FrozenBackbone on device '{cfg.backbone.device}'...")
    backbone = FrozenBackbone(config=cfg)
    assert backbone.has_learned_delta_head, "delta head not loaded — material gate inert"
    agent = AtlasAgent(backbone=backbone, cfg=cfg)

    target_state = paths["runs_dir"] / "state.pt"
    if target_state.exists():
        assert "archive" not in str(paths["runs_dir"]).lower(), (
            f"Target run dir {paths['runs_dir']} is under an archive path; cannot resume from archive."
        )

    loop = SelfLearnLoop(
        agent=agent,
        run_id=args.run_id,
        run_dir=paths["runs_dir"],
        shards_dir=paths["shards_dir"],
        results_dir=paths["results_dir"],
        cfg=cfg,
    )
    loop.enable_audit = not args.no_audit

    if args.phase == "selflearn":
        print("=" * 70)
        print(f"=== [ATLAS] Starting SELFLEARN Phase ({args.iters} iterations) ===")
        print(f"Saving shards to: {paths['shards_dir']}")
        print(f"Saving state checkpoints to: {paths['runs_dir']}")
        print(f"Saving dev screen reports to: {paths['results_dir']}")
        if args.minutes:
            print(f"Wall-clock budget: {args.minutes} minutes")
        print("=" * 70)

        marker_file = paths["runs_dir"] / "REMINED"
        if args.fresh:
            print("[SelfLearnLoop] Starting fresh (--fresh flag active). Skipping load_state and re-mine.")
            loop.start_iter = 0
            agent.resumed_from = {"run_id": "fresh", "iteration": 0}
            agent.memory_source = "scratch"
            agent.remine = False
        else:
            resume_iter = loop.load_state()
            if resume_iter >= 39 and not marker_file.exists():
                print(f"[SelfLearnLoop] Performing one-time memory repair pass (re-mine) at iteration {resume_iter}...")
                remine_stats = loop.remine_shards()
                print(f"[SelfLearnLoop] Re-mine complete: {remine_stats}")
            agent.resumed_from = {"run_id": args.run_id, "iteration": resume_iter}
            agent.memory_source = "own_play_v2"
            agent.remine = True

        loop.run_iterations(num_iters=args.iters, minutes_budget=args.minutes, fresh=args.fresh)
        sync_results_to_repo(paths["results_dir"], drive_root=paths.get("drive_root"), runs_dir=paths.get("runs_dir"))

    elif args.phase == "screen":
        print("=" * 70)
        print(f"=== [ATLAS] Starting SCREEN Phase (tag: {args.tag}) ===")
        print(f"Loading state from: {paths['runs_dir']}")
        print(f"Writing final protocol JSON to: {paths['results_dir']}")
        print("=" * 70)

        resume_it = loop.load_state()
        agent.resumed_from = {"run_id": args.run_id, "iteration": resume_it}
        agent.memory_source = "own_play_v2"
        agent.remine = True

        report = screen_agent(
            agent=agent,
            seeds=cfg.screen.seeds,
            num_episodes=cfg.screen.num_episodes,
            tag=args.tag,
            out_dir=paths["results_dir"],
        )
        sync_results_to_repo(paths["results_dir"], drive_root=paths.get("drive_root"), runs_dir=paths.get("runs_dir"))

    print("\n[ATLAS] Phase finished successfully.")


if __name__ == "__main__":
    main()
