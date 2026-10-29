"""Colab entrypoint for Project ATLAS with dedicated Google Drive integration.

Storage Contract:
- Main Google Drive directory: /content/drive/MyDrive/learn-a-world (shared dataset & baselines)
- Input data read from: /content/drive/MyDrive/learn-a-world/data/sequences/*.npz
- Backbone checkpoint read from: /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_state_readout_v2.pt
- Dedicated ATLAS subfolder: /content/drive/MyDrive/learn-a-world/atlas/
    ├── shards/       (live collected shards & cached .latents.npz)
    ├── runs/<id>/    (atomic state.pt, train_log.jsonl, goal_bank, lift, topomap)
    ├── checkpoints/  (exported head models)
    └── results/      (protocol screening JSON reports)

Phases:
  bootstrap: offline pass over Drive sequence shards to initialize heads, GoalBank & PredecessorLift
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
        choices=["bootstrap", "selflearn", "screen"],
        help="Execution phase (bootstrap | selflearn | screen)",
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
    parser.add_argument("--run-id", type=str, default="atlas_v1", help="Unique run identifier")
    parser.add_argument("--tag", type=str, default="atlas_v1", help="Screening tag")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit backbone checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda | mps | cpu | auto)")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke verification pass")
    return parser.parse_args()


def resolve_paths(args) -> dict:
    """Configures input and output directories adhering to the Drive layout contract."""
    drive_root = resolve_drive_root(args.drive_dir)

    if drive_root is not None and drive_root.exists():
        print(f"[Drive] Detected main Drive directory: {drive_root}")
        atlas_dir = Path(args.atlas_dir) if args.atlas_dir else (drive_root / "atlas")
        bootstrap_seq_dir = drive_root / "data" / "sequences"
        drive_ckpt = drive_root / "checkpoints" / "jepa_crafter_v3_state_readout_v2.pt"
    else:
        print("[Drive] Google Drive not mounted; falling back to local workspace.")
        atlas_dir = REPO_ROOT / "data" / "atlas"
        bootstrap_seq_dir = REPO_ROOT / "data" / "sequences"
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
        "bootstrap_seq_dir": bootstrap_seq_dir,
        "drive_ckpt": drive_ckpt,
    }


def sync_results_to_repo(results_dir: Path):
    """Copies screening JSON results back to the local repo results/ folder."""
    repo_results = REPO_ROOT / "results"
    repo_results.mkdir(parents=True, exist_ok=True)
    if results_dir.exists() and results_dir != repo_results:
        for f in results_dir.glob("atlas_screen_*.json"):
            shutil.copy2(f, repo_results / f.name)


def run_bootstrap_phase(loop: SelfLearnLoop, paths: dict, iters: int):
    """Offline bootstrap on Drive sequence shards (e.g. data/sequences/*.npz)."""
    print("=" * 70)
    print("=== [ATLAS] Starting BOOTSTRAP Phase ===")
    print(f"Reading sequence shards from: {paths['bootstrap_seq_dir']}")
    print(f"Caching latents to: {paths['shards_dir']}")
    print("=" * 70)

    shard_files: List[Path] = []
    if paths["bootstrap_seq_dir"].exists():
        shard_files.extend(sorted(paths["bootstrap_seq_dir"].glob("*.npz")))

    # Fallback to local repo sequences if drive empty
    local_seq = REPO_ROOT / "data" / "sequences"
    if not shard_files and local_seq.exists():
        shard_files.extend(sorted(local_seq.glob("*.npz")))

    if not shard_files:
        print("[Bootstrap] No sequence shards found. Skipping bootstrap.")
        loop.save_state(0)
        return

    # Filter out latents sidecars
    shard_files = [f for f in shard_files if not f.name.endswith(".latents.npz")]
    print(f"[Bootstrap] Found {len(shard_files)} bootstrap shards.")

    num_shards_to_use = min(len(shard_files), max(1, iters))
    selected_shards = shard_files[:num_shards_to_use]

    all_episodes = []
    for idx, s_path in enumerate(selected_shards):
        print(f"[Bootstrap {idx+1}/{len(selected_shards)}] Caching latents for {s_path.name}...")
        lat_path = loop.trainer.cache_latents(s_path, cache_dir=paths["shards_dir"])

        # Load metadata/actions/targets without loading pixel observations
        shard_data = ShardLoader.load_shard(s_path, load_obs=False)
        with np.load(lat_path) as lat_data:
            z_full = np.array(lat_data["z"])
            h_full = np.array(lat_data["h"])
            res_full = np.array(lat_data["target_residuals"])

        ep_lengths = shard_data["episode_lengths"]

        start = 0
        for ep_len in ep_lengths:
            end = start + ep_len
            all_episodes.append({
                "z": z_full[start:end].copy(),
                "h": h_full[start:end].copy(),
                "target_residuals": res_full[start:end].copy(),
                "actions": shard_data["actions"][start:end].copy(),
                "rewards": shard_data["rewards"][start:end].copy(),
                "dones": shard_data["dones"][start:end].copy(),
                "events": shard_data["events"][start:end].copy(),
                "mask_before": shard_data["mask_before"][start:end].copy(),
                "inventory": shard_data["inventory"][start:end].copy(),
            })
            start = end

        del z_full, h_full, res_full, shard_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform != "darwin":
                rss /= 1024  # Linux: ru_maxrss is in KB
            else:
                rss /= (1024 * 1024)  # macOS: bytes
            print(f"  -> Ingested cleanly (RAM max RSS: {rss:.1f} MB)")
        except Exception:
            pass

    print(f"[Bootstrap] Ingested {len(all_episodes)} episodes. Training initial heads...")
    val_count = max(1, int(0.1 * len(all_episodes)))
    train_eps = all_episodes[:-val_count]
    val_eps = all_episodes[-val_count:]

    loop.trainer.train_epoch(train_eps, val_eps, iter_idx=0)

    print("[Bootstrap] Mining GoalBank and PredecessorLift from bootstrap trajectories...")
    mine_goals_and_lift(loop.agent, all_episodes, iter_idx=0)

    loop.save_state(0)
    print(
        f"[Bootstrap Complete] GoalBank populated with {len(loop.agent.goals.entries)} goals. "
        f"State saved to {loop.state_file}"
    )


def main():
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
    agent = AtlasAgent(backbone=backbone, cfg=cfg)

    loop = SelfLearnLoop(
        agent=agent,
        run_id=args.run_id,
        run_dir=paths["runs_dir"],
        shards_dir=paths["shards_dir"],
        results_dir=paths["results_dir"],
        cfg=cfg,
    )

    if args.phase == "bootstrap":
        run_bootstrap_phase(loop, paths, iters=args.iters)

    elif args.phase == "selflearn":
        print("=" * 70)
        print(f"=== [ATLAS] Starting SELFLEARN Phase ({args.iters} iterations) ===")
        print(f"Saving shards to: {paths['shards_dir']}")
        print(f"Saving state checkpoints to: {paths['runs_dir']}")
        print(f"Saving dev screen reports to: {paths['results_dir']}")
        if args.minutes:
            print(f"Wall-clock budget: {args.minutes} minutes")
        print("=" * 70)

        loop.run_iterations(num_iters=args.iters, minutes_budget=args.minutes)
        sync_results_to_repo(paths["results_dir"])

    elif args.phase == "screen":
        print("=" * 70)
        print(f"=== [ATLAS] Starting SCREEN Phase (tag: {args.tag}) ===")
        print(f"Loading state from: {paths['runs_dir']}")
        print(f"Writing final protocol JSON to: {paths['results_dir']}")
        print("=" * 70)

        loop.load_state()
        report = screen_agent(
            agent=agent,
            seeds=cfg.screen.seeds,
            num_episodes=cfg.screen.num_episodes,
            tag=args.tag,
            out_dir=paths["results_dir"],
        )
        sync_results_to_repo(paths["results_dir"])

    print("\n[ATLAS] Phase finished successfully.")


if __name__ == "__main__":
    main()
