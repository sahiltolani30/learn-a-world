"""Run one explicitly unlocked MOSAIC phase in Colab or locally."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

from mosaic.src.baseline import BaselineConfig, run_baseline


def _run(command: Sequence[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def _prepare_drive(root: Path) -> None:
    for name in ("checkpoints", "logs", "shards", "results", "manifests", "caches", "notebooks"):
        (root / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MOSAIC phase")
    parser.add_argument("--phase", choices=("p0_contract", "p1_baseline"), required=True)
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/learn-a-world/mosaic",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--algorithm", choices=("random", "ppo", "recurrent_ppo"), default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--eval-seed-start", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--permutation-seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    _prepare_drive(drive_root)
    run_id = args.run_id or f"{args.phase}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result_dir = drive_root / "results" / run_id
    manifest = {
        "format": "mosaic_phase_manifest_v1",
        "phase": args.phase,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.phase == "p0_contract":
        _run([sys.executable, "-m", "pytest", "mosaic/tests", "-q"])
        manifest["status"] = "PASS"
    else:
        values = {
            "algorithm": "ppo",
            "train_steps": 10_000,
            "eval_episodes": 5,
            "eval_seed_start": 900_000,
            "train_seed": 12345,
            "max_episode_steps": 1000,
            "permutation_seed": 12345,
            "device": "auto",
            "num_envs": 1,
        }
        if args.config is not None:
            config_path = Path(args.config)
            loaded = yaml.safe_load(config_path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError("phase config must contain a mapping")
            values.update(loaded)
        for key in tuple(values):
            value = getattr(args, key)
            if value is not None:
                values[key] = value
        config = BaselineConfig(
            **values,
            output_dir=str(result_dir),
        )
        result = run_baseline(config)
        manifest["status"] = "PASS"
        manifest["result"] = result

    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
