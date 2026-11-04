"""Main execution entrypoint for local and Google Colab execution."""
import os
import sys
import argparse
from pathlib import Path

# Ensure package root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.utils.config import load_config, save_config
from src.training.train_agent import run_training


def main():
    parser = argparse.ArgumentParser(description="AEON JEPA-Crafter Training & Evaluation Runner")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "configs" / "crafter_t4.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default=None,
        help="Google Drive master folder (e.g. /content/drive/MyDrive/learn-a-world/aeon)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints and artifacts",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum environment steps (e.g. 10000 for smoke test, 1000000 for full run)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel games/environments to step in lockstep (e.g. 8)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.num_envs is not None:
        config.collection.num_env_workers = int(args.num_envs)

    # If Drive root is specified, route outputs directly to Drive
    if args.drive_root:
        drive_path = Path(args.drive_root)
        if str(drive_path).startswith("/content/drive") and not os.path.exists("/content/drive/MyDrive"):
            raise RuntimeError(
                "CRITICAL: /content/drive/MyDrive does not exist! Google Drive is NOT mounted. "
                "Please run Cell 2 ('drive.mount(\"/content/drive\")') before launching training, "
                "otherwise all checkpoints will be written to temporary VM disk and lost upon disconnect!"
            )
        for sub in ["runs", "checkpoints", "artifacts", "results"]:
            (drive_path / sub).mkdir(parents=True, exist_ok=True)
        config.project.output_dir = str(drive_path / "runs")
        print(f"[Drive Storage] Verified Google Drive output directory active: {config.project.output_dir}")
    elif args.output_dir:
        config.project.output_dir = args.output_dir

    if args.device:
        config.project.device = args.device

    print(f"[AEON Runner] Loaded config from {args.config}")
    print(f"[AEON Runner] Target environment steps: {args.max_steps or config.environment.max_environment_steps}")

    results = run_training(config=config, max_steps=args.max_steps)
    print("\n=======================================================")
    print(f"AEON run completed! Final report: {results['final_report_path']}")
    print(f"Crafter Score: {results['achievement_summary']['crafter_score']:.2f}%")
    print("=======================================================")


if __name__ == "__main__":
    main()
