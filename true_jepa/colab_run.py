"""Minimal Colab/T4 entry point for the clean pixel-only self-play run."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from true_jepa.config import JEPAConfig, TrainConfig, as_dict
from true_jepa.env import PixelCrafter
from true_jepa.model import TrueJEPA
from true_jepa.self_play import CollectionStats, SelfPlayCoordinator
from true_jepa.drive import TrueJEPADrive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the clean true-JEPA agent")
    parser.add_argument("--drive-root", default=None, help="learn-a-world root; auto-detected on Colab")
    parser.add_argument("--run-dir", default=None, help="local override for the model root")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--collection-policy",
        choices=("mixed", "random"),
        default="mixed",
        help="data collection policy; random is used for a clean Phase 1 dataset",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="collect and save replay without optimizer updates",
    )
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="train the already-saved replay without collecting new episodes",
    )
    parser.add_argument(
        "--world-only",
        action="store_true",
        help="train JEPA/reward dynamics while leaving actor/value policy unchanged",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    storage = TrueJEPADrive(root=args.run_dir or args.drive_root)
    seeds = itertools.count(int(args.seed))

    def factory() -> PixelCrafter:
        return PixelCrafter(next(seeds), max_steps=args.max_steps)

    probe = factory()
    try:
        num_actions = probe.num_actions
    finally:
        probe.close()

    checkpoint = storage.load_checkpoint()
    if checkpoint is not None:
        model_cfg = JEPAConfig(**checkpoint["config"])
        train_cfg = TrainConfig(**checkpoint["train_config"])
        model = TrueJEPA(model_cfg, num_actions=int(checkpoint["num_actions"]))
        replay_path = storage.shards / "replay.npz"
        runner = SelfPlayCoordinator.load(
            str(storage.checkpoints / "state.pt"),
            factory,
            model,
            train_cfg=train_cfg,
            device=args.device,
            seed=args.seed,
            num_envs=args.num_envs,
            collection_policy=args.collection_policy,
            replay_path=str(replay_path) if replay_path.exists() else None,
        )
        print(f"Resuming true_jepa at update {runner.update}, frames {runner.env_frames}")
    else:
        model = TrueJEPA(JEPAConfig(), num_actions=num_actions)
        train_cfg = TrainConfig()
        runner = SelfPlayCoordinator(
            factory,
            model,
            train_cfg=train_cfg,
            device=args.device,
            seed=args.seed,
            num_envs=args.num_envs,
            collection_policy=args.collection_policy,
        )
    collection = CollectionStats() if args.train_only else runner.collect(args.episodes)
    metrics = (
        []
        if args.collect_only
        else runner.train(args.updates, train_policy=not args.world_only)
    )
    storage.save_checkpoint({
        "format": "true_jepa_run_v1",
        "config": dict(runner.model.cfg.__dict__),
        "train_config": dict(train_cfg.__dict__),
        "num_actions": runner.model.num_actions,
        "model": runner.model.state_dict(),
        "target_encoder": runner.model.target_encoder.state_dict(),
        "world_optimizer": runner.world_optimizer.state_dict(),
        "actor_optimizer": runner.actor_optimizer.state_dict(),
        "value_optimizer": runner.value_optimizer.state_dict(),
        "update": runner.update,
        "env_frames": runner.env_frames,
        "relations": runner.relations.state_dict(),
    })
    storage.save_replay(runner.replay)
    log_path = storage.logs / "train_metrics.jsonl"
    with log_path.open("a") as log_file:
        for metric in metrics:
            log_file.write(
                json.dumps(
                    {"update": runner.update, "env_frames": runner.env_frames, **metric},
                    sort_keys=True,
                )
                + "\n"
            )
    storage.save_json(
        {
            "jepa": as_dict(model.cfg),
            "train": as_dict(train_cfg),
            "num_actions": num_actions,
            "device": args.device,
            "collection": collection.__dict__,
            "last_metrics": metrics[-1] if metrics else {},
        },
        "config.json",
    )
    print(json.dumps({"collection": collection.__dict__, "last_metrics": metrics[-1] if metrics else {}}, indent=2))


if __name__ == "__main__":
    main()
