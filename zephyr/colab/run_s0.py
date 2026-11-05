"""Zephyr Stage-0 Colab runner: collect -> train world model -> gate report.

Usage (Colab, T4):
    %cd /content/learn-a-world
    !python3 zephyr/colab/run_s0.py \
        --data-dir  /content/drive/MyDrive/learn-a-world/zephyr/data \
        --ckpt-dir  /content/drive/MyDrive/learn-a-world/zephyr/checkpoints \
        --report    /content/drive/MyDrive/learn-a-world/zephyr/reports/zephyr_stage0_report.json \
        --frames 250000 --steps 50000 --num-envs 4 --device cuda

Resume-safe: finished shards and checkpoints are reused; only the missing
frame/step budget is executed. --smoke runs a minutes-long end-to-end pass.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from dataclasses import replace
from typing import Dict

# Allow running as a script from anywhere: repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from zephyr.buffer import ZephyrReplay  # noqa: E402
from zephyr.config import ZephyrConfig  # noqa: E402
from zephyr.heads import ZephyrHeads  # noqa: E402
from zephyr.train_loop import FrameCollector, RandomPolicy  # noqa: E402
from zephyr.train_wm import evaluate_gates, train_world_model  # noqa: E402
from zephyr.vec_env import ZephyrVecEnv  # noqa: E402
from zephyr.wm import ZephyrWM  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _load_shards(replay: ZephyrReplay, data_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(data_dir, "zephyr_shard_*.npz")))
    for path in paths:
        shard = ZephyrReplay.load_shard(path)
        for rec in shard.records:
            replay.add_episode(rec)
    return len(paths)


def smoke_config(cfg: ZephyrConfig) -> ZephyrConfig:
    """Minutes-long end-to-end pass on CPU (laptop/CI sanity, not a benchmark)."""
    return replace(
        cfg,
        world_model=replace(
            cfg.world_model,
            latent_dim=64, belief_dim=64, action_embed_dim=16,
            predictor_hidden=128, ema_ramp_steps=100,
        ),
        encoder=replace(cfg.encoder, channels=(8, 16, 16, 32), latent_dim=64),
        heads=replace(
            cfg.heads, reward_bins=33, critic_bins=33,
            reward_symlog_range=10.0, critic_symlog_range=10.0,
        ),
        ac=replace(cfg.ac, imagination_horizon=5, imag_start_stride=2),
        explore=replace(cfg.explore, ensemble_k=2, ensemble_hidden=64),
        data=replace(
            cfg.data, batch_size=4, seq_len=8, num_envs=1,
            replay_capacity_frames=20_000,
            episode_max_steps=20,
        ),
        optim=replace(cfg.optim, fp16=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", default=None)
    parser.add_argument("--ckpt-dir", "--ckpt_dir", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--drive-dir", "--drive_dir", default=None,
                        help="convenience root: sets data-dir, ckpt-dir, and report automatically")
    parser.add_argument("--frames", "--collect_frames", type=int, default=250_000,
                        help="target collected frames (env steps)")
    parser.add_argument("--steps", "--train_steps", type=int, default=50_000,
                        help="world-model optimizer steps for this session")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None,
                        help="override data batch size")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--collect-only", "--collect_only", action="store_true")
    parser.add_argument("--eval-only", "--eval_only", action="store_true")
    parser.add_argument("--fresh-training", "--fresh_training", action="store_true",
                        help="ignore existing checkpoints and retrain from scratch")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--in-process", "--in_process", action="store_true",
                        help="run envs without multiprocessing (macOS/tests)")
    parser.add_argument("--log-every", "--log_every", type=int, default=500)
    args = parser.parse_args()

    if args.drive_dir:
        if args.data_dir is None:
            args.data_dir = os.path.join(args.drive_dir, "data")
        if args.ckpt_dir is None:
            args.ckpt_dir = os.path.join(args.drive_dir, "checkpoints")
        if args.report is None:
            args.report = os.path.join(args.drive_dir, "reports", "zephyr_stage0_report.json")

    if not args.data_dir or not args.ckpt_dir or not args.report:
        parser.error("Either --drive-dir OR all of (--data-dir, --ckpt-dir, --report) must be provided")

    cfg = ZephyrConfig()
    if args.smoke:
        cfg = smoke_config(cfg)
    num_envs = args.num_envs if args.num_envs is not None else (
        cfg.data.num_envs if not args.smoke else 1
    )
    batch_size = args.batch_size if args.batch_size is not None else cfg.data.batch_size
    cfg = replace(cfg, data=replace(cfg.data, num_envs=num_envs, batch_size=batch_size))
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    for d in (args.data_dir, args.ckpt_dir, os.path.dirname(os.path.abspath(args.report))):
        os.makedirs(d, exist_ok=True)

    # ---- data ----
    replay = ZephyrReplay(capacity_frames=cfg.data.replay_capacity_frames)
    n_shards = _load_shards(replay, args.data_dir)
    print(f"[zephyr-s0] loaded {n_shards} shards -> {len(replay)} frames, "
          f"{replay.episodes} episodes, labeled={replay.labeled_fraction():.3f}")

    if len(replay) < args.frames:
        env = ZephyrVecEnv(
            num_envs=cfg.data.num_envs,
            base_seed=cfg.data.train_seed_base,
            max_steps=cfg.data.episode_max_steps,
            in_process=args.in_process,
        )
        collector = FrameCollector(
            env, RandomPolicy(cfg.data.num_envs, seed=cfg.data.split_seed),
            replay, source="random", shard_dir=args.data_dir,
            shard_every_episodes=20 if not args.smoke else 2,
        )
        print(f"[zephyr-s0] collecting {args.frames - len(replay)} frames "
              f"with {cfg.data.num_envs} envs (random policy)...")
        stats = collector.collect(args.frames - len(replay))
        if replay.episodes < 2:
            collector.flush_open_episodes()
        env.close()
        print(f"[zephyr-s0] collected {stats['frames']} frames, "
              f"{replay.episodes} episodes total, "
              f"{stats['achievement_events']} achievement events")
    else:
        stats = {"frames": 0, "episodes": replay.episodes,
                 "achievement_events": sum(replay.achievement_counts().values())}

    if args.collect_only:
        print("[zephyr-s0] --collect-only: done")
        return

    # ---- split ----
    train_replay, val_replay = replay.split(
        cfg.data.val_ratio, seed=cfg.data.split_seed
    )
    print(f"[zephyr-s0] split: train {train_replay.episodes} eps / "
          f"{len(train_replay)} frames; val {val_replay.episodes} eps / "
          f"{len(val_replay)} frames")

    wm_path = os.path.join(args.ckpt_dir, "zephyr_crafter_v1_stage0_wm.pt")
    heads_path = os.path.join(args.ckpt_dir, "zephyr_crafter_v1_stage0_heads.pt")

    if not args.eval_only:
        resume_from = 0
        if args.fresh_training or not (
            os.path.exists(wm_path) and os.path.exists(heads_path)
        ):
            wm = ZephyrWM(cfg).to(device)
            heads = ZephyrHeads(cfg).to(device)
        else:
            wm, wm_meta = ZephyrWM.load_checkpoint(wm_path, device=device)
            heads, _ = ZephyrHeads.load_checkpoint(heads_path, device=device)
            if wm.cfg.to_dict() != cfg.to_dict():
                raise RuntimeError(
                    "checkpoint config differs from this run's config; "
                    "use --fresh-training to restart training deliberately"
                )
            resume_from = int(wm_meta.get("step") or 0)
            print(f"[zephyr-s0] resuming training from step {resume_from}")
        remaining = max(0, args.steps - resume_from)
        if remaining == 0:
            print(f"[zephyr-s0] target {args.steps} steps already reached; "
                  "skipping to gate evaluation")
            final = {}
        else:
            print(f"[zephyr-s0] training world model: {remaining} steps "
                  f"({resume_from} -> {args.steps}) on {device}...")
            final = train_world_model(
                wm, heads, train_replay, cfg, device=device, steps=remaining,
                log_every=args.log_every, ckpt_path=wm_path, heads_ckpt_path=heads_path,
                ckpt_every=max(1000, remaining // 10), initial_step=resume_from,
                on_log=lambda s, m: print(
                    f"[zephyr-s0] step {s}: loss={m['loss']:.4f} "
                    f"jepa={m['loss_jepa_mse']:.4f} readout={m['loss_readout']:.4f} "
                    f"effect={m['loss_effect']:.4f} "
                    f"latent_std={m['latent_std_online']:.3f}"
                ),
            )
    else:
        wm, _ = ZephyrWM.load_checkpoint(wm_path, device=device)
        heads, _ = ZephyrHeads.load_checkpoint(heads_path, device=device)
        final = {}

    # ---- gates ----
    print("[zephyr-s0] evaluating gates ZG1/ZG2/ZG3/ZG5 on held-out episodes...")
    gates = evaluate_gates(wm, heads, val_replay, cfg, device=device)

    import json

    report: Dict = {
        "stage": "s0",
        "protocol_id": cfg.protocol_id,
        "created_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "git_commit": _git_commit(),
        "config": cfg.to_dict(),
        "data": {
            "shards": n_shards,
            "frames": len(replay),
            "episodes": replay.episodes,
            "labeled_fraction": replay.labeled_fraction(),
            "achievement_counts": replay.achievement_counts(),
            "collect_stats": stats,
        },
        "train_steps": args.steps,
        "final_train_metrics": final,
        "gates": gates,
        "gate_labels": {
            "zg1": "JEPA health / no collapse",
            "zg2": "free-run beats persistence at horizons 1/5/10/15",
            "zg3": "readout beats persistence readout by >=30%",
            "zg5": "termination head recall>=0.5, precision>=0.5, AUC>=0.9",
        },
    }
    tmp = args.report + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    os.replace(tmp, args.report)

    print("[zephyr-s0] GATE REPORT")
    print(f"  ZG1 collapse/latent-std : {gates['zg1']['latent_std_online']:.3f} "
          f"-> {'PASS' if gates['zg1']['passed'] else 'FAIL'}")
    for h, info in gates["zg2"]["horizons"].items():
        if info.get("available"):
            print(f"  ZG2 horizon {h:>2}       : freerun {info['freerun_mse']:.5f} "
                  f"vs persist {info['persistence_mse']:.5f} "
                  f"({info['relative_improvement_pct']:+.1f}%)")
    print(f"  ZG2 overall             : {'PASS' if gates['zg2']['passed'] else 'FAIL'}")
    print(f"  ZG3 readout vs persist  : {gates['zg3']['readout_mse']:.5f} vs "
          f"{gates['zg3']['persistence_mse']:.5f} "
          f"({gates['zg3']['relative_improvement_pct']:+.1f}%) -> "
          f"{'PASS' if gates['zg3']['passed'] else 'FAIL'}")
    if gates["zg5"].get("available"):
        print(f"  ZG5 termination         : recall {gates['zg5']['recall_at_0p5']:.3f} "
              f"precision {gates['zg5']['precision_at_0p5']:.3f} "
              f"AUC {gates['zg5']['auc']:.3f} -> "
              f"{'PASS' if gates['zg5']['passed'] else 'FAIL'}")
    else:
        print("  ZG5 termination         : no positive done events in val split -> FAIL")
    print(f"  report: {args.report}")
    print(f"  checkpoints: {wm_path} / {heads_path}")


if __name__ == "__main__":
    main()
