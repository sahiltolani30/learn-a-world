"""Zephyr Stage-1 Colab runner: joint WM + Actor-Critic training loop.

Usage (Colab, T4/L4/A100):
    %cd /content/ML
    python3 -m zephyr.colab.run_s1 \
        --drive-dir /content/drive/MyDrive/learn-a-world/zephyr \
        --total-frames 2000000 \
        --screen-every 500000 \
        --device cuda

Resume-safe: checkpoints and replay shards are saved to --drive-dir.
If interrupted by Colab timeouts, re-running the exact same command automatically
detects the saved state and resumes training without lost progress.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

# Allow running as a script from anywhere: repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from zephyr.colab.run_s0 import smoke_config  # noqa: E402
from zephyr.config import ZephyrConfig  # noqa: E402
from zephyr.train_loop import ZephyrTrainer  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-dir", "--drive_dir", default=None,
                        help="root Google Drive directory for persistent storage")
    parser.add_argument("--run-dir", "--run_dir", default=None,
                        help="local/scratch run directory (defaults to drive-dir if provided)")
    parser.add_argument("--total-frames", "--total_frames", type=int, default=2_000_000,
                        help="total environment frames target for Stage 1")
    parser.add_argument("--screen-every", "--screen_every", type=int, default=500_000,
                        help="frequency of dev screening in frames")
    parser.add_argument("--frames-per-cycle", "--frames_per_cycle", type=int, default=5_000,
                        help="frames collected per outer loop cycle")
    parser.add_argument("--train-steps-per-cycle", "--train_steps_per_cycle", type=int, default=2_500,
                        help="training optimizer steps per outer loop cycle")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-wm-ckpt", "--init_wm_ckpt", default=None,
                        help="optional Stage-0 WM checkpoint to initialize weights from")
    parser.add_argument("--init-heads-ckpt", "--init_heads_ckpt", default=None,
                        help="optional Stage-0 heads checkpoint to initialize weights from")
    parser.add_argument("--smoke", action="store_true",
                        help="fast sanity pass on CPU")
    parser.add_argument("--in-process", "--in_process", action="store_true",
                        help="run envs without multiprocessing (macOS/tests)")
    parser.add_argument("--log-every", "--log_every", type=int, default=200)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if args.drive_dir:
            run_dir = os.path.join(args.drive_dir, "runs", "s1")
        else:
            run_dir = "zephyr_runs/s1"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)

    cfg = ZephyrConfig()
    if args.smoke:
        cfg = smoke_config(cfg)
        args.frames_per_cycle = min(args.frames_per_cycle, 50)
        args.train_steps_per_cycle = min(args.train_steps_per_cycle, 5)
        args.total_frames = min(args.total_frames, 50)

    from dataclasses import replace
    if args.num_envs is not None:
        cfg = replace(cfg, data=replace(cfg.data, num_envs=args.num_envs))
    elif args.smoke:
        cfg = replace(cfg, data=replace(cfg.data, num_envs=1))


    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    state_path = os.path.join(run_dir, "state.pt")
    if os.path.exists(state_path):
        print(f"[zephyr-s1] found existing state at {state_path}; resuming...")
        trainer = ZephyrTrainer.load_run(state_path, device=device)
        print(f"[zephyr-s1] resumed: {trainer.env_frames} frames, "
              f"{trainer.train_steps} steps, {trainer.total_episodes} episodes")
    else:
        # Check if Stage-0 checkpoints are provided or available in default locations
        load_ckpts = None
        init_wm = args.init_wm_ckpt
        init_heads = args.init_heads_ckpt
        if init_wm is None and args.drive_dir:
            cand_wm = os.path.join(args.drive_dir, "checkpoints", "zephyr_crafter_v1_stage0_wm.pt")
            cand_heads = os.path.join(args.drive_dir, "checkpoints", "zephyr_crafter_v1_stage0_heads.pt")
            if os.path.exists(cand_wm) and os.path.exists(cand_heads):
                init_wm, init_heads = cand_wm, cand_heads

        if init_wm and init_heads and os.path.exists(init_wm) and os.path.exists(init_heads):
            print(f"[zephyr-s1] initializing from Stage-0 checkpoints: {init_wm} / {init_heads}")
            load_ckpts = (init_wm, init_heads)

        trainer = ZephyrTrainer(
            cfg=cfg,
            device=device,
            run_dir=run_dir,
            load_checkpoints=load_ckpts,
            env_in_process=args.in_process,
        )

    print(f"[zephyr-s1] starting training loop on {device}: "
          f"target={args.total_frames} frames, current={trainer.env_frames} frames")

    last_screen_frame = trainer.env_frames
    while trainer.env_frames < args.total_frames:
        frames_to_collect = min(args.frames_per_cycle, args.total_frames - trainer.env_frames)
        print(f"[zephyr-s1] collecting {frames_to_collect} frames (policy: actor)...")
        c_stats = trainer.collect(frames_to_collect, source="actor")
        if trainer.replay.episodes < 2 and trainer.collector is not None:
            trainer.collector.flush_open_episodes()
        print(f"[zephyr-s1] buffer: {len(trainer.replay)} frames, {trainer.replay.episodes} episodes, "
              f"{c_stats['achievement_events']} new achievement events")

        if len(trainer.replay) >= cfg.data.batch_size:
            print(f"[zephyr-s1] training {args.train_steps_per_cycle} steps (WM + AC)...")
            train_metrics = trainer.train(
                args.train_steps_per_cycle,
                log_every=args.log_every,
                on_log=lambda s, m: print(
                    f"[zephyr-s1] step {s}: wm_loss={m.get('loss', 0.0):.4f} "
                    f"actor_loss={m.get('actor_loss', 0.0):.4f} "
                    f"critic_loss={m.get('critic_loss', 0.0):.4f} "
                    f"ret_mean={m.get('lambda_return_mean', 0.0):.3f}"
                ),
            )
        else:
            print(f"[zephyr-s1] replay size {len(trainer.replay)} < batch_size {cfg.data.batch_size}; skipping train step")

        # Save run state
        trainer.save_run(state_path)
        wm_best = os.path.join(run_dir, "ckpt", "zephyr_wm_latest.pt")
        heads_best = os.path.join(run_dir, "ckpt", "zephyr_heads_latest.pt")
        trainer.save_checkpoints(wm_best, heads_best)

        # Periodic screening
        if trainer.env_frames - last_screen_frame >= args.screen_every:
            last_screen_frame = trainer.env_frames
            screen_report_path = os.path.join(
                run_dir, "reports", f"dev_screen_frame_{trainer.env_frames:07d}.json"
            )
            print(f"[zephyr-s1] running dev screen at frame {trainer.env_frames}...")
            report = trainer.run_dev_screen(screen_report_path, agent_mode="actor")
            print(f"[zephyr-s1] dev screen complete: score={report['crafter_score']:.2f}% "
                  f"mean_surv={report['mean_survival_length']:.1f}")

    print(f"[zephyr-s1] target {args.total_frames} frames reached! Finished.")
    print(f"[zephyr-s1] final checkpoints saved in {os.path.join(run_dir, 'ckpt')}")


if __name__ == "__main__":
    main()
