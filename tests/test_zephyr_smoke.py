"""End-to-end Zephyr smoke: collect -> train -> gates -> resume -> screen.

Tiny config, in-process env, CPU. Exercises the full Stage-0 pipeline and the
Mode-A/Mode-B agents on real Crafter episodes. Runtime: a few minutes.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from zephyr_test_utils import tiny_cfg
from zephyr.buffer import ZephyrReplay
from zephyr.heads import ZephyrHeads
from zephyr.train_loop import FrameCollector, RandomPolicy, ZephyrTrainer
from zephyr.train_wm import evaluate_gates, train_world_model
from zephyr.vec_env import ZephyrVecEnv
from zephyr.wm import ZephyrWM

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _collect(tmp_path, cfg, frames):
    replay = ZephyrReplay(capacity_frames=cfg.data.replay_capacity_frames)
    env = ZephyrVecEnv(
        num_envs=cfg.data.num_envs,
        base_seed=cfg.data.train_seed_base,
        max_steps=cfg.data.episode_max_steps,
        in_process=True,
    )
    collector = FrameCollector(
        env, RandomPolicy(cfg.data.num_envs, seed=cfg.data.split_seed),
        replay, source="random", shard_dir=str(tmp_path / "data"),
        shard_every_episodes=2,
    )
    stats = collector.collect(frames)
    env.close()
    return replay, stats


def test_full_stage0_pipeline(tmp_path):
    cfg = tiny_cfg()
    replay, stats = _collect(tmp_path, cfg, frames=350)
    assert stats["frames"] >= 350
    assert replay.labeled_fraction() == 1.0  # 100%-dense privileged labels
    assert len(list(os.listdir(tmp_path / "data"))) >= 1  # shards persisted

    train_replay, val_replay = replay.split(cfg.data.val_ratio, seed=cfg.data.split_seed)
    assert train_replay.episodes >= 1 and val_replay.episodes >= 1

    device = "cpu"
    wm = ZephyrWM(cfg).to(device)
    heads = ZephyrHeads(cfg).to(device)
    final = train_world_model(
        wm, heads, train_replay, cfg, device=device, steps=12, log_every=6,
        ckpt_path=str(tmp_path / "wm.pt"), heads_ckpt_path=str(tmp_path / "heads.pt"),
        ckpt_every=6,
    )
    assert "loss" in final and np.isfinite(final["loss"])

    gates = evaluate_gates(wm, heads, val_replay, cfg, device=device)
    assert "zg1" in gates and "zg2" in gates and "zg3" in gates and "zg5" in gates
    assert gates["zg1"]["passed"], "tiny smoke should not collapse"

    # checkpoint reload works and matches
    wm2, _ = ZephyrWM.load_checkpoint(str(tmp_path / "wm.pt"))
    heads2, _ = ZephyrHeads.load_checkpoint(str(tmp_path / "heads.pt"))
    obs = next(iter(train_replay.records)).obs[0]
    import torch
    with torch.no_grad():
        z1 = wm.encode(torch.from_numpy(obs).unsqueeze(0))
        z2 = wm2.encode(torch.from_numpy(obs).unsqueeze(0))
    assert torch.equal(z1, z2)


def test_trainer_collect_train_resume(tmp_path):
    cfg = tiny_cfg()
    trainer = ZephyrTrainer(
        cfg, device="cpu", run_dir=str(tmp_path / "run"), env_in_process=True
    )
    stats = trainer.collect(200, source="random")
    assert stats["frames"] >= 200
    trainer.train(4, log_every=2)
    assert trainer.train_steps == 4
    path = trainer.save_run()
    reloaded = ZephyrTrainer.load_run(path, device="cpu")
    assert reloaded.train_steps == 4
    assert reloaded.env_frames == trainer.env_frames
    assert len(reloaded.replay) == len(trainer.replay)
    reloaded.train(2)
    assert reloaded.train_steps == 6


def test_screen_actor_and_cem_modes(tmp_path):
    from zephyr.screen import run_screen

    cfg = tiny_cfg()
    replay, _ = _collect(tmp_path, cfg, frames=400)
    train_replay, _ = replay.split(cfg.data.val_ratio, seed=cfg.data.split_seed)
    wm = ZephyrWM(cfg)
    heads = ZephyrHeads(cfg)
    train_world_model(
        wm, heads, train_replay, cfg, device="cpu", steps=3, log_every=3,
    )
    wm_path = str(tmp_path / "s_wm.pt")
    heads_path = str(tmp_path / "s_heads.pt")
    wm.save_checkpoint(wm_path, step=3)
    heads.save_checkpoint(heads_path, step=3)

    out = str(tmp_path / "screen_actor.json")
    report = run_screen(
        wm_path, heads_path, out_path=out, num_episodes=2, seed_start=2050,
        max_steps=20, agent_mode="actor", device="cpu",
    )
    with open(out) as f:
        on_disk = json.load(f)
    assert on_disk == json.loads(json.dumps(report))  # JSON-serializable
    assert report["protocol_id"] == "strict_visual_zephyr_clean_v1"
    assert report["strict_inference"] is True
    assert report["privileged_inventory_used"] is False
    assert report["achievements_used_for_scoring_only"] is True
    assert report["adaptation_mode"] == "frozen"
    assert report["num_episodes"] == 2
    assert len(report["episode_logs"]) == 2
    assert all(ep["seed"] == 2050 + i for i, ep in enumerate(report["episode_logs"]))
    assert "crafter_score" in report and report["crafter_score"] >= 0.0
    assert len(report["unlock_percentages"]) == 22

    out_cem = str(tmp_path / "screen_cem.json")
    report_cem = run_screen(
        wm_path, heads_path, out_path=out_cem, num_episodes=1, seed_start=2050,
        max_steps=8, agent_mode="cem", device="cpu",
    )
    assert report_cem["agent_mode"] == "cem"


def test_screen_cli_entrypoint(tmp_path):
    import subprocess
    import sys

    cfg = tiny_cfg()
    wm = ZephyrWM(cfg)
    heads = ZephyrHeads(cfg)
    wm_path = str(tmp_path / "cli_wm.pt")
    heads_path = str(tmp_path / "cli_heads.pt")
    wm.save_checkpoint(wm_path, step=1)
    heads.save_checkpoint(heads_path, step=1)
    out_json = str(tmp_path / "cli_screen.json")

    cmd = [
        sys.executable, "-m", "zephyr.screen",
        "--checkpoint_wm", wm_path,
        "--checkpoint_heads", heads_path,
        "--out_path", out_json,
        "--num_episodes", "1",
        "--seed_start", "2050",
        "--max_steps", "5",
        "--agent_mode", "actor",
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"screen CLI failed: {res.stderr}"
    assert os.path.exists(out_json)
    with open(out_json) as f:
        data = json.load(f)
    assert data["protocol_id"] == "strict_visual_zephyr_clean_v1"
    assert data["num_episodes"] == 1


def test_colab_runners_cli(tmp_path):
    import subprocess
    import sys

    # Test run_s0 CLI with --drive-dir and --smoke
    drive_dir = str(tmp_path / "drive_s0")
    cmd_s0 = [
        sys.executable, "zephyr/colab/run_s0.py",
        "--smoke", "--in-process",
        "--drive-dir", drive_dir,
        "--frames", "40",
        "--steps", "2",
        "--device", "cpu",
    ]
    res_s0 = subprocess.run(cmd_s0, capture_output=True, text=True, timeout=60)
    assert res_s0.returncode == 0, f"run_s0 failed: {res_s0.stderr}\n{res_s0.stdout}"
    assert os.path.exists(os.path.join(drive_dir, "reports", "zephyr_stage0_report.json"))

    # Test run_s1 CLI with --drive-dir and --smoke
    run_dir = str(tmp_path / "run_s1")
    cmd_s1 = [
        sys.executable, "-m", "zephyr.colab.run_s1",
        "--smoke", "--in-process",
        "--run-dir", run_dir,
        "--total-frames", "40",
        "--frames-per-cycle", "40",
        "--train-steps-per-cycle", "2",
        "--device", "cpu",
    ]
    res_s1 = subprocess.run(cmd_s1, capture_output=True, text=True, timeout=60)
    assert res_s1.returncode == 0, f"run_s1 failed: {res_s1.stderr}\n{res_s1.stdout}"
    assert os.path.exists(os.path.join(run_dir, "state.pt"))

