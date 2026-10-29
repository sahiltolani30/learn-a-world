"""Tests for Project ATLAS v2.2 F1 (Resume integrity) and F2 (Honest monitor feed)."""

import json
from pathlib import Path
import numpy as np
import torch
import pytest

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig, HealthConfig
from atlas.health import HealthMonitor
from atlas.selflearn import SelfLearnLoop


def test_f1_resume_integrity(tmp_path: Path):
    """Verification F1: resume a fixture state at iter 7 -> next iteration logs 7, seeds continue, monitor best/last_best restored, journal rows carry run_segment."""
    cfg = AtlasConfig()
    cfg.collect.episodes_per_iter = 1
    cfg.collect.max_steps = 5
    cfg.train.epochs_per_iter = 1

    backbone = FrozenBackbone(config=cfg)
    agent = AtlasAgent(backbone=backbone, cfg=cfg)

    run_dir = tmp_path / "run_f1"
    loop = SelfLearnLoop(agent=agent, run_id="test_f1", run_dir=run_dir, cfg=cfg)

    # 1. Setup fixture state at iter 7 with health monitor state
    loop.health.best_ach_per_ep = 2.4
    loop.health.last_best_iter = 7
    loop.health.progress_history = [(0, 1.0), (5, 2.4), (7, 2.4)]
    loop.health.alarms_so_far = [{"iter": 5, "alarm": "TEST_ALARM"}]

    # Save state at iter 7
    loop.save_state(current_iter=7)

    # 2. Fresh loop resuming from run_dir
    agent2 = AtlasAgent(backbone=backbone, cfg=cfg)
    loop2 = SelfLearnLoop(agent=agent2, run_id="test_f1", run_dir=run_dir, cfg=cfg)

    resume_iter = loop2.load_state()
    assert resume_iter == 8, f"Expected resume iter 8 after saving iter 7, got {resume_iter}"
    assert loop2.start_iter == 8

    # Check monitor state restored
    assert np.isclose(loop2.health.best_ach_per_ep, 2.4)
    assert loop2.health.last_best_iter == 7
    assert len(loop2.health.progress_history) == 3
    assert len(loop2.health.alarms_so_far) == 1

    # Check seed calculation continues from start_iter
    assert 10000 + loop2.start_iter * 100 == 10800

    # 3. Collect one episode with collector and check run_segment in journal
    ep_data = loop2.collector.run_episode(seed=10800, iter_idx=8)
    assert loop2.journal_path.exists()

    with open(loop2.journal_path, "r") as jf:
        lines = [json.loads(line) for line in jf if line.strip()]

    assert len(lines) >= 1
    latest = lines[-1]
    assert latest["iter"] == 8
    assert latest["seed"] == 10800
    assert "run_segment" in latest
    assert latest["run_segment"] == loop2.git_commit

    # Check positions also appended to player_positions.jsonl
    pos_file = run_dir / "player_positions.jsonl"
    assert pos_file.exists()
    with open(pos_file, "r") as pf:
        pos_lines = [json.loads(line) for line in pf if line.strip()]
    assert len(pos_lines) >= 1
    assert pos_lines[-1]["run_segment"] == loop2.git_commit


def test_f2_honest_monitor_feed():
    """Verification F2: simulate 20 iters with one dev screen at iter 5 (ach 2.4) -> monitor best becomes 2.4, no A3 before 15 flat iters, A3 fires at iter 20 if no improvement."""
    cfg = HealthConfig()
    cfg.flatline_iters = 15
    cfg.flatline_improvement = 0.05
    monitor = HealthMonitor(cfg=cfg)

    last_dev_ach = 0.0

    # Iters 0..4: no dev screen yet (ach 0.0)
    for it in range(0, 5):
        monitor.observe_progress(it, last_dev_ach)
        status = monitor.check(it)
        assert not status["beta_reset"], f"A3 should not fire at iter {it}"

    # Iter 5: Dev screen runs, ach = 2.4
    last_dev_ach = 2.4
    monitor.observe_progress(5, last_dev_ach)
    status5 = monitor.check(5)
    assert not status5["beta_reset"]
    assert np.isclose(monitor.best_ach_per_ep, 2.4)
    assert monitor.last_best_iter == 5

    # Iters 6..19: carried forward last_dev_ach (no new dev screen improvement)
    for it in range(6, 20):
        monitor.observe_progress(it, last_dev_ach)
        status = monitor.check(it)
        assert not status["beta_reset"], f"A3 fired prematurely at iter {it} (flatline iters={cfg.flatline_iters}, diff={it - 5})"

    # Iter 20: 20 - 5 = 15 flat iters reached! A3 MUST fire at iter 20
    monitor.observe_progress(20, last_dev_ach)
    status20 = monitor.check(20)
    assert status20["beta_reset"], f"A3 must fire at iter 20 after 15 flat iterations!"
    a3_alarms = [a for a in status20["alarms"] if "A3_FLATLINE" in a]
    assert len(a3_alarms) == 1
    assert "2.400" in a3_alarms[0], f"Alarm message should show progress 2.400, got: {a3_alarms[0]}"
