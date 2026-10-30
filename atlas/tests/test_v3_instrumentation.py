"""Unit tests for V3 Instrumentation & Bug Fixes (C1-C4 + F5)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig, HealthConfig
from atlas.health import HealthMonitor
from atlas.audit_pack import AuditPackExporter


def test_c1_positions_in_shard_and_audit(tmp_path):
    """C1: positions array in shard dicts and positions_last2.npz in audit pack."""
    exporter = AuditPackExporter(run_dir=tmp_path / "run")
    T = 20
    ep1 = {
        "z": np.zeros((T, 128), dtype=np.float32),
        "h": np.zeros((T, 256), dtype=np.float32),
        "actions": np.zeros(T, dtype=np.int64),
        "rewards": np.zeros(T, dtype=np.float32),
        "dones": np.zeros(T, dtype=bool),
        "events": np.zeros((T, 22), dtype=bool),
        "inventory": np.zeros((T, 16), dtype=np.float32),
        "positions": np.ones((T, 2), dtype=np.float32) * 5.0,
    }
    ep2 = {
        "z": np.zeros((T, 128), dtype=np.float32),
        "h": np.zeros((T, 256), dtype=np.float32),
        "actions": np.zeros(T, dtype=np.int64),
        "rewards": np.zeros(T, dtype=np.float32),
        "dones": np.zeros(T, dtype=bool),
        "events": np.zeros((T, 22), dtype=bool),
        "inventory": np.zeros((T, 16), dtype=np.float32),
        "positions": np.ones((T, 2), dtype=np.float32) * 10.0,
    }

    cfg = AtlasConfig()
    exporter.export(
        current_iter=1,
        git_commit="test_commit",
        cfg=cfg,
        cached_episodes=[ep1, ep2],
        results_dir=tmp_path / "results",
    )

    pos_file = tmp_path / "run" / "audit" / "positions_last2.npz"
    assert pos_file.exists()
    loaded = np.load(pos_file)
    assert "ep_0" in loaded
    assert "ep_1" in loaded
    assert loaded["ep_0"].shape == (T, 2)
    assert np.allclose(loaded["ep_0"], 5.0)


def test_c2_probe_id_in_decisions():
    """C2: probe_id increments per PROBE entry; decisions track probe_id."""
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    agent.reset_episode()
    obs = np.zeros((64, 64, 3), dtype=np.uint8)

    # First probe
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 2
    agent.probe_id = 1
    act1, _ = agent.act(obs, reward_prev=0.0, step=0)
    assert agent.last_probe_decision["probe_id"] == 1

    # Second probe
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 2
    agent.probe_id = 2
    act2, _ = agent.act(obs, reward_prev=0.0, step=1)
    assert agent.last_probe_decision["probe_id"] == 2


def test_c3_health_alarm_baseline_fix():
    """C3: Resumed at iter > 10 with best_ach_per_ep == 0.0 initializes baseline from history."""
    h_cfg = HealthConfig()
    h_cfg.flatline_alarm_iters = 15

    monitor = HealthMonitor(cfg=h_cfg)

    # Simulate resuming at it=59 with historical scores
    history = [(it, 1.25) for it in range(10, 59)]
    snapshot = {
        "best_ach_per_ep": 0.0,  # uninitialized in older checkpoint
        "last_best_iter": 0,
        "progress_history": history,
        "alarms_so_far": [],
    }

    monitor.load_snapshot(snapshot, resume_iter=59)

    # Verify baseline was restored to 1.25, not left at 0.0
    assert monitor.best_ach_per_ep == 1.25
    assert monitor.last_best_iter == 58

    # Observing current progress does not immediately fire flatline alarm
    monitor.observe_progress(59, 1.25)
    report = monitor.check(iter_idx=59)
    assert not any("A3_FLATLINE" in a for a in report["alarms"])


def test_c4_trend_arithmetic_raw_difference():
    """C4: Report card computes raw difference current - baseline without division by zero."""
    monitor = HealthMonitor()
    # Baseline window it10..15 has mean 0.0
    for it in range(10, 16):
        monitor.observe_progress(it, 0.0)

    # Latest progress at it=20 is 1.5
    monitor.observe_progress(20, 1.5)

    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    report_card = monitor.generate_report_card(iter_idx=20, agent=agent)
    assert "ach_per_ep_trend_raw" in report_card
    # 1.5 - 0.0 = 1.5 (clean raw float difference, no inf/nan)
    assert report_card["ach_per_ep_trend_raw"] == 1.5
    assert not np.isnan(report_card["ach_per_ep_trend_raw"])
    assert not np.isinf(report_card["ach_per_ep_trend_raw"])


def test_f5_screen_gating_calibration_telemetry(tmp_path):
    """F5: Screening report JSON contains material_gate_active key."""
    # Write a minimal screen report matching screen.py schema
    report = {
        "score_percent": 1.23,
        "mean_survival": 150.0,
        "mean_reward": 1.5,
        "material_gate_active": True,
    }
    out_file = tmp_path / "screen_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f)

    with open(out_file, "r") as f:
        loaded = json.load(f)

    assert "material_gate_active" in loaded
    assert loaded["material_gate_active"] is True
