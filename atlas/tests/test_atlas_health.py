"""Unit tests for HealthMonitor: alarms A1, A2, A3, interventions, and report cards."""

import json
import tempfile
from pathlib import Path
import numpy as np

from atlas.config import HealthConfig
from atlas.health import HealthMonitor
from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig


def test_alarm_a1_action_diversity_triggers_boost():
    cfg = HealthConfig(window_steps=200, action_share_alarm=0.60, epsilon_boost=2.0, boost_steps=100)
    monitor = HealthMonitor(cfg)

    # Feed 200 identical actions (action 0)
    for _ in range(200):
        monitor.observe_action(0)

    res = monitor.check(iter_idx=1)
    assert any("A1_ACTION_DIVERSITY" in a for a in res["alarms"])
    assert res["epsilon_boost_active"] is True
    assert res["boost_steps_left"] == 100

    # Step notifications decrement boost
    for _ in range(50):
        monitor.notify_step()
    assert monitor.boost_steps_left == 50
    assert monitor.epsilon_boost_active is True

    for _ in range(50):
        monitor.notify_step()
    assert monitor.boost_steps_left == 0
    assert monitor.epsilon_boost_active is False


def test_alarm_a2_confidence():
    cfg = HealthConfig(confidence_alarm=0.85)
    monitor = HealthMonitor(cfg)

    monitor.observe_confidence(0.90)
    res = monitor.check(iter_idx=1)
    assert any("A2_CONFIDENCE" in a for a in res["alarms"])


def test_alarm_a3_flatline_triggers_beta_reset():
    cfg = HealthConfig(flatline_iters=15, flatline_improvement=0.10)
    monitor = HealthMonitor(cfg)

    # Feed 15 flat progress readings
    for i in range(15):
        monitor.observe_progress(iter_idx=i, ach_per_ep=1.0)

    res = monitor.check(iter_idx=15)
    assert any("A3_FLATLINE" in a for a in res["alarms"])
    assert res["beta_reset"] is True


def test_report_card_schema():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    monitor = HealthMonitor(cfg.health)
    monitor.observe_action(1)
    monitor.observe_confidence(0.70)
    monitor.observe_progress(iter_idx=10, ach_per_ep=2.0)
    monitor.observe_progress(iter_idx=30, ach_per_ep=2.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir) / "health_report.json"
        card = monitor.generate_report_card(iter_idx=30, agent=agent, out_path=out_p)

        assert out_p.exists()
        with open(out_p, "r") as f:
            data = json.load(f)

        required_keys = [
            "top_action_share",
            "optionnet_confidence",
            "goal_bank_largest_dim_share",
            "goal_bank_per_dim_counts",
            "ach_per_ep_trend_vs_it10",
            "lift_edges_used",
            "max_ci_low",
            "alarms_so_far",
            "pass",
        ]
        for k in required_keys:
            assert k in data, f"Missing required key '{k}' in report card"

        pass_keys = ["diversity", "humility", "album_variety", "trend", "lift"]
        for pk in pass_keys:
            assert pk in data["pass"], f"Missing pass subkey '{pk}'"
