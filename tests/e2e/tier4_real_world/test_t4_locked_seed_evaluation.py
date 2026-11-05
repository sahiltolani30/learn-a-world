"""Tier 4: Scenario 1 - Locked Evaluation Seeds (2050..2099) Acceptance Harness.

Covers:
- F1 (Pixels Only), F2 (17 Actions), F8 (Actor), F20 (Locked Seeds Evaluation),
  F21 (Score Metric), F22 (Survival Steps).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import types
import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

# Cleanly bind eval.metrics without executing eval/__init__.py (which imports legacy models/planning/training)
if "eval" not in sys.modules:
    eval_mod = types.ModuleType("eval")
    sys.modules["eval"] = eval_mod
if "eval.metrics" not in sys.modules:
    spec = importlib.util.spec_from_file_location("eval.metrics", str(REPO_ROOT / "eval" / "metrics.py"))
    metrics_mod = importlib.util.module_from_spec(spec)
    sys.modules["eval.metrics"] = metrics_mod
    spec.loader.exec_module(metrics_mod)
    setattr(sys.modules["eval"], "metrics", metrics_mod)

from envs.crafter_wrapper import CrafterEnv
from zephyr.config import ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.screen import run_screen
from zephyr.wm import ZephyrWM


def test_t4_locked_01_seed_range_quarantine_and_deterministic_bounds(crafter_env_factory):
    """T4-LOCKED-01: Seed range 2050..2099 quarantine and deterministic reproducibility."""
    cfg = ZephyrConfig()

    bench_start = cfg.data.benchmark_seed_start
    bench_eps = cfg.data.benchmark_episodes
    bench_range = set(range(bench_start, bench_start + bench_eps))

    assert bench_start == 2050
    assert bench_eps == 50
    assert len(bench_range) == 50

    # Strict Quarantine: No overlap with dev screening (6150..6159) or training seeds (7000+)
    dev_range = set(range(cfg.data.dev_screen_seed_start, cfg.data.dev_screen_seed_start + cfg.data.dev_screen_episodes))
    train_range = set(range(cfg.data.train_seed_base, cfg.data.train_seed_base + 1000))

    assert len(bench_range.intersection(dev_range)) == 0, "Benchmark seeds overlap with dev screening"
    assert len(bench_range.intersection(train_range)) == 0, "Benchmark seeds overlap with training pool"

    # Deterministic reproducibility on seed 2050
    env1 = crafter_env_factory(seed=2050, max_steps=10)
    o1, _ = env1.reset(seed=2050)
    for _ in range(5):
        o1, _, _, _, _ = env1.step(0)

    env2 = crafter_env_factory(seed=2050, max_steps=10)
    o2, _ = env2.reset(seed=2050)
    for _ in range(5):
        o2, _, _, _, _ = env2.step(0)

    assert np.array_equal(o1, o2), "Crafter reset on seed 2050 is not bit-for-bit deterministic"


def test_t4_locked_02_evaluation_screen_execution(tmp_path, tiny_test_cfg):
    """T4-LOCKED-02: End-to-end evaluation benchmark screen via run_screen (Mode A Actor)."""
    wm = ZephyrWM(tiny_test_cfg).to("cpu")
    heads = ZephyrHeads(tiny_test_cfg).to("cpu")

    wm_path = os.path.join(str(tmp_path), "wm.pt")
    heads_path = os.path.join(str(tmp_path), "heads.pt")
    out_json = os.path.join(str(tmp_path), "benchmark_report.json")

    wm.save_checkpoint(wm_path, step=100)
    heads.save_checkpoint(heads_path, step=100)

    report = run_screen(
        checkpoint_wm=wm_path,
        checkpoint_heads=heads_path,
        out_path=out_json,
        num_episodes=2,
        seed_start=2050,
        max_steps=10,
        agent_mode="actor",
        device="cpu",
    )

    assert os.path.exists(out_json)
    assert report["protocol_id"] == "strict_visual_zephyr_clean_v1"
    assert report["seed_start"] == 2050
    assert report["seed_end"] == 2051
    assert report["num_episodes"] == 2
    assert report["agent_mode"] == "actor"
    assert report["strict_inference"] is True
    assert report["privileged_inventory_used"] is False
    assert len(report["episode_logs"]) == 2


def test_t4_locked_03_provenance_metadata_and_cryptographic_integrity(tmp_path, tiny_test_cfg):
    """T4-LOCKED-03: Cryptographic provenance auditing (SHA-256 and strict inference flags)."""
    wm = ZephyrWM(tiny_test_cfg).to("cpu")
    heads = ZephyrHeads(tiny_test_cfg).to("cpu")

    wm_path = os.path.join(str(tmp_path), "wm.pt")
    heads_path = os.path.join(str(tmp_path), "heads.pt")
    out_json = os.path.join(str(tmp_path), "provenance_report.json")

    wm.save_checkpoint(wm_path, step=50)
    heads.save_checkpoint(heads_path, step=50)

    report = run_screen(
        checkpoint_wm=wm_path,
        checkpoint_heads=heads_path,
        out_path=out_json,
        num_episodes=1,
        seed_start=2050,
        max_steps=5,
        agent_mode="actor",
        device="cpu",
    )

    with open(wm_path, "rb") as f:
        expected_wm_hash = hashlib.sha256(f.read()).hexdigest()
    with open(heads_path, "rb") as f:
        expected_heads_hash = hashlib.sha256(f.read()).hexdigest()

    assert report["checkpoint_wm_sha256"] == expected_wm_hash
    assert report["checkpoint_heads_sha256"] == expected_heads_hash
    assert isinstance(report["git_commit"], str)
    assert report["strict_inference"] is True
    assert report["privileged_inventory_used"] is False
    assert report["achievements_used_for_scoring_only"] is True


def test_t4_locked_04_cem_agent_mode_b_ablation_execution(tmp_path, tiny_test_cfg):
    """T4-LOCKED-04: CEM Mode B ablation planner execution on locked evaluation seeds."""
    wm = ZephyrWM(tiny_test_cfg).to("cpu")
    heads = ZephyrHeads(tiny_test_cfg).to("cpu")

    wm_path = os.path.join(str(tmp_path), "wm.pt")
    heads_path = os.path.join(str(tmp_path), "heads.pt")
    out_json = os.path.join(str(tmp_path), "cem_ablation_report.json")

    wm.save_checkpoint(wm_path, step=50)
    heads.save_checkpoint(heads_path, step=50)

    report = run_screen(
        checkpoint_wm=wm_path,
        checkpoint_heads=heads_path,
        out_path=out_json,
        num_episodes=1,
        seed_start=2050,
        max_steps=5,
        agent_mode="cem",
        device="cpu",
    )

    assert report["agent_mode"] == "cem"
    assert report["planner_configuration"]["agent_mode"] == "cem"
    assert len(report["episode_logs"]) == 1
    assert os.path.exists(out_json)
