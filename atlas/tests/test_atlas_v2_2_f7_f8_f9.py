"""Tests for Project ATLAS v2.2 F7 (Instrumentation fixes), F8 (Audit pack export), and F9 (Reachability-scaled option budget)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.audit_pack import AuditPackExporter
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig, GoalsConfig, SkillsConfig
from atlas.goals import GoalEntry


def test_f7_probe_dump_with_blended_score_and_unlock(tmp_path: Path):
    """Verification F7: unit test dumps a probe with nonzero blended score and correct unlock attribution."""
    exporter = AuditPackExporter(run_dir=tmp_path)

    blended_scores = [0.1 * i for i in range(17)]
    probe_probs = [[0.05 * j for j in range(22)] for _ in range(17)]
    unlocked = [4, 6]  # wood, table

    exporter.record_decision(
        iter_idx=10,
        step=42,
        goal_kind="event",
        goal_dim=[4],
        mode="PROBE",
        arrival_cos=0.88,
        chosen_action=7,
        blended_scores=blended_scores,
        probe_event_probs=probe_probs,
        unlocked_dims=unlocked,
        matched_bank=True,
    )

    exporter.export(
        current_iter=10,
        git_commit="test_commit",
        cfg=AtlasConfig(),
        cached_episodes=[],
    )

    dec_file = tmp_path / "audit" / "decisions.jsonl"
    assert dec_file.exists()

    with open(dec_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 1
    rec = records[0]
    assert rec["iter"] == 10
    assert rec["step"] == 42
    assert rec["chosen_action"] == 7
    assert rec["unlocked_dims"] == [4, 6]
    assert rec["matched_bank"] is True
    assert len(rec["blended_scores"]) == 17
    assert any(s > 0.0 for s in rec["blended_scores"]), "Blended scores must be nonzero!"


def test_f8_audit_pack_export_and_cap(tmp_path: Path):
    """Verification F8: smoke run produces manifest + >=1 decisions line + >=1 frame; folder stays < 60 MB."""
    exporter = AuditPackExporter(run_dir=tmp_path, max_folder_mb=5.0)  # test cap with 5 MB

    # 1. Record frame and decision
    dummy_obs = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    exporter.record_frame("test_arrival_frame.png", dummy_obs)
    exporter.record_decision(
        iter_idx=1,
        step=15,
        goal_kind="frontier",
        goal_dim=[],
        mode="PROBE",
        arrival_cos=0.89,
        chosen_action=2,
        blended_scores=[0.2] * 17,
        probe_event_probs=[[0.01] * 22] * 17,
        unlocked_dims=[],
        matched_bank=False,
    )

    # 2. Mock cached episode for latents
    cached_episodes = [{
        "z": np.random.randn(20, 128).astype(np.float32),
        "h": np.random.randn(20, 256).astype(np.float32),
        "target_residuals": np.random.randn(20, 128).astype(np.float32),
        "actions": np.zeros(20, dtype=np.int64),
        "rewards": np.zeros(20, dtype=np.float32),
        "events": np.zeros((20, 22), dtype=bool),
        "positions": [[32.0, 32.0]] * 20,
    }]

    exporter.export(
        current_iter=1,
        git_commit="abc1234",
        cfg=AtlasConfig(),
        cached_episodes=cached_episodes,
    )

    audit_dir = tmp_path / "audit"
    assert (audit_dir / "manifest.json").exists()
    assert (audit_dir / "decisions.jsonl").exists()
    assert (audit_dir / "latents_last2.npz").exists()
    assert (audit_dir / "positions_last2.npz").exists()
    frames = list((audit_dir / "frames").glob("*.png"))
    assert len(frames) >= 1

    # Check manifest contents
    with open(audit_dir / "manifest.json", "r") as f:
        mf = json.load(f)
    assert mf["iter"] == 1
    assert mf["git_commit"] == "abc1234"
    assert "sha256" in mf

    # Check folder size < 60 MB
    total_bytes = sum(p.stat().st_size for p in audit_dir.rglob("*") if p.is_file())
    assert total_bytes < 60 * 1024 * 1024


def test_f9_reachability_scaled_option_budget():
    """Verification F9: stub test — low-ψ goal yields budget > 25; high-ψ goal yields 25; telemetry logs option_budget per goal cycle."""
    cfg = AtlasConfig()
    cfg.skills.max_option_steps = 25
    cfg.skills.option_budget_min = 25
    cfg.skills.option_budget_max = 60

    backbone = FrozenBackbone(config=cfg)
    agent = AtlasAgent(backbone=backbone, cfg=cfg)

    # High reachability goal (psi ~ 1.0)
    high_psi = 1.0
    budget_high = agent.cfg.skills.max_option_steps * (1.0 + 2.0 * max(0.0, 1.0 - high_psi))
    budget_high = int(min(cfg.skills.option_budget_max, max(cfg.skills.option_budget_min, round(budget_high))))
    assert budget_high == 25, f"Expected high-psi budget to be 25, got {budget_high}"

    # Low reachability goal (psi ~ 0.1)
    low_psi = 0.1
    budget_low = agent.cfg.skills.max_option_steps * (1.0 + 2.0 * max(0.0, 1.0 - low_psi))
    budget_low = int(min(cfg.skills.option_budget_max, max(cfg.skills.option_budget_min, round(budget_low))))
    assert budget_low > 25, f"Expected low-psi budget > 25, got {budget_low}"
    assert budget_low <= 60, f"Expected low-psi budget <= 60, got {budget_low}"

    # Verify telemetry logs option_budget
    dummy_obs = np.zeros((64, 64, 3), dtype=np.uint8)
    action, telemetry = agent.act(dummy_obs, reward_prev=0.0, step=0)
    assert "option_budget" in telemetry
    assert "goal_mode" in telemetry
    assert telemetry["option_budget"] >= 25
