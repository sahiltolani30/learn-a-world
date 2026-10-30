"""Verification tests for T1 Decision Trace (atlas/trace.py)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from atlas.arbiter import AtlasAgent
from atlas.config import AtlasConfig, NUM_ACTIONS, NUM_ACHIEVEMENTS
from atlas.trace import TraceWriter


class StubBackbone:
    def __init__(self):
        self.device = "cpu"
        self.has_learned_delta_head = False
        self.checkpoint_path = "mock.pt"
        self.provenance = {}

    def encode(self, obs_uint8):
        return torch.randn(1, 128)

    def recurrent_step(self, z, a_prev, h_prev):
        return torch.randn(1, 256)

    def composite(self, z, h):
        return torch.cat([z, h], dim=-1)

    def event_probs_all_actions(self, s, mask):
        return torch.full((NUM_ACTIONS, NUM_ACHIEVEMENTS), 0.05)

    def event_logits_all_actions(self, s, mask):
        return torch.full((s.shape[0], NUM_ACTIONS, NUM_ACHIEVEMENTS), 0.05)

    def event_probs(self, s, a, mask):
        return torch.full((1, NUM_ACHIEVEMENTS), 0.05)

    def prior_logits_weighted(self, s, mask):
        return torch.zeros(1, NUM_ACTIONS)

    def q_mean(self, s, a):
        return torch.tensor([0.0])


class StubEpistemic:
    def disagreement_all_actions(self, s):
        return torch.full((1, NUM_ACTIONS), 0.1)

    def disagreement(self, s, a):
        return torch.tensor([0.1])


class StubSafety:
    def __init__(self, override_step: int = -1):
        self.override_step = override_step
        self.current_step = 0

    def risk_all_actions(self, s):
        risks = torch.full((1, NUM_ACTIONS), 0.05)
        if self.current_step == self.override_step:
            risks[0, :] = 0.02
            risks[0, 0] = 0.90  # Force override on action 0
        self.current_step += 1
        return risks


def test_t1_signature_unchanged():
    sig = inspect.signature(AtlasAgent.act)
    params = list(sig.parameters.keys())
    assert params == ["self", "obs_uint8", "reward_prev", "step"], f"Signature changed: {params}"


def test_t1_trace_schema_and_keys(tmp_path: Path):
    cfg = AtlasConfig()
    cfg.skills.arrival_cosine = 0.85
    agent = AtlasAgent(cfg=cfg, backbone=StubBackbone())
    agent.epistemic_net = StubEpistemic()
    agent.safety_net = StubSafety()

    writer = TraceWriter(run_dir=tmp_path, max_folder_mb=50.0)
    agent.trace_enabled = True

    writer.begin_episode(iter_idx=10, ep_idx=1, seed=1234)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    for t in range(30):
        action, tel = agent.act(obs, reward_prev=0.0, step=t)
        writer.write_step(agent, step=t, reward_prev=0.0)

    writer.end_episode(length=30, total_reward=0.0, died=False)

    trace_file = tmp_path / "trace" / "trace_it10.jsonl"
    assert trace_file.exists()

    lines = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]
    assert len(lines) == 30

    contractual_keys = {
        "iter", "ep_idx", "seed", "t", "mode", "prev_mode", "goal",
        "scores_source", "scores", "a_intended", "a_final",
        "epsilon_fired", "epsilon", "beta_t", "override", "risks",
        "event_p_max", "m_soft", "newly_unlocked", "probe_id",
        "abort_reason", "arrived", "material_gate", "r_prev", "r_int",
    }

    for idx, row in enumerate(lines):
        assert contractual_keys.issubset(set(row.keys())), f"Missing keys at step {idx}: {contractual_keys - set(row.keys())}"
        assert len(row["scores"]) == 17
        assert len(row["risks"]) == 17
        assert len(row["event_p_max"]) == 22
        assert len(row["m_soft"]) == 22


def test_t1_epsilon_zero_no_fires(tmp_path: Path):
    cfg = AtlasConfig()
    agent = AtlasAgent(cfg=cfg, backbone=StubBackbone())
    agent.epsilon = 0.0
    agent.epistemic_net = StubEpistemic()
    agent.safety_net = StubSafety()

    writer = TraceWriter(run_dir=tmp_path)
    agent.trace_enabled = True
    writer.begin_episode(iter_idx=1, ep_idx=0, seed=42)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    for t in range(20):
        agent.act(obs, reward_prev=0.0, step=t)
        writer.write_step(agent, step=t, reward_prev=0.0)
    writer.end_episode(length=20, total_reward=0.0, died=False)

    trace_file = tmp_path / "trace" / "trace_it1.jsonl"
    lines = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]
    for row in lines:
        assert row["epsilon_fired"] is False
        # When override didn't fire and epsilon didn't fire, intended == final
        if not row["override"]["fired"]:
            assert row["a_intended"] == row["a_final"]


def test_t1_override_trace(tmp_path: Path):
    cfg = AtlasConfig()
    agent = AtlasAgent(cfg=cfg, backbone=StubBackbone())
    agent.epsilon = 0.0
    agent.epistemic_net = StubEpistemic()
    # Force override at step 5
    safety = StubSafety(override_step=5)
    agent.safety_net = safety

    writer = TraceWriter(run_dir=tmp_path)
    agent.trace_enabled = True
    writer.begin_episode(iter_idx=2, ep_idx=0, seed=42)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    for t in range(10):
        agent.act(obs, reward_prev=0.0, step=t)
        writer.write_step(agent, step=t, reward_prev=0.0)
    writer.end_episode(length=10, total_reward=0.0, died=False)

    trace_file = tmp_path / "trace" / "trace_it2.jsonl"
    lines = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]

    overrides = [row for row in lines if row["override"]["fired"]]
    assert len(overrides) == 1
    assert overrides[0]["t"] == 5
    assert overrides[0]["override"]["a_intended"] != overrides[0]["override"]["a_final"]


def test_t1_scores_source_modes(tmp_path: Path):
    cfg = AtlasConfig()
    agent = AtlasAgent(cfg=cfg, backbone=StubBackbone())
    agent.epistemic_net = StubEpistemic()
    agent.safety_net = StubSafety()

    writer = TraceWriter(run_dir=tmp_path)
    agent.trace_enabled = True
    writer.begin_episode(iter_idx=3, ep_idx=0, seed=42)

    # Test FALLBACK mode
    agent.mode = "FALLBACK"
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    agent.act(obs, reward_prev=0.0, step=0)
    writer.write_step(agent, step=0, reward_prev=0.0)

    # Test WANDER mode
    agent.mode = "WANDER"
    agent.wander_steps_remaining = 5
    agent.act(obs, reward_prev=0.0, step=1)
    writer.write_step(agent, step=1, reward_prev=0.0)

    # Test PROBE mode
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 3
    agent.act(obs, reward_prev=0.0, step=2)
    writer.write_step(agent, step=2, reward_prev=0.0)

    writer.end_episode(length=3, total_reward=0.0, died=False)
    trace_file = tmp_path / "trace" / "trace_it3.jsonl"
    lines = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]

    sources = [row["scores_source"] for row in lines]
    assert sources[0] == "fallback_logits"
    assert sources[1] == "wander_logits"
    assert sources[2] == "probe_blend"


def test_t1_pruning_under_cap(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(parents=True)

    # Create dummy files
    for it in range(10):
        p = trace_dir / f"trace_it{it}.jsonl"
        p.write_text("x" * 1024 * 100)  # 100 KB each -> 1 MB total

    # Set cap to 0.5 MB -> should delete oldest files
    writer = TraceWriter(run_dir=tmp_path, max_folder_mb=0.5)
    writer._enforce_size_cap()

    remaining_files = list(trace_dir.glob("trace_it*.jsonl"))
    total_bytes = sum(f.stat().st_size for f in remaining_files)
    assert total_bytes <= 0.5 * 1024 * 1024
    # Ensure oldest (trace_it0) was deleted
    assert not (trace_dir / "trace_it0.jsonl").exists()


def test_t5_relations_snapshot(tmp_path: Path):
    """T5 Verification: relations snapshot matches contractual schema and anonymity."""
    from atlas.audit_pack import export_relations_snapshot
    from atlas.goals import GoalEntry, GoalBank, PredecessorLift

    cfg = AtlasConfig()
    agent = AtlasAgent(cfg=cfg, backbone=StubBackbone())
    agent.lift = PredecessorLift()
    # Set 1 edge with ci_low > 0
    agent.lift.ci_low[1, 7] = 1.35
    agent.lift.lift[1, 7] = 2.40

    # 2-goal bank
    agent.goal_bank = GoalBank(cfg=cfg.goals)
    g1 = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[7],
        kind="event",
        n_try=3,
        n_succ=1,
    )
    g1.world_frequency = 0.28
    g1.born_iter = 40
    g2 = GoalEntry(
        key=np.zeros(128, dtype=np.float32),
        context=np.zeros(256, dtype=np.float32),
        action_profile=np.zeros((17, 22), dtype=np.float32),
        event_dims=[1],
        kind="frontier",
        n_try=1,
        n_succ=0,
    )
    g2.world_frequency = 0.05
    g2.born_iter = 50
    agent.goal_bank.entries = [g1, g2]

    # Item event map
    agent.item_event_map = {0: 7, 1: None}

    audit_dir = tmp_path / "audit"
    out_file = export_relations_snapshot(audit_dir=audit_dir, iter_idx=121, agent=agent)

    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)

    assert data["iter"] == 121
    assert "lift_edges" in data
    assert "item_event_map" in data
    assert "bank" in data

    # 1-edge lift: [[1, 7, 1.35, 2.4]]
    assert len(data["lift_edges"]) == 1
    edge = data["lift_edges"][0]
    assert edge[0] == 1 and edge[1] == 7
    assert abs(edge[2] - 1.35) < 1e-3
    assert abs(edge[3] - 2.40) < 1e-3

    # Assert no string field in bank other than 'kind'
    assert len(data["bank"]) == 2
    for b in data["bank"]:
        for k, v in b.items():
            if k == "kind":
                assert isinstance(v, str)
            else:
                assert not isinstance(v, str), f"Field '{k}' in bank is a string: {v}"

    # Assert item_event_map values are ints or null
    for k, v in data["item_event_map"].items():
        assert v is None or isinstance(v, int), f"item_event_map value for {k} is not int/None: {v}"
    assert data["item_event_map"]["0"] == 7
    assert data["item_event_map"]["1"] is None

