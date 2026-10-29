"""Unit tests for ATLAS v2.1 Re-mine protocol and loop instrumentation (F5 & F6)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.config import AtlasConfig
from atlas.goals import GoalEntry
from atlas.selflearn import IterationInstrumenter, SelfLearnLoop


def test_remine_path_and_idempotency():
    """Verification F6: Fake shard set is remined, event goals enter bank, marker is written and respected."""
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    # Pre-populate bank with 10 frontier goals
    for i in range(10):
        entry = GoalEntry(
            key=np.random.randn(128).astype(np.float32),
            context=np.random.randn(256).astype(np.float32),
            action_profile=np.zeros((17, 22), dtype=np.float32),
            event_dims=[],
            kind="frontier",
            born_iter=10,
        )
        agent.goals.add(entry)

    assert len(agent.goals.entries) == 10
    assert sum(1 for g in agent.goals.entries if g.kind == "event") == 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        shards_dir = root / "shards"
        runs_dir = root / "runs" / "test_run"
        shards_dir.mkdir(parents=True)
        runs_dir.mkdir(parents=True)

        # Create a fake shard with 2 episodes (T=15 each)
        T_ep = 15
        T_total = T_ep * 2
        obs = np.zeros((T_total, 64, 64, 3), dtype=np.uint8)
        actions = np.zeros(T_total, dtype=np.int64)
        rewards = np.zeros(T_total, dtype=np.float32)
        dones = np.zeros(T_total, dtype=bool)
        truncated = np.zeros(T_total, dtype=bool)
        discounts = np.ones(T_total, dtype=np.float32)
        events = np.zeros((T_total, 22), dtype=bool)
        mask_before = np.zeros((T_total, 22), dtype=bool)
        inventory = np.zeros((T_total, 16), dtype=np.float32)

        # Episode 1 unlocks achievement 4 (sapling) at step 5 and 21 (wake_up) at step 10
        events[5, 4] = True
        events[10, 21] = True
        rewards[5] = 1.0
        rewards[10] = 1.0

        shard_path = shards_dir / "shard_0_2.npz"
        np.savez_compressed(
            shard_path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncated=truncated,
            discounts=discounts,
            events=events,
            mask_before=mask_before,
            inventory=inventory,
            episode_lengths=np.array([T_ep, T_ep], dtype=np.int64),
            seeds=np.array([42, 43], dtype=np.int64),
            source="atlas_live",
        )

        # Create matching latents sidecar
        latents_path = shards_dir / "shard_0_2.latents.npz"
        z_arr = np.random.randn(T_total, 128).astype(np.float32)
        h_arr = np.random.randn(T_total, 256).astype(np.float32)
        target_residuals = np.random.randn(T_total, 128).astype(np.float32)
        np.savez_compressed(
            latents_path,
            z=z_arr,
            h=h_arr,
            target_residuals=target_residuals,
        )

        loop = SelfLearnLoop(
            agent=agent,
            run_id="test_run",
            run_dir=runs_dir,
            shards_dir=shards_dir,
            cfg=cfg,
        )

        # 1. Run re-mine
        marker_file = runs_dir / "REMINED"
        assert not marker_file.exists(), "REMINED marker should not exist before re-mine"

        stats = loop.remine_shards()

        # 2. Verify mined event goals appear in bank
        assert stats["episodes"] == 2
        assert stats["event_goals"] > 0, "Event goals should have been mined from unlock steps"
        event_count = sum(1 for g in agent.goals.entries if g.kind == "event")
        assert event_count == stats["event_goals"]
        assert event_count >= 2, f"Expected at least 2 event goals, got {event_count}"

        # 3. Verify REMINED marker is written atomically
        assert marker_file.exists(), "REMINED marker was not written"
        with open(marker_file, "r") as mf:
            marker_data = json.load(mf)
        assert marker_data["event_goals"] == stats["event_goals"]
        assert marker_data["episodes"] == 2

        # 4. Verify train_log.jsonl has remine_stats
        train_log = runs_dir / "train_log.jsonl"
        assert train_log.exists(), "train_log.jsonl not created"
        with open(train_log, "r") as tlf:
            lines = [json.loads(line) for line in tlf if line.strip()]
        assert any("remine_stats" in l for l in lines)

        # 5. Idempotency: second run sees marker and skips re-mine in run_atlas flow
        assert marker_file.exists()


def test_iteration_instrumenter():
    """Verification F5: IterationInstrumenter records telemetry without modifying agent decisions."""
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with IterationInstrumenter(agent) as inst:
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        # Act 5 steps
        for step in range(5):
            a, telem = agent.act(obs, reward_prev=0.0, step=step)
            assert isinstance(a, int)
            assert "mode" in telem

    # Stats dict structure is populated
    stats = inst.stats
    assert "goals_selected_by_kind" in stats
    assert "event" in stats["goals_selected_by_kind"]
    assert "vital" in stats["goals_selected_by_kind"]
    assert "frontier" in stats["goals_selected_by_kind"]
    assert "arrivals" in stats
    assert "probes_started" in stats
    assert "probe_unlocks" in stats
    assert "wander_starts" in stats
