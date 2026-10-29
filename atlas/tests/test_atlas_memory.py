"""Unit tests for TopoMap, GoalBank, and PredecessorLift."""

import tempfile
from pathlib import Path
import numpy as np

from atlas.config import TopoMapConfig, GoalsConfig
from atlas.goals import GoalBank, GoalEntry, PredecessorLift
from atlas.placemap import TopoMap


def test_topomap_add_match_persist():
    cfg = TopoMapConfig(scene_change_threshold=0.45, min_keyframe_interval=3, match_threshold=0.93)
    topo = TopoMap(cfg)

    z1 = np.ones(128, dtype=np.float32)
    h1 = np.zeros(256, dtype=np.float32)

    # First view -> initial keyframe
    kf1 = topo.observe(z1, h1, step=0)
    assert kf1 is not None
    assert len(topo.episode_keyframes) == 1

    # Identical view within min_interval -> no new keyframe, match exists
    kf_none = topo.observe(z1, h1, step=1)
    assert kf_none is None
    matched = topo.match(z1)
    assert matched is not None
    assert matched[1] >= 0.99  # High cosine similarity

    # Drastically different view after min_interval
    z2 = -np.ones(128, dtype=np.float32)
    kf2 = topo.observe(z2, h1, step=5)
    assert kf2 is not None
    assert len(topo.episode_keyframes) == 2

    # Reset episode: clears episode ring, persistent remains
    topo.reset_episode()
    assert len(topo.episode_keyframes) == 0
    assert len(topo.persistent_keyframes) == 2

    # Match in persistent store still succeeds
    assert topo.match(z1) is not None

    # Test save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_p = Path(tmp_dir) / "topomap.npz"
        topo.save(save_p)

        new_topo = TopoMap(cfg)
        new_topo.load(save_p)
        assert len(new_topo.persistent_keyframes) == 2
        assert new_topo.match(z1) is not None


def test_goalbank_merge_and_select():
    cfg = GoalsConfig()
    bank = GoalBank(cfg)

    z = np.ones(128, dtype=np.float32)
    h = np.zeros(256, dtype=np.float32)
    profile = np.zeros((17, 22), dtype=np.float32)
    profile[0, 5] = 0.8  # Action 0 unlocks event 5

    entry1 = GoalEntry(
        key=z,
        context=h,
        action_profile=profile,
        event_dims=[5],
        kind="event",
        n_try=1,
        n_succ=0,
    )
    bank.add(entry1)
    assert len(bank.entries) == 1

    # Add duplicate (cosine = 1.0 > 0.97) -> merges stats
    entry_dup = GoalEntry(
        key=z.copy(),
        context=h.copy(),
        action_profile=profile,
        event_dims=[5, 6],
        kind="event",
        n_try=2,
        n_succ=1,
    )
    bank.add(entry_dup)
    assert len(bank.entries) == 1
    assert bank.entries[0].n_try == 3
    assert bank.entries[0].n_succ == 1
    assert 6 in bank.entries[0].event_dims

    # Test selection
    m_soft = np.zeros(22, dtype=np.float32)
    best_goal, score = bank.select(
        z=z,
        h=h,
        m_soft=m_soft,
        psi_fn=lambda s, g: 0.9,
    )
    assert best_goal is not None
    assert score > 0.0

    # Save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "bank"
        bank.save(p)

        new_bank = GoalBank(cfg)
        new_bank.load(p)
        assert len(new_bank.entries) == 1
        assert new_bank.entries[0].n_try == 3


def test_predecessor_lift_mining():
    lift = PredecessorLift()

    # Synthetic pattern: 5 episodes where event 0 (wood) fires at t=10, event 1 (table) fires at t=20
    episodes = []
    for _ in range(5):
        ep = np.zeros((50, 22), dtype=bool)
        ep[10, 0] = True  # wood
        ep[20, 1] = True  # table
        episodes.append(ep)

    lift.mine(episodes, window=30)

    # Event 0 strictly precedes event 1 within window=30
    assert lift.lift[0, 1] > 1.0, f"Expected lift[0, 1] > 1.0, got {lift.lift[0, 1]}"
