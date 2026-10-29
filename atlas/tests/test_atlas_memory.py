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


def test_goalbank_dedupe_and_select():
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
    added = bank.add(entry1)
    assert added is not None
    assert len(bank.entries) == 1
    assert np.isclose(np.linalg.norm(bank.entries[0].key), 1.0, atol=1e-5)

    # Add duplicate (cosine = 1.0 > 0.90) -> rejected, rejected_same_kind incremented
    entry_dup = GoalEntry(
        key=z.copy(),
        context=h.copy(),
        action_profile=profile,
        event_dims=[5],
        kind="event",
        n_try=2,
        n_succ=1,
    )
    res = bank.add(entry_dup)
    assert res is None
    assert bank.rejected_same_kind == 1
    assert len(bank.entries) == 1

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
        assert new_bank.entries[0].n_try == 1
        assert new_bank._dim_counts[5] == 1


def test_goalbank_dim_cap_and_norms():
    cfg = GoalsConfig(max_goals_per_dim=15)
    bank = GoalBank(cfg)

    # Insert 100 same-dim event goals (with orthogonal or distinct keys to bypass dedupe)
    rng = np.random.RandomState(42)
    for _ in range(100):
        key = rng.randn(128).astype(np.float32)
        entry = GoalEntry(
            key=key,
            context=np.zeros(256, dtype=np.float32),
            action_profile=np.zeros((17, 22), dtype=np.float32),
            event_dims=[3],
            kind="event",
        )
        bank.add(entry)

    # Must hold <= 15 goals for dim 3
    dim_3_goals = [g for g in bank.entries if 3 in g.event_dims]
    assert len(dim_3_goals) <= 15
    assert bank._dim_counts[3] <= 15

    # All stored keys have norm 1.0 +- 1e-6
    for g in bank.entries:
        norm = np.linalg.norm(g.key)
        assert np.isclose(norm, 1.0, atol=1e-5), f"Key norm {norm} is not 1.0"


def test_predecessor_lift_mining():
    lift = PredecessorLift()

    # 1. Rare-event pattern: d fires once in 1 episode -> base tiny, count_d < 20
    # Must NOT produce any edge above 1.2
    episodes_rare = []
    for _ in range(10):
        ep = np.zeros((50, 22), dtype=bool)
        episodes_rare.append(ep)
    episodes_rare[0][10, 0] = True
    episodes_rare[0][20, 1] = True

    lift.mine(episodes_rare, window=30)
    assert lift.ci_low[0, 1] == 0.0
    assert lift.lift[0, 1] == 1.0
    assert not np.any(lift.ci_low > 1.2)

    # 2. Strong common pattern: 25 episodes where event 0 fires at t=10 and event 1 at t=20
    # count_d = 25 >= 20, k = 25 >= 3, n_e = 25 >= 1
    episodes_common = []
    for _ in range(25):
        ep = np.zeros((50, 22), dtype=bool)
        ep[10, 0] = True  # wood
        ep[20, 1] = True  # table
        episodes_common.append(ep)

    lift.mine(episodes_common, window=30)
    assert lift.lift[0, 1] > 2.0, f"Expected lift > 2.0, got {lift.lift[0, 1]}"
    assert lift.ci_low[0, 1] <= lift.lift[0, 1] * 1.01 + 1e-6
    lift.assert_consistent()


def test_goalbank_eviction_when_full():
    """Verify that GoalBank eviction at capacity works without numpy array equality ambiguity."""
    cfg = GoalsConfig(max_goals=10, max_goals_per_dim=50)
    bank = GoalBank(cfg)
    rng = np.random.RandomState(123)

    for i in range(25):
        key = rng.randn(128).astype(np.float32)
        entry = GoalEntry(
            key=key,
            context=rng.randn(256).astype(np.float32),
            action_profile=np.zeros((17, 22), dtype=np.float32),
            event_dims=[i % 22],
            kind="event",
            n_try=i,
            n_succ=0,
            born_iter=i,
        )
        bank.add(entry)

    assert len(bank.entries) == 10

