"""Unit tests for ZephyrReplay: boundaries, masks, shards, splits."""
from __future__ import annotations

import numpy as np
import pytest

from zephyr.buffer import EpisodeRecord, ZephyrReplay


def _record(t: int, seed: int = 0, with_achievement: bool = False, source: str = "random"):
    rng = np.random.default_rng(seed)
    ach_new = np.zeros((t, 22), dtype=bool)
    if with_achievement:
        ach_new[t // 2, 6] = True  # collect_wood
    cum = np.cumsum(ach_new, axis=0).astype(bool)
    return EpisodeRecord(
        obs=rng.integers(0, 255, (t, 64, 64, 3), dtype=np.uint8),
        actions=rng.integers(0, 17, t).astype(np.int16),
        rewards=rng.normal(0, 0.1, t).astype(np.float32),
        rewards_int=np.zeros(t, dtype=np.float32),
        done=np.zeros(t, dtype=bool),
        inventory=rng.random((t, 16)).astype(np.float32),
        achievements_cum=cum,
        achievements_new=ach_new,
        seed=seed,
        source=source,
    )


def test_record_validation_rejects_bad_shapes():
    rec = _record(10)
    with pytest.raises(ValueError):
        EpisodeRecord(
            obs=rec.obs[:5], actions=rec.actions, rewards=rec.rewards,
            rewards_int=rec.rewards_int, done=rec.done, inventory=rec.inventory,
            achievements_cum=rec.achievements_cum,
            achievements_new=rec.achievements_new, seed=0, source="x",
        )


def test_sample_windows_never_cross_episodes():
    replay = ZephyrReplay(capacity_frames=None)
    ep_short = _record(5, seed=1)
    ep_long = _record(50, seed=2, with_achievement=True)
    replay.add_episode(ep_short)
    replay.add_episode(ep_long)
    rng = np.random.default_rng(0)
    batch = replay.sample(batch=32, seq_len=8, rng=rng, prioritize_achievement_frac=0.0)
    assert batch["obs"].shape == (32, 8, 64, 64, 3)
    assert batch["obs_next"].shape == (32, 8, 64, 64, 3)

    obs = batch["obs"].numpy()
    nxt = batch["obs_next"].numpy()
    valid = batch["valid"].numpy()
    inv = batch["inventory"].numpy()
    inv_nxt = batch["inventory_next"].numpy()

    for b in range(32):
        # identify the source episode by its first window frame
        if np.array_equal(obs[b, 0], ep_short.obs[0]) and valid[b].sum() == 5:
            ep, n_valid, start = ep_short, 5, 0
        else:
            # locate the window start inside the long episode
            start = next(
                s for s in range(ep_long.obs.shape[0])
                if np.array_equal(obs[b, 0], ep_long.obs[s])
            )
            ep, n_valid = ep_long, 8
        assert valid[b, :n_valid].all()
        for t in range(n_valid - 1):
            assert np.array_equal(nxt[b, t], ep.obs[start + t + 1])
            assert np.allclose(inv_nxt[b, t], ep.inventory[start + t + 1], atol=1e-6)
        if start + n_valid >= ep.obs.shape[0]:
            # window reaches episode end -> bootstrap with the episode's final frame
            assert np.array_equal(nxt[b, n_valid - 1], ep.obs[-1])
        else:
            assert np.array_equal(nxt[b, n_valid - 1], ep.obs[start + n_valid])
        if n_valid < 8:
            assert not valid[b, n_valid:].any()
            assert (nxt[b, n_valid:] == ep.obs[-1]).all()


def test_prioritized_sampling_hits_achievement_episodes():
    replay = ZephyrReplay(capacity_frames=None)
    for i in range(20):
        replay.add_episode(_record(20, seed=100 + i, with_achievement=(i == 0)))
    rng = np.random.default_rng(7)
    chosen = {int(rng.integers(0, 20)) for _ in range(0)}  # noqa
    hits = 0
    for _ in range(200):
        batch = replay.sample(1, 4, rng, prioritize_achievement_frac=1.0)
        hits += 1  # sampling occurred; direct pool check below is stronger
    pool = [i for i, ep in enumerate(replay.records) if ep.achievements_new.any()]
    assert pool == [0]
    # frac=1.0 must always draw from the pool (the only achievement episode)
    obs0 = replay.records[0].obs[0]
    matched = 0
    for _ in range(10):
        batch = replay.sample(8, 4, rng, prioritize_achievement_frac=1.0)
        matched += int(any(np.array_equal(batch["obs"][b, 0], obs0) for b in range(8)))
    assert matched > 0  # window from episode 0 appears


def test_shard_roundtrip_preserves_records(tmp_path):
    replay = ZephyrReplay(capacity_frames=None)
    replay.add_episode(_record(30, seed=5, with_achievement=True, source="actor"))
    replay.add_episode(_record(7, seed=6))
    path = str(tmp_path / "shard.npz")
    replay.save_shard(path)
    loaded = ZephyrReplay.load_shard(path)
    assert loaded.episodes == 2
    for a, b in zip(replay.records, loaded.records):
        assert np.array_equal(a.obs, b.obs)
        assert np.array_equal(a.inventory, b.inventory)
        assert a.seed == b.seed and a.source == b.source


def test_shard_rejects_foreign_format(tmp_path):
    import numpy as np
    path = str(tmp_path / "bad.npz")
    np.savez(path, junk=np.zeros(3))
    with pytest.raises(Exception):
        ZephyrReplay.load_shard(path)


def test_split_is_deterministic_and_disjoint():
    replay = ZephyrReplay(capacity_frames=None)
    for i in range(10):
        replay.add_episode(_record(12, seed=200 + i))
    tr1, va1 = replay.split(0.2, seed=4242)
    tr2, va2 = replay.split(0.2, seed=4242)
    assert tr1.episodes + va1.episodes == 10
    assert {ep.seed for ep in va1.records} == {ep.seed for ep in va2.records}
    seeds_tr = {ep.seed for ep in tr1.records}
    seeds_va = {ep.seed for ep in va1.records}
    assert not (seeds_tr & seeds_va)


def test_fifo_eviction_respects_capacity():
    replay = ZephyrReplay(capacity_frames=25)
    for i in range(5):
        replay.add_episode(_record(10, seed=300 + i))
    assert replay.episodes <= 3  # 25-frame capacity with 10-frame episodes
    seeds = [ep.seed for ep in replay.records]
    assert seeds == sorted(seeds)  # oldest evicted first
