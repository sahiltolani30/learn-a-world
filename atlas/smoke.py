"""ATLAS local end-to-end smoke test.

Runs 2 episodes (seeds 6150/6151, max 200 steps) with AtlasAgent.
Asserts:
- Episodes complete successfully.
- Telemetry is populated on every step.
- Beliefs are monotone (m_bin and m_soft never decrease).
- Shard is written and verifies round-trip through ShardLoader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.collect import EpisodeCollector, ShardLoader
from atlas.config import AtlasConfig
from envs.crafter_wrapper import CrafterEnv


def run_smoke():
    print("=== [ATLAS Smoke Test] Starting ===")
    cfg = AtlasConfig()
    cfg.collect.max_steps = 200

    backbone = FrozenBackbone(config=cfg)
    agent = AtlasAgent(backbone=backbone, cfg=cfg)

    # 1. Run 2 episodes manually to inspect step-by-step telemetry and belief monotonicity
    test_seeds = [6150, 6151]
    for seed in test_seeds:
        print(f"Running episode seed={seed}...")
        env = CrafterEnv(seed=seed)
        obs, reset_info = env.reset(seed=seed)
        agent.reset_episode(seed=seed)

        prev_m_soft = agent.belief.m_soft.copy()
        prev_m_bin = agent.belief.m_bin.copy()
        reward_prev = 0.0

        for step in range(200):
            obs_uint8 = np.asarray(obs, dtype=np.uint8)
            action, telemetry = agent.act(obs_uint8, reward_prev, step)

            # Assert telemetry keys present
            for k in ["mode", "goal_score", "arrival_cos", "psi", "risk_intended", "override_fired", "r_int"]:
                assert k in telemetry, f"Missing telemetry key: {k}"
            assert telemetry["mode"] in ["PROBE", "NAVIGATE", "FALLBACK"], f"Unexpected mode: {telemetry['mode']}"

            # Assert belief monotonicity
            cur_m_soft = agent.belief.m_soft
            cur_m_bin = agent.belief.m_bin
            assert np.all(cur_m_soft >= prev_m_soft - 1e-6), "m_soft decreased!"
            assert np.all(cur_m_bin >= prev_m_bin - 1e-6), "m_bin decreased!"
            prev_m_soft = cur_m_soft.copy()
            prev_m_bin = cur_m_bin.copy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            discount = float(info.get("discount", 0.0 if terminated else 1.0))
            agent.observe_result(next_obs, reward, terminated, discount)

            obs = next_obs
            reward_prev = float(reward)

            if terminated or truncated:
                print(f"Episode seed={seed} terminated at step {step + 1}")
                break

    # 2. Test EpisodeCollector shard output
    with tempfile.TemporaryDirectory() as tmp_dir:
        collector = EpisodeCollector(agent=agent, out_dir=tmp_dir, cfg=cfg.collect)
        shard_path = collector.run_episodes(n=2, seed_base=6150, iter_idx=0)
        assert Path(shard_path).exists(), f"Shard was not created: {shard_path}"

        loaded = ShardLoader.load_shard(shard_path)
        assert len(loaded["episode_lengths"]) == 2
        assert len(loaded["obs"]) == sum(loaded["episode_lengths"])
        assert loaded["obs"].dtype == np.uint8
        assert loaded["actions"].dtype == np.int64
        print(f"Shard verified successfully with {len(loaded['obs'])} total steps.")

    print("=== [ATLAS Smoke Test] PASSED successfully! ===")


if __name__ == "__main__":
    run_smoke()
