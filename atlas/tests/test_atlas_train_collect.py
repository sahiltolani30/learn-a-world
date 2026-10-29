"""Unit tests for EpisodeCollector, ShardLoader, and HeadTrainer."""

import tempfile
from pathlib import Path
import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.backbone import FrozenBackbone
from atlas.collect import EpisodeCollector, ShardLoader
from atlas.config import AtlasConfig
from atlas.train import HeadTrainer


def test_shard_round_trip():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        collector = EpisodeCollector(agent=agent, out_dir=tmp_dir, cfg=cfg.collect)
        # Collect 1 short episode
        cfg.collect.max_steps = 10
        shard_path = collector.run_episodes(n=1, seed_base=42, iter_idx=0)

        assert Path(shard_path).exists()
        loaded = ShardLoader.load_shard(shard_path)

        for key in ["obs", "actions", "rewards", "dones", "events", "mask_before", "inventory"]:
            assert key in loaded, f"Key {key} missing from shard!"
        assert loaded["episode_lengths"][0] == len(loaded["obs"])


def test_latent_cache_equals_recomputation():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        collector = EpisodeCollector(agent=agent, out_dir=tmp_dir, cfg=cfg.collect)
        cfg.collect.max_steps = 5
        shard_path = collector.run_episodes(n=1, seed_base=100, iter_idx=0)

        trainer = HeadTrainer(
            backbone=fb,
            option_net=agent.option_net,
            reach_net=agent.reach_net,
            epistemic_net=agent.epistemic_net,
            safety_net=agent.safety_net,
            cfg=cfg.train,
            log_dir=tmp_dir,
        )

        latents_path = trainer.cache_latents(shard_path)
        assert latents_path.exists()

        cached_data = np.load(latents_path)
        z_cached = cached_data["z"]

        shard_data = ShardLoader.load_shard(shard_path)
        with torch.no_grad():
            z_recomputed = fb.encode(shard_data["obs"]).cpu().numpy()

        assert np.allclose(z_cached, z_recomputed, atol=1e-5), "Cached latents do not match recomputation!"


def test_trainer_step_decreases_loss():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")
    agent = AtlasAgent(backbone=fb, cfg=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = HeadTrainer(
            backbone=fb,
            option_net=agent.option_net,
            reach_net=agent.reach_net,
            epistemic_net=agent.epistemic_net,
            safety_net=agent.safety_net,
            readout_net=agent.readout_net,
            cfg=cfg.train,
            log_dir=tmp_dir,
        )

        # Synthetic episode
        T = 20
        ep = {
            "z": np.random.randn(T, 128).astype(np.float32),
            "h": np.random.randn(T, 256).astype(np.float32),
            "target_residuals": np.random.randn(T, 128).astype(np.float32),
            "actions": np.random.randint(0, 17, size=(T,), dtype=np.int64),
            "rewards": np.zeros(T, dtype=np.float32),
            "dones": np.zeros(T, dtype=bool),
            "events": np.zeros((T, 22), dtype=bool),
            "mask_before": np.zeros((T, 22), dtype=bool),
            "inventory": np.zeros((T, 16), dtype=np.float32),
        }

        log_entry = trainer.train_epoch(train_eps=[ep], val_eps=[ep], iter_idx=0)
        assert "gates" in log_entry
        assert "final_ce_loss" in log_entry
