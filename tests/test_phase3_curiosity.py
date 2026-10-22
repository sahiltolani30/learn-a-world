"""
Phase 3 Mac CPU Smoke Tests — Plan2Explore Curiosity Pipeline.

Covers:
  1. test_plan2explore_forward_shapes        — K predictors return [K, B, latent_dim]
  2. test_plan2explore_bonus_positive        — bonus > 0 with independent random inits
  3. test_plan2explore_predictor_loss_grad   — loss differentiable, .grad not None
  4. test_collector_v3_smoke                 — 2 env steps collected, shard written, manifest updated
  5. test_manifest_resume_logic              — load existing manifest, correct seed skip
  6. test_curiosity_guided_plan_smoke        — 1 planning call with augmented reward, action in [0,16]

All tests must pass on Mac CPU before any Colab run.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models.curiosity import Plan2ExploreCuriosity

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

LATENT_DIM = 128
RECURRENT_DIM = 256
NUM_ACTIONS = 17
B = 4  # batch size for tests


@pytest.fixture(scope="module")
def curiosity_module():
    """Shared Plan2ExploreCuriosity instance (3 predictors, CPU)."""
    return Plan2ExploreCuriosity(
        latent_dim=LATENT_DIM,
        recurrent_dim=RECURRENT_DIM,
        num_predictors=3,
        num_actions=NUM_ACTIONS,
    ).cpu()


@pytest.fixture
def zha(curiosity_module):
    """Random z, h, action tensors on CPU."""
    z = torch.randn(B, LATENT_DIM)
    h = torch.randn(B, RECURRENT_DIM)
    a = torch.randint(0, NUM_ACTIONS, (B,))
    return z, h, a


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Forward shapes
# ──────────────────────────────────────────────────────────────────────────────

def test_plan2explore_forward_shapes(curiosity_module, zha):
    """K predictors must return tensor of shape [K, B, latent_dim]."""
    z, h, a = zha
    preds = curiosity_module.forward(z, h, a)
    assert preds.shape == (3, B, LATENT_DIM), (
        f"Expected [3, {B}, {LATENT_DIM}], got {preds.shape}"
    )
    assert preds.dtype == torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Curiosity bonus is positive
# ──────────────────────────────────────────────────────────────────────────────

def test_plan2explore_bonus_positive(curiosity_module, zha):
    """With independently-initialized random weights, bonus must be > 0."""
    z, h, a = zha
    bonus = curiosity_module.curiosity_bonus(z, h, a)
    assert bonus.shape == (B,), f"Expected [B={B}], got {bonus.shape}"
    # All bonuses should be strictly positive (random inits → disagreement)
    assert (bonus > 0).all(), f"Expected all bonuses > 0, got min={bonus.min().item():.6f}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Predictor loss is differentiable
# ──────────────────────────────────────────────────────────────────────────────

def test_plan2explore_predictor_loss_grad(curiosity_module, zha):
    """Loss must be differentiable; all predictor parameter grads must be non-None."""
    curiosity_module.train()
    # Fresh module to avoid interference with module fixture
    fresh = Plan2ExploreCuriosity(
        latent_dim=LATENT_DIM,
        recurrent_dim=RECURRENT_DIM,
        num_predictors=3,
        num_actions=NUM_ACTIONS,
    ).cpu()
    z, h, a = zha
    z_target = torch.randn(B, LATENT_DIM)

    loss = fresh.predictor_loss(z, h, a, z_target)
    assert loss.requires_grad, "Loss must require grad"
    loss.backward()

    for name, param in fresh.named_parameters():
        assert param.grad is not None, f"Gradient is None for param: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Collector smoke — 2 env steps, shard written, manifest updated
# ──────────────────────────────────────────────────────────────────────────────

def test_collector_v3_smoke(tmp_path):
    """
    Smoke-test the collector: run ~2 env steps, write a shard, update manifest.
    Uses the actual Crafter env on CPU with the v3 checkpoint.
    """
    from training.collector_v3 import CollectorV3

    ckpt_path = repo_root / "checkpoints" / "jepa_crafter_v3_phase2.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    drive_dir = tmp_path / "smoke_collect"
    collector = CollectorV3(
        model_checkpoint=str(ckpt_path),
        curiosity_checkpoint=None,
        drive_dir=str(drive_dir),
        tranche_id=1,
        max_steps_per_ep=10,       # very short episodes for smoke test
        shard_every_transitions=50, # tiny shard threshold
        lambda_task=0.3,
        lambda_cur=1.0,
        device="cpu",
    )

    # Collect just 2 seeds (2 very short episodes)
    summary = collector.collect(
        target_new_transitions=30,
        seed_start=5000,
    )

    # Shard written
    shards = list(drive_dir.glob("shard_curiosity_v3_t01_*.npz"))
    assert len(shards) >= 1, "Expected at least one shard written"

    # Manifest updated
    manifest_path = drive_dir / "dataset_index.json"
    assert manifest_path.exists(), "dataset_index.json not written"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["total_transitions"] > 0, "Manifest has 0 transitions"
    assert "shards" in manifest and len(manifest["shards"]) > 0

    # Shard loadable
    shard_data = np.load(str(shards[0]), allow_pickle=True)
    for key in ("obs", "actions", "rewards", "dones", "events", "mask_before", "episode_lengths"):
        assert key in shard_data, f"Missing key '{key}' in shard"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Manifest resume logic — seed skip
# ──────────────────────────────────────────────────────────────────────────────

def test_manifest_resume_logic(tmp_path):
    """
    Write a manifest claiming 500 transitions already collected starting at seed 5000.
    Verify CollectorV3 resumes from the correct next seed.
    """
    from training.collector_v3 import CollectorV3

    ckpt_path = repo_root / "checkpoints" / "jepa_crafter_v3_phase2.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    drive_dir = tmp_path / "resume_test"
    drive_dir.mkdir(parents=True)

    # Craft a fake shard
    fake_shard_name = "shard_curiosity_v3_t01_0000.npz"
    fake_shard_path = drive_dir / fake_shard_name
    T = 500
    np.savez_compressed(
        str(fake_shard_path),
        obs=np.zeros((T, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(T, dtype=np.int64),
        rewards=np.zeros(T, dtype=np.float32),
        dones=np.zeros(T, dtype=np.bool_),
        events=np.zeros((T, 22), dtype=np.uint8),
        mask_before=np.zeros((T, 22), dtype=np.uint8),
        episode_lengths=np.array([100, 100, 100, 100, 100], dtype=np.int64),
        sources=np.array(["curiosity_v3"] * 5, dtype=object),
    )

    # Write manifest with 500 transitions and shard info
    manifest = {
        "total_transitions": T,
        "total_episodes": 5,
        "shards": [
            {
                "shard_name": fake_shard_name,
                "tranche": 1,
                "source_policy": "curiosity_v3",
                "seed_range": [5000, 5004],
                "transition_count": T,
                "per_achievement_event_counts": {k: 0 for k in [
                    "collect_coal", "collect_diamond", "collect_drink",
                    "collect_iron", "collect_sapling", "collect_stone",
                    "collect_wood", "defeat_skeleton", "defeat_zombie",
                    "eat_cow", "eat_plant", "make_iron_pickaxe",
                    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
                    "make_wood_pickaxe", "make_wood_sword", "place_furnace",
                    "place_plant", "place_stone", "place_table", "wake_up",
                ]},
            }
        ],
    }
    (drive_dir / "dataset_index.json").write_text(json.dumps(manifest, indent=2))

    collector = CollectorV3(
        model_checkpoint=str(ckpt_path),
        curiosity_checkpoint=None,
        drive_dir=str(drive_dir),
        tranche_id=1,
        max_steps_per_ep=5,
        shard_every_transitions=50,
        device="cpu",
    )

    # The resume seed should start AFTER the 5 episodes already recorded → seed 5005
    assert collector.resume_seed == 5005, (
        f"Expected resume from seed 5005, got {collector.resume_seed}"
    )
    assert collector.already_collected == T, (
        f"Expected {T} already_collected, got {collector.already_collected}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Curiosity-augmented planning call
# ──────────────────────────────────────────────────────────────────────────────

def test_curiosity_guided_plan_smoke():
    """
    1 planning call with curiosity-augmented reward.
    Action must be in valid range [0, 16].
    """
    from training.collector_v3 import CuriosityAugmentedPlanner

    ckpt_path = repo_root / "checkpoints" / "jepa_crafter_v3_phase2.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    curiosity = Plan2ExploreCuriosity(
        latent_dim=128,
        recurrent_dim=256,
        num_predictors=3,
        num_actions=17,
    ).cpu()

    planner = CuriosityAugmentedPlanner(
        model_checkpoint=str(ckpt_path),
        curiosity=curiosity,
        lambda_task=0.3,
        lambda_cur=1.0,
        horizon=3,           # very short for smoke test
        num_candidates=8,    # small for speed
        cem_iterations=1,
        device="cpu",
    )

    # Random observation
    obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    action, debug = planner.plan(obs)

    assert 0 <= action <= 16, f"Action {action} out of valid range [0, 16]"
    assert "curiosity_bonus_mean" in debug, "Debug info missing curiosity_bonus_mean"
    assert debug["curiosity_bonus_mean"] >= 0.0
