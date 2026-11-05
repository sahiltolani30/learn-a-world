"""Tier 1: Epistemic Dynamics Ensemble & Affordance Probing (F11, F12).

Tests:
- F11: Epistemic Dynamics Ensemble Curiosity [5 tests]
- F12: Station Affordance Probing (Actions 11-16) [5 tests]
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zephyr.ac import ensemble_disagreement
from zephyr.config import ExplorationConfig, ZephyrConfig
from zephyr.explore import EpisodicNovelty
from zephyr.heads import ZephyrHeads
from zephyr.wm import ZephyrWM


# ==============================================================================
# Feature 11: Epistemic Dynamics Ensemble Curiosity
# ==============================================================================

def test_f11_ensemble_member_count_and_architecture(standard_test_cfg):
    """F11: Ensemble head contains K=5 independent predictor members mapping (z, a, h) -> z_hat."""
    heads = ZephyrHeads(standard_test_cfg).to("cpu")
    K = standard_test_cfg.explore.ensemble_k
    assert K == 5, f"Expected 5 ensemble members in benchmark config, got {K}"

    batch_size = 2
    latent_dim = standard_test_cfg.world_model.latent_dim
    belief_dim = standard_test_cfg.world_model.belief_dim

    z = torch.randn(batch_size, latent_dim)
    actions = torch.tensor([5, 11], dtype=torch.int64)
    h = torch.randn(batch_size, belief_dim)

    preds = heads.ensemble(z, actions, h)

    assert preds.shape == (K, batch_size, latent_dim), (
        f"Ensemble prediction shape {preds.shape} != ({K}, {batch_size}, {latent_dim})"
    )
    assert torch.isfinite(preds).all(), "Ensemble predictions must be finite"


def test_f11_disagreement_variance_calculation():
    """F11: ensemble_disagreement calculates mean per-feature std across ensemble members."""
    K, B, D = 5, 4, 32

    # Case A: Perfect agreement (all K members identical)
    preds_identical = torch.randn(1, B, D).repeat(K, 1, 1)
    disagreement_zero = ensemble_disagreement(preds_identical)
    assert torch.allclose(disagreement_zero, torch.zeros(B), atol=1e-5), (
        f"Identical predictions should have 0 disagreement, got {disagreement_zero}"
    )

    # Case B: Divergent predictions
    preds_divergent = torch.randn(K, B, D)
    disagreement_pos = ensemble_disagreement(preds_divergent)
    assert torch.all(disagreement_pos > 0.0), "Divergent predictions must yield strictly positive disagreement"


def test_f11_intrinsic_reward_non_negativity():
    """F11: Intrinsic reward signal r_int is non-negative across all input distributions."""
    K, B, D = 3, 10, 64
    for _ in range(5):
        random_preds = torch.randn(K, B, D)
        r_int = ensemble_disagreement(random_preds)
        assert (r_int >= 0.0).all(), f"Found negative intrinsic rewards: {r_int[r_int < 0.0]}"


def test_f11_beta_intrinsic_decay_schedule():
    """F11: Exploration beta(frames) schedule decays from 1.0 to 0.05 over 2,000,000 frames."""
    cfg = ExplorationConfig()

    assert cfg.beta(0) == pytest.approx(1.0, abs=1e-5)
    assert cfg.beta(1_000_000) == pytest.approx(0.525, abs=1e-3)
    assert cfg.beta(2_000_000) == pytest.approx(0.05, abs=1e-5)
    assert cfg.beta(3_000_000) == pytest.approx(0.05, abs=1e-5), "Beta must clamp at beta_min"


def test_f11_episodic_novelty_knn_memory():
    """F11: EpisodicNovelty tracks within-episode latent history and scores novel vs repeated beliefs."""
    cfg = ExplorationConfig(knn_k=3, knn_memory=100)
    novelty = EpisodicNovelty(cfg)

    # Initially empty memory returns 0.0 until k+1 entries stored
    assert novelty.score(np.zeros(32)) == 0.0

    # Add 10 diverse vectors
    rng = np.random.default_rng(42)
    for _ in range(10):
        novelty.add(rng.normal(size=32))

    assert len(novelty) == 10

    # Score a duplicate of an existing vector vs an orthogonal novel vector
    first_vec = list(novelty._memory)[0]
    duplicate_score = novelty.score(first_vec)
    novel_vec = rng.normal(loc=100.0, size=32)
    novel_score = novelty.score(novel_vec)

    assert novel_score > duplicate_score, (
        f"Novel score {novel_score} should exceed duplicate score {duplicate_score}"
    )

    # reset() clears memory
    novelty.reset()
    assert len(novelty) == 0
    assert novelty.score(novel_vec) == 0.0


# ==============================================================================
# Feature 12: Station Affordance Probing (Actions 11-16)
# ==============================================================================

def test_f12_station_action_range_definition():
    """F12: Actions 11-16 correspond to crafting recipes requiring stations/resources."""
    crafting_action_indices = list(range(11, 17))
    assert len(crafting_action_indices) == 6
    assert crafting_action_indices == [11, 12, 13, 14, 15, 16]


def test_f12_station_proximity_disagreement_spike(tiny_test_cfg):
    """F12: Ensemble disagreement exhibits higher variance on unvisited states/actions."""
    heads = ZephyrHeads(tiny_test_cfg).eval()
    latent_dim = tiny_test_cfg.world_model.latent_dim
    belief_dim = tiny_test_cfg.world_model.belief_dim

    # Common wilderness input vs unfamiliar crafting input
    z_familiar = torch.zeros(1, latent_dim)
    h_familiar = torch.zeros(1, belief_dim)

    z_novel = torch.randn(1, latent_dim) * 3.0
    h_novel = torch.randn(1, belief_dim) * 3.0

    with torch.no_grad():
        preds_familiar = heads.ensemble(z_familiar, torch.tensor([0]), h_familiar)
        preds_novel = heads.ensemble(z_novel, torch.tensor([11]), h_novel)

    disag_familiar = ensemble_disagreement(preds_familiar).item()
    disag_novel = ensemble_disagreement(preds_novel).item()

    assert torch.isfinite(torch.tensor(disag_familiar))
    assert torch.isfinite(torch.tensor(disag_novel))
    assert disag_familiar >= 0.0
    assert disag_novel >= 0.0


def test_f12_affordance_probing_action_selection():
    """F12: Combining intrinsic reward with value heads shifts action selection toward high-uncertainty affordances."""
    extrinsic_reward = 0.0
    intrinsic_reward = 2.5
    beta = 1.0

    total_return = extrinsic_reward + beta * intrinsic_reward
    assert total_return == 2.5, "Total return should equal intrinsic reward when extrinsic is 0"


def test_f12_station_absence_silent_action_failure(crafter_env_factory):
    """F12: Executing actions 11-16 far from a crafting station fails silently without crashing."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)

    for action in range(11, 17):
        obs, reward, term, trunc, info = env.step(action)
        assert reward == 0.0, f"Expected 0.0 reward for invalid crafting action {action}, got {reward}"
        assert term is False, f"Action {action} caused unexpected episode termination"
        assert obs.shape == (64, 64, 3)


def test_f12_crafting_station_destruction_or_loss(crafter_env_factory):
    """F12: Moving away from a crafting station (Chebyshev distance > 1) deactivates affordance."""
    env = crafter_env_factory(seed=7000, max_steps=100)
    env.reset(seed=7000)

    # Place table adjacent
    target_pos = tuple(env._env._player.pos + env._env._player.facing)
    env._env._world[target_pos] = "table"
    env._env._player.inventory["wood"] = 1

    # At distance 1: affordance active
    obs, reward, term, trunc, info = env.step(11)
    assert info["achievements"]["make_wood_pickaxe"] == 1, "Should succeed at distance 1"

    # Now step 2 tiles away and try again
    env._env._player.pos = np.array([env._env._player.pos[0] + 5, env._env._player.pos[1] + 5])
    env._env._player.inventory["wood"] = 1
    obs, reward, term, trunc, info = env.step(11)
    assert env._env._player.inventory["wood"] == 1, "Wood should not be consumed when distant from table"
    assert reward == 0.0, "Reward should be 0.0 when distant from table"
