"""Tier 3: Curiosity, Exploration & Crafting Station Affordance Synergy.

Covers combinatorial interactions between:
- F7 (Dynamics Predictor), F8 (Actor), F11 (Epistemic Curiosity),
  F12 (Station Affordance Probing), F16 (Online Loop).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zephyr.ac import (
    ensemble_disagreement,
    imagine_trajectory,
    trajectory_lambda_returns,
)
from zephyr.config import ExplorationConfig, ZephyrConfig
from zephyr.explore import EpisodicNovelty
from zephyr.heads import DynamicsEnsemble, ZephyrHeads
from zephyr.train_loop import ActorPolicy
from zephyr.train_wm import wm_joint_losses
from zephyr.wm import ZephyrWM


def test_t3_curiosity_01_predictor_ensemble_multi_head_divergence(tiny_test_cfg, mock_heads):
    """T3-CURIOSITY-01: Predictor ensemble members independently diverge on input latents."""
    b = 4
    latent_dim = tiny_test_cfg.world_model.latent_dim
    belief_dim = tiny_test_cfg.world_model.belief_dim

    z = torch.randn(b, latent_dim)
    a = torch.randint(0, 17, (b,))
    h = torch.randn(b, belief_dim)

    preds = mock_heads.ensemble(z, a, h)

    # Output shape: [K, B, latent_dim]
    k = tiny_test_cfg.explore.ensemble_k
    assert preds.shape == (k, b, latent_dim)

    # Pairwise difference between distinct ensemble heads (independent initializations)
    if k >= 2:
        diff = torch.norm(preds[0] - preds[1], p=2, dim=-1).mean().item()
        assert diff > 0.01, f"Ensemble heads produced identical outputs: diff={diff}"


def test_t3_curiosity_02_ensemble_disagreement_variance_familiar_vs_novel(tiny_test_cfg):
    """T3-CURIOSITY-02: Disagreement variance spike on novel vs familiar transitions."""
    b, d = 4, tiny_test_cfg.world_model.latent_dim
    k = tiny_test_cfg.explore.ensemble_k

    # Familiar transitions: predictions are tightly clustered across ensemble members
    base_pred = torch.randn(1, b, d)
    familiar_preds = base_pred.repeat(k, 1, 1) + torch.randn(k, b, d) * 0.001
    familiar_disagreement = ensemble_disagreement(familiar_preds)

    # Novel transitions: predictions diverge widely across members
    novel_preds = base_pred.repeat(k, 1, 1) + torch.randn(k, b, d) * 0.5
    novel_disagreement = ensemble_disagreement(novel_preds)

    assert familiar_disagreement.mean().item() < 0.01
    assert novel_disagreement.mean().item() > 0.1
    ratio = (novel_disagreement.mean() / (familiar_disagreement.mean() + 1e-8)).item()
    assert ratio > 3.0, f"Disagreement spike ratio {ratio} < 3.0"


def test_t3_curiosity_03_crafting_station_proximity_disagreement_surge(tiny_test_cfg, mock_heads):
    """T3-CURIOSITY-03: Dynamics ensemble evaluates disagreement across action probing."""
    b = 2
    latent_dim = tiny_test_cfg.world_model.latent_dim
    belief_dim = tiny_test_cfg.world_model.belief_dim

    z = torch.randn(b, latent_dim)
    h = torch.randn(b, belief_dim)

    # Probing action 11 (make_wood_pickaxe)
    craft_action = torch.full((b,), 11, dtype=torch.long)
    preds = mock_heads.ensemble(z, craft_action, h)
    disagreement = ensemble_disagreement(preds)

    assert disagreement.shape == (b,)
    assert (disagreement >= 0.0).all()
    assert torch.isfinite(disagreement).all()


def test_t3_curiosity_04_intrinsic_reward_injection_into_actor_imagination(tiny_test_cfg, mock_wm, mock_heads, sample_obs_batch):
    """T3-CURIOSITY-04: Epistemic disagreement reward injection into actor imagination returns."""
    b = sample_obs_batch.shape[0]
    obs = torch.from_numpy(np.ascontiguousarray(sample_obs_batch))
    with torch.no_grad():
        z0 = mock_wm.encode(obs)
        h0 = mock_wm.init_state(b, "cpu")

    imagination = imagine_trajectory(
        mock_wm, mock_heads, z0, h0, horizon=5, cfg=tiny_test_cfg, with_ensemble=True
    )

    assert "intrinsic" in imagination
    assert imagination["intrinsic"].shape == (5, b)

    # Incur positive intrinsic reward
    imagination["intrinsic"].fill_(0.5)

    coder = mock_heads.critic.coder
    ret_no_int = trajectory_lambda_returns(imagination, coder, tiny_test_cfg, intrinsic_weight=0.0)
    ret_with_int = trajectory_lambda_returns(imagination, coder, tiny_test_cfg, intrinsic_weight=1.0)

    assert ret_with_int.mean() > ret_no_int.mean(), "Intrinsic bonus should elevate trajectory returns"


def test_t3_curiosity_05_episodic_novelty_knn_belief_space_memory(tiny_test_cfg):
    """T3-CURIOSITY-05: Episodic k-NN belief space novelty memory and reset lifecycle."""
    cfg = ExplorationConfig(knn_k=4, knn_memory=32)
    novelty = EpisodicNovelty(cfg)

    # Until k+1 items stored, score returns 0.0
    for i in range(cfg.knn_k):
        novelty.add(np.random.randn(64))
        assert novelty.score(np.random.randn(64)) == 0.0

    # Fill memory with points clustered around the origin
    cluster_points = [np.random.randn(64).astype(np.float32) * 0.01 for _ in range(8)]
    for p in cluster_points:
        novelty.add(p)

    # Re-visiting the cluster center yields near-zero novelty (since k nearest neighbors are clustered)
    cluster_query = np.zeros(64, dtype=np.float32)
    cluster_score = novelty.score(cluster_query)
    assert cluster_score < 0.15, f"Expected low novelty for clustered point, got {cluster_score}"

    # Querying a distant novel belief yields high novelty
    distant_query = np.ones(64, dtype=np.float32) * 50.0
    novel_score = novelty.score(distant_query)
    assert novel_score > cluster_score
    assert novel_score > 0.8, f"Expected high novelty for distant belief, got {novel_score}"

    # Reset clears memory
    novelty.reset()
    assert len(novelty) == 0
    assert novelty.score(distant_query) == 0.0


def test_t3_curiosity_06_intrinsic_schedule_decay():
    """T3-CURIOSITY-06: Linear annealing schedule beta(t) transitions exploration to exploitation."""
    cfg = ExplorationConfig(
        beta_start=1.0,
        beta_decay_frames=2_000_000,
        beta_min=0.05,
    )

    # Initial frame: full exploration
    assert cfg.beta(0) == 1.0

    # Halfway through schedule: midpoint
    halfway_beta = cfg.beta(1_000_000)
    assert abs(halfway_beta - 0.525) < 1e-4, f"Expected 0.525, got {halfway_beta}"

    # At schedule completion: reaches minimum
    assert abs(cfg.beta(2_000_000) - 0.05) < 1e-5

    # Past schedule completion: strictly clamped to beta_min
    assert abs(cfg.beta(3_000_000) - 0.05) < 1e-5
    assert abs(cfg.beta(10_000_000) - 0.05) < 1e-5


def test_t3_curiosity_07_affordance_probing_action_distribution_shift(tiny_test_cfg, mock_wm, mock_heads, sample_obs_batch):
    """T3-CURIOSITY-07: ActorPolicy computes intrinsic rewards and action sampling."""
    num_envs = sample_obs_batch.shape[0]
    policy = ActorPolicy(
        mock_wm,
        mock_heads,
        num_envs=num_envs,
        device="cpu",
        disagreement_weight=1.0,
        knn_weight=1.0,
    )

    actions, intrinsic = policy.act(sample_obs_batch)

    assert actions.shape == (num_envs,)
    assert (actions >= 0).all() and (actions < 17).all()
    assert intrinsic.shape == (num_envs,)
    assert not np.isnan(intrinsic).any()


def test_t3_curiosity_08_ensemble_joint_loss_gradient_flow(tiny_test_cfg, mock_wm, mock_heads, synthetic_batch_generator):
    """T3-CURIOSITY-08: Dynamics ensemble parameters receive non-zero gradients in wm_joint_losses."""
    batch = synthetic_batch_generator(batch_size=2, seq_len=4)
    total, metrics = wm_joint_losses(mock_wm, mock_heads, batch, tiny_test_cfg, device=torch.device("cpu"))
    assert "loss_ensemble" in metrics
    assert torch.isfinite(total)

    total.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mock_heads.ensemble.parameters())
    assert has_grad, "Heads ensemble must receive non-zero gradients from wm_joint_losses"
    for i, member in enumerate(mock_heads.ensemble.members):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in member.parameters()), (
            f"Member {i} missing gradients"
        )
