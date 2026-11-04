"""Tests for MOSAIC Phase P2: Predictive Visual Belief & Dynamics."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch

from mosaic.src.checkpoint import load_checkpoint, save_atomic
from mosaic.src.models.encoder import VisualEncoder, AnchorDecoder
from mosaic.src.models.belief import RecurrentStochasticBelief
from mosaic.src.models.dynamics import ActionEmbedding, DynamicsEnsemble
from mosaic.src.models.world_model import MosaicWorldModel
from mosaic.src.models.diagnostics import (
    evaluate_action_conditioning,
    evaluate_heldout_prediction,
    evaluate_representation_collapse,
)


def test_visual_encoder_and_anchor_shapes() -> None:
    encoder = VisualEncoder(latent_dim=128)
    decoder = AnchorDecoder(latent_dim=128)

    # Test uint8 input
    x_uint8 = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    tokens, pooled = encoder(x_uint8)
    assert tokens.shape == (2, 64, 128)
    assert pooled.shape == (2, 128)

    # Test float input and reconstruction
    x_float = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    tokens_fl, pooled_fl = encoder(x_float)
    assert tokens_fl.shape == (2, 64, 128)
    assert pooled_fl.shape == (2, 128)

    recon = decoder(tokens_fl)
    assert recon.shape == (2, 3, 64, 64)
    assert recon.min() >= 0.0 and recon.max() <= 1.0


def test_recurrent_stochastic_belief_step_and_kl() -> None:
    B = 3
    belief = RecurrentStochasticBelief(
        visual_dim=128, action_dim=64, reward_dim=16, hidden_dim=256, stochastic_dim=32
    )
    h_0, z_0 = belief.initial_state(B)
    assert h_0.shape == (B, 256)
    assert z_0.shape == (B, 32)

    e_t = torch.randn(B, 128)
    a_prev = torch.randn(B, 64)
    r_prev = torch.randn(B, 1)

    # Step with posterior
    b_t, h_t, post_dist, prior_dist = belief.step(
        e_t=e_t, a_prev_emb=a_prev, r_prev=r_prev, h_prev=h_0, z_prev=z_0, use_posterior=True
    )
    assert b_t.shape == (B, 288)  # 256 + 32
    assert h_t.shape == (B, 256)

    # KL divergence test
    kl = post_dist.kl(prior_dist, free_bits=0.1)
    assert kl.shape == (B,)
    assert (kl >= 0.1).all()


def test_dynamics_ensemble_and_uncertainty() -> None:
    B = 4
    belief_dim = 288
    action_dim = 64
    ensemble = DynamicsEnsemble(
        belief_dim=belief_dim, action_dim=action_dim, num_members=3, hidden_dim=128, target_dim=128
    )

    belief = torch.randn(B, belief_dim)
    action_emb = torch.randn(B, action_dim)

    out = ensemble(belief, action_emb)
    assert out["next_belief_mean"].shape == (B, belief_dim)
    assert out["next_belief_members"].shape == (3, B, belief_dim)
    assert out["latent_pred_mean"].shape == (B, 128)
    assert out["latent_pred_members"].shape == (3, B, 128)
    assert out["reward_mean"].shape == (B, 1)
    assert out["continue_logit_mean"].shape == (B, 1)
    assert out["disagreement"].shape == (B,)
    assert (out["disagreement"] >= 0.0).all()


def test_action_conditioning_divergence() -> None:
    wm = MosaicWorldModel(num_actions=17, latent_dim=128, action_dim=64, hidden_dim=128, stochastic_dim=32)
    B = 2
    h_0, z_0, b_0 = wm.initial_belief(B)

    metrics = evaluate_action_conditioning(wm, b_0, action_token_1=1, action_token_2=2)
    assert metrics["is_action_sensitive"] is True
    assert metrics["belief_divergence"] > 0.0
    assert metrics["latent_divergence"] > 0.0


def test_world_model_loss_computation_and_backward() -> None:
    B, T = 2, 4
    wm = MosaicWorldModel(
        num_actions=17,
        latent_dim=64,
        action_dim=32,
        reward_dim=16,
        hidden_dim=128,
        stochastic_dim=32,
        num_ensemble_members=2,
    )

    obs_seq = torch.rand(B, T, 3, 64, 64)
    action_seq = torch.randint(0, 17, (B, T))
    reward_seq = torch.randn(B, T)
    continue_seq = torch.ones(B, T)

    losses = wm.compute_losses(
        obs_seq=obs_seq,
        action_seq=action_seq,
        reward_seq=reward_seq,
        continue_seq=continue_seq,
        lambda_dyn=1.0,
        lambda_anchor=0.1,
        lambda_r=1.0,
        lambda_c=1.0,
        lambda_kl=0.1,
    )

    assert "total_loss" in losses
    assert torch.isfinite(losses["total_loss"])

    # Backward pass verification
    losses["total_loss"].backward()
    for name, param in wm.encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Encoder param {name} missing grad"
            break


def test_target_encoder_ema_update() -> None:
    wm = MosaicWorldModel(latent_dim=64, hidden_dim=64, stochastic_dim=16, ema_decay=0.9)
    # Mutate online encoder weights
    with torch.no_grad():
        for p in wm.encoder.parameters():
            p.add_(1.0)

    target_param_before = next(wm.target_encoder.parameters()).clone()
    wm.update_target_network()
    target_param_after = next(wm.target_encoder.parameters())

    assert not torch.equal(target_param_before, target_param_after)


def test_representation_collapse_diagnostics() -> None:
    # 1. Non-collapsed diverse features
    features_healthy = torch.randn(100, 32)
    diag_healthy = evaluate_representation_collapse(features_healthy)
    assert diag_healthy["is_collapsed"] is False
    assert diag_healthy["std_mean"] > 0.5
    assert diag_healthy["effective_rank"] > 10.0

    # 2. Collapsed features (constant)
    features_collapsed = torch.ones(100, 32) * 5.0
    diag_collapsed = evaluate_representation_collapse(features_collapsed)
    assert diag_collapsed["is_collapsed"] is True
    assert diag_collapsed["std_mean"] < 1e-4


def test_world_model_checkpoint_roundtrip() -> None:
    wm = MosaicWorldModel(num_actions=17, latent_dim=64, hidden_dim=64, stochastic_dim=16)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "world_model.pt"
        save_atomic(
            ckpt_path,
            {
                "format": "mosaic_runtime_v1",
                "phase": "p2_world_model",
                "state_dict": wm.state_dict(),
            },
        )
        loaded = load_checkpoint(ckpt_path)
        assert loaded["format"] == "mosaic_runtime_v1"
        assert loaded["phase"] == "p2_world_model"

        wm2 = MosaicWorldModel(num_actions=17, latent_dim=64, hidden_dim=64, stochastic_dim=16)
        wm2.load_state_dict(loaded["state_dict"])

        # Check prediction equality
        dummy_b = torch.randn(2, wm.belief.belief_dim)
        dummy_a = torch.tensor([0, 1])
        with torch.no_grad():
            out1 = wm.imagine_transition(dummy_b, dummy_a)
            out2 = wm2.imagine_transition(dummy_b, dummy_a)
            assert torch.allclose(out1["latent_pred_mean"], out2["latent_pred_mean"])
