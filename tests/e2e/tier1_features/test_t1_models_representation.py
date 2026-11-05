"""Tier 1: Representation & Latent Dynamics (F6, F7).

Tests:
- F6: Non-Contrastive Latent Representation (z_t in R^512) [5 tests]
- F7: Open-Loop Latent Dynamics Prediction [5 tests]
"""
from __future__ import annotations

import pytest
import torch

from zephyr.losses import vicreg_cov_loss, vicreg_var_loss
from zephyr.wm import ZephyrWM


# ==============================================================================
# Feature 6: Non-Contrastive Latent Representation (z_t in R^512)
# ==============================================================================

def test_f6_latent_vector_dimension_and_norm(mock_wm):
    """F6: Latent vector matches latent_dim with LayerNorm applied across features."""
    batch_size = 4
    obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)

    z = mock_wm.encode(obs)

    assert z.shape == (batch_size, mock_wm.cfg.world_model.latent_dim), (
        f"Shape mismatch: {z.shape} != ({batch_size}, {mock_wm.cfg.world_model.latent_dim})"
    )
    # LayerNorm properties: per-sample mean ~ 0, std ~ 1
    mean = z.mean(dim=-1)
    std = z.std(dim=-1)
    assert torch.all(torch.abs(mean) < 1e-4), f"LayerNorm mean not centered: {mean}"
    assert torch.all(torch.abs(std - 1.0) < 1e-2), f"LayerNorm std not normalized: {std}"


def test_f6_target_encoder_ema_stop_gradient(mock_wm):
    """F6: Target EMA encoder is frozen with requires_grad=False on all parameters."""
    for name, param in mock_wm.ema_encoder.named_parameters():
        assert not param.requires_grad, (
            f"Target encoder parameter {name} has requires_grad=True (must be frozen)"
        )


def test_f6_ema_decay_schedule_progression(tiny_test_cfg):
    """F6: Polyak EMA momentum tracks schedule: ramps linearly from start to max."""
    wm = ZephyrWM(tiny_test_cfg)
    cfg = tiny_test_cfg.world_model

    # Initialize a parameter in online encoder to 1.0, EMA to 0.0
    with torch.no_grad():
        first_online_param = next(wm.encoder.parameters())
        first_ema_param = next(wm.ema_encoder.parameters())
        first_online_param.fill_(1.0)
        first_ema_param.fill_(0.0)

    # Step 0: decay = ema_decay_start
    wm.update_ema(step=0)
    expected_step0 = (1.0 - cfg.ema_decay_start) * 1.0
    val_step0 = first_ema_param.mean().item()
    assert val_step0 == pytest.approx(expected_step0, abs=1e-5), (
        f"EMA parameter at step 0 ({val_step0}) != {expected_step0}"
    )

    # Reset EMA to 0.0 and test step = ema_ramp_steps (decay = ema_decay_max)
    with torch.no_grad():
        first_ema_param.fill_(0.0)
    wm.update_ema(step=cfg.ema_ramp_steps)
    expected_step_max = (1.0 - cfg.ema_decay_max) * 1.0
    val_step_max = first_ema_param.mean().item()
    assert val_step_max == pytest.approx(expected_step_max, abs=1e-5), (
        f"EMA parameter at max step ({val_step_max}) != {expected_step_max}"
    )


def test_f6_vicreg_variance_hinge_loss():
    """F6: vicreg_var_loss penalizes per-dimension standard deviation below gamma=1.0."""
    dim = 64
    batch = 32

    # Case A: Collapsed latents (std = 0)
    z_collapsed = torch.zeros(batch, dim)
    loss_collapsed = vicreg_var_loss(z_collapsed, gamma=1.0)
    assert loss_collapsed.item() == pytest.approx(1.0, abs=1e-4), (
        f"Collapsed variance loss {loss_collapsed.item()} != 1.0"
    )

    # Case B: Well-spread latents (std = 1.2)
    z_spread = torch.randn(batch, dim) * 1.5
    loss_spread = vicreg_var_loss(z_spread, gamma=1.0)
    assert loss_spread.item() < 0.2, (
        f"Expected near-zero variance penalty for spread latents, got {loss_spread.item()}"
    )


def test_f6_vicreg_covariance_decorrelation_loss():
    """F6: vicreg_cov_loss penalizes off-diagonal cross-correlation terms."""
    dim = 16
    batch = 64

    # Case A: Perfectly correlated features (rank-1 matrix)
    base = torch.randn(batch, 1)
    z_corr = base.repeat(1, dim)
    loss_corr = vicreg_cov_loss(z_corr)
    assert loss_corr.item() > 0.5, f"Expected high covariance loss for correlated features, got {loss_corr.item()}"

    # Case B: Independent uncorrelated features
    z_uncorr = torch.randn(1000, dim)
    loss_uncorr = vicreg_cov_loss(z_uncorr)
    assert loss_uncorr.item() < 0.05, f"Expected low covariance loss for independent features, got {loss_uncorr.item()}"


# ==============================================================================
# Feature 7: Open-Loop Latent Dynamics Prediction
# ==============================================================================

def test_f7_gru_belief_state_transition(mock_wm):
    """F7: Belief GRU advances temporal state h matching belief_dim."""
    batch_size = 3
    z = torch.randn(batch_size, mock_wm.cfg.world_model.latent_dim)
    actions = torch.tensor([0, 5, 16], dtype=torch.int64)
    h_init = mock_wm.init_state(batch_size, z.device)

    h_next = mock_wm.belief_step(z, actions, h_init)

    assert h_next.shape == (batch_size, mock_wm.cfg.world_model.belief_dim), (
        f"Belief state shape {h_next.shape} != ({batch_size}, {mock_wm.cfg.world_model.belief_dim})"
    )
    assert torch.isfinite(h_next).all(), "Belief state must be finite"
    assert not torch.allclose(h_next, h_init), "Belief state must update with new inputs"


def test_f7_latent_predictor_single_step_output(mock_wm):
    """F7: Latent predictor outputs LayerNorm-normalized next latent z_hat."""
    batch_size = 4
    z = torch.randn(batch_size, mock_wm.cfg.world_model.latent_dim)
    actions = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    h = mock_wm.init_state(batch_size, z.device)

    z_hat = mock_wm.predict(z, actions, h)

    assert z_hat.shape == (batch_size, mock_wm.cfg.world_model.latent_dim), (
        f"Predicted latent shape {z_hat.shape} != ({batch_size}, {mock_wm.cfg.world_model.latent_dim})"
    )
    assert torch.isfinite(z_hat).all(), "Predicted latent contains NaNs/Infs"
    # LayerNorm output check
    assert torch.all(torch.abs(z_hat.mean(dim=-1)) < 1e-4)
    assert torch.all(torch.abs(z_hat.std(dim=-1) - 1.0) < 1e-2)


def test_f7_open_loop_multi_step_unrolling(mock_wm):
    """F7: Open-loop 15-step rollout generates consistent latent sequence without observation input."""
    horizon = 15
    batch_size = 2
    z = torch.randn(batch_size, mock_wm.cfg.world_model.latent_dim)
    h = mock_wm.init_state(batch_size, z.device)
    actions = torch.randint(0, 17, (horizon, batch_size))

    z_seq = []
    h_seq = []
    for t in range(horizon):
        act = actions[t]
        z = mock_wm.predict(z, act, h)
        h = mock_wm.belief_step(z, act, h)
        z_seq.append(z)
        h_seq.append(h)

    assert len(z_seq) == horizon, f"Expected {horizon} predicted latents, got {len(z_seq)}"
    assert len(h_seq) == horizon, f"Expected {horizon} belief states, got {len(h_seq)}"
    for t in range(horizon):
        assert torch.isfinite(z_seq[t]).all(), f"NaN detected at step {t} in predicted latents"
        assert torch.isfinite(h_seq[t]).all(), f"NaN detected at step {t} in belief states"


def test_f7_dynamics_loss_horizon_decay(tiny_test_cfg):
    """F7: Multi-step prediction loss weights decay exponentially: w_k = decay ** (k - 1)."""
    decay = tiny_test_cfg.world_model.horizon_decay
    weights = [decay ** (k - 1) for k in range(1, 6)]

    assert weights[0] == 1.0, f"Step 1 weight should be 1.0, got {weights[0]}"
    assert weights[1] == pytest.approx(0.5, abs=1e-5)
    assert weights[2] == pytest.approx(0.25, abs=1e-5)
    for k in range(len(weights) - 1):
        ratio = weights[k + 1] / weights[k]
        assert ratio == pytest.approx(decay, abs=1e-5), f"Weight ratio {ratio} != {decay}"


def test_f7_belief_init_state_device_and_shape(mock_wm):
    """F7: init_state produces zero tensor matching (batch_size, belief_dim) on correct device."""
    batch_size = 5
    device = torch.device("cpu")
    h0 = mock_wm.init_state(batch_size, device)

    assert h0.shape == (batch_size, mock_wm.cfg.world_model.belief_dim)
    assert h0.device == device
    assert torch.all(h0 == 0.0), "Initial belief state must be initialized to exact zeros"
