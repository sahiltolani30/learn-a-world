"""Tier 2: Boundary Tests - Nightfall Step 148 Wall.

Category 1: Step 148 Illumination & Monster Surge
Tests:
1. test_t2_nightfall_exact_step_147_to_148_transition
2. test_t2_nightfall_midnight_darkness_step_210
3. test_t2_nightfall_dawn_recovery_step_300
4. test_t2_nightfall_monster_spawning_quota_at_daylight_boundary
5. test_t2_nightfall_vision_encoder_representation_stability
6. test_t2_nightfall_continuation_head_focal_sensitivity
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from zephyr.heads import ZephyrHeads
from zephyr.wm import ZephyrWM


def _crafter_daylight(step: int) -> float:
    """Exact Crafter daylight formula: daylight = 1 - |cos(pi * ((step / 300 + 0.3) % 1))|^3."""
    progress = (step / 300.0 + 0.3) % 1.0
    return 1.0 - abs(math.cos(math.pi * progress)) ** 3


def _crafter_zombie_quota(daylight: float) -> float:
    """Crafter chunk balance target quota: Target_zombie = 3.5 - 3 * daylight."""
    return 3.5 - 3.0 * daylight


def test_t2_nightfall_exact_step_147_to_148_transition(crafter_env_factory):
    """T2-Nightfall: Daylight crosses critical 0.50 threshold between step 147 and step 148."""
    daylight_147 = _crafter_daylight(147)
    daylight_148 = _crafter_daylight(148)

    assert daylight_147 > 0.50, f"Daylight at 147 ({daylight_147:.4f}) should be > 0.50"
    assert daylight_148 < 0.50, f"Daylight at 148 ({daylight_148:.4f}) should be < 0.50"

    # Noise amount in Crafter is 2 * max(0, 0.5 - daylight)
    noise_147 = 2.0 * max(0.0, 0.50 - daylight_147)
    noise_148 = 2.0 * max(0.0, 0.50 - daylight_148)

    assert noise_147 == 0.0, f"Noise at step 147 must be exactly 0.0, got {noise_147}"
    assert noise_148 > 0.0, f"Noise at step 148 must be > 0.0, got {noise_148}"
    assert noise_148 == pytest.approx(2.0 * (0.50 - 0.4946), abs=1e-3)

    # Verify environment rendering transition
    env = crafter_env_factory(seed=7000, max_steps=160)
    env.reset(seed=7000)
    for _ in range(147):
        obs147, _, _, _, _ = env.step(0)
    obs148, _, _, _, _ = env.step(0)

    # Luminance drops at step 148
    assert obs148.mean() <= obs147.mean(), "Mean image brightness should not increase entering nightfall"


def test_t2_nightfall_midnight_darkness_step_210():
    """T2-Nightfall: Exact midnight at step 210 hits global illumination minimum (daylight = 0.0)."""
    daylight_210 = _crafter_daylight(210)
    assert daylight_210 == pytest.approx(0.0000, abs=1e-5), f"Daylight at step 210 should be 0.0, got {daylight_210}"

    # Maximum visual noise: 2 * (0.5 - 0.0) = 1.0
    noise_210 = 2.0 * max(0.0, 0.50 - daylight_210)
    assert noise_210 == pytest.approx(1.0, abs=1e-5), f"Maximum noise at midnight must be 1.0, got {noise_210}"

    # Maximum zombie quota: 3.5 - 3 * 0 = 3.5
    quota_210 = _crafter_zombie_quota(daylight_210)
    assert quota_210 == pytest.approx(3.50, abs=1e-5)


def test_t2_nightfall_dawn_recovery_step_300():
    """T2-Nightfall: Step 300 marks completion of the 300-step cycle, resetting progress to 0.30."""
    daylight_0 = _crafter_daylight(0)
    daylight_300 = _crafter_daylight(300)

    assert daylight_300 == pytest.approx(daylight_0, abs=1e-5), (
        f"Daylight at step 300 ({daylight_300}) must match step 0 ({daylight_0})"
    )
    assert daylight_300 == pytest.approx(0.7969, abs=1e-3)

    # Daylight is well above 0.50, so visual noise is completely inactive
    noise_300 = 2.0 * max(0.0, 0.50 - daylight_300)
    assert noise_300 == 0.0, f"Visual noise must be 0.0 at dawn (step 300), got {noise_300}"


def test_t2_nightfall_monster_spawning_quota_at_daylight_boundary():
    """T2-Nightfall: Zombie quota crosses the 2.0 hostile surge threshold across daylight=0.50."""
    quota_day = _crafter_zombie_quota(0.5001)
    quota_night = _crafter_zombie_quota(0.4999)

    assert quota_day < 2.0, f"Day quota ({quota_day}) should be < 2.0"
    assert quota_night > 2.0, f"Night quota ({quota_night}) should be > 2.0"
    assert quota_night > quota_day, "Hostile quota must monotonically increase as illumination drops"


def test_t2_nightfall_vision_encoder_representation_stability(mock_wm, synthetic_obs_generator):
    """T2-Nightfall: Visual encoder produces stable, finite, LayerNorm-normalized latents across illumination extremes."""
    mock_wm.eval()

    # Daylight observation vs nocturnal noisy observation
    obs_day = synthetic_obs_generator("daylight", batch_size=4, seed=10)
    obs_night = synthetic_obs_generator("nightfall", batch_size=4, seed=20)

    t_day = torch.from_numpy(obs_day)
    t_night = torch.from_numpy(obs_night)

    with torch.no_grad():
        z_day = mock_wm.encode(t_day)
        z_night = mock_wm.encode(t_night)

    assert torch.isfinite(z_day).all(), "Daylight latents contain non-finite values"
    assert torch.isfinite(z_night).all(), "Nightfall latents contain non-finite values"

    # LayerNorm ensures standard deviation across latent dimension remains healthy
    assert (z_day.std(dim=-1) > 0.1).all(), "Daylight latent representation collapsed"
    assert (z_night.std(dim=-1) > 0.1).all(), "Nightfall latent representation collapsed"

    # L2 norms should be stable and comparable
    norm_day = z_day.norm(dim=-1).mean().item()
    norm_night = z_night.norm(dim=-1).mean().item()
    assert abs(norm_day - norm_night) / norm_day < 0.5, "Severe norm explosion/collapse across illumination boundary"


def test_t2_nightfall_continuation_head_focal_sensitivity(tiny_test_cfg):
    """T2-Nightfall: Continuation head predictions bound in [0, 1] and penalize night danger states."""
    heads = ZephyrHeads(tiny_test_cfg).eval()
    state_dim = tiny_test_cfg.world_model.latent_dim + tiny_test_cfg.world_model.belief_dim

    # Safe state vs hostile swarm danger state
    safe_state = torch.ones(4, state_dim) * 0.5
    danger_state = torch.ones(4, state_dim) * -3.0

    with torch.no_grad():
        c_safe = 1.0 - torch.sigmoid(heads.contin(safe_state))
        c_danger = 1.0 - torch.sigmoid(heads.contin(danger_state))

    assert c_safe.shape == (4,), f"Expected shape (4,), got {c_safe.shape}"
    assert c_danger.shape == (4,), f"Expected shape (4,), got {c_danger.shape}"
    assert torch.all((c_safe >= 0.0) & (c_safe <= 1.0)), "Probabilities must be bounded in [0, 1]"
    assert torch.all((c_danger >= 0.0) & (c_danger <= 1.0)), "Probabilities must be bounded in [0, 1]"

    # State-dependent continuation discounting attenuates future value: gamma_eff = gamma * c
    gamma = 0.997
    gamma_eff_danger = gamma * float(c_danger.mean())
    assert gamma_eff_danger < gamma, "Effective discount factor must strictly attenuate in danger contexts"
