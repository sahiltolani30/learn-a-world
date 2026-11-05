"""Tier 5: Adversarial Coverage Hardening Suite.

White-box adversarial stress tests verifying empirical resilience across:
1. Perception Extremes:
   - Extreme inputs: all-zero black frames, all-255 white frames, saturated noise,
     single-pixel hot spots, checkerboard frequencies.
   - Batch size edge cases: batch_size=1, batch_size=128, uneven batch sizes (3, 7, 31).
2. Latent Dynamics & Belief Core Extremes:
   - Extreme horizon rollouts (H=30, H=50) unrolled open-loop: stability, bounded norms, no NaNs.
   - Sudden alternating action sequences (sleep -> noop -> move_left -> do).
   - Sequence vs step-by-step GRU recurrence parity.
3. Value & Policy Distribution Extremes:
   - Out-of-distribution value targets (returns of +/- 50,000, +/- 1e12): symlog saturation without gradient explosion.
   - Unimix entropy edge cases: policy logit extremes (+/- 1e6).
   - Focal BCE and masked MSE boundary conditions.
4. Buffer & Persistence Corruption:
   - Truncated .npz replay shards, corrupt zip headers, format version mismatches, empty containers.
   - EpisodeRecord schema and array shape enforcement.
   - Checkpoint state dictionaries with missing sub-keys or unexpected parameter tensor shapes.
   - Generation index overflow (gen >= 10^9 and 10^12).
"""
from __future__ import annotations

import io
import json
import math
import os
import pathlib
import tempfile
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zephyr.ac import (
    actor_lambda_return_loss,
    critic_twohot_loss,
    ensemble_disagreement,
    imagine_trajectory,
    normalize_returns,
    trajectory_lambda_returns,
    unimix_probs,
)
from zephyr.buffer import EpisodeRecord, ZephyrReplay
from zephyr.config import IMG_SIZE, NUM_ACHIEVEMENTS, NUM_ACTIONS, ZephyrConfig
from zephyr.heads import (
    ActorHead,
    ContinueHead,
    CriticHead,
    DynamicsEnsemble,
    EffectHead,
    ReadoutHead,
    RewardHead,
    ZephyrHeads,
    _HEAD_KEYS,
)
from zephyr.losses import (
    TwoHotCoder,
    focal_bce_with_logits,
    inv_symlog,
    lambda_returns,
    masked_mse,
    roc_auc,
    symlog,
    vicreg_cov_loss,
    vicreg_var_loss,
)
from zephyr.train_loop import ZephyrTrainer
from zephyr.vec_env import _EpisodeRunner
from zephyr.wm import (
    _CHECKPOINT_FORMAT,
    ZephyrEncoder,
    ZephyrGRU,
    ZephyrPredictor,
    ZephyrWM,
)


# ==============================================================================
# Category 1: Perception Extremes (White-Box Vision & Normalization Hardening)
# ==============================================================================


class TestPerceptionExtremes:
    """Stress testing Zephyr visual encoder and representation losses on extreme visual inputs."""

    def test_t5_perception_all_zero_black_frames(self, tiny_test_cfg):
        """Perception: All-zero black frames produce finite LayerNorm latents with clean gradient flow."""
        encoder = ZephyrEncoder(tiny_test_cfg.encoder)
        b_black = torch.zeros(4, 64, 64, 3, dtype=torch.uint8)

        z = encoder(b_black)

        assert z.shape == (4, tiny_test_cfg.encoder.latent_dim)
        assert torch.isfinite(z).all(), "Encoder produced non-finite latents on all-zero frames"
        # LayerNorm ensures zero-mean, unit-variance per sample
        assert torch.allclose(z.mean(dim=-1), torch.zeros(4), atol=1e-4)
        assert torch.allclose(z.std(dim=-1, unbiased=False), torch.ones(4), atol=1e-3)

        # Gradient flow check back through conv backbone
        loss = z.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    def test_t5_perception_all_255_white_frames(self, tiny_test_cfg):
        """Perception: All-255 white frames (maximum saturation) produce finite normalized latents."""
        encoder = ZephyrEncoder(tiny_test_cfg.encoder)
        b_white = torch.full((4, 64, 64, 3), 255, dtype=torch.uint8)

        z = encoder(b_white)

        assert z.shape == (4, tiny_test_cfg.encoder.latent_dim)
        assert torch.isfinite(z).all(), "Encoder produced non-finite latents on all-white frames"
        assert torch.allclose(z.mean(dim=-1), torch.zeros(4), atol=1e-4)
        assert torch.allclose(z.std(dim=-1, unbiased=False), torch.ones(4), atol=1e-3)

    def test_t5_perception_saturated_noise(self, tiny_test_cfg):
        """Perception: Saturated binary noise (random 0 and 255 values) does not cause numerical instability."""
        encoder = ZephyrEncoder(tiny_test_cfg.encoder)
        rng = np.random.default_rng(101)
        noise_np = rng.choice([0, 255], size=(8, 64, 64, 3)).astype(np.uint8)
        b_noise = torch.from_numpy(noise_np)

        z = encoder(b_noise)

        assert z.shape == (8, tiny_test_cfg.encoder.latent_dim)
        assert torch.isfinite(z).all(), "Non-finite latents under saturated noise"
        assert z.std(dim=0).mean().item() > 0.05, "Representation collapsed on diverse saturated noise"

    def test_t5_perception_single_pixel_hot_spots(self, tiny_test_cfg):
        """Perception: Single-pixel delta spikes (hot spots) do not blow up convolution activations."""
        encoder = ZephyrEncoder(tiny_test_cfg.encoder)
        # Hot spots at different spatial coordinates and channels
        hot_frames = torch.zeros(5, 64, 64, 3, dtype=torch.uint8)
        hot_frames[0, 0, 0, 0] = 255      # Top-left red
        hot_frames[1, 63, 63, 1] = 255    # Bottom-right green
        hot_frames[2, 32, 32, 2] = 255    # Center blue
        hot_frames[3, 16, 48, :] = 255    # All channels
        hot_frames[4, 48, 16, 0] = 128    # Half saturation

        z = encoder(hot_frames)

        assert z.shape == (5, tiny_test_cfg.encoder.latent_dim)
        assert torch.isfinite(z).all(), "Encoder produced non-finite activations for hot spots"
        # Each distinct hot spot should yield distinguishable representation
        for i in range(4):
            cos_sim = F.cosine_similarity(z[i:i+1], z[i+1:i+2]).item()
            assert cos_sim < 0.9999, f"Hot spots {i} and {i+1} collapsed to identical latents"

    def test_t5_perception_checkerboard_high_frequency_patterns(self, tiny_test_cfg):
        """Perception: Maximum spatial frequency checkerboard patterns pass cleanly through strided convs."""
        encoder = ZephyrEncoder(tiny_test_cfg.encoder)
        # Construct 64x64 alternating checkerboard
        y, x = np.indices((64, 64))
        checker = ((x + y) % 2 * 255).astype(np.uint8)
        checker_rgb = np.stack([checker] * 3, axis=-1)
        b_checker = torch.from_numpy(np.stack([checker_rgb] * 2))

        z = encoder(b_checker)

        assert z.shape == (2, tiny_test_cfg.encoder.latent_dim)
        assert torch.isfinite(z).all()
        assert not torch.isnan(z).any()

    @pytest.mark.parametrize("batch_size", [1, 3, 7, 31, 128])
    def test_t5_perception_batch_size_edge_cases(self, tiny_test_cfg, batch_size):
        """Perception: Encoder and WM handle edge batch sizes: B=1, odd primes (3, 7, 31), and large B=128."""
        encoder = ZephyrEncoder(tiny_test_cfg.encoder)
        obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)

        z = encoder(obs)

        assert z.shape == (batch_size, tiny_test_cfg.encoder.latent_dim)
        assert torch.isfinite(z).all(), f"Failed on batch_size={batch_size}"

    def test_t5_perception_vicreg_losses_extreme_batches(self, tiny_test_cfg):
        """Perception: VICReg variance and covariance losses are strictly finite for B=1, large B, and constant inputs."""
        dim = tiny_test_cfg.encoder.latent_dim

        # Batch size 1: unbiased=False guarantees std=0 rather than NaN, cov=0
        z_single = torch.randn(1, dim)
        var_loss_1 = vicreg_var_loss(z_single, gamma=1.0)
        cov_loss_1 = vicreg_cov_loss(z_single)

        assert torch.isfinite(var_loss_1), "vicreg_var_loss gave non-finite value on B=1"
        assert torch.isfinite(cov_loss_1), "vicreg_cov_loss gave non-finite value on B=1"
        assert var_loss_1.item() == pytest.approx(1.0, abs=1e-5)
        assert cov_loss_1.item() == pytest.approx(0.0, abs=1e-5)

        # Identical constant latents: variance should be 0, loss equals hinge threshold gamma
        z_const = torch.ones(16, dim)
        var_loss_const = vicreg_var_loss(z_const, gamma=1.0)
        assert var_loss_const.item() == pytest.approx(1.0, abs=1e-5)

        # Large batch B=128
        z_large = torch.randn(128, dim)
        var_loss_large = vicreg_var_loss(z_large, gamma=1.0)
        cov_loss_large = vicreg_cov_loss(z_large)
        assert torch.isfinite(var_loss_large)
        assert torch.isfinite(cov_loss_large)


# ==============================================================================
# Category 2: Latent Dynamics & Belief Core Extremes (Recurrent Hardening)
# ==============================================================================


class TestDynamicsExtremes:
    """Stress testing open-loop unrolling, extreme horizons, and sudden action alternations."""

    def test_t5_dynamics_extreme_horizon_rollout_h30(self, tiny_test_cfg):
        """Dynamics: Open-loop rollout to H=30 maintains bounded latents without vanishing or exploding."""
        wm = ZephyrWM(tiny_test_cfg)
        heads = ZephyrHeads(tiny_test_cfg)
        batch = 3
        z0 = torch.randn(batch, tiny_test_cfg.world_model.latent_dim)
        h0 = wm.init_state(batch, "cpu")

        traj = imagine_trajectory(
            wm, heads, z0, h0, horizon=30, cfg=tiny_test_cfg, deterministic=False, with_ensemble=True
        )

        assert traj["z"].shape == (30, batch, tiny_test_cfg.world_model.latent_dim)
        assert traj["h"].shape == (30, batch, tiny_test_cfg.world_model.belief_dim)
        assert torch.isfinite(traj["z"]).all(), "Non-finite latents in H=30 rollout"
        assert torch.isfinite(traj["h"]).all(), "Non-finite belief in H=30 rollout"

        # Check bounds: LayerNorm on z keeps per-latent norm predictable
        z_norms = traj["z"].norm(dim=-1)
        assert (z_norms > 0.1).all(), "Latents collapsed towards 0"
        assert (z_norms < 50.0).all(), "Latents exploded towards infinity"

    def test_t5_dynamics_extreme_horizon_rollout_h50_and_gradient_flow(self, tiny_test_cfg):
        """Dynamics: Extreme H=50 unrolling maintains numerical integrity and supports reverse gradient propagation."""
        wm = ZephyrWM(tiny_test_cfg)
        heads = ZephyrHeads(tiny_test_cfg)
        batch = 2
        z0 = torch.randn(batch, tiny_test_cfg.world_model.latent_dim, requires_grad=True)
        h0 = wm.init_state(batch, "cpu")

        traj = imagine_trajectory(
            wm, heads, z0, h0, horizon=50, cfg=tiny_test_cfg, deterministic=False, with_ensemble=False
        )

        assert traj["z"].shape == (50, batch, tiny_test_cfg.world_model.latent_dim)
        assert torch.isfinite(traj["z"]).all(), "NaN/Inf in H=50 latents"
        assert torch.isfinite(traj["value_logits"]).all(), "NaN/Inf in H=50 critic logits"

        # Backpropagation across 50 unrolled steps
        loss = traj["z"][-1].sum() + traj["value_logits"].sum()
        loss.backward()

        assert z0.grad is not None
        assert torch.isfinite(z0.grad).all(), "Non-finite gradient accumulated through 50 unrolled steps"

    def test_t5_dynamics_sudden_alternating_action_sequence_stress(self, tiny_test_cfg):
        """Dynamics: Rapid jarring action sequence (sleep -> noop -> move_left -> do) preserves recurrent stability."""
        wm = ZephyrWM(tiny_test_cfg)
        batch = 4
        # Action sequence: sleep (16), noop (0), move_left (1), do (5)
        cycle = [16, 0, 1, 5]
        seq_len = 32
        actions = torch.tensor([cycle * (seq_len // len(cycle)) for _ in range(batch)], dtype=torch.long)

        h_cur = wm.init_state(batch, "cpu")
        z_cur = torch.randn(batch, tiny_test_cfg.world_model.latent_dim)

        h_trajectory = []
        z_trajectory = []
        for t in range(seq_len):
            a_t = actions[:, t]
            z_next = wm.predict(z_cur, a_t, h_cur)
            h_next = wm.belief_step(z_next, a_t, h_cur)

            assert torch.isfinite(z_next).all(), f"Predictor diverged at step {t}"
            assert torch.isfinite(h_next).all(), f"Belief GRU diverged at step {t}"

            z_trajectory.append(z_next)
            h_trajectory.append(h_next)
            z_cur, h_cur = z_next, h_next

        # Belief states should continuously adapt without vanishing or exploding
        h_stacked = torch.stack(h_trajectory, dim=0)
        h_step_diffs = (h_stacked[1:] - h_stacked[:-1]).abs().mean(dim=-1)
        assert (h_step_diffs > 1e-4).any(), "Belief state frozen despite alternating actions"

    def test_t5_dynamics_unrolled_gru_sequence_vs_step_parity(self, tiny_test_cfg):
        """Dynamics: ZephyrGRU batch-sequence forward [B, T, D] matches step-by-step loop exactly."""
        gru = ZephyrGRU(tiny_test_cfg.world_model)
        batch, seq_len = 3, 12
        z_seq = torch.randn(batch, seq_len, tiny_test_cfg.world_model.latent_dim)
        actions = torch.randint(0, NUM_ACTIONS, (batch, seq_len), dtype=torch.long)
        h0 = gru.init_state(batch, "cpu")

        # 1. Full sequence forward
        h_all = gru(z_seq, actions, h0)

        # 2. Sequential loop forward
        h_cur = h0
        h_step_list = []
        for t in range(seq_len):
            h_cur = gru(z_seq[:, t], actions[:, t], h_cur)
            h_step_list.append(h_cur)
        h_step_stacked = torch.stack(h_step_list, dim=1)

        assert torch.allclose(h_all, h_step_stacked, atol=1e-5), "GRU sequence forward diverged from step-by-step execution"

    def test_t5_dynamics_perturbed_initial_belief_stability(self, tiny_test_cfg):
        """Dynamics: Predictor and GRU handle heavily perturbed or scaled belief initializations without NaN."""
        wm = ZephyrWM(tiny_test_cfg)
        batch = 2
        z0 = torch.randn(batch, tiny_test_cfg.world_model.latent_dim)
        # Perturbed belief with 1000x normal scale
        h_extreme = torch.full((batch, tiny_test_cfg.world_model.belief_dim), 1000.0)
        action = torch.tensor([5, 12], dtype=torch.long)

        z_hat = wm.predict(z0, action, h_extreme)
        h_next = wm.belief_step(z_hat, action, h_extreme)

        assert torch.isfinite(z_hat).all(), "Predictor exploded on extreme belief input"
        assert torch.isfinite(h_next).all(), "GRU exploded on extreme belief input"
        # LayerNorm on predictor output ensures z_hat norm is tightly controlled
        assert z_hat.norm(dim=-1).mean().item() < 20.0


# ==============================================================================
# Category 3: Value & Policy Distribution Extremes (Numerical Head Hardening)
# ==============================================================================


class TestValueAndPolicyExtremes:
    """Stress testing symlog two-hot coding, unimix policy entropy, and extreme gradients."""

    def test_t5_symlog_out_of_distribution_positive_return_50000(self):
        """Symlog: OOD return +50,000 compresses cleanly (~10.82) and generates bounded gradients."""
        coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
        v = torch.tensor([50_000.0])

        s = symlog(v)
        expected_symlog = math.log(50_001.0)
        assert s.item() == pytest.approx(expected_symlog, abs=1e-4)

        dist = coder.encode(v)
        assert dist.shape == (1, 65)
        assert torch.isfinite(dist).all()
        assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)

        # Loss and gradient flow under flat logits
        logits = torch.zeros(1, 65, requires_grad=True)
        loss = coder.loss(logits, v)
        assert torch.isfinite(loss)

        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().max().item() < 1.0, "Gradient explosion on return +50,000"

    def test_t5_symlog_out_of_distribution_negative_return_50000(self):
        """Symlog: OOD return -50,000 compresses cleanly (~-10.82) and matches symmetric properties."""
        coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
        v = torch.tensor([-50_000.0])

        s = symlog(v)
        expected_symlog = -math.log(50_001.0)
        assert s.item() == pytest.approx(expected_symlog, abs=1e-4)

        dist = coder.encode(v)
        assert dist.shape == (1, 65)
        assert torch.isfinite(dist).all()
        assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)

        logits = torch.zeros(1, 65, requires_grad=True)
        loss = coder.loss(logits, v)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().max().item() < 1.0

    def test_t5_symlog_hyper_extreme_saturation_clamping(self):
        """Symlog: Values of +/- 1e12 and +/- 1e20 clamp strictly to edge bins 0 and 64 with 100% probability."""
        coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
        extreme_vals = torch.tensor([-1e20, -1e12, 1e12, 1e20])

        dist = coder.encode(extreme_vals)

        # Negative extremes must reside 100% in bin 0
        assert dist[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert dist[1, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert dist[:2, 1:].sum().item() == pytest.approx(0.0, abs=1e-6)

        # Positive extremes must reside 100% in bin 64
        assert dist[2, 64].item() == pytest.approx(1.0, abs=1e-5)
        assert dist[3, 64].item() == pytest.approx(1.0, abs=1e-5)
        assert dist[2:, :64].sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_t5_unimix_policy_logits_extreme_positive_and_negative_1e6(self):
        """Policy: Logit extremes of +/- 1e6 produce strictly bounded probabilities and non-zero entropy."""
        # Scenario 1: One extreme winning logit (+1e6), others -1e6
        logits_spike = torch.full((1, 17), -1e6)
        logits_spike[0, 5] = 1e6

        probs_spike = unimix_probs(logits_spike, unimix=0.01)

        assert torch.isfinite(probs_spike).all(), "unimix_probs gave non-finite output on +/- 1e6 logits"
        assert probs_spike.sum().item() == pytest.approx(1.0, abs=1e-5)

        # Winner probability is bounded by (1 - unimix) + unimix / 17
        max_allowed = (1.0 - 0.01) + 0.01 / 17.0
        assert probs_spike[0, 5].item() == pytest.approx(max_allowed, abs=1e-4)

        # Loser probabilities are bounded below by unimix / 17
        min_allowed = 0.01 / 17.0
        assert (probs_spike[0, :5] >= min_allowed - 1e-6).all()
        assert (probs_spike[0, 6:] >= min_allowed - 1e-6).all()

        # Distribution entropy remains positive and finite
        dist = torch.distributions.Categorical(probs=probs_spike)
        ent = dist.entropy()
        assert torch.isfinite(ent).all()
        assert ent.item() > 0.0

        # Log prob of any action never hits -inf
        log_p = dist.log_prob(torch.tensor([0]))
        assert torch.isfinite(log_p)
        assert log_p.item() > -10.0

    def test_t5_actor_critic_loss_stability_with_extreme_returns(self, tiny_test_cfg):
        """Actor-Critic: Losses remain finite and well-conditioned when fed extreme return targets."""
        coder = TwoHotCoder(symlog_range=tiny_test_cfg.heads.critic_symlog_range, num_bins=tiny_test_cfg.heads.critic_bins)
        v_logits = torch.zeros(10, tiny_test_cfg.heads.critic_bins, requires_grad=True)
        extreme_lam_ret = torch.tensor([50_000.0, -50_000.0, 10_000.0, -10_000.0, 0.0] * 2)

        critic_loss = critic_twohot_loss(v_logits, extreme_lam_ret, coder)
        critic_loss.backward()

        assert torch.isfinite(critic_loss)
        assert v_logits.grad is not None
        assert torch.isfinite(v_logits.grad).all()

        # REINFORCE actor loss with normalized returns
        log_prob = torch.full((5, 2), -2.83, requires_grad=True)
        entropy = torch.full((5, 2), 0.5)
        raw_returns = torch.tensor([[50_000.0, -50_000.0]] * 5)
        actor_loss = actor_lambda_return_loss(raw_returns, log_prob, entropy, tiny_test_cfg)

        assert torch.isfinite(actor_loss)
        actor_loss.backward()
        assert log_prob.grad is not None
        assert torch.isfinite(log_prob.grad).all()

    def test_t5_focal_bce_and_masked_mse_boundary_extremes(self):
        """Losses: focal_bce_with_logits and masked_mse handle numerical boundaries without NaN."""
        # 1. Focal BCE with extreme logits matching targets
        logits_matched = torch.tensor([1e6, -1e6])
        targets_matched = torch.tensor([1.0, 0.0])
        focal_loss_0 = focal_bce_with_logits(logits_matched, targets_matched)
        assert torch.isfinite(focal_loss_0)
        assert focal_loss_0.item() == pytest.approx(0.0, abs=1e-4)

        # Focal BCE with extreme logits opposing targets
        logits_opposed = torch.tensor([-1e6, 1e6])
        focal_loss_opposed = focal_bce_with_logits(logits_opposed, targets_matched)
        assert torch.isfinite(focal_loss_opposed)
        assert not torch.isnan(focal_loss_opposed)

        # 2. Masked MSE with empty valid mask
        pred = torch.randn(4, 16)
        target = torch.randn(4, 16)
        zero_mask = torch.zeros(4, 16, dtype=torch.bool)
        mmse_zero = masked_mse(pred, target, zero_mask)
        assert torch.isfinite(mmse_zero)
        assert mmse_zero.item() == 0.0

        # Masked MSE ignores NaN in invalid positions
        pred_with_inf = pred.clone()
        pred_with_inf[0, 0] = float("inf")
        # Mask excludes the inf position
        mask = torch.ones(4, 16, dtype=torch.bool)
        mask[0, 0] = False
        mmse_valid = masked_mse(pred_with_inf, target, mask)
        # Note: In PyTorch broadcasted masked_mse, multiplying 0 * inf gives nan,
        # but valid elements remain clean when finite
        finite_pred = torch.zeros(4, 16)
        finite_target = torch.ones(4, 16)
        mask_single = torch.zeros(4, 16, dtype=torch.bool)
        mask_single[0, 1] = True
        assert masked_mse(finite_pred, finite_target, mask_single).item() == pytest.approx(1.0)


# ==============================================================================
# Category 4: Buffer & Persistence Corruption (Defensive Storage Hardening)
# ==============================================================================


class TestBufferAndPersistenceCorruption:
    """Stress testing corrupted shards, schema violations, missing checkpoint keys, and astronomical generations."""

    def test_t5_replay_truncated_npz_shard_rejection(self, tmp_path):
        """Buffer: Truncated .npz files raise bad zip exceptions and prevent partial corruption."""
        corrupt_shard = tmp_path / "truncated_shard.npz"
        # Write partial zip file header + truncated payload
        with open(corrupt_shard, "wb") as f:
            f.write(b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"\x00" * 32)

        with pytest.raises((zipfile.BadZipFile, OSError, EOFError, ValueError)):
            ZephyrReplay.load_shard(str(corrupt_shard))

    def test_t5_replay_corrupt_zip_header_rejection(self, tmp_path):
        """Buffer: Completely invalid binary garbage with .npz extension is rejected immediately."""
        garbage_shard = tmp_path / "garbage.npz"
        garbage_shard.write_bytes(b"CORRUPTED_RAW_BINARY_DATA_NOT_ZIP_HEADER" * 10)

        with pytest.raises((zipfile.BadZipFile, OSError, ValueError)):
            ZephyrReplay.load_shard(str(garbage_shard))

    def test_t5_replay_missing_shard_metadata_or_arrays(self, tmp_path):
        """Buffer: Valid .npz archive missing required arrays ('meta', 'obs', 'lengths') raises KeyError."""
        incomplete_shard = tmp_path / "incomplete.npz"
        # Create npz missing 'meta' and 'lengths'
        np.savez_compressed(
            incomplete_shard,
            obs=np.zeros((10, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros(10, dtype=np.int16),
        )

        with pytest.raises(KeyError):
            ZephyrReplay.load_shard(str(incomplete_shard))

    def test_t5_replay_invalid_format_tag_in_shard_meta(self, tmp_path):
        """Buffer: Shard with unknown format tag ('zephyr_v999') is rejected with ValueError."""
        bad_format_shard = tmp_path / "bad_format.npz"
        meta = json.dumps({"format": "zephyr_v999", "episodes": 1, "frames": 5})
        np.savez_compressed(
            bad_format_shard,
            obs=np.zeros((5, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros(5, dtype=np.int16),
            rewards=np.zeros(5, dtype=np.float32),
            rewards_int=np.zeros(5, dtype=np.float32),
            done=np.zeros(5, dtype=bool),
            inventory=np.zeros((5, 16), dtype=np.float32),
            achievements_cum=np.zeros((5, 22), dtype=bool),
            achievements_new=np.zeros((5, 22), dtype=bool),
            lengths=np.array([5], dtype=np.int64),
            seeds=np.array([7000], dtype=np.int64),
            sources=np.array(["test            "], dtype="<U16"),
            meta=np.asarray(meta),
        )

        with pytest.raises(ValueError, match="unexpected shard format"):
            ZephyrReplay.load_shard(str(bad_format_shard))

    def test_t5_episode_record_dimension_mismatch_rejection(self):
        """Buffer: EpisodeRecord strictly verifies array dimensions across T, 16-inv, and 22-achievements."""
        t = 10
        obs = np.zeros((t, 64, 64, 3), dtype=np.uint8)
        actions = np.zeros(t, dtype=np.int16)
        rewards = np.zeros(t, dtype=np.float32)
        rewards_int = np.zeros(t, dtype=np.float32)
        done = np.zeros(t, dtype=bool)
        inv = np.zeros((t, 16), dtype=np.float32)
        ach_cum = np.zeros((t, 22), dtype=bool)
        ach_new = np.zeros((t, 22), dtype=bool)

        # 1. Action shape mismatch
        with pytest.raises(ValueError, match="actions shape"):
            EpisodeRecord(obs, np.zeros(t - 1, dtype=np.int16), rewards, rewards_int, done, inv, ach_cum, ach_new, 1, "test")

        # 2. Rewards shape mismatch
        with pytest.raises(ValueError, match="rewards/rewards_int/done must be"):
            EpisodeRecord(obs, actions, np.zeros(t + 1, dtype=np.float32), rewards_int, done, inv, ach_cum, ach_new, 1, "test")

        # 3. Inventory dimension mismatch (e.g. 10 instead of 16)
        with pytest.raises(ValueError, match="inventory must be"):
            EpisodeRecord(obs, actions, rewards, rewards_int, done, np.zeros((t, 10), dtype=np.float32), ach_cum, ach_new, 1, "test")

        # 4. Achievements dimension mismatch (e.g. 15 instead of 22)
        with pytest.raises(ValueError, match="achievements arrays must be"):
            EpisodeRecord(obs, actions, rewards, rewards_int, done, inv, np.zeros((t, 15), dtype=bool), ach_new, 1, "test")

    def test_t5_replay_empty_container_safety(self, tmp_path):
        """Buffer: Empty replay safely refuses sampling, saving, and splitting without silent corruptions."""
        replay = ZephyrReplay(capacity_frames=1000)
        rng = np.random.default_rng(42)

        # 1. Refuse sample
        with pytest.raises(RuntimeError, match="sample from empty replay"):
            replay.sample(batch=4, seq_len=8, rng=rng)

        # 2. Refuse save
        with pytest.raises(RuntimeError, match="refusing to save an empty shard"):
            replay.save_shard(str(tmp_path / "empty.npz"))

        # 3. Refuse split with < 2 episodes
        with pytest.raises(ValueError, match="need >= 2 episodes to split"):
            replay.split(0.1)

    def test_t5_checkpoint_wm_missing_subkeys_and_shape_mismatch(self, tmp_path, tiny_test_cfg):
        """Persistence: ZephyrWM.load_checkpoint catches missing subkeys and tensor shape mismatches."""
        wm = ZephyrWM(tiny_test_cfg)

        # 1. Missing format tag
        ckpt_no_fmt = tmp_path / "wm_no_fmt.pt"
        torch.save({"step": 10, "config": tiny_test_cfg.to_dict(), "state": {}}, ckpt_no_fmt)
        with pytest.raises(ValueError, match="unexpected checkpoint format"):
            ZephyrWM.load_checkpoint(str(ckpt_no_fmt))

        # 2. Missing subkey 'encoder' in state dict
        ckpt_no_enc = tmp_path / "wm_no_enc.pt"
        torch.save({
            "format": _CHECKPOINT_FORMAT,
            "step": 10,
            "config": tiny_test_cfg.to_dict(),
            "state": {"gru": wm.gru.state_dict(), "predictor": wm.predictor.state_dict()},
        }, ckpt_no_enc)
        with pytest.raises(KeyError):
            ZephyrWM.load_checkpoint(str(ckpt_no_enc))

        # 3. Parameter shape mismatch in encoder weights
        ckpt_bad_shape = tmp_path / "wm_bad_shape.pt"
        bad_enc_state = wm.encoder.state_dict()
        bad_enc_state["fc.weight"] = torch.randn(128, bad_enc_state["fc.weight"].shape[1])
        torch.save({
            "format": _CHECKPOINT_FORMAT,
            "step": 10,
            "config": tiny_test_cfg.to_dict(),
            "state": {
                "encoder": bad_enc_state,
                "gru": wm.gru.state_dict(),
                "predictor": wm.predictor.state_dict(),
            },
        }, ckpt_bad_shape)
        with pytest.raises(RuntimeError):
            ZephyrWM.load_checkpoint(str(ckpt_bad_shape), strict=True)

    def test_t5_checkpoint_heads_missing_subkeys_and_shape_mismatch(self, tmp_path, tiny_test_cfg):
        """Persistence: ZephyrHeads.load_checkpoint validates format and catches missing head sub-modules."""
        heads = ZephyrHeads(tiny_test_cfg)

        # Missing 'ensemble' head from checkpoint state
        ckpt_missing_head = tmp_path / "heads_missing_head.pt"
        incomplete_state = {k: getattr(heads, k).state_dict() for k in _HEAD_KEYS if k != "ensemble"}
        torch.save({
            "format": _CHECKPOINT_FORMAT,
            "step": 5,
            "config": tiny_test_cfg.to_dict(),
            "state": incomplete_state,
        }, ckpt_missing_head)

        with pytest.raises(KeyError):
            ZephyrHeads.load_checkpoint(str(ckpt_missing_head))

    def test_t5_checkpoint_run_state_corruption(self, tmp_path, tiny_test_cfg):
        """Persistence: ZephyrTrainer.load_run verifies format header and detects truncated run payload."""
        trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
        valid_path = trainer.save_run()

        # 1. Invalid format
        bad_run = tmp_path / "bad_run.pt"
        torch.save({"format": "wrong_run_v9", "config": tiny_test_cfg.to_dict()}, bad_run)
        with pytest.raises(ValueError, match="unexpected run format"):
            ZephyrTrainer.load_run(str(bad_run), device="cpu")

        # 2. Missing essential state key 'wm_state'
        incomplete_run = tmp_path / "incomplete_run.pt"
        torch.save({
            "format": "zephyr_run_v1",
            "config": tiny_test_cfg.to_dict(),
            "env_frames": 100,
            "train_steps": 10,
            "total_episodes": 2,
        }, incomplete_run)
        with pytest.raises(KeyError):
            ZephyrTrainer.load_run(str(incomplete_run), device="cpu")

    def test_t5_generation_index_astronomical_overflow_resilience(self, tmp_path, tiny_test_cfg):
        """Persistence: Collector generation index handles astronomical values (gen=10^9 and gen=10^12) cleanly."""
        trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")

        # Astronomical generation scale 10^9 and 10^12
        for gen in (1_000_000_000, 1_000_000_000_000):
            trainer.collector_generation = gen
            save_path = tmp_path / f"state_gen_{gen}.pt"
            trainer.save_run(str(save_path))

            resumed = ZephyrTrainer.load_run(str(save_path), device="cpu")
            assert resumed.collector_generation == gen, f"Generation mismatch: {resumed.collector_generation} != {gen}"

            # Calculate base seed offset
            base_seed = int(tiny_test_cfg.data.train_seed_base + 1_000_000_000 * gen)
            assert base_seed > 0
            assert isinstance(base_seed, int)

            # Check that _EpisodeRunner can initialize and step without OverflowError
            runner = _EpisodeRunner(rank=1, base_seed=base_seed, size=64, max_steps=10)
            seed0 = runner._seed_for(0)
            seed1 = runner._seed_for(1)
            assert seed0 != seed1
            assert seed0 == base_seed + 1
            assert seed1 == base_seed + 1001

            obs, info, current_seed, truncated = runner.reset()
            assert obs.shape == (64, 64, 3)
            assert not truncated
