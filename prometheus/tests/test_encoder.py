"""
tests/test_encoder.py

Comprehensive unit tests verifying ObservationEncoder and TargetEncoder invariants.
"""

import unittest
import time
import torch
import torch.nn as nn

from src.models.encoder import ObservationEncoder, ResNetTinyEncoder, ResBlock
from src.models.target import TargetEncoder


class TestObservationEncoderAndTarget(unittest.TestCase):
    """Unit and integration tests for ObservationEncoder and TargetEncoder."""

    def setUp(self):
        torch.manual_seed(42)

    def test_01_encoder_forward_shape_contract(self):
        """Verifies forward pass transforms (B, 3, 64, 64) -> (B, 256)."""
        enc = ObservationEncoder(in_channels=3, latent_dim=256)
        x = torch.randn(8, 3, 64, 64)
        z = enc(x)
        self.assertEqual(z.shape, (8, 256))
        self.assertTrue(z.requires_grad)

    def test_02_encoder_layernorm_properties(self):
        """Verifies output has mean ≈ 0.0 and std ≈ 1.0 across feature axis."""
        enc = ObservationEncoder(in_channels=3, latent_dim=256)
        x = torch.randn(4, 3, 64, 64)
        z = enc(x)
        mean = z.mean(dim=-1)
        std = z.std(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-4))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=2e-2))

    def test_03_encoder_parameter_budget(self):
        """Verifies parameter count is strictly under 800k."""
        enc = ObservationEncoder()
        params = sum(p.numel() for p in enc.parameters())
        self.assertIn(params, (300_000, 304224, 793344))
        self.assertLess(params, 800_000, f"Params {params} exceeds 800,000 budget")

    def test_04_encoder_cpu_latency_under_2ms(self):
        """Verifies average forward pass on CPU is < 2.0 ms per observation."""
        enc = ObservationEncoder()
        enc.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            for _ in range(10):
                _ = enc(x)  # Warmup
            t0 = time.perf_counter()
            N = 50
            for _ in range(N):
                _ = enc(x)
            avg_ms = (time.perf_counter() - t0) / N * 1000
        self.assertLess(avg_ms, 2.0, f"CPU latency {avg_ms:.2f}ms exceeds 2.0ms")

    def test_05_encoder_gradient_flow_to_all_layers(self):
        """Verifies backward pass populates gradients for every trainable parameter."""
        enc = ObservationEncoder()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        tokens, z = enc.forward_tokens(x)
        recon = enc.decode_tokens(tokens)
        loss = (z ** 2).sum() + (recon ** 2).sum()
        loss.backward()
        for name, param in enc.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
            self.assertGreater(param.grad.abs().sum().item(), 0.0, f"Zero gradient for {name}")

    def test_06_target_stop_gradient_guarantee(self):
        """Verifies target parameters and outputs are completely detached."""
        enc = ObservationEncoder()
        tgt = TargetEncoder(enc, tau=0.99)
        # Verify both target_encoder and outer TargetEncoder expose only frozen parameters
        for param in tgt.target_encoder.parameters():
            self.assertFalse(param.requires_grad)
        for param in tgt.parameters():
            self.assertFalse(param.requires_grad)
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        z = tgt(x)
        self.assertFalse(z.requires_grad)

    def test_07_target_polyak_ema_numerical_accuracy(self):
        """Verifies theta_target = tau * theta_target + (1 - tau) * theta_online."""
        enc = ObservationEncoder()
        tgt = TargetEncoder(enc, tau=0.99)
        with torch.no_grad():
            for p_o in enc.parameters():
                p_o.fill_(10.0)
            for p_t in tgt.target_encoder.parameters():
                p_t.fill_(0.0)
        tgt.update()  # 0.99 * 0.0 + 0.01 * 10.0 = 0.1
        for p_t in tgt.target_encoder.parameters():
            self.assertTrue(torch.allclose(p_t, torch.tensor(0.1), atol=1e-6))

    def test_08_target_hard_update_synchronization(self):
        """Verifies hard_update instantly synchronizes weights."""
        enc = ObservationEncoder()
        tgt = TargetEncoder(enc, tau=0.99)
        with torch.no_grad():
            for p in enc.parameters():
                p.fill_(42.0)
        tgt.hard_update()
        for p_t in tgt.target_encoder.parameters():
            self.assertTrue(torch.allclose(p_t, torch.tensor(42.0)))

    def test_09_spatial_hud_gradient_sensitivity(self):
        """Verifies HUD status rows 51-54 receive strong gradient attention."""
        enc = ObservationEncoder()
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        z = enc(x)
        loss = (z ** 2).sum()
        loss.backward()
        grad_map = x.grad.abs().squeeze(0).mean(dim=0)
        hud_grad = grad_map[51:55, :].mean().item()
        terrain_grad = grad_map[0:50, :].mean().item()
        self.assertGreater(hud_grad, 1e-5)
        sensitivity_ratio = hud_grad / (terrain_grad + 1e-8)
        self.assertGreater(sensitivity_ratio, 0.5, f"HUD sensitivity ratio {sensitivity_ratio:.2f} is too low")

    def test_10_generalization_atari_and_unbatched_inputs(self):
        """Verifies generalization to Atari (84x84x4) and unbatched (3, 64, 64) inputs."""
        # Atari ALE 4-frame stack
        enc_atari = ObservationEncoder(in_channels=4, latent_dim=256)
        x_atari = torch.randn(2, 4, 84, 84)
        z_atari = enc_atari(x_atari)
        self.assertEqual(z_atari.shape, (2, 256))

        # Unbatched Crafter observation (3, 64, 64)
        enc_crafter = ObservationEncoder(in_channels=3, latent_dim=256)
        x_single = torch.randn(3, 64, 64)
        z_single = enc_crafter(x_single)
        self.assertEqual(z_single.shape, (256,))

        # Uint8 format conversion
        x_uint8 = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
        z_uint8 = enc_crafter(x_uint8)
        self.assertEqual(z_uint8.shape, (2, 256))

    def test_11_alias_and_eval_mode_lock(self):
        """Verifies class alias ResNetTinyEncoder and TargetEncoder train mode lock."""
        self.assertIs(ResNetTinyEncoder, ObservationEncoder)
        enc = ObservationEncoder()
        tgt = TargetEncoder(enc, tau=0.99)
        enc.train(True)
        self.assertTrue(enc.training)
        tgt.train(True)
        self.assertFalse(tgt.target_encoder.training)
        # TargetEncoder mode changes must NEVER alter online encoder training mode
        self.assertTrue(enc.training, "TargetEncoder.train(True) mutated online encoder mode!")
        tgt.train(False)
        self.assertTrue(enc.training, "TargetEncoder.train(False) mutated online encoder mode!")
        tgt.eval()
        self.assertTrue(enc.training, "TargetEncoder.eval() mutated online encoder mode!")

    def test_12_invalid_input_shape_raises(self):
        """Verifies invalid input shape raises ValueError."""
        enc = ObservationEncoder()
        with self.assertRaises(ValueError):
            enc(torch.randn(64, 64))
        with self.assertRaises(ValueError):
            enc(torch.randn(1, 2, 3, 64, 64))

    def test_13_target_encoder_submodule_isolation_and_parameter_count(self):
        """Verifies TargetEncoder does NOT register online_encoder as child submodule."""
        enc = ObservationEncoder()
        tgt = TargetEncoder(enc, tau=0.99)

        # 1. Parameter count check: TargetEncoder must match ObservationEncoder parameter count (< 800k)
        tgt_params = list(tgt.parameters())
        enc_params = list(enc.parameters())
        self.assertEqual(len(tgt_params), len(enc_params))
        total_tgt_params = sum(p.numel() for p in tgt_params)
        self.assertIn(total_tgt_params, (300_000, 304_224, 793_344))
        self.assertLess(total_tgt_params, 800_000)

        # 2. All target parameters must have requires_grad=False
        trainable_tgt_params = sum(p.numel() for p in tgt_params if p.requires_grad)
        self.assertEqual(trainable_tgt_params, 0, f"Expected 0 trainable params in target, got {trainable_tgt_params}")
        self.assertFalse(any(p.requires_grad for p in tgt_params))

        # 3. Submodule isolation: online_encoder must NOT be in named_children
        children_dict = dict(tgt.named_children())
        self.assertNotIn("online_encoder", children_dict, "online_encoder leaked into child modules!")
        self.assertIn("target_encoder", children_dict)

        # 4. Mode isolation: mutating target mode preserves online encoder mode
        enc.train(True)
        tgt.train(True)
        self.assertTrue(enc.training)
        tgt.train(False)
        self.assertTrue(enc.training)
        tgt.eval()
        self.assertTrue(enc.training)


if __name__ == "__main__":
    unittest.main()

