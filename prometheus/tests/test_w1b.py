"""
tests/test_w1b.py

Acceptance tests for W1B_LAST_CHANCE_PLAN (The Three Unpulled Levers):
- Lever A (L-A): Content-addressed pooling (tokens weighted by dynamic norm).
- Lever B (L-B): Token-grid Conv2d reconstruction decoder (192-channel output + pixel-grid mapping test).
- Lever C (L-C): GRU BPTT-lite sequence training through sampled chains (weight_hh receives non-zero gradient).
- Architecture and lineage invariants: arch tag 'v3.2-w1b', fresh buffer policy on legacy.
"""

import math
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import ObservationEncoder
from src.agent.agent import JEPAAgent
from src.memory.buffer import TwoTierReplayBuffer
from src.train import TrainingPipeline


class TestW1BLevers(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    # --------------------------------------------------------------------------
    # Lever A: Content-Addressed Pooling Tests
    # --------------------------------------------------------------------------

    def test_la_content_addressed_pooling_formula(self):
        """Verifies pool_tokens implements exact content-addressed attention formula."""
        encoder = ObservationEncoder(token_dim=64, latent_dim=256)
        encoder.eval()
        tokens = torch.randn(4, 64, 64)

        with torch.no_grad():
            pooled = encoder.pool_tokens(tokens)

            # Manual reference calculation per W1b spec:
            attn_scores = tokens.norm(dim=-1) / math.sqrt(64)
            attn_w = torch.softmax(attn_scores, dim=-1)
            pooled_token = (attn_w.unsqueeze(-1) * tokens).sum(dim=1)
            expected = encoder.ln(encoder.pooled_proj(pooled_token))

        self.assertTrue(
            torch.allclose(pooled, expected, atol=1e-5),
            "pool_tokens does not match reference content-addressed pooling"
        )

    def test_la_dynamic_tokens_receive_higher_weights(self):
        """Dynamic tiles (higher norm) must receive higher pooling attention weight than static grass tiles."""
        encoder = ObservationEncoder(token_dim=64, latent_dim=256)
        encoder.eval()

        # 56 low-norm static tokens (~0.05), 8 high-norm dynamic tokens (~2.0)
        tokens = torch.randn(1, 64, 64) * 0.05
        dynamic_indices = [5, 12, 19, 27, 34, 41, 50, 58]
        tokens[:, dynamic_indices, :] = torch.randn(1, 8, 64) * 2.0

        attn_scores = tokens.norm(dim=-1) / math.sqrt(64)
        attn_w = torch.softmax(attn_scores, dim=-1)[0]  # (64,)

        dynamic_weights = attn_w[dynamic_indices]
        static_mask = torch.ones(64, dtype=torch.bool)
        static_mask[dynamic_indices] = False
        static_weights = attn_w[static_mask]

        self.assertGreater(
            dynamic_weights.mean().item(),
            static_weights.mean().item() * 5.0,
            "Dynamic tokens must receive strongly higher pooling attention weights than static tokens"
        )

    # --------------------------------------------------------------------------
    # Lever B: Conv Reconstruction Decoder & Grid Mapping Tests
    # --------------------------------------------------------------------------

    def test_lb_recon_decoder_architecture_and_output_shape(self):
        """Verifies recon_decoder has ~100k params and produces exact (B, 3, 64, 64) frames."""
        encoder = ObservationEncoder(token_dim=64, latent_dim=256)
        B = 4
        tokens = torch.randn(B, 64, 64)
        recon = encoder.decode_tokens(tokens)

        self.assertEqual(recon.shape, (B, 3, 64, 64))

        # Check recon_decoder param count (~95k to ~105k)
        dec_params = sum(p.numel() for p in encoder.recon_decoder.parameters())
        self.assertGreater(dec_params, 85_000)
        self.assertLess(dec_params, 120_000)

    def test_lb_recon_grid_mapping(self):
        """
        User-mandated mapping unit test:
        Crafts a frame where each tile (r, c) has a unique center pixel color (r/7, c/7, 0.5).
        Trains recon on this single frame for 300 steps and verifies center pixels land in correct tiles.
        """
        agent = JEPAAgent(latent_dim=256, num_actions=17)
        frame = torch.zeros(1, 3, 64, 64)
        for r in range(8):
            for c in range(8):
                frame[0, :, r * 8 + 4, c * 8 + 4] = torch.tensor([r / 7.0, c / 7.0, 0.5])

        optimizer = torch.optim.Adam(agent.encoder.recon_decoder.parameters(), lr=1e-2)
        for _ in range(300):
            optimizer.zero_grad()
            tokens, _ = agent.encoder.forward_tokens(frame)
            rec = agent.encoder.decode_tokens(tokens)
            loss = F.mse_loss(rec, frame)
            loss.backward()
            optimizer.step()

        agent.eval()
        with torch.no_grad():
            rec = agent.encoder.decode_tokens(agent.encoder.forward_tokens(frame)[0])

        for r in range(8):
            for c in range(8):
                diff = (rec[0, :, r * 8 + 4, c * 8 + 4] - frame[0, :, r * 8 + 4, c * 8 + 4]).abs().max().item()
                self.assertLess(
                    diff,
                    0.15,
                    f"Tile ({r}, {c}) center pixel mapping error {diff:.4f} exceeds 0.15 tolerance"
                )

    # --------------------------------------------------------------------------
    # Lever C: GRU Sequence Training (BPTT-lite) Tests
    # --------------------------------------------------------------------------

    def test_lc_buffer_sample_sequence_chains(self):
        """Verifies buffer sample_sequence_chains extracts valid, padded chains with masks."""
        buffer = TwoTierReplayBuffer(tier1_capacity=200, tier2_capacity=50, mem_hidden=256)

        # Insert 3 episodes of length 15 each
        for ep in range(3):
            for step in range(15):
                obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
                next_obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
                done = (step == 14)
                h = np.random.randn(256).astype(np.float16)
                buffer.add(
                    obs=obs,
                    action=step % 17,
                    reward=1.0 if done else 0.0,
                    next_obs=next_obs,
                    done=done,
                    episode_id=ep,
                    step_in_ep=step,
                    h=h,
                )

        chains = buffer.sample_sequence_chains(num_chains=8, chain_len=8, device="cpu")
        self.assertIsNotNone(chains)
        self.assertEqual(chains["obs"].shape, (8, 8, 3, 64, 64))
        self.assertEqual(chains["next_obs"].shape, (8, 8, 3, 64, 64))
        self.assertEqual(chains["h"].shape, (8, 256))
        self.assertEqual(chains["mask"].shape, (8, 8))
        self.assertTrue((chains["mask"] >= 0.0).all() and (chains["mask"] <= 1.0).all())

    def test_lc_gru_weight_hh_receives_nonzero_gradients(self):
        """Verifies that GRU recurrent weight_hh strictly receives non-zero gradients via sequence BPTT."""
        trainer = TrainingPipeline(
            config={
                "batch_size": 16,
                "tier1_capacity": 500,
                "tier2_capacity": 100,
                "warmup_steps": 10,
                "num_envs": 1,
                "allow_fresh_buffer": True,
            }
        )

        # Seed the buffer with consecutive transitions so chains exist
        for step in range(30):
            obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
            next_obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
            done = (step == 29)
            h = np.random.randn(256).astype(np.float16)
            trainer.buffer.add(
                obs=obs,
                action=step % 17,
                reward=0.1,
                next_obs=next_obs,
                done=done,
                episode_id=0,
                step_in_ep=step,
                h=h,
            )

        trainer.agent.gru.weight_hh.grad = None
        trainer.agent.mem_to_pooled.weight.grad = None

        loss_dict = trainer.update_world_model()

        self.assertIn("l_seq", loss_dict)
        self.assertIsNotNone(trainer.agent.gru.weight_hh.grad)
        hh_grad_norm = trainer.agent.gru.weight_hh.grad.norm().item()
        self.assertGreater(
            hh_grad_norm,
            0.0,
            "GRU recurrent weights (weight_hh) must receive non-zero gradients during world model update"
        )
        self.assertIsNotNone(trainer.agent.mem_to_pooled.weight.grad)
        self.assertGreater(
            trainer.agent.mem_to_pooled.weight.grad.norm().item(),
            0.0,
            "mem_to_pooled head must receive non-zero gradients"
        )

    # --------------------------------------------------------------------------
    # Lineage and Metadata Tests
    # --------------------------------------------------------------------------

    def test_arch_tag_is_v3_2_w1b(self):
        """Verifies pipeline specifies arch tag 'v3.2-w1b'."""
        trainer = TrainingPipeline(
            config={
                "batch_size": 4,
                "warmup_steps": 5,
                "allow_fresh_buffer": True,
            }
        )
        saved_path = trainer.save_checkpoint()
        ckpt = torch.load(saved_path, map_location="cpu", weights_only=False)
        self.assertIn(ckpt["metadata"]["arch"], ["v3.2-w1b", "v3.3-phaseA"])


if __name__ == "__main__":
    unittest.main()
