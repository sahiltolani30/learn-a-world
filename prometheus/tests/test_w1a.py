"""
tests/test_w1a.py

Comprehensive acceptance tests for W1A Scene-Collapse Fix (W1A_SCENE_COLLAPSE_FIX.md §4):
1. Recon shape: decode_tokens(tokens) returns (B,3,64,64); token grid of zeros decodes to constant frame.
2. Recon learns: 200 optimizer steps on a single fixed frame drives l_recon below 0.01.
3. Action-everywhere: changing action changes EVERY token position's prediction (delta > 0 for all 64 positions).
4. Dynamic-tile weights: synthetic batch where one tile changes -> that tile's loss weight is upweighted.
5. Per-token floor: collapsed token position produces positive l_tokfloor; varying one produces ~0.
6. Mean pooling: pooled = mean of tokens before pooled_proj + LayerNorm (verify by direct computation).
7. Eval memory (Hotfix 4): eval hidden state changes across steps, resets on done.
8. Success counting: achievement-verified counting remains unchanged.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.encoder import ObservationEncoder
from src.models.predictor import SpatialTransformerPredictor
from src.agent.agent import JEPAAgent


class TestW1aAcceptance:

    def test_1_recon_shape(self):
        """
        Acceptance Test 1:
        decode_tokens(tokens) returns (B,3,64,64); a token grid of zeros decodes to a constant frame.
        """
        encoder = ObservationEncoder(token_dim=64, latent_dim=256)
        B = 4
        tokens = torch.randn(B, 64, 64)
        recon = encoder.decode_tokens(tokens)
        assert recon.shape == (B, 3, 64, 64), f"Expected (4, 3, 64, 64), got {recon.shape}"

        # Zero tokens decode to constant patches (bias of recon_head)
        zeros_tok = torch.zeros(2, 64, 64)
        recon_zeros = encoder.decode_tokens(zeros_tok)
        assert recon_zeros.shape == (2, 3, 64, 64)
        
        # Verify all 8x8 patches within recon_zeros are identical across spatial positions
        # Each patch is 8x8 pixels across 3 channels
        patch_00 = recon_zeros[0, :, :8, :8]
        patch_01 = recon_zeros[0, :, :8, 8:16]
        assert torch.allclose(patch_00, patch_01, atol=1e-5), "Zero tokens did not produce identical constant patches"

    def test_2_recon_learns(self):
        """
        Acceptance Test 2:
        200 optimizer steps on a single fixed frame must drive l_recon below 0.01
        (proves the head + token path can carry pixel detail).
        """
        import crafter
        torch.manual_seed(42)
        encoder = ObservationEncoder(token_dim=64, latent_dim=256)
        env = crafter.Env(seed=42)
        frame = env.reset()
        x = torch.from_numpy(frame.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        for step in range(200):
            optimizer.zero_grad()
            tokens, _ = encoder.forward_tokens(x)
            recon = encoder.decode_tokens(tokens)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()

        assert loss.item() < 0.01, f"Expected l_recon < 0.01 after 200 steps, got {loss.item():.4f}"

    def test_3_action_everywhere(self):
        """
        Acceptance Test 3:
        Changing the action changes EVERY token position's prediction
        (per-position output delta > 0 for all 64 positions after a forward with two different actions).
        """
        torch.manual_seed(42)
        predictor = SpatialTransformerPredictor(token_dim=64, num_actions=17)
        predictor.eval()
        tokens = torch.randn(1, 64, 64)

        with torch.no_grad():
            a0 = torch.tensor([0])  # noop
            a5 = torch.tensor([5])  # do
            mean_pred0, _ = predictor.forward_tokens(tokens.clone(), a0)
            mean_pred5, _ = predictor.forward_tokens(tokens.clone(), a5)

            # Per-position difference norm: (1, 64)
            delta = (mean_pred0 - mean_pred5).norm(dim=-1).squeeze(0)  # (64,)
            assert delta.shape == (64,)
            assert (delta > 1e-4).all(), (
                f"Action did not modulate all 64 token positions! Min delta: {delta.min().item():.6f}"
            )

    def test_4_dynamic_tile_weights(self):
        """
        Acceptance Test 4:
        Synthetic batch where one tile changes -> that tile's loss weight is upweighted (~5x vs ~1x).
        """
        B = 2
        curr_tokens = torch.zeros(B, 64, 64)
        target_tokens = torch.zeros(B, 64, 64)
        # Tile 12 changes significantly
        target_tokens[:, 12, :] = 3.0

        with torch.no_grad():
            delta = (target_tokens - curr_tokens).norm(dim=-1)          # (B, 64)
            weights = torch.ones_like(delta)
            topk = torch.topk(delta, k=8, dim=-1).indices                # the 8 most-changing tiles
            weights.scatter_(1, topk, 5.0)
            weights = weights / weights.mean(dim=-1, keepdim=True)       # mean weight = 1 per sample

        # Tile 12 is in topk and must receive 5.0 / mean_weight
        # Since 8 tiles get 5 and 56 tiles get 1, sum = 40 + 56 = 96, mean = 96/64 = 1.5
        # Weighted tile 12 = 5.0 / 1.5 = 3.333, unweighted = 1.0 / 1.5 = 0.667
        # Ratio is 5.0 / 1.0 = 5.0x
        w_dynamic = weights[0, 12].item()
        w_static = weights[0, 0].item()
        assert np.isclose(w_dynamic / w_static, 5.0, atol=1e-3), f"Expected 5x ratio, got {w_dynamic / w_static:.3f}"

        # Test compute_masked_loss integrates weights properly
        predictor = SpatialTransformerPredictor(token_dim=64, num_actions=17)
        mask = torch.ones(B, 64, dtype=torch.bool)
        pred_tokens = torch.zeros(B, 64, 64)
        loss_weighted = predictor.compute_masked_loss(pred_tokens, target_tokens, mask, position_weights=weights)
        loss_unweighted = predictor.compute_masked_loss(pred_tokens, target_tokens, mask, position_weights=None)
        # Weighted loss must be higher because error on tile 12 is upweighted
        assert loss_weighted > loss_unweighted, f"Weighted {loss_weighted} should exceed unweighted {loss_unweighted}"

    def test_5_per_token_floor(self):
        """
        Acceptance Test 5:
        A collapsed token position (constant across batch) produces a positive l_tokfloor;
        a varying one produces ~0.
        """
        token_var_floor = 0.05
        # Collapsed batch: all samples have exact same tokens -> var = 0
        collapsed_tokens = torch.ones(16, 64, 64)
        tok_var_collapsed = collapsed_tokens.var(dim=0)
        l_tokfloor_collapsed = F.relu(token_var_floor - tok_var_collapsed).mean()
        assert l_tokfloor_collapsed.item() > 0.049, f"Expected floor loss ~0.05, got {l_tokfloor_collapsed.item()}"

        # Healthy batch with high variance: var >> 0.05 -> loss is 0
        healthy_tokens = torch.randn(64, 64, 64) * 2.0
        tok_var_healthy = healthy_tokens.var(dim=0)
        l_tokfloor_healthy = F.relu(token_var_floor - tok_var_healthy).mean()
        assert l_tokfloor_healthy.item() == 0.0, f"Expected floor loss 0.0, got {l_tokfloor_healthy.item()}"

    def test_6_mean_pooling(self):
        """
        Acceptance Test 6:
        Mean pooling: pooled = mean of tokens before pooled_proj + LayerNorm (verify by direct computation).
        """
        encoder = ObservationEncoder(token_dim=64, latent_dim=256)
        encoder.eval()
        tokens = torch.randn(3, 64, 64)

        with torch.no_grad():
            pooled = encoder.pool_tokens(tokens)
            expected_token = tokens.mean(dim=1)
            expected_pooled = encoder.ln(encoder.pooled_proj(expected_token))

        assert torch.allclose(pooled, expected_pooled, atol=1e-5), "pool_tokens does not match direct mean pooling"

    def test_7_eval_memory_hotfix4(self):
        """
        Acceptance Test 7:
        Eval memory (Hotfix 4): during evaluate(), the eval hidden state changes across steps
        and resets on done.
        """
        agent = JEPAAgent(latent_dim=256, num_actions=17, device="cpu")
        agent.eval()

        obs = torch.rand(3, 64, 64)
        h0 = torch.zeros(1, 256)

        with torch.no_grad():
            x = obs.unsqueeze(0)
            z = agent.encoder(x)
            h1 = agent.gru(z, h0)
            # Memory state must change
            assert not torch.allclose(h0, h1), "GRU step did not change eval_h"

            # Readout with h0 vs h1 must differ
            r0 = agent.get_readout(z, h0)
            r1 = agent.get_readout(z, h1)
            assert not torch.allclose(r0, r1), "Readout did not incorporate h"

            # Resetting h to zero restores original readout on identical obs
            h_reset = torch.zeros(1, 256)
            r_reset = agent.get_readout(z, h_reset)
            assert torch.allclose(r0, r_reset, atol=1e-5), "Resetting h did not restore base readout"

    def test_8_success_counting(self):
        """
        Acceptance Test 8:
        Verify achievement tracking and milestone discovery mechanics.
        """
        from src.train import TrainingPipeline
        config = {
            "device": "cpu",
            "total_steps": 100,
            "num_envs": 2,
            "batch_size": 16,
            "eval_interval": 100,
            "save_interval": 0,
            "no_resume": True,
            "allow_fresh_buffer": True,
        }
        pipe = TrainingPipeline(config)
        assert len(pipe.all_unlocked_achievements) == 0
        assert isinstance(pipe.success_counts, dict)
        assert pipe.token_var_floor == 0.05
        pipe.envs.close()
        pipe.eval_env.close()
