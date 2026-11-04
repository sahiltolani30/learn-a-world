"""Phase I: Integration & Regression Verification Suite for Variant D in JepaWorldModel.

Tests:
1. Checkpoint compatibility: Old state_dict containing final_norm loads strictly without error.
2. Predictor output invariants: Shape [B, 64, 128], dtype, device, and unnormalized representation space.
3. Gradient flow: Gradients flow cleanly to current latents, context state, action embeddings, and transformer.
4. Ensemble disagreement and auxiliary heads: Operates cleanly with residual predictions.
5. Autoregressive rollout: 8-step autoregressive rollout produces finite tensors and correct horizons.
6. Checkpoint save/load round-trip: Exact restoration of all parameters.
"""
import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.jepa_world_model import JepaWorldModel
from src.models.latent_predictor import LatentPredictorMember
from src.models.action_embedding import ActionEmbedding
from src.learning.optimizer import build_optimizer_and_scheduler
from src.learning.learner import JepaLearner


def test_checkpoint_loading_strict_compatibility():
    """Verify that a state_dict containing legacy final_norm parameters loads strictly with 0 errors."""
    model = JepaWorldModel()
    original_sd = model.state_dict()

    # Confirm final_norm keys exist in state_dict (for compatibility)
    final_norm_keys = [k for k in original_sd.keys() if "final_norm" in k]
    assert len(final_norm_keys) == 6, f"Expected 6 final_norm keys (weight+bias for 3 members), got {final_norm_keys}"

    # Verify strict loading into a fresh model
    new_model = JepaWorldModel()
    load_result = new_model.load_state_dict(original_sd, strict=True)
    assert len(load_result.missing_keys) == 0, f"Unexpected missing keys: {load_result.missing_keys}"
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    # Verify weights are identical
    for k in original_sd.keys():
        assert torch.equal(new_model.state_dict()[k], original_sd[k])


def test_predictor_output_invariants():
    """Verify output shape [B, 64, 128], dtype, device, and residual unnormalized representation space."""
    predictor = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64)
    B = 4
    zt = torch.randn(B, 64, 128) + 0.64
    ht = torch.randn(B, 256)
    a_emb = torch.randn(B, 64)

    out = predictor(zt, ht, a_emb)
    assert out.shape == (B, 64, 128)
    assert out.dtype == torch.float32
    assert out.device == zt.device
    assert torch.isfinite(out).all()

    # In Variant D, output is NOT forced to zero-mean LayerNorm
    mean_channel = out.mean(dim=-1).abs().mean().item()
    assert mean_channel > 0.1, f"Output should retain raw token space mean, got {mean_channel}"


def test_gradient_flow_residual_and_action():
    """Verify gradients flow backward into zt, ht, and action embeddings."""
    predictor = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64)
    act_table = ActionEmbedding(num_actions=17, embedding_dim=64)

    B = 4
    zt = torch.randn(B, 64, 128, requires_grad=True)
    ht = torch.randn(B, 256, requires_grad=True)
    actions = torch.tensor([0, 3, 7, 12])
    target = zt.clone().detach() + torch.randn(B, 64, 128) * 0.1

    a_emb = act_table(actions)
    out = predictor(zt, ht, a_emb)

    pn = F.normalize(out, dim=-1, eps=1e-6)
    tn = F.normalize(target, dim=-1, eps=1e-6)
    loss = (1.0 - (pn * tn).sum(dim=-1)).mean()
    loss.backward()

    assert zt.grad is not None and torch.norm(zt.grad).item() > 0, "No gradient flow to input latents zt"
    assert ht.grad is not None and torch.norm(ht.grad).item() > 0, "No gradient flow to context ht"
    assert act_table.embedding.weight.grad is not None, "No gradient flow to action embeddings"
    assert torch.norm(act_table.embedding.weight.grad).item() > 0, "Zero gradient magnitude on action embeddings"
    assert torch.norm(predictor.cond_mlp[0].weight.grad).item() > 0, "Zero gradient on cond_mlp"
    assert torch.norm(predictor.transformer.layers[0].linear1.weight.grad).item() > 0, "Zero gradient on transformer"


def test_ensemble_disagreement_and_auxiliary_heads():
    """Verify ensemble prediction, disagreement estimation, and auxiliary prediction heads."""
    model = JepaWorldModel()
    B = 2
    zt = torch.randn(B, 64, 128)
    ht = torch.randn(B, 256)
    actions = torch.tensor([1, 5])

    preds, mean_pooled, disagreement = model.predict_one_step_ensemble(zt, ht, actions)
    assert len(preds) == 3, f"Expected 3 ensemble members, got {len(preds)}"
    for p in preds:
        assert p.shape == (B, 64, 128)
        assert torch.isfinite(p).all()
    assert mean_pooled.shape == (B, 128)
    assert disagreement.shape == (B,)
    assert torch.all(disagreement >= 0.0)
    assert torch.isfinite(disagreement).all()

    # Auxiliary heads check
    aux = model.auxiliary_heads(mean_pooled, ht)
    assert aux.reward.shape == (B, 1)
    assert aux.done_probability.shape == (B, 1)
    assert torch.isfinite(aux.reward).all()
    assert torch.isfinite(aux.done_probability).all()


def test_autoregressive_rollout_stability():
    """Verify 8-step autoregressive rollout produces valid horizons without numerical issues."""
    model = JepaWorldModel()
    B = 2
    zt = torch.randn(B, 64, 128)
    ht = torch.randn(B, 256)
    future_actions = torch.randint(0, 17, (B, 8))

    rollout = model.autoregressive_rollout(zt, ht, future_actions, member_idx=0)
    for h in [1, 2, 4, 8]:
        assert h in rollout["predictions_by_horizon"], f"Missing horizon {h}"
        z_h = rollout["predictions_by_horizon"][h]
        assert z_h.shape == (B, 64, 128)
        assert torch.isfinite(z_h).all(), f"Non-finite prediction at horizon {h}"
        u_h = rollout["pooled_by_horizon"][h]
        assert u_h.shape == (B, 128)
        assert torch.isfinite(u_h).all()


def test_checkpoint_save_load_roundtrip():
    """Verify exact state_dict save and load round-trip with the modified predictor."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_roundtrip.pt")
        model1 = JepaWorldModel()
        torch.save({"model_state_dict": model1.state_dict()}, ckpt_path)

        model2 = JepaWorldModel()
        loaded = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(loaded["model_state_dict"], strict=True)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Parameter mismatch after roundtrip"
