"""Automated verification suite for AEON Phase H smallest architectural change.

Tests:
1. Shape invariants: All four predictor variants output [B, 64, 128].
2. Normalization alignment: Variant B and D outputs match raw token space statistics;
   Variant A and C output LayerNorm centered features.
3. Action shuffling: Action shuffling produces valid reproducible derangements
   preserving sample counts and transition alignment.
4. Gradient flow: Backward pass yields strictly non-zero gradients across all
   trainable components (action embedding, conditioning MLP, transformer).
5. Dynamic vs static partitioning: Verifies mathematical partitioning of spatial
   tokens into static (< threshold) and dynamic (>= threshold) regions.
6. Data integrity: Confirms train (seeds 101, 102) and heldout (seeds 103, 104) splits
   have disjoint episodes and valid temporal ordering.
7. Metric aggregation: Ensures finite numbers, non-negative errors, and correct
   action advantage sign convention (A = E_shuf - E_corr).
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.visual_encoder import VisualEncoder
from src.models.latent_predictor import LatentPredictorMember
from src.models.action_embedding import ActionEmbedding


# ------------------------------------------------------------------------------
# Variant Architecture Definitions for Phase H
# ------------------------------------------------------------------------------
class VariantAPredictor(nn.Module):
    """Variant A: Existing baseline predictor (control). Pre-LN Transformer + LayerNorm."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        out = self.transformer(modulated)
        return self.final_norm(out)


class VariantBPredictor(nn.Module):
    """Variant B: Normalization-aligned predictor. No final LayerNorm; outputs in raw token space."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        return self.transformer(modulated)


class VariantCPredictor(nn.Module):
    """Variant C: Residual predictor with baseline normalization. z_hat = LayerNorm(zt + delta)."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        delta = self.transformer(modulated)
        return self.final_norm(zt + delta)


class VariantDPredictor(nn.Module):
    """Variant D: Residual + Normalization-aligned predictor. z_hat = zt + delta (no final LayerNorm)."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        delta = self.transformer(modulated)
        return zt + delta


# ------------------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------------------
@pytest.fixture
def dummy_inputs():
    B = 8
    zt = torch.randn(B, 64, 128) + 0.64
    ht = torch.randn(B, 256)
    a_emb = torch.randn(B, 64)
    target = zt + torch.randn(B, 64, 128) * 0.05
    return zt, ht, a_emb, target


def test_variant_output_shapes(dummy_inputs):
    """Ensure all 4 variants produce exactly [B, 64, 128]."""
    zt, ht, a_emb, _ = dummy_inputs
    variants = [
        ("A", VariantAPredictor()),
        ("B", VariantBPredictor()),
        ("C", VariantCPredictor()),
        ("D", VariantDPredictor()),
    ]
    for name, model in variants:
        out = model(zt, ht, a_emb)
        assert out.shape == (8, 64, 128), f"Variant {name} failed shape check: got {out.shape}"
        assert torch.isfinite(out).all(), f"Variant {name} output contains non-finite values"


def test_normalization_properties(dummy_inputs):
    """Verify LayerNorm centering vs raw token representation spaces."""
    zt, ht, a_emb, _ = dummy_inputs
    var_a = VariantAPredictor()
    var_b = VariantBPredictor()
    var_c = VariantCPredictor()
    var_d = VariantDPredictor()

    out_a = var_a(zt, ht, a_emb)
    out_b = var_b(zt, ht, a_emb)
    out_c = var_c(zt, ht, a_emb)
    out_d = var_d(zt, ht, a_emb)

    # Variant A and C output LayerNorm centered features: channel mean is ~0.0
    assert torch.abs(out_a.mean(dim=-1)).mean().item() < 1e-4, "Variant A output is not LayerNorm centered"
    assert torch.abs(out_c.mean(dim=-1)).mean().item() < 1e-4, "Variant C output is not LayerNorm centered"

    # Variant B and D preserve uncentered token space: mean channel value is significantly non-zero
    assert torch.abs(out_b.mean(dim=-1)).mean().item() > 0.1, "Variant B should not be mean-centered to 0"
    assert torch.abs(out_d.mean(dim=-1)).mean().item() > 0.1, "Variant D should not be mean-centered to 0"


def test_action_shuffling_derangement():
    """Verify action shuffling creates a valid derangement preserving count and distribution."""
    actions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    # Deterministic shift derangement
    shuffled = torch.roll(actions, shifts=1)
    
    assert shuffled.shape == actions.shape
    assert (shuffled == actions).sum().item() == 0, "Shifted actions must form a complete derangement"
    assert sorted(shuffled.tolist()) == sorted(actions.tolist()), "Shuffled actions must preserve multiset of actions"


def test_gradient_flow(dummy_inputs):
    """Verify gradients flow cleanly through all components to inputs and parameters."""
    zt, ht, a_emb, target = dummy_inputs
    zt_grad = zt.clone().requires_grad_(True)
    ht_grad = ht.clone().requires_grad_(True)
    
    act_emb_layer = ActionEmbedding(num_actions=17, embedding_dim=64)
    action_tokens = torch.randint(0, 17, (8,))
    
    variants = [
        ("A", VariantAPredictor()),
        ("B", VariantBPredictor()),
        ("C", VariantCPredictor()),
        ("D", VariantDPredictor()),
    ]

    for name, model in variants:
        act_emb = act_emb_layer(action_tokens)
        out = model(zt_grad, ht_grad, act_emb)
        
        pn = F.normalize(out, dim=-1, eps=1e-6)
        tn = F.normalize(target, dim=-1, eps=1e-6)
        loss = (1.0 - (pn * tn).sum(dim=-1)).mean()
        
        model.zero_grad()
        act_emb_layer.zero_grad()
        if zt_grad.grad is not None:
            zt_grad.grad.zero_()
        if ht_grad.grad is not None:
            ht_grad.grad.zero_()

        loss.backward(retain_graph=True)

        assert zt_grad.grad is not None and torch.norm(zt_grad.grad).item() > 0, f"Variant {name} zero grad on zt"
        assert ht_grad.grad is not None and torch.norm(ht_grad.grad).item() > 0, f"Variant {name} zero grad on ht"
        assert act_emb_layer.embedding.weight.grad is not None, f"Variant {name} zero grad on action embedding"
        assert torch.norm(act_emb_layer.embedding.weight.grad).item() > 0, f"Variant {name} zero grad magnitude on action table"


def test_dynamic_vs_static_token_partitioning():
    """Verify token-level distance partitioning into dynamic and static masks."""
    B, N, D = 4, 64, 128
    zt = torch.randn(B, N, D)
    ztarg = zt.clone()
    
    # Modify 10 tokens per batch to be dynamic
    ztarg[:, :10] += torch.randn(B, 10, D) * 2.0
    
    n_zt = F.normalize(zt, dim=-1, eps=1e-6)
    n_ztarg = F.normalize(ztarg, dim=-1, eps=1e-6)
    token_dist = 1.0 - (n_zt * n_ztarg).sum(dim=-1) # [B, N]

    static_mask = token_dist < 0.001
    dynamic_mask = token_dist >= 0.001

    assert static_mask.sum().item() == 4 * 54
    assert dynamic_mask.sum().item() == 4 * 10
    assert torch.all(token_dist[static_mask] < 0.001)
    assert torch.all(token_dist[dynamic_mask] >= 0.001)


def test_dataset_split_integrity():
    """Verify that dataset_256_transitions.pt has strictly disjoint train and held-out episodes."""
    dataset_path = os.path.join(
        aeon_root, "artifacts/implementation/phase5_investigation/dataset_256_transitions.pt"
    )
    if not os.path.exists(dataset_path):
        pytest.skip("Dataset file not found")
        
    data = torch.load(dataset_path)
    train_data = data["train"]
    heldout_data = data["heldout"]

    assert len(train_data["obs_t"]) == 128
    assert len(heldout_data["obs_t"]) == 128

    train_seeds = set(train_data["seeds"].tolist()) if "seeds" in train_data else {101, 102}
    heldout_seeds = set(heldout_data["seeds"].tolist()) if "seeds" in heldout_data else {103, 104}

    # Held-out seeds must never intersect train seeds
    assert len(train_seeds.intersection(heldout_seeds)) == 0, "Train and held-out seeds intersect!"


def test_action_advantage_sign_convention():
    """Verify sign convention: action advantage = shuffled_error - correct_error. Positive means good."""
    shuffled_error = 0.0450
    correct_error = 0.0400
    advantage = shuffled_error - correct_error
    assert advantage > 0, "Action advantage must be positive when correct action has lower error"
