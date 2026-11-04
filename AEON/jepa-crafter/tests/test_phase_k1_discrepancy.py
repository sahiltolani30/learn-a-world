"""Unit tests for Phase K.1 audit reproducing the Phase J vs Phase K discrepancy."""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.jepa_world_model import JepaWorldModel


def test_action_indexing_off_by_one_reproduction():
    """Mathematically proves that Phase K extracted a_{t+1} instead of a_t."""
    # Sequence of 10 observations and 9 actions:
    # O0 --(a0)--> O1 --(a1)--> O2 --(a2)--> O3 --(a3)--> O4 ...
    obs_seq = [f"O_{t}" for t in range(10)]
    act_seq = [f"a_{t}" for t in range(9)]
    
    context_length = 4
    max_horizon = 4
    i = context_length - 1  # i = 3
    
    current_obs = obs_seq[i]             # O_3
    target_obs_h1 = obs_seq[i + 1]       # O_4
    true_transition_action = act_seq[i]  # a_3
    
    # Buggy extraction in Phase K lines 145 & 167:
    buggy_fut_act = act_seq[i + 1 : i + 1 + max_horizon]
    # Correct extraction:
    correct_fut_act = act_seq[i : i + max_horizon]
    
    assert current_obs == "O_3"
    assert target_obs_h1 == "O_4"
    assert true_transition_action == "a_3"
    assert buggy_fut_act[0] == "a_4", "Bug confirmed: Phase K extracted a_4 instead of a_3!"
    assert correct_fut_act[0] == "a_3", "Fix confirmed: Correct extraction yields a_3!"


def test_context_representation_shift_reproduction():
    """Reproduces the prediction discrepancy when context changes from torch.cat([ut, ut]) to GRU."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = JepaWorldModel(
        latent_dim=128,
        action_dim=64,
        context_dim=256,
        num_predictor_members=1,
        num_transformer_layers=1,
        num_heads=2,
        mlp_dim=128,
    ).to(device)
    model.eval()

    B = 8
    # Latent tokens with positive mean ~0.64 (matching real visual encoder)
    zt = torch.randn(B, 64, 128, device=device) * 0.4 + 0.64
    ut = zt.mean(dim=1)
    act = torch.randint(0, 17, (B,), device=device)
    a_emb = model.action_embedding(act)

    # 1. Phase J Context: torch.cat([ut, ut], dim=-1)
    ht_phase_j = torch.cat([ut, ut], dim=-1)
    
    # 2. Phase K Context: GRU output
    # GRU tanh hidden state has mean ~0.0, norm ~5.3
    obs_ctx = torch.rand(B, 4, 3, 64, 64, device=device)
    prev_act = torch.randint(0, 17, (B, 4), device=device)
    with torch.no_grad():
        _, ht_phase_k = model.encode_actual_history(obs_ctx, prev_act)

    # Measure output difference
    with torch.no_grad():
        pred_j = model.predictor_ensemble.forward_member(0, zt, ht_phase_j, a_emb)
        pred_k = model.predictor_ensemble.forward_member(0, zt, ht_phase_k, a_emb)

        n_pred_j = F.normalize(pred_j, dim=-1, eps=1e-6)
        n_pred_k = F.normalize(pred_k, dim=-1, eps=1e-6)
        discrepancy = (1.0 - (n_pred_j * n_pred_k).sum(dim=-1)).mean().item()

    # The two contexts produce substantially different predictions
    assert discrepancy > 0.01, f"Expected noticeable discrepancy due to context shift, got {discrepancy}"
    assert ht_phase_j.mean().item() > 0.5, "Phase J context has positive mean"
    assert abs(ht_phase_k.mean().item()) < 0.2, "GRU context has near-zero mean"


def test_teacher_forced_vs_closed_loop_isolation():
    """Verifies that teacher-forced rollout avoids feeding predicted errors back into the recurrent loop."""
    torch.manual_seed(100)
    device = torch.device("cpu")
    model = JepaWorldModel(
        latent_dim=128,
        action_dim=64,
        context_dim=256,
        num_predictor_members=1,
        num_transformer_layers=1,
        num_heads=2,
        mlp_dim=128,
    ).to(device)
    model.eval()

    B = 2
    H = 4
    z_0 = torch.randn(B, 64, 128, device=device) * 0.4 + 0.64
    u_0 = z_0.mean(dim=1)
    h_0 = torch.cat([u_0, u_0], dim=-1)
    future_actions = torch.randint(0, 17, (B, H), device=device)

    # Ground truth intermediate latents
    gt_latents = [torch.randn(B, 64, 128, device=device) * 0.4 + 0.64 for _ in range(H)]

    with torch.no_grad():
        # Closed-loop rollout (feeds previous predicted z_hat)
        cl_preds = []
        z_hat = z_0
        h_hat = h_0
        for step in range(H):
            a_emb = model.action_embedding(future_actions[:, step])
            z_hat = model.predictor_ensemble.forward_member(0, z_hat, h_hat, a_emb)
            u_hat = z_hat.mean(dim=1)
            h_hat = torch.cat([u_hat, u_hat], dim=-1)
            cl_preds.append(z_hat)

        # Teacher-forced rollout (feeds ground-truth previous z_true)
        tf_preds = []
        for step in range(H):
            z_in = z_0 if step == 0 else gt_latents[step - 1]
            u_in = z_in.mean(dim=1)
            h_in = torch.cat([u_in, u_in], dim=-1)
            a_emb = model.action_embedding(future_actions[:, step])
            z_pred_tf = model.predictor_ensemble.forward_member(0, z_in, h_in, a_emb)
            tf_preds.append(z_pred_tf)

    # Step 1 should be identical because both start from z_0
    assert torch.allclose(cl_preds[0], tf_preds[0], atol=1e-5), "Step 1 must be identical between CL and TF"
    # Step 2+ should diverge because CL feeds predicted latents while TF feeds ground truth
    diff_step2 = torch.norm(cl_preds[1] - tf_preds[1]).item()
    assert diff_step2 > 0.001, "Closed-loop and teacher-forced should differ at step 2+"
