"""Unit tests for zephyr.losses primitives."""
from __future__ import annotations

import math

import pytest
import torch

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


def test_symlog_roundtrip():
    x = torch.tensor([0.0, 0.5, -0.5, 3.0, -3.0, 20.0, -100.0])
    assert torch.allclose(inv_symlog(symlog(x)), x, atol=1e-5)


def test_twohot_encode_sums_to_one_and_decodes():
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    v = torch.tensor([-19.9, -1.0, 0.0, 0.37, 5.0, 19.9])
    dist = coder.encode(v)
    assert dist.shape == (6, 65)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(6), atol=1e-5)
    decoded = coder.decode(torch.log(dist.clamp(min=1e-9)))
    assert torch.allclose(decoded, v, atol=0.35)


def test_twohot_exact_hit():
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    dist = coder.encode(torch.tensor([0.0]))
    center_idx = 32  # symlog(0)=0 is the middle bin of 65
    assert float(dist[0, center_idx]) == pytest.approx(1.0, abs=1e-5)


def test_twohot_loss_decreases_toward_target():
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    v = torch.tensor([3.0])
    flat = torch.zeros(1, coder.num_bins)
    closer = torch.zeros(1, coder.num_bins)
    idx = int(torch.argmin((coder.bin_centers - symlog(v)) ** 2))
    closer[0, idx] = 4.0
    assert coder.loss(closer, v) < coder.loss(flat, v)


def test_lambda_returns_monte_carlo_case():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    continues = torch.ones(3, 1)
    values = torch.zeros(3, 1)
    bootstrap = torch.zeros(1)
    ret = lambda_returns(rewards, continues, values, bootstrap, gamma=1.0, td_lambda=1.0)
    assert ret.flatten().tolist() == [6.0, 5.0, 3.0]


def test_lambda_returns_zero_continue_truncates():
    rewards = torch.tensor([[1.0], [2.0]])
    continues = torch.tensor([[1.0], [0.0]])  # episode ends at t=1
    values = torch.zeros(2, 1)
    ret = lambda_returns(rewards, continues, values, torch.zeros(1), gamma=1.0, td_lambda=0.5)
    assert ret[1].item() == pytest.approx(2.0)   # terminal step: return is its own reward
    # t=0 continues: G_0 = 1 + gamma*c*((1-lam)*V_1 + lam*G_1) = 1 + 0.5*2
    assert ret[0].item() == pytest.approx(2.0)


def test_vicreg_terms():
    collapsed = torch.zeros(16, 64)
    assert vicreg_var_loss(collapsed) > 0.9  # std ~ 0 -> full hinge
    healthy = torch.randn(512, 64)
    assert vicreg_var_loss(healthy, gamma=1.0) < 0.05
    assert vicreg_cov_loss(healthy) >= 0.0


def test_focal_bce_rare_positive_learning():
    torch.manual_seed(0)
    logits = torch.full((512,), -3.0)          # predicts "never done"
    targets = torch.zeros(512)
    targets[::16] = 1.0                        # 1/16 positives
    before = focal_bce_with_logits(logits, targets)
    logits_pos = logits.clone()
    logits_pos[targets > 0.5] = 3.0            # predict positives
    after = focal_bce_with_logits(logits_pos, targets)
    assert after < before


def test_masked_mse_masks_and_never_nan():
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.zeros(1, 2, 2)
    mask = torch.tensor([[True, False]])
    val = masked_mse(pred, target, mask)
    assert val.item() == pytest.approx((1.0 + 4.0) / 2)
    empty = masked_mse(pred, target, torch.zeros(1, 2, dtype=torch.bool))
    assert float(empty) == 0.0


def test_roc_auc_perfect_and_random():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, targets) == pytest.approx(1.0)
    inverted = roc_auc(-scores, targets)
    assert inverted == pytest.approx(0.0)
    tied = roc_auc(torch.ones(4), targets)
    assert tied == pytest.approx(0.5)
    one_class = roc_auc(torch.tensor([0.5, 0.6]), torch.tensor([1.0, 1.0]))
    assert one_class == 0.5
