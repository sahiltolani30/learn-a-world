"""Tier 2: Boundary Tests - Symlog Extremes in Distributional Critic.

Category 4: Extreme Returns & Symlog Two-Hot Edges
Tests:
19. test_t2_symlog_extreme_positive_saturation
20. test_t2_symlog_extreme_negative_saturation
21. test_t2_symlog_exact_zero_and_infinitesimal_symmetry
22. test_t2_symlog_boundary_range_limit_20
23. test_t2_twohot_loss_with_extreme_logits
24. test_t2_critic_flat_prediction_collapse_guard
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from zephyr.losses import TwoHotCoder, inv_symlog, symlog


def test_t2_symlog_extreme_positive_saturation():
    """T2-Symlog: Return value +10,000 compresses to ~9.21 in symlog space; encodes and decodes smoothly."""
    v = torch.tensor([10_000.0], dtype=torch.float32)
    s = symlog(v)
    expected_symlog = math.log(10_001.0)
    assert s.item() == pytest.approx(expected_symlog, abs=1e-4)

    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    dist = coder.encode(v)

    assert dist.shape == (1, 65)
    assert torch.isfinite(dist).all(), "Encoded distribution contains non-finite values"
    assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)

    # Convert probability distribution to logits and decode
    # Uniform logits with probabilities matching dist
    logits = torch.log(dist + 1e-12)
    decoded_v = coder.decode(logits)
    assert torch.isfinite(decoded_v).all()
    # Decoding two-hot interpolation reconstructs value accurately within bin step resolution
    assert decoded_v.item() == pytest.approx(10_000.0, rel=0.05)


def test_t2_symlog_extreme_negative_saturation():
    """T2-Symlog: Return value -10,000 compresses to -9.21 in symlog space; encodes and decodes smoothly."""
    v = torch.tensor([-10_000.0], dtype=torch.float32)
    s = symlog(v)
    expected_symlog = -math.log(10_001.0)
    assert s.item() == pytest.approx(expected_symlog, abs=1e-4)

    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    dist = coder.encode(v)

    assert dist.shape == (1, 65)
    assert torch.isfinite(dist).all(), "Encoded distribution contains non-finite values"
    assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)

    logits = torch.log(dist + 1e-12)
    decoded_v = coder.decode(logits)
    assert torch.isfinite(decoded_v).all()
    assert decoded_v.item() == pytest.approx(-10_000.0, rel=0.05)


def test_t2_symlog_exact_zero_and_infinitesimal_symmetry():
    """T2-Symlog: Symlog around 0 is strictly anti-symmetric and maps 0.0 cleanly to center bin 32."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    eps_vals = torch.tensor([1e-7, 0.0, -1e-7], dtype=torch.float32)

    s = symlog(eps_vals)
    # Anti-symmetry: symlog(-x) == -symlog(x)
    assert s[0].item() == pytest.approx(-s[2].item(), abs=1e-10)
    assert s[1].item() == 0.0

    # At 0.0, center bin is index 32
    dist_zero = coder.encode(torch.tensor([0.0]))
    assert dist_zero[0, 32].item() == pytest.approx(1.0, abs=1e-5)
    assert dist_zero[0, :32].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert dist_zero[0, 33:].sum().item() == pytest.approx(0.0, abs=1e-6)


def test_t2_symlog_boundary_range_limit_20():
    """T2-Symlog: Values at and exceeding the +-20 symlog range clamp cleanly to edge bins 0 and 64."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    # Value corresponding to symlog = 20.0 is exp(20) - 1 ≈ 4.85e8
    extreme_pos = inv_symlog(torch.tensor([20.0]))
    extreme_neg = inv_symlog(torch.tensor([-20.0]))
    beyond_pos = torch.tensor([1e15], dtype=torch.float32)
    beyond_neg = torch.tensor([-1e15], dtype=torch.float32)

    dist_pos = coder.encode(extreme_pos)
    dist_neg = coder.encode(extreme_neg)
    dist_beyond_pos = coder.encode(beyond_pos)
    dist_beyond_neg = coder.encode(beyond_neg)

    # Edge bins are 0 and 64
    assert dist_pos[0, 64].item() == pytest.approx(1.0, abs=1e-4)
    assert dist_neg[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert dist_beyond_pos[0, 64].item() == pytest.approx(1.0, abs=1e-4)
    assert dist_beyond_neg[0, 0].item() == pytest.approx(1.0, abs=1e-4)


def test_t2_twohot_loss_with_extreme_logits():
    """T2-Symlog: Log-sum-exp numerical stability prevents NaNs/Infs even with extreme logits +-1000."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    extreme_logits = torch.zeros(2, 65)
    extreme_logits[0, 0] = -1000.0
    extreme_logits[0, 64] = 1000.0
    extreme_logits[1, 32] = 1000.0

    targets = torch.tensor([10.0, 0.0])
    loss = coder.loss(extreme_logits, targets)

    assert torch.isfinite(loss), f"Loss computed from extreme logits was not finite: {loss}"
    assert loss.item() >= 0.0


def test_t2_critic_flat_prediction_collapse_guard():
    """T2-Symlog: Gradients on flat logits are non-zero and point directly towards true target bins."""
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    # Flat logits representing uninformative / collapsed prediction
    flat_logits = torch.zeros(1, 65, requires_grad=True)

    # Target is positive return +5.0 (symlog(5.0) = ln(6) ≈ 1.79)
    target = torch.tensor([5.0])
    loss = coder.loss(flat_logits, target)

    loss.backward()

    assert flat_logits.grad is not None
    assert torch.isfinite(flat_logits.grad).all()
    # Target bin should have negative gradient (increasing logit lowers loss)
    target_dist = coder.encode(target)
    target_bin = int(target_dist.argmax(dim=-1).item())
    assert flat_logits.grad[0, target_bin].item() < 0.0, "Gradient did not pull target bin upward"
