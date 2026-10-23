"""
Unit & Smoke Tests for EventHeadCalibrator (Part 1).
Runs purely on CPU with synthetic tensors (under 1 second).
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from models.calibration import EventHeadCalibrator, compute_binary_ece
from models.heads import EventHead
from models.jepa_v3 import JEPAv3


def test_event_head_calibrator_synthetic():
    """Smoke test: EventHeadCalibrator fits, reduces ECE, and preserves frontier heads."""
    np.random.seed(42)
    N = 1000
    num_heads = 22

    # Synthesize logits and labels
    # Head 0: Active head with strong signal but uncalibrated logit offset
    true_prob_0 = np.random.uniform(0.1, 0.4, size=N)
    labels_0 = (np.random.rand(N) < true_prob_0).astype(np.int32)
    # Model under-predicts logits (e.g. -4.0 offset)
    logits_0 = np.log(true_prob_0 / (1.0 - true_prob_0)) - 3.5

    # Head 1: Frontier zero-data head
    labels_1 = np.zeros(N, dtype=np.int32)
    logits_1 = np.random.normal(-6.0, 1.0, size=N)

    logits = np.random.normal(-5.0, 1.0, size=(N, num_heads))
    labels = np.zeros((N, num_heads), dtype=np.int32)

    logits[:, 0] = logits_0
    labels[:, 0] = labels_0
    logits[:, 1] = logits_1
    labels[:, 1] = labels_1

    calibrator = EventHeadCalibrator(num_heads=num_heads, fit_bias=True, min_positives=5)
    calibrator.fit(logits, labels)

    # 1. Frontier head must remain identity (T=1, b=0)
    assert calibrator.temperatures[1] == 1.0
    assert calibrator.biases[1] == 0.0
    assert calibrator.active_mask[1] == False

    # 2. Active head 0 must be calibrated and active
    assert calibrator.active_mask[0] == True
    assert calibrator.temperatures[0] > 0.0

    # 3. Check apply on NumPy array
    cal_probs_np = calibrator.apply(logits)
    assert cal_probs_np.shape == (N, num_heads)
    assert np.all(cal_probs_np >= 0.0) and np.all(cal_probs_np <= 1.0)

    # Active head recall@0.5 should increase or ECE should decrease
    ece_before = compute_binary_ece(1.0 / (1.0 + np.exp(-logits[:, 0])), labels[:, 0])
    ece_after = compute_binary_ece(cal_probs_np[:, 0], labels[:, 0])
    assert ece_after <= ece_before + 1e-3, f"ECE did not improve: before={ece_before}, after={ece_after}"

    # 4. Check apply on PyTorch Tensor
    t_logits = torch.from_numpy(logits).float()
    cal_probs_t = calibrator.apply(t_logits)
    assert isinstance(cal_probs_t, torch.Tensor)
    assert cal_probs_t.shape == (N, num_heads)
    assert torch.all(cal_probs_t >= 0.0) and torch.all(cal_probs_t <= 1.0)
    np.testing.assert_allclose(cal_probs_t.numpy(), cal_probs_np, atol=1e-5)


def test_calibrator_save_load_roundtrip():
    """Smoke test: JSON serialization and deserialization."""
    calibrator = EventHeadCalibrator(num_heads=22, fit_bias=True)
    calibrator.temperatures[3] = 1.85
    calibrator.biases[3] = 0.42
    calibrator.active_mask[3] = True

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "calibrator.json"
        calibrator.save(json_path)
        assert json_path.exists()

        loaded = EventHeadCalibrator.load(json_path)
        assert loaded.num_heads == 22
        assert loaded.fit_bias == True
        np.testing.assert_allclose(loaded.temperatures, calibrator.temperatures)
        np.testing.assert_allclose(loaded.biases, calibrator.biases)
        np.testing.assert_array_equal(loaded.active_mask, calibrator.active_mask)


def test_event_head_integration():
    """Smoke test: EventHead and JEPAv3 seamlessly use attached calibrator."""
    head = EventHead(latent_dim=64, num_actions=17, num_achievements=22)
    calibrator = EventHeadCalibrator(num_heads=22, fit_bias=True)
    calibrator.biases[0] = 5.0  # Big bias on head 0 to guarantee noticeable shift

    z = torch.randn(4, 64)
    action = torch.zeros(4, dtype=torch.long)
    mask = torch.zeros(4, 22)

    # 1. Without calibrator: standard sigmoid
    raw_p = head.predict_proba(z, action, mask)
    assert raw_p.shape == (4, 22)

    # 2. Attach calibrator
    head.set_calibrator(calibrator)
    cal_p = head.predict_proba(z, action, mask, use_calibration=True)
    assert cal_p.shape == (4, 22)

    # Head 0 probability should be significantly higher due to bias
    assert (cal_p[:, 0] > raw_p[:, 0]).all()

    # 3. Can bypass calibration with use_calibration=False
    uncal_p = head.predict_proba(z, action, mask, use_calibration=False)
    torch.testing.assert_close(uncal_p, raw_p)
