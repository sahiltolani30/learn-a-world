"""Verification tests for G4 Calibrated Survival Repair (atlas/safety.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atlas.safety import ContinueNet, calibrate_risk_threshold


def test_g4_calibrate_risk_threshold_selects_optimal():
    """Synthetic risk scores for 10 deaths (mostly 0.45-0.55) + 1000 survived (mostly 0.05-0.35)
    -> grid selects 0.40, deaths_flagged >= 60%.
    """
    rng = np.random.RandomState(42)

    # 10 deaths: 8 around 0.45-0.55 (mean ~0.50), 2 around 0.25
    death_risks = np.array([0.48, 0.52, 0.46, 0.49, 0.54, 0.47, 0.51, 0.53, 0.25, 0.22])
    # 1000 survived steps: mostly 0.05..0.38 (some in 0.30..0.38 penalizing tau <= 0.35)
    survived_risks = rng.uniform(0.05, 0.38, size=1000)

    chosen_thresh, summary = calibrate_risk_threshold(death_risks, survived_risks)

    # At threshold 0.40:
    # 8 of 10 deaths are flagged (80% >= 60%)
    # Zero or near-zero survived steps are flagged (fpr ≈ 0)
    # Utility at 0.40 is 0.8 - 0 = 0.8
    # At 0.50: only 4 deaths flagged (40% < 60%)
    assert chosen_thresh == 0.40, f"Expected 0.40, got {chosen_thresh}"
    assert summary["deaths_flagged_rate"] >= 0.60


def test_g4_all_survived_fixture():
    """All-survived fixture -> threshold unchanged at 0.50, null grid row for deaths."""
    survived_risks = np.array([0.1, 0.2, 0.15, 0.05])
    death_risks = np.array([])  # 0 deaths

    chosen_thresh, summary = calibrate_risk_threshold(death_risks, survived_risks)
    assert chosen_thresh == 0.50
    assert summary["grid"] is None
    assert summary["deaths_flagged_rate"] is None
