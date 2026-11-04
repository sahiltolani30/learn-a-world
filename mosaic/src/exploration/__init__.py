"""Exploration, calibration, and active probing package for MOSAIC Phase P3."""

from .calibration import compute_calibration_curve, rank_correlation
from .active_probe import DualValueHeads, ActiveProbeScorer, ActiveProbePolicy
from .runner_p3 import Phase3Config, run_phase_3

__all__ = [
    "compute_calibration_curve",
    "rank_correlation",
    "DualValueHeads",
    "ActiveProbeScorer",
    "ActiveProbePolicy",
    "Phase3Config",
    "run_phase_3",
]
