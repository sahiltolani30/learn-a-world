"""
Metrics and scoring engine for Godfather JEPA.
"""

from .crafter_score import (
    ACHIEVEMENT_TIERS,
    ACHIEVEMENT_TO_IDX,
    CRAFTER_ACHIEVEMENTS,
    compute_crafter_score,
    matrix_to_records,
    records_to_matrix,
)

__all__ = [
    "ACHIEVEMENT_TIERS",
    "ACHIEVEMENT_TO_IDX",
    "CRAFTER_ACHIEVEMENTS",
    "compute_crafter_score",
    "matrix_to_records",
    "records_to_matrix",
]
