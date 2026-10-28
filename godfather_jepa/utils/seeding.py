"""
Seeding utilities and module alias for Godfather JEPA.

Re-exports all constants, functions, and exceptions from `seed_guard.py`
to satisfy the module interface documented in PROJECT.md (`godfather_jepa/utils/seeding.py`).
"""
from __future__ import annotations

from godfather_jepa.utils.seed_guard import (
    DEV_SCREENING_COUNT,
    DEV_SCREENING_MAX,
    DEV_SCREENING_MIN,
    EPISODE_STRIDE,
    FORBIDDEN_EVAL_RANGES,
    GENERALIZATION_MAX,
    GENERALIZATION_MIN,
    GENERATION_STRIDE,
    LOCKED_BENCHMARK_COUNT,
    LOCKED_BENCHMARK_MAX,
    LOCKED_BENCHMARK_MIN,
    PROCEDURAL_SPAN,
    PROCEDURAL_TRAIN_MAX,
    PROCEDURAL_TRAIN_MIN,
    RANK_STRIDE,
    SeedLeakageError,
    assert_training_seed_pure,
    generate_procedural_seed,
    is_dev_screening_seed,
    is_evaluation_seed,
    is_generalization_seed,
    is_locked_benchmark_seed,
    is_procedural_training_seed,
)

__all__ = [
    "DEV_SCREENING_COUNT",
    "DEV_SCREENING_MAX",
    "DEV_SCREENING_MIN",
    "EPISODE_STRIDE",
    "FORBIDDEN_EVAL_RANGES",
    "GENERALIZATION_MAX",
    "GENERALIZATION_MIN",
    "GENERATION_STRIDE",
    "LOCKED_BENCHMARK_COUNT",
    "LOCKED_BENCHMARK_MAX",
    "LOCKED_BENCHMARK_MIN",
    "PROCEDURAL_SPAN",
    "PROCEDURAL_TRAIN_MAX",
    "PROCEDURAL_TRAIN_MIN",
    "RANK_STRIDE",
    "SeedLeakageError",
    "assert_training_seed_pure",
    "generate_procedural_seed",
    "is_dev_screening_seed",
    "is_evaluation_seed",
    "is_generalization_seed",
    "is_locked_benchmark_seed",
    "is_procedural_training_seed",
]
