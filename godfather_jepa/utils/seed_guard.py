"""
Seed Space Partitioning and Runtime Guard for Godfather JEPA.

Enforces strict physical separation between training procedural seeds and held-out
evaluation seeds, guaranteeing zero benchmark leakage and zero seed-memorization.

Seed Space Partition Map:
  - [0, 2049]           : Legacy / Archive Seed Space (Unused)
  - [2050, 2099] (50)   : 🔒 HELD-OUT LOCKED BENCHMARK (Strictly Evaluation Only)
  - [2100, 6149]        : Reserved Buffer Space
  - [6150, 6199] (50)   : 🔒 HELD-OUT DEV SCREENING (Validation Screening Only)
  - [6200, 9999]        : Reserved Buffer Space
  - [10000, 99999]      : Arbitrary Unseen Generalization Test Pool (Evaluations)
  - [100000, 10^9 - 1]  : Procedural Training Seed Stream (Training Interactive Envs)
"""
from __future__ import annotations

from typing import Tuple, Union
import numpy as np

# =====================================================================
# Seed Space Partition Constants
# =====================================================================

# 1. Held-out Locked Benchmark Range (Strictly Evaluation Only)
LOCKED_BENCHMARK_MIN: int = 2050
LOCKED_BENCHMARK_MAX: int = 2099
LOCKED_BENCHMARK_COUNT: int = LOCKED_BENCHMARK_MAX - LOCKED_BENCHMARK_MIN + 1  # 50

# 2. Held-out Dev Screening Validation Range (Strictly Evaluation Only)
DEV_SCREENING_MIN: int = 6150
DEV_SCREENING_MAX: int = 6199
DEV_SCREENING_COUNT: int = DEV_SCREENING_MAX - DEV_SCREENING_MIN + 1  # 50

# Tuple of all forbidden evaluation ranges: (start, end) inclusive
FORBIDDEN_EVAL_RANGES: Tuple[Tuple[int, int], ...] = (
    (LOCKED_BENCHMARK_MIN, LOCKED_BENCHMARK_MAX),
    (DEV_SCREENING_MIN, DEV_SCREENING_MAX),
)

# 3. Arbitrary Unseen Generalization Test Pool (Unseen Layout Evaluations)
GENERALIZATION_MIN: int = 10_000
GENERALIZATION_MAX: int = 99_999

# 4. Procedural Training Seed Stream (Training Interactive Environments)
PROCEDURAL_TRAIN_MIN: int = 100_000
PROCEDURAL_TRAIN_MAX: int = 999_999_999
PROCEDURAL_SPAN: int = PROCEDURAL_TRAIN_MAX - PROCEDURAL_TRAIN_MIN + 1  # 999,900,000

# Procedural Generator Strides (Mixed-Radix Decomposition)
RANK_STRIDE: int = 1
EPISODE_STRIDE: int = 1_000
GENERATION_STRIDE: int = 100_000_000


# =====================================================================
# Exception Definition
# =====================================================================

class SeedLeakageError(ValueError, RuntimeError):
    """
    Raised when training environments or procedural generators attempt to
    access or train on held-out evaluation seeds.
    Inherits from both ValueError and RuntimeError for comprehensive exception catching.
    """
    pass


# =====================================================================
# Runtime Assertions & Queries
# =====================================================================

def assert_training_seed_pure(seed: Union[int, np.integer]) -> None:
    """
    Verifies that a given seed is strictly pure for training and does not
    leak or overlap with held-out evaluation ranges.

    Args:
        seed: Integer seed to validate (int or np.integer).

    Raises:
        TypeError: If seed is not an integer (or is a boolean).
        ValueError: If seed is negative.
        SeedLeakageError: If seed falls in locked benchmark [2050, 2099]
                          or dev screening [6150, 6199].
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}: {seed}")

    seed = int(seed)

    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    for s_min, s_max in FORBIDDEN_EVAL_RANGES:
        if s_min <= seed <= s_max:
            range_name = (
                "Locked Benchmark Evaluation Harness"
                if (s_min, s_max) == (LOCKED_BENCHMARK_MIN, LOCKED_BENCHMARK_MAX)
                else "Dev Screening Validation Pool"
            )
            raise SeedLeakageError(
                f"FATAL SEED LEAKAGE DETECTED: Seed {seed} falls within the forbidden "
                f"held-out range [{s_min}, {s_max}] ({range_name}). "
                f"Training environments must NEVER touch or train on evaluation seeds!"
            )


def is_evaluation_seed(seed: int) -> bool:
    """Returns True if seed falls in any reserved evaluation range."""
    return any(s_min <= seed <= s_max for s_min, s_max in FORBIDDEN_EVAL_RANGES)


def is_locked_benchmark_seed(seed: int) -> bool:
    """Returns True if seed falls in locked benchmark evaluation range [2050, 2099]."""
    return LOCKED_BENCHMARK_MIN <= seed <= LOCKED_BENCHMARK_MAX


def is_dev_screening_seed(seed: int) -> bool:
    """Returns True if seed falls in dev screening validation range [6150, 6199]."""
    return DEV_SCREENING_MIN <= seed <= DEV_SCREENING_MAX


def is_procedural_training_seed(seed: int) -> bool:
    """Returns True if seed falls in procedural training range [100000, 999999999]."""
    return PROCEDURAL_TRAIN_MIN <= seed <= PROCEDURAL_TRAIN_MAX


def is_generalization_seed(seed: int) -> bool:
    """Returns True if seed falls in arbitrary generalization test pool [10000, 99999]."""
    return GENERALIZATION_MIN <= seed <= GENERALIZATION_MAX


# =====================================================================
# Procedural Training Seed Generator
# =====================================================================

def generate_procedural_seed(
    rank: Union[int, np.integer],
    episode: Union[int, np.integer] = 0,
    generation: Union[int, np.integer] = 0,
) -> int:
    """
    Generates a deterministic procedural training seed for a given worker rank,
    episode index, and resumption generation counter.

    Guarantees:
    1. Returns an integer in [100,000, 999,999,999].
    2. Mathematically guaranteed never to intersect held-out ranges:
       [2050, 2099] and [6150, 6199].
    3. Guarantees zero duplicate collisions across standard training regimes:
       up to 9 full checkpoint generations (0..8) with 100,000 episodes and 1,000 ranks.
    4. Automatically invokes assert_training_seed_pure(seed) before returning
       as a defense-in-depth safety assertion.

    Args:
        rank: Worker environment index (>= 0).
        episode: Episode counter for the worker (>= 0).
        generation: Checkpoint resumption generation counter (>= 0).

    Returns:
        int: Procedural training seed >= 100,000 and <= 999,999,999.

    Raises:
        TypeError: If any parameter is not an integer.
        ValueError: If any parameter is negative.
        SeedLeakageError: If defense-in-depth purity assertion fails.
    """
    for param_name, val in [("rank", rank), ("episode", episode), ("generation", generation)]:
        if not isinstance(val, (int, np.integer)) or isinstance(val, (bool, np.bool_)):
            raise TypeError(
                f"Parameter '{param_name}' must be an integer, got {type(val).__name__}: {val}"
            )
        if int(val) < 0:
            raise ValueError(f"Parameter '{param_name}' must be non-negative, got {val}")

    rank_int = int(rank)
    episode_int = int(episode)
    generation_int = int(generation)

    offset = (
        generation_int * GENERATION_STRIDE
        + episode_int * EPISODE_STRIDE
        + rank_int * RANK_STRIDE
    ) % PROCEDURAL_SPAN

    seed = int(PROCEDURAL_TRAIN_MIN + offset)

    # Defense-in-depth runtime assertion
    assert_training_seed_pure(seed)

    return seed
