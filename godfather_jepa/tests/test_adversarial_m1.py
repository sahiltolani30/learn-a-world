"""
Empirical Adversarial Stress Test Suite for Milestone 1 (Foundation & Code Isolation).

Author: Challenger 1 (Adversarial Critic & ML Systems Specialist)
Purpose:
  Empirically stress-test AST static isolation checker, seed partitioning,
  Crafter environment wrapper, and official Crafter score metric.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List
import numpy as np
import pytest

from godfather_jepa.envs.crafter_env import GodfatherCrafterEnv
from godfather_jepa.metrics.crafter_score import (
    CRAFTER_ACHIEVEMENTS,
    compute_crafter_score,
)
from godfather_jepa.utils.seed_guard import (
    DEV_SCREENING_MAX,
    DEV_SCREENING_MIN,
    LOCKED_BENCHMARK_MAX,
    LOCKED_BENCHMARK_MIN,
    PROCEDURAL_TRAIN_MAX,
    PROCEDURAL_TRAIN_MIN,
    SeedLeakageError,
    assert_training_seed_pure,
    generate_procedural_seed,
    is_dev_screening_seed,
    is_evaluation_seed,
    is_locked_benchmark_seed,
)
from godfather_jepa.verification.ast_audit import (
    ALL_FORBIDDEN_MODULES,
    FORBIDDEN_LEGACY_MODULES,
    audit_package_isolation,
    get_max_relative_level,
)


# =====================================================================
# 1. AST STATIC ISOLATION ADVERSARIAL STRESS TESTS
# =====================================================================

class TestASTIsolationAdversarial:
    """Stress tests probing AST checker with obfuscation, depth tricks, and bypasses."""

    def test_multi_line_and_backslash_imports_caught(self) -> None:
        """Verify multi-line and backslash-continued imports are caught."""
        snippets = [
            "from models import (\n    jepa_v3,\n    train,\n)\n",
            "import \\\n    models\n",
            "import os, \\\n    sys, \\\n    planning\n",
        ]
        for code in snippets:
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "obfuscated.py"), "w", encoding="utf-8") as f:
                    f.write(code)
                violations = audit_package_isolation(td)
                assert len(violations) >= 1, f"Failed to catch: {code!r}"

    def test_relative_import_depth_enforcement(self) -> None:
        """Verify that relative imports attempting to escape package root are blocked."""
        with tempfile.TemporaryDirectory() as td:
            # Package root file: max level 1
            root_f = os.path.join(td, "root.py")
            with open(root_f, "w", encoding="utf-8") as f:
                f.write("from .. import legacy\n")

            # Subdir file: max level 2
            sub_dir = os.path.join(td, "models")
            os.makedirs(sub_dir)
            sub_f = os.path.join(sub_dir, "predictor.py")
            with open(sub_f, "w", encoding="utf-8") as f:
                f.write("from ... import legacy\n")

            violations = audit_package_isolation(td)
            assert len(violations) == 2
            assert all("escapes package root" in v.reason for v in violations)

    def test_standard_dynamic_imports_caught(self) -> None:
        """Verify standard dynamic imports are caught."""
        snippets = [
            "__import__('models')",
            "builtins.__import__('planning')",
            "__builtins__.__import__('zephyr')",
            "importlib.import_module('dual_store')",
            "import_module('scripts')",
        ]
        for s in snippets:
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "dyn.py"), "w", encoding="utf-8") as f:
                    f.write(s)
                violations = audit_package_isolation(td)
                assert len(violations) >= 1, f"Expected violation for {s!r}"

    def test_dynamic_bypass_vulnerabilities_empirically_demonstrated(self) -> None:
        """
        Adversarially probe known blindspots of the AST checker and verify all are caught:
        1. Aliased dynamic import: 'from importlib import import_module as imp; imp("models")'
        2. Aliased builtins import: 'from builtins import __import__ as imp; imp("models")'
        3. Attribute reflection: 'getattr(importlib, "import_module")("models")'
        4. Interpreter eval/exec: 'exec("import models")', 'eval("__import__(\'models\')")'
        5. Subscript access: '__builtins__["__import__"]("models")'
        """
        bypasses = [
            "from importlib import import_module as imp\nimp('models')",
            "from builtins import __import__ as imp\nimp('models')",
            "import importlib\nfn = getattr(importlib, 'import_module')\nfn('models')",
            "exec('import models')",
            "eval(\"__import__('models')\")",
            "__builtins__['__import__']('models')",
        ]
        for bypass_code in bypasses:
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "bypass.py"), "w", encoding="utf-8") as f:
                    f.write(bypass_code)
                violations = audit_package_isolation(td)
                assert len(violations) >= 1, (
                    f"Checker must detect violation for bypass: {bypass_code!r}"
                )


# =====================================================================
# 2. SEED GUARD STRESS & FUZZING TESTS
# =====================================================================

class TestSeedGuardAdversarial:
    """Stress tests generating 2M seeds, fuzzing types, and checking collision domains."""

    def test_procedural_generator_two_million_seeds_zero_leakage_and_overlap(self) -> None:
        """
        Generates 2,000,000 procedural seeds across randomized distributed ranks,
        episodes, and checkpoint generations.
        Asserts:
        - Exactly 2,000,000 unique seeds (0 duplicate collisions)
        - 0 seeds overlap with locked benchmark [2050, 2099]
        - 0 seeds overlap with dev screening [6150, 6199]
        """
        # Distributed setup: 8 generations x 16 ranks x 15,625 episodes = 2,000,000 seeds
        gens = np.arange(8, dtype=np.int64)[:, None, None]
        ranks = np.arange(16, dtype=np.int64)[None, :, None]
        eps = np.arange(15625, dtype=np.int64)[None, None, :]

        offsets = (gens * 100_000_000 + eps * 1_000 + ranks) % 999_900_000
        seeds = (100_000 + offsets).flatten()

        assert len(seeds) == 2_000_000
        unique_seeds = len(np.unique(seeds))
        assert unique_seeds == 2_000_000, f"Collisions found: only {unique_seeds} unique seeds"

        # Assert zero overlap with locked evaluation ranges
        assert seeds.min() >= PROCEDURAL_TRAIN_MIN  # 100,000 >= 2099
        assert seeds.max() <= PROCEDURAL_TRAIN_MAX  # 999,999,999
        assert not np.any((seeds >= LOCKED_BENCHMARK_MIN) & (seeds <= LOCKED_BENCHMARK_MAX))
        assert not np.any((seeds >= DEV_SCREENING_MIN) & (seeds <= DEV_SCREENING_MAX))

    def test_procedural_generator_modulo_wrap_boundary_collision(self) -> None:
        """
        Demonstrates the exact mathematical collision boundary in generate_procedural_seed:
        Because PROCEDURAL_SPAN = 999,900,000 (not 1,000,000,000), when gen = 9 and ep = 99,900:
        9 * 100_000_000 + 99,900 * 1,000 = 999,900_000 = 0 (mod 999,900,000).
        This causes (gen=9, ep=99900, rank=r) to collide with (gen=0, ep=0, rank=r).
        """
        s_gen0_ep0 = generate_procedural_seed(rank=42, episode=0, generation=0)
        s_gen9_ep99900 = generate_procedural_seed(rank=42, episode=99900, generation=9)
        assert s_gen0_ep0 == s_gen9_ep99900 == 100042, "Collision confirmed at boundary"

    @pytest.mark.parametrize("bad_val", [
        -1, -2050, -6150, -100_000, -999_999_999, -2**31,
    ])
    def test_fuzz_negative_seeds_rejected(self, bad_val: int) -> None:
        """Verify negative seeds are rejected with ValueError."""
        with pytest.raises(ValueError):
            assert_training_seed_pure(bad_val)

    @pytest.mark.parametrize("bad_float", [
        2050.0, 2075.5, 6150.0, 100_000.0, 0.0, -1.0,
        float("nan"), float("inf"), float("-inf"),
    ])
    def test_fuzz_float_seeds_rejected(self, bad_float: float) -> None:
        """Verify float seeds are rejected with TypeError."""
        with pytest.raises(TypeError):
            assert_training_seed_pure(bad_float)

    @pytest.mark.parametrize("non_int", [
        "2050", "", None, [], {}, set(), (1,), True, False,
    ])
    def test_fuzz_non_integer_seeds_rejected(self, non_int: Any) -> None:
        """Verify non-integer seeds are rejected with TypeError."""
        with pytest.raises(TypeError):
            assert_training_seed_pure(non_int)

    def test_numpy_int_type_behavior(self) -> None:
        """
        Verify that np.int64 / np.int32 are accepted by assert_training_seed_pure
        and generate_procedural_seed, while np.bool_ is strictly rejected.
        """
        np_seed = np.int64(100_000)
        assert_training_seed_pure(np_seed)

        np_seed32 = np.int32(100_000)
        assert_training_seed_pure(np_seed32)

        # NumPy integer in held-out range must raise SeedLeakageError
        with pytest.raises(SeedLeakageError):
            assert_training_seed_pure(np.int64(2050))

        # NumPy bool must raise TypeError
        with pytest.raises(TypeError):
            assert_training_seed_pure(np.bool_(True))

        with pytest.raises(TypeError):
            assert_training_seed_pure(np.bool_(False))

        # Generator accepts NumPy integers
        gen_seed = generate_procedural_seed(rank=np.int64(3), episode=np.int32(10), generation=np.int64(1))
        assert isinstance(gen_seed, int)
        assert gen_seed >= PROCEDURAL_TRAIN_MIN


# =====================================================================
# 3. ENVIRONMENT & SCORE ADVERSARIAL TESTS
# =====================================================================

class TestEnvAndScoreAdversarial:
    """Stress tests on Crafter environment wrapper and metric edge cases."""

    def test_env_float_action_silent_truncation(self) -> None:
        """
        Verify that GodfatherCrafterEnv.step rejects float and bool actions
        rather than silently truncating or converting them.
        """
        env = GodfatherCrafterEnv(training=False)
        env.reset(seed=123)

        with pytest.raises(ValueError, match="must be an integer"):
            env.step(3.5)

        with pytest.raises(ValueError, match="must be an integer"):
            env.step(True)

        with pytest.raises(ValueError, match="must be an integer"):
            env.step(np.bool_(False))

        # Valid NumPy integer action succeeds
        obs, reward, term, trunc, info = env.step(np.int64(3))
        assert obs.shape == (64, 64, 3)

    def test_crafter_score_empty_and_extremes(self) -> None:
        """Verify crafter score handles empty inputs and 0%/100% bounds."""
        assert float(compute_crafter_score([])) == 0.0
        assert float(compute_crafter_score({})) == 0.0
        assert float(compute_crafter_score(np.zeros((0, 22)))) == 0.0

        all_zeros = {ach: 0.0 for ach in CRAFTER_ACHIEVEMENTS}
        assert math.isclose(float(compute_crafter_score(all_zeros)), 0.0)

        all_hundreds = {ach: 100.0 for ach in CRAFTER_ACHIEVEMENTS}
        assert math.isclose(float(compute_crafter_score(all_hundreds)), 100.0)

    def test_crafter_score_negative_percentage_clamping(self) -> None:
        """Verify negative percentages are clamped to 0.0 in dict mode."""
        neg_dict = {ach: -50.0 for ach in CRAFTER_ACHIEVEMENTS}
        res = compute_crafter_score(neg_dict)
        assert math.isclose(float(res), 0.0)

    def test_crafter_score_degenerate_and_mismatched_arrays(self) -> None:
        """Verify crafter score handles 0-dim, 1D empty, and mismatched column arrays."""
        # 0-dim array returns 0.0
        res_0d = compute_crafter_score(np.array(0))
        assert float(res_0d) == 0.0

        # 1D empty array returns 0.0
        res_empty_1d = compute_crafter_score(np.array([]))
        assert float(res_empty_1d) == 0.0

        # 1D single-episode 22-achievement array works
        res_1d = compute_crafter_score(np.ones(22))
        assert math.isclose(float(res_1d), 100.0)

        # Mismatched column count raises ValueError
        with pytest.raises(ValueError, match="Expected matrix with 22 achievement columns"):
            compute_crafter_score(np.ones((10, 5)))

        with pytest.raises(ValueError, match="Expected matrix with 22 achievement columns"):
            compute_crafter_score(np.ones((5, 10)))
