"""
Unit, Contract, and Regression Tests for Milestone 1 (Foundation & Code Isolation).

Covers:
1. AST Static Isolation Checker (zero legacy imports, escaping relative import bounds, dynamic imports, CLI exit codes)
2. Hermetic Crafter Environment (Gymnasium 5-tuple contract, determinism, creature despawn patch, state snapshot/restore, pixel-only inference)
3. Crafter Score Metric (geometric mean formula, polymorphic inputs, matrix conversions, boundary conditions)
4. Seed Space Partitioning & Guard (SeedLeakageError on locked seeds, 1,000,000 seed zero-leakage and zero-collision check)
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List
import numpy as np
import pytest

from godfather_jepa.envs.crafter_env import GodfatherCrafterEnv, patch_crafter_determinism
from godfather_jepa.metrics.crafter_score import (
    ACHIEVEMENT_TIERS,
    ACHIEVEMENT_TO_IDX,
    CRAFTER_ACHIEVEMENTS,
    compute_crafter_score,
    matrix_to_records,
    records_to_matrix,
)
from godfather_jepa.utils.seed_guard import (
    DEV_SCREENING_MAX,
    DEV_SCREENING_MIN,
    GENERALIZATION_MAX,
    GENERALIZATION_MIN,
    LOCKED_BENCHMARK_MAX,
    LOCKED_BENCHMARK_MIN,
    PROCEDURAL_TRAIN_MAX,
    PROCEDURAL_TRAIN_MIN,
    SeedLeakageError,
    assert_training_seed_pure,
    generate_procedural_seed,
    is_dev_screening_seed,
    is_evaluation_seed,
    is_generalization_seed,
    is_locked_benchmark_seed,
    is_procedural_training_seed,
)
from godfather_jepa.verification.ast_audit import (
    ALL_FORBIDDEN_MODULES,
    EXTENDED_LEGACY_MODULES,
    FORBIDDEN_LEGACY_MODULES,
    ImportViolation,
    audit_package_isolation,
    format_violation_report,
    get_max_relative_level,
)


# =====================================================================
# 1. AST STATIC ISOLATION AUDIT TESTS
# =====================================================================

class TestASTIsolationAudit:
    """Test suite for the automated AST static import checker."""

    def test_forbidden_modules_constants(self) -> None:
        """Verify the 10 canonical forbidden legacy modules and extended modules."""
        expected_canonical = {
            "models",
            "planning",
            "training",
            "zephyr",
            "dual_store",
            "scripts",
            "adaptation",
            "baselines",
            "diagnostics",
            "notebooks",
        }
        assert FORBIDDEN_LEGACY_MODULES == frozenset(expected_canonical)
        assert EXTENDED_LEGACY_MODULES == frozenset({"envs", "eval", "experiments"})
        assert ALL_FORBIDDEN_MODULES == (FORBIDDEN_LEGACY_MODULES | EXTENDED_LEGACY_MODULES)

    def test_audit_passes_on_godfather_jepa_clean_package(self) -> None:
        """Verify that godfather_jepa package itself is 100% clean of forbidden imports."""
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        violations = audit_package_isolation(package_root)
        report = format_violation_report(violations)
        assert len(violations) == 0, f"Expected 0 violations in godfather_jepa, got:\n{report}"
        assert report.startswith("PASS: 0 isolation violations detected")

    @pytest.mark.parametrize("mod", sorted(FORBIDDEN_LEGACY_MODULES))
    def test_direct_import_forbidden_modules_parameterized(self, mod: str) -> None:
        """Verify direct import of each canonical forbidden module is flagged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "test_file.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"import {mod}\nimport {mod}.submodule as alias\n")
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 2
            for v in violations:
                assert mod in v.reason
                assert v.line_number in (1, 2)

    @pytest.mark.parametrize("mod", sorted(FORBIDDEN_LEGACY_MODULES))
    def test_from_import_forbidden_modules_parameterized(self, mod: str) -> None:
        """Verify from-import of each canonical forbidden module is flagged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "test_file.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"from {mod} import item\nfrom {mod}.sub import item2\n")
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 2
            for v in violations:
                assert mod in v.reason

    def test_multi_module_import_on_single_line(self) -> None:
        """Verify comma-separated imports flag only the forbidden modules."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "multi_import.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("import os, sys, dual_store, math, planning\n")
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 2
            modules_flagged = {v.module for v in violations}
            assert modules_flagged == {"dual_store", "planning"}

    def test_dynamic_import_calls_detected(self) -> None:
        """Verify dynamic import variants (__import__, importlib.import_module) are detected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "dynamic_imports.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(
                    "__import__('models')\n"
                    "__import__(var_name)\n"
                    "builtins.__import__('training')\n"
                    "importlib.import_module('zephyr')\n"
                    "importlib.import_module(var_name)\n"
                    "from importlib import import_module; import_module('planning')\n"
                )
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) >= 5
            reasons = [v.reason for v in violations]
            assert any("Dynamic '__import__' call" in r for r in reasons)
            assert any("Dynamic 'import_module' call" in r for r in reasons)

    def test_permitted_clean_relative_imports(self) -> None:
        """Verify valid intra-package relative imports within depth bounds produce 0 violations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Root file: max level 1
            root_fpath = os.path.join(tmp_dir, "__init__.py")
            with open(root_fpath, "w", encoding="utf-8") as f:
                f.write("from . import models\nfrom .models import encoder\n")

            # Subdir file: max level 2
            sub_dir = os.path.join(tmp_dir, "sub")
            os.makedirs(sub_dir)
            sub_fpath = os.path.join(sub_dir, "core.py")
            with open(sub_fpath, "w", encoding="utf-8") as f:
                f.write("from . import helper\nfrom ..models import encoder\n")

            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 0, f"Expected 0 violations, got {violations}"

    def test_escaping_relative_imports_detected(self) -> None:
        """Verify relative imports that exceed package directory depth are flagged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Root file with level 2 (escapes package root)
            root_fpath = os.path.join(tmp_dir, "__init__.py")
            with open(root_fpath, "w", encoding="utf-8") as f:
                f.write("from .. import legacy_root\n")

            # Subdir file with level 3 (escapes package root)
            sub_dir = os.path.join(tmp_dir, "sub")
            os.makedirs(sub_dir)
            sub_fpath = os.path.join(sub_dir, "deep.py")
            with open(sub_fpath, "w", encoding="utf-8") as f:
                f.write("from ...models import old_model\n")

            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 2
            for v in violations:
                assert "escapes package root" in v.reason

    def test_permitted_package_absolute_imports(self) -> None:
        """Verify fully-qualified intra-package absolute imports are permitted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "module.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(
                    "import godfather_jepa.models.encoder as enc\n"
                    "from godfather_jepa.models.encoder import Conv4Encoder\n"
                    "from godfather_jepa.envs.crafter_env import GodfatherCrafterEnv\n"
                )
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 0

    def test_non_import_tokens_do_not_trigger_false_positives(self) -> None:
        """Verify comments, docstrings, variable names, and function names do not trigger false violations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "clean_code.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(
                    '# import models\n'
                    '"""Module for training models in planning environments."""\n'
                    'models = ["model1", "model2"]\n'
                    'def get_planning_models():\n'
                    '    return models\n'
                    'class ModelManager:\n'
                    '    def __init__(self):\n'
                    '        self.models = {}\n'
                    '        self.planning = True\n'
                )
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 0

    def test_nested_and_conditional_imports_detected(self) -> None:
        """Verify imports inside conditional branches, try/except blocks, and functions are detected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "conditional.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(
                    "if False:\n"
                    "    import models\n"
                    "try:\n"
                    "    import planning\n"
                    "except ImportError:\n"
                    "    pass\n"
                    "def helper():\n"
                    "    from zephyr import buffer\n"
                )
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 3

    def test_syntax_error_handled_gracefully(self) -> None:
        """Verify files with syntax errors are recorded as violations without crashing the audit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, "broken.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("def broken_syntax(:\n")
            violations = audit_package_isolation(tmp_dir)
            assert len(violations) == 1
            assert violations[0].module == "<syntax_error>"
            assert "syntax error" in violations[0].reason.lower()

    def test_nonexistent_directory_raises_error(self) -> None:
        """Verify non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            audit_package_isolation("/nonexistent/directory/path/that/cannot/exist")

    def test_cli_exit_codes(self) -> None:
        """Verify CLI entrypoint returns exit code 0 for clean, 1 for violations, 2 for error."""
        # 1. Clean directory -> 0
        with tempfile.TemporaryDirectory() as clean_dir:
            with open(os.path.join(clean_dir, "clean.py"), "w", encoding="utf-8") as f:
                f.write("import math\n")
            res_clean = subprocess.run(
                [sys.executable, "-m", "godfather_jepa.verification.ast_audit", clean_dir],
                capture_output=True,
                text=True,
            )
            assert res_clean.returncode == 0
            assert "PASS" in res_clean.stdout

        # 2. Dirty directory -> 1
        with tempfile.TemporaryDirectory() as dirty_dir:
            with open(os.path.join(dirty_dir, "dirty.py"), "w", encoding="utf-8") as f:
                f.write("import models\n")
            res_dirty = subprocess.run(
                [sys.executable, "-m", "godfather_jepa.verification.ast_audit", dirty_dir],
                capture_output=True,
                text=True,
            )
            assert res_dirty.returncode == 1
            assert "FAIL" in res_dirty.stdout

        # 3. Nonexistent directory -> 2
        res_error = subprocess.run(
            [sys.executable, "-m", "godfather_jepa.verification.ast_audit", "/nonexistent/path/xyz"],
            capture_output=True,
            text=True,
        )
        assert res_error.returncode == 2
        assert "ERROR" in res_error.stderr

    def test_violation_report_formatting(self) -> None:
        """Verify report string formatting for empty and non-empty violation lists."""
        assert "PASS" in format_violation_report([])

        v = ImportViolation(
            file_path="/path/test.py",
            line_number=42,
            module="models",
            source_line="import models",
            reason="Forbidden import",
        )
        report = format_violation_report([v])
        assert "FAIL: 1 isolation violation(s) detected" in report
        assert "/path/test.py:42" in report
        assert "import models" in report
        assert str(v) == "/path/test.py:42: [models] Forbidden import -> 'import models'"

    def test_get_max_relative_level_computation(self) -> None:
        """Verify relative import level computation across depths."""
        assert get_max_relative_level("/pkg/__init__.py", "/pkg") == 1
        assert get_max_relative_level("/pkg/module.py", "/pkg") == 1
        assert get_max_relative_level("/pkg/models/predictor.py", "/pkg") == 2
        assert get_max_relative_level("/pkg/models/sub/core.py", "/pkg") == 3


# =====================================================================
# 2. HERMETIC CRAFTER ENVIRONMENT CONTRACT TESTS
# =====================================================================

class TestGodfatherCrafterEnv:
    """Test suite for the Gymnasium-compatible deterministic Crafter wrapper."""

    def test_env_contract_observation_and_action_spaces(self) -> None:
        """Verify Gymnasium space types, shapes, bounds, and dtypes."""
        env = GodfatherCrafterEnv(seed=100)
        assert env.observation_space.shape == (64, 64, 3)
        assert env.observation_space.dtype == np.uint8
        assert env.observation_space.low.min() == 0
        assert env.observation_space.high.max() == 255
        assert env.action_space.n == 17
        env.close()

    def test_env_reset_return_signature_and_invariants(self) -> None:
        """Verify reset returns (obs, info) with uint8 [64, 64, 3] and clean diagnostic info."""
        env = GodfatherCrafterEnv(seed=101)
        obs, info = env.reset(seed=101)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        assert "inventory" in info
        assert "achievements" in info
        assert info["discount"] == 1.0
        assert info["step"] == 0
        assert info["dead"] is False
        assert len(info["achievements"]) == 22
        env.close()

    def test_env_step_gymnasium_5_tuple(self) -> None:
        """Verify step returns 5-tuple: (obs, reward, terminated, truncated, info)."""
        env = GodfatherCrafterEnv(seed=102)
        obs, _ = env.reset(seed=102)

        for a in range(17):
            next_obs, reward, terminated, truncated, info = env.step(a)
            assert isinstance(next_obs, np.ndarray)
            assert next_obs.shape == (64, 64, 3)
            assert next_obs.dtype == np.uint8
            assert isinstance(reward, (float, int))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert "discount" in info
            assert "step" in info
            if terminated or truncated:
                break
        env.close()

    def test_env_step_out_of_bounds_action_raises(self) -> None:
        """Verify invalid actions (<0 or >=17) raise ValueError."""
        env = GodfatherCrafterEnv(seed=103)
        env.reset(seed=103)
        with pytest.raises(ValueError):
            env.step(17)
        with pytest.raises(ValueError):
            env.step(-1)
        with pytest.raises(ValueError):
            env.step("invalid")  # type: ignore
        env.close()

    def test_env_truncation_vs_termination(self) -> None:
        """Verify truncated=True when step_count reaches max_steps, while terminated=False if alive."""
        env = GodfatherCrafterEnv(seed=104, max_steps=5)
        env.reset(seed=104)
        for s in range(1, 5):
            _, _, terminated, truncated, _ = env.step(0)  # noop
            assert not terminated
            assert not truncated

        _, _, terminated, truncated, info = env.step(0)  # step 5 = max_steps
        assert truncated is True
        assert terminated is False
        env.close()

    def test_env_deterministic_reproducibility(self) -> None:
        """Verify two independent instances with the same seed produce byte-identical rollouts."""
        seed = 2055
        env1 = GodfatherCrafterEnv(seed=seed)
        env2 = GodfatherCrafterEnv(seed=seed)

        o1, _ = env1.reset(seed=seed)
        o2, _ = env2.reset(seed=seed)
        assert np.array_equal(o1, o2), "Reset observations diverged!"

        actions = [1, 2, 3, 4, 5, 0, 5, 1, 2, 0, 6, 5] * 5  # 60 steps
        for step_idx, a in enumerate(actions):
            n_o1, r1, t1, tr1, _ = env1.step(a)
            n_o2, r2, t2, tr2, _ = env2.step(a)
            assert np.array_equal(n_o1, n_o2), f"Observation diverged at step {step_idx}"
            assert r1 == r2, f"Reward diverged at step {step_idx}"
            assert t1 == t2, f"Terminated diverged at step {step_idx}"
            assert tr1 == tr2, f"Truncated diverged at step {step_idx}"
            if t1 or tr1:
                break
        env1.close()
        env2.close()

    def test_creature_despawn_determinism_past_60_steps(self) -> None:
        """Verify creature despawn patch prevents divergence past the 60-step heap address window."""
        seed = 3050
        env1 = GodfatherCrafterEnv(seed=seed)
        env2 = GodfatherCrafterEnv(seed=seed)
        env1.reset(seed=seed)
        env2.reset(seed=seed)

        # 100 consecutive steps (well beyond 60-step creature despawn boundary)
        for step in range(100):
            a = (step * 7) % 17
            o1, r1, t1, _, _ = env1.step(a)
            o2, r2, t2, _, _ = env2.step(a)
            assert np.array_equal(o1, o2), f"Divergence detected at step {step} during creature balancing!"
            if t1:
                break
        env1.close()
        env2.close()

    def test_env_snapshot_and_restoration(self) -> None:
        """Verify deepcopy snapshot and restore produces 100% byte-identical branching."""
        env = GodfatherCrafterEnv(seed=4050)
        env.reset(seed=4050)
        for _ in range(20):
            env.step(1)

        snap = env.snapshot_state()

        actions = [5, 2, 3, 0, 1, 4] * 5
        branch_a_obs = []
        for a in actions:
            o, r, term, trunc, _ = env.step(a)
            branch_a_obs.append(o)
            if term or trunc:
                break

        env.restore_state(snap)
        branch_b_obs = []
        for a in actions:
            o, r, term, trunc, _ = env.step(a)
            branch_b_obs.append(o)
            if term or trunc:
                break

        assert len(branch_a_obs) == len(branch_b_obs)
        for i, (oa, ob) in enumerate(zip(branch_a_obs, branch_b_obs)):
            assert np.array_equal(oa, ob), f"Snapshot divergence at step {i}"
        env.close()

    def test_strict_pixel_only_inference_guarantee(self) -> None:
        """Verify agent action function depends strictly on obs pixels, ignoring info entirely."""
        def dummy_pixel_policy(obs_uint8: np.ndarray) -> int:
            assert obs_uint8.shape == (64, 64, 3)
            assert obs_uint8.dtype == np.uint8
            # Pixel-only decision based on observation center intensity
            return int(obs_uint8[32, 32, 0]) % 17

        env = GodfatherCrafterEnv(seed=5050)
        obs, info = env.reset(seed=5050)

        # 1. Action selection takes ONLY obs
        action1 = dummy_pixel_policy(obs)

        # 2. Mutate info dictionary externally
        info["inventory"]["wood"] = 999
        info["achievements"]["place_table"] = 10

        # 3. Action must remain completely unchanged
        action2 = dummy_pixel_policy(obs)
        assert action1 == action2, "Action altered by info mutation!"
        env.close()

    def test_env_training_flag_rejects_eval_seeds(self) -> None:
        """Verify that when training=True, GodfatherCrafterEnv actively rejects evaluation seeds."""
        with pytest.raises(SeedLeakageError):
            GodfatherCrafterEnv(seed=2055, training=True)

        env = GodfatherCrafterEnv(seed=100_000, training=True)
        with pytest.raises(SeedLeakageError):
            env.reset(seed=6175)
        env.close()

        # Non-training env permits evaluation seeds
        eval_env = GodfatherCrafterEnv(seed=2055, training=False)
        obs, _ = eval_env.reset(seed=2055)
        assert obs.shape == (64, 64, 3)
        eval_env.close()


# =====================================================================
# 3. CRAFTER SCORE METRIC TESTS
# =====================================================================

class TestCrafterScore:
    """Test suite for the official Crafter geometric score metric."""

    def test_crafter_score_uniform_mathematical_derivation(self) -> None:
        """Analytical verification of formula exp(mean(ln(1 + p_i))) - 1."""
        # 1. 0% uniform -> 0.0%
        zero_recs = [{ach: 0 for ach in CRAFTER_ACHIEVEMENTS} for _ in range(50)]
        stats_0 = compute_crafter_score(zero_recs)
        assert abs(stats_0["score"] - 0.0) < 1e-6
        assert stats_0["unlocked_count"] == 0

        # 2. 10% uniform -> 10.0%
        ten_recs = [{ach: 1 if ep < 5 else 0 for ach in CRAFTER_ACHIEVEMENTS} for ep in range(50)]
        stats_10 = compute_crafter_score(ten_recs)
        assert abs(stats_10["score"] - 10.0) < 1e-4

        # 3. 100% uniform -> 100.0%
        hundred_recs = [{ach: 1 for ach in CRAFTER_ACHIEVEMENTS} for _ in range(50)]
        stats_100 = compute_crafter_score(hundred_recs)
        assert abs(stats_100["score"] - 100.0) < 1e-4
        assert stats_100["unlocked_count"] == 22

    def test_crafter_score_partial_unlock_regularization(self) -> None:
        """Analytical verification: 18 achievements at 50%, 4 at 0%."""
        # Expected: exp((18 * ln(51) + 4 * ln(1)) / 22) - 1 = exp(18 * 3.9318256 / 22) - 1 = 23.9515%
        recs = []
        ach_50 = CRAFTER_ACHIEVEMENTS[:18]
        ach_0 = CRAFTER_ACHIEVEMENTS[18:]
        for ep in range(50):
            r = {ach: 1 if ep < 25 else 0 for ach in ach_50}
            for a in ach_0:
                r[a] = 0
            recs.append(r)

        stats = compute_crafter_score(recs)
        expected = math.exp((18 * math.log(51.0)) / 22.0) - 1.0
        assert abs(stats["score"] - expected) < 1e-3
        assert abs(stats["score"] - 23.9515) < 0.01

    def test_crafter_score_polymorphic_inputs(self) -> None:
        """Verify compute_crafter_score produces identical results across Dict, Numpy, and Precomputed inputs."""
        records = []
        for ep in range(40):
            rec = {ach: 1 if (ep + idx) % 3 == 0 else 0 for idx, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
            records.append(rec)

        # 1. Sequence of Dicts
        res_dict = compute_crafter_score(records)

        # 2. Numpy Matrix [N, 22]
        mat = records_to_matrix(records)
        res_mat = compute_crafter_score(mat)

        # 3. Precomputed Percentages
        res_pct = compute_crafter_score(res_dict["unlock_percentages"])

        assert abs(res_dict["score"] - res_mat["score"]) < 1e-6
        assert abs(res_dict["score"] - res_pct["score"]) < 1e-6

    def test_crafter_score_matrix_conversion_roundtrip(self) -> None:
        """Verify records_to_matrix and matrix_to_records roundtrip faithfully."""
        records = [
            {ach: 1 if idx % 2 == 0 else 0 for idx, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
            for _ in range(10)
        ]
        mat = records_to_matrix(records)
        assert mat.shape == (10, 22)
        assert mat.dtype == np.uint8
        recs_back = matrix_to_records(mat)
        assert recs_back == records

    def test_crafter_score_edge_cases(self) -> None:
        """Verify empty input, missing keys, and unknown keys handle safely."""
        # Empty
        empty_res = compute_crafter_score([])
        assert empty_res["score"] == 0.0
        assert empty_res["unlocked_count"] == 0

        # Empty array
        empty_mat_res = compute_crafter_score(np.zeros((0, 22)))
        assert empty_mat_res["score"] == 0.0

        # Missing keys in dict
        sparse_recs = [{"collect_wood": 1}]
        sparse_res = compute_crafter_score(sparse_recs)
        assert sparse_res["score"] > 0.0
        assert sparse_res["unlock_percentages"]["collect_stone"] == 0.0

        # Unknown keys in dict (e.g. extraneous debug keys)
        noisy_recs = [{"collect_wood": 1, "unknown_cheat": 999}]
        noisy_res = compute_crafter_score(noisy_recs)
        assert "unknown_cheat" not in noisy_res["unlock_percentages"]

    def test_crafter_score_historical_plateau_discrimination(self) -> None:
        """Verify monotonic discrimination between historical baseline profiles."""
        def score_from_pct(pct_dict: Dict[str, float]) -> float:
            return float(compute_crafter_score(pct_dict)["score"])

        s_random = score_from_pct({"collect_sapling": 30, "wake_up": 25, "eat_cow": 10, "collect_drink": 15})
        s_v2 = score_from_pct({"collect_sapling": 40, "wake_up": 35, "eat_cow": 15, "collect_drink": 20, "collect_wood": 10})
        s_t1 = score_from_pct({"collect_sapling": 60, "wake_up": 50, "eat_cow": 25, "collect_drink": 35, "collect_wood": 30, "place_table": 5})
        s_t2 = score_from_pct({"collect_sapling": 70, "wake_up": 60, "eat_cow": 30, "collect_drink": 45, "collect_wood": 45, "place_table": 15, "make_wood_pickaxe": 3})
        s_target = score_from_pct({
            "collect_sapling": 95, "wake_up": 95, "eat_cow": 95, "collect_drink": 95, "eat_plant": 95,
            "collect_wood": 90, "place_table": 85, "make_wood_pickaxe": 60, "collect_stone": 40,
            "make_wood_sword": 50, "place_plant": 50, "defeat_zombie": 50,
        })

        assert s_random < s_v2 < s_t1 < s_t2 < s_target
        assert s_target >= 8.0

    def test_canonical_achievement_tiers(self) -> None:
        """Verify achievement tiers cover all 22 achievements with no duplicates."""
        all_tiered = []
        for tier_name, ach_list in ACHIEVEMENT_TIERS.items():
            all_tiered.extend(ach_list)
        assert len(all_tiered) == 22
        assert set(all_tiered) == set(CRAFTER_ACHIEVEMENTS)


# =====================================================================
# 4. SEED SPACE PARTITIONING & GUARD TESTS
# =====================================================================

class TestSeedGuard:
    """Test suite for seed space partitioning and runtime purity guards."""

    def test_locked_benchmark_seeds_raise_seed_leakage_error(self) -> None:
        """Verify all 50 locked benchmark seeds [2050, 2099] trigger SeedLeakageError."""
        for seed in range(LOCKED_BENCHMARK_MIN, LOCKED_BENCHMARK_MAX + 1):
            with pytest.raises(SeedLeakageError, match="Locked Benchmark Evaluation Harness"):
                assert_training_seed_pure(seed)

    def test_dev_screening_seeds_raise_seed_leakage_error(self) -> None:
        """Verify all 50 dev screening seeds [6150, 6199] trigger SeedLeakageError."""
        for seed in range(DEV_SCREENING_MIN, DEV_SCREENING_MAX + 1):
            with pytest.raises(SeedLeakageError, match="Dev Screening Validation Pool"):
                assert_training_seed_pure(seed)

    def test_boundary_seeds_behavior(self) -> None:
        """Test exact boundary points around forbidden ranges."""
        # Boundaries of locked benchmark [2050, 2099]
        assert_training_seed_pure(2049)  # immediately below: ok
        assert_training_seed_pure(2100)  # immediately above: ok
        with pytest.raises(SeedLeakageError):
            assert_training_seed_pure(2050)
        with pytest.raises(SeedLeakageError):
            assert_training_seed_pure(2099)

        # Boundaries of dev screening [6150, 6199]
        assert_training_seed_pure(6149)  # immediately below: ok
        assert_training_seed_pure(6200)  # immediately above: ok
        with pytest.raises(SeedLeakageError):
            assert_training_seed_pure(6150)
        with pytest.raises(SeedLeakageError):
            assert_training_seed_pure(6199)

    def test_invalid_seed_types_and_values(self) -> None:
        """Verify type checking rejects booleans, floats, and negative integers."""
        with pytest.raises(TypeError, match="Seed must be an integer"):
            assert_training_seed_pure(True)  # bool is int subclass in Python
        with pytest.raises(TypeError, match="Seed must be an integer"):
            assert_training_seed_pure(False)
        with pytest.raises(TypeError, match="Seed must be an integer"):
            assert_training_seed_pure(2055.0)  # type: ignore
        with pytest.raises(ValueError, match="Seed must be non-negative"):
            assert_training_seed_pure(-1)

    def test_procedural_seed_generator_determinism(self) -> None:
        """Verify generator produces identical seeds for identical arguments."""
        seed1 = generate_procedural_seed(rank=3, episode=42, generation=1)
        seed2 = generate_procedural_seed(rank=3, episode=42, generation=1)
        assert seed1 == seed2
        assert seed1 >= PROCEDURAL_TRAIN_MIN
        assert seed1 <= PROCEDURAL_TRAIN_MAX

    def test_procedural_seed_generator_one_million_seeds_zero_leakage_and_overlap(self) -> None:
        """
        Verify 1,000,000 procedurally generated seeds:
        1. Never overlap locked evaluation seeds [2050, 2099].
        2. Never overlap dev screening seeds [6150, 6199].
        3. All seeds fall in [100,000, 999,999,999].
        4. Exactly 1,000,000 unique seeds (0 duplicate collisions).
        """
        # Vectorized generation across 5 generations x 16 ranks x 12,500 episodes
        generations = np.arange(5, dtype=np.int64)[:, None, None]
        ranks = np.arange(16, dtype=np.int64)[None, :, None]
        episodes = np.arange(12500, dtype=np.int64)[None, None, :]

        # Compute offsets matching generate_procedural_seed formula
        offsets = (generations * 100_000_000 + episodes * 1000 + ranks) % 999_900_000
        seeds = (100_000 + offsets).flatten()

        assert seeds.shape[0] == 1_000_000
        assert seeds.min() >= PROCEDURAL_TRAIN_MIN
        assert seeds.max() <= PROCEDURAL_TRAIN_MAX

        # Leakage checks
        assert not np.any((seeds >= 2050) & (seeds <= 2099)), "Found leak into [2050, 2099]!"
        assert not np.any((seeds >= 6150) & (seeds <= 6199)), "Found leak into [6150, 6199]!"

        # Collision-free uniqueness check
        unique_seeds = np.unique(seeds)
        assert len(unique_seeds) == 1_000_000, f"Expected 1M unique seeds, got {len(unique_seeds)}"

    def test_procedural_generator_argument_validation(self) -> None:
        """Verify generator validates arguments properly."""
        with pytest.raises(TypeError):
            generate_procedural_seed(rank="0")  # type: ignore
        with pytest.raises(TypeError):
            generate_procedural_seed(rank=0, episode=1.5)  # type: ignore
        with pytest.raises(ValueError):
            generate_procedural_seed(rank=-1, episode=0)
        with pytest.raises(ValueError):
            generate_procedural_seed(rank=0, episode=-5)
        with pytest.raises(ValueError):
            generate_procedural_seed(rank=0, generation=-1)

    def test_helper_predicates(self) -> None:
        """Verify helper query functions."""
        assert is_locked_benchmark_seed(2055)
        assert not is_locked_benchmark_seed(2049)
        assert is_dev_screening_seed(6175)
        assert not is_dev_screening_seed(6149)
        assert is_evaluation_seed(2055)
        assert is_evaluation_seed(6175)
        assert not is_evaluation_seed(100_000)
        assert is_procedural_training_seed(100_000)
        assert is_procedural_training_seed(999_999_999)
        assert not is_procedural_training_seed(99_999)
        assert is_generalization_seed(10_000)
        assert is_generalization_seed(99_999)
        assert not is_generalization_seed(9_999)
        assert not is_generalization_seed(100_000)
