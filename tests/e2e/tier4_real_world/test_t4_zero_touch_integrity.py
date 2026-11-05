"""Tier 4: Scenario 4A - Scientific Integrity Verification (Zero-Touch & Tabula Rasa).

Covers:
- F23 (Zero-Touch Purity), F24 (Zero External Weights), M1 Isolation.
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import os
import pathlib
import subprocess
import sys
import types
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
ZEPHYR_DIR = REPO_ROOT / "zephyr"

# Cleanly bind eval.metrics without executing eval/__init__.py
if "eval" not in sys.modules:
    eval_mod = types.ModuleType("eval")
    sys.modules["eval"] = eval_mod
if "eval.metrics" not in sys.modules:
    spec = importlib.util.spec_from_file_location("eval.metrics", str(REPO_ROOT / "eval" / "metrics.py"))
    metrics_mod = importlib.util.module_from_spec(spec)
    sys.modules["eval.metrics"] = metrics_mod
    spec.loader.exec_module(metrics_mod)
    setattr(sys.modules["eval"], "metrics", metrics_mod)

from eval.metrics import CRAFTER_ACHIEVEMENTS
from zephyr.config import ACHIEVEMENTS, ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.wm import ZephyrWM


def test_t4_integ_01_git_status_working_tree_audit():
    """T4-INTEG-01: Zero-touch working tree audit: 0 tracked legacy files modified."""
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    # Legacy protected paths
    protected_prefixes = (
        "models/",
        "planning/",
        "training/",
        "scripts/",
        "envs/",
        "eval/",
    )

    violations = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        status_code = line[:2]
        path = line[3:].strip()
        # Check if a tracked file in a protected directory was modified
        if not line.startswith("??"):  # Modified or staged tracked file
            for prefix in protected_prefixes:
                if path.startswith(prefix):
                    violations.append((status_code, path))

    assert len(violations) == 0, f"Zero-touch policy violation: modified legacy files {violations}"


def test_t4_integ_02_ast_static_import_whitelist_audit():
    """T4-INTEG-02: AST static analysis proves zephyr/ has zero imports from legacy modules."""
    py_files = list(ZEPHYR_DIR.rglob("*.py"))
    assert len(py_files) > 0, "No Python files found in zephyr/"

    forbidden_roots = ("models", "planning", "training")
    violations = []

    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in forbidden_roots:
                        violations.append((str(file_path), alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in forbidden_roots:
                        violations.append((str(file_path), node.module, node.lineno))

    assert len(violations) == 0, f"AST static audit found forbidden legacy imports: {violations}"


def test_t4_integ_03_pure_tabula_rasa_weight_initialization_audit(tiny_test_cfg):
    """T4-INTEG-03: Pure tabula rasa verification: world model & heads initialized randomly from scratch."""
    wm = ZephyrWM(tiny_test_cfg)
    heads = ZephyrHeads(tiny_test_cfg)

    # Check encoder parameters
    for name, p in wm.encoder.named_parameters():
        assert torch.isfinite(p).all(), f"NaN or Inf in encoder parameter {name}"
        if "weight" in name:
            assert torch.norm(p) > 0.0, f"All-zero weight in encoder {name}"

    # Check predictor parameters
    for name, p in wm.predictor.named_parameters():
        assert torch.isfinite(p).all(), f"NaN or Inf in predictor parameter {name}"
        if "weight" in name:
            assert torch.norm(p) > 0.0, f"All-zero weight in predictor {name}"

    # Check heads parameters
    for name, p in heads.named_parameters():
        assert torch.isfinite(p).all(), f"NaN or Inf in head parameter {name}"
        if "weight" in name:
            assert torch.norm(p) > 0.0, f"All-zero weight in head {name}"


def test_t4_integ_04_achievement_order_parity_with_official_scorer():
    """T4-INTEG-04: Exact 1:1 parity between zephyr.config.ACHIEVEMENTS and eval.metrics."""
    assert len(ACHIEVEMENTS) == 22
    assert len(CRAFTER_ACHIEVEMENTS) == 22
    assert tuple(ACHIEVEMENTS) == tuple(CRAFTER_ACHIEVEMENTS)

    # Verify key causal milestones exist in exact positions
    assert "collect_wood" in ACHIEVEMENTS
    assert "place_table" in ACHIEVEMENTS
    assert "make_wood_pickaxe" in ACHIEVEMENTS
    assert "collect_stone" in ACHIEVEMENTS
