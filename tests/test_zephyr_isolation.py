"""Isolation contract: zephyr/ must never import the previous agent line.

Enforces docs/ZEPHYR_AGENT_PLAN.md section 1: the only cross-line imports
allowed are `envs` (deterministic Crafter wrapper) and `eval.metrics` (official
scorer). Everything else must come from zephyr/ itself or the stdlib/numpy/torch.
"""
from __future__ import annotations

import ast
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZEPHYR_DIR = os.path.join(REPO_ROOT, "zephyr")

FORBIDDEN_PREFIXES = ("models", "planning", "training")
ALLOWED_LOCAL_PREFIXES = ("zephyr", "envs", "eval.metrics", "eval")

def _imports_of(path: str):
    with open(path, "r") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield module, node.lineno


def test_zephyr_package_exists_with_expected_modules():
    for name in ("config.py", "losses.py", "wm.py", "heads.py", "vec_env.py",
                 "buffer.py", "ac.py", "explore.py", "planner.py", "screen.py",
                 "train_wm.py", "train_loop.py"):
        assert os.path.exists(os.path.join(ZEPHYR_DIR, name)), f"missing {name}"


def test_no_imports_from_previous_agent_line():
    violations = []
    for root, _dirs, files in os.walk(ZEPHYR_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            for module, lineno in _imports_of(path):
                top = module.split(".")[0]
                if top in FORBIDDEN_PREFIXES:
                    violations.append(f"{path}:{lineno} imports {module}")
    assert not violations, "forbidden imports:\n" + "\n".join(violations)


def test_cross_line_imports_are_whitelisted_only():
    allowed = {"envs", "eval.metrics"}
    violations = []
    for root, _dirs, files in os.walk(ZEPHYR_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            for module, lineno in _imports_of(path):
                top = module.split(".")[0]
                if top in FORBIDDEN_PREFIXES or top in ("zephyr",):
                    continue
                if module.startswith("eval.metrics"):
                    continue
                if top in allowed:
                    continue
                if top in ("__future__",):
                    continue
                # stdlib / numpy / torch / everything else is fine; only flag
                # imports from THIS repo outside the whitelist.
                if module.split(".")[0] in (
                    "models", "planning", "training", "envs", "eval", "experiments",
                    "adaptation", "baselines", "diagnostics", "notebooks", "scripts",
                ):
                    if not module.startswith("eval.metrics"):
                        violations.append(f"{path}:{lineno} non-whitelisted repo import {module}")
    assert not violations, "non-whitelisted repo imports:\n" + "\n".join(violations)


def test_achievements_order_matches_official_scorer():
    from zephyr.config import ACHIEVEMENTS
    from eval.metrics import CRAFTER_ACHIEVEMENTS

    assert tuple(CRAFTER_ACHIEVEMENTS) == ACHIEVEMENTS
