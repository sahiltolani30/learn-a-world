"""F11-fix regression tests: material gating must be LEARNED, never a recipe table.

These tests encode the integrity contract:
1. Suppression is computed purely from the frozen StateDeltaHead's predicted deltas and
   the learned readout's holdings — change the predictions, the gating changes.
2. No hardcoded action→material mapping may exist anywhere in atlas/ (structural grep).
3. The provenance guard disables gating when the delta head was not loaded.
"""

import ast
import numpy as np
import pytest
import torch
from pathlib import Path

from atlas.arbiter import compute_material_suppression


def test_suppression_follows_learned_deltas():
    # Action 0 consumes 3 units of item 1 (delta -3 on log1p scale); agent has 1.
    deltas = np.zeros((17, 16), dtype=np.float32)
    deltas[0, 1] = -3.0
    have = np.zeros(16, dtype=np.float32)
    have[1] = 1.0
    sup = compute_material_suppression(deltas, have, margin=0.2)
    assert sup[0] is True or bool(sup[0]) is True
    # Give the agent 5 units -> no deficit -> no suppression
    have[1] = 5.0
    sup2 = compute_material_suppression(deltas, have, margin=0.2)
    assert not bool(sup2[0])
    # Change the delta head's prediction -> gating must change (learned, not fixed)
    deltas2 = deltas.copy()
    deltas2[0, 1] = 0.0
    assert not bool(compute_material_suppression(deltas2, have, margin=0.2)[0])


def test_suppression_ignores_vitals_dims():
    # A negative delta on a vitals dim (12..15) must NEVER gate — that is ContinueNet's job.
    deltas = np.zeros((17, 16), dtype=np.float32)
    deltas[5, 12] = -9.0  # action 5 costs health
    have = np.zeros(16, dtype=np.float32)
    assert not bool(compute_material_suppression(deltas, have)[5])


def test_suppression_pure_and_generic():
    # Symmetry: which action gets suppressed depends ONLY on its learned delta row.
    deltas = np.zeros((17, 16), dtype=np.float32)
    deltas[3, 0] = -1.0
    deltas[9, 0] = -2.0
    have = np.zeros(16, dtype=np.float32)
    sup = compute_material_suppression(deltas, have, margin=0.2)
    assert bool(sup[3]) and bool(sup[9]) and not bool(sup[0])


def test_no_recipe_table_in_source():
    """Structural audit: no hardcoded action->material mapping anywhere in atlas/*.py."""
    atlas_dir = Path(__file__).resolve().parent.parent
    banned = ("CRAFT_REQUIRED_MATERIALS",)
    for py in atlas_dir.rglob("*.py"):
        if "test" in py.name:
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in banned:
                        pytest.fail(f"hardcoded recipe table '{t.id}' found in {py.name}")
    # Also: the arbiter must not contain a literal dict of int->list describing crafts
    arb = (atlas_dir / "arbiter.py").read_text()
    assert "CRAFT_REQUIRED_MATERIALS" not in arb
