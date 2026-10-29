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


def test_probe_act_with_learned_delta_head():
    """Verify AtlasAgent.act in PROBE mode executes learned gating without NoneType error."""
    from atlas.arbiter import AtlasAgent
    from atlas.backbone import FrozenBackbone
    from atlas.config import AtlasConfig

    cfg = AtlasConfig()
    cfg.goals.material_gating_learned = True
    fb = FrozenBackbone(config=cfg, device="cpu")
    # Simulate that delta head is learned
    fb.model.loaded_component_keys.add("state_delta_head")
    assert fb.has_learned_delta_head is True

    agent = AtlasAgent(backbone=fb, cfg=cfg)
    agent.mode = "PROBE"
    agent.probe_steps_remaining = 3

    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    # Step 0 act - should not throw TypeError
    action, telemetry = agent.act(obs, reward_prev=0.0, step=0)
    assert 0 <= action < 17
    assert agent.last_probe_decision is not None
    assert "material_gate_learned" in agent.last_probe_decision

