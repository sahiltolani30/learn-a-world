"""Tests for Project ATLAS v2.2 F3 (Selection rotation — break the monopoly)."""

import numpy as np
import pytest

from atlas.config import GoalsConfig
from atlas.goals import GoalBank, GoalEntry


def test_f3_selection_rotation_breaks_monopoly():
    """Verification F3: with a stub bank where one goal always scores highest, over 200 selections >= 15% are rotate-mode and >= 5% frontier-mode; the monopoly goal's selection share drops below 60%."""
    cfg = GoalsConfig(selection_rotate_share=0.25, selection_frontier_share=0.10)
    bank = GoalBank(cfg=cfg)

    # Add 1 monopoly goal (scores 10.0 on locked dim)
    monopoly_entry = GoalEntry(
        key=np.random.randn(128).astype(np.float32),
        context=np.random.randn(256).astype(np.float32),
        action_profile=np.full((17, 22), 10.0, dtype=np.float32),
        event_dims=[0],
        kind="event",
        born_iter=0,
    )
    bank.add(monopoly_entry)

    # Add 9 other competitive event goals (scores 5.0 on locked dim)
    for i in range(1, 10):
        e = GoalEntry(
            key=np.random.randn(128).astype(np.float32),
            context=np.random.randn(256).astype(np.float32),
            action_profile=np.full((17, 22), 5.0, dtype=np.float32),
            event_dims=[1],
            kind="event",
            born_iter=0,
        )
        bank.add(e)

    # Add 5 frontier goals
    for i in range(5):
        e = GoalEntry(
            key=np.random.randn(128).astype(np.float32),
            context=np.random.randn(256).astype(np.float32),
            action_profile=np.zeros((17, 22), dtype=np.float32),
            event_dims=[],
            kind="frontier",
            born_iter=0,
        )
        bank.add(e)

    np.random.seed(42)
    counts = {"argmax": 0, "rotate": 0, "frontier": 0}
    monopoly_selections = 0

    dummy_z = np.zeros(128, dtype=np.float32)
    dummy_h = np.zeros(256, dtype=np.float32)
    m_soft = np.zeros(22, dtype=np.float32)
    psi_fn = lambda s, g: 0.5

    total_selections = 200
    for it in range(total_selections):
        g, score = bank.select(dummy_z, dummy_h, m_soft, psi_fn, current_iter=it)
        mode = bank.last_selection_mode
        counts[mode] += 1
        if g is monopoly_entry:
            monopoly_selections += 1
        bank.mark_used(g.key, success=True, current_iter=it)

    rotate_share = counts["rotate"] / total_selections
    frontier_share = counts["frontier"] / total_selections
    monopoly_share = monopoly_selections / total_selections

    # Assertions per PLAN_V2_2 F3 spec
    assert rotate_share >= 0.15, f"Expected rotate_share >= 0.15, got {rotate_share:.3f}"
    assert frontier_share >= 0.05, f"Expected frontier_share >= 0.05, got {frontier_share:.3f}"
    assert monopoly_share < 0.60, f"Expected monopoly_share < 0.60, got {monopoly_share:.3f}"
