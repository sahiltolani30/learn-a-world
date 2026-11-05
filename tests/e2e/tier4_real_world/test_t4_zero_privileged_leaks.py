"""Tier 4: Scenario 4B - Scientific Integrity Verification (Zero Privileged Leaks).

Covers:
- F1 (Pixels Only), F25 (Zero Privileged Leaks).
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import numpy as np
import pytest
import torch

from envs.crafter_wrapper import CrafterEnv
from zephyr.planner import ZephyrAgent, ZephyrCEMAgent

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
SCREEN_PY = REPO_ROOT / "zephyr" / "screen.py"


def test_t4_leak_01_strict_action_path_signature_inspection():
    """T4-LEAK-01: Action path signatures take ONLY raw observation pixels (no info/inventory)."""
    # Mode A: ZephyrAgent.act
    sig_a = inspect.signature(ZephyrAgent.act)
    params_a = list(sig_a.parameters.keys())
    assert params_a == ["self", "obs_uint8"], f"ZephyrAgent.act has unexpected parameters: {params_a}"

    # Mode B: ZephyrCEMAgent.act
    sig_b = inspect.signature(ZephyrCEMAgent.act)
    params_b = list(sig_b.parameters.keys())
    assert params_b == ["self", "obs_uint8"], f"ZephyrCEMAgent.act has unexpected parameters: {params_b}"

    # Assert forbidden argument names do not exist
    forbidden_args = {"info", "inventory", "reward", "done", "term", "trunc", "achievements"}
    for arg in forbidden_args:
        assert arg not in params_a
        assert arg not in params_b


def test_t4_leak_02_hidden_engine_state_invariance_audit(mock_wm, mock_heads, crafter_env_factory):
    """T4-LEAK-02: Hidden engine state mutations produce identical actions given unchanged pixels."""
    agent = ZephyrAgent(mock_wm, mock_heads, device="cpu")
    env = crafter_env_factory(seed=7000, max_steps=50)
    obs_real, _ = env.reset(seed=7000)

    # Action on unmodified state
    agent.reset()
    action1 = agent.act(obs_real)

    # Mutate hidden engine internal inventory and vitals without changing the observation frame
    env._env._player.inventory["wood"] = 99
    env._env._player.inventory["stone"] = 99
    env._env._player.inventory["health"] = 1
    env._env._player.inventory["energy"] = 1

    # Pass the identical observation frame again
    agent.reset()
    action2 = agent.act(obs_real)

    assert action1 == action2, (
        f"Agent action shifted ({action1} -> {action2}) under identical pixel observation, "
        "indicating privileged information leakage"
    )


def test_t4_leak_03_post_episode_environment_info_quarantine():
    """T4-LEAK-03: AST static audit of screen.py proves info is never fed to agent.act()."""
    with open(SCREEN_PY, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(SCREEN_PY))

    # Look for Call nodes to agent.act(...)
    act_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "act":
                act_calls.append(node)

    assert len(act_calls) > 0, "No agent.act() call found in screen.py"

    for call in act_calls:
        # Must have exactly 1 argument: obs
        assert len(call.args) == 1, f"agent.act() called with {len(call.args)} args, expected 1"
        arg = call.args[0]
        assert isinstance(arg, ast.Name) and arg.id == "obs", f"agent.act argument is {arg}, expected obs"


def test_t4_leak_04_auxiliary_readout_head_insulation_audit(mock_wm, mock_heads, sample_obs_batch):
    """T4-LEAK-04: Auxiliary readout and effect heads are never queried during action selection."""
    call_counts = {"readout": 0, "effect": 0}

    real_readout = mock_heads.readout.forward
    real_effect = mock_heads.effect.forward

    def spy_readout(*args, **kwargs):
        call_counts["readout"] += 1
        return real_readout(*args, **kwargs)

    def spy_effect(*args, **kwargs):
        call_counts["effect"] += 1
        return real_effect(*args, **kwargs)

    mock_heads.readout.forward = spy_readout
    mock_heads.effect.forward = spy_effect

    try:
        agent = ZephyrAgent(mock_wm, mock_heads, device="cpu")
        agent.reset()
        _ = agent.act(sample_obs_batch[0])

        assert call_counts["readout"] == 0, "ReadoutHead was called during action inference"
        assert call_counts["effect"] == 0, "EffectHead was called during action inference"
    finally:
        mock_heads.readout.forward = real_readout
        mock_heads.effect.forward = real_effect
