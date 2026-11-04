"""P0 clean-room contract tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mosaic.src.agent import RandomVisualPolicy
from mosaic.src.api import (
    validate_observation,
    validate_reward,
    validate_step,
)
from mosaic.src.checkpoint import load_checkpoint, save_atomic
from mosaic.src.env_boundary import CrafterPixelEnv, OpaqueActionPermutation


def test_public_action_signature_is_exact() -> None:
    assert list(inspect.signature(RandomVisualPolicy.act).parameters) == [
        "self",
        "obs_uint8",
        "reward_prev",
        "step",
    ]


def test_observation_reward_and_step_validation() -> None:
    observation = np.zeros((64, 64, 3), dtype=np.uint8)
    assert validate_observation(observation).flags.c_contiguous
    assert validate_reward(np.float32(0.0)) == 0.0
    assert validate_step(np.int64(0)) == 0
    with pytest.raises(ValueError):
        validate_observation(np.zeros((64, 64, 3), dtype=np.float32))
    with pytest.raises(TypeError):
        validate_reward({"reward": 1.0})
    with pytest.raises(ValueError):
        validate_step(-1)


def test_action_permutation_is_bijective_and_semantics_free() -> None:
    first = OpaqueActionPermutation(17, seed=11)
    second = OpaqueActionPermutation(17, seed=12)
    values = [first.token_to_environment(index) for index in range(17)]
    assert sorted(values) == list(range(17))
    assert first.provenance()["num_actions"] == 17
    assert first.provenance()["mapping_sha256"] != second.provenance()["mapping_sha256"]
    assert "permutation" not in first.provenance()


def test_crafter_boundary_returns_no_metadata() -> None:
    environment = CrafterPixelEnv(seed=3, max_episode_steps=2, permutation_seed=5)
    try:
        observation = environment.reset(seed=3)
        assert observation.dtype == np.uint8
        next_observation, reward, terminated, truncated = environment.step(0)
        assert next_observation.shape == observation.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        environment.close()


def test_atomic_runtime_checkpoint_roundtrip(tmp_path) -> None:
    path = tmp_path / "runtime.pt"
    save_atomic(path, {"format": "mosaic_runtime_v1", "step": 7, "state": [1, 2]})
    assert load_checkpoint(path)["step"] == 7
    assert not path.with_suffix(".pt.tmp").exists()


def test_mosaic_source_contains_no_recipe_table_symbol() -> None:
    from pathlib import Path

    forbidden = (
        "CRAFT" + "_REQUIRED" + "_MATERIALS",
        "ACTION" + "_TO_INGREDIENT",
        "RECIPE" + "_TABLE",
    )
    for source in Path("mosaic/src").rglob("*.py"):
        text = source.read_text()
        assert not any(token in text for token in forbidden), source
