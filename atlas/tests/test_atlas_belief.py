"""Unit tests for AchievementBelief."""

import inspect
import numpy as np
import pytest
import torch

from atlas.belief import AchievementBelief


def test_belief_threshold_update():
    belief = AchievementBelief(unlock_threshold=0.25, reward_fallback=0.9)
    probs = np.zeros(22, dtype=np.float32)
    probs[3] = 0.40  # Above threshold

    newly = belief.update(probs, reward=0.0)
    assert newly == [3]
    assert belief.m_bin[3] == 1.0
    assert np.isclose(belief.m_soft[3], 0.40)
    assert belief.m_bin[0] == 0.0

    # Repeat: should not re-unlock dimension 3
    newly2 = belief.update(probs, reward=0.0)
    assert newly2 == []


def test_belief_reward_fallback():
    belief = AchievementBelief(unlock_threshold=0.25, reward_fallback=0.9)
    probs = np.zeros(22, dtype=np.float32)
    # All below 0.25 threshold, but dim 7 is highest
    probs[7] = 0.20
    probs[2] = 0.10

    newly = belief.update(probs, reward=1.0)
    assert newly == [7]
    assert belief.m_bin[7] == 1.0
    assert belief.m_soft[7] == 1.0
    assert belief.last_fallback == [7]


def test_belief_refuses_env_info():
    belief = AchievementBelief()
    sig = inspect.signature(belief.update)
    param_names = list(sig.parameters.keys())
    assert param_names == ["event_probs", "reward"], f"Unexpected parameters in update: {param_names}"

    # Passing env info dict raises TypeError or ValueError
    with pytest.raises((ValueError, TypeError)):
        belief.update({"achievements": [1] * 22}, 0.0)

    # Passing wrong shape raises ValueError
    with pytest.raises(ValueError):
        belief.update(np.zeros((10,)), 0.0)


def test_belief_reset():
    belief = AchievementBelief()
    probs = np.ones(22, dtype=np.float32)
    belief.update(probs, reward=1.0)
    assert np.all(belief.m_bin == 1.0)

    belief.reset()
    assert np.all(belief.m_bin == 0.0)
    assert np.all(belief.m_soft == 0.0)
    assert belief.last_fallback == []
