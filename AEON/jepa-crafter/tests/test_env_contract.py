"""Tests verifying environment interface contract."""
import pytest
import numpy as np
from src.env.crafter_adapter import CrafterAdapter
from src.env.opaque_actions import OpaqueActionMapping


def test_observation_and_action_contract():
    adapter = CrafterAdapter(seed=42)
    obs = adapter.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    
    # Check invalid action token raises ValueError
    with pytest.raises(ValueError):
        adapter.step(17)
    with pytest.raises(ValueError):
        adapter.step(-1)
        
    # Check valid step
    next_obs, reward, terminated, truncated = adapter.step(0)
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (64, 64, 3)
    assert next_obs.dtype == np.uint8
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    adapter.close()


def test_reward_free_mode():
    adapter = CrafterAdapter(env_id="CrafterNoReward-v1", reward_mode="none", seed=42)
    adapter.reset()
    for _ in range(5):
        _, reward, terminated, truncated = adapter.step(0)
        assert reward == 0.0
        if terminated or truncated:
            break
    adapter.close()
