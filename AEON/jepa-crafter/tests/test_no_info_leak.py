"""Tests ensuring zero information leakage from info/internals into learner space."""
import pytest
import numpy as np
from src.env.crafter_adapter import CrafterAdapter
from src.env.learner_observation_filter import LearnerObservationFilter
from src.env.evaluator_tap import EvaluatorTap


def test_no_info_leak_in_step_return():
    tap = EvaluatorTap()
    adapter = CrafterAdapter(seed=123, evaluator_tap=tap)
    adapter.reset()
    
    # Inject a secret key into the underlying env step to simulate internal metadata
    orig_step = adapter._env.step
    def mock_step(action):
        obs, reward, done, info = orig_step(action)
        info["FORBIDDEN_LEAK_SECRET_TOKEN"] = "CLASSIFIED_INFO_12345"
        info["inventory"] = {"wood": 99}
        return obs, reward, done, info
    adapter._env.step = mock_step

    ret = adapter.step(0)
    assert len(ret) == 4, "Learner must strictly receive 4-tuple (obs, reward, terminated, truncated)"
    
    next_obs, reward, terminated, truncated = ret
    # Check that return elements contain no dictionary or nested leaked attributes
    assert isinstance(next_obs, np.ndarray)
    assert not hasattr(next_obs, "info")
    assert not hasattr(next_obs, "inventory")
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    # Run recursive inspection against leakage
    LearnerObservationFilter.inspect_against_leakage(ret)
    
    adapter.close()


def test_leak_filter_catches_forbidden_keys():
    leaky_dict = {
        "obs": np.zeros((64, 64, 3), dtype=np.uint8),
        "info": {"secret": 1},
    }
    with pytest.raises(AssertionError):
        LearnerObservationFilter.inspect_against_leakage(leaky_dict)
        
    leaky_dict2 = {
        "inventory": {"wood": 1},
    }
    with pytest.raises(AssertionError):
        LearnerObservationFilter.inspect_against_leakage(leaky_dict2)
