"""Tests verifying that evaluator tap records metrics without leaking to adapter output."""
import numpy as np
from src.env.crafter_adapter import CrafterAdapter
from src.env.evaluator_tap import EvaluatorTap
from src.env.learner_observation_filter import LearnerObservationFilter


def test_evaluator_isolation():
    tap = EvaluatorTap()
    adapter = CrafterAdapter(seed=100, evaluator_tap=tap)
    adapter.reset()
    
    # Step environment
    for i in range(10):
        ret = adapter.step(i % 17)
        # Ensure returned object to learner has no evaluator or info reference
        assert len(ret) == 4
        LearnerObservationFilter.inspect_against_leakage(ret)
        
    summary = tap.get_summary()
    assert "mean_reward" in summary
    assert "achievement_histories" in summary
    
    adapter.close()
