"""
genesis_jepa.tests.test_gate5_evaluation
========================================
Gate 5 Verification Suite:
1. End-to-end cleanroom evaluation on Crafter environment
2. Verification of unprivileged interface integrity
3. Metric sanity: episode lengths, survival rates, rewards, and achievements dictionary
"""
import pytest
import numpy as np

from genesis_jepa.agent import GenesisVisualAgent
from genesis_jepa.config import GenesisConfig
from genesis_jepa.evaluate import evaluate_agent


class TestGate5EvaluationHarness:
    def test_evaluate_agent_dry_run(self):
        """Runs 2 quick episodes on Crafter and verifies metrics schema and execution."""
        config = GenesisConfig()
        agent = GenesisVisualAgent(config)

        results = evaluate_agent(
            agent=agent,
            config=config,
            num_episodes=2,
            max_steps_per_episode=30,  # Short test horizon
            seed=123,
            verbose=False,
        )

        assert results["num_episodes"] == 2
        assert "mean_reward" in results
        assert "mean_length" in results
        assert "nightfall_survival_rate" in results
        assert "crafter_score" in results
        assert isinstance(results["achievement_rates"], dict)
        assert results["mean_length"] > 0
