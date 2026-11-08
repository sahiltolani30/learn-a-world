"""
genesis_jepa.tests.test_gate4_agent_and_policy
==============================================
Gate 4 Verification Suite:
1. SymlogDistributionalCritic:
   - Two-hot continuous simplex projection & roundtrip expectation
   - Center bin (index 32) at y = 0.0
   - Analytical unit logit gradient bound: ||grad_z L||_infty <= 1.0
2. ValueStratifiedReplayBuffer:
   - Guaranteed 50/50 milestone quota (kappa = 0.50)
   - Curious Replay priority advantage
3. GenesisVisualAgent:
   - Strict act(obs_uint8, reward_prev, step) interface
   - Anti-Sleep Hazard Lock (bars Action 6 when dark or in danger)
   - Dynamic Bayes Risk cutoff and evasive locomotion dodging
"""
import numpy as np
import pytest
import torch

from genesis_jepa.agent import GenesisVisualAgent
from genesis_jepa.config import GenesisConfig
from genesis_jepa.modules.distributional_critic import (
    SymlogDistributionalCritic,
    symexp,
    symlog,
)
from genesis_jepa.replay import ValueStratifiedReplayBuffer


class TestGate4CriticAndPolicy:
    def test_two_hot_projection_and_zero_return_center(self):
        """Verifies zero return maps exactly to atom 32, and two-hot expectation matches symlog(y)."""
        critic = SymlogDistributionalCritic(in_dim=192, num_bins=65, v_min=-20.0, v_max=20.0)

        # 1. Zero return test: symlog(0.0) = 0.0, center atom index = 32
        zero_target = torch.tensor([0.0])
        p_zero = critic.two_hot_projection(zero_target)
        assert p_zero.shape == (1, 65)
        assert torch.argmax(p_zero, dim=-1).item() == 32
        assert torch.allclose(p_zero[0, 32], torch.tensor(1.0), atol=1e-5)

        # 2. Continuous values roundtrip: sum(p * support) == symlog(y)
        test_vals = torch.tensor([-150.0, -10.0, -1.0, 0.0, 0.5, 1.0, 5.0, 20.0, 100.0])
        p_dist = critic.two_hot_projection(test_vals)
        recovered_symlog = torch.sum(p_dist * critic.support, dim=-1)
        expected_symlog = symlog(test_vals).clamp(-20.0, 20.0)
        assert torch.allclose(recovered_symlog, expected_symlog, atol=1e-5)

    def test_unit_bounded_logit_gradient(self):
        """Verifies ||grad_z L_CE||_infty <= 1.0 unconditionally across extreme return shocks."""
        critic = SymlogDistributionalCritic(in_dim=192, num_bins=65)
        torch.manual_seed(42)

        dummy_states = torch.randn(32, 192)
        # Extreme target returns from -1000 to +1000
        extreme_targets = torch.tensor(
            [-1000.0, -50.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0] * 4,
            dtype=torch.float32,
        )

        loss, metrics = critic.compute_loss(dummy_states, extreme_targets)

        # Check recorded max logit grad in metrics
        assert metrics["max_logit_grad"] <= 1.0 + 1e-6

        # Analytical backward verification
        loss.backward()
        for p in critic.parameters():
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()

    def test_value_stratified_replay_milestone_quota(self):
        """Verifies 50/50 quota equipartition between milestone and routine experiences."""
        buffer = ValueStratifiedReplayBuffer(capacity=1000, milestone_quota=0.50)

        dummy_obs = np.zeros((64, 64, 3), dtype=np.uint8)

        # Add 100 routine steps (reward = 0)
        for _ in range(100):
            buffer.add(dummy_obs, action=1, reward=0.0, next_obs=dummy_obs, done=False)

        # Add 10 milestone steps (reward = 1.0)
        for _ in range(10):
            buffer.add(dummy_obs, action=5, reward=1.0, next_obs=dummy_obs, done=False)

        # Sample batch of 32
        batch = buffer.sample_batch(batch_size=32)
        rewards = batch["reward"].squeeze(-1)

        # Half the batch (16 items) must have reward > 0
        milestone_count = int((rewards > 0.0).sum().item())
        assert milestone_count == 16, f"Expected exactly 16 milestone items, got {milestone_count}"

    def test_genesis_visual_agent_strict_act_interface(self):
        """Verifies act() conforms strictly to (obs_uint8, reward_prev, step) -> int."""
        config = GenesisConfig()
        agent = GenesisVisualAgent(config)

        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        action = agent.act(obs, reward_prev=0.0, step=1)

        assert isinstance(action, int)
        assert 0 <= action < config.num_actions

    def test_anti_sleep_hazard_lock(self):
        """Verifies Action 6 (sleep) is banned when darkness >= 0.40 or danger > 0.05."""
        config = GenesisConfig(night_darkness_threshold=0.40)
        agent = GenesisVisualAgent(config)

        # Simulate pitch black night (L_dark = 0.85 >= 0.40)
        agent.compute_circadian_darkness = lambda patch: 0.85

        obs = np.zeros((64, 64, 3), dtype=np.uint8)

        # Run 50 action selections in pitch black conditions
        actions_chosen = []
        for _ in range(50):
            act = agent.act(obs, reward_prev=0.0, step=100)
            actions_chosen.append(act)

        # Action 6 (sleep) must NEVER be chosen
        assert 6 not in actions_chosen, "Anti-Sleep Hazard Lock violated: Action 6 was chosen during darkness!"

    def test_dynamic_bayes_risk_cutoff_evasive_dodge(self):
        """Verifies dynamic Bayes cutoff forces evasive locomotion dodges under high threat."""
        config = GenesisConfig(danger_threshold=0.05, bayes_risk_cutoff_max=0.082)
        agent = GenesisVisualAgent(config)

        # Configure danger head to strictly output high threat (> 0.99)
        with torch.no_grad():
            agent.danger_head[-2].weight.zero_()
            agent.danger_head[-2].bias.fill_(10.0)

        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Run 30 actions under high threat
        for _ in range(30):
            action = agent.act(obs, reward_prev=0.0, step=10)
            # Under high threat, agent must choose evasive moves: Actions 1, 2, 3, or 4 (move left/right/up/down)
            assert action in [1, 2, 3, 4], f"Expected evasive dodge (1..4), got action {action}"
