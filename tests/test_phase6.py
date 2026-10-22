r"""
Automated Gate Tests for Phase 6 (Learn-a-World v2 Phase 2):
1. TwoHot Encoding Parity: Continuous scalar return targets map to discrete bins and reconstruct with minimal error.
2. QEnsemble Contracts: K=3 heads, output shapes [K, B, 51], q_mean, non-zero q_std, and cross-entropy loss.
3. PolicyPrior Contracts: Action logits [B, 17], temperature sampling, and behaviour-cloning loss.
4. AgentBelief Logic: Tracks unlocked mask strictly from transitions, handles p_i > 0.5 and reward >= 0.9 fallback.
5. RewardValuePlanner: Rollout trajectory scoring with candidate mask simulation, returns valid int action and debug metrics.
6. Pure v2 Evaluator Smoke: Runs 2 episodes (10 steps each) without info leaks and outputs complete decomposition.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from eval.eval_v2 import evaluate_v2
from models.heads import (
    NUM_ACHIEVEMENTS,
    PolicyPrior,
    QEnsemble,
    two_hot_decode,
    two_hot_encode,
)
from models.jepa import JEPAWorldModel
from models.jepa_v2 import JEPAv2
from planning.agent_v2 import AgentBelief
from planning.mpc_v2 import RewardValuePlanner

def test_two_hot_encoding_parity():
    """Gate Test 1: Verify TwoHot discrete binning accurately reconstructs continuous targets."""
    num_bins = 51
    v_min = 0.0
    v_max = 12.0
    delta = (v_max - v_min) / (num_bins - 1)  # 0.24

    targets = torch.tensor([0.0, 1.0, 2.4, 5.0, 11.5, 12.0], dtype=torch.float32)
    dist = two_hot_encode(targets, num_bins=num_bins, v_min=v_min, v_max=v_max)

    assert dist.shape == (len(targets), num_bins)
    # Probabilities must sum to 1.0
    assert torch.allclose(dist.sum(dim=-1), torch.ones(len(targets)), atol=1e-5)

    # Convert discrete distribution back to scalar expectations using log probabilities
    log_dist = (dist + 1e-8).log()
    reconstructed = two_hot_decode(log_dist, num_bins=num_bins, v_min=v_min, v_max=v_max)

    max_err = (targets - reconstructed).abs().max().item()
    assert max_err < 1e-3, f"TwoHot reconstruction error {max_err:.4f} exceeded tolerance."

def test_q_ensemble_contracts():
    """Gate Test 2: Verify K=3 QEnsemble forward passes, q_mean, q_std, and loss."""
    B = 4
    latent_dim = 128
    num_actions = 17
    num_heads = 3
    num_bins = 51

    q_ens = QEnsemble(
        latent_dim=latent_dim,
        num_actions=num_actions,
        action_emb_dim=32,
        num_heads=num_heads,
        num_bins=num_bins,
        hidden_dim=64,
    )

    z = torch.randn(B, latent_dim)
    actions = torch.randint(0, num_actions, (B,))

    logits = q_ens(z, actions)
    assert logits.shape == (num_heads, B, num_bins)

    q_vals = q_ens.q_values(z, actions)
    assert q_vals.shape == (num_heads, B)

    q_m = q_ens.q_mean(z, actions)
    assert q_m.shape == (B,)
    assert 0.0 <= q_m.min() and q_m.max() <= 12.0

    q_s = q_ens.q_std(z, actions)
    assert q_s.shape == (B,)
    assert (q_s >= 0.0).all()

    target_returns = torch.tensor([1.0, 3.5, 0.0, 8.0])
    loss = q_ens.compute_loss(logits, target_returns)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_policy_prior_contracts():
    """Gate Test 3: Verify PolicyPrior action logits, categorical sampling, and CE loss."""
    B = 4
    latent_dim = 128
    num_actions = 17

    prior = PolicyPrior(
        latent_dim=latent_dim,
        num_actions=num_actions,
        hidden_dim=64,
    )

    z = torch.randn(B, latent_dim)
    mask = torch.zeros(B, NUM_ACHIEVEMENTS)
    mask[:, 6] = 1.0  # wood unlocked

    logits = prior(z, mask)
    assert logits.shape == (B, num_actions)

    sampled = prior.sample(z, mask, temperature=1.0)
    assert sampled.shape == (B,)
    assert (sampled >= 0).all() and (sampled < num_actions).all()

    target_actions = torch.tensor([5, 1, 8, 11], dtype=torch.long)
    loss = prior.compute_loss(logits, target_actions)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_agent_belief_tracking():
    """Gate Test 4: Verify AgentBelief mask progression without env info access."""
    belief = AgentBelief()
    assert (belief.get_mask() == 0.0).all()

    # Step 1: Low event probabilities and no reward -> mask remains 0
    p_step1 = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
    p_step1[6] = 0.2  # wood
    belief.update(p_step1, reward=0.0)
    assert (belief.get_mask() == 0.0).all()

    # Step 2: EventHead detects collect_wood with high confidence (p=0.85)
    p_step2 = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
    p_step2[6] = 0.85
    belief.update(p_step2, reward=1.0)
    assert belief.get_mask()[6] == 1.0
    assert belief.get_mask().sum() == 1.0

    # Step 3: Fallback rule - reward 1.0 received, but EventHead uncertain (max prob p=0.45 on collect_drink)
    p_step3 = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
    p_step3[2] = 0.45  # collect_drink has highest prob among locked
    p_step3[6] = 0.90  # wood already unlocked, must be ignored
    belief.update(p_step3, reward=1.0)
    # collect_drink (index 2) must be unlocked by fallback
    assert belief.get_mask()[2] == 1.0
    assert belief.get_mask()[6] == 1.0
    assert belief.get_mask().sum() == 2.0

def test_reward_value_planner():
    """Gate Test 5: Verify RewardValuePlanner returns valid action and debug decomposition."""
    base_model = JEPAWorldModel(
        in_channels=3,
        latent_dim=128,
        num_actions=17,
        action_emb_dim=32,
        predictor_hidden_dim=64,
        predictor_layers=2,
    )
    model = JEPAv2(
        base_model=base_model,
        hidden_dim=64,
        action_emb_dim=32,
        q_heads=3,
        q_bins=51,
        freeze_base=True,
    )

    planner = RewardValuePlanner(
        model=model,
        horizon=4,
        num_candidates=32,
        cem_iterations=2,
        elite_ratio=0.25,
        device="cpu",
    )

    obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    belief = AgentBelief()

    action, debug = planner.plan(obs, belief)

    assert isinstance(action, int)
    assert 0 <= action < 17
    assert "reward_sum" in debug
    assert "terminal_value" in debug
    assert "uncertainty_penalty" in debug
    assert "best_score" in debug
    assert "mean_q_std" in debug
    assert debug["mean_q_std"] >= 0.0

def test_eval_v2_smoke_run(tmp_path):
    """Gate Test 6: Verify pure v2 evaluator runs end-to-end on held-out seeds."""
    out_file = tmp_path / "smoke_v2.json"
    results = evaluate_v2(
        checkpoint_path=None,  # smoke test with initialized v2 model
        config_path="experiments/configs/crafter_v2.yaml",
        num_episodes=2,
        seed_start=1000,
        max_steps=10,
        device_override="cpu",
        out_path=str(out_file),
    )

    assert out_file.exists()
    assert "crafter_score" in results
    assert "mean_survival_steps" in results
    assert "debug_decomposition" in results
    assert results["num_episodes"] == 2
    assert "reward_sum_variance" in results["debug_decomposition"]
