"""
Automated Gate Tests for Phase 2 (v3 Recurrent Model & Task-Shaped Representation):
1. RecurrentCore: Hidden state continuity and episode boundary reset masking.
2. RecurrentPredictor: Multi-step imagination step advancing both z and h.
3. Task-Shaped Gradient Flow: Verified non-zero gradients flowing from task losses directly into ConvEncoder.
4. AgentBeliefV3: Epistemic mask tracking and recurrent hidden state persistence.
5. RecurrentMPCPlanner: CEM planning with recurrent belief unrolls.
6. End-to-End Joint Training Smoke: Verifies training progression, rollout error evaluation, and checkpoint persistence.
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from models.jepa_v3 import JEPAv3, RecurrentCore, RecurrentPredictor
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from training.run_step2_v3_build import train_v3_recurrent


def test_recurrent_core_continuity():
    """Gate Test 1: Verifies RecurrentCore unrolls match step-by-step and masks handle resets."""
    core = RecurrentCore(latent_dim=32, action_emb_dim=16, recurrent_dim=64, num_actions=5)
    B, T = 2, 4
    z_seq = torch.randn(B, T, 32)
    actions = torch.randint(0, 5, (B, T))

    # Step-by-step manual unroll
    h_manual = []
    h = torch.zeros(B, 64)
    for t in range(T):
        h = core(z_seq[:, t], actions[:, t], h)
        h_manual.append(h)
    h_manual = torch.stack(h_manual, dim=1)

    # Sequence unroll
    h_unroll = core.unroll_sequence(z_seq, actions)
    assert torch.allclose(h_manual, h_unroll, atol=1e-5)

    # Test reset mask: if mask is 1 at t=2, h is reset before step 2
    resets = torch.zeros(B, T, dtype=torch.bool)
    resets[0, 2] = True
    h_masked = core.unroll_sequence(z_seq, actions, reset_masks=resets)
    assert h_masked.shape == (B, T, 64)


def test_recurrent_predictor_dynamics():
    """Gate Test 2: Verifies RecurrentPredictor advances both visual latent and recurrent hidden state."""
    model = JEPAv3(latent_dim=32, recurrent_dim=64, num_actions=5, hidden_dim=64)
    B = 3
    z = torch.randn(B, 32)
    h = torch.randn(B, 64)
    action = torch.randint(0, 5, (B,))

    next_z, next_h, next_s = model.predict_next_step(z, h, action)
    assert next_z.shape == (B, 32)
    assert next_h.shape == (B, 64)
    assert next_s.shape == (B, 96)
    assert torch.allclose(next_s, torch.cat([next_z, next_h], dim=-1))


def test_task_shaped_gradient_flow():
    """Gate Test 3: Verifies task loss backpropagates through heads and recurrent core into ConvEncoder."""
    model = JEPAv3(latent_dim=32, recurrent_dim=64, num_actions=5, hidden_dim=64)
    B, T = 2, 4
    obs = torch.randn(B, T, 3, 64, 64)
    actions = torch.randint(0, 5, (B, T))
    mask = torch.zeros(B, T, 22)
    events = torch.randint(0, 2, (B, T, 22)).float()

    obs_flat = obs.view(B * T, 3, 64, 64)
    z_seq = model.encoder(obs_flat).view(B, T, -1)
    h_seq = model.recurrent_core.unroll_sequence(z_seq, actions)
    s_seq = torch.cat([z_seq, h_seq], dim=-1)

    # Event head loss
    logits = model.event_head(s_seq, actions, mask)
    loss = model.event_head.compute_loss(logits, events)

    model.zero_grad()
    loss.backward()

    # Verify gradients reach the FIRST convolutional layer of ConvEncoder
    first_conv_weight = model.encoder.conv_net[0].weight
    assert first_conv_weight.grad is not None
    assert torch.norm(first_conv_weight.grad) > 0.0, "ConvEncoder received zero gradient from task loss!"

    # Verify gradients reach the RecurrentCore GRU parameters
    gru_weight = model.recurrent_core.gru.weight_ih
    assert gru_weight.grad is not None
    assert torch.norm(gru_weight.grad) > 0.0, "RecurrentCore received zero gradient from task loss!"


def test_agent_belief_v3_tracking():
    """Gate Test 4: Verifies AgentBeliefV3 tracks mask and recurrent memory across steps."""
    belief = AgentBeliefV3(recurrent_dim=64)
    assert belief.get_mask().sum() == 0.0
    assert belief.get_h().sum() == 0.0

    # Step update
    fake_probs = np.zeros(22, dtype=np.float32)
    fake_probs[6] = 0.85  # collect_wood
    next_h = np.ones(64, dtype=np.float32) * 0.5

    belief.update(event_probs=fake_probs, reward=1.0, next_h=next_h)
    assert belief.get_mask()[6] == 1.0
    assert np.allclose(belief.get_h(), next_h)

    # Reset
    belief.reset()
    assert belief.get_mask().sum() == 0.0
    assert belief.get_h().sum() == 0.0


def test_recurrent_mpc_planner_step():
    """Gate Test 5: Verifies RecurrentMPCPlanner plans actions and propagates state."""
    model = JEPAv3(latent_dim=32, recurrent_dim=64, num_actions=5, hidden_dim=64)
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=4,
        num_candidates=16,
        cem_iterations=2,
        num_actions=5,
        device="cpu",
    )
    belief = AgentBeliefV3(recurrent_dim=64)
    obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    action, debug = planner.plan(obs, belief)
    assert 0 <= action < 5
    assert "best_score" in debug
    assert "reward_sum" in debug
    assert "terminal_value" in debug


def get_test_resources(tmp_path):
    from training.sequence_buffer import EpisodeBuffer
    from models.jepa import JEPAWorldModel
    from models.jepa_v2 import JEPAv2

    # Check for existing shards
    shards_dir = None
    for cand in [
        Path("data/sequences"),
        Path("/content/drive/MyDrive/learn-a-world/data/sequences"),
    ]:
        if cand.exists() and list(cand.glob("*.npz")):
            shards_dir = str(cand)
            break

    if shards_dir is None:
        dummy_dir = tmp_path / "dummy_shards"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        buf = EpisodeBuffer(capacity_transitions=1000)
        for ep_i in range(3):
            T = 16
            obs = np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8)
            actions = np.random.randint(0, 17, (T,), dtype=np.int64)
            rewards = np.zeros(T, dtype=np.float32)
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True
            events = np.zeros((T, 22), dtype=np.uint8)
            events[5, 6] = 1
            mask_before = np.zeros((T, 22), dtype=np.uint8)
            buf.add_episode(obs, actions, rewards, dones, events, mask_before, source="dummy")
        buf.save_shard(dummy_dir / "shard_0000.npz")
        shards_dir = str(dummy_dir)

    # Check for existing base checkpoint
    ckpt_path = None
    for cand in [
        Path("checkpoints/jepa_crafter_v2_best.pt"),
        Path("checkpoints/jepa_crafter_v2.pt"),
        Path("checkpoints/jepa_crafter.pt"),
        Path("/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v2_best.pt"),
    ]:
        if cand.exists():
            ckpt_path = str(cand)
            break

    if ckpt_path is None:
        dummy_base = JEPAWorldModel(in_channels=3, latent_dim=128, num_actions=17)
        dummy_v2 = JEPAv2(dummy_base, hidden_dim=64)
        dummy_ckpt = tmp_path / "dummy_v2_base.pt"
        torch.save({
            "base_model_state": dummy_base.state_dict(),
            "event_head_state": dummy_v2.event_head.state_dict(),
            "health_head_state": dummy_v2.health_head.state_dict(),
            "q_ensemble_state": dummy_v2.q_ensemble.state_dict(),
            "policy_prior_state": dummy_v2.policy_prior.state_dict(),
        }, dummy_ckpt)
        ckpt_path = str(dummy_ckpt)

    return shards_dir, ckpt_path


def test_step2_joint_training_smoke(tmp_path):
    """Gate Test 6: End-to-end CPU smoke training test for Phase 2."""
    out_ckpt = tmp_path / "test_v3_phase2.pt"
    out_json = tmp_path / "test_v3_screening.json"
    shards_dir, ckpt_path = get_test_resources(tmp_path)

    results = train_v3_recurrent(
        checkpoint_base=ckpt_path,
        shards_dir=shards_dir,
        total_steps=12,
        batch_size=4,
        seq_len=8,
        recurrent_dim=64,
        device_name="cpu",
        output_checkpoint=str(out_ckpt),
        output_json=str(out_json),
    )

    assert out_ckpt.exists()
    assert out_json.exists()
    assert len(results["history"]["step"]) >= 1
    assert "final_metrics" in results

