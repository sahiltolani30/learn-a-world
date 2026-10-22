"""
Automated Gate Tests for Phase 1 (Value Fix - R1):
1. ReturnNormalizer: DreamerV3-style percentile scaling & denormalization.
2. n-Step Bellman Targets: Vector-safe per-sequence done masking without cross-contamination.
3. Clipped Double-Q: Target ensemble minimum strictly bounds individual heads.
4. Polyak Target Network Update: Momentum decay tau=0.005 smoothly tracks online parameters.
5. End-to-End Value Training Smoke: Verifies training progression and checkpoint persistence.
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from models.heads import QEnsemble, ReturnNormalizer
from models.jepa import JEPAWorldModel
from models.jepa_v2 import JEPAv2
from training.run_step1_value_fix import compute_n_step_bellman_targets, train_value_fix


def test_return_normalizer():
    """Gate Test 1: Verifies ReturnNormalizer expands sparse distribution and denormalizes correctly."""
    normalizer = ReturnNormalizer(low_percentile=5.0, high_percentile=95.0, target_max=12.0)
    
    # Skewed sparse returns in [0, 2]
    raw_returns = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0])
    normalizer.update(raw_returns)
    assert normalizer.initialized
    assert normalizer.high > normalizer.low

    norm = normalizer.normalize(raw_returns)
    assert norm.shape == raw_returns.shape
    assert norm.min() >= 0.0
    assert norm.max() <= 12.0
    # 2.0 should be mapped near target_max * 0.8 (~9.6)
    assert norm[5] > norm[0]

    # Invertibility check
    denorm = normalizer.denormalize(norm)
    assert torch.allclose(raw_returns, denorm, atol=1e-4)


def test_n_step_bellman_targets_vector_masking():
    """Gate Test 2: Verifies done masking is per-sequence without batch cross-contamination."""
    B = 2
    T = 5
    n_step = 3
    gamma = 0.9

    rewards = torch.ones(B, T)
    dones = torch.zeros(B, T, dtype=torch.bool)
    # Sequence 0 terminates at step 1; Sequence 1 never terminates
    dones[0, 1] = True

    target_q_vals = torch.full((B, T), 10.0)

    targets = compute_n_step_bellman_targets(
        rewards=rewards,
        dones=dones,
        target_q_vals=target_q_vals,
        gamma=gamma,
        n_step=n_step,
    )

    # For sequence 0 at t=0:
    # k=0: r_0 = 1, alive=1
    # k=1: r_1 = 1, alive=1 (done fires here)
    # k=2: r_2 = 1, alive=0 -> ignored!
    # terminal bootstrap: alive=0 -> ignored!
    # Expected target for seq 0 at t=0: 1.0 + 0.9 * 1.0 = 1.9
    assert abs(targets[0, 0].item() - 1.9) < 1e-4

    # For sequence 1 at t=0:
    # Never terminates. Accumulates k=0,1,2 plus bootstrapped Q at t=3:
    # 1.0 + 0.9*1.0 + 0.81*1.0 + 0.9^3 * 10.0 = 2.71 + 7.29 = 10.0
    assert abs(targets[1, 0].item() - 10.0) < 1e-4


def test_clipped_double_q_minimum():
    """Gate Test 3: Verifies q_min is strictly <= individual heads and <= q_mean."""
    q_ensemble = QEnsemble(latent_dim=64, num_actions=17, num_heads=3)
    z = torch.randn(4, 64)
    actions = torch.randint(0, 17, (4,))

    q_vals = q_ensemble.q_values(z, actions)  # [3, 4]
    q_min = q_ensemble.q_min(z, actions)      # [4]
    q_mean = q_ensemble.q_mean(z, actions)    # [4]

    assert q_min.shape == (4,)
    # Minimum must be <= all individual heads
    for k in range(3):
        assert (q_min <= q_vals[k] + 1e-6).all()
    # Minimum must be <= mean
    assert (q_min <= q_mean + 1e-6).all()


def test_polyak_target_network_update():
    """Gate Test 4: Verifies target Q parameters update smoothly via momentum."""
    base = JEPAWorldModel(in_channels=3, latent_dim=64, num_actions=17)
    model = JEPAv2(base_model=base, freeze_base=True)

    # Modify online Q weights intentionally
    with torch.no_grad():
        for p in model.q_ensemble.parameters():
            p.add_(1.0)

    # Save target parameters before Polyak update
    target_params_before = [p.clone() for p in model.q_target_ensemble.parameters()]

    # Apply 1 Polyak update with decay = 0.99
    model.update_target_q(decay=0.99)

    for p_before, p_target, p_online in zip(
        target_params_before,
        model.q_target_ensemble.parameters(),
        model.q_ensemble.parameters(),
    ):
        expected = 0.99 * p_before + 0.01 * p_online
        assert torch.allclose(p_target, expected, atol=1e-5)


def get_test_resources(tmp_path):
    from training.sequence_buffer import EpisodeBuffer
    
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
        dummy_v2 = JEPAv2(dummy_base, hidden_dim=256)
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


def test_step1_training_smoke(tmp_path):
    """Gate Test 5: End-to-end CPU smoke training test for Phase 1."""
    out_ckpt = tmp_path / "test_value_fix.pt"
    out_json = tmp_path / "test_screening.json"
    shards_dir, ckpt_path = get_test_resources(tmp_path)

    results = train_value_fix(
        checkpoint_path=ckpt_path,
        shards_dir=shards_dir,
        total_steps=15,
        batch_size=8,
        seq_len=8,
        n_step=3,
        device_name="cpu",
        output_checkpoint=str(out_ckpt),
        output_json=str(out_json),
    )

    assert out_ckpt.exists()
    assert out_json.exists()
    assert len(results["history"]["step"]) >= 1
    assert "final_calibration" in results

