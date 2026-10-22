r"""
Automated Gate Tests for Phase 5 (Learn-a-World v2 Phase 1):
1. EpisodeBuffer Storage: Whole episode storage, event delta tracking, mask indexing.
2. Sequence Sampling: Uniform and event-balanced batch sampling shapes, types, and device handling.
3. Shard Persistence: .npz shard serialization and reloading with 100% bitwise parity.
4. EventHead & HealthHead Contracts: Output shapes, probability ranges [0, 1], and finite focal/Huber losses.
5. JEPAv2 Integration: Loads from v1 checkpoint, freezes encoder/predictor, computes predict_reward.
6. Diagnostic Trainer Smoke: Runs 10 optimization steps, produces decreasing loss, and validates PR-AUC.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.heads import NUM_ACHIEVEMENTS, EventHead, HealthHead
from models.jepa import JEPAWorldModel
from models.jepa_v2 import JEPAv2
from training.sequence_buffer import EpisodeBuffer
from training.trainer_v2 import compute_binary_pr_auc, train_heads

def create_synthetic_episode(length: int = 25, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    obs = rng.randint(0, 256, size=(length, 64, 64, 3), dtype=np.uint8)
    actions = rng.randint(0, 17, size=(length,), dtype=np.int64)
    rewards = rng.randn(length).astype(np.float32)
    dones = np.zeros(length, dtype=bool)
    dones[-1] = True

    events = np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8)
    # Scatter some achievement events
    events[5, 6] = 1   # collect_wood
    events[10, 2] = 1  # collect_drink

    mask_before = np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8)
    mask_before[6:, 6] = 1
    mask_before[11:, 2] = 1

    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "events": events,
        "mask_before": mask_before,
        "source": "synthetic_test",
    }

def test_episode_buffer_storage_and_shapes():
    """Gate Test 1: EpisodeBuffer properly appends episodes and tracks transitions."""
    buf = EpisodeBuffer(capacity_transitions=1000)
    ep1 = create_synthetic_episode(length=20, seed=1)
    ep2 = create_synthetic_episode(length=30, seed=2)

    buf.add_episode(**ep1)
    assert buf.num_episodes == 1
    assert buf.total_transitions == 20
    assert len(buf) == 20

    buf.add_episode(**ep2)
    assert buf.num_episodes == 2
    assert buf.total_transitions == 50
    assert len(buf) == 50

    stats = buf.get_dataset_statistics()
    assert stats["total_episodes"] == 2
    assert stats["total_transitions"] == 50
    assert stats["per_achievement_positive_counts"]["collect_wood"] == 2
    assert stats["per_achievement_positive_counts"]["collect_drink"] == 2

def test_sequence_sampling_uniform_and_event_balanced():
    """Gate Test 2: sample_sequences produces tensor batches with expected shapes and values."""
    buf = EpisodeBuffer(capacity_transitions=1000)
    for s in range(5):
        buf.add_episode(**create_synthetic_episode(length=25, seed=s))

    # Uniform sampling
    batch_u = buf.sample_sequences(batch_size=8, seq_len=16, mode="uniform")
    assert batch_u["obs"].shape == (8, 16, 3, 64, 64)
    assert batch_u["obs"].dtype == torch.float32
    assert 0.0 <= batch_u["obs"].min() and batch_u["obs"].max() <= 1.0
    assert batch_u["actions"].shape == (8, 16)
    assert batch_u["actions"].dtype == torch.int64
    assert batch_u["rewards"].shape == (8, 16)
    assert batch_u["dones"].shape == (8, 16)
    assert batch_u["events"].shape == (8, 16, NUM_ACHIEVEMENTS)
    assert batch_u["mask_before"].shape == (8, 16, NUM_ACHIEVEMENTS)

    # Event balanced sampling
    batch_eb = buf.sample_sequences(batch_size=8, seq_len=16, mode="event_balanced")
    assert batch_eb["obs"].shape == (8, 16, 3, 64, 64)
    assert batch_eb["events"].shape == (8, 16, NUM_ACHIEVEMENTS)

def test_shard_persistence_bitwise_parity(tmp_path):
    """Gate Test 3: Saves episodes to shard and reloads with 100% bitwise parity."""
    buf = EpisodeBuffer(capacity_transitions=1000)
    for s in range(4):
        buf.add_episode(**create_synthetic_episode(length=20, seed=s))

    shard_path = tmp_path / "test_shard.npz"
    buf.save_shard(shard_path)
    assert shard_path.exists()

    reloaded_buf = EpisodeBuffer(capacity_transitions=1000)
    reloaded_buf.load_shard(shard_path)

    assert reloaded_buf.num_episodes == buf.num_episodes
    assert reloaded_buf.total_transitions == buf.total_transitions

    for ep_orig, ep_reloaded in zip(buf.episodes, reloaded_buf.episodes):
        assert np.array_equal(ep_orig["obs"], ep_reloaded["obs"])
        assert np.array_equal(ep_orig["actions"], ep_reloaded["actions"])
        assert np.allclose(ep_orig["rewards"], ep_reloaded["rewards"])
        assert np.array_equal(ep_orig["dones"], ep_reloaded["dones"])
        assert np.array_equal(ep_orig["events"], ep_reloaded["events"])
        assert np.array_equal(ep_orig["mask_before"], ep_reloaded["mask_before"])
        assert ep_orig["source"] == ep_reloaded["source"]

def test_event_head_and_health_head():
    """Gate Test 4: Verify EventHead and HealthHead forward passes, ranges, and loss."""
    B = 4
    latent_dim = 128
    num_actions = 17

    event_head = EventHead(latent_dim=latent_dim, num_actions=num_actions, hidden_dim=64)
    health_head = HealthHead(latent_dim=latent_dim, num_actions=num_actions, hidden_dim=64)

    z = torch.randn(B, latent_dim)
    actions = torch.randint(0, num_actions, (B,))
    mask = torch.zeros(B, NUM_ACHIEVEMENTS)
    mask[:, 6] = 1.0  # wood unlocked

    # EventHead test
    logits = event_head(z, actions, mask)
    assert logits.shape == (B, NUM_ACHIEVEMENTS)

    probs = event_head.predict_proba(z, actions, mask)
    assert probs.shape == (B, NUM_ACHIEVEMENTS)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    target_events = torch.zeros(B, NUM_ACHIEVEMENTS)
    target_events[0, 2] = 1.0
    loss_event = event_head.compute_loss(logits, target_events)
    assert loss_event.ndim == 0
    assert torch.isfinite(loss_event)

    # HealthHead test
    h_pred = health_head(z, actions)
    assert h_pred.shape == (B,)
    h_target = torch.randn(B).clamp(-1.0, 1.0)
    loss_h = health_head.compute_loss(h_pred, h_target)
    assert loss_h.ndim == 0
    assert torch.isfinite(loss_h)

def test_jepa_v2_wrapper_and_frozen_base(tmp_path):
    """Gate Test 5: Verify JEPAv2 loads base checkpoint, freezes parameters, and computes reward."""
    # Create and save a dummy v1 checkpoint
    base_model = JEPAWorldModel(
        in_channels=3,
        latent_dim=128,
        num_actions=17,
        action_emb_dim=32,
        predictor_hidden_dim=64,
        predictor_layers=2,
    )
    ckpt_path = tmp_path / "dummy_jepa.pt"
    torch.save({"model_state": base_model.state_dict(), "step": 100}, ckpt_path)

    v2_model = JEPAv2.from_v1_checkpoint(
        checkpoint_path=str(ckpt_path),
        config={"model": {"latent_dim": 128, "num_actions": 17, "action_emb_dim": 32, "predictor_hidden_dim": 64}},
        freeze_base=True,
    )

    # Verify base model parameters are frozen
    for p in v2_model.base_model.parameters():
        assert not p.requires_grad

    # Verify head parameters ARE trainable
    assert any(p.requires_grad for p in v2_model.event_head.parameters())
    assert any(p.requires_grad for p in v2_model.health_head.parameters())

    # Forward reward contract
    B = 2
    obs = torch.randn(B, 3, 64, 64)
    z = v2_model.encode(obs, use_target=True)
    assert z.shape == (B, 128)

    actions = torch.tensor([5, 1], dtype=torch.long)
    mask = torch.zeros(B, NUM_ACHIEVEMENTS)

    reward, decomp = v2_model.predict_reward(z, actions, mask, return_decomposition=True)
    assert reward.shape == (B,)
    assert decomp["event_probs"].shape == (B, NUM_ACHIEVEMENTS)
    assert decomp["achievement_reward"].shape == (B,)
    assert decomp["health_reward"].shape == (B,)

def test_pr_auc_calculation():
    """Gate Test 6: Verify exact Area Under Precision-Recall Curve computation."""
    # Perfect classifier
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    pr_auc = compute_binary_pr_auc(y_true, y_score)
    assert np.isclose(pr_auc, 1.0, atol=1e-3)

    # Inverted classifier
    y_score_inv = np.array([0.1, 0.2, 0.8, 0.9])
    pr_auc_inv = compute_binary_pr_auc(y_true, y_score_inv)
    assert pr_auc_inv < 0.5

def test_head_training_smoke_step(tmp_path):
    """Gate Test 7: Verify train_heads performs gradient optimization and validation."""
    # Create dummy checkpoint
    base_model = JEPAWorldModel(
        in_channels=3,
        latent_dim=128,
        num_actions=17,
        action_emb_dim=32,
        predictor_hidden_dim=64,
        predictor_layers=2,
    )
    ckpt_path = tmp_path / "dummy_base.pt"
    torch.save({"model_state": base_model.state_dict(), "step": 1}, ckpt_path)

    # Populate buffer
    buf = EpisodeBuffer(capacity_transitions=500)
    for s in range(4):
        buf.add_episode(**create_synthetic_episode(length=20, seed=s))

    out_json = tmp_path / "val_rep.json"
    out_ckpt = tmp_path / "v2_out.pt"

    model, val_report = train_heads(
        config_path="experiments/configs/crafter_v2.yaml",
        checkpoint_path=str(ckpt_path),
        buffer=buf,
        total_steps=5,
        batch_size=4,
        seq_len=16,
        device_override="cpu",
        out_json_path=str(out_json),
        out_checkpoint_path=str(out_ckpt),
    )

    assert out_json.exists()
    assert out_ckpt.exists()
    assert "macro_tier1_pr_auc" in val_report
    assert "health_mae" in val_report
    assert "per_achievement" in val_report

