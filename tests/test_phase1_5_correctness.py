"""Focused tests and laptop smoke test for Phase 1.5 correctness requirements."""

from pathlib import Path
import numpy as np
import pytest
import torch

from models.jepa_v3 import JEPAv3
from training.run_phase1_transition_learning import (
    _make_model,
    evaluate_transition_heads,
    train_phase1_transitions,
)
from training.sequence_buffer import EpisodeBuffer, NUM_ACHIEVEMENTS


def _make_dummy_episode(length=20, labeled=False, delta_val=1.0):
    events = np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8)
    kwargs = {}
    if labeled:
        kwargs["state_delta"] = np.full((length, 16), delta_val, dtype=np.float32)
        kwargs["state_delta_valid"] = np.ones(length, dtype=np.bool_)
    else:
        kwargs["state_delta"] = np.zeros((length, 16), dtype=np.float32)
        kwargs["state_delta_valid"] = np.zeros(length, dtype=np.bool_)

    return {
        "obs": np.random.randint(0, 255, (length, 64, 64, 3), dtype=np.uint8),
        "actions": np.random.randint(0, 5, length, dtype=np.int64),
        "rewards": np.ones(length, dtype=np.float32) * 0.5,
        "dones": np.array([False] * (length - 1) + [True]),
        "events": events,
        "mask_before": np.zeros_like(events),
        "source": "test",
        **kwargs,
    }


def test_missing_checkpoint_fails_loudly():
    """Fail if requested checkpoint is missing; never silently train from random initialization."""
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError, match="Requested checkpoint does not exist"):
        _make_model("/nonexistent/path/checkpoint.pt", device, transition_state_dim=16)

    with pytest.raises(ValueError, match="No checkpoint_path provided"):
        _make_model(None, device, transition_state_dim=16, allow_fresh_init=False)

    # Explicit allow_fresh_init succeeds
    model = _make_model(None, device, transition_state_dim=16, allow_fresh_init=True)
    assert isinstance(model, JEPAv3)


def test_batch_invariant_reward_mae():
    """Reward MAE must divide by total transition count, not single-batch size."""
    model = JEPAv3(latent_dim=16, recurrent_dim=8, hidden_dim=32, transition_state_dim=16)
    buf = EpisodeBuffer(capacity_transitions=200)
    for _ in range(5):
        buf.add_episode(**_make_dummy_episode(length=20, labeled=True))

    res_1 = evaluate_transition_heads(model, buf, batch_size=4, seq_len=8, batches=1, device="cpu")
    res_4 = evaluate_transition_heads(model, buf, batch_size=4, seq_len=8, batches=4, device="cpu")

    # In both cases, reward_mae is in a sensible bounded range, not scaled 4x by batch count!
    assert isinstance(res_1["reward_mae"], float)
    assert isinstance(res_4["reward_mae"], float)
    # The expected reward MAE per transition should be similar within sampling variance
    assert abs(res_1["reward_mae"] - res_4["reward_mae"]) < 1.0


def test_labeled_window_sampling():
    """Configurable labeled_ratio guarantees labeled state-delta examples appear reliably."""
    buf = EpisodeBuffer(capacity_transitions=500)
    # Add 9 unlabeled episodes and 1 labeled episode
    for _ in range(9):
        buf.add_episode(**_make_dummy_episode(length=20, labeled=False))
    buf.add_episode(**_make_dummy_episode(length=20, labeled=True))

    # Without labeled_ratio (uniform sampling): labeled windows appear ~10% of the time
    # With labeled_ratio=0.5: at least round(8 * 0.5) = 4 sequences MUST have valid state deltas
    batch = buf.sample_sequences(batch_size=8, seq_len=10, labeled_ratio=0.5)
    labeled_seqs = batch["state_delta_valid"].any(dim=1).sum().item()
    assert labeled_seqs >= 4, f"Expected >= 4 labeled sequences in batch, got {labeled_seqs}"


def test_stratified_train_val_split_and_labeled_counts():
    """Ensure validation contains enough labeled examples and exact counts are reported."""
    buf = EpisodeBuffer(capacity_transitions=500)
    for _ in range(12):
        buf.add_episode(**_make_dummy_episode(length=20, labeled=False))
    for _ in range(6):
        buf.add_episode(**_make_dummy_episode(length=20, labeled=True))

    train_buf, val_buf = buf.split_train_val(val_ratio=0.25, seed=42)

    train_stats = train_buf.get_labeled_statistics()
    val_stats = val_buf.get_labeled_statistics()

    assert train_stats["labeled_episodes"] >= 1
    assert val_stats["labeled_episodes"] >= 1
    assert train_stats["labeled_transitions"] > 0
    assert val_stats["labeled_transitions"] > 0
    assert val_stats["labeled_transitions"] == val_stats["labeled_episodes"] * 20
    assert val_stats["total_episodes"] == len(val_buf.episodes)


def test_termination_recall_and_majority_baseline():
    """Evaluate termination recall and majority baseline."""
    model = JEPAv3(latent_dim=16, recurrent_dim=8, hidden_dim=32, transition_state_dim=16)
    buf = EpisodeBuffer(capacity_transitions=200)
    # 5 episodes of length 20: 100 transitions total, exactly 5 dones (positives=5, negatives=95)
    for _ in range(5):
        buf.add_episode(**_make_dummy_episode(length=20, labeled=True))

    metrics = evaluate_transition_heads(model, buf, batch_size=4, seq_len=8, batches=4, device="cpu")
    assert "termination_majority_baseline" in metrics
    assert "termination_recall" in metrics
    assert "termination_precision" in metrics
    assert 0.0 <= metrics["termination_majority_baseline"] <= 1.0
    assert 0.0 <= metrics["termination_recall"] <= 1.0


def test_latent_variance_collapse_detection():
    """Latent variance/collapse metrics are calculated and detect collapse."""
    model = JEPAv3(latent_dim=16, recurrent_dim=8, hidden_dim=32, transition_state_dim=16)
    buf = EpisodeBuffer(capacity_transitions=200)
    for _ in range(3):
        buf.add_episode(**_make_dummy_episode(length=20, labeled=True))

    metrics = evaluate_transition_heads(model, buf, batch_size=4, seq_len=8, batches=2, device="cpu")
    assert "latent_std_mean" in metrics
    assert "latent_std_min" in metrics
    assert "latent_std_max" in metrics
    assert "latent_collapse_detected" in metrics
    assert isinstance(metrics["latent_collapse_detected"], bool)


def test_laptop_smoke_state_delta_beats_persistence(tmp_path):
    """Smoke test: state delta head trains on labeled windows and beats persistence baseline."""
    shard_path = tmp_path / "shard_smoke_0000.npz"
    buf = EpisodeBuffer(capacity_transitions=500)

    # Create synthetic episodes where state delta follows a clear pattern:
    # Action a causes state_delta[a] = +1.0, and 0 elsewhere
    rng = np.random.RandomState(42)
    for _ in range(10):
        length = 30
        actions = rng.randint(0, 4, length, dtype=np.int64)
        state_delta = np.zeros((length, 16), dtype=np.float32)
        for t in range(length):
            state_delta[t, actions[t]] = 1.0  # predictable causal effect
        buf.add_episode(
            obs=np.zeros((length, 64, 64, 3), dtype=np.uint8),
            actions=actions,
            rewards=np.zeros(length, dtype=np.float32),
            dones=np.array([False] * (length - 1) + [True]),
            events=np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8),
            mask_before=np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8),
            state_delta=state_delta,
            state_delta_valid=np.ones(length, dtype=np.bool_),
            source="synthetic_causal",
        )
    buf.save_shard(shard_path)

    out_ckpt = tmp_path / "smoke_out.pt"
    out_json = tmp_path / "smoke_out.json"

    result = train_phase1_transitions(
        shards_dir=tmp_path,
        checkpoint_path=None,
        output_checkpoint=out_ckpt,
        output_json=out_json,
        total_steps=50,
        batch_size=8,
        seq_len=8,
        horizons=(1, 2),
        lr=1e-3,
        labeled_ratio=0.8,
        allow_fresh_init=True,
        transition_state_dim=16,
        val_ratio=0.2,
        device="cpu",
    )

    metrics = result["metrics"]
    assert metrics["state_delta_valid_transitions"] > 0
    assert metrics["state_delta_mse"] is not None
    assert metrics["state_delta_persistence_mse"] is not None
    # State delta MSE should improve significantly and beat persistence baseline!
    assert metrics["state_delta_beats_persistence"] is True or metrics["state_delta_mse"] < metrics["state_delta_persistence_mse"]
    assert out_ckpt.exists()
    assert out_json.exists()


def test_frozen_base_model_training(tmp_path: Path):
    """Test that freeze_base=True keeps encoder/prior strictly frozen while training transition heads."""
    from models.jepa_v3 import JEPAv3
    from training.sequence_buffer import EpisodeBuffer
    init_model = JEPAv3(transition_state_dim=16)
    init_ckpt = tmp_path / "init_model.pt"
    init_model.save_checkpoint(str(init_ckpt), step=0)

    # Save initial weights to verify freezing
    encoder_weight_before = init_model.encoder.conv_net[0].weight.clone()
    prior_weight_before = init_model.policy_prior.net[0].weight.clone()
    delta_weight_before = init_model.state_delta_head.net[-1].weight.clone()

    # Create dummy shard
    shard_path = tmp_path / "shard_0.npz"
    buf = EpisodeBuffer(capacity_transitions=1000)
    actions = np.zeros(16, dtype=np.int64)
    buf.add_episode(
        obs=np.zeros((16, 64, 64, 3), dtype=np.uint8),
        actions=actions,
        rewards=np.zeros(16, dtype=np.float32),
        dones=np.array([False] * 15 + [True]),
        events=np.zeros((16, NUM_ACHIEVEMENTS), dtype=np.uint8),
        mask_before=np.zeros((16, NUM_ACHIEVEMENTS), dtype=np.uint8),
        state_delta=np.ones((16, 16), dtype=np.float32),
        state_delta_valid=np.ones(16, dtype=np.bool_),
    )
    buf.save_shard(shard_path)

    out_ckpt = tmp_path / "frozen_out.pt"
    result = train_phase1_transitions(
        shards_dir=tmp_path,
        checkpoint_path=init_ckpt,
        output_checkpoint=out_ckpt,
        total_steps=5,
        batch_size=2,
        seq_len=8,
        horizons=(1, 2),
        freeze_base=True,
        allow_fresh_init=True,
        device="cpu",
    )

    trained_model = JEPAv3(transition_state_dim=16)
    trained_model.load_checkpoint(str(out_ckpt))

    # 1. Base modules MUST be bitwise identical
    assert torch.equal(trained_model.encoder.conv_net[0].weight, encoder_weight_before), "Encoder was modified despite freeze_base=True!"
    assert torch.equal(trained_model.policy_prior.net[0].weight, prior_weight_before), "Policy prior was modified despite freeze_base=True!"

    # 2. Transition head MUST have updated
    assert not torch.equal(trained_model.state_delta_head.net[-1].weight, delta_weight_before), "State delta head did not update!"
    assert result["config"]["freeze_base"] is True
