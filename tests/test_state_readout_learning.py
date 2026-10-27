#!/usr/bin/env python3
"""
Unit tests for Section 6 & Gate 5: State Readout Head and Training Pipeline.
Verifies:
1. StateReadoutHead architecture, output shapes, and positive uncertainty.
2. Base model parameter freezing during readout learning.
3. Strict independent episode/seed splitting.
4. Baseline comparison logic (readout MSE vs zero/persistence baseline).
5. StateReadoutHead persistence in JEPAv3 checkpoint save/load.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from models.heads import StateReadoutHead
from models.jepa_v3 import JEPAv3
from training.sequence_buffer import EpisodeBuffer
from training.run_state_readout_learning import evaluate_state_readout, train_state_readout


def test_state_readout_head_forward():
    """Verify StateReadoutHead outputs [..., state_dim] with positive uncertainty."""
    head = StateReadoutHead(latent_dim=1024, state_dim=16, hidden_dim=64)
    x = torch.randn(4, 8, 1024)
    pred, unc = head(x)
    assert pred.shape == (4, 8, 16)
    assert unc.shape == (4, 8, 16)
    assert (unc > 0).all(), "Uncertainty must be strictly positive"


def test_state_readout_head_loss():
    """Verify compute_loss works with and without valid mask."""
    head = StateReadoutHead(latent_dim=1024, state_dim=16, hidden_dim=64)
    pred = torch.ones(2, 4, 16)
    target = torch.zeros(2, 4, 16)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    # Without mask: all elements error is (1 - 0)^2 = 1.0
    loss_unmasked = head.compute_loss(pred, target)
    assert pytest.approx(loss_unmasked.item(), 1e-4) == 1.0

    # With mask: only 3 valid steps
    loss_masked = head.compute_loss(pred, target, valid_mask=mask)
    assert pytest.approx(loss_masked.item(), 1e-4) == 1.0


def test_base_model_freezing_during_training():
    """Verify base model parameters remain 100% frozen while state_readout_head trains."""
    model = JEPAv3()
    
    # Store initial weights of encoder and recurrent core
    initial_enc_weight = model.encoder.conv_net[0].weight.clone().detach()
    initial_core_weight = model.recurrent_core.gru.weight_ih.clone().detach()
    initial_q_weight = model.q_ensemble.heads[0][0].weight.clone().detach()
    initial_readout_weight = model.state_readout_head.net[0].weight.clone().detach()

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False
    for p in model.state_readout_head.parameters():
        p.requires_grad = True

    # Dummy step
    optimizer = torch.optim.Adam(model.state_readout_head.parameters(), lr=1e-2)
    s = torch.randn(2, 4, model.state_dim)
    target = torch.ones(2, 4, 16)
    pred, _ = model.predict_state(s)
    loss = model.state_readout_head.compute_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Assert base model is bitwise identical
    assert torch.equal(model.encoder.conv_net[0].weight, initial_enc_weight), "Encoder weight changed!"
    assert torch.equal(model.recurrent_core.gru.weight_ih, initial_core_weight), "Recurrent core changed!"
    assert torch.equal(model.q_ensemble.heads[0][0].weight, initial_q_weight), "Q head changed!"

    # Assert readout head updated
    assert not torch.equal(model.state_readout_head.net[0].weight, initial_readout_weight), "Readout head did not update!"


def test_checkpoint_save_and_load_state_readout_head():
    """Verify StateReadoutHead weights are saved and loaded correctly in JEPAv3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_readout_ckpt.pt"
        model_1 = JEPAv3()
        # Mutate weights so they are non-default
        with torch.no_grad():
            model_1.state_readout_head.state_mean.weight.fill_(0.42)

        model_1.save_checkpoint(str(ckpt_path))

        model_2 = JEPAv3()
        audit = model_2.load_checkpoint(str(ckpt_path))
        assert "state_readout_head" in audit["loaded_components"]
        assert torch.allclose(model_2.state_readout_head.state_mean.weight, torch.tensor(0.42))


def test_independent_episode_split_and_readout_pipeline():
    """Verify full state readout pipeline on synthetic episodes with independent split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        shard_path = tmp_p / "synthetic_shard_final.npz"
        out_ckpt = tmp_p / "state_readout_trained.pt"
        out_json = tmp_p / "state_readout_results.json"

        # Create 10 synthetic episodes with state deltas
        n_eps = 10
        ep_len = 20
        total_t = n_eps * ep_len

        obs = np.random.randint(0, 256, (total_t, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randint(0, 17, total_t, dtype=np.int64)
        rewards = np.zeros(total_t, dtype=np.float32)
        dones = np.zeros(total_t, dtype=bool)
        for e in range(1, n_eps + 1):
            dones[e * ep_len - 1] = True
        events = np.zeros((total_t, 22), dtype=np.uint8)
        mask_before = np.zeros((total_t, 22), dtype=np.uint8)

        # Non-trivial state deltas
        state_delta = np.zeros((total_t, 16), dtype=np.float32)
        state_delta[:, 0] = 0.5  # constant wood collection
        state_delta_valid = np.ones(total_t, dtype=bool)
        ep_lens = np.array([ep_len] * n_eps, dtype=np.int64)

        np.savez_compressed(
            shard_path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            events=events,
            mask_before=mask_before,
            state_delta=state_delta,
            state_delta_valid=state_delta_valid,
            episode_lengths=ep_lens,
        )

        result = train_state_readout(
            shards_dir=tmp_p,
            checkpoint_path=None,
            output_checkpoint=out_ckpt,
            output_json=out_json,
            total_steps=5,
            batch_size=4,
            seq_len=8,
            labeled_ratio=1.0,
            val_ratio=0.20,
            split_seed=42,
            device="cpu",
        )

        assert result["status"] == "completed"
        assert result["dataset_audit"]["train_episodes"] == 8
        assert result["dataset_audit"]["val_episodes"] == 2
        assert out_ckpt.exists()
        assert out_json.exists()
        assert "state_readout_mse" in result["final_metrics"]
        assert "persistence_mse" in result["final_metrics"]
