"""Focused gates for the implemented causal-chain Phase 0/1 work."""

import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.phase0_trace import ChainTraceRecorder
from models.jepa_v3 import JEPAv3
from training.sequence_buffer import EpisodeBuffer, NUM_ACHIEVEMENTS
from training.collector_v3 import _load_jepa_v3


def _episode(length=6, with_delta=True):
    kwargs = {}
    if with_delta:
        kwargs["state_delta"] = np.ones((length, 16), dtype=np.float32)
        kwargs["state_delta_valid"] = np.ones(length, dtype=np.bool_)
    return dict(
        obs=np.zeros((length, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(length, dtype=np.int64),
        rewards=np.zeros(length, dtype=np.float32),
        dones=np.array([False] * (length - 1) + [True]),
        events=np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8),
        mask_before=np.zeros((length, NUM_ACHIEVEMENTS), dtype=np.uint8),
        **kwargs,
    )


def test_phase1_transition_heads_are_action_conditioned_and_checkpointable(tmp_path):
    model = JEPAv3(latent_dim=16, recurrent_dim=8, hidden_dim=32, transition_state_dim=4)
    obs = torch.rand(2, 3, 64, 64)
    z = model.encode(obs)
    s, _ = model.recurrent_step(z, torch.tensor([0, 0]), torch.zeros(2, 8))
    first = model.predict_transition(s, torch.tensor([0, 0]))["state_delta"]
    second = model.predict_transition(s, torch.tensor([1, 1]))["state_delta"]
    assert first.shape == (2, 4)
    assert not torch.allclose(first, second)

    path = tmp_path / "phase1.pt"
    model.save_checkpoint(str(path))
    restored = JEPAv3(latent_dim=16, recurrent_dim=8, hidden_dim=32, transition_state_dim=4)
    audit = restored.load_checkpoint(str(path), device="cpu")
    assert "state_delta_head" in audit["loaded_components"]
    assert "transition_head" in audit["loaded_components"]


def test_sequence_buffer_keeps_legacy_shards_and_new_labels(tmp_path):
    new_buf = EpisodeBuffer(capacity_transitions=100)
    new_buf.add_episode(**_episode(with_delta=True), source="diagnostic")
    new_path = tmp_path / "new.npz"
    new_buf.save_shard(new_path)
    loaded = EpisodeBuffer(capacity_transitions=100)
    loaded.load_shard(new_path)
    batch = loaded.sample_sequences(batch_size=1, seq_len=4)
    assert batch["state_delta"].shape == (1, 4, 16)
    assert batch["state_delta_valid"].all()

    legacy = EpisodeBuffer(capacity_transitions=100)
    legacy.add_episode(**_episode(with_delta=False), source="legacy")
    legacy_path = tmp_path / "legacy.npz"
    legacy.save_shard(legacy_path)
    reloaded = EpisodeBuffer(capacity_transitions=100)
    reloaded.load_shard(legacy_path)
    legacy_batch = reloaded.sample_sequences(batch_size=1, seq_len=4)
    assert legacy_batch["state_delta"].shape[-1] == 16
    assert not legacy_batch["state_delta_valid"].any()


def test_phase0_trace_labels_privileged_state_only():
    strict = ChainTraceRecorder(track="strict_visual_jepa", ablation="test")
    strict.start_episode(1)
    strict.record_step(step=0, action=5, reward=0.0, done=False, info={"inventory": {"wood": 2}})
    result = strict.finish_episode({"achievements": {"collect_wood": 1}}, 0.0)
    assert "actual_inventory" not in result["steps"][0]
    assert result["first_failed_link"] == "table_execution"

    privileged = ChainTraceRecorder(track="privileged_diagnostic", ablation="test")
    privileged.start_episode(1)
    privileged.record_step(step=0, action=5, reward=0.0, done=False, info={"inventory": {"wood": 2}})
    result = privileged.finish_episode({"achievements": {}}, 0.0)
    assert result["steps"][0]["actual_inventory"]["wood"] == 2


def test_collector_loads_native_v3_checkpoint_components(tmp_path):
    source = JEPAv3()
    checkpoint = tmp_path / "native_v3.pt"
    source.save_checkpoint(str(checkpoint))
    loaded = _load_jepa_v3(str(checkpoint), device="cpu")
    key = next(iter(source.encoder.state_dict()))
    assert torch.allclose(
        source.encoder.state_dict()[key],
        loaded.encoder.state_dict()[key],
    )
