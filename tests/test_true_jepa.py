"""Focused integrity and smoke tests for the clean true_jepa line."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import TrueJEPA
from true_jepa.objectives import train_world_model_step
from true_jepa.relations import AnonymousRelationMemory, TransitionEvidence
from true_jepa.replay import PixelEpisode, PixelReplay
from true_jepa.self_play import SelfPlayCoordinator
from true_jepa.validate import validate_replay


def tiny_model() -> TrueJEPA:
    return TrueJEPA(
        JEPAConfig(
            token_dim=32,
            belief_dim=40,
            predictor_heads=4,
            predictor_layers=1,
            predictor_mlp_dim=64,
            ensemble_size=2,
            rollout_horizon=2,
        ),
        num_actions=5,
    )


def make_episode(length: int = 6) -> PixelEpisode:
    rng = np.random.default_rng(12)
    return PixelEpisode(
        obs=rng.integers(0, 256, size=(length + 1, 64, 64, 3), dtype=np.uint8),
        actions=rng.integers(0, 5, size=length, dtype=np.int64),
        rewards=rng.normal(size=length).astype(np.float32),
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        seed=17,
        source="synthetic",
    )


def test_model_is_action_conditioned_and_target_is_ema_separate():
    model = tiny_model()
    obs = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    first = model.encode(obs)
    hidden = model.initial_belief(2, "cpu")
    action_a = torch.zeros(2, dtype=torch.long)
    action_b = torch.ones(2, dtype=torch.long)
    prediction_a = model.predict(first, action_a, hidden).pooled
    prediction_b = model.predict(first, action_b, hidden).pooled
    assert not torch.allclose(prediction_a, prediction_b)
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    online_before = [value.detach().clone() for value in model.encoder.parameters()]
    target_before = [value.detach().clone() for value in model.target_encoder.parameters()]
    with torch.no_grad():
        for value in model.encoder.parameters():
            value.add_(0.1)
    model.update_target(1)
    assert any(not torch.equal(a, b) for a, b in zip(target_before, model.target_encoder.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(online_before, model.encoder.parameters()))


def test_replay_windows_are_boundary_safe_and_round_trip(tmp_path):
    replay = PixelReplay()
    replay.add(make_episode())
    batch = replay.sample(3, 4, np.random.default_rng(3))
    assert batch["obs"].shape == (3, 5, 64, 64, 3)
    assert batch["actions"].shape == (3, 4)
    path = tmp_path / "replay.npz"
    replay.save(str(path))
    restored = PixelReplay.load(str(path))
    assert restored.episodes == 1
    assert len(restored.records[0].actions) == 6


def test_one_closed_loop_jepa_update_and_checkpoint(tmp_path):
    model = tiny_model()
    replay = PixelReplay()
    replay.add(make_episode())
    batch = replay.sample(2, 4, np.random.default_rng(4))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    metrics = train_world_model_step(
        model,
        batch,
        optimizer,
        torch.device("cpu"),
        TrainConfig(batch_size=2, sequence_length=4, amp=False),
        update=1,
    )
    assert np.isfinite(metrics["loss"])
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path), update=1)
    restored, meta = TrueJEPA.load_checkpoint(str(path))
    assert meta["update"] == 1
    assert torch.equal(model.target_encoder.position, restored.target_encoder.position)


def test_relation_memory_needs_repeated_evidence_and_has_no_semantic_table():
    memory = AnonymousRelationMemory(
        effect_similarity=0.95,
        minimum_support=4,
        minimum_episodes=2,
        success_margin=0.1,
    )
    direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    for episode in range(3):
        for step in range(2):
            memory.observe(
                TransitionEvidence(
                    context=direction,
                    action=2,
                    effect=direction,
                    outcome=1.0,
                    episode=episode,
                    step=step,
                )
            )
    assert len(memory.relations) == 1
    assert memory.usable(0)
    assert memory.rank_actions(direction, [1, 2])[0][0] == 2
    assert isinstance(memory.state_dict()["relations"], list)


def test_strict_agent_signature_and_source_isolation():
    signature = inspect.signature(VisualJEPAAgent.act)
    assert list(signature.parameters) == ["self", "obs_uint8", "reward_prev", "step"]
    source_root = Path(__file__).parents[1] / "true_jepa"
    source = "\n".join(path.read_text() for path in source_root.glob("*.py"))
    for banned in ("CRAFT_REQUIRED_MATERIALS", "INVENTORY_KEYS", "ACHIEVEMENTS"):
        assert banned not in source


def test_self_play_load_accepts_parallel_environment_count():
    signature = inspect.signature(SelfPlayCoordinator.load)
    assert "num_envs" in signature.parameters
    assert signature.parameters["num_envs"].default == 1


def test_random_collection_mode_is_explicit():
    signature = inspect.signature(SelfPlayCoordinator.__init__)
    assert signature.parameters["collection_policy"].default == "mixed"


def test_validation_reports_action_reward_and_representation_gates():
    model = tiny_model()
    replay = PixelReplay()
    replay.add(make_episode())
    report = validate_replay(
        model,
        replay,
        torch.device("cpu"),
        batches=1,
        batch_size=2,
        sequence_length=2,
        seed=8,
    )
    assert report["valid_transitions"] > 0
    assert "action_conditioning" in report
    assert "reward_prediction" in report
    assert "representation" in report
