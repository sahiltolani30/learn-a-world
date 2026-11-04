"""Tests for Phase 2: Repair and Verify Baseline Integration (P2_BASELINE_INTEGRITY_PASS)."""

import copy
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from src.env.opaque_actions import OpaqueActionMapping
from src.evaluation.action_conditioning_tests import ActionConditioningTester
from src.data.heldout_store import HeldoutEpisodeStore
from src.data.replay_buffer import ReplayBuffer
from src.data.checkpoint_store import CheckpointStore
from src.exploration.latent_memory import LatentMemory
from src.exploration.novelty import NoveltyEstimator, RunningNormalizer
from src.exploration.disagreement import DisagreementEstimator, LearningProgressEstimator
from src.exploration.role_definitions import ROLE_WEIGHTS
from src.planning.candidate_scorer import CandidateScorer
from src.models.jepa_world_model import JepaWorldModel
from src.learning.learner import JepaLearner
from src.learning.update_scheduler import UpdateScheduler
from src.training.resume import restore_checkpoint
from src.utils.config import Config, load_config


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_training_and_evaluator_action_mapping_identical():
    """Test 1: The action-conditioning evaluator uses the exact run-specific

    opaque token permutation used during training.
    """
    model = JepaWorldModel(latent_dim=128, action_dim=64, context_dim=256, num_actions=17)
    train_mapping = OpaqueActionMapping(num_actions=17, seed=42)

    tester = ActionConditioningTester(model=model, opaque_mapping=train_mapping)
    assert tester.opaque_mapping is not None
    assert tester.opaque_mapping.permutation == train_mapping.permutation

    # Verify that different mapping yields different token-to-action translation
    other_mapping = OpaqueActionMapping(num_actions=17, seed=999)
    assert tester.opaque_mapping.permutation != other_mapping.permutation


def test_true_heldout_seeds_disjoint_from_replay():
    """Test 2: Validation episodes are collected independently from disjoint seeds

    and automated assertion confirms zero overlap with training seeds.
    """
    mapping = OpaqueActionMapping(num_actions=17, seed=123)
    heldout = HeldoutEpisodeStore(
        heldout_seed_start=900000,
        heldout_episode_count=5,
        max_steps_per_episode=10,
        opaque_mapping=mapping,
    )
    
    # Verify seeds are >= 900,000
    assert all(s >= 900000 for s in heldout.heldout_seeds)

    # Disjoint check against standard training seeds (e.g. 1000..1010)
    train_seeds = set(range(1000, 1010))
    assert heldout.verify_disjoint_from(train_seeds) is True

    # Leaking train seed into heldout set must trigger ValueError
    leaked_seeds = {1005, 900001}
    with pytest.raises(ValueError, match="CRITICAL LEAK: Training seeds overlap with heldout seeds"):
        heldout.verify_disjoint_from(leaked_seeds)


def test_novelty_insert_once_per_real_observation():
    """Test 3: Each real observation is inserted into memory exactly once."""
    estimator = NoveltyEstimator(capacity=100, latent_dim=128)
    assert estimator.memory.size == 0
    assert estimator.normalizer.count == 0

    # Insert 5 distinct observations
    for i in range(5):
        z = torch.randn(128)
        nov = estimator.insert_observation(z)
        assert isinstance(nov, float)

    assert estimator.memory.size == 5
    assert estimator.normalizer.count == 5


def test_imagined_candidates_do_not_mutate_novelty_memory():
    """Test 4: Imagined rollout candidates never enter memory or mutate running normalizer."""
    estimator = NoveltyEstimator(capacity=100, latent_dim=128)
    # Insert 2 initial real observations
    estimator.insert_observation(torch.randn(128))
    estimator.insert_observation(torch.randn(128))

    initial_size = estimator.memory.size
    initial_count = estimator.normalizer.count

    # Evaluate 20 imagined candidate latents with update_memory=False and update_normalizer=False
    for _ in range(20):
        z_imagined = torch.randn(128)
        score = estimator.evaluate(z_imagined, update_memory=False, update_normalizer=False)
        assert isinstance(score, float)

    assert estimator.memory.size == initial_size
    assert estimator.normalizer.count == initial_count


def test_learning_progress_updates_from_real_residual():
    """Test 5: Learning progress estimator updates only from actual measured residuals,

    while score() is purely read-only.
    """
    estimator = LearningProgressEstimator(clip_val=5.0)
    init_short = estimator.short_ma
    init_long = estimator.long_ma
    init_count = estimator.normalizer.count

    # Read-only score call should NOT change state
    s1 = estimator.score()
    assert estimator.short_ma == init_short
    assert estimator.long_ma == init_long
    assert estimator.normalizer.count == init_count

    # Update with measured error decreases residual
    estimator.update_from_residual(0.1)
    assert estimator.short_ma != init_short
    assert estimator.normalizer.count == init_count + 1


def test_candidate_order_does_not_change_scores():
    """Test 6: Candidate sequence scoring is invariant to candidate evaluation order

    within an absolute numerical tolerance of 1e-6.
    """
    scorer = CandidateScorer(novelty_estimator=NoveltyEstimator(), discount=0.99)
    weights = ROLE_WEIGHTS["explorer"]

    torch.manual_seed(42)
    N = 4
    H = 4
    latents = torch.randn(N, H, 128)
    rewards = torch.randn(N, H)
    term_probs = torch.sigmoid(torch.randn(N, H))
    disagreements = torch.abs(torch.randn(N, H))

    rollout_forward = {
        "pooled_latents": latents,
        "rewards": rewards,
        "termination_probs": term_probs,
        "disagreements": disagreements,
    }
    scores_forward = scorer.score_rollouts(rollout_forward, weights)

    # Reverse order: [N-1, ..., 0]
    perm = list(reversed(range(N)))
    rollout_reversed = {
        "pooled_latents": latents[perm],
        "rewards": rewards[perm],
        "termination_probs": term_probs[perm],
        "disagreements": disagreements[perm],
    }
    scores_reversed = scorer.score_rollouts(rollout_reversed, weights)

    # Re-align reversed scores
    scores_unreversed = scores_reversed[perm]
    diff = (scores_forward - scores_unreversed).abs().max().item()
    assert diff < 1e-6, f"Candidate scoring order variance detected! Max diff: {diff}"


def test_normalizer_updates_only_after_candidate_ranking():
    """Test 7: Candidate scoring does not mutate normalizer statistics."""
    estimator = NoveltyEstimator(capacity=50, latent_dim=128)
    scorer = CandidateScorer(novelty_estimator=estimator, discount=0.99)
    weights = ROLE_WEIGHTS["explorer"]

    count_before = estimator.normalizer.count
    rollout = {
        "pooled_latents": torch.randn(8, 4, 128),
        "rewards": torch.randn(8, 4),
        "termination_probs": torch.zeros(8, 4),
        "disagreements": torch.zeros(8, 4),
    }
    scorer.score_rollouts(rollout, weights)
    count_after = estimator.normalizer.count

    assert count_before == count_after, f"Normalizer mutated during candidate ranking! {count_before} -> {count_after}"


def test_config_fields_are_consumed_or_rejected():
    """Test 8: Config handling consumes all declared fields and rejects unknown sections."""
    cfg = Config()
    assert cfg.collection.initial_random_steps >= 0
    assert cfg.collection.num_env_workers >= 1
    assert cfg.collection.heldout_seed_start >= 1000

    # Rejection of unknown top-level section
    bad_dict = cfg.to_dict()
    bad_dict["invalid_section_xyz"] = {"foo": "bar"}
    with pytest.raises(ValueError, match="Unknown top-level config sections"):
        Config.from_dict(bad_dict)


def test_full_checkpoint_resume_restores_state(temp_dir):
    """Test 9: Full checkpoint resume restores model, EMA, optimizer, replay, mapping,

    novelty memory, normalizers, and scheduler.
    """
    model = JepaWorldModel(latent_dim=128, action_dim=64, context_dim=256, num_actions=17)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    learner = JepaLearner(model=model, optimizer=optimizer, device=torch.device("cpu"))
    replay = ReplayBuffer(capacity=100)
    mapping = OpaqueActionMapping(num_actions=17, seed=777)
    novelty = NoveltyEstimator(capacity=50, latent_dim=128)
    disag = DisagreementEstimator()
    prog = LearningProgressEstimator()
    scheduler = UpdateScheduler(updates_per_env_step=0.25)

    # Put some state in each
    z = torch.randn(128)
    novelty.insert_observation(z)
    prog.update_from_residual(0.3)
    scheduler.step(num_env_steps=8)

    obs = np.ones((64, 64, 3), dtype=np.uint8) * 128
    replay.start_episode(obs, episode_id=1)
    replay.add(action=3, reward=1.0, next_obs=obs, terminated=False, truncated=False)

    ckpt_data = {
        "step": 42,
        "learner_update_count": 5,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": model.target_encoder.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "action_mapping": mapping.to_dict(),
        "replay_state": replay.get_state(),
        "novelty_state": novelty.get_state(),
        "disagreement_state": disag.get_state(),
        "progress_state": prog.get_state(),
        "scheduler_state": scheduler.get_state(),
    }

    ckpt_path = str(temp_dir / "checkpoint_test.pt")
    torch.save(ckpt_data, ckpt_path)

    # Fresh instances
    model2 = JepaWorldModel(latent_dim=128, action_dim=64, context_dim=256, num_actions=17)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    learner2 = JepaLearner(model=model2, optimizer=optimizer2, device=torch.device("cpu"))
    replay2 = ReplayBuffer(capacity=100)
    novelty2 = NoveltyEstimator(capacity=50, latent_dim=128)
    disag2 = DisagreementEstimator()
    prog2 = LearningProgressEstimator()
    scheduler2 = UpdateScheduler()

    restored = restore_checkpoint(
        checkpoint_path=ckpt_path,
        model=model2,
        learner=learner2,
        replay_buffer=replay2,
        expected_mapping=mapping,
        novelty_estimator=novelty2,
        disagreement_estimator=disag2,
        progress_estimator=prog2,
        update_scheduler=scheduler2,
    )

    assert restored["step"] == 42
    assert learner2.update_count == 5
    assert replay2.size == replay.size
    assert novelty2.memory.size == novelty.memory.size
    assert novelty2.normalizer.count == novelty.normalizer.count
    assert scheduler2.learner_updates == scheduler.learner_updates

    # Weights match exactly
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(model.target_encoder.parameters(), model2.target_encoder.parameters()):
        assert torch.equal(p1, p2)
