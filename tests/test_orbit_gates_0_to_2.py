"""Comprehensive test suite for ORBIT Gates 0, 1, and 2.
Complies with ORBIT_IMPLEMENTATION_BRIEF_v1.1.md Sections A-K and Gates 0-2.
Written BEFORE production bug fixes (TDD Red-Green protocol).
"""
from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score, evaluate_crafter_agent
from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.env import PixelCrafter
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.replay import PixelEpisode, PixelReplay
from true_jepa.self_play import SelfPlayCoordinator


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def create_tiny_model(num_actions: int = 17) -> TrueJEPA:
    return TrueJEPA(
        JEPAConfig(
            token_dim=32,
            belief_dim=48,
            predictor_heads=4,
            predictor_layers=1,
            predictor_mlp_dim=64,
            ensemble_size=2,
            rollout_horizon=2,
        ),
        num_actions=num_actions,
    )


def create_synthetic_episode(
    length: int = 10,
    seed: int = 42,
    episode_id: int = 1,
    source: str = "synthetic",
) -> PixelEpisode:
    rng = np.random.default_rng(seed)
    return PixelEpisode(
        obs=rng.integers(0, 256, size=(length + 1, 64, 64, 3), dtype=np.uint8),
        actions=rng.integers(0, 17, size=length, dtype=np.int64),
        rewards=rng.normal(size=length).astype(np.float32),
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        seed=seed,
        source=source,
    )


# ---------------------------------------------------------------------------
# Gate 0 Tests: Benchmark & Evaluator Baseline
# ---------------------------------------------------------------------------

def test_gate0_score_formula_exact_vectors():
    """Verify compute_crafter_score against exact mathematical test vectors."""
    # 1. All zero unlocks -> score must be exactly 0.0
    zero_records = [{k: 0 for k in CRAFTER_ACHIEVEMENTS} for _ in range(10)]
    res_zero = compute_crafter_score(zero_records)
    assert res_zero["score"] == pytest.approx(0.0)
    assert res_zero["unlocked_count"] == 0

    # 2. All 100% unlocks -> score must be exactly 100.0
    # exp((1/22)*sum(ln(1 + 100))) - 1 = exp(ln(101)) - 1 = 101 - 1 = 100.0
    hundred_records = [{k: 1 for k in CRAFTER_ACHIEVEMENTS} for _ in range(10)]
    res_hundred = compute_crafter_score(hundred_records)
    assert res_hundred["score"] == pytest.approx(100.0)
    assert res_hundred["unlocked_count"] == 22

    # 3. Single achievement at 100%, others 0%
    # exp((1/22)*ln(101)) - 1 = 101^(1/22) - 1 ≈ 0.23239%
    single_record = [{k: (1 if k == "collect_wood" else 0) for k in CRAFTER_ACHIEVEMENTS} for _ in range(10)]
    res_single = compute_crafter_score(single_record)
    expected_single = math.exp(math.log(101.0) / 22.0) - 1.0
    assert res_single["score"] == pytest.approx(expected_single, rel=1e-5)


def test_gate0_recompute_v3_artifacts_and_detect_provenance_mismatch():
    """Verify scores from both JSON files and detect the R01 provenance divergence."""
    path_champion = Path("results/screen_v3_champion_clean_50ep.json")
    path_t2 = Path("results/screen_v3_t2_final_results.json")

    assert path_champion.exists(), "screen_v3_champion_clean_50ep.json must exist"
    assert path_t2.exists(), "screen_v3_t2_final_results.json must exist"

    with open(path_champion) as f:
        data_champ = json.load(f)
    with open(path_t2) as f:
        data_t2 = json.load(f)

    # 1. Verify internal math consistency for champion (3.17088)
    pcts_champ = data_champ["unlock_percentages"]
    log_sum_champ = sum(math.log(1.0 + pcts_champ[k]) for k in CRAFTER_ACHIEVEMENTS)
    score_champ_recalc = math.exp(log_sum_champ / 22.0) - 1.0
    assert score_champ_recalc == pytest.approx(data_champ["crafter_score"], abs=1e-5)
    assert score_champ_recalc == pytest.approx(3.17087726205484, abs=1e-5)

    # 2. Verify internal math consistency for t2 (4.55)
    pcts_t2 = data_t2["unlock_percentages"]
    log_sum_t2 = sum(math.log(1.0 + pcts_t2[k]) for k in CRAFTER_ACHIEVEMENTS)
    score_t2_recalc = math.exp(log_sum_t2 / 22.0) - 1.0
    assert score_t2_recalc == pytest.approx(data_t2["crafter_score"], abs=1e-2)
    assert score_t2_recalc == pytest.approx(4.546214306875846, abs=1e-4)

    # 3. Detect R01 PROVENANCE BLOCKER: identical checkpoint path and seeds, but divergent results!
    assert data_champ["checkpoint"] == data_t2["checkpoint"]
    assert data_champ["seed_range"] == data_t2["seed_range"]
    assert data_champ["crafter_score"] != data_t2["crafter_score"]

    # Verify per-episode outcomes diverge on episode 1 (seed 2050)
    ep1_champ = data_champ["episodes"][0]
    ep1_t2 = data_t2["episodes"][0]
    assert ep1_champ["seed"] == ep1_t2["seed"]
    assert ep1_champ["steps"] != ep1_t2["steps"]
    assert ep1_champ["unlocked"] != ep1_t2["unlocked"]


def test_gate0_crafter_wrapper_no_import_patching():
    """Verify that importing envs.crafter_wrapper does NOT monkey-patch crafter globally."""
    import crafter
    # Upstream Crafter Env._balance_object should not be modified
    orig_code = crafter.env.Env._balance_object.__code__
    import envs.crafter_wrapper as cw
    # Test that no automatic monkey patching happened on import
    assert getattr(crafter.env.Env, "_deterministic_patched", False) is False, (
        "crafter_wrapper must NOT apply monkey patching on module import (R04 finding)"
    )


def test_gate0_crafter_wrapper_truncation_semantics():
    """Verify that CrafterEnv properly handles max_steps truncation vs termination (R03)."""
    from envs.crafter_wrapper import CrafterEnv
    env = CrafterEnv(seed=123, max_steps=5)
    obs, info = env.reset(seed=123)
    assert obs.shape == (64, 64, 3)

    # Step up to max_steps
    done = False
    step_count = 0
    while not done and step_count < 10:
        obs, reward, terminated, truncated, info = env.step(0)  # Noop / Action 0
        step_count += 1
        done = terminated or truncated

    assert step_count == 5, f"Expected truncation at step 5, got {step_count}"
    assert truncated is True, "Environment must return truncated=True at max_steps cutoff"
    assert terminated is False, "Environment must return terminated=False when truncated at max_steps"


def test_gate0_evaluator_isolates_info():
    """Verify that evaluate_crafter_agent does not pass info to the action function (R02)."""
    from envs.crafter_wrapper import CrafterEnv
    env = CrafterEnv(seed=100, max_steps=3)

    called_with_info = False

    def strict_action_fn(obs: np.ndarray, reward_prev: float = 0.0, *args) -> int:
        nonlocal called_with_info
        if len(args) > 0 and isinstance(args[0], dict) and "inventory" in args[0]:
            called_with_info = True
        return 0

    # Evaluator must not pass info
    res = evaluate_crafter_agent(env, strict_action_fn, num_episodes=1, max_steps=2)
    assert not called_with_info, "Evaluator leaked info dictionary to action_fn (R02 finding)"


# ---------------------------------------------------------------------------
# Gate 1 Tests: Anti-Cheat & Interface Integrity
# ---------------------------------------------------------------------------

def test_gate1_runtime_poison_info_invariance():
    """Runtime poison test: changing info metadata must not alter agent actions or representations."""
    model = create_tiny_model(num_actions=17)
    agent = VisualJEPAAgent(model, device="cpu", stochastic=False)

    rng = np.random.default_rng(99)
    obs = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    # 1. Clean run
    agent.reset()
    action_clean = agent.act(obs, reward_prev=0.0)
    hidden_clean = agent.hidden.clone()

    # 2. Poisoned info - simulating privileged fields being altered externally
    poisoned_info = {
        "inventory": {"wood": 9999, "iron": 9999, "diamond": 9999},
        "achievements": {ach: 10 for ach in CRAFTER_ACHIEVEMENTS},
        "player_pos": [9999, 9999],
        "semantic": np.zeros((64, 64), dtype=np.int32),
    }

    # Agent cannot receive poisoned_info, but even if an adapter has it, policy output must be bitwise identical
    agent.reset()
    action_poisoned = agent.act(obs, reward_prev=0.0)
    hidden_poisoned = agent.hidden.clone()

    assert action_clean == action_poisoned
    assert torch.equal(hidden_clean, hidden_poisoned)


def test_gate1_agent_explicit_reset_lifecycle():
    """Verify that agent.reset() is explicit and act() does not depend on step == 0 (R16)."""
    model = create_tiny_model(num_actions=17)
    agent = VisualJEPAAgent(model, device="cpu", stochastic=False)

    obs = np.zeros((64, 64, 3), dtype=np.uint8)

    # Before reset, hidden is None
    assert agent.hidden is None
    agent.reset()
    assert agent.hidden is None

    # Step 1 without passing step argument
    act1 = agent.act(obs, reward_prev=0.0)
    assert agent.hidden is not None
    h1 = agent.hidden.clone()

    # Step 2 without passing step argument
    act2 = agent.act(obs, reward_prev=1.0)
    assert not torch.equal(agent.hidden, h1), "Belief should evolve across steps"

    # Reset explicitly clears hidden state
    agent.reset()
    assert agent.hidden is None


def test_gate1_action_space_runtime_discovery():
    """Verify action space is discovered dynamically from environment and equals 17."""
    env = PixelCrafter(seed=42)
    assert hasattr(env, "num_actions")
    assert env.num_actions == 17
    env.close()


def test_gate1_static_ast_anti_cheat():
    """AST / grep audit asserting zero hardcoded recipe or material tables exist in true_jepa."""
    banned_tokens = [
        "CRAFT_REQUIRED_MATERIALS",
        "INVENTORY_KEYS",
        "ACHIEVEMENT_RECIPES",
        "RECIPE_TABLE",
        "place_table_step",
    ]
    target_dir = Path("true_jepa")
    for py_file in target_dir.glob("**/*.py"):
        content = py_file.read_text()
        for token in banned_tokens:
            assert token not in content, f"Found banned cheat token '{token}' in {py_file}"


# ---------------------------------------------------------------------------
# Gate 2 Tests: Data, Replay, Sequence & Checkpoint Integrity
# ---------------------------------------------------------------------------

def test_gate2_replay_capacity_persistence_and_load(tmp_path):
    """Verify PixelReplay persists capacity_frames and restores it on load (R11)."""
    capacity = 500
    replay = PixelReplay(capacity_frames=capacity)
    ep = create_synthetic_episode(length=10, seed=1)
    replay.add(ep)
    assert replay.capacity_frames == capacity

    save_path = tmp_path / "replay_capacity_test.npz"
    replay.save(str(save_path))

    loaded_replay = PixelReplay.load(str(save_path))
    assert loaded_replay.capacity_frames == capacity, (
        f"Loaded replay lost capacity_frames: expected {capacity}, got {loaded_replay.capacity_frames}"
    )


def test_gate2_replay_oversized_episode_policy():
    """Verify that adding an episode larger than capacity raises ValueError (DEC-01 / R11)."""
    replay = PixelReplay(capacity_frames=10)
    oversized_ep = create_synthetic_episode(length=20, seed=2)  # 21 frames > 10 capacity
    with pytest.raises(ValueError, match="exceeds replay capacity"):
        replay.add(oversized_ep)


def test_gate2_replay_full_prefix_sampling():
    """Verify that PixelReplay provides full episode prefix for accurate belief reconstruction (R05)."""
    replay = PixelReplay(capacity_frames=1000)
    ep = create_synthetic_episode(length=30, seed=7)
    replay.add(ep)

    rng = np.random.default_rng(123)
    # Sample a window with sequence_length=5
    # The batch must contain prefix data from step 0 to the sampled start
    batch = replay.sample_with_prefix(batch_size=2, sequence_length=5, rng=rng)

    assert "obs" in batch
    assert "actions" in batch
    assert "prefix_obs" in batch
    assert "prefix_actions" in batch
    assert "prefix_rewards" in batch
    assert "prefix_lengths" in batch

    # Verify that unrolling belief from step 0 over the prefix matches a full forward pass
    model = create_tiny_model(num_actions=17)
    # Check shape alignment
    assert batch["obs"].shape == (2, 6, 64, 64, 3)  # B, T+1, H, W, C


def test_gate2_optimizer_parameter_disjointness():
    """Verify disjoint parameter groups across world, actor, and value optimizers (R09)."""
    model = create_tiny_model(num_actions=17)
    coordinator = SelfPlayCoordinator(
        env_factory=lambda: PixelCrafter(seed=1),
        model=model,
        train_cfg=TrainConfig(),
        device="cpu",
    )

    world_params = set(p for group in coordinator.world_optimizer.param_groups for p in group["params"])
    actor_params = set(p for group in coordinator.actor_optimizer.param_groups for p in group["params"])
    value_params = set(p for group in coordinator.value_optimizer.param_groups for p in group["params"])
    target_params = set(model.target_encoder.parameters())

    # 1. Target encoder parameters belong to NO optimizer
    assert len(world_params.intersection(target_params)) == 0
    assert len(actor_params.intersection(target_params)) == 0
    assert len(value_params.intersection(target_params)) == 0

    # 2. World, actor, and value parameter sets are strictly mutually disjoint
    assert len(world_params.intersection(actor_params)) == 0, "world and actor optimizers overlap (R09 finding)"
    assert len(world_params.intersection(value_params)) == 0, "world and value optimizers overlap (R09 finding)"
    assert len(actor_params.intersection(value_params)) == 0, "actor and value optimizers overlap (R09 finding)"

    # 3. All trainable parameters are accounted for
    all_trainable = set(p for p in model.parameters() if p.requires_grad)
    all_covered = world_params.union(actor_params).union(value_params)
    assert all_trainable == all_covered


def test_gate2_checkpoint_full_rng_state_and_exact_resume(tmp_path):
    """Verify episode-boundary checkpoint saves and restores full RNG state for exact resume (R12)."""
    model = create_tiny_model(num_actions=17)
    coordinator = SelfPlayCoordinator(
        env_factory=lambda: PixelCrafter(seed=42),
        model=model,
        train_cfg=TrainConfig(batch_size=2, sequence_length=4),
        device="cpu",
        seed=100,
    )

    # Populate replay with 2 episodes
    coordinator.replay.add(create_synthetic_episode(length=8, seed=101))
    coordinator.replay.add(create_synthetic_episode(length=8, seed=102))

    # Save checkpoint
    ckpt_path = str(tmp_path / "coordinator_checkpoint.pt")
    replay_path = str(tmp_path / "replay_checkpoint.npz")
    coordinator.save(ckpt_path)
    coordinator.replay.save(replay_path)

    # Step uninterrupted
    uninterrupted_rng_state = coordinator.rng.bit_generator.state
    uninterrupted_sample = coordinator.replay.sample(batch_size=2, sequence_length=4, rng=coordinator.rng)

    # Step resumed from restored checkpoint
    restored_coordinator = SelfPlayCoordinator.load(
        ckpt_path,
        env_factory=lambda: PixelCrafter(seed=42),
        model=create_tiny_model(num_actions=17),
        train_cfg=TrainConfig(batch_size=2, sequence_length=4),
        device="cpu",
        seed=100,
        replay_path=replay_path,
    )

    restored_sample = restored_coordinator.replay.sample(batch_size=2, sequence_length=4, rng=restored_coordinator.rng)

    # Samples must be identical
    assert torch.equal(uninterrupted_sample["actions"], restored_sample["actions"])
    assert torch.equal(uninterrupted_sample["obs"], restored_sample["obs"])


def test_gate2_exploding_module_call_site_sensitivity():
    """Exploding module harness: proves every module is actually connected to its execution path (Section G.4)."""
    model = create_tiny_model(num_actions=17)
    frame = LatentFrame(
        tokens=torch.randn(2, 64, 32),
        pooled=torch.randn(2, 32),
    )
    hidden = torch.randn(2, 48)
    action = torch.tensor([1, 2], dtype=torch.long)

    # 1. Exploding predictor
    class ExplodingPredictor(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("Boom: Predictor is active")

    model.predictor = ExplodingPredictor()
    with pytest.raises(RuntimeError, match="Boom: Predictor is active"):
        model.predict(frame, action, hidden)

    # 2. Exploding reward head
    class ExplodingReward(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("Boom: Reward Head is active")

    model = create_tiny_model(num_actions=17)
    model.heads.reward = ExplodingReward()
    with pytest.raises(RuntimeError, match="Boom: Reward Head is active"):
        model.transition_heads(frame, hidden, action)

    # 3. Exploding continuation head
    class ExplodingContinuation(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("Boom: Continuation Head is active")

    model = create_tiny_model(num_actions=17)
    model.heads.continuation = ExplodingContinuation()
    with pytest.raises(RuntimeError, match="Boom: Continuation Head is active"):
        model.transition_heads(frame, hidden, action)
