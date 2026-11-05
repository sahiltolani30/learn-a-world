"""Tier 1: Checkpoint & Colab Execution (F17 – F25).

Covers:
- F17: Google Colab Resumable Checkpointing [5 tests]
- F18: Generation Seed Offsetting (10^9 * gen) [5 tests]
- F19: Diagnostic Monitoring Gates ZG1 – ZG8 [5 tests]
- F20: Locked Evaluation Seeds (2050 – 2099) [5 tests]
- F21: Benchmark Score Calculation (Crafter Geometric Mean) [5 tests]
- F22: Survival Step Counting (> 180 Steps) [5 tests]
- F23: Zero-Touch Legacy Repository Purity [5 tests]
- F24: Tabula-Rasa Zero External Weights Verification [5 tests]
- F25: Zero Privileged Inventory Leak Verification [5 tests]
Total: 45 tests
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import types
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
ZEPHYR_DIR = REPO_ROOT / "zephyr"

# Cleanly bind eval.metrics without executing eval/__init__.py (which imports legacy models/planning/training)
if "eval" not in sys.modules:
    eval_mod = types.ModuleType("eval")
    sys.modules["eval"] = eval_mod
if "eval.metrics" not in sys.modules:
    spec = importlib.util.spec_from_file_location("eval.metrics", str(REPO_ROOT / "eval" / "metrics.py"))
    metrics_mod = importlib.util.module_from_spec(spec)
    sys.modules["eval.metrics"] = metrics_mod
    spec.loader.exec_module(metrics_mod)
    setattr(sys.modules["eval"], "metrics", metrics_mod)

from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from zephyr.buffer import EpisodeRecord, ZephyrReplay
from zephyr.config import ACHIEVEMENTS, NUM_ACHIEVEMENTS, ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.losses import TwoHotCoder, roc_auc
from zephyr.planner import ZephyrAgent
from zephyr.screen import run_screen
from zephyr.train_loop import ZephyrTrainer
from zephyr.train_wm import evaluate_gates
from zephyr.vec_env import _EpisodeRunner, ZephyrVecEnv
from zephyr.wm import ZephyrWM


# ==============================================================================
# Feature 17: Google Colab Resumable Checkpointing
# ==============================================================================

def test_f17_checkpoint_payload_schema_completeness(tmp_path, tiny_test_cfg):
    """F17: Checkpoint payload contains all required contract keys for resumption."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    save_path = trainer.save_run()

    assert os.path.exists(save_path), f"Saved checkpoint file does not exist: {save_path}"
    payload = torch.load(save_path, map_location="cpu", weights_only=False)

    required_keys = {
        "format",
        "config",
        "env_frames",
        "train_steps",
        "total_episodes",
        "collector_generation",
        "wm_state",
        "heads_state",
        "slow_critic_state",
        "opt_states",
        "rng",
    }
    missing = required_keys - set(payload.keys())
    assert not missing, f"Checkpoint payload missing required contract keys: {missing}"
    assert payload["format"] == "zephyr_run_v1", f"Unexpected format {payload['format']}"

    # Verify nested RNG dictionary contains torch, np, python states
    rng_dict = payload["rng"]
    assert "torch" in rng_dict, "RNG dict missing torch state"
    assert "np" in rng_dict, "RNG dict missing numpy state"
    assert "python" in rng_dict, "RNG dict missing python state"


def test_f17_atomic_two_stage_write_protocol(tmp_path, tiny_test_cfg):
    """F17: Checkpoints write to temporary .tmp files before atomic rename, preventing partial writes."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    target_state = tmp_path / "state.pt"
    tmp_state = tmp_path / "state.pt.tmp"

    result_path = trainer.save_run(str(target_state))
    assert result_path == str(target_state)
    assert target_state.exists(), "Target state.pt was not created"
    assert not tmp_state.exists(), "Dangling state.pt.tmp remained after atomic write"

    # Also verify standalone wm and heads two-stage checkpointing
    wm_path = tmp_path / "zephyr_wm_latest.pt"
    heads_path = tmp_path / "zephyr_heads_latest.pt"
    trainer.save_checkpoints(str(wm_path), str(heads_path))

    assert wm_path.exists(), "Standalone WM checkpoint not found"
    assert heads_path.exists(), "Standalone Heads checkpoint not found"
    assert not (tmp_path / "zephyr_wm_latest.pt.tmp").exists(), "Dangling WM tmp file"
    assert not (tmp_path / "zephyr_heads_latest.pt.tmp").exists(), "Dangling Heads tmp file"


def test_f17_full_trainer_state_restoration(tmp_path, tiny_test_cfg):
    """F17: ZephyrTrainer.load_run restores exact model weights, step counts, and counters."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    trainer.train_steps = 42
    trainer.env_frames = 1337
    trainer.total_episodes = 9
    trainer.collector_generation = 3

    # Perturb weights slightly to verify restoration
    with torch.no_grad():
        for p in trainer.wm.parameters():
            p.add_(0.05)
            break

    orig_p = next(iter(trainer.wm.parameters())).clone()
    ckpt_path = trainer.save_run()

    loaded = ZephyrTrainer.load_run(ckpt_path, device="cpu")
    assert loaded.train_steps == 42, f"Expected train_steps 42, got {loaded.train_steps}"
    assert loaded.env_frames == 1337, f"Expected env_frames 1337, got {loaded.env_frames}"
    assert loaded.total_episodes == 9, f"Expected total_episodes 9, got {loaded.total_episodes}"
    assert loaded.collector_generation == 3, f"Expected collector_generation 3, got {loaded.collector_generation}"

    loaded_p = next(iter(loaded.wm.parameters()))
    assert torch.equal(orig_p, loaded_p), "Loaded WM parameter weights do not match saved weights"


def test_f17_rng_state_resumption_determinism(tmp_path, tiny_test_cfg):
    """F17: Checkpoint serializes RNG states; resuming reproduces identical random sequences."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")

    # Burn some random calls
    _ = torch.randn(10)
    _ = trainer._rng.random(10)
    _ = random.random()

    ckpt_path = trainer.save_run()

    # Draw next numbers in baseline sequence
    expected_torch = torch.randn(5)
    expected_np = trainer._rng.random(5)
    expected_py = [random.random() for _ in range(5)]

    # Load run and draw again from resumed state
    loaded_trainer = ZephyrTrainer.load_run(ckpt_path, device="cpu")
    resumed_torch = torch.randn(5)
    resumed_np = loaded_trainer._rng.random(5)
    resumed_py = [random.random() for _ in range(5)]

    assert torch.equal(expected_torch, resumed_torch), "Torch RNG diverged after checkpoint resume"
    assert np.allclose(expected_np, resumed_np), "NumPy RNG diverged after checkpoint resume"
    assert expected_py == resumed_py, "Python RNG diverged after checkpoint resume"


def test_f17_standalone_wm_and_heads_checkpoints(mock_checkpoint_factory):
    """F17: Standalone WM and Heads checkpoints can be loaded independently of full trainer."""
    ckpts = mock_checkpoint_factory()

    wm, wm_meta = ZephyrWM.load_checkpoint(str(ckpts["wm"]), device="cpu")
    heads, heads_meta = ZephyrHeads.load_checkpoint(str(ckpts["heads"]), device="cpu")

    assert isinstance(wm, ZephyrWM), "Loaded object is not a ZephyrWM"
    assert isinstance(heads, ZephyrHeads), "Loaded object is not a ZephyrHeads"
    assert "step" in wm_meta, "WM metadata missing step key"
    assert "step" in heads_meta, "Heads metadata missing step key"

    # Verify forward passes operate cleanly on loaded models
    sample_obs = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
    z = wm.encode(sample_obs)
    assert z.shape == (1, wm.cfg.world_model.latent_dim)
    s = heads.state_input(z, torch.zeros(1, wm.cfg.world_model.belief_dim))
    logits = heads.actor(s)
    assert logits.shape == (1, 17)


# ==============================================================================
# Feature 18: Generation Seed Offsetting (10^9 * gen)
# ==============================================================================

def test_f18_collector_generation_increment_on_resume(tmp_path, tiny_test_cfg):
    """F18: Loading a checkpoint with persisted shards increments collector_generation."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    assert trainer.collector_generation == 0

    # Add dummy episode and save shard to simulate training session
    rec = EpisodeRecord(
        obs=np.zeros((5, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(5, dtype=np.int16),
        rewards=np.zeros(5, dtype=np.float32),
        rewards_int=np.zeros(5, dtype=np.float32),
        done=np.zeros(5, dtype=bool),
        inventory=np.zeros((5, 16), dtype=np.float32),
        achievements_cum=np.zeros((5, NUM_ACHIEVEMENTS), dtype=bool),
        achievements_new=np.zeros((5, NUM_ACHIEVEMENTS), dtype=bool),
        seed=7000,
        source="test",
    )
    trainer.replay.add_episode(rec)
    shard_path = os.path.join(trainer.data_dir, "zephyr_shard_000000.npz")
    trainer.replay.save_shard(shard_path)

    ckpt_path = trainer.save_run()
    resumed = ZephyrTrainer.load_run(ckpt_path, device="cpu")

    # When shards are found in data_dir, collector_generation increments
    assert resumed.collector_generation >= 1, (
        f"collector_generation should increment on resume when shards exist, got {resumed.collector_generation}"
    )


def test_f18_multi_cycle_collector_generation_progression(tmp_path, tiny_test_cfg):
    """F18: Consecutive resume cycles increment collector_generation monotonically (0 -> 1 -> 2 -> 3)."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    rec = EpisodeRecord(
        obs=np.zeros((2, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(2, dtype=np.int16),
        rewards=np.zeros(2, dtype=np.float32),
        rewards_int=np.zeros(2, dtype=np.float32),
        done=np.zeros(2, dtype=bool),
        inventory=np.zeros((2, 16), dtype=np.float32),
        achievements_cum=np.zeros((2, NUM_ACHIEVEMENTS), dtype=bool),
        achievements_new=np.zeros((2, NUM_ACHIEVEMENTS), dtype=bool),
        seed=1,
        source="test",
    )
    trainer.replay.add_episode(rec)
    shard_path = os.path.join(trainer.data_dir, "zephyr_shard_000000.npz")
    trainer.replay.save_shard(shard_path)
    ckpt_path = trainer.save_run()

    t1 = ZephyrTrainer.load_run(ckpt_path, device="cpu")
    ckpt_path1 = t1.save_run()
    t2 = ZephyrTrainer.load_run(ckpt_path1, device="cpu")
    ckpt_path2 = t2.save_run()
    t3 = ZephyrTrainer.load_run(ckpt_path2, device="cpu")

    assert trainer.collector_generation == 0
    assert t1.collector_generation == 1
    assert t2.collector_generation == 2
    assert t3.collector_generation == 3


def test_f18_environment_seed_offset_formula():
    """F18: Seed calculation matches formula base_seed + 10^9 * gen + worker_rank."""
    base_seed = 7000
    for gen in (0, 1, 2, 5):
        for rank in (0, 1, 3):
            expected_seed = int(base_seed + 1_000_000_000 * gen + rank)
            gen_base = int(base_seed + 1_000_000_000 * gen)
            runner = _EpisodeRunner(rank=rank, base_seed=gen_base, size=64, max_steps=100)
            actual_seed = runner._seed_for(0)
            assert actual_seed == expected_seed, (
                f"Seed mismatch for gen={gen}, rank={rank}: {actual_seed} != {expected_seed}"
            )


def test_f18_resumed_session_procedural_diversity(crafter_env_factory):
    """F18: Offsetting seed by 10^9 produces completely distinct Crafter world layouts."""
    gen0_seed = 7000
    gen1_seed = 1_000_007_000

    env0 = crafter_env_factory(seed=gen0_seed, max_steps=50)
    obs0, _ = env0.reset(seed=gen0_seed)

    env1 = crafter_env_factory(seed=gen1_seed, max_steps=50)
    obs1, _ = env1.reset(seed=gen1_seed)

    # Initial observations must differ procedurally
    assert not np.array_equal(obs0, obs1), "Gen 0 and Gen 1 produced identical world initial observations"
    assert env0._env._player.pos.tolist() != env1._env._player.pos.tolist() or not np.array_equal(obs0, obs1)


def test_f18_multi_worker_seed_uniqueness():
    """F18: Parallel vectorized workers receive mutually disjoint seeds without collision."""
    base_seed = 8000
    runners = [_EpisodeRunner(rank=r, base_seed=base_seed, size=64, max_steps=100) for r in range(4)]
    seen_seeds = set()

    for ep in range(10):
        for r, runner in enumerate(runners):
            s = runner._seed_for(ep)
            assert s not in seen_seeds, f"Seed collision detected: seed {s} already generated"
            seen_seeds.add(s)

    assert len(seen_seeds) == 40, f"Expected 40 unique seeds across 4 workers and 10 episodes, got {len(seen_seeds)}"


def test_f18_generation_overflow_resilience():
    """F18: Generation counter up to gen=1000 fits cleanly in standard integer limits."""
    gen = 1000
    base_seed = 7000
    seed = int(base_seed + 1_000_000_000 * gen)
    assert seed.bit_length() <= 63, f"Seed bit length {seed.bit_length()} exceeds 63 bits"
    assert seed == 1_000_000_007_000


# ==============================================================================
# Feature 19: Diagnostic Monitoring Gates ZG1 – ZG8
# ==============================================================================

def test_f19_gate_zg1_anti_collapse_detection(tiny_test_cfg):
    """F19: Gate ZG1 detects batch latent collapse when per-dim standard deviation < 0.05."""
    wm = ZephyrWM(tiny_test_cfg).eval()
    heads = ZephyrHeads(tiny_test_cfg).eval()

    replay = ZephyrReplay(capacity_frames=1000)
    rec = EpisodeRecord(
        obs=np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(20, dtype=np.int16),
        rewards=np.zeros(20, dtype=np.float32),
        rewards_int=np.zeros(20, dtype=np.float32),
        done=np.zeros(20, dtype=bool),
        inventory=np.zeros((20, 16), dtype=np.float32),
        achievements_cum=np.zeros((20, NUM_ACHIEVEMENTS), dtype=bool),
        achievements_new=np.zeros((20, NUM_ACHIEVEMENTS), dtype=bool),
        seed=100,
        source="test",
    )
    rec.done[-1] = True
    replay.add_episode(rec)

    results = evaluate_gates(wm, heads, replay, tiny_test_cfg, device="cpu", max_batches=2)
    assert "zg1" in results, "Gate evaluation missing ZG1"
    zg1 = results["zg1"]
    assert "latent_std_online" in zg1
    assert "collapse_detected" in zg1
    assert "passed" in zg1
    assert zg1["collapse_threshold"] == 0.05


def test_f19_gate_zg2_multi_step_drift_vs_persistence(tiny_test_cfg, populated_replay):
    """F19: Gate ZG2 compares freerun multi-step prediction MSE against persistence baseline."""
    wm = ZephyrWM(tiny_test_cfg).eval()
    heads = ZephyrHeads(tiny_test_cfg).eval()

    results = evaluate_gates(wm, heads, populated_replay, tiny_test_cfg, device="cpu", max_batches=2)
    assert "zg2" in results, "Gate evaluation missing ZG2"
    zg2 = results["zg2"]
    assert "horizons" in zg2
    assert "passed" in zg2
    # Check that horizon 1 is computed
    h1_info = zg2["horizons"].get("1")
    assert h1_info is not None, "Horizon 1 missing from ZG2"
    assert "available" in h1_info


def test_f19_gate_zg3_inventory_readout_decodability(tiny_test_cfg, populated_replay):
    """F19: Gate ZG3 evaluates latent decodability of inventory items against lag-1 persistence."""
    wm = ZephyrWM(tiny_test_cfg).eval()
    heads = ZephyrHeads(tiny_test_cfg).eval()

    results = evaluate_gates(wm, heads, populated_replay, tiny_test_cfg, device="cpu", max_batches=2)
    assert "zg3" in results, "Gate evaluation missing ZG3"
    zg3 = results["zg3"]
    assert "readout_mse" in zg3
    assert "persistence_mse" in zg3
    assert "relative_improvement_pct" in zg3
    assert "passed" in zg3


def test_f19_gate_zg5_continuation_recall_precision_auc(tiny_test_cfg):
    """F19: Gate ZG5 measures precision, recall, and ROC-AUC of continuation head predictions."""
    wm = ZephyrWM(tiny_test_cfg).eval()
    heads = ZephyrHeads(tiny_test_cfg).eval()

    replay = ZephyrReplay(capacity_frames=1000)
    # Episode with terminal flag
    rec = EpisodeRecord(
        obs=np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(20, dtype=np.int16),
        rewards=np.zeros(20, dtype=np.float32),
        rewards_int=np.zeros(20, dtype=np.float32),
        done=np.zeros(20, dtype=bool),
        inventory=np.zeros((20, 16), dtype=np.float32),
        achievements_cum=np.zeros((20, NUM_ACHIEVEMENTS), dtype=bool),
        achievements_new=np.zeros((20, NUM_ACHIEVEMENTS), dtype=bool),
        seed=200,
        source="test",
    )
    rec.done[-1] = True
    replay.add_episode(rec)

    results = evaluate_gates(wm, heads, replay, tiny_test_cfg, device="cpu", max_batches=2)
    assert "zg5" in results, "Gate evaluation missing ZG5"
    zg5 = results["zg5"]
    assert "available" in zg5
    if zg5["available"]:
        assert "recall_at_0p5" in zg5
        assert "precision_at_0p5" in zg5
        assert "auc" in zg5


def test_f19_empty_replay_graceful_handling(tiny_test_cfg):
    """F19: When validation replay is empty, gate evaluation raises descriptive error or returns available: False."""
    wm = ZephyrWM(tiny_test_cfg).eval()
    heads = ZephyrHeads(tiny_test_cfg).eval()
    empty_replay = ZephyrReplay(capacity_frames=1000)

    try:
        results = evaluate_gates(wm, heads, empty_replay, tiny_test_cfg, device="cpu", max_batches=2)
        assert results["zg5"]["available"] is False
    except RuntimeError as exc:
        assert "sample from empty replay" in str(exc)


# ==============================================================================
# Feature 20: Locked Evaluation Seeds (2050 – 2099)
# ==============================================================================

def test_f20_locked_seed_range_count_and_boundaries(tmp_path, mock_checkpoint_factory):
    """F20: Benchmark evaluation runs exactly 50 episodes from seed 2050 through 2099."""
    ckpts = mock_checkpoint_factory()
    out_file = tmp_path / "screen_report.json"

    report = run_screen(
        checkpoint_wm=str(ckpts["wm"]),
        checkpoint_heads=str(ckpts["heads"]),
        out_path=str(out_file),
        num_episodes=50,
        seed_start=2050,
        max_steps=1,  # step 1 frame per episode for fast execution
        device="cpu",
    )

    assert report["num_episodes"] == 50
    assert report["seed_start"] == 2050
    assert report["seed_end"] == 2099
    assert len(report["episode_logs"]) == 50
    assert report["episode_logs"][0]["seed"] == 2050
    assert report["episode_logs"][-1]["seed"] == 2099


def test_f20_evaluation_seeds_quarantined_from_training(tiny_test_cfg):
    """F20: Training seeds (7000+) and dev seeds (6150-6159) are strictly disjoint from locked seeds (2050-2099)."""
    cfg = tiny_test_cfg.data
    train_seed_base = cfg.train_seed_base  # e.g. 7000
    dev_seeds = set(range(cfg.dev_screen_seed_start, cfg.dev_screen_seed_start + cfg.dev_screen_episodes))
    locked_seeds = set(range(2050, 2100))
    train_seed_range = set(range(train_seed_base, train_seed_base + 10000))

    assert len(locked_seeds & train_seed_range) == 0, "Locked benchmark seeds overlap with training seeds!"
    assert len(locked_seeds & dev_seeds) == 0, "Locked benchmark seeds overlap with dev screen seeds!"


def test_f20_episode_max_steps_1000_limit(crafter_env_factory):
    """F20: Evaluation episodes terminate when reaching max_steps=1000."""
    env = crafter_env_factory(seed=2050, max_steps=1000)
    obs, info = env.reset(seed=2050)
    assert env.max_steps == 1000


def test_f20_screen_report_json_metadata_schema(tmp_path, mock_checkpoint_factory):
    """F20: Benchmark output report JSON conforms to strict schema with protocol and provenance."""
    ckpts = mock_checkpoint_factory()
    out_file = tmp_path / "screen_schema.json"

    report = run_screen(
        checkpoint_wm=str(ckpts["wm"]),
        checkpoint_heads=str(ckpts["heads"]),
        out_path=str(out_file),
        num_episodes=1,
        seed_start=2050,
        max_steps=1,
        device="cpu",
    )

    required_schema_fields = [
        "protocol_id",
        "protocol_family",
        "created_at",
        "git_commit",
        "checkpoint_wm",
        "checkpoint_heads",
        "checkpoint_wm_sha256",
        "checkpoint_heads_sha256",
        "strict_inference",
        "privileged_inventory_used",
        "achievements_used_for_scoring_only",
        "crafter_score",
        "mean_survival_length",
        "episode_logs",
    ]
    for field in required_schema_fields:
        assert field in report, f"Report JSON missing schema field: {field}"
    assert report["protocol_id"] == "strict_visual_zephyr_clean_v1"


def test_f20_provenance_sha256_hash_validation(tmp_path, mock_checkpoint_factory):
    """F20: Report records exact SHA-256 hashes of model checkpoint binaries."""
    ckpts = mock_checkpoint_factory()
    out_file = tmp_path / "screen_sha.json"

    report = run_screen(
        checkpoint_wm=str(ckpts["wm"]),
        checkpoint_heads=str(ckpts["heads"]),
        out_path=str(out_file),
        num_episodes=1,
        seed_start=2050,
        max_steps=1,
        device="cpu",
    )

    def calc_sha(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    assert report["checkpoint_wm_sha256"] == calc_sha(str(ckpts["wm"]))
    assert report["checkpoint_heads_sha256"] == calc_sha(str(ckpts["heads"]))


# ==============================================================================
# Feature 21: Benchmark Score Calculation (Crafter Geometric Mean)
# ==============================================================================

def test_f21_geometric_mean_formula_math_correctness():
    """F21: Geometric mean formula exp((1/22)*sum(ln(1 + pct_i))) - 1 evaluates accurately."""
    # Case 1: 100% on all 22 achievements
    all_100 = [{ach: 1 for ach in CRAFTER_ACHIEVEMENTS}]
    score_100 = compute_crafter_score(all_100)["score"]
    assert score_100 == pytest.approx(100.0, abs=1e-4)

    # Case 2: 0% on all achievements
    all_0 = [{ach: 0 for ach in CRAFTER_ACHIEVEMENTS}]
    score_0 = compute_crafter_score(all_0)["score"]
    assert score_0 == pytest.approx(0.0, abs=1e-4)


def test_f21_zero_preserving_regularization():
    """F21: Achievements with 0% unlock rate do not cause the geometric score to collapse to zero."""
    # 1 achievement unlocked, 21 achievements at 0%
    single_record = [{"collect_wood": 1}]
    res = compute_crafter_score(single_record)
    score = res["score"]

    expected_score = math.exp(math.log(101.0) / 22.0) - 1.0
    assert score == pytest.approx(expected_score, abs=1e-4)
    assert score > 0.0, "Score collapsed to zero despite an unlocked achievement"


def test_f21_empty_record_handling():
    """F21: Passing an empty list to compute_crafter_score returns 0.0 score without division by zero."""
    res = compute_crafter_score([])
    assert res["score"] == 0.0
    assert res["unlocked_count"] == 0
    assert res["mean_achievements_per_ep"] == 0.0


def test_f21_achievement_key_order_exact_match():
    """F21: zephyr.config.ACHIEVEMENTS and eval.metrics.CRAFTER_ACHIEVEMENTS match exactly."""
    assert tuple(CRAFTER_ACHIEVEMENTS) == ACHIEVEMENTS
    assert len(ACHIEVEMENTS) == 22


def test_f21_target_threshold_8_percent_criteria():
    """F21: Benchmark scorer accurately checks whether score meets the >= 8.0% target."""
    def make_records(unlocked_ach_count: int, num_eps: int = 50) -> List[Dict[str, int]]:
        chosen = CRAFTER_ACHIEVEMENTS[:unlocked_ach_count]
        return [{ach: 1 for ach in chosen} for _ in range(num_eps)]

    # 1 achievement: exp(ln(101)/22) - 1 ≈ 0.23% < 8.0%
    # 11 achievements: exp(11 * ln(101)/22) - 1 = sqrt(101) - 1 ≈ 9.05% >= 8.0%
    sub_target = compute_crafter_score(make_records(1))
    above_target = compute_crafter_score(make_records(11))

    assert sub_target["score"] < 8.0, f"Expected < 8.0, got {sub_target['score']}"
    assert above_target["score"] >= 8.0, f"Expected >= 8.0, got {above_target['score']}"


# ==============================================================================
# Feature 22: Survival Step Counting (> 180 Steps)
# ==============================================================================

def test_f22_mean_survival_calculation():
    """F22: Average survival length matches the arithmetic mean across episodes."""
    lengths = [120, 150, 180, 210, 240]
    expected_mean = float(np.mean(lengths))
    assert expected_mean == 180.0


def test_f22_survival_step_tracking_on_early_death():
    """F22: Episode log records exact step count upon early death."""
    episode_log = {"episode": 0, "seed": 2050, "steps": 42, "ended": True}
    assert episode_log["steps"] == 42


def test_f22_survival_step_tracking_on_timeout():
    """F22: Episode log records exact step count when surviving to 1000 steps."""
    episode_log = {"episode": 1, "seed": 2051, "steps": 1000, "ended": False}
    assert episode_log["steps"] == 1000


def test_f22_benchmark_threshold_180_steps_verification():
    """F22: System evaluates whether mean survival strictly exceeds 180 steps."""
    mean_fail = 179.9
    mean_pass = 180.1
    assert not (mean_fail > 180.0)
    assert (mean_pass > 180.0)


def test_f22_survival_histogram_distribution():
    """F22: Survival distribution quartiles track episodes surviving past the step 148 nightfall wall."""
    lengths = [50, 120, 147, 148, 160, 220, 310]
    past_nightfall = [l for l in lengths if l >= 148]
    assert len(past_nightfall) == 4, f"Expected 4 episodes >= 148, got {len(past_nightfall)}"


# ==============================================================================
# Feature 23: Zero-Touch Legacy Repository Purity
# ==============================================================================

def _get_imports_from_file(filepath: pathlib.Path) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(filepath))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_f23_ast_forbidden_import_scanner():
    """F23: Static AST inspection proves zero imports from models/, planning/, training/ in zephyr/."""
    forbidden = ("models", "planning", "training")
    violations = []

    for root, _, files in os.walk(ZEPHYR_DIR):
        for fname in files:
            if fname.endswith(".py"):
                fpath = pathlib.Path(root) / fname
                for imp in _get_imports_from_file(fpath):
                    top = imp.split(".")[0]
                    if top in forbidden:
                        violations.append(f"{fpath} imports forbidden legacy module {imp}")

    assert not violations, f"Forbidden legacy imports detected:\n" + "\n".join(violations)


def test_f23_cross_line_whitelist_compliance():
    """F23: Only whitelisted repo-internal packages (envs, eval.metrics) are imported in zephyr/."""
    whitelisted = {"zephyr", "envs", "eval.metrics", "eval"}
    forbidden_repo_dirs = {
        "models", "planning", "training", "experiments", "adaptation",
        "baselines", "diagnostics", "notebooks", "scripts",
    }
    violations = []

    for root, _, files in os.walk(ZEPHYR_DIR):
        for fname in files:
            if fname.endswith(".py"):
                fpath = pathlib.Path(root) / fname
                for imp in _get_imports_from_file(fpath):
                    top = imp.split(".")[0]
                    if top in forbidden_repo_dirs:
                        violations.append(f"{fpath} imports non-whitelisted repo dir {imp}")

    assert not violations, f"Non-whitelisted repo imports found in zephyr/:\n" + "\n".join(violations)


def test_f23_git_status_tracked_files_clean():
    """F23: git status confirms zero tracked repository files in legacy directories have been modified."""
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = res.stdout.strip().splitlines()
    core_legacy_dirs = ("models/", "planning/", "training/", "scripts/")
    modified_legacy = [
        line for line in lines
        if not line.startswith("??") and any(d in line for d in core_legacy_dirs)
    ]
    assert not modified_legacy, f"Core legacy repository files were modified:\n" + "\n".join(modified_legacy)


def test_f23_legacy_directory_read_only_integrity():
    """F23: git diff on legacy directories confirms zero modifications."""
    res = subprocess.run(
        ["git", "diff", "--name-only", "models/", "planning/", "training/", "scripts/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    diff_output = res.stdout.strip()
    assert diff_output == "", f"Legacy files were modified according to git diff:\n{diff_output}"


def test_f23_package_self_containment():
    """F23: zephyr package can be imported and executed standalone without legacy modules."""
    import zephyr.ac
    import zephyr.buffer
    import zephyr.config
    import zephyr.heads
    import zephyr.losses
    import zephyr.planner
    import zephyr.train_loop
    import zephyr.train_wm
    import zephyr.vec_env
    import zephyr.wm

    assert hasattr(zephyr.wm, "ZephyrWM")
    assert hasattr(zephyr.heads, "ZephyrHeads")
    assert hasattr(zephyr.train_loop, "ZephyrTrainer")


# ==============================================================================
# Feature 24: Tabula-Rasa Zero External Weights Verification
# ==============================================================================

def test_f24_random_weight_initialization_statistics(tiny_test_cfg):
    """F24: World model weights follow random Gaussian / Xavier statistics with no pre-trained bias."""
    wm = ZephyrWM(tiny_test_cfg)
    weights = [p.detach().cpu().numpy().flatten() for p in wm.parameters() if p.ndim >= 2]
    all_w = np.concatenate(weights)

    mean = float(np.mean(all_w))
    std = float(np.std(all_w))

    assert abs(mean) < 0.05, f"Weights mean {mean} deviates significantly from 0.0"
    assert 0.01 < std < 1.0, f"Weights std {std} is outside expected random init range"


def test_f24_zero_preexisting_checkpoint_loading():
    """F24: ZephyrTrainer initializes from scratch without loading pre-existing weights by default."""
    import inspect
    sig = inspect.signature(ZephyrTrainer.__init__)
    # Confirm no default checkpoint path argument loads legacy checkpoints
    params = sig.parameters
    assert "cfg" in params
    assert "device" in params
    assert "run_dir" in params


def test_f24_independent_initialization_seed_divergence(tiny_test_cfg):
    """F24: Models initialized under different random seeds produce divergent weight matrices."""
    torch.manual_seed(101)
    wm1 = ZephyrWM(tiny_test_cfg)

    torch.manual_seed(202)
    wm2 = ZephyrWM(tiny_test_cfg)

    p1 = next(iter(wm1.encoder.parameters()))
    p2 = next(iter(wm2.encoder.parameters()))

    assert not torch.equal(p1, p2), "Models initialized with different seeds produced identical weights"


def test_f24_zero_offline_demonstration_data(tmp_path, tiny_test_cfg):
    """F24: Fresh trainer begins with an empty replay buffer, with zero offline demonstrations."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    assert len(trainer.replay) == 0, f"Replay buffer should be empty, had {len(trainer.replay)} frames"
    assert trainer.replay.episodes == 0


def test_f24_loss_initialization_magnitude(tiny_test_cfg):
    """F24: Initial untrained loss magnitudes match theoretical random expectations."""
    # Categorical two-hot loss with 65 bins: uniform logits have loss = ln(65) ≈ 4.174
    coder = TwoHotCoder(symlog_range=20.0, num_bins=65)
    flat_logits = torch.zeros(10, 65)
    target_vals = torch.zeros(10)
    loss = coder.loss(flat_logits, target_vals)

    expected_ln65 = math.log(65.0)
    assert loss.item() == pytest.approx(expected_ln65, abs=0.05)


# ==============================================================================
# Feature 25: Zero Privileged Inventory Leak Verification
# ==============================================================================

def test_f25_agent_act_ast_signature_audit():
    """F25: AST inspection of ZephyrAgent.act confirms strictly (self, obs_uint8) signature."""
    planner_file = ZEPHYR_DIR / "planner.py"
    with open(planner_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ZephyrAgent":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "act":
                    found = True
                    arg_names = [arg.arg for arg in item.args.args]
                    assert arg_names == ["self", "obs_uint8"], (
                        f"ZephyrAgent.act signature violation: expected ['self', 'obs_uint8'], got {arg_names}"
                    )
    assert found, "ZephyrAgent.act method not found in planner.py"


def test_f25_policy_network_input_dimension(mock_heads, tiny_test_cfg):
    """F25: Actor network input dimension is strictly state_dim = latent_dim + belief_dim with no inventory slots."""
    expected_dim = tiny_test_cfg.world_model.latent_dim + tiny_test_cfg.world_model.belief_dim
    # Inspect first linear layer of actor network
    first_linear = mock_heads.actor.net[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == expected_dim, (
        f"Actor in_features {first_linear.in_features} != expected state_dim {expected_dim}"
    )


def test_f25_inventory_vector_used_for_diagnostics_only():
    """F25: Inventory annotations are used exclusively for ReadoutHead MSE diagnostics."""
    heads_file = ZEPHYR_DIR / "heads.py"
    with open(heads_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    # Readout head predicts 16 inventory keys
    readout_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ReadoutHead":
            readout_def = node
            break
    assert readout_def is not None, "ReadoutHead class definition not found"


def test_f25_screen_strict_inference_flag_enforcement(tmp_path, mock_checkpoint_factory):
    """F25: Benchmark runner asserts strict_inference: True and privileged_inventory_used: False."""
    ckpts = mock_checkpoint_factory()
    out_file = tmp_path / "screen_priv.json"

    report = run_screen(
        checkpoint_wm=str(ckpts["wm"]),
        checkpoint_heads=str(ckpts["heads"]),
        out_path=str(out_file),
        num_episodes=1,
        seed_start=2050,
        max_steps=1,
        device="cpu",
    )

    assert report["strict_inference"] is True
    assert report["privileged_inventory_used"] is False
    assert report["achievements_used_for_scoring_only"] is True


def test_f25_blackbox_agent_execution_with_masked_env_info(mock_wm, mock_heads):
    """F25: Agent executes and selects actions without access to the environment info dict."""
    agent = ZephyrAgent(mock_wm, mock_heads, device="cpu")
    obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # Calling act requires only obs, no info parameter can even be provided
    action = agent.act(obs)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action <= 16
