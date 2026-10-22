"""
Unit and Integration Tests for Phase 3 and Phase 4 of Learn-a-World v2:
- Multi-step unroll consistency loss (no NaN)
- Recurrent GRU belief module temporal stability across 16 steps
- Collect-retrain round execution and cumulative history logging
- Phase 4 promotion rule criteria audit and markdown report generation
- End-to-end evaluation runner smoke test
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from eval.run_final_comparison import generate_markdown_report, run_evaluation_suite
from models.heads import GRUBeliefModule
from models.jepa import JEPAWorldModel
from models.jepa_v2 import JEPAv2
from training.sequence_buffer import EpisodeBuffer
from training.trainer_v2 import run_step3_rounds, train_joint

@pytest.fixture
def mini_buffer() -> EpisodeBuffer:
    """Creates a tiny synthetic buffer for fast CPU testing."""
    buf = EpisodeBuffer(capacity_transitions=100)
    for ep_id in range(3):
        T = 8
        obs = np.random.randint(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randint(0, 17, size=(T,), dtype=np.int64)
        rewards = np.random.uniform(-0.1, 1.0, size=(T,)).astype(np.float32)
        dones = np.zeros(T, dtype=np.bool_)
        dones[-1] = True
        events = np.zeros((T, 22), dtype=np.uint8)
        events[1, 0] = 1  # simulated collect_wood event
        masks = np.zeros((T, 22), dtype=np.uint8)
        buf.add_episode(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            events=events,
            mask_before=masks,
            source="synthetic_test",
        )
    return buf

@pytest.fixture
def cpu_v2_model() -> JEPAv2:
    """Creates a lightweight JEPAv2 model on CPU."""
    base = JEPAWorldModel(
        in_channels=3,
        latent_dim=64,
        num_actions=17,
        action_emb_dim=16,
        predictor_hidden_dim=64,
        predictor_layers=1,
    )
    return JEPAv2(
        base_model=base,
        hidden_dim=64,
        action_emb_dim=16,
        q_heads=2,
        q_bins=11,
        freeze_base=False,
    )

def test_multi_step_unroll_loss_no_nan(cpu_v2_model: JEPAv2, mini_buffer: EpisodeBuffer):
    """Verifies that multi-step imagination unroll training runs without NaN."""
    history = train_joint(
        model=cpu_v2_model,
        buffer=mini_buffer,
        total_steps=3,
        batch_size=2,
        seq_len=4,
        unroll_steps=2,
        lr_base=1e-4,
        lr_heads=3e-4,
        device="cpu",
    )
    assert len(history["loss"]) > 0
    for loss_val in history["loss"]:
        assert not np.isnan(loss_val), "Found NaN in joint training loss!"
        assert not np.isinf(loss_val), "Found Inf in joint training loss!"

def test_gru_belief_temporal_stability():
    """Verifies that GRU belief hidden state runs 16 steps without exploding or NaN."""
    B = 4
    T = 16
    latent_dim = 64
    hidden_dim = 128
    gru = GRUBeliefModule(
        latent_dim=latent_dim,
        num_actions=17,
        action_emb_dim=16,
        hidden_dim=hidden_dim,
    )

    h_t = None
    for t in range(T):
        z_t = torch.randn(B, latent_dim)
        a_prev = torch.randint(0, 17, (B,), dtype=torch.int64)
        h_t = gru(z_t, a_prev, h_t)

        assert h_t.shape == (B, hidden_dim)
        assert not torch.isnan(h_t).any(), f"NaN in GRU state at step {t}"
        assert torch.norm(h_t).item() < 100.0, f"Exploding GRU state at step {t}"

def test_phase4_promotion_decision_logic():
    """Verifies the exact promotional criteria rule (§6 Step 4)."""
    # Rule: v2_pure_score > v1_hybrid_score AND v2_pure_score >= v1_pure_score + 1.0

    # Case 1: Passes both
    v2_pure = 4.2
    v1_hybrid = 3.5
    v1_pure = 1.0
    pass_case = (v2_pure > v1_hybrid) and (v2_pure >= v1_pure + 1.0)
    assert pass_case is True

    # Case 2: Beats pure + 1.0, but fails hybrid
    v2_pure = 2.8
    v1_hybrid = 3.5
    v1_pure = 1.0
    fail_case_1 = (v2_pure > v1_hybrid) and (v2_pure >= v1_pure + 1.0)
    assert fail_case_1 is False

    # Case 3: Beats hybrid, but doesn't beat pure + 1.0
    v2_pure = 3.6
    v1_hybrid = 3.5
    v1_pure = 3.0
    fail_case_2 = (v2_pure > v1_hybrid) and (v2_pure >= v1_pure + 1.0)
    assert fail_case_2 is False

def test_markdown_report_generation(tmp_path: Path):
    """Verifies that results/promotion_decision.md is correctly written with all tables."""
    dummy_data = {
        "timestamp": "2026-10-22T10:00:00",
        "protocol": {"episodes": 100, "seed_start": 1000, "seed_end": 1099, "max_steps": 1000},
        "promotion_decision": {
            "promoted": True,
            "rule": "v2_pure_score > v1_hybrid_score AND v2_pure_score >= v1_pure_score + 1.0",
            "v2_pure_score": 4.5,
            "v1_hybrid_score": 3.2,
            "v1_pure_score": 0.0,
            "condition_beat_hybrid": True,
            "condition_beat_pure_plus_one": True,
            "margin_over_hybrid": 1.3,
            "margin_over_pure": 4.5,
        },
        "agents": {
            "v1_hybrid": {
                "evaluator": "eval/eval_crafter_v1_ablation.py",
                "checkpoint": "checkpoints/jepa_crafter.pt",
                "reflexes": "on",
                "crafter_score": 3.2,
                "unlocked_count": 6,
                "mean_survival_steps": 125.0,
                "mean_reward": 5.2,
                "per_achievement_rates": {"collect_wood": 80.0},
            },
            "v2_pure": {
                "evaluator": "eval/eval_v2.py",
                "checkpoint": "checkpoints/jepa_crafter_v2_best.pt",
                "reflexes": "none",
                "crafter_score": 4.5,
                "unlocked_count": 8,
                "mean_survival_steps": 150.0,
                "mean_reward": 6.8,
                "per_achievement_rates": {"collect_wood": 90.0},
            },
        },
    }

    out_file = tmp_path / "promotion_decision.md"
    generate_markdown_report(dummy_data, str(out_file))

    assert out_file.exists()
    content = out_file.read_text()
    assert "PROMOTED TO V2" in content
    assert "Executive Summary & Comparison Table" in content
    assert "v2_pure" in content
    assert "v1_hybrid" in content

def test_final_comparison_runner_smoke(tmp_path: Path):
    """Runs a 1-episode smoke run of the comparison suite on random baseline."""
    out_json = tmp_path / "smoke_comparison.json"
    out_md = tmp_path / "smoke_decision.md"

    res = run_evaluation_suite(
        episodes=1,
        max_steps=5,
        seed_start=1000,
        agents=["random"],
        out_json=str(out_json),
        out_md=str(out_md),
    )

    assert out_json.exists()
    assert out_md.exists()
    assert "random" in res["agents"]
    assert "promotion_decision" in res

def test_phase3_mini_round_smoke(tmp_path: Path):
    """Executes a 1-round mini collect-retrain smoke cycle on Mac/CPU."""
    ckpt_dir = tmp_path / "checkpoints"
    out_dir = tmp_path / "results"

    res = run_step3_rounds(
        base_checkpoint_path="checkpoints/jepa_crafter.pt",
        config_path="experiments/configs/crafter_v2.yaml",
        rounds=1,
        transitions_per_round=20,
        joint_steps_per_round=2,
        eval_episodes=1,
        eval_max_steps=5,
        out_dir=str(out_dir),
        ckpt_dir_override=str(ckpt_dir),
        device_override="cpu",
    )
    assert "rounds" in res
    assert len(res["rounds"]) == 1
    assert "crafter_score" in res["rounds"][0]
    rounds_json = out_dir / "v2_step3_rounds.json"
    assert rounds_json.exists()
    data = json.loads(rounds_json.read_text())
    assert "rounds" in data
    assert "best_score" in data
