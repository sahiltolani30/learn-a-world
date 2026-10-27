from pathlib import Path
import json
import pytest
import torch

from models.jepa_v3 import JEPAv3
from scripts.trace_gameplay_failure_path import run_failure_path_trace


def test_failure_path_trace_smoke(tmp_path):
    # Create dummy checkpoint
    model = JEPAv3()
    ckpt_path = tmp_path / "dummy_trace.pt"
    model.save_checkpoint(str(ckpt_path))

    out_json = tmp_path / "trace_out.json"
    summary = run_failure_path_trace(
        checkpoint_path=str(ckpt_path),
        calibrator_path="results/calibration/event_head_calibrator.json",
        num_episodes=1,
        seed_start=2050,
        max_steps=3,
        horizon=3,
        num_candidates=8,
        cem_iterations=1,
        device="cpu",
        output_json=str(out_json),
    )

    assert out_json.exists()
    with open(out_json, "r") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "episodes" in data
    assert len(data["episodes"]) == 1
    ep0 = data["episodes"][0]
    assert "steps" in ep0
    assert len(ep0["steps"]) == 3
    s0 = ep0["steps"][0]
    assert "inventory_diagnostic" in s0
    assert "achievement_diagnostic" in s0
    assert "event_probabilities" in s0
    assert "q_values" in s0
    assert "policy_prior" in s0
    assert "latent_norm" in s0
    assert "first_failed_link" in ep0["summary"]
