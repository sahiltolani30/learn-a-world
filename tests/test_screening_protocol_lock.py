import pytest
from pathlib import Path
from scripts.screen_v3_50eps import _resolve_calibrator, _sha256_file, _get_git_commit, run_screening
from models.jepa_v3 import JEPAv3
import torch


def test_calibrator_resolution_and_hashing():
    cal_path = "results/calibration/event_head_calibrator.json"
    res = _resolve_calibrator(cal_path)
    assert res is not None
    assert res.exists()
    h = _sha256_file(res)
    assert h is not None
    assert len(h) == 64


def test_git_commit_retrieval():
    commit = _get_git_commit()
    assert commit != "unknown"
    assert len(commit) == 40


def test_screening_fails_on_missing_calibrator_without_diagnostic_flag(tmp_path):
    # Dummy checkpoint
    model = JEPAv3()
    ckpt_path = tmp_path / "dummy.pt"
    model.save_checkpoint(str(ckpt_path))

    with pytest.raises(RuntimeError, match="Protocol Lock Violation"):
        run_screening(
            checkpoint_path=str(ckpt_path),
            calibrator_path="non_existent_calibrator_path.json",
            allow_uncalibrated_diagnostic=False,
            num_episodes=1,
            max_steps=2,
            output_json=str(tmp_path / "out.json"),
        )


def test_screening_allows_uncalibrated_diagnostic_flag(tmp_path):
    model = JEPAv3()
    ckpt_path = tmp_path / "dummy.pt"
    model.save_checkpoint(str(ckpt_path))

    summary = run_screening(
        checkpoint_path=str(ckpt_path),
        calibrator_path="non_existent_calibrator_path.json",
        allow_uncalibrated_diagnostic=True,
        num_episodes=1,
        max_steps=2,
        output_json=str(tmp_path / "out.json"),
    )
    assert summary["protocol_id"] == "uncalibrated_diagnostic_not_a_benchmark"
    assert summary["calibrator_sha256"] is None
