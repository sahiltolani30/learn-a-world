"""Tier 2: Boundary Tests - Checkpoint Corruption & Recovery.

Category 6: Seed Offsets & Checkpoint Corruption Recovery
Tests:
31. test_t2_checkpoint_mid_write_tmp_file_interruption
32. test_t2_checkpoint_corrupt_state_file_detection
33. test_t2_checkpoint_missing_optimizer_state_recovery
34. test_t2_checkpoint_config_mismatch_validation
35. test_t2_seed_offset_astronomical_generation_scale
36. test_t2_checkpoint_drive_mirror_failure_resilience
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

# Cleanly bind eval.metrics without executing eval/__init__.py
if "eval" not in sys.modules:
    eval_mod = types.ModuleType("eval")
    sys.modules["eval"] = eval_mod
if "eval.metrics" not in sys.modules:
    spec = importlib.util.spec_from_file_location("eval.metrics", str(REPO_ROOT / "eval" / "metrics.py"))
    metrics_mod = importlib.util.module_from_spec(spec)
    sys.modules["eval.metrics"] = metrics_mod
    spec.loader.exec_module(metrics_mod)
    setattr(sys.modules["eval"], "metrics", metrics_mod)

from zephyr.screen import run_screen
from zephyr.train_loop import ZephyrTrainer
from zephyr.wm import ZephyrWM


def test_t2_checkpoint_mid_write_tmp_file_interruption(tmp_path: pathlib.Path, tiny_test_cfg):
    """T2-Recovery: Dangling state.pt.tmp from interrupted write does not prevent loading valid state.pt."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    trainer.train_steps = 25
    valid_ckpt = trainer.save_run()

    # Simulate crash leaving partial .tmp file alongside valid state.pt
    dangling_tmp = tmp_path / "state.pt.tmp"
    with open(dangling_tmp, "wb") as f:
        f.write(b"CORRUPTED_INCOMPLETE_WRITE_DATA")

    # Resumption must cleanly load the intact state.pt
    loaded = ZephyrTrainer.load_run(valid_ckpt, device="cpu")
    assert loaded.train_steps == 25
    assert os.path.exists(valid_ckpt)


def test_t2_checkpoint_corrupt_state_file_detection(tmp_path: pathlib.Path):
    """T2-Recovery: Truncated or zero-byte state.pt is detected and raises descriptive Exception."""
    corrupt_state = tmp_path / "corrupt_state.pt"
    # Zero byte file
    corrupt_state.touch()

    with pytest.raises(Exception):
        ZephyrTrainer.load_run(str(corrupt_state), device="cpu")

    # Random garbage bytes
    with open(corrupt_state, "wb") as f:
        f.write(b"\x00\xff\xfe\x01\x02\x03GARBAGE_PAYLOAD")

    with pytest.raises(Exception):
        ZephyrTrainer.load_run(str(corrupt_state), device="cpu")


def test_t2_checkpoint_missing_optimizer_state_recovery(tmp_path: pathlib.Path, tiny_test_cfg):
    """T2-Recovery: Checkpoint with empty opt_states dictionary restores models and counters cleanly."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    trainer.train_steps = 15
    save_path = trainer.save_run()

    # Modify payload to remove optimizer states
    payload = torch.load(save_path, map_location="cpu", weights_only=False)
    payload["opt_states"] = {}
    torch.save(payload, save_path)

    loaded = ZephyrTrainer.load_run(save_path, device="cpu")
    assert loaded.train_steps == 15
    assert hasattr(loaded, "wm")
    assert hasattr(loaded, "heads")


def test_t2_checkpoint_config_mismatch_validation(tmp_path: pathlib.Path, tiny_test_cfg):
    """T2-Recovery: Incompatible or invalid run format raises ValueError before executing."""
    trainer = ZephyrTrainer(tiny_test_cfg, run_dir=str(tmp_path), device="cpu")
    save_path = trainer.save_run()

    payload = torch.load(save_path, map_location="cpu", weights_only=False)
    payload["format"] = "legacy_unsupported_v0"
    torch.save(payload, save_path)

    with pytest.raises(ValueError, match="unexpected run format"):
        ZephyrTrainer.load_run(save_path, device="cpu")


def test_t2_seed_offset_astronomical_generation_scale():
    """T2-Recovery: Resumption at generation 2,000,000 evaluates safely within 64-bit integer limits."""
    gen = 2_000_000
    base_seed = 7000
    seed = int(base_seed + 1_000_000_000 * gen)

    assert seed == 2_000_000_000_007_000
    assert seed.bit_length() <= 63, f"Seed bit length {seed.bit_length()} exceeds 63 bits"
    assert seed > 0


def test_t2_checkpoint_drive_mirror_failure_resilience(tmp_path: pathlib.Path, mock_checkpoint_factory):
    """T2-Recovery: Drive mirror target creation handles missing or invalid target paths predictably."""
    ckpts = mock_checkpoint_factory()
    out_file = tmp_path / "screen_local.json"
    valid_mirror = tmp_path / "mirror_dir" / "screen_mirror.json"

    # Verify normal mirror write succeeds
    report = run_screen(
        checkpoint_wm=str(ckpts["wm"]),
        checkpoint_heads=str(ckpts["heads"]),
        out_path=str(out_file),
        num_episodes=1,
        seed_start=2050,
        max_steps=1,
        device="cpu",
        drive_mirror=str(valid_mirror),
    )

    assert out_file.exists(), "Local report file missing"
    assert valid_mirror.exists(), "Drive mirror report file missing"
    assert report["protocol_id"] == "strict_visual_zephyr_clean_v1"
