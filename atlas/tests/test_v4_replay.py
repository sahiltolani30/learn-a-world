"""Verification tests for T4 Replay CLI (atlas/replay.py)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture
def golden_trace_dir(tmp_path: Path) -> Path:
    """Creates a temporary run_dir with a golden trace file of 20 steps."""
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / "trace_it121.jsonl"

    lines = []
    for t in range(20):
        # Line 5 has an unlock
        newly_unlocked = [7] if t == 5 else []
        # Line 10 has an override
        override_fired = (t == 10)
        a_intended = 4
        a_final = 2 if override_fired else 4
        # Line 15 has an abort
        abort_reason = "stagnation" if t == 15 else "none"
        # Line 18 has arrived
        arrived = (t == 18)

        line = {
            "iter": 121,
            "ep_idx": 3,
            "seed": 6234,
            "t": t,
            "mode": "NAVIGATE" if t < 18 else "PROBE",
            "prev_mode": "NAVIGATE",
            "goal": {
                "kind": "event",
                "dims": [7],
                "cos": 0.71,
                "psi": 0.42,
                "budget": 44,
                "goal_mode": "material_subgoal",
                "n_try": 3,
                "n_succ": 1,
            },
            "scores_source": "option_logits",
            "scores": [0.0] * 17,
            "a_intended": a_intended,
            "a_final": a_final,
            "epsilon_fired": False,
            "epsilon": 0.12,
            "beta_t": 0.35,
            "override": {
                "fired": override_fired,
                "risk_intended": 0.85 if override_fired else 0.11,
                "min_risk": 0.03,
                "a_intended": a_intended,
                "a_final": a_final,
            },
            "risks": [0.05] * 17,
            "event_p_max": [0.01] * 22,
            "m_soft": [False] * 22,
            "newly_unlocked": newly_unlocked,
            "probe_id": 12,
            "abort_reason": abort_reason,
            "arrived": arrived,
            "material_gate": {"suppressed_count": 0},
            "r_prev": 1.0 if t == 5 else 0.0,
            "r_int": 1.32,
        }
        lines.append(line)

    with open(trace_file, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return tmp_path


def test_t4_story_subcommand(golden_trace_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "atlas.replay",
        "story",
        "--run-dir",
        str(golden_trace_dir),
        "--iter",
        "121",
        "--ep",
        "3",
        "--from",
        "5",
        "--to",
        "15",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Command failed: {res.stderr}"
    stdout = res.stdout
    # Must contain override marker
    assert "!!! [OVERRIDE]" in stdout
    # Must contain unlock marker
    assert "!!! [UNLOCK]" in stdout
    # Must contain abort marker
    assert "!!! [ABORT]" in stdout


def test_t4_filter_subcommand(golden_trace_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "atlas.replay",
        "filter",
        "--run-dir",
        str(golden_trace_dir),
        "--override",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    lines = [line for line in res.stdout.strip().splitlines() if line]
    assert len(lines) == 1, f"Expected 1 override line, got {len(lines)}"
    assert "override" in lines[0]


def test_t4_summary_subcommand(golden_trace_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "atlas.replay",
        "summary",
        "--run-dir",
        str(golden_trace_dir),
        "--iter",
        "121",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    stdout = res.stdout
    assert "Overrides:  1" in stdout
    assert "NAVIGATE: 18" in stdout
    assert "PROBE: 2" in stdout


def test_t4_goal_subcommand(golden_trace_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "atlas.replay",
        "goal",
        "--run-dir",
        str(golden_trace_dir),
        "--iter",
        "121",
        "--ep",
        "3",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    stdout = res.stdout
    assert "material_subgoal" in stdout
    assert "event[7]" in stdout


def test_t4_missing_file_exit_code_2(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "atlas.replay",
        "story",
        "--run-dir",
        str(tmp_path),
        "--iter",
        "999",
        "--ep",
        "0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}: {res.stderr}"
    assert "Missing" in res.stderr or "not found" in res.stderr.lower()
