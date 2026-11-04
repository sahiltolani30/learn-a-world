"""Regression test suite for Investigation A: Evaluation and Episode-Accounting Integrity."""
import os
import tempfile
import csv
import pytest
from src.env.evaluator_tap import EvaluatorTap
from src.evaluation.achievement_recorder import AchievementRecorder
from src.evaluation.report import ReportGenerator


def test_interleaved_environments_remain_independent():
    """1. Interleave two environments with different rewards, lengths, and achievements."""
    tap = EvaluatorTap()

    # Step env 0
    tap.record_step(info={"achievements": {"collect_wood": 1}}, reward=1.0, done=False, env_id=0)
    # Step env 1
    tap.record_step(info={"achievements": {"collect_drink": 1}}, reward=0.5, done=False, env_id=1)
    # Step env 0 again
    tap.record_step(info={"achievements": {"place_table": 1}}, reward=2.0, done=True, env_id=0)

    # Env 0 finished (length 2, reward 3.0, wood + table)
    # Env 1 is still running!
    summary = tap.get_summary()
    assert summary["num_episodes"] == 1
    assert summary["episode_rewards"][0] == 3.0
    assert summary["episode_lengths"][0] == 2
    assert summary["achievement_histories"][0].get("collect_wood") == 1
    assert summary["achievement_histories"][0].get("place_table") == 1
    assert summary["achievement_histories"][0].get("collect_drink") is None

    # Step env 1 to completion
    tap.record_step(info={"achievements": {}}, reward=0.5, done=True, env_id=1)
    summary2 = tap.get_summary()
    assert summary2["num_episodes"] == 2
    assert summary2["episode_rewards"][1] == 1.0  # 0.5 + 0.5
    assert summary2["episode_lengths"][1] == 2
    assert summary2["achievement_histories"][1].get("collect_drink") == 1
    assert summary2["achievement_histories"][1].get("collect_wood") is None


def test_reset_one_env_does_not_corrupt_active_episode():
    """2. Reset one environment while another is in progress."""
    tap = EvaluatorTap()

    # Env 0 starts and accumulates reward
    tap.record_step(info={"achievements": {"collect_sapling": 1}}, reward=1.5, done=False, env_id=0)
    # Env 1 resets explicitly
    tap.reset_episode(env_id=1)

    # Env 0 completes
    tap.record_step(info={}, reward=0.5, done=True, env_id=0)
    summary = tap.get_summary()
    assert summary["num_episodes"] == 1
    assert summary["episode_rewards"][0] == 2.0
    assert summary["episode_lengths"][0] == 2
    assert summary["achievement_histories"][0].get("collect_sapling") == 1


def test_terminate_multiple_environments_different_orders():
    """3. Terminate multiple environments in different orders; each recorded exactly once."""
    tap = EvaluatorTap()

    tap.record_step(info={}, reward=1.0, done=False, env_id=0)
    tap.record_step(info={}, reward=2.0, done=False, env_id=1)
    tap.record_step(info={}, reward=3.0, done=False, env_id=2)

    # Terminate env 1 first
    tap.record_step(info={}, reward=0.0, done=True, env_id=1)
    # Terminate env 2 second
    tap.record_step(info={}, reward=0.0, done=True, env_id=2)
    # Terminate env 0 third
    tap.record_step(info={}, reward=0.0, done=True, env_id=0)

    summary = tap.get_summary()
    assert summary["num_episodes"] == 3
    # Check that rewards correspond to order of completion: env 1 (2.0), env 2 (3.0), env 0 (1.0)
    assert summary["episode_rewards"] == [2.0, 3.0, 1.0]


def test_checkpoint_state_persistence_restores_episode_history():
    """4. State persistence enables resumption without losing episode history."""
    rec = AchievementRecorder()
    rec.record_episode({"collect_wood": 1, "place_table": 1}, total_reward=5.0, length=42)
    rec.record_episode({"collect_drink": 1}, total_reward=1.0, length=15)

    state = rec.get_state()
    assert state["num_episodes"] == 2

    # New recorder restores from state
    rec_resumed = AchievementRecorder()
    rec_resumed.load_state(state)
    assert rec_resumed.num_episodes == 2
    summary = rec_resumed.get_summary()
    assert summary["num_episodes"] == 2
    assert summary["mean_reward"] == 3.0
    assert summary["achievement_rates"]["collect_wood"] == 50.0


def test_report_with_zero_evaluation_episodes_marked_invalid():
    """5. Generate a report with zero evaluation episodes; verify it says NOT_RUN/INVALID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report_gen = ReportGenerator(run_dir=tmpdir)
        summary = {
            "num_episodes": 0,
            "crafter_score": None,
            "mean_reward": None,
            "mean_length": None,
            "is_valid": False,
            "status": "NOT_RUN",
            "achievement_rates": {},
        }
        report_path = report_gen.generate_final_report(
            config={},
            heldout_metrics={},
            action_metrics={},
            achievement_summary=summary,
            diagnostics={},
        )
        with open(report_path, "r") as f:
            content = f.read()
        assert "NOT_RUN" in content or "INVALID" in content
        assert "Official Crafter Benchmark Score: 0.00% across 0 evaluated episodes" not in content


def test_benchmark_csv_distinguishes_runs():
    """6. Generate two runs with different run IDs; verify old rows are distinguished."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "benchmark.csv")
        rec1 = AchievementRecorder()
        rec1.record_episode({"collect_wood": 1}, total_reward=2.0, length=20)
        rec1.export_csv(csv_path, seed=123, run_name="run_alpha", checkpoint_id="step_50k")

        rec2 = AchievementRecorder()
        rec2.record_episode({"collect_drink": 1}, total_reward=1.0, length=10)
        rec2.export_csv(csv_path, seed=456, run_name="run_beta", checkpoint_id="step_10k")

        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["run_name"] == "run_alpha"
        assert rows[0]["checkpoint_id"] == "step_50k"
        assert rows[1]["run_name"] == "run_beta"
        assert rows[1]["checkpoint_id"] == "step_10k"
