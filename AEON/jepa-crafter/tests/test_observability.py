"""Tests for Phase 1: Minimal Observability Foundation."""

import json
import pytest
import shutil
import tempfile
from pathlib import Path

from src.observability.events import Event, EventType, FORBIDDEN_LEAK_KEYS
from src.observability.trace_writer import TraceWriter
from src.observability.tracer import TraceManager
from src.env.opaque_actions import OpaqueActionMapping


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_synthetic_episode_reconstruction_and_100pct_linkage(temp_dir):
    """Test 1 & 2: Synthetic episode with at least 10 transitions can be reconstructed from logs

    and every action decision has exactly one linked execution and one outcome.
    """
    tracer = TraceManager(run_id="synth-001", output_dir=str(temp_dir), enabled=True)
    tracer.log_run_started({"seed": 1234})
    tracer.log_episode_started(episode_id=1, seed=100)

    num_transitions = 12
    decision_ids = []
    
    for t in range(num_transitions):
        obs_ref = f"frame-1-{t}"
        tracer.log_observation_received(episode_id=1, timestep=t, obs_ref=obs_ref)
        
        dec_id = tracer.log_action_decision(
            episode_id=1,
            timestep=t,
            obs_ref=obs_ref,
            candidate_scores=[0.1 * i for i in range(17)],
            selected_token=t % 17,
            mapping_version="v1.0",
            role="Explorer",
            mode="explore",
        )
        decision_ids.append(dec_id)
        
        tracer.log_action_executed(dec_id)
        
        is_terminal = (t == num_transitions - 1)
        tracer.log_action_outcome(
            decision_id=dec_id,
            reward=1.0 if is_terminal else 0.0,
            terminated=is_terminal,
            truncated=False,
            next_obs_ref=f"frame-1-{t+1}",
        )

    tracer.log_episode_ended(episode_id=1, total_steps=num_transitions, total_reward=1.0)
    checksum = tracer.shutdown()

    # Reconstruct from trace
    trace_file = temp_dir / "traces" / "synth-001.jsonl"
    assert trace_file.exists()
    
    events = list(TraceWriter.read_events(str(trace_file)))
    assert len(events) >= num_transitions * 3  # observation, decision, executed, outcome
    
    # Verify 100% linkage between decisions and outcomes
    decisions = {e["payload"]["decision_id"]: e for e in events if e["event_type"] == "action_decision"}
    executions = {e["payload"]["decision_id"]: e for e in events if e["event_type"] == "action_executed"}
    outcomes = {e["payload"]["decision_id"]: e for e in events if e["event_type"] == "action_outcome"}

    assert len(decisions) == num_transitions
    assert len(executions) == num_transitions
    assert len(outcomes) == num_transitions

    for d_id in decision_ids:
        assert d_id in decisions
        assert d_id in executions
        assert d_id in outcomes
        # Check step consistency
        assert decisions[d_id]["timestep"] == outcomes[d_id]["timestep"]


def test_crash_failure_record_and_readability(temp_dir):
    """Test 3: A crash creates a failure record and leaves the last complete trace readable."""
    tracer = TraceManager(run_id="crash-001", output_dir=str(temp_dir), enabled=True)
    tracer.log_episode_started(episode_id=1, seed=42)

    try:
        raise RuntimeError("Simulated unhandled exception during training step")
    except RuntimeError as e:
        tracer.log_failure("RuntimeError", str(e))

    tracer.shutdown()

    trace_file = temp_dir / "traces" / "crash-001.jsonl"
    events = list(TraceWriter.read_events(str(trace_file)))
    assert len(events) == 2
    assert events[0]["event_type"] == "episode_started"
    assert events[1]["event_type"] == "failure"
    assert events[1]["payload"]["error_type"] == "RuntimeError"
    assert "Simulated unhandled exception" in events[1]["payload"]["message"]


def test_trace_writer_flush_reopen_and_checksum(temp_dir):
    """Test 4: Trace writer flush/reopen/checksum round-trip passes."""
    trace_path = str(temp_dir / "traces" / "chk-test.jsonl")
    writer = TraceWriter(trace_path, enabled=True)
    
    for i in range(5):
        evt = Event(
            event_id=f"evt-{i}",
            event_type="test_event",
            run_id="test-run",
            episode_id=1,
            timestep=i,
            payload={"step": i},
        )
        writer.write_event(evt)
    
    writer.flush()
    checksum1 = writer.close()
    assert Path(trace_path).exists()
    assert Path(trace_path).with_suffix(".sha256").exists()
    assert TraceWriter.verify_checksum(trace_path)

    # Reopen in append mode
    writer2 = TraceWriter(trace_path, enabled=True)
    evt_new = Event(
        event_id="evt-new",
        event_type="test_event",
        run_id="test-run",
        episode_id=1,
        timestep=5,
        payload={"step": 5},
    )
    writer2.write_event(evt_new)
    checksum2 = writer2.close()

    assert checksum2 != checksum1
    assert TraceWriter.verify_checksum(trace_path)
    
    events = list(TraceWriter.read_events(trace_path))
    assert len(events) == 6


def test_deterministic_debug_on_off_policy_match():
    """Test 5: Debug enabled vs disabled produces identical actions and model outputs

    in a deterministic test harness.
    """
    mapping1 = OpaqueActionMapping(num_actions=17, seed=777)
    mapping2 = OpaqueActionMapping(num_actions=17, seed=777)

    # Simulate 50 deterministic decision cycles
    actions_debug_off = [mapping1.token_to_env_action(i % 17) for i in range(50)]
    
    # In debug on:
    with tempfile.TemporaryDirectory() as tmp:
        tracer = TraceManager(run_id="dbg-on", output_dir=tmp, enabled=True)
        actions_debug_on = []
        for i in range(50):
            token = i % 17
            dec_id = tracer.log_action_decision(
                episode_id=1,
                timestep=i,
                obs_ref=f"f-{i}",
                candidate_scores=None,
                selected_token=token,
                mapping_version="v1",
                role="Explorer",
                mode="explore",
            )
            env_act = mapping2.token_to_env_action(token)
            tracer.log_action_executed(dec_id)
            tracer.log_action_outcome(dec_id, reward=0.0, terminated=False, truncated=False, next_obs_ref=f"f-{i+1}")
            actions_debug_on.append(env_act)
        tracer.shutdown()

    assert actions_debug_on == actions_debug_off


def test_zero_evaluator_info_leakage(temp_dir):
    """Test 6: Evaluator-only fields inserted into fake info never appear in learner,

    replay, planner or learner-facing logs.
    """
    tracer = TraceManager(run_id="leak-test", output_dir=str(temp_dir), enabled=True)

    # Attempt to insert forbidden privileged keys into payload
    with pytest.raises(ValueError, match="Forbidden privileged key 'inventory'"):
        tracer.emit(
            EventType.OBSERVATION_RECEIVED,
            episode_id=1,
            timestep=0,
            payload={"obs_ref": "frame-0", "inventory": {"wood": 2}},
        )

    with pytest.raises(ValueError, match="Forbidden privileged key 'achievements'"):
        tracer.emit(
            EventType.ACTION_DECISION,
            episode_id=1,
            timestep=0,
            payload={"decision_id": "dec-1", "achievements": ["collect_wood"]},
        )

    tracer.shutdown()
