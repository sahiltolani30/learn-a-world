"""
Stage 8 (online cross-game learning loop) unit tests.

Spec: docs/PHASE2_MEMORY_REDESIGN_PLAN.md §3 Stage 8, §4.
Covers: strict-inference recorder guard (achievements rejection), bank-contract
step dict construction, the confident-firing margin rule, episode rollover into
the bank, belief reward-fallback provenance (N7), and bounded subgoal provenance.

All tests are CPU-only and fast; the bank is a Mock (its implementation lands in
parallel — the recorder is coded against the learn_from_episode contract only).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.agent_v3 import AgentBeliefV3
from planning.playtime_recorder import (
    GROUNDED_STATE_DIM,
    NUM_EVENT_DIMS,
    OBS_LATENT_DIM,
    PlaytimeRecorder,
)
from scripts.screen_v3_50eps import collect_subgoal_trace

BANK_CONTRACT_KEYS = {
    "obs_latent",
    "grounded_state",
    "action",
    "reward",
    "done",
    "event_vec",
    "mask_before_vec",
    "confident_events",
    "fallback_unlocked",
}


def _base_step_kwargs() -> Dict[str, Any]:
    """Minimal valid record_step payload (strict-inference shapes)."""
    return dict(
        z_t=torch.zeros(OBS_LATENT_DIM),
        x_hat=torch.zeros(GROUNDED_STATE_DIM),
        action=3,
        reward=0.0,
        done=False,
        mask_before=np.zeros(NUM_EVENT_DIMS),
        newly_unlocked=[],
    )


def _record_and_capture(recorder: PlaytimeRecorder) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Mock]:
    """Flushes the buffered episode into a mock bank; returns (learned steps, learn kwargs, bank)."""
    bank = Mock()
    bank.learn_from_episode.return_value = {"created": 0}
    recorder.end_episode(bank)
    args, kwargs = bank.learn_from_episode.call_args
    return args[0], kwargs, bank


def test_record_step_rejects_engine_achievements():
    """Engine achievements are scoring-only; handing one to the recorder must raise."""
    recorder = PlaytimeRecorder()
    with pytest.raises(ValueError, match="scoring-only"):
        recorder.record_step(
            **_base_step_kwargs(),
            achievements={"collect_wood": 1},
        )
    # Nothing may have been buffered by the rejected call.
    assert len(recorder) == 0


def test_record_step_converts_tensors_to_lists():
    """Torch/numpy inputs land in the bank contract as plain lists of the right dims."""
    recorder = PlaytimeRecorder()
    z_t = torch.arange(OBS_LATENT_DIM, dtype=torch.float32)
    x_hat = torch.arange(GROUNDED_STATE_DIM, dtype=torch.float32)
    mask_before = np.zeros(NUM_EVENT_DIMS, dtype=np.float32)
    mask_before[:7] = 1.0

    recorder.record_step(
        z_t=z_t,
        x_hat=x_hat,
        action=np.int64(5),
        reward=np.float64(1.0),
        done=np.bool_(True),
        mask_before=mask_before,
        newly_unlocked=np.array([3, 11]),
        fallback_unlocked=np.array([3]),
        event_probs=torch.full((NUM_EVENT_DIMS,), 0.9),
        thresholds=np.full(NUM_EVENT_DIMS, 0.5),
    )

    steps, kwargs, _ = _record_and_capture(recorder)

    assert len(steps) == 1
    step = steps[0]
    assert set(step.keys()) == BANK_CONTRACT_KEYS
    assert isinstance(step["obs_latent"], list)
    assert len(step["obs_latent"]) == OBS_LATENT_DIM
    assert all(isinstance(v, float) for v in step["obs_latent"])
    assert step["obs_latent"][0] == 0.0
    assert step["obs_latent"][OBS_LATENT_DIM - 1] == float(OBS_LATENT_DIM - 1)
    assert isinstance(step["grounded_state"], list)
    assert len(step["grounded_state"]) == GROUNDED_STATE_DIM
    assert step["action"] == 5 and isinstance(step["action"], int)
    assert step["reward"] == 1.0
    assert step["done"] is True
    assert step["event_vec"] == [3, 11]
    assert step["mask_before_vec"] == [1] * 7 + [0] * (NUM_EVENT_DIMS - 7)
    assert step["confident_events"] == [3, 11]
    assert step["fallback_unlocked"] == [3]
    assert kwargs["episode_id"] == 0


def test_confident_events_margin_rule():
    """An unlock is confident only when prob > threshold + margin (0.15)."""
    # Above margin: 0.5 > 0.25 + 0.15 -> confident.
    recorder = PlaytimeRecorder()
    kwargs = _base_step_kwargs()
    kwargs.update(newly_unlocked=[6], event_probs=[0.5] * NUM_EVENT_DIMS, thresholds=[0.25] * NUM_EVENT_DIMS)
    recorder.record_step(**kwargs)
    steps, _, _ = _record_and_capture(recorder)
    assert steps[0]["confident_events"] == [6]

    # Below margin: 0.3 <= 0.25 + 0.15 -> not confident.
    recorder = PlaytimeRecorder()
    kwargs = _base_step_kwargs()
    kwargs.update(newly_unlocked=[6], event_probs=[0.3] * NUM_EVENT_DIMS, thresholds=[0.25] * NUM_EVENT_DIMS)
    recorder.record_step(**kwargs)
    steps, _, _ = _record_and_capture(recorder)
    assert steps[0]["confident_events"] == []

    # Missing probs/thresholds -> nothing counts as event-margin evidence.
    recorder = PlaytimeRecorder()
    kwargs = _base_step_kwargs()
    kwargs["newly_unlocked"] = [6]
    recorder.record_step(**kwargs)
    steps, _, _ = _record_and_capture(recorder)
    assert steps[0]["confident_events"] == []


def test_end_episode_calls_bank_and_clears():
    """Whole buffered episode goes to learn_from_episode, edges refresh, buffer rolls over."""
    recorder = PlaytimeRecorder()
    kwargs_a = _base_step_kwargs()
    kwargs_a["action"] = 1
    kwargs_b = _base_step_kwargs()
    kwargs_b.update(action=2, done=True, newly_unlocked=[4])
    recorder.record_step(**kwargs_a)
    recorder.record_step(**kwargs_b)

    bank = Mock()
    expected_stats = {"created": 2, "confirmed": 0, "denied": 0}
    bank.learn_from_episode.return_value = expected_stats

    stats = recorder.end_episode(bank)

    assert stats == expected_stats
    assert bank.learn_from_episode.call_count == 1
    args, kwargs = bank.learn_from_episode.call_args
    assert kwargs["episode_id"] == 0
    learned = args[0]
    assert len(learned) == 2
    assert learned[0]["action"] == 1
    assert learned[1]["action"] == 2 and learned[1]["event_vec"] == [4]
    bank.refresh_edges.assert_called_once_with()
    assert len(recorder) == 0

    # Second episode rolls the episode id forward.
    recorder.record_step(**kwargs_a)
    recorder.end_episode(bank)
    assert bank.learn_from_episode.call_args.kwargs["episode_id"] == 1


def test_belief_fallback_provenance_recorded():
    """Reward fallback (N7) exposes its pinned dim; the next normal update clears it."""
    belief = AgentBeliefV3(recurrent_dim=64, frontier_threshold=0.9, strict_inference=True)
    zeros = torch.zeros(NUM_EVENT_DIMS)

    # reward >= 0.9 with no head firing: fallback pins argmax over locked dims (all-zero probs -> dim 0).
    belief.update(zeros, reward=1.0)
    assert belief.history[-1]["fallback_unlocked"] == [0]
    assert belief.last_fallback_unlocked == [0]
    # Legacy belief behavior is untouched.
    assert belief.history[-1]["newly_unlocked"] == [0]
    assert belief.mask[0] == 1.0

    # Next normal (non-fallback) update clears the provenance.
    belief.update(zeros, reward=0.0)
    assert belief.last_fallback_unlocked == []
    assert belief.history[-1]["fallback_unlocked"] == []


def test_episode_records_carry_subgoal_trace_cap():
    """Subgoal provenance keeps at most `cap` entries per episode, dropping empty statuses."""
    entries = [(i, "EXECUTE", f"card_a{i % 3}_e{i}") for i in range(30)]
    trace = collect_subgoal_trace(entries, cap=20)
    assert len(trace) == 20
    assert trace[0] == {"step": 0, "status": "EXECUTE", "target": "card_a0_e0"}
    assert trace[-1]["step"] == 19

    # Empty statuses are filtered before the cap is applied.
    mixed = [(0, "", "ignored"), (1, "READY", "card_a0_e5"), (2, "", "ignored2")]
    assert collect_subgoal_trace(mixed, cap=20) == [
        {"step": 1, "status": "READY", "target": "card_a0_e5"}
    ]
