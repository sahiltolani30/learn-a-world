"""High-level TraceManager coordinating observability events."""

import atexit
import traceback
import uuid
from typing import Any, Dict, List, Optional

from .events import Event, EventType
from .trace_writer import TraceWriter


class TraceManager:
    """Coordinates lifecycle, action, training, and failure event tracking."""

    def __init__(
        self,
        run_id: str,
        output_dir: str,
        code_sha: str = "8829fe0",
        config_hash: str = "default",
        enabled: bool = True,
    ):
        self.run_id = run_id
        self.output_dir = output_dir
        self.code_sha = code_sha
        self.config_hash = config_hash
        self.enabled = enabled

        self._seq = 0
        trace_file = f"{output_dir}/traces/{run_id}.jsonl"
        self.writer = TraceWriter(trace_file, enabled=enabled)

        # Track active decision IDs to guarantee 100% decision-outcome linkage
        self._active_decisions: Dict[str, dict] = {}

        if self.enabled:
            atexit.register(self.shutdown)

    def _next_event_id(self) -> str:
        self._seq += 1
        return f"evt-{self.run_id}-{self._seq:08d}"

    def emit(
        self,
        event_type: EventType,
        episode_id: int,
        timestep: int,
        payload: Dict[str, Any],
    ) -> Event:
        event = Event(
            event_id=self._next_event_id(),
            event_type=event_type.value,
            run_id=self.run_id,
            episode_id=episode_id,
            timestep=timestep,
            code_sha=self.code_sha,
            config_hash=self.config_hash,
            payload=payload,
        )
        self.writer.write_event(event)
        return event

    # --- Lifecycle ---
    def log_run_started(self, config_dict: dict) -> Event:
        # Bounded payload: strip internal or excessive configs
        safe_cfg = {k: v for k, v in config_dict.items() if k != "secrets"}
        return self.emit(
            EventType.RUN_STARTED,
            episode_id=0,
            timestep=0,
            payload={"config": safe_cfg},
        )

    def log_episode_started(self, episode_id: int, seed: int) -> Event:
        return self.emit(
            EventType.EPISODE_STARTED,
            episode_id=episode_id,
            timestep=0,
            payload={"seed": seed},
        )

    def log_episode_ended(self, episode_id: int, total_steps: int, total_reward: float) -> Event:
        return self.emit(
            EventType.EPISODE_ENDED,
            episode_id=episode_id,
            timestep=total_steps,
            payload={"total_steps": total_steps, "total_reward": float(total_reward)},
        )

    # --- Actions & Outcomes (Strict 1-to-1 Linkage) ---
    def log_observation_received(
        self, episode_id: int, timestep: int, obs_ref: str
    ) -> Event:
        return self.emit(
            EventType.OBSERVATION_RECEIVED,
            episode_id=episode_id,
            timestep=timestep,
            payload={"obs_ref": obs_ref},
        )

    def log_action_decision(
        self,
        episode_id: int,
        timestep: int,
        obs_ref: str,
        candidate_scores: Optional[List[float]],
        selected_token: int,
        mapping_version: str,
        role: str,
        mode: str,
    ) -> str:
        decision_id = f"dec-{self.run_id}-{episode_id}-{timestep}"
        self.emit(
            EventType.ACTION_DECISION,
            episode_id=episode_id,
            timestep=timestep,
            payload={
                "decision_id": decision_id,
                "obs_ref": obs_ref,
                "candidate_scores": candidate_scores,
                "selected_token": int(selected_token),
                "mapping_version": mapping_version,
                "role": role,
                "mode": mode,
            },
        )
        self._active_decisions[decision_id] = {
            "episode_id": episode_id,
            "timestep": timestep,
            "selected_token": int(selected_token),
        }
        return decision_id

    def log_action_executed(self, decision_id: str) -> Event:
        info = self._active_decisions.get(decision_id, {"episode_id": 0, "timestep": 0, "selected_token": 0})
        return self.emit(
            EventType.ACTION_EXECUTED,
            episode_id=info["episode_id"],
            timestep=info["timestep"],
            payload={
                "decision_id": decision_id,
                "selected_token": info["selected_token"],
            },
        )

    def log_action_outcome(
        self,
        decision_id: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs_ref: str,
    ) -> Event:
        info = self._active_decisions.pop(decision_id, {"episode_id": 0, "timestep": 0})
        return self.emit(
            EventType.ACTION_OUTCOME,
            episode_id=info["episode_id"],
            timestep=info["timestep"],
            payload={
                "decision_id": decision_id,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "next_obs_ref": next_obs_ref,
            },
        )

    # --- Replay & Training ---
    def log_replay_inserted(
        self, episode_id: int, timestep: int, buffer_index: int
    ) -> Event:
        return self.emit(
            EventType.REPLAY_INSERTED,
            episode_id=episode_id,
            timestep=timestep,
            payload={"buffer_index": buffer_index},
        )

    def log_sequence_sampled(
        self, batch_size: int, horizons: List[int], context_length: int
    ) -> Event:
        return self.emit(
            EventType.SEQUENCE_SAMPLED,
            episode_id=0,
            timestep=0,
            payload={
                "batch_size": batch_size,
                "horizons": horizons,
                "context_length": context_length,
            },
        )

    def log_training_update_started(self, update_id: int, step: int) -> Event:
        return self.emit(
            EventType.TRAINING_UPDATE_STARTED,
            episode_id=0,
            timestep=step,
            payload={"update_id": update_id},
        )

    def log_training_update_finished(
        self,
        update_id: int,
        step: int,
        losses: Dict[str, float],
        grad_norm: float,
        lr: float,
    ) -> Event:
        return self.emit(
            EventType.TRAINING_UPDATE_FINISHED,
            episode_id=0,
            timestep=step,
            payload={
                "update_id": update_id,
                "losses": {k: float(v) for k, v in losses.items()},
                "grad_norm": float(grad_norm),
                "lr": float(lr),
            },
        )

    def log_ema_updated(self, step: int, momentum: float) -> Event:
        return self.emit(
            EventType.EMA_UPDATED,
            episode_id=0,
            timestep=step,
            payload={"step": step, "momentum": float(momentum)},
        )

    # --- Checkpoints & Failures ---
    def log_checkpoint_saved(self, checkpoint_path: str, step: int) -> Event:
        return self.emit(
            EventType.CHECKPOINT_SAVED,
            episode_id=0,
            timestep=step,
            payload={"checkpoint_path": str(checkpoint_path)},
        )

    def log_checkpoint_loaded(self, checkpoint_path: str, step: int) -> Event:
        return self.emit(
            EventType.CHECKPOINT_LOADED,
            episode_id=0,
            timestep=step,
            payload={"checkpoint_path": str(checkpoint_path)},
        )

    def log_failure(
        self, error_type: str, message: str, stack_trace: Optional[str] = None
    ) -> Event:
        if stack_trace is None:
            stack_trace = traceback.format_exc()
        return self.emit(
            EventType.FAILURE,
            episode_id=0,
            timestep=0,
            payload={
                "error_type": error_type,
                "message": message,
                "stack_trace": stack_trace,
            },
        )

    def log_gate_result(
        self, gate_id: str, status: str, metrics: Dict[str, Any]
    ) -> Event:
        return self.emit(
            EventType.GATE_RESULT,
            episode_id=0,
            timestep=0,
            payload={"gate_id": gate_id, "status": status, "metrics": metrics},
        )

    def shutdown(self) -> str:
        """Close tracer, write checksum file, and return checksum."""
        return self.writer.close()
