"""Event definitions and serialization for AEON Observability Foundation."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    # Lifecycle
    RUN_STARTED = "run_started"
    EPISODE_STARTED = "episode_started"
    EPISODE_ENDED = "episode_ended"
    
    # Environment & Action Decision
    OBSERVATION_RECEIVED = "observation_received"
    ACTION_DECISION = "action_decision"
    ACTION_EXECUTED = "action_executed"
    ACTION_OUTCOME = "action_outcome"
    
    # Data & Replay
    REPLAY_INSERTED = "replay_inserted"
    SEQUENCE_SAMPLED = "sequence_sampled"
    
    # Training & Updates
    TRAINING_UPDATE_STARTED = "training_update_started"
    TRAINING_UPDATE_FINISHED = "training_update_finished"
    EMA_UPDATED = "ema_updated"
    
    # Checkpointing
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    
    # Diagnostics & Failures
    FAILURE = "failure"
    GATE_RESULT = "gate_result"


# Strict list of forbidden keys to prevent info leakage into logs
FORBIDDEN_LEAK_KEYS = {
    "inventory", "achievements", "player_pos", "player_facing", 
    "semantic_action", "craft_material", "action_name"
}


@dataclass
class Event:
    event_id: str
    event_type: str
    run_id: str
    episode_id: int
    timestep: int
    timestamp: float = field(default_factory=time.time)
    code_sha: str = "8829fe0"
    config_hash: str = "default"
    component_version: str = "1.0.0"
    payload: Dict[str, Any] = field(default_factory=dict)

    def validate_no_leakage(self) -> None:
        """Assert that no forbidden privileged keys leak into event payloads."""
        def _check(d: Any):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k.lower() in FORBIDDEN_LEAK_KEYS:
                        raise ValueError(f"Forbidden privileged key '{k}' detected in event {self.event_id}")
                    _check(v)
            elif isinstance(d, list):
                for item in d:
                    _check(item)

        _check(self.payload)

    def to_dict(self) -> Dict[str, Any]:
        self.validate_no_leakage()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def compute_checksum(self) -> str:
        content = self.to_json()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
