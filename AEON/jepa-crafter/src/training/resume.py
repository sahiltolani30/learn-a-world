"""Resume manager and integrity verification for restarted runs."""
import os
from typing import Dict, Any, Tuple, Optional
import torch

from src.models.jepa_world_model import JepaWorldModel
from src.learning.learner import JepaLearner
from src.data.replay_buffer import ReplayBuffer
from src.data.checkpoint_store import CheckpointStore
from src.env.opaque_actions import OpaqueActionMapping
from src.utils.seeding import set_rng_states


def restore_checkpoint(
    checkpoint_path: str,
    model: JepaWorldModel,
    learner: JepaLearner,
    replay_buffer: ReplayBuffer,
    expected_mapping: Optional[OpaqueActionMapping] = None,
) -> Dict[str, Any]:
    """Restore checkpoint and perform strict validation checks."""
    store = CheckpointStore(os.path.dirname(checkpoint_path))
    ckpt = torch.load(checkpoint_path, map_location=learner.device, weights_only=False)

    # 1. Action permutation integrity check
    if "action_mapping" in ckpt and expected_mapping is not None:
        saved_perm = ckpt["action_mapping"]["permutation"]
        if saved_perm != expected_mapping.permutation:
            raise ValueError(f"Action permutation mismatch upon resume! Saved {saved_perm} vs Expected {expected_mapping.permutation}")

    # 2. Model weights
    model.load_state_dict(ckpt["model_state_dict"])
    
    # 3. Optimizer and scaler
    learner.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt and learner.scaler is not None:
        learner.scaler.load_state_dict(ckpt["scaler_state_dict"])
    learner.update_count = ckpt.get("learner_update_count", 0)

    # 4. Replay buffer
    if "replay_state" in ckpt:
        replay_buffer.set_state(ckpt["replay_state"])

    # 5. Restore RNG states
    if "rng_states" in ckpt:
        set_rng_states(ckpt["rng_states"])

    return ckpt
