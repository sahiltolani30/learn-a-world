"""Tests validating checkpoint serialization and exact restoration."""
import os
import shutil
import tempfile
import numpy as np
import torch
from src.data.replay_buffer import ReplayBuffer
from src.data.checkpoint_store import CheckpointStore
from src.env.opaque_actions import OpaqueActionMapping


def test_checkpoint_atomic_save_and_resume():
    temp_dir = tempfile.mkdtemp()
    try:
        store = CheckpointStore(temp_dir)
        mapping = OpaqueActionMapping(num_actions=17, seed=777)
        
        # Populate a buffer
        buffer = ReplayBuffer(capacity=50)
        buffer.start_episode(np.full((64, 64, 3), 10, dtype=np.uint8), episode_id=1)
        for i in range(12):
            buffer.add(i % 17, float(i), np.full((64, 64, 3), 10 + i + 1, dtype=np.uint8), terminated=False, truncated=False)

        checkpoint_data = {
            "update_count": 42,
            "env_steps": 12,
            "replay_state": buffer.get_state(),
            "action_mapping": mapping.to_dict(),
            "dummy_model_weights": {"weight": torch.tensor([1.23, 4.56])},
        }

        saved_path = store.save(step=12, checkpoint_data=checkpoint_data)
        assert os.path.exists(saved_path)
        assert store.has_checkpoint()

        loaded = store.load()
        assert loaded["step"] == 12
        assert loaded["update_count"] == 42
        assert loaded["action_mapping"]["permutation"] == mapping.permutation
        assert torch.allclose(loaded["dummy_model_weights"]["weight"], torch.tensor([1.23, 4.56]))

        # Restore replay buffer and test equality
        buffer_restored = ReplayBuffer(capacity=50)
        buffer_restored.set_state(loaded["replay_state"])
        assert buffer_restored.size == buffer.size
        assert (buffer_restored.actions[:12] == buffer.actions[:12]).all()
        assert (buffer_restored.rewards[:12] == buffer.rewards[:12]).all()

    finally:
        shutil.rmtree(temp_dir)
