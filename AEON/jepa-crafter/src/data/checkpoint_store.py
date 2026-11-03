"""Atomic checkpoint serialization and restoration."""
import os
import shutil
import tempfile
from typing import Dict, Any, Optional
import torch

from src.utils.seeding import get_rng_states, set_rng_states


class CheckpointStore:
    """Handles atomic saving and validation of complete training checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        step: int,
        checkpoint_data: Dict[str, Any],
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> str:
        """Save checkpoint atomically: write to temp file, verify, then rename."""
        if filename is None:
            filename = f"checkpoint_{step:08d}.pt"
        final_path = os.path.join(self.checkpoint_dir, filename)

        # Ensure RNG state is included if not already
        if "rng_states" not in checkpoint_data:
            checkpoint_data["rng_states"] = get_rng_states()
        checkpoint_data["step"] = step

        # Write to temporary file in the same filesystem directory to ensure atomic rename
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=self.checkpoint_dir, suffix=".tmp")
        temp_path = temp_file.name
        temp_file.close()

        try:
            torch.save(checkpoint_data, temp_path)
            
            # Verify it can be loaded cleanly
            verify_load = torch.load(temp_path, map_location="cpu", weights_only=False)
            assert "step" in verify_load, "Corrupted checkpoint: missing 'step'"
            
            # Atomic rename
            shutil.move(temp_path, final_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"Failed to atomically save checkpoint at {final_path}: {e}")

        # Update latest pointer/copy
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        shutil.copyfile(final_path, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            shutil.copyfile(final_path, best_path)

        return final_path

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint from explicit path or latest."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def has_checkpoint(self) -> bool:
        """Check if a latest checkpoint exists."""
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        return os.path.exists(latest_path)
