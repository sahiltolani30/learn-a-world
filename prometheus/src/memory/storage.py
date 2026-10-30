"""
src/memory/storage.py

CheckpointManager for Atomic Model & Memory Persistence in JEPA World Model.

Fulfills requirements:
- R2: Cross-episode skill persistence and checkpoint resumption.
- R5: Atomic, preemption-resilient checkpoint serialization to local disk
      or Google Drive mount (/content/drive/MyDrive/...).
- Multi-component serialization: network weights (encoder, predictor, target, policy, critic),
  replay buffer state, running Welford normalization statistics, and training metrics.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import os
import glob
import re
import uuid
import shutil
import gc
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages atomic checkpoint saving, loading, and directory maintenance
    for local training and Google Drive environments.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        keep_last_n: Optional[int] = 5,
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory where checkpoint files are saved/loaded.
            keep_last_n: Number of recent step checkpoints to retain (None for unlimited).
        """
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.keep_last_n = keep_last_n
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_google_drive_mounted() -> bool:
        """Check if Google Drive mount path exists."""
        return os.path.exists("/content/drive/MyDrive")

    def _extract_state(self, obj: Any) -> Any:
        """Extract state_dict if available; otherwise return raw object."""
        if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
            return obj.state_dict()
        return obj

    @staticmethod
    def save_atomic_checkpoint(payload: Dict[str, Any], filepath: Union[str, Path]) -> str:
        """
        Rule 2: Preemption-Resilient Atomic Checkpointing.
        Saves directly to a .pt.tmp file first, flushes, and performs atomic rename via os.replace.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".pt.tmp")
        try:
            torch.save(payload, str(tmp_path))
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed to atomically write checkpoint to {path}: {e}") from e
        return str(path.resolve())

    def save(
        self,
        filepath: Union[str, Path],
        components: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        best_filepath: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Atomically save components and metadata to disk.
        
        Uses a two-phase write pattern (write to temp file in same directory,
        then atomic os.replace) to ensure zero checkpoint corruption even if
        interrupted by preemption, crash, or Drive disconnection.
        
        Args:
            filepath: Target checkpoint path.
            components: Dict of models, optimizers, replay buffers, or state dicts.
            metadata: Optional dictionary with training step, episode, metrics, etc.
            is_best: If True, also saves copy to best_filepath.
            best_filepath: Target path for the best model (defaults to best_checkpoint.pt).
            
        Returns:
            Resolved absolute path string of saved checkpoint.
        """
        target_path = Path(filepath)
        if not target_path.is_absolute():
            target_path = self.checkpoint_dir / target_path

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Assemble checkpoint payload
        payload: Dict[str, Any] = {
            "components": {k: self._extract_state(v) for k, v in components.items()},
            "metadata": metadata or {},
        }

        # Write atomically via temp file in same directory (guarantees same filesystem)
        temp_filename = f"{target_path.name}.tmp.{uuid.uuid4().hex}"
        temp_path = target_path.parent / temp_filename

        try:
            torch.save(payload, str(temp_path))
            # Flush to disk and atomic rename
            os.replace(str(temp_path), str(target_path))
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise IOError(f"Failed to atomically write checkpoint to {target_path}: {e}") from e

        # Handle best checkpoint copy if requested via efficient file copy (zero RAM overhead)
        if is_best:
            if best_filepath is None:
                b_path = target_path.parent / "best_checkpoint.pt"
            else:
                b_path = Path(best_filepath)
                if not b_path.is_absolute():
                    b_path = self.checkpoint_dir / b_path

            b_temp = b_path.parent / f"{b_path.name}.tmp.{uuid.uuid4().hex}"
            try:
                shutil.copyfile(str(target_path), str(b_temp))
                os.replace(str(b_temp), str(b_path))
            except Exception as e:
                if b_temp.exists():
                    try:
                        b_temp.unlink()
                    except OSError:
                        pass
                raise IOError(f"Failed to write best checkpoint to {b_path}: {e}") from e

        # Explicit RAM cleanup to prevent Google Colab OOM
        del payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if hasattr(os, "sync"):
                os.sync()
        except Exception:
            pass

        # Prune older checkpoints if configured
        if self.keep_last_n is not None and self.keep_last_n > 0:
            self.prune(self.keep_last_n)

        return str(target_path.resolve())

    def load(
        self,
        filepath: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """
        Load raw checkpoint dictionary from disk.
        
        Args:
            filepath: Path to checkpoint file.
            device: PyTorch device mapping.
            
        Returns:
            Dictionary containing 'components' and 'metadata'.
        """
        target_path = Path(filepath)
        if not target_path.is_absolute():
            target_path = self.checkpoint_dir / target_path

        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {target_path}")

        try:
            checkpoint = torch.load(str(target_path), map_location=device, weights_only=False)
        except Exception as e:
            raise IOError(f"Failed to load checkpoint file at {target_path}: {e}") from e

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format at {target_path}: expected dict, got {type(checkpoint)}")

        # Normalize format if saved directly without outer components wrapper
        if "components" not in checkpoint:
            checkpoint = {
                "components": checkpoint,
                "metadata": {},
            }

        return checkpoint

    def restore(
        self,
        filepath: Union[str, Path],
        components: Dict[str, Any],
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state into provided live component instances.
        
        Args:
            filepath: Path to checkpoint file.
            components: Dict of live instances (e.g. {'encoder': enc, 'buffer': buf}).
            device: Device to map tensors to.
            strict: If True, raises error if a component's key is missing in checkpoint.
            
        Returns:
            Loaded metadata dictionary.
        """
        checkpoint = self.load(filepath, device=device)
        comp_states = checkpoint.get("components", {})

        for name, comp in components.items():
            if name not in comp_states:
                if strict:
                    raise KeyError(
                        f"Component '{name}' requested for restoration, but not found in checkpoint. "
                        f"Available keys: {list(comp_states.keys())}"
                    )
                continue

            state = comp_states[name]
            if hasattr(comp, "load_state_dict") and callable(getattr(comp, "load_state_dict")):
                if isinstance(comp, nn.Module):
                    comp.load_state_dict(state, strict=strict)
                else:
                    comp.load_state_dict(state)
            else:
                raise TypeError(
                    f"Component '{name}' does not implement load_state_dict()."
                )

        metadata = checkpoint.get("metadata", {})
        del checkpoint, comp_states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metadata

    def find_latest_checkpoint(
        self,
        pattern: str = "checkpoint_*.pt",
    ) -> Optional[str]:
        """
        Find the most recent checkpoint matching pattern by modification time or step index.
        
        Args:
            pattern: Glob pattern to search for in checkpoint_dir.
            
        Returns:
            Absolute path string of latest checkpoint, or None if none found.
        """
        # Rule 3: Prioritize active verified checkpoint 'state.pt' if present
        state_file = self.checkpoint_dir / "state.pt"
        if state_file.exists():
            return str(state_file.resolve())

        search_path = str(self.checkpoint_dir / pattern)
        files = glob.glob(search_path)
        if not files:
            return None

        # Try to sort numerically by step number if filename contains _step_(\d+)
        step_pattern = re.compile(r"_step_(\d+)")
        numbered_files = []
        for f in files:
            match = step_pattern.search(f)
            if match:
                numbered_files.append((int(match.group(1)), f))

        if numbered_files:
            numbered_files.sort(key=lambda x: x[0])
            return str(Path(numbered_files[-1][1]).resolve())

        # Fall back to file modification time
        files.sort(key=os.path.getmtime)
        return str(Path(files[-1]).resolve())

    def prune(
        self,
        keep_last_n: int,
        pattern: str = "checkpoint_step_*.pt",
    ) -> List[str]:
        """
        Prune older checkpoints, keeping only the most recent keep_last_n files.
        Never removes 'best_checkpoint.pt' or 'checkpoint_latest.pt'.
        
        Args:
            keep_last_n: Maximum number of step checkpoints to preserve.
            pattern: Glob pattern for pruning candidates.
            
        Returns:
            List of pruned file paths.
        """
        if keep_last_n <= 0:
            return []

        search_path = str(self.checkpoint_dir / pattern)
        files = glob.glob(search_path)

        # Filter out protected files
        candidates = [
            f for f in files
            if not f.endswith("best_checkpoint.pt") and not f.endswith("checkpoint_latest.pt")
        ]

        if len(candidates) <= keep_last_n:
            return []

        # Sort by step number if possible, else modification time
        step_pattern = re.compile(r"_step_(\d+)")
        numbered = []
        for f in candidates:
            match = step_pattern.search(f)
            if match:
                numbered.append((int(match.group(1)), f))

        if len(numbered) == len(candidates):
            numbered.sort(key=lambda x: x[0])
            sorted_files = [f for _, f in numbered]
        else:
            candidates.sort(key=os.path.getmtime)
            sorted_files = candidates

        to_remove = sorted_files[:-keep_last_n]
        removed = []
        for f in to_remove:
            try:
                os.remove(f)
                removed.append(f)
            except OSError:
                pass

        return removed
