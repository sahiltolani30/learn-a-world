"""Dedicated, atomic storage for true-JEPA Colab runs.

The model owns ``learn-a-world/true_jepa`` and never reads another model's
checkpoint or replay.  Local runs fall back to ``runs/true_jepa``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class TrueJEPADrive:
    """Create and manage the true-JEPA Drive subfolder."""

    def __init__(self, root: str | Path | None = None, model_name: str = "true_jepa") -> None:
        if root is not None:
            base = Path(root)
        elif Path("/content/drive/MyDrive").exists():
            base = Path("/content/drive/MyDrive/learn-a-world")
        else:
            base = Path("runs")
        self.root = base / model_name
        self.checkpoints = self.root / "checkpoints"
        self.logs = self.root / "logs"
        self.shards = self.root / "shards"
        self.results = self.root / "results"
        for path in (self.checkpoints, self.logs, self.shards, self.results):
            path.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, payload: Dict[str, Any], filename: str = "state.pt") -> Path:
        target = self.checkpoints / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return target

    def load_checkpoint(self, filename: str = "state.pt") -> Optional[Dict[str, Any]]:
        target = self.checkpoints / filename
        if not target.exists():
            return None
        return torch.load(target, map_location="cpu", weights_only=False)

    def save_replay(self, replay: Any, filename: str = "replay.npz") -> Path:
        target = self.shards / filename
        temporary = self.shards / (filename + ".tmp")
        replay.save(str(temporary))
        generated = Path(str(temporary) + ".npz")
        os.replace(generated, target)
        return target

    def save_json(self, payload: Any, filename: str) -> Path:
        import json

        target = self.results / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, target)
        return target
