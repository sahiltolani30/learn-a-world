"""Atomic runtime checkpoint helpers for P0/P1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch


def save_atomic(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a payload and atomically replace the destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load a MOSAIC runtime payload without compatibility fallbacks."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "mosaic_runtime_v1":
        raise ValueError("checkpoint is not a MOSAIC runtime checkpoint")
    return payload
