"""Phase 1.5 dataset discovery and deduplicated manifest handling.

Phase 1.5 is a data-preparation gate, not a new model.  Drive contains
multiple copies of cumulative snapshots and, in some cases, an unlabeled copy
of a transition shard alongside the state-delta-labeled copy.  This module
creates a reproducible manifest that prefers:

1. ``*_final.npz`` over numbered cumulative snapshots;
2. state-delta-labeled data over an unlabeled duplicate of the same transitions;
3. one copy of byte-identical shards.

The manifest records source/event statistics so demonstration-assisted and
strict/self-play data remain auditable.  Inventory/state vectors remain
training targets only; this module never changes strict inference inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from eval.metrics import CRAFTER_ACHIEVEMENTS


_NUMBERED_RE = re.compile(r"^(?P<prefix>.+)_(?P<number>\d+)$")


def _family_key(path: Path) -> Tuple[str, bool]:
    """Return filename family and whether this is a cumulative final shard."""
    stem = path.stem
    if stem.endswith("_final"):
        return stem[: -len("_final")], True
    match = _NUMBERED_RE.match(stem)
    if match:
        return match.group("prefix"), False
    return stem, False


def _update_hash(hasher: "hashlib._Hash", value: np.ndarray) -> None:
    arr = np.ascontiguousarray(value)
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(repr(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())


def _file_and_core_hashes(path: Path, data: Any) -> Tuple[str, str]:
    """Hash the archive and transition identity, excluding optional labels."""
    file_hash = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            file_hash.update(chunk)

    core_hash = hashlib.sha256()
    for key in ("obs", "actions", "rewards", "dones", "events", "mask_before"):
        if key in data:
            _update_hash(core_hash, data[key])
    return file_hash.hexdigest(), core_hash.hexdigest()


def _source_names(data: Any) -> List[str]:
    if "sources" not in data:
        return ["unknown"]
    values = np.asarray(data["sources"]).reshape(-1)
    names = sorted({str(v) for v in values.tolist()})
    return names or ["unknown"]


def inspect_shard(path: Union[str, Path]) -> Dict[str, Any]:
    """Read one shard's metadata and causal-event counts."""
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        if "obs" not in data or "actions" not in data:
            raise ValueError(f"Not a sequence shard: {p}")
        transitions = int(data["actions"].shape[0])
        if "episode_lengths" in data:
            episodes = int(len(data["episode_lengths"]))
        elif "dones" in data:
            dones = data["dones"]
            done_indices = np.where(dones)[0]
            episodes = max(1, len(done_indices))
        else:
            episodes = 1
        events = np.asarray(data["events"]) if "events" in data else np.zeros((transitions, len(CRAFTER_ACHIEVEMENTS)), dtype=np.uint8)
        has_delta = "state_delta" in data and data["state_delta"].ndim == 2
        valid_count = int(np.asarray(data["state_delta_valid"]).sum()) if "state_delta_valid" in data else 0
        event_counts = {
            name: int(events[:, idx].sum())
            for idx, name in enumerate(CRAFTER_ACHIEVEMENTS)
            if idx < events.shape[1]
        }
        file_hash, core_hash = _file_and_core_hashes(p, data)
        return {
            "path": str(p.resolve()),
            "filename": p.name,
            "family": _family_key(p)[0],
            "is_final": _family_key(p)[1],
            "transitions": transitions,
            "episodes": episodes,
            "sources": _source_names(data),
            "has_state_delta": bool(has_delta),
            "valid_state_delta_count": valid_count,
            "event_counts": event_counts,
            "file_sha256": file_hash,
            "core_sha256": core_hash,
        }


def _candidate_paths(roots: Sequence[Union[str, Path]]) -> List[Path]:
    found: Dict[str, Path] = {}
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix == ".npz":
            found[str(root.resolve())] = root
            continue
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        for path in root.rglob("*.npz"):
            found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: str(p))


def _choose_better(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Choose the more useful copy of identical transition content."""
    a_score = (
        int(a["has_state_delta"]),
        int(a["valid_state_delta_count"]),
        int("phase1_transitions" in a["path"]),
        int(a["transitions"]),
        str(a["path"]),
    )
    b_score = (
        int(b["has_state_delta"]),
        int(b["valid_state_delta_count"]),
        int("phase1_transitions" in b["path"]),
        int(b["transitions"]),
        str(b["path"]),
    )
    return a if a_score >= b_score else b


def discover_active_shards(
    roots: Sequence[Union[str, Path]],
    output_manifest: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Discover and deduplicate active shards across one or more directories."""
    paths = _candidate_paths(roots)
    if not paths:
        raise FileNotFoundError(f"No .npz shards found under: {list(map(str, roots))}")
    records = [inspect_shard(path) for path in paths]
    excluded: List[Dict[str, Any]] = []

    # A final cumulative shard supersedes all numbered snapshots in the same
    # family, even when the copies live in different Drive directories.
    final_families = {r["family"] for r in records if r["is_final"]}
    active: List[Dict[str, Any]] = []
    for record in records:
        if not record["is_final"] and record["family"] in final_families:
            excluded.append({"path": record["path"], "reason": "superseded_by_final"})
        else:
            active.append(record)

    # Keep one final copy per family. Prefer the largest/latest final copy.
    by_final_family: Dict[str, Dict[str, Any]] = {}
    remainder: List[Dict[str, Any]] = []
    for record in active:
        if record["is_final"]:
            previous = by_final_family.get(record["family"])
            if previous is None or (
                record["transitions"], record["valid_state_delta_count"], record["path"]
            ) > (
                previous["transitions"], previous["valid_state_delta_count"], previous["path"]
            ):
                if previous is not None:
                    excluded.append({"path": previous["path"], "reason": "duplicate_final_family"})
                by_final_family[record["family"]] = record
            else:
                excluded.append({"path": record["path"], "reason": "duplicate_final_family"})
        else:
            remainder.append(record)
    active = remainder + list(by_final_family.values())

    # Exact copies and copies with identical transition content are collapsed.
    # The latter is important when a labeled phase1 shard and an unlabeled copy
    # are stored in separate directories.
    by_core: Dict[str, Dict[str, Any]] = {}
    for record in sorted(active, key=lambda r: r["path"]):
        previous = by_core.get(record["core_sha256"])
        if previous is None:
            by_core[record["core_sha256"]] = record
        else:
            winner = _choose_better(previous, record)
            loser = record if winner is previous else previous
            excluded.append({"path": loser["path"], "reason": "duplicate_transition_content"})
            by_core[record["core_sha256"]] = winner
    selected = sorted(by_core.values(), key=lambda r: r["path"])

    totals = {
        "transitions": int(sum(r["transitions"] for r in selected)),
        "episodes": int(sum(r["episodes"] for r in selected)),
        "valid_state_delta_transitions": int(sum(r["valid_state_delta_count"] for r in selected)),
        "event_counts": {
            name: int(sum(r["event_counts"].get(name, 0) for r in selected))
            for name in CRAFTER_ACHIEVEMENTS
        },
    }
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "phase": "phase1.5",
        "roots": [str(Path(root).resolve()) for root in roots],
        "selected_shards": selected,
        "excluded_shards": excluded,
        "totals": totals,
        "state_delta_labels_are_training_targets_only": True,
        "strict_inference_uses_inventory_input": False,
    }
    if output_manifest is not None:
        output = Path(output_manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported Phase 1.5 manifest schema: {manifest.get('schema_version')}")
    return manifest


def selected_paths_from_manifest(manifest_path: Union[str, Path]) -> List[Path]:
    manifest = load_manifest(manifest_path)
    paths = [Path(item["path"]) for item in manifest.get("selected_shards", [])]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing shards: {missing[:5]}")
    return paths
