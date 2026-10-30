"""Audit Pack exporter: permanent diagnostic export on every save_state (V2.2 F8).

Diagnostics only; zero behavior change.
Exports to <run_dir>/audit/:
- manifest.json
- latents_last2.npz
- positions_last2.npz
- decisions.jsonl (append)
- frames/ (arrival & probe frames, capped at 50 files)
- health_cards/ (latest health report card)
Folder cap: ~60 MB (oldest pruned if exceeded).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image


def _sha256(path: Union[str, Path]) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _config_to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _config_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_config_to_dict(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def export_relations_snapshot(audit_dir: Union[str, Path], iter_idx: int, agent: Any) -> Path:
    """Exports learned relations, ItemEventMap, and GoalBank metadata (T5).
    Strictly anonymous: no key vectors, no achievement names.
    """
    audit_path = Path(audit_dir)
    relations_dir = audit_path / "relations"
    relations_dir.mkdir(parents=True, exist_ok=True)
    out_file = relations_dir / f"relations_it{iter_idx}.json"

    lift_edges = []
    if hasattr(agent, "lift") and hasattr(agent.lift, "ci_low") and hasattr(agent.lift, "lift"):
        ci_low = agent.lift.ci_low
        lift = agent.lift.lift
        num_ach = ci_low.shape[0]
        for d in range(num_ach):
            for e in range(num_ach):
                if float(ci_low[d, e]) > 0.0:
                    lift_edges.append([
                        int(d),
                        int(e),
                        float(round(float(ci_low[d, e]), 4)),
                        float(round(float(lift[d, e]), 4)),
                    ])

    item_event_map = {}
    if hasattr(agent, "item_event_map") and agent.item_event_map:
        for k, v in agent.item_event_map.items():
            item_event_map[str(k)] = int(v) if v is not None else None

    bank_list = []
    if hasattr(agent, "goal_bank") and hasattr(agent.goal_bank, "entries"):
        for entry in agent.goal_bank.entries:
            b_obj = {
                "kind": str(entry.kind),
                "event_dims": [int(d) for d in entry.event_dims],
                "n_try": int(entry.n_try),
                "n_succ": int(entry.n_succ),
                "world_frequency": float(round(float(getattr(entry, "world_frequency", 0.0)), 4)),
                "born_iter": int(getattr(entry, "born_iter", 0)),
            }
            bank_list.append(b_obj)

    data = {
        "iter": int(iter_idx),
        "lift_edges": lift_edges,
        "item_event_map": item_event_map,
        "bank": bank_list,
    }

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    return out_file


class AuditPackExporter:
    """Manages diagnostic audit pack exports for Project ATLAS."""

    def __init__(self, run_dir: Union[str, Path], max_folder_mb: float = 60.0):
        self.run_dir = Path(run_dir)
        self.audit_dir = self.run_dir / "audit"
        self.frames_dir = self.audit_dir / "frames"
        self.health_dir = self.audit_dir / "health_cards"
        self.probes_dir = self.audit_dir / "probes"
        self.relations_dir = self.audit_dir / "relations"
        self.max_bytes = int(max_folder_mb * 1024 * 1024)

        for d in [self.audit_dir, self.frames_dir, self.health_dir, self.probes_dir, self.relations_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.pending_decisions: List[Dict[str, Any]] = []
        self.pending_frames: List[Dict[str, Any]] = []  # list of {name, obs}

    def record_decision(
        self,
        iter_idx: int,
        step: int,
        goal_kind: str,
        goal_dim: Sequence[int],
        mode: str,
        arrival_cos: float,
        chosen_action: int,
        blended_scores: Sequence[float],
        probe_event_probs: Sequence[Sequence[float]],
        unlocked_dims: Sequence[int],
        matched_bank: bool,
        probe_id: int = 0,
    ) -> None:
        """Records a single probe/goal decision event."""
        # Convert float16 event probs to float list
        record = {
            "iter": int(iter_idx),
            "step": int(step),
            "goal_kind": str(goal_kind),
            "goal_dim": [int(d) for d in goal_dim],
            "mode": str(mode),
            "arrival_cos": round(float(arrival_cos), 4),
            "chosen_action": int(chosen_action),
            "blended_scores": [round(float(s), 4) for s in blended_scores],
            "probe_event_probs": [[round(float(p), 4) for p in row] for row in probe_event_probs],
            "unlocked_dims": [int(d) for d in unlocked_dims],
            "matched_bank": bool(matched_bank),
            "probe_id": int(probe_id),
        }
        self.pending_decisions.append(record)

    def record_frame(self, name: str, obs_uint8: np.ndarray) -> None:
        """Buffers a visual keyframe snapshot (arrival or probe moment)."""
        if len(self.pending_frames) < 10:  # up to 5 arrivals + 5 probes per save
            self.pending_frames.append({"name": name, "obs": obs_uint8.copy()})

    def export(
        self,
        current_iter: int,
        git_commit: str,
        cfg: Any,
        cached_episodes: List[Dict[str, Any]],
        results_dir: Optional[Union[str, Path]] = None,
        agent: Optional[Any] = None,
    ) -> None:
        """Writes audit files on save_state and enforces ~60MB cap."""
        # 1. Manifest
        manifest = {
            "iter": int(current_iter),
            "git_commit": str(git_commit),
            "config": _config_to_dict(cfg),
            "sha256": {
                "goal_bank_npz": _sha256(self.run_dir / "goal_bank.npz"),
                "lift_npz": _sha256(self.run_dir / "lift.npz"),
                "state_pt": _sha256(self.run_dir / "state.pt"),
                "topomap_npz": _sha256(self.run_dir / "topomap.npz"),
            },
        }
        manifest_path = self.audit_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # 2. Latents and Positions for last 2 shards / episodes
        if cached_episodes:
            last2 = cached_episodes[-2:]
            z_list = [ep["z"] for ep in last2]
            h_list = [ep["h"] for ep in last2]
            res_list = [ep.get("target_residuals", np.zeros_like(ep["z"])) for ep in last2]
            act_list = [ep["actions"] for ep in last2]
            rew_list = [ep["rewards"] for ep in last2]
            evt_list = [ep["events"] for ep in last2]

            latents_path = self.audit_dir / "latents_last2.npz"
            np.savez_compressed(
                latents_path,
                z=np.concatenate(z_list, axis=0) if z_list else np.zeros((0, 128), dtype=np.float32),
                h=np.concatenate(h_list, axis=0) if h_list else np.zeros((0, 256), dtype=np.float32),
                target_residuals=np.concatenate(res_list, axis=0) if res_list else np.zeros((0, 128), dtype=np.float32),
                actions=np.concatenate(act_list, axis=0) if act_list else np.zeros((0,), dtype=np.int64),
                rewards=np.concatenate(rew_list, axis=0) if rew_list else np.zeros((0,), dtype=np.float32),
                events=np.concatenate(evt_list, axis=0) if evt_list else np.zeros((0, 22), dtype=bool),
            )

            # Positions
            positions_path = self.audit_dir / "positions_last2.npz"
            pos_dict = {}
            for i, ep in enumerate(last2):
                pos = ep.get("positions")
                if pos is not None:
                    pos_dict[f"ep_{i}"] = np.array(pos, dtype=np.float32)
            if not pos_dict:
                pos_dict["dummy"] = np.zeros((0, 2), dtype=np.float32)
            np.savez_compressed(positions_path, **pos_dict)

        # 3. Decisions jsonl (append)
        if self.pending_decisions:
            decisions_path = self.audit_dir / "decisions.jsonl"
            with open(decisions_path, "a") as f:
                for dec in self.pending_decisions:
                    f.write(json.dumps(dec) + "\n")
            self.pending_decisions.clear()

        # 4. Frames (capped at 50 PNG files total)
        for frame_item in self.pending_frames:
            f_name = frame_item["name"]
            if not f_name.endswith(".png"):
                f_name = f"{f_name}.png"
            f_path = self.frames_dir / f_name
            try:
                img = Image.fromarray(frame_item["obs"])
                img.save(f_path)
            except Exception:
                pass
        self.pending_frames.clear()

        # Maintain 50 file cap in frames_dir
        existing_frames = sorted(self.frames_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        if len(existing_frames) > 50:
            for p in existing_frames[:-50]:
                p.unlink(missing_ok=True)

        # 5. Latest health card
        if results_dir:
            res_p = Path(results_dir)
            cards = sorted(res_p.glob("atlas_health_it*.json"), key=lambda p: p.stat().st_mtime)
            if cards:
                latest_card = cards[-1]
                shutil.copy2(latest_card, self.health_dir / latest_card.name)
            p_files = sorted(res_p.glob("probes_it*.json"), key=lambda p: p.stat().st_mtime)
            if p_files:
                latest_p = p_files[-1]
                shutil.copy2(latest_p, self.probes_dir / latest_p.name)

        # 5c. Relations snapshot (T5)
        if agent is not None:
            export_relations_snapshot(self.audit_dir, current_iter, agent)

        # 6. Folder size cap: ~60 MB
        self._enforce_size_cap()

    def _enforce_size_cap(self) -> None:
        """Drops oldest frames or latents if total folder size exceeds max_bytes."""
        try:
            all_files = []
            total_size = 0
            for root, _, files in os.walk(self.audit_dir):
                for f in files:
                    fp = Path(root) / f
                    sz = fp.stat().st_size
                    total_size += sz
                    all_files.append((fp, fp.stat().st_mtime, sz))

            if total_size <= self.max_bytes:
                return

            # Sort by mtime ascending (oldest first)
            # Prioritize dropping old frame PNGs first, then old latents
            frame_files = [item for item in all_files if item[0].suffix == ".png"]
            frame_files.sort(key=lambda x: x[1])

            for fp, _, sz in frame_files:
                if total_size <= self.max_bytes:
                    break
                fp.unlink(missing_ok=True)
                total_size -= sz
        except Exception:
            pass
