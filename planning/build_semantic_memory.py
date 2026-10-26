#!/usr/bin/env python3
"""
planning/build_semantic_memory.py — Autonomous Memory Extraction Tool
======================================================================

Extracts landmark breakthrough transitions (Delta e_t > 0) from the agent's
own self-play rollouts and base sequence shards.
Zero human-engineered rules: strictly encodes transitions the agent itself discovered.

Outputs:
  checkpoints/semantic_memory_bank.pt
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.semantic_memory import LongTermSemanticMemory, NUM_ACHIEVEMENTS
from scripts.screen_v3_50eps import _resolve_checkpoint


def extract_memories_from_shard(
    shard_path: Path,
    model: JEPAv3,
    memory_bank: LongTermSemanticMemory,
    device: torch.device,
    max_per_achievement: int = 15,
) -> int:
    """
    Scans a single .npz shard, finds breakthrough transitions where milestones unlocked,
    and encodes the observation into a visual concept key in memory.
    """
    if not shard_path.exists():
        return 0

    print(f"  [Extractor] Scanning shard: {shard_path.name}...")
    data = np.load(str(shard_path), allow_pickle=True)
    obs = data.get("obs")
    actions = data.get("actions")
    rewards = data.get("rewards")
    events = data.get("events")
    mask_before = data.get("mask_before")

    if obs is None or events is None or actions is None:
        return 0

    N = len(actions)
    if N == 0:
        return 0

    # Find indices where an achievement was unlocked
    milestone_indices = np.where(np.any(events > 0, axis=-1))[0]
    if len(milestone_indices) == 0:
        return 0

    ach_counts: Dict[str, int] = {}
    added = 0

    # Batch process milestone observations through frozen encoder
    batch_obs = []
    batch_metadata = []

    for idx in milestone_indices:
        ev = events[idx]
        unlocked_indices = np.where(ev > 0)[0]
        unlocked_names = [CRAFTER_ACHIEVEMENTS[i] for i in unlocked_indices if i < len(CRAFTER_ACHIEVEMENTS)]
        primary_name = unlocked_names[0] if unlocked_names else "milestone"

        # Cap per achievement type to avoid memory bias toward common events
        curr_count = ach_counts.get(primary_name, 0)
        if curr_count >= max_per_achievement:
            continue
        ach_counts[primary_name] = curr_count + 1

        img = obs[idx]  # [64, 64, 3] uint8
        batch_obs.append(img)
        batch_metadata.append((
            mask_before[idx] if mask_before is not None else np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8),
            int(actions[idx]),
            ev,
            float(rewards[idx]) if rewards is not None else 1.0,
            ", ".join(unlocked_names),
        ))

    if not batch_obs:
        return 0

    # Encode in one batch
    img_tensor = torch.from_numpy(np.array(batch_obs)).permute(0, 3, 1, 2).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        z_batch = model.encode(img_tensor, use_target=True)

    for i, (m_before, act, ev_step, rew, name) in enumerate(batch_metadata):
        z_i = z_batch[i]
        memory_bank.add_entry(
            key=z_i,
            preconditions=m_before,
            action=act,
            outcome=ev_step,
            realized_return=rew,
            outcome_name=name,
        )
        added += 1

    print(f"    -> Extracted {added} landmark transitions ({dict(ach_counts)})")
    return added


def build_memory_bank(
    checkpoint: str = "checkpoints/jepa_crafter_v3_t2_final.pt",
    drive_dir: Optional[str] = None,
    output_path: str = "checkpoints/semantic_memory_bank.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_per_achievement: int = 15,
) -> LongTermSemanticMemory:
    """Builds and saves the complete long-term semantic memory bank."""
    dev = torch.device(device)
    resolved_ckpt = _resolve_checkpoint(checkpoint, drive_dir=drive_dir)
    print("=" * 80)
    print("AUTONOMOUS SEMANTIC MEMORY BANK EXTRACTION")
    print(f"  Checkpoint: {resolved_ckpt}")
    print(f"  Output    : {output_path}")
    print(f"  Device    : {dev}")
    print("=" * 80)

    # 1. Load model for visual encoding
    model = JEPAv3(latent_dim=128, recurrent_dim=256, num_actions=17, action_emb_dim=32)
    model.load_checkpoint(resolved_ckpt, device=dev)
    model.to(dev)
    model.eval()

    memory_bank = LongTermSemanticMemory(latent_dim=128, device=dev)

    # 2. Collect candidate shards (prioritize the agent's recent self-play rollouts)
    shard_dirs = []
    if drive_dir:
        dp = Path(drive_dir)
        shard_dirs.extend([
            dp / "data" / "self_play_shards",
            dp / "data" / "sequences",
            dp / "data" / "combined_shards",
        ])
    shard_dirs.extend([
        REPO_ROOT / "data" / "self_play_shards",
        REPO_ROOT / "data" / "sequences",
    ])

    total_extracted = 0
    scanned_shards = 0
    t0 = time.time()

    for s_dir in shard_dirs:
        if not s_dir.exists():
            continue
        all_npz = list(s_dir.glob("*.npz"))
        final_stems = {p.stem.replace("_final", "") for p in all_npz if "_final" in p.name}
        shards = []
        for p in all_npz:
            base_stem = p.stem.rsplit("_", 1)[0] if "_" in p.stem else p.stem
            if base_stem in final_stems and "_final" not in p.name:
                continue
            shards.append(p)
        shards.sort()
        for sp in shards:
            scanned_shards += 1
            extracted = extract_memories_from_shard(
                shard_path=sp,
                model=model,
                memory_bank=memory_bank,
                device=dev,
                max_per_achievement=max_per_achievement,
            )
            total_extracted += extracted

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"Memory Extraction Complete: {len(memory_bank)} entries compiled from {scanned_shards} shards in {elapsed:.1f}s")
    
    # Summary of stored outcomes
    summary = {}
    for name in memory_bank.outcome_names:
        summary[name] = summary.get(name, 0) + 1
    print(f"Stored Milestones: {summary}")
    print("=" * 80)

    # Save to disk and Drive
    out_p = memory_bank.save(output_path)
    print(f"[Memory Bank] Successfully saved to: {out_p}")

    if drive_dir and Path(drive_dir).exists():
        drive_out = Path(drive_dir) / "checkpoints" / out_p.name
        drive_out.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(out_p), str(drive_out))
        print(f"[Drive] Mirrored memory bank to: {drive_out}")

    return memory_bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Autonomous Semantic Memory Bank from Experience Shards.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_t2_final.pt")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--output_path", type=str, default="checkpoints/semantic_memory_bank.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_per_achievement", type=int, default=15)
    args = parser.parse_args()

    build_memory_bank(
        checkpoint=args.checkpoint,
        drive_dir=args.drive_dir,
        output_path=args.output_path,
        device=args.device,
        max_per_achievement=args.max_per_achievement,
    )
