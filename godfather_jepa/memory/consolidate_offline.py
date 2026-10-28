"""
Standalone Offline Memory Consolidation Script for Godfather JEPA.

Ingests offline sequence shards (e.g. from Google Drive data/sequences/),
extracts empirical transitions, enforces the Four Iron Rules of Causal Consolidation,
and builds the persistent SemanticConsolidatedBank before or alongside training.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from godfather_jepa.memory.online_store import OnlineEpisodicStore
from godfather_jepa.memory.semantic_bank import SemanticConsolidatedBank
from godfather_jepa.memory.consolidation import OfflineConsolidationEngine
from godfather_jepa.metrics.crafter_score import CRAFTER_ACHIEVEMENTS


def consolidate_shards(
    shard_dir: str,
    output_path: str = "checkpoints/godfather_memory_bank.pt",
    max_episodes: int = 500,
) -> Dict[str, Any]:
    """
    Ingests sequence shards and consolidates them into the semantic bank.
    """
    bank = SemanticConsolidatedBank(persistence_path=output_path)
    engine = OfflineConsolidationEngine(semantic_bank=bank)

    shard_files = sorted(Path(shard_dir).glob("*.npz"))
    if not shard_files:
        print(f"[Consolidation] Warning: No .npz files found in {shard_dir}")
        return {"total_cards": len(bank.cards), "episodes_processed": 0}

    print(f"[Consolidation] Found {len(shard_files)} shard files in {shard_dir}")
    total_episodes = 0

    for sf in shard_files:
        try:
            data = np.load(str(sf), allow_pickle=True)
            if "obs" not in data or "dones" not in data:
                continue

            obs = data["obs"]
            acts = data["actions"] if "actions" in data else np.zeros(len(obs), dtype=np.int64)
            rews = data["rewards"] if "rewards" in data else np.zeros(len(obs), dtype=np.float32)
            dones = data["dones"]
            events = data["events"] if "events" in data else None

            n_trans = len(obs)
            if n_trans == 0:
                continue

            ep_start = 0
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            for ep_end in ep_indices:
                ep_acts = acts[ep_start : ep_end + 1]
                ep_rews = rews[ep_start : ep_end + 1]
                ep_events = events[ep_start : ep_end + 1] if events is not None else None

                store = OnlineEpisodicStore()
                z_dummy = np.zeros(512, dtype=np.float32)

                for t_idx in range(len(ep_acts)):
                    a = int(ep_acts[t_idx])
                    r = float(ep_rews[t_idx])
                    unlocked = None

                    if ep_events is not None:
                        for a_idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
                            if a_idx < ep_events.shape[1] and ep_events[t_idx, a_idx] > 0:
                                unlocked = ach_name
                                break

                    item_deltas = {}
                    if a == 5 and r > 0 and unlocked == "collect_wood":
                        item_deltas["wood"] = 1
                    elif a == 8:
                        item_deltas["wood"] = -2
                    elif a == 11:
                        item_deltas["wood"] = -1
                    elif a == 5 and r > 0 and unlocked == "collect_stone":
                        item_deltas["stone"] = 1

                    store.record_step(
                        step_idx=t_idx,
                        z_t=z_dummy,
                        action=a,
                        reward=r,
                        done=(t_idx == len(ep_acts) - 1),
                        z_next=z_dummy,
                        item_deltas=item_deltas,
                        unlocked_achievement=unlocked,
                    )

                engine.consolidate_episode(store)
                total_episodes += 1
                ep_start = ep_end + 1

                if total_episodes >= max_episodes:
                    break

            if total_episodes >= max_episodes:
                break
        except Exception as e:
            continue

    # Save consolidated bank to disk and JSON
    bank.save(output_path)
    print(f"[Consolidation] Consolidated {total_episodes} episodes into {output_path}")
    print(f"[Consolidation] Active Causal Cards in Memory Bank: {len(bank.cards)}")
    for name, card in bank.cards.items():
        print(f"  - Card '{name}': Action {card.action}, Successes: {card.success_count}, Failures: {card.failure_count}")

    return {"total_cards": len(bank.cards), "episodes_processed": total_episodes}


def main():
    parser = argparse.ArgumentParser(description="Godfather JEPA Offline Memory Consolidation")
    parser.add_argument("--shard-dir", type=str, default="/content/drive/MyDrive/learn-a-world/data/sequences/")
    parser.add_argument("--out-bank", type=str, default="checkpoints/godfather_memory_bank.pt")
    parser.add_argument("--max-episodes", type=int, default=500)
    args = parser.parse_args()

    consolidate_shards(
        shard_dir=args.shard_dir,
        output_path=args.out_bank,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
