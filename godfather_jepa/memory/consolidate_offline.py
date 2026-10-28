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
from godfather_jepa.utils.visual_decoder import decode_inventory_from_frame


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
                ep_obs = obs[ep_start : ep_end + 1]
                ep_acts = acts[ep_start : ep_end + 1]
                ep_rews = rews[ep_start : ep_end + 1]
                ep_events = events[ep_start : ep_end + 1] if events is not None else None

                store = OnlineEpisodicStore()

                craftable_items = [
                    "health", "food", "drink", "energy",
                    "sapling", "wood", "stone", "coal", "iron",
                    "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                    "wood_sword", "stone_sword", "iron_sword",
                ]

                for t_idx in range(len(ep_acts)):
                    a = int(ep_acts[t_idx])
                    r = float(ep_rews[t_idx])
                    curr_f = ep_obs[t_idx]
                    next_f = ep_obs[t_idx + 1] if t_idx < len(ep_acts) - 1 else curr_f
                    unlocked = None

                    if ep_events is not None:
                        for a_idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
                            if a_idx < ep_events.shape[1] and ep_events[t_idx, a_idx] > 0:
                                unlocked = ach_name
                                break

                    # Empirical visual inventory detection via exact font pattern matching
                    c1 = decode_inventory_from_frame(curr_f)
                    c2 = decode_inventory_from_frame(next_f)
                    item_deltas = {}
                    for item in c1:
                        diff = c2[item] - c1[item]
                        if diff != 0:
                            item_deltas[item] = diff

                    if unlocked:
                        if unlocked.startswith("collect_"):
                            r_name = unlocked[len("collect_"):]
                            if r_name in c1 and item_deltas.get(r_name, 0) <= 0:
                                item_deltas[r_name] = 1
                        elif unlocked.startswith("make_"):
                            t_name = unlocked[len("make_"):]
                            if t_name in c1 and item_deltas.get(t_name, 0) <= 0:
                                item_deltas[t_name] = 1

                    # Empirical displacement calculation
                    delta_p = np.zeros(2, dtype=np.float32)
                    if a in (1, 2, 3, 4):
                        w1 = curr_f[:49, :63].mean(-1).astype(float)
                        w2 = next_f[:49, :63].mean(-1).astype(float)
                        mask = np.ones((35, 49), dtype=np.float32)
                        mask[11:24, 18:31] = 0.0
                        im1 = w1[7:42, 7:56]
                        dz = (np.abs(im1 - w2[7:42, 7:56]) * mask).mean()
                        if a == 1:
                            dm = (np.abs(im1 - w2[7:42, 14:63]) * mask).mean()
                            nom = np.array([-1.0, 0.0], dtype=np.float32)
                        elif a == 2:
                            dm = (np.abs(im1 - w2[7:42, 0:49]) * mask).mean()
                            nom = np.array([1.0, 0.0], dtype=np.float32)
                        elif a == 3:
                            dm = (np.abs(im1 - w2[14:49, 7:56]) * mask).mean()
                            nom = np.array([0.0, -1.0], dtype=np.float32)
                        elif a == 4:
                            dm = (np.abs(im1 - w2[0:35, 7:56]) * mask).mean()
                            nom = np.array([0.0, 1.0], dtype=np.float32)

                        if not (dm >= 3.5 and dm > dz * 1.1):
                            delta_p = nom

                    z_t = curr_f.reshape(-1)[:512].astype(np.float32) / 255.0
                    z_next = next_f.reshape(-1)[:512].astype(np.float32) / 255.0

                    store.record_step(
                        step_idx=t_idx,
                        z_t=z_t,
                        action=a,
                        reward=r,
                        done=(t_idx == len(ep_acts) - 1),
                        z_next=z_next,
                        item_deltas=item_deltas,
                        unlocked_achievement=unlocked,
                        delta_p=delta_p,
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
