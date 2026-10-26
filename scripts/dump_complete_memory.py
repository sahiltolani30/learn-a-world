#!/usr/bin/env python3
"""
scripts/dump_complete_memory.py — Full Memory Bank Auditor & Clean Optimizer
============================================================================

Extracts and audits EVERY single memory entry in the long-term memory bank.
1. Identifies all 1-to-1 causal relationships discovered.
2. Identifies all duplicate entries (e.g. 19 table placements, 35 wood harvests).
3. Identifies all unwanted junk data (passive hunger/thirst decay noise).
4. Optionally creates a clean, optimized memory bank with 0 duplicates and 0 junk:
   checkpoints/clean_optimal_memory_bank.pt

Usage:
    python scripts/dump_complete_memory.py
    python scripts/dump_complete_memory.py --save_clean
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from planning.semantic_memory import (
    CRAFTER_INVENTORY_KEYS,
    LongTermSemanticMemory,
    vector_to_inventory_dict,
)

CRAFTER_ACTION_NAMES = [
    "noop (0)", "move_left (1)", "move_right (2)", "move_up (3)", "move_down (4)",
    "do/harvest (5)", "sleep (6)", "place_stone (7)", "place_table (8)",
    "place_furnace (9)", "place_plant (10)", "make_wood_pickaxe (11)",
    "make_stone_pickaxe (12)", "make_iron_pickaxe (13)", "make_wood_sword (14)",
    "make_stone_sword (15)", "make_iron_sword (16)"
]


def resolve_bank_path(user_path: str) -> Path:
    candidates = [
        Path(user_path),
        REPO_ROOT / user_path,
        Path("/content/drive/MyDrive/learn-a-world") / user_path,
        Path("/content/learn-a-world") / user_path,
        REPO_ROOT / "checkpoints/tabula_rasa_memory_bank.pt",
        Path("/content/drive/MyDrive/learn-a-world/checkpoints/tabula_rasa_memory_bank.pt"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return Path(user_path)


def format_inventory(inv_vec: np.ndarray) -> str:
    res = []
    for i in range(12):
        val = int(round(inv_vec[i]))
        if val > 0:
            res.append(f"{CRAFTER_INVENTORY_KEYS[i]}={val}")
    res_str = ", ".join(res) if res else "empty"
    return f"[{res_str}]"


def format_delta(delta_vec: np.ndarray) -> str:
    changes = []
    for i, k in enumerate(CRAFTER_INVENTORY_KEYS):
        d = delta_vec[i]
        if abs(d) > 0.01:
            sign = "+" if d > 0 else ""
            changes.append(f"{k}: {sign}{int(round(d))}")
    return ", ".join(changes) if changes else "None (0.0)"


def is_passive_decay(delta_vec: np.ndarray) -> bool:
    """True if change is purely passive hunger/thirst/energy decay (-1 with no resource change)."""
    # Tools/resources (0..11)
    if np.any(np.abs(delta_vec[:12]) > 0.01):
        return False
    # Vital replenishment (> 0) is deliberate (eating/drinking/sleeping)
    if np.any(delta_vec[12:16] > 0.01):
        return False
    # Acute damage (< -1.0) is combat
    if delta_vec[12] < -1.0:
        return False
    # Passive hunger/thirst/energy drop of -1
    return bool(np.any(delta_vec[12:16] < -0.01))


def dump_and_audit(bank_path: Path, save_clean: bool = False):
    if not bank_path.exists():
        print(f"[-] Error: File not found: {bank_path}")
        return

    data = torch.load(bank_path, map_location="cpu")
    keys = data["keys"]
    inventories = data["inventories"].numpy()
    actions = data["actions"].numpy()
    returns = data["returns"].numpy()
    deltas = data["inventory_deltas"].numpy()
    names = data["outcome_names"]
    N = len(keys)

    print("=" * 80)
    print(f"      📋 COMPLETE MEMORY BANK INVENTORY & AUDIT ({N} TOTAL ENTRIES)")
    print(f"      File: {bank_path}")
    print("=" * 80)

    # 1. Group into 1-to-1 Causal Relationships
    # A pure relation is defined by: (Action, Resource State Before, Resource Delta)
    relations: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    passive_junk: List[int] = []
    mistakes: List[int] = []

    for i in range(N):
        act = int(actions[i])
        inv_str = format_inventory(inventories[i])
        delta_str = format_delta(deltas[i])
        ret = float(returns[i])
        name = names[i]

        # Check for passive clock tick noise
        if is_passive_decay(deltas[i]):
            passive_junk.append(i)
            continue

        # Check for mistakes (fruitless actions with G=0)
        if ret <= 0.01 and np.all(np.abs(deltas[i]) < 0.01):
            mistakes.append(i)
            continue

        # Genuine Causal Relation
        rel_key = (act, inv_str, delta_str)
        if rel_key not in relations:
            relations[rel_key] = {
                "action": act,
                "action_name": CRAFTER_ACTION_NAMES[act],
                "inventory_before": inv_str,
                "delta": delta_str,
                "tag": name,
                "max_return": ret,
                "occurrences": 1,
                "sample_indices": [i],
            }
        else:
            relations[rel_key]["occurrences"] += 1
            if ret > relations[rel_key]["max_return"]:
                relations[rel_key]["max_return"] = ret
                relations[rel_key]["tag"] = name
            relations[rel_key]["sample_indices"].append(i)

    # Print Summary of What Was Found
    print(f"\n[+] Total Raw Entries in Memory       : {N}")
    print(f"[+] Unique 1-to-1 Causal Relations    : {len(relations)} (Pillar 1: What works)")
    print(f"[+] Mistake Entries (Suppressed G=0.0): {len(mistakes)} (Pillar 2: What fails)")
    print(f"[+] Passive Vital Clock Ticks (Junk)  : {len(passive_junk)} (❌ UNWANTED NOISE)")

    # 2. Print All 1-to-1 Causal Relations
    print("\n" + "=" * 80)
    print(" 🌟 PART 1: THE UNIQUE 1-TO-1 CAUSAL RELATIONS (THE ONLY DATA AGENT NEEDS)")
    print("=" * 80)
    sorted_rels = sorted(relations.values(), key=lambda r: -r["max_return"])
    for idx, r in enumerate(sorted_rels, 1):
        print(f"\nRelation #{idx:02d}: {r['tag']} (Observed {r['occurrences']}x)")
        print(f"  • Action Pressed     : {r['action_name']}")
        print(f"  • Backpack Prereq    : {r['inventory_before']}")
        print(f"  • Physical Outcome   : {r['delta']}")
        print(f"  • Max Expected Return: G = {r['max_return']:.3f}")

    # 3. Print Duplicate Audit
    print("\n" + "=" * 80)
    print(" 🔄 PART 2: DUPLICATE DATA ANALYSIS")
    print("=" * 80)
    high_dupes = [r for r in sorted_rels if r["occurrences"] > 1]
    total_dupes = sum(r["occurrences"] - 1 for r in high_dupes)
    print(f"Found {total_dupes} duplicate entries across {len(high_dupes)} relations:")
    for r in high_dupes:
        print(f"  • {r['action_name']:<25} | In: {r['inventory_before']:<18} | Out: {r['delta']:<25} -> {r['occurrences']} copies (Only 1 needed!)")

    # 4. Print Mistakes Audit
    print("\n" + "=" * 80)
    print(" 🛑 PART 3: MISTAKE DATA ANALYSIS (SUPPRESSED ACTIONS)")
    print("=" * 80)
    mistake_groups: Dict[int, int] = {}
    for idx in mistakes:
        act = int(actions[idx])
        mistake_groups[act] = mistake_groups.get(act, 0) + 1

    print(f"Found {len(mistakes)} mistake entries across {len(mistake_groups)} actions:")
    for act, cnt in sorted(mistake_groups.items(), key=lambda x: -x[1]):
        act_name = CRAFTER_ACTION_NAMES[act]
        print(f"  • {act_name:<28} : {cnt:>2} failed attempts with G = 0.0 (Suppresses action when materials missing)")

    # 5. Purge and Save Clean Optimal Bank (if requested)
    if save_clean:
        print("\n" + "=" * 80)
        print(" 🧹 PART 4: PURGING UNWANTED NOISE & SAVING CLEAN OPTIMAL BANK")
        print("=" * 80)
        clean_bank = LongTermSemanticMemory(latent_dim=keys.shape[1], device="cpu")

        # A. Add 1 best prototype for each unique relation (max return, 0 duplicates)
        for r in sorted_rels:
            best_idx = r["sample_indices"][0]
            clean_bank.add_entry(
                key=keys[best_idx],
                inventory=inventories[best_idx],
                action=int(actions[best_idx]),
                inventory_delta=deltas[best_idx],
                realized_return=r["max_return"],
                outcome_name=r["tag"],
            )

        # B. Add 2 representative failure prototypes per mistake action (enough to suppress)
        for act in sorted(mistake_groups.keys()):
            act_mistakes = [idx for idx in mistakes if int(actions[idx]) == act]
            for best_idx in act_mistakes[:2]:
                clean_bank.add_entry(
                    key=keys[best_idx],
                    inventory=inventories[best_idx],
                    action=int(actions[best_idx]),
                    inventory_delta=deltas[best_idx],
                    realized_return=0.0,
                    outcome_name=names[best_idx],
                )

        clean_path = bank_path.parent / "clean_optimal_memory_bank.pt"
        clean_bank.save(clean_path)
        print(f"[+] Successfully purged {passive_junk.__len__()} passive noise entries!")
        print(f"[+] Successfully collapsed {total_dupes} duplicate entries into single best prototypes!")
        print(f"[+] Clean Optimal Bank Size: {len(clean_bank)} pure entries (down from {N})")
        print(f"[+] Saved clean memory bank to: {clean_path}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Full Memory Bank Inventory & Clean Purge")
    parser.add_argument("--bank", type=str, default="checkpoints/tabula_rasa_memory_bank.pt", help="Path to memory bank .pt file")
    parser.add_argument("--save_clean", action="store_true", help="Purge noise and save clean_optimal_memory_bank.pt")
    args = parser.parse_args()

    resolved = resolve_bank_path(args.bank)
    dump_and_audit(resolved, save_clean=args.save_clean)


if __name__ == "__main__":
    main()
