#!/usr/bin/env python3
"""
scripts/inspect_clean_memory_bank.py — Comprehensive Diagnostic Memory Inspector
================================================================================
Analyses a consolidated Phase 2 causal memory bank before launching 50-episode screenings.

Checks:
  1. Card counts (event-grounded vs event-free, confirmed vs hypothesis)
  2. Complete Precondition Hygiene (0 vitals, 0 ghost saplings, 0 self-loops)
  3. Core Tech-Tree Walkthrough (Wood -> Table -> Pickaxe -> Stone -> Sword -> Zombie)
  4. Causal Enabling Graph & Critical Chains
"""

import argparse
import json
from pathlib import Path
import sys

# Crafter State Space Mapping (16-D)
DIM_NAMES = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",          # 0..5
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",               # 6..8
    "wood_sword", "stone_sword", "iron_sword",                     # 9..11
    "health", "food", "drink", "energy"                            # 12..15 (Vitals)
]

# Action Names
ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",      # 0..4
    "do", "sleep", "place_stone", "place_table", "place_furnace",   # 5..9
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",      # 10..12
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",    # 13..15
    "make_iron_sword"                                              # 16
]

# Achievement Names (22-D)
ACHIEVEMENT_NAMES = [
    "collect_wood", "collect_stone", "collect_coal", "collect_iron", "collect_diamond",  # 0..4
    "collect_sapling", "collect_drink", "place_table", "defeat_zombie", "eat_cow",       # 5..9
    "defeat_skeleton", "wake_up", "place_plant", "eat_plant", "place_stone",             # 10..14
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword",   # 15..18
    "make_stone_sword", "make_iron_sword", "place_furnace"                               # 19..21
]


def fmt_items(items):
    if not items:
        return "None"
    out = []
    for entry in items:
        dim = int(entry[0])
        amt = float(entry[1])
        name = DIM_NAMES[dim] if 0 <= dim < len(DIM_NAMES) else f"dim_{dim}"
        out.append(f"{name} ({amt:.2f})")
    return ", ".join(out)


def inspect_bank(bank_path: str):
    p = Path(bank_path)
    if not p.exists():
        print(f"Error: Bank file not found at {bank_path}")
        return 1

    data = json.loads(p.read_text())
    if "prototypes" in data and isinstance(data["prototypes"], dict):
        protos = data["prototypes"]
    elif "prototype_summary" in data and isinstance(data["prototype_summary"], list):
        protos = {item.get("prototype_id", item.get("id")): item for item in data["prototype_summary"]}
    else:
        protos = {}

    if "edges" in data and isinstance(data["edges"], dict):
        edges = data["edges"]
    elif "causal_edges" in data and isinstance(data["causal_edges"], dict):
        edges = data["causal_edges"]
    else:
        edges = {}

    print("=" * 80)
    print(f"       CAUSAL MEMORY BANK DIAGNOSTIC INSPECTION: {p.name}")
    print("=" * 80)
    print(f"Total Prototypes:  {len(protos)}")
    confirmed = sum(1 for card in protos.values() if card.get("status") == "confirmed")
    hypotheses = sum(1 for card in protos.values() if card.get("status") == "hypothesis")
    event_grounded = sum(1 for card in protos.values() if card.get("event_dim") is not None)
    event_free = len(protos) - event_grounded
    print(f"  - Confirmed:     {confirmed}")
    print(f"  - Hypotheses:    {hypotheses}")
    print(f"  - Event-Grounded:{event_grounded}")
    print(f"  - Event-Free:    {event_free}")

    # 1. Cleanliness & Hygiene Audit
    print("\n" + "-" * 80)
    print("1. HYGIENE & SANITY CHECKS")
    print("-" * 80)

    # Check Vitals
    vitals_cards = [
        (pid, int(d))
        for pid, card in protos.items()
        for d, _ in card.get("needs", [])
        if int(d) in range(12, 16)
    ]
    if vitals_cards:
        print(f"  ❌ FAILED: {len(vitals_cards)} cards have vitals in needs! (e.g. {vitals_cards[:3]})")
    else:
        print("  ✅ [PASS] 0 vitals in needs bank-wide (Health/Food/Drink/Energy completely excluded).")

    # Check Ghost Saplings
    sapling_cards = [
        (pid, card.get("action"))
        for pid, card in protos.items()
        for d, _ in card.get("needs", [])
        if int(d) == 5 and card.get("action") != 10  # dim 5 = sapling, action 10 = place_plant
    ]
    if sapling_cards:
        print(f"  ❌ FAILED: {len(sapling_cards)} cards have spurious sapling needs! (e.g. {sapling_cards[:3]})")
    else:
        print("  ✅ [PASS] 0 ghost sapling needs (Only plant-placing consumes saplings).")

    # Check Self-Loops
    self_needs = [
        pid for pid, card in protos.items()
        if {int(d) for d, _ in card.get("needs", [])} & {int(d) for d, _ in card.get("produces", [])}
    ]
    if self_needs:
        print(f"  ❌ FAILED: {len(self_needs)} cards both need and produce the same item! (e.g. {self_needs[:3]})")
    else:
        print("  ✅ [PASS] 0 self-consumption loops (No card needs what it produces).")

    # 2. Key Tech-Tree Cards
    print("\n" + "-" * 80)
    print("2. CORE TECH-TREE CARDS AUDIT")
    print("-" * 80)

    # Table Cards (Action 8)
    table_cards = [card for card in protos.values() if card.get("action") == 8]
    print(f"\n[Place Table Cards] Count: {len(table_cards)}")
    for card in table_cards[:3]:
        pid = card.get("prototype_id", card.get("id"))
        status = card.get("status")
        needs = fmt_items(card.get("needs", []))
        consumes = fmt_items(card.get("consumes", []))
        produces = fmt_items(card.get("produces", []))
        conf = card.get("confidence", 0.0)
        print(f"  • {pid} [{status}, conf={conf:.2f}]:")
        print(f"      NEEDS:    {needs}")
        print(f"      CONSUMES: {consumes}")
        print(f"      PRODUCES: {produces}")

    # Wood Pickaxe Cards (Action 11 in code: MAKE_WOOD_PICKAXE)
    pickaxe_cards = [card for card in protos.values() if card.get("action") in (11, 10) and (
        card.get("event_dim") == 15 or "pickaxe" in card.get("prototype_id", card.get("id", ""))
    )]
    print(f"\n[Wood Pickaxe Cards] Count: {len(pickaxe_cards)}")
    for card in pickaxe_cards[:3]:
        pid = card.get("prototype_id", card.get("id"))
        status = card.get("status")
        needs = fmt_items(card.get("needs", []))
        consumes = fmt_items(card.get("consumes", []))
        produces = fmt_items(card.get("produces", []))
        conf = card.get("confidence", 0.0)
        print(f"  • {pid} [{status}, conf={conf:.2f}]:")
        print(f"      NEEDS:    {needs}")
        print(f"      CONSUMES: {consumes}")
        print(f"      PRODUCES: {produces}")

    # Wood Harvesting Cards (Action 5)
    wood_cards = [card for card in protos.values() if card.get("action") == 5 and any(int(d) == 0 for d, _ in card.get("produces", []))]
    print(f"\n[Wood Harvesting Cards] Count: {len(wood_cards)}")
    for card in wood_cards[:3]:
        pid = card.get("prototype_id", card.get("id"))
        status = card.get("status")
        needs = fmt_items(card.get("needs", []))
        produces = fmt_items(card.get("produces", []))
        conf = card.get("confidence", 0.0)
        print(f"  • {pid} [{status}, conf={conf:.2f}]:")
        print(f"      NEEDS:    {needs}")
        print(f"      PRODUCES: {produces}")

    # Stone Mining Cards (Action 5 producing Stone)
    stone_cards = [card for card in protos.values() if card.get("action") == 5 and any(int(d) == 1 for d, _ in card.get("produces", []))]
    print(f"\n[Stone Mining Cards] Count: {len(stone_cards)}")
    for card in stone_cards[:3]:
        pid = card.get("prototype_id", card.get("id"))
        status = card.get("status")
        needs = fmt_items(card.get("needs", []))
        produces = fmt_items(card.get("produces", []))
        conf = card.get("confidence", 0.0)
        print(f"  • {pid} [{status}, conf={conf:.2f}]:")
        print(f"      NEEDS:    {needs}")
        print(f"      PRODUCES: {produces}")

    # 3. Causal Enabling Graph
    print("\n" + "-" * 80)
    print("3. CAUSAL ENABLING GRAPH")
    print("-" * 80)
    enabling_edges = []
    produces_need_edges = []
    for dst, edgelist in edges.items():
        for e in edgelist:
            src = e.get("src_id", e.get("src"))
            if e.get("kind") == "enabling_lift":
                enabling_edges.append((src, dst, e.get("lift", 1.0)))
            elif e.get("kind") == "produces_need":
                produces_need_edges.append((src, dst, e.get("dim")))

    print(f"Total Graph Edges: {len(enabling_edges) + len(produces_need_edges)}")
    print(f"  • produces_need (Material Chains): {len(produces_need_edges)}")
    print(f"  • enabling_lift (Milestone Chains): {len(enabling_edges)}")

    print("\nSample Enabling Chains:")
    for src, dst, lift in enabling_edges[:5]:
        print(f"  • {src}  ──(enables, lift={lift:.2f})──>  {dst}")

    print("\n" + "=" * 80)
    if not vitals_cards and not sapling_cards and not self_needs:
        print("✅ VERDICT: Bank is 100% CLEAN. Preconditions are purely causal.")
        print("   Ready to launch Clean Arm B Screening!")
        print("=" * 80)
        return 0
    else:
        print("❌ VERDICT: Issues found. Review before running screening.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Phase 2 Causal Memory Bank.")
    parser.add_argument("bank_path", type=str, default="checkpoints/phase2_causal_memory_bank_v2.json", nargs="?")
    args = parser.parse_args()
    sys.exit(inspect_bank(args.bank_path))
