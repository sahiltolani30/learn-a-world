#!/usr/bin/env python3
"""
scripts/read_phase2_cards.py — Human-readable card reader for the Phase 2 memory bank.

Prints every learned causal card as a plain sentence so a human can verify WHAT THE
AGENT LEARNED — this is evaluation tooling: it may name achievements/items (the agent
itself never does). Zero model / GPU needed — reads the bank JSON directly.

Usage:
    python3 scripts/read_phase2_cards.py checkpoints/phase2_causal_memory_bank_v2.json
    python3 scripts/read_phase2_cards.py <bank.json> --event <n>     # cards grounding event n
    python3 scripts/read_phase2_cards.py <bank.json> --action 8      # cards for action 8
    python3 scripts/read_phase2_cards.py <bank.json> --edges-only
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS  # noqa: E402

# Display names only (tooling). Dim order: CRAFTER_INVENTORY_KEYS (resources, tools, vitals).
DIM_NAMES = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
    "health", "food", "drink", "energy",
]


def _pairs(pairs):
    return ", ".join(f"{DIM_NAMES[int(d)]}>={float(a):.2f}" for d, a in pairs) if pairs else "nothing"


def card_sentence(pid, c):
    action = c.get("action")
    ev = c.get("event_dim")
    unlocks = CRAFTER_ACHIEVEMENTS[ev] if ev is not None else "(no achievement — resource/source card)"
    status = c.get("status", "?")
    k = float(c.get("success_count", 0))
    n = k + float(c.get("failure_count", 0))
    lines = [
        f"[{pid}]  action {action}  ->  UNLOCKS {unlocks}",
        f"    WHEN     : visual context (cue dispersion {float(c.get('visual_dispersion', 1.0)):.2f} — "
        + ("homogeneous/informative" if float(c.get("visual_dispersion", 1.0)) >= 0.70 else "varied/cue-weak") + ")",
        f"    NEEDS    : {_pairs(c.get('needs', []))}",
        f"    COSTS    : {_pairs(c.get('consumes', []))}",
        f"    PRODUCES : {_pairs(c.get('produces', []))}",
        f"    EVIDENCE : status={status}, trust(Wilson LB)={float(c.get('wilson_lb', 0.0)):.2f}, "
        f"unlock_prob={float(c.get('unlock_prob', 0.0)):.2f}, {k:.0f} successes / {n - k:.0f} failures",
    ]
    return "\n".join(lines)


def edge_sentences(edges):
    out = []
    for dst, edgelist in edges.items():
        for e in edgelist:
            kind = e.get("kind")
            if kind == "enabling_lift":
                out.append(f"  {e['src_id']}  --enables(lift {float(e.get('lift', 0)):.2f}x, "
                           f"support {e.get('support', 0)}, dt {float(e.get('mean_dt', 0)):.0f} steps)-->  {dst}")
            else:
                out.append(f"  {e['src_id']}  --produces {DIM_NAMES[int(e['dim'])]} needed by-->  {dst}")
    return out


def sanity_checks(data):
    """Plain-language version of the audit gate, for eyeballing."""
    protos = data.get("prototypes", {})
    print("=" * 78)
    print("SANITY LENS (what correct learning should and should not look like)")
    print("=" * 78)
    vitals = [(pid, d) for pid, c in protos.items() for d, _ in c.get("needs", []) if int(d) >= 12]
    print(f"  cards demanding vitals (should be 0):            {len(vitals)} {vitals[:4]}")
    tables = {pid: c for pid, c in protos.items() if c.get("action") == 8}
    bad_tables = [(pid, d) for pid, c in tables.items() for d, _ in c.get("needs", []) if int(d) != 0]
    print(f"  place_table cards:                                {len(tables)}; non-wood needs: {len(bad_tables)} {bad_tables[:4]}")
    wood_prod = [pid for pid, c in protos.items()
                 if c.get("action") == 5 and any(int(d) == 0 for d, _ in c.get("produces", []))]
    print(f"  action-5 cards producing wood (should be >= 1):   {len(wood_prod)} {wood_prod[:4]}")
    sapling_needs = [pid for pid, c in protos.items() if any(int(d) == 5 for d, _ in c.get("needs", []))]
    print(f"  cards still demanding saplings (the old bug):     {len(sapling_needs)} {sapling_needs[:4]}")
    hyp = sum(1 for c in protos.values() if c.get("status") == "hypothesis")
    ev = sum(1 for c in protos.values() if c.get("event_dim") is not None)
    print(f"  cards: {len(protos)} total | {ev} event-grounded | {hyp} hypothesis (unproven, explorable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--event", type=int, default=None, help="Only cards grounding this achievement index")
    ap.add_argument("--action", type=int, default=None, help="Only cards for this action index")
    ap.add_argument("--edges-only", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.bank).read_text())
    protos = data.get("prototypes", {})
    edges = data.get("edges", {})

    sanity_checks(data)

    print("\n" + "=" * 78)
    print("LEARNED CAUSAL EDGES (the 'from X I can make Y' sentences)")
    print("=" * 78)
    es = edge_sentences(edges)
    print("\n".join(es) if es else "  (no edges learned)")

    if not args.edges_only:
        print("\n" + "=" * 78)
        print("LEARNED CARDS (the notebook)")
        print("=" * 78)
        for pid, c in sorted(protos.items()):
            if args.event is not None and c.get("event_dim") != args.event:
                continue
            if args.action is not None and c.get("action") != args.action:
                continue
            print("\n" + card_sentence(pid, c))


if __name__ == "__main__":
    main()
