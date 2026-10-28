#!/usr/bin/env python3
"""
Online Memory Exporter & Reviewer for Phase 2 Causal Memory Bank.

Extracts all online-created, confirmed, and denied causal cards, their
associated causal graph edges, and the unlock ledger from:
  /content/drive/MyDrive/learn-a-world/checkpoints/phase2_causal_memory_bank_online.json

Outputs:
  1. A structured JSON dump containing only online-active memory:
     results/online_memory_dump.json
  2. A comprehensive, human-readable Markdown review report:
     results/online_memory_review.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS

CRAFTER_ACTIONS = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

INVENTORY_NAMES = [
    "wood", "stone", "table", "furnace", "iron", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
    "health", "food", "drink", "energy"
]


def _format_item_amount(dim: int, amt: float) -> str:
    name = INVENTORY_NAMES[dim] if 0 <= dim < len(INVENTORY_NAMES) else f"dim_{dim}"
    return f"{amt:.1f} {name}" if abs(amt - round(amt)) > 1e-3 else f"{int(round(amt))} {name}"


def export_online_memory_review(
    bank_path: str,
    output_md: str = "results/online_memory_review.md",
    output_json: str = "results/online_memory_dump.json",
) -> None:
    p = Path(bank_path)
    if not p.exists():
        print(f"❌ Error: Bank file not found: {bank_path}")
        sys.exit(1)

    print(f"🔍 Loading bank from: {bank_path} ({p.stat().st_size / 1024:.1f} KB)")
    raw = json.loads(p.read_text())
    prototypes: Dict[str, Dict[str, Any]] = raw.get("prototypes", {})
    edges: Dict[str, List[Dict[str, Any]]] = raw.get("edges", {})
    unlock_ledger = raw.get("unlock_ledger", [])

    total_cards = len(prototypes)
    online_created = [card for card in prototypes.values() if card.get("created_from") == "online"]
    online_reinforced = [
        card for card in prototypes.values()
        if card.get("success_online", 0.0) > 0 or card.get("failure_online", 0.0) > 0
    ]
    online_active_ids = {c["prototype_id"] for c in online_created} | {c["prototype_id"] for c in online_reinforced}

    # Filter edges relevant to online cards
    relevant_edges = []
    for dst_id, edge_list in edges.items():
        for edge in edge_list:
            src_id = edge.get("src_id")
            if src_id in online_active_ids or dst_id in online_active_ids:
                relevant_edges.append(edge)

    print("=" * 80)
    print(f"📊 ONLINE MEMORY BANK AUDIT SUMMARY")
    print("=" * 80)
    print(f"• Total Prototypes in Bank:   {total_cards}")
    print(f"• Online-Created Cards:       {len(online_created)}")
    print(f"• Online-Reinforced Cards:    {len(online_reinforced)}")
    print(f"• Total Active Online Cards:  {len(online_active_ids)}")
    print(f"• Causal Edges (Online):      {len(relevant_edges)}")
    print("=" * 80)

    # 1. Build Markdown Report
    lines = [
        "# Online Causal Memory Bank Review Report",
        "",
        f"**Source Bank File**: `{p.name}`  ",
        f"**Total Prototypes**: {total_cards}  ",
        f"**Online-Created Cards**: {len(online_created)}  ",
        f"**Online-Reinforced Cards**: {len(online_reinforced)}  ",
        f"**Causal Edges Involving Online Cards**: {len(relevant_edges)}  ",
        "",
        "---",
        "",
        "## 1. Discovered Online Cards (Hypotheses & Confirmed)",
        "",
        "| Card ID | Action | Event Achievement | Status | Wilson Conf | Online Succ / Fail | Needs | Produces | Consumes |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |",
    ]

    for c in online_created:
        pid = c.get("prototype_id", "")
        act_idx = c.get("action", 0)
        act_name = CRAFTER_ACTIONS[act_idx] if 0 <= act_idx < len(CRAFTER_ACTIONS) else f"act_{act_idx}"
        ed = c.get("event_dim")
        ed_name = CRAFTER_ACHIEVEMENTS[ed] if (ed is not None and 0 <= ed < len(CRAFTER_ACHIEVEMENTS)) else "None"
        status = c.get("status", "hypothesis")
        conf = c.get("confidence", 0.0)
        s_on = c.get("success_online", 0.0)
        f_on = c.get("failure_online", 0.0)

        needs_strs = [_format_item_amount(int(d), amt) for d, amt in c.get("needs", [])]
        prod_strs = [_format_item_amount(int(d), amt) for d, amt in c.get("produces", [])]
        cons_strs = [_format_item_amount(int(d), amt) for d, amt in c.get("consumes", [])]

        lines.append(
            f"| `{pid}` | `{act_name}` ({act_idx}) | **{ed_name}** | `{status}` | {conf:.3f} | "
            f"**{s_on:.1f}** / {f_on:.1f} | {', '.join(needs_strs) or '—'} | "
            f"{', '.join(prod_strs) or '—'} | {', '.join(cons_strs) or '—'} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Offline Cards Reinforced During Gameplay",
        "",
        "Cards originally formed offline whose Bayesian confidence was sharpened by live online confirmations or near-miss denials:",
        "",
        "| Card ID | Action | Event | Offline S/F | Online S/F | New Confidence | Needs | Produces |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ])

    offline_reinforced = [c for c in online_reinforced if c.get("created_from") != "online"]
    for c in offline_reinforced[:25]:
        pid = c.get("prototype_id", "")
        act_idx = c.get("action", 0)
        act_name = CRAFTER_ACTIONS[act_idx] if 0 <= act_idx < len(CRAFTER_ACTIONS) else f"act_{act_idx}"
        ed = c.get("event_dim")
        ed_name = CRAFTER_ACHIEVEMENTS[ed] if (ed is not None and 0 <= ed < len(CRAFTER_ACHIEVEMENTS)) else "None"
        conf = c.get("confidence", 0.0)
        s_off = c.get("success_offline", 0.0)
        f_off = c.get("failure_offline", 0.0)
        s_on = c.get("success_online", 0.0)
        f_on = c.get("failure_online", 0.0)

        needs_strs = [_format_item_amount(int(d), amt) for d, amt in c.get("needs", [])]
        prod_strs = [_format_item_amount(int(d), amt) for d, amt in c.get("produces", [])]

        lines.append(
            f"| `{pid}` | `{act_name}` | {ed_name} | {s_off:.0f}/{f_off:.0f} | "
            f"**+{s_on:.1f}** / **-{f_on:.1f}** | **{conf:.3f}** | {', '.join(needs_strs) or '—'} | "
            f"{', '.join(prod_strs) or '—'} |"
        )
    if len(offline_reinforced) > 25:
        lines.append(f"| ... | ... | ... | ... | ... | ... | *(+{len(offline_reinforced)-25} more reinforced cards)* | ... |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Causal Dependency Graph (Online Relations)",
        "",
        "| Source Card (Enabler / Producer) | -> | Target Card (Consumer / Milestone) | Relation Type | Resource / Lift |",
        "| :--- | :---: | :--- | :--- | :--- |",
    ])

    for e in relevant_edges[:30]:
        src = e.get("src_id", "")
        dst = e.get("dst_id", "")
        kind = e.get("kind", "")
        dim = e.get("dim")
        lift = e.get("lift", 1.0)
        dim_str = INVENTORY_NAMES[dim] if (dim is not None and 0 <= dim < len(INVENTORY_NAMES)) else "None"
        detail = f"Resource: **{dim_str}**" if kind == "produces_need" else f"Temporal Lift: **{lift:.2f}x**"
        lines.append(f"| `{src}` | ➔ | `{dst}` | `{kind}` | {detail} |")

    if len(relevant_edges) > 30:
        lines.append(f"| ... | ➔ | ... | ... | *(+{len(relevant_edges)-30} more causal edges)* |")

    # 4. Write files
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Human-readable Markdown review exported to: {output_md}")

    dump_payload = {
        "summary": {
            "bank_path": bank_path,
            "total_prototypes": total_cards,
            "online_created_count": len(online_created),
            "online_reinforced_count": len(online_reinforced),
            "online_edges_count": len(relevant_edges),
        },
        "online_created_cards": online_created,
        "online_reinforced_cards": offline_reinforced,
        "online_edges": relevant_edges,
        "unlock_ledger": unlock_ledger,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(dump_payload, indent=2), encoding="utf-8")
    print(f"✅ Filtered Online Memory JSON exported to:     {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Export & Review Online Memory Bank.")
    parser.add_argument("bank_path", type=str, nargs="?", default="/content/drive/MyDrive/learn-a-world/checkpoints/phase2_causal_memory_bank_online.json")
    parser.add_argument("--output_md", type=str, default="results/online_memory_review.md")
    parser.add_argument("--output_json", type=str, default="results/online_memory_dump.json")
    args = parser.parse_args()

    export_online_memory_review(args.bank_path, args.output_md, args.output_json)


if __name__ == "__main__":
    main()
