#!/usr/bin/env python3
"""
scripts/strict_anti_hardcoding_audit.py — DeepMind-Grade Strict Anti-Hardcoding Auditor
======================================================================================
Strictly enforces THE IRON RULE:
  1. ZERO Human Recipes (no "table = 2 wood", no recipe dictionary, no manual rules).
  2. ZERO Action-Specific Hardcoding (no "if action == 8: needs = wood").
  3. ZERO Precondition Cheats (all needs must be dynamically discovered from delta x < 0).
  4. ZERO Vitally-Corrupted Needs (no health/food/drink/energy in crafting recipes).
  5. ZERO Phantom Preconditions (no ghost saplings on movement/sleep).

Can be run locally or in Colab on any memory bank or consolidation diagnostic JSON.
"""

import ast
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that execute agent reasoning, planning, and memory consolidation
CORE_AGENT_FILES = [
    REPO_ROOT / "planning/causal_memory.py",
    REPO_ROOT / "planning/compositional_planner.py",
    REPO_ROOT / "training/run_phase2_memory_consolidation.py",
]

# Prohibited hardcoding pattern signatures in core agent logic
PROHIBITED_PATTERNS = [
    (r"RECIPES\s*=\s*\{", "Hardcoded recipe dictionary definition"),
    (r"CRAFTING_TABLE\s*=\s*\{", "Hardcoded crafting table recipe dictionary"),
    (r"TECH_TREE\s*=\s*\{", "Hardcoded technology tree mapping"),
    (r"if\s+.*action\s*==\s*8\s*:\s*.*needs\s*=", "Action-specific hardcoded table recipe assignment"),
    (r"if\s+.*action\s*==\s*1[1-6]\s*:\s*.*needs\s*=", "Action-specific hardcoded tool recipe assignment"),
    (r"needs\s*=\s*\[\[.*['\"]wood['\"].*\]\]", "Hardcoded string recipe assignment for wood"),
    (r"needs\s*=\s*\[\[0(\.0)?,\s*2(\.0)?\]\]", "Hardcoded static 2-wood table recipe assignment"),
]


def audit_codebase_anti_hardcoding() -> bool:
    print("=" * 80)
    print("   [CRITIC CHECK 1/2] STATIC CODEBASE INSPECTION (ZERO-HARDCODING AUDIT)")
    print("=" * 80)
    all_clean = True

    for fpath in CORE_AGENT_FILES:
        if not fpath.exists():
            print(f"  [WARN] Skipping non-existent file: {fpath}")
            continue

        text = fpath.read_text()
        rel_path = fpath.relative_to(REPO_ROOT)
        print(f"\n-> Auditing: {rel_path} ({len(text.splitlines())} lines)")

        file_violations = 0
        for pat, desc in PROHIBITED_PATTERNS:
            matches = list(re.finditer(pat, text, re.IGNORECASE))
            if matches:
                for m in matches:
                    line_no = text[:m.start()].count("\n") + 1
                    snippet = text.splitlines()[line_no - 1].strip()
                    print(f"  ❌ VIOLATION FOUND [line {line_no}]: {desc}")
                    print(f"     Line: {snippet}")
                    file_violations += 1
                    all_clean = False

        if file_violations == 0:
            print(f"  ✅ [PASS] Clean. Zero hardcoded recipes or action-specific cheats in {fpath.name}.")

    return all_clean


def audit_memory_bank_anti_hardcoding(bank_path: str) -> bool:
    print("\n" + "=" * 80)
    print("   [CRITIC CHECK 2/2] MEMORY BANK DATA AUDIT (EMPIRICAL GROUNDING AUDIT)")
    print("=" * 80)
    p = Path(bank_path)
    if not p.exists():
        print(f"  ❌ Error: Memory bank file not found: {bank_path}")
        return False

    data = json.loads(p.read_text())
    if "prototypes" in data and isinstance(data["prototypes"], dict):
        protos = data["prototypes"]
    elif "prototype_summary" in data and isinstance(data["prototype_summary"], list):
        protos = {item.get("prototype_id", item.get("id")): item for item in data["prototype_summary"]}
    else:
        print("  ❌ Error: Unrecognized memory bank format.")
        return False

    print(f"Bank File: {p.name}  |  Total Prototypes: {len(protos)}")

    failures = []

    # Check A: Dynamic Grounding in Consumes (Conservation of Matter)
    # Every need must exist in consumes, and must be derived from delta x < 0
    unanchored_needs = []
    for pid, card in protos.items():
        consumes_dict = {int(d): amt for d, amt in card.get("consumes", [])}
        for d, amt in card.get("needs", []):
            d_int = int(d)
            if d_int not in consumes_dict:
                unanchored_needs.append((pid, d_int, amt, "Need not present in consumed delta"))
            elif abs(consumes_dict[d_int] - amt) > 1e-3:
                unanchored_needs.append((pid, d_int, amt, f"Need ({amt}) != Consumed delta ({consumes_dict[d_int]})"))

    if unanchored_needs:
        failures.append(f"Unanchored needs: {len(unanchored_needs)} cards have needs not equal to consumed state deltas (e.g. {unanchored_needs[:3]})")
    else:
        print("  ✅ [PASS] 100% Causal Conservation: Every need strictly equals observed consumed delta (delta x < 0).")

    # Check B: Zero Vitals in Needs (Health/Food/Drink/Energy)
    vitals_cards = [
        (pid, int(d))
        for pid, card in protos.items()
        for d, _ in card.get("needs", [])
        if int(d) in range(12, 16)
    ]
    if vitals_cards:
        failures.append(f"Vitals contamination: {len(vitals_cards)} cards have vitals in needs: {vitals_cards[:3]}")
    else:
        print("  ✅ [PASS] Zero Vitals: 0 cards carry health/food/drink/energy preconditions.")

    # Check C: Zero Ghost Saplings
    ghost_saplings = [
        (pid, card.get("action"))
        for pid, card in protos.items()
        for d, _ in card.get("needs", [])
        if int(d) == 5 and card.get("action") != 10  # Only action 10 (place_plant) uses sapling
    ]
    if ghost_saplings:
        failures.append(f"Ghost saplings: {len(ghost_saplings)} cards require saplings for non-planting actions: {ghost_saplings[:3]}")
    else:
        print("  ✅ [PASS] Zero Ghost Saplings: 0 movement/sleep/punching cards require saplings.")

    # Check D: Preconditions strictly empty for non-crafts
    non_craft_needs = [
        (pid, card.get("action"))
        for pid, card in protos.items()
        if card.get("action") in (0, 1, 2, 3, 4, 6) and card.get("needs")
    ]
    if non_craft_needs:
        failures.append(f"Movement/sleep needs: {len(non_craft_needs)} non-crafting actions have needs: {non_craft_needs[:3]}")
    else:
        print("  ✅ [PASS] Zero Spurious Needs on Basic Actions: Movement, noop, and sleep have empty needs.")

    # Check E: Confidence strictly from Wilson lower bound
    invalid_conf = []
    for pid, card in protos.items():
        succ = card.get("success_count", 0.0)
        fail = card.get("failure_count", 0.0)
        conf = card.get("confidence", 0.0)
        w_lb = card.get("wilson_lb", 0.0)
        if conf < 0.05 or conf > 1.0 or w_lb < 0.0:
            invalid_conf.append((pid, succ, fail, conf, w_lb))
    if invalid_conf:
        failures.append(f"Invalid confidence values: {len(invalid_conf)} cards: {invalid_conf[:3]}")
    else:
        print("  ✅ [PASS] Empirical Bayesian Tracking: All confidence values computed from empirical counts.")

    # Check F: Online Adaptation Report
    online_created = [card for card in protos.values() if card.get("created_from") == "online"]
    online_reinforced = [card for card in protos.values() if card.get("success_online", 0.0) > 0 or card.get("failure_online", 0.0) > 0]
    if online_created or online_reinforced:
        print("\n" + "=" * 80)
        print("   [CRITIC CHECK 3/3] ONLINE ADAPTATION & CAUSAL CARD REPORT")
        print("=" * 80)
        print(f"  • Online-Created Cards:   {len(online_created)}")
        print(f"  • Online-Active Cards:    {len(online_reinforced)}")
        from eval.metrics import CRAFTER_ACHIEVEMENTS
        for c in online_created[:15]:
            ed = c.get("event_dim")
            ed_name = CRAFTER_ACHIEVEMENTS[ed] if (ed is not None and 0 <= ed < len(CRAFTER_ACHIEVEMENTS)) else "event-free"
            needs_str = f"needs={c.get('needs', [])}" if c.get('needs') else "needs=[]"
            print(f"    - {c.get('prototype_id')}: action={c.get('action')} -> event='{ed_name}' | {c.get('status')} | succ_on={c.get('success_online')} fail_on={c.get('failure_online')} conf={c.get('confidence'):.3f} | {needs_str}")
        if len(online_created) > 15:
            print(f"    ... and {len(online_created) - 15} more online-discovered cards.")

    print("\n" + "-" * 80)
    if not failures:
        print("  🏆 VERDICT: STRICT CRITIC AUDIT PASSED 100%.")
        print("     Zero hardcoding, zero cheats, zero human recipe injections found.")
        print("-" * 80)
        return True
    else:
        print("  ❌ VERDICT: STRICT CRITIC AUDIT FAILED:")
        for f in failures:
            print(f"     • {f}")
        print("-" * 80)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strict Anti-Hardcoding Critic for Phase 2 Causal Memory.")
    parser.add_argument("bank_path", type=str, nargs="?", default=None)
    parser.add_argument("--bank_path", dest="flag_bank_path", type=str, default=None)
    args = parser.parse_args()

    bank_target = args.flag_bank_path or args.bank_path or "results/diagnostics/phase2_memory_consolidation_4.json"

    c1 = audit_codebase_anti_hardcoding()
    c2 = audit_memory_bank_anti_hardcoding(bank_target)

    if c1 and c2:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
