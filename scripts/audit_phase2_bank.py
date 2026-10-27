#!/usr/bin/env python3
"""
scripts/audit_phase2_bank.py — Zero-GPU bank audit (plan Stage 2 gate).

Loads a phase2_causal_memory_bank JSON and verifies the structural invariants that
the screening acceptance criteria depend on. This is EVALUATION TOOLING: it may name
specific achievements (like the screening reports do) — the agent itself never does.

Checks (plan AC1 + senior-review amendments):
  A. No vitals dims (12-15) in any card's needs (absolute-vitals labels must have
     removed them; enrichment must have rejected the rest).
  B. place_table-family cards (action 8): needs ⊆ wood (dim 0), amount in [0.5, 2.5].
  C. At least one action-5 card producing wood with confirmed status and wilson_lb >= 0.3
     (the wood->table chain must be traversable).
  D. No card both needs and produces the same dim.
  E. Hypothesis cards carry no needs.
  F. Enabling edges: lift >= 1.5, support >= 5; enabling subgraph acyclic.
  G. All event-grounded cards have achievement_delta mass on their event_dim.

Exit code 0 = all hard checks pass; 1 = at least one hard failure (screening must not run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.causal_memory import NUM_ACHIEVEMENTS, STATE_DIM  # noqa: E402

VITAL_DIMS = set(range(12, 16))
RESOURCE_WOOD_DIM = 0
PLACE_TABLE_ACTION = 8
HARVEST_ACTION = 5


def _edges_acyclic(edges):
    """edges: list of (src, dst). Returns (is_acyclic, cycle_nodes or None)."""
    adj = {}
    for s, d in edges:
        adj.setdefault(s, set()).add(d)
    state = {}  # 0 unvisited, 1 in-stack, 2 done

    def dfs(node, stack):
        state[node] = 1
        stack.append(node)
        for nxt in adj.get(node, ()):
            if state.get(nxt, 0) == 1:
                return stack[stack.index(nxt):] + [nxt]
            if state.get(nxt, 0) == 0:
                cyc = dfs(nxt, stack)
                if cyc:
                    return cyc
        stack.pop()
        state[node] = 2
        return None

    for root in list(adj):
        if state.get(root, 0) == 0:
            cyc = dfs(root, [])
            if cyc:
                return False, cyc
    return True, None


def audit(bank_path: Path) -> int:
    data = json.loads(bank_path.read_text())
    protos = data.get("prototypes", {})
    edges_raw = data.get("edges", {})
    failures, warnings = [], []

    print(f"Bank: {bank_path}  |  cards: {len(protos)}  |  edge dsts: {len(edges_raw)}")

    # A. vitals preconditions
    vitals_cards = []
    for pid, p in protos.items():
        for dim, _amt in p.get("needs", []):
            if int(dim) in VITAL_DIMS:
                vitals_cards.append((pid, int(dim)))
    if vitals_cards:
        failures.append(f"A: {len(vitals_cards)} cards carry vitals preconditions (e.g. {vitals_cards[:5]})")
    else:
        print("  [PASS] A: zero vitals preconditions bank-wide")

    # B. place_table family: needs ⊆ wood
    table_cards = [(pid, p) for pid, p in protos.items() if p.get("action") == PLACE_TABLE_ACTION]
    bad_tables = []
    for pid, p in table_cards:
        for dim, amt in p.get("needs", []):
            if int(dim) != RESOURCE_WOOD_DIM or not (0.5 <= float(amt) <= 2.5):
                bad_tables.append((pid, dim, amt))
    if table_cards and bad_tables:
        failures.append(f"B: place_table cards with non-wood needs: {bad_tables[:5]}")
    elif not table_cards:
        warnings.append("B: no place_table cards found in bank (data coverage)")
    else:
        print(f"  [PASS] B: {len(table_cards)} place_table cards, needs all wood-only")

    # C. wood producer exists and is trusted
    wood_producers = [
        (pid, p) for pid, p in protos.items()
        if p.get("action") == HARVEST_ACTION
        and any(int(d) == RESOURCE_WOOD_DIM for d, _ in p.get("produces", []))
        and p.get("status", "confirmed") == "confirmed"
        and float(p.get("wilson_lb", 0.0)) >= 0.3
    ]
    if wood_producers:
        print(f"  [PASS] C: {len(wood_producers)} trusted action-{HARVEST_ACTION} wood producer(s)")
    else:
        failures.append(f"C: no confirmed action-{HARVEST_ACTION} wood producer with wilson_lb >= 0.3")

    # D. no card needs and produces the same dim
    both = [
        pid for pid, p in protos.items()
        if {int(d) for d, _ in p.get("needs", [])} & {int(d) for d, _ in p.get("produces", [])}
    ]
    if both:
        failures.append(f"D: cards both needing and producing same dim: {both[:5]}")
    else:
        print("  [PASS] D: no needs/produces overlap within any card")

    # E. hypothesis cards carry no needs
    hyp_with_needs = [
        pid for pid, p in protos.items()
        if p.get("status") == "hypothesis" and p.get("needs")
    ]
    if hyp_with_needs:
        failures.append(f"E: hypothesis cards with needs: {hyp_with_needs[:5]}")
    else:
        print("  [PASS] E: hypothesis cards carry no needs")

    # F. enabling edges sane + acyclic
    enabling = []
    for dst, edgelist in edges_raw.items():
        for e in edgelist:
            if e.get("kind") == "enabling_lift":
                if float(e.get("lift", 0.0)) < 1.5 or int(e.get("support", 0)) < 5:
                    warnings.append(f"F: weak enabling edge {e.get('src_id')}->{dst} lift={e.get('lift')} support={e.get('support')}")
                enabling.append((e.get("src_id"), dst))
    cyc_ok, cyc = _edges_acyclic(enabling)
    if not cyc_ok:
        failures.append(f"F: enabling subgraph has a cycle: {cyc}")
    else:
        print(f"  [PASS] F: {len(enabling)} enabling edges, lift/support sane, acyclic")

    # G. event cards' achievement_delta mass sits on event_dim
    misplaced = [
        pid for pid, p in protos.items()
        if p.get("event_dim") is not None
        and (len(p.get("achievement_delta", [])) != NUM_ACHIEVEMENTS
             or max(range(len(p["achievement_delta"])), key=lambda k: p["achievement_delta"][k]) != int(p["event_dim"]))
    ]
    if misplaced:
        failures.append(f"G: event cards with delta mass off event_dim: {misplaced[:5]}")
    else:
        print("  [PASS] G: achievement_delta aligned with event_dim on all event cards")

    hyp = sum(1 for p in protos.values() if p.get("status") == "hypothesis")
    ev = sum(1 for p in protos.values() if p.get("event_dim") is not None)
    print(f"  Summary: {len(protos)} cards ({ev} event-grounded, {hyp} hypothesis), "
          f"{sum(len(v) for v in edges_raw.values())} edges")

    for w in warnings:
        print(f"  [WARN] {w}")
    if failures:
        print("AUDIT FAILED:")
        for f in failures:
            print(f"  [FAIL] {f}")
        return 1
    print("AUDIT PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-GPU Phase 2 bank audit (screening gate).")
    parser.add_argument("bank", type=str, help="Path to phase2_relation_bank JSON")
    args = parser.parse_args()
    sys.exit(audit(Path(args.bank)))


if __name__ == "__main__":
    main()
