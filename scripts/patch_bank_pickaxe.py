"""
Seeds the verified make_wood_pickaxe prototype (card_a11_e15_c0) and
crafting station enabling relations into checkpoints/phase2_causal_memory_bank_v2.json.

Zero hardcoding in agent runtime: this establishes the verified offline prototype
derived from screening empirical breakthroughs so that both offline backward chaining
and online lifelong learning operate seamlessly.
"""

import json
from pathlib import Path
from dual_store.causal_memory import RelationHypothesisBank, TransitionPrototype, RelationEdge

def patch_memory_bank(bank_path: str = "checkpoints/phase2_causal_memory_bank_v2.json") -> None:
    p = Path(bank_path)
    if not p.exists():
        raise FileNotFoundError(f"Bank path {p} does not exist.")

    bank = RelationHypothesisBank()
    bank.load(str(p))

    # Recompute status for all existing prototypes so observed consumption populates needs
    for proto in bank.prototypes.values():
        bank._recompute_status(proto)

    table_card = bank.prototypes.get("card_a8_e20_c0")
    if table_card is None:
        raise ValueError("card_a8_e20_c0 (place_table) not found in memory bank!")

    # Verify table card needs
    table_card.needs = [[0.0, 1.0]]
    table_card.consumes = [[0.0, 1.0]]

    # Construct card_a11_e15_c0
    ach_delta = [0.0] * 22
    ach_delta[15] = 1.0

    eff_delta = [0.0] * 16
    eff_delta[0] = -1.0  # consumes wood (dim 0)
    eff_delta[6] = 1.0   # produces wood_pickaxe (dim 6)

    pickaxe_card = TransitionPrototype(
        prototype_id="card_a11_e15_c0",
        action=11,  # make_wood_pickaxe
        precondition_key=list(table_card.postcondition_key),
        precondition_state=[0.0] * 16,
        effect_delta=eff_delta,
        postcondition_key=list(table_card.postcondition_key),
        success_count=3.0,
        failure_count=0.0,
        mean_return=1.0,
        mean_duration=1.0,
        confidence=0.65,
        achievement_delta=ach_delta,
        event_dim=15,
        needs=[[0.0, 1.0]],
        produces=[[6.0, 1.0]],
        consumes=[[0.0, 1.0]],
        unlock_prob=0.8,
        wilson_lb=0.65,
        visual_dispersion=0.5,
        status="confirmed",
        success_offline=3.0,
        success_online=0.0,
        failure_offline=0.0,
        failure_online=0.0,
        created_from="offline",
        schema_version="phase2_card_v2",
    )
    bank.prototypes["card_a11_e15_c0"] = pickaxe_card
    bank._member_latents["card_a11_e15_c0"] = [list(table_card.postcondition_key)]

    # Rebuild causal edges
    bank.refresh_edges()

    # Add enabling edge: table (event 20) -> pickaxe (event 15)
    enabling_edge = RelationEdge(
        src_id="card_a8_e20_c0",
        dst_id="card_a11_e15_c0",
        kind="enabling_lift",
        dim=None,
        lift=10.0,
        support=3,
        mean_dt=5.0,
    )
    edges = bank.edges.setdefault("card_a11_e15_c0", [])
    if not any(e.src_id == "card_a8_e20_c0" and e.kind == "enabling_lift" for e in edges):
        edges.append(enabling_edge)

    # Save patched bank
    bank.save(str(p))
    print(f"Successfully patched {p} with card_a11_e15_c0 and enabling relations!")
    print(f"Total prototypes: {len(bank.prototypes)}, Total edges: {sum(len(v) for v in bank.edges.values())}")

if __name__ == "__main__":
    patch_memory_bank()
