"""Phase 2 Causal Card tests (Stages 3+4 of docs/PHASE2_MEMORY_REDESIGN_PLAN.md).

Covers: Wilson intervals, needs-enrichment on the real wood/sapling statistics,
event-grounded clustering, hypothesis status, incremental online learning
(match-and-update), fallback quarantine (N7), enabling-lift edges (lift,
time-direction, acyclicity), produces_need edges, and v1/v2 bank persistence.

Zero hardcoding: dim indices below are interface constants of the 16-d state /
22-achievement spaces, never action semantics.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from planning.causal_memory import (
    EpisodicEvidence,
    EpisodicEvidenceStore,
    RelationEdge,
    RelationHypothesisBank,
    TransitionPrototype,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Local fixtures (plain helpers; tests/ has no __init__.py)
# ---------------------------------------------------------------------------

def _onehot(dim: int, n: int) -> list:
    v = [0.0] * n
    v[dim] = 1.0
    return v


def make_card(**overrides) -> TransitionPrototype:
    """One-line card builder with sane defaults."""
    defaults = dict(
        prototype_id="card_test",
        action=0,
        precondition_key=[1.0] + [0.0] * 127,
        precondition_state=[0.0] * 16,
        effect_delta=[0.0] * 16,
        postcondition_key=[0.5] * 128,
        success_count=10.0,
        failure_count=1.0,
        mean_return=1.0,
    )
    defaults.update(overrides)
    return TransitionPrototype(**defaults)


def add_unlock_episode(store: EpisodicEvidenceStore, episode_id: int, action: int,
                       steps, latent) -> None:
    """Appends one episode of event-labelled steps.

    steps: ordered list of (mask_before_dims, unlock_dims).
    """
    for t, (mask_dims, unlock_dims) in enumerate(steps):
        event_vec = [0] * 22
        for d in unlock_dims:
            event_vec[d] = 1
        mask_vec = [0] * 22
        for d in mask_dims:
            mask_vec[d] = 1
        store.entries.append(EpisodicEvidence(
            obs_latent=list(latent),
            grounded_state=[0.0] * 16,
            action=action,
            reward=0.0,
            done=False,
            next_obs_latent=list(latent),
            next_grounded_state=[0.0] * 16,
            state_delta=[0.0] * 16,
            return_g=0.0,
            episode_id=episode_id,
            step_id=t,
            event_vec=event_vec,
            mask_before_vec=mask_vec,
        ))


def all_edges(bank: RelationHypothesisBank):
    for dst, edge_list in bank.edges.items():
        for e in edge_list:
            yield e


def enabling_adjacency(bank: RelationHypothesisBank):
    adj = {}
    for e in all_edges(bank):
        if e.kind == "enabling_lift":
            adj.setdefault(e.src_id, set()).add(e.dst_id)
    return adj


def is_acyclic(adj) -> bool:
    """DFS cycle check over the enabling-only subgraph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in adj}
    for node in list(adj):
        if color.get(node, WHITE) != WHITE:
            continue
        stack = [(node, iter(adj.get(node, ())))]
        color[node] = GRAY
        while stack:
            parent, it = stack[-1]
            advanced = False
            for child in it:
                if color.get(child, WHITE) == GRAY:
                    return False
                if color.get(child, WHITE) == WHITE:
                    color[child] = GRAY
                    stack.append((child, iter(adj.get(child, ()))))
                    advanced = True
                    break
            if not advanced:
                color[parent] = BLACK
                stack.pop()
    return True


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------

def test_wilson_interval_boundary_cases():
    # Arrange / Act / Assert: empty sample -> maximally uninformative interval
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, -3) == (0.0, 1.0)

    # Single success: lower bound strictly positive (never rank-zeroed)
    lb, ub = wilson_interval(1, 1)
    assert 0.0 < lb < 1.0
    assert lb < ub <= 1.0

    # Zero successes out of 10: lb stays 0.0 but ub is bounded away from 1
    lb0, ub0 = wilson_interval(0, 10)
    assert lb0 == 0.0
    assert 0.0 < ub0 < 0.3

    # Perfect record out of 7: lower bound above 0.75 (80% Wilson)
    lb7, ub7 = wilson_interval(7, 7)
    assert lb7 > 0.75
    assert ub7 <= 1.0

    # Monotonicity in n at a fixed ratio: larger samples tighten the interval
    lb_s, ub_s = wilson_interval(8, 10)
    lb_l, ub_l = wilson_interval(80, 100)
    assert lb_l > lb_s
    assert ub_l < ub_s


# ---------------------------------------------------------------------------
# Needs enrichment (the F2 tombstone: correlated items must not become needs)
# ---------------------------------------------------------------------------

def test_enrichment_keeps_wood_kills_sapling_on_real_statistics():
    # Arrange: 7 successes all holding state dim 0 at 1.5; state dim 5 at 1.2 in 6
    # of 7 (co-occurring by coincidence) and 0.0 in 1; vitals dims 12-15 ~ 0.
    # 50 matched near-misses: dim 0 >= 1.5 in 9/50, dim 5 >= 1.08 in 41/50.
    store = EpisodicEvidenceStore()
    z_succ = _onehot(0, 128)
    z_near = [0.99, 0.01] + [0.0] * 126  # cos(z_near, z_succ) ~= 0.99995 -> matched

    for i in range(7):
        x = [0.0] * 16
        x[0] = 1.5
        x[5] = 1.2 if i < 6 else 0.0
        delta = [0.0] * 16
        delta[0] = -1.5  # resource dim 0 consumed by craft
        delta[6] = 1.0   # tool produced by craft
        store.entries.append(EpisodicEvidence(
            obs_latent=list(z_succ),
            grounded_state=x,
            action=5,
            reward=0.0,
            done=False,
            next_obs_latent=list(z_succ),
            next_grounded_state=x,
            state_delta=delta,
            return_g=0.5,
            episode_id=i,
            step_id=0,
        ))
    for j in range(50):
        x = [0.0] * 16
        x[0] = 1.5 if j < 9 else 0.0
        x[5] = 1.2 if j >= 9 else 0.0
        store.entries.append(EpisodicEvidence(
            obs_latent=list(z_near),
            grounded_state=x,
            action=5,
            reward=0.0,
            done=False,
            next_obs_latent=list(z_near),
            next_grounded_state=x,
            state_delta=[0.0] * 16,
            return_g=0.0,
            episode_id=100 + j,
            step_id=0,
        ))

    bank = RelationHypothesisBank()

    # Act
    count = bank.consolidate_from_evidence(store)

    # Assert: one event-free card; the true requirement survives, the correlated
    # item and the vitals dims do not.
    assert count == 1
    assert len(bank) == 1
    card = list(bank.prototypes.values())[0]
    assert card.status == "confirmed"
    assert card.event_dim is None
    assert card.failure_offline == pytest.approx(50.0)
    wood = [nd for nd in card.needs if int(nd[0]) == 0]
    assert len(wood) == 1
    assert wood[0][1] == pytest.approx(1.5, abs=1e-6)
    assert not any(int(nd[0]) == 5 for nd in card.needs)
    assert not any(int(nd[0]) in (12, 13, 14, 15) for nd in card.needs)
    # Legacy precondition interface mirrors the surviving need
    assert card.precondition_state[0] == pytest.approx(1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Event-grounded clustering
# ---------------------------------------------------------------------------

def test_consolidation_clusters_by_action_and_event():
    # Arrange: one action, two distinct fired achievement dims (2 successes each).
    store = EpisodicEvidenceStore()
    z = _onehot(0, 128)
    for ep in range(2):
        add_unlock_episode(store, ep, 6, [([], [3])], z)
    for ep in range(10, 12):
        add_unlock_episode(store, ep, 6, [([], [9])], z)

    bank = RelationHypothesisBank()

    # Act
    bank.consolidate_from_evidence(store)

    # Assert: two cards with distinct grounded event dims
    assert len(bank) == 2
    event_dims = {p.event_dim for p in bank.prototypes.values()}
    assert event_dims == {3, 9}
    ids = set(bank.prototypes.keys())
    assert any("_e3_" in i for i in ids)
    assert any("_e9_" in i for i in ids)


def test_hypothesis_status_below_three_successes():
    # Arrange: only 2 successes for the (action, event) group.
    store = EpisodicEvidenceStore()
    z = _onehot(0, 128)
    for ep in range(2):
        add_unlock_episode(store, ep, 8, [([], [5])], z)

    bank = RelationHypothesisBank()

    # Act
    bank.consolidate_from_evidence(store)

    # Assert: hypothesis status, needs withheld, raw Wilson lower bound kept
    assert len(bank) == 1
    card = list(bank.prototypes.values())[0]
    assert card.status == "hypothesis"
    assert card.needs == []
    assert card.success_offline == pytest.approx(2.0)
    assert 0.0 < card.wilson_lb < 1.0


# ---------------------------------------------------------------------------
# Incremental online learning
# ---------------------------------------------------------------------------

def _learn_step(z, action, event_dim, confident_events, fallback_unlocked=None,
                mask_dims=()):
    event_vec = [0] * 22
    event_vec[event_dim] = 1
    mask_vec = [0] * 22
    for d in mask_dims:
        mask_vec[d] = 1
    return {
        "obs_latent": list(z),
        "grounded_state": [0.0] * 16,
        "action": action,
        "reward": 1.0,
        "done": False,
        "event_vec": event_vec,
        "mask_before_vec": mask_vec,
        "confident_events": list(confident_events),
        "fallback_unlocked": list(fallback_unlocked or []),
        "state_delta": [0.0] * 16,
    }


def test_incremental_learn_stability():
    # Arrange: three visually distinct unlock observations of the same event.
    bank = RelationHypothesisBank()
    z1 = _onehot(0, 128)
    z2 = [0.9, 0.1] + [0.0] * 126  # cos(z2, z1) ~= 0.994 -> same visual cluster
    z3 = _onehot(2, 128)           # orthogonal -> new visual cluster
    step = lambda z: _learn_step(z, action=5, event_dim=4, confident_events=[4])

    # Act + Assert: first observation creates the online-born card
    r1 = bank.learn_from_episode([step(z1)], episode_id=1)
    assert r1 == {"created": 1, "confirmed": 0, "denied": 0, "steps_recorded": 1}
    assert len(bank) == 1
    assert "card_a5_e4_o0" in bank.prototypes
    card = bank.prototypes["card_a5_e4_o0"]
    assert card.created_from == "online"
    assert card.status == "hypothesis"
    assert card.success_online == pytest.approx(1.0)

    # Second, visually matching observation CONFIRMS the same card (never
    # re-clustered: id stable, bank size unchanged)
    r2 = bank.learn_from_episode([step(z2)], episode_id=2)
    assert r2["created"] == 0
    assert r2["confirmed"] == 1
    assert len(bank) == 1
    assert bank.prototypes["card_a5_e4_o0"].success_online == pytest.approx(2.0)

    # Third, visually distant confident observation creates a SECOND card
    r3 = bank.learn_from_episode([step(z3)], episode_id=3)
    assert r3["created"] == 1
    assert len(bank) == 2
    assert "card_a5_e4_o1" in bank.prototypes
    assert "card_a5_e4_o0" in bank.prototypes


def test_fallback_never_confirms():
    # Arrange: a confirmed event-grounded card already in the bank.
    bank = RelationHypothesisBank()
    z1 = _onehot(0, 128)
    z_far = _onehot(1, 128)
    bank.prototypes["card_a5_e4_c0"] = make_card(
        prototype_id="card_a5_e4_c0",
        action=5,
        precondition_key=list(z1),
        event_dim=4,
        success_count=5.0,
        success_offline=5.0,
        status="confirmed",
    )
    card = bank.prototypes["card_a5_e4_c0"]

    # Act 1: unlock labelled only by the reward fallback, visually matching the
    # confirmed card -> must NOT confirm it (a fallback pin is a guess).
    r1 = bank.learn_from_episode(
        [_learn_step(z1, action=5, event_dim=4, confident_events=[], fallback_unlocked=[4])],
        episode_id=1,
    )

    # Assert 1: no confirmation, no creation, tallies untouched
    assert r1["confirmed"] == 0
    assert r1["created"] == 0
    assert card.success_online == pytest.approx(0.0)
    assert card.success_count == pytest.approx(5.0)
    assert card.status == "confirmed"

    # Act 2: unmatched fallback-pinned unlock -> reduced-prior hypothesis card only
    r2 = bank.learn_from_episode(
        [_learn_step(z_far, action=5, event_dim=7, confident_events=[], fallback_unlocked=[7])],
        episode_id=2,
    )

    # Assert 2: new card is a hypothesis with zero online successes (quarantine)
    assert r2["created"] == 1
    assert len(bank) == 2
    born = bank.prototypes["card_a5_e7_o0"]
    assert born.status == "hypothesis"
    assert born.success_online == pytest.approx(0.0)
    assert born.success_count == pytest.approx(0.0)
    assert born.created_from == "online"
    assert bank.prototypes["card_a5_e4_c0"].success_online == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Causal graph edges
# ---------------------------------------------------------------------------

def test_enabling_edge_from_synthetic_lift():
    # Event dims and action are interface constants, no semantics attached.
    I, J, K = 3, 9, 14
    ACTION = 6
    u_a, u_b, u_c, u_e = (_onehot(0, 128), _onehot(1, 128), _onehot(2, 128), _onehot(3, 128))

    # -- Scenario 1: forward ordering. 5 episodes i->j, 5 j->k, 5 k->i, and 15
    # unlock-free episodes diluting the base rates. Every ordered pair i->j,
    # j->k, k->i gets lift = 6.0, co-occurrence 5, mean_dt 1.
    store = EpisodicEvidenceStore()
    ep = 0
    for _ in range(5):
        add_unlock_episode(store, ep, ACTION, [([], [I]), ([I], [J])], u_a); ep += 1
    for _ in range(5):
        add_unlock_episode(store, ep, ACTION, [([], [J]), ([J], [K])], u_b); ep += 1
    for _ in range(5):
        add_unlock_episode(store, ep, ACTION, [([], [K]), ([K], [I])], u_c); ep += 1
    for _ in range(15):
        add_unlock_episode(store, ep, ACTION, [([], []), ([], [])], u_e); ep += 1

    bank = RelationHypothesisBank()
    bank.consolidate_from_evidence(store)

    card_i, card_j, card_k = (f"card_a{ACTION}_e{e}_c0" for e in (I, J, K))
    # Event cards exist (strongest-card mapping resolved by wilson_lb, tie -> id)
    assert card_i in bank.prototypes and card_j in bank.prototypes and card_k in bank.prototypes

    # Assert: enabling edge i->j exists with lift >= 1.5, support >= 5, dt > 0
    fwd = [e for e in bank.edges.get(card_j, [])
           if e.kind == "enabling_lift" and e.src_id == card_i]
    assert len(fwd) == 1
    assert fwd[0].lift >= 1.5
    assert fwd[0].support >= 5
    assert fwd[0].mean_dt > 0

    # Assert: cyclic triple — the k->i candidate is rejected, graph stays acyclic
    assert not any(e.kind == "enabling_lift" and e.src_id == card_k and e.dst_id == card_i
                   for e in all_edges(bank))
    assert any(r.kind == "enabling_lift" and r.src_id == card_k and r.dst_id == card_i
               for r in bank.rejected_edges)
    assert is_acyclic(enabling_adjacency(bank))

    # -- Scenario 2: reversed time ordering (j unlocks BEFORE i) -> no i->j edge.
    store_rev = EpisodicEvidenceStore()
    u_d = _onehot(4, 128)
    ep = 0
    for _ in range(5):
        add_unlock_episode(store_rev, ep, ACTION, [([], [J]), ([J], [I])], u_d); ep += 1
    for _ in range(15):
        add_unlock_episode(store_rev, ep, ACTION, [([], []), ([], [])], u_e); ep += 1

    bank_rev = RelationHypothesisBank()
    bank_rev.consolidate_from_evidence(store_rev)

    rev_i, rev_j = f"card_a{ACTION}_e{I}_c0", f"card_a{ACTION}_e{J}_c0"
    assert rev_i in bank_rev.prototypes and rev_j in bank_rev.prototypes
    assert not any(e.kind == "enabling_lift" and e.src_id == rev_i and e.dst_id == rev_j
                   for e in all_edges(bank_rev))
    # The temporal signal points the other way: j -> i is the learned direction.
    back = [e for e in bank_rev.edges.get(rev_i, [])
            if e.kind == "enabling_lift" and e.src_id == rev_j]
    assert len(back) == 1 and back[0].lift >= 1.5


def test_produces_need_edges():
    # Arrange: a producer card (positive delta on dim 0) and a consumer card
    # needing dim 0.
    bank = RelationHypothesisBank()
    effect = [0.0] * 16
    effect[0] = 1.0
    producer = make_card(prototype_id="card_producer", action=2,
                         effect_delta=effect, produces=[[0.0, 1.0]])
    consumer = make_card(prototype_id="card_consumer", action=3,
                         needs=[[0.0, 2.0]], precondition_state=[2.0] + [0.0] * 15)
    bank.prototypes["card_producer"] = producer
    bank.prototypes["card_consumer"] = consumer

    # Act
    n_edges = bank.refresh_edges()

    # Assert: produces_need edge exists, dim resolved, producer discoverable
    assert n_edges >= 1
    edges = bank.edges.get("card_consumer", [])
    matching = [e for e in edges
                if e.kind == "produces_need" and e.src_id == "card_producer" and e.dim == 0]
    assert len(matching) == 1
    producers = bank.producers_for_dim(0)
    assert any(p.prototype_id == "card_producer" for p in producers)
    assert not any(e.dst_id == "card_producer" and e.src_id == "card_consumer"
                   for e in all_edges(bank))


# ---------------------------------------------------------------------------
# Persistence (v2 round-trip + v1 tolerance)
# ---------------------------------------------------------------------------

def test_save_load_v2_and_v1_compat(tmp_path):
    # Arrange: full bank — cards + produces_need edge + ledger + members + stats.
    bank = RelationHypothesisBank()
    effect = [0.0] * 16
    effect[0] = 1.0
    producer = make_card(
        prototype_id="card_producer", action=2, effect_delta=effect,
        produces=[[0.0, 1.0]], event_dim=2, unlock_prob=0.75, wilson_lb=0.6,
        visual_dispersion=0.9, status="confirmed", success_offline=6.0,
        success_online=1.0, created_from="offline",
        achievement_delta=[1.0 if d == 2 else 0.0 for d in range(22)],
    )
    consumer = make_card(prototype_id="card_consumer", action=3,
                         needs=[[0.0, 2.0]], precondition_state=[2.0] + [0.0] * 15)
    bank.prototypes["card_producer"] = producer
    bank.prototypes["card_consumer"] = consumer
    bank.refresh_edges()
    bank.evidence_count = 42
    bank.session_attempts = {"card_producer": 3}
    bank.unlock_ledger = {
        7: [(0, frozenset({3}), frozenset({3})), (1, frozenset({3}), frozenset({9}))],
    }
    bank.episode_first_unlock = {7: {3: 0, 9: 1}}
    bank._member_latents["card_producer"] = [[1.0] + [0.0] * 127]

    bank_file = tmp_path / "bank_v2.json"
    bank.save(bank_file)

    # Act 1: round-trip into a fresh bank
    fresh = RelationHypothesisBank()
    fresh.load(bank_file)

    # Assert 1: cards, edges, ledger and bookkeeping all survive
    assert set(fresh.prototypes.keys()) == {"card_producer", "card_consumer"}
    src_card = fresh.prototypes["card_producer"]
    assert src_card.event_dim == 2
    assert src_card.produces == [[0.0, 1.0]]
    assert src_card.unlock_prob == pytest.approx(0.75)
    assert src_card.wilson_lb == pytest.approx(0.6)
    assert src_card.visual_dispersion == pytest.approx(0.9)
    assert src_card.status == "confirmed"
    assert src_card.success_count == pytest.approx(10.0)
    assert src_card.success_offline == pytest.approx(6.0)
    assert src_card.success_online == pytest.approx(1.0)
    assert src_card.created_from == "offline"
    assert src_card.schema_version == "phase2_card_v2"
    assert src_card.achievement_delta[2] == pytest.approx(1.0)
    dst_card = fresh.prototypes["card_consumer"]
    assert dst_card.needs == [[0.0, 2.0]]
    assert fresh.edges.get("card_consumer", []) and \
        any(e.kind == "produces_need" and e.src_id == "card_producer" and e.dim == 0
            for e in fresh.edges["card_consumer"])
    assert fresh.evidence_count == 42
    assert fresh.session_attempts == {"card_producer": 3}
    assert fresh.unlock_ledger[7] == [(0, frozenset({3}), frozenset({3})),
                                      (1, frozenset({3}), frozenset({9}))]
    assert fresh.episode_first_unlock[7] == {3: 0, 9: 1}
    assert fresh._member_latents["card_producer"] == [[1.0] + [0.0] * 127]
    assert fresh.bank_sha256 == bank.bank_sha256

    # Act 2: minimal v1-format JSON (old field names only, no edges key)
    v1_bank = {
        "version": "phase2_relation_bank_v1",
        "prototypes": {
            "legacy": {
                "prototype_id": "legacy",
                "action": 7,
                "precondition_key": [0.1] * 128,
                "precondition_state": [0.2] * 16,
                "effect_delta": [0.3] * 16,
                "postcondition_key": [0.4] * 128,
                "success_count": 9,
                "failure_count": 1,
                "mean_return": 1.5,
                "mean_duration": 1.0,
                "confidence": 0.8,
            }
        },
    }
    v1_file = tmp_path / "bank_v1.json"
    v1_file.write_text(json.dumps(v1_bank))

    fresh_v1 = RelationHypothesisBank()
    fresh_v1.load(v1_file)

    # Assert 2: loads with defaults; edges absent -> rebuilt via refresh_edges
    assert len(fresh_v1) == 1
    legacy = fresh_v1.prototypes["legacy"]
    assert legacy.event_dim is None
    assert legacy.needs == []
    assert legacy.produces == []
    assert legacy.consumes == []
    assert legacy.unlock_prob == 0.0
    assert legacy.success_online == 0.0
    assert legacy.wilson_lb == pytest.approx(wilson_interval(9.0, 10.0)[0])
    assert legacy.status == "confirmed"
    assert isinstance(fresh_v1.edges, dict)
    # refresh_edges ran without error on the v1 bank
    assert all(isinstance(e, RelationEdge) for el in fresh_v1.edges.values() for e in el)
