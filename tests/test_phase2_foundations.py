"""Foundation tests (plan review fixes): vitals labels, encoder convention,
bank-audit gate, final-transition inclusion. Plan §4 tests 10/11/13/14."""
import json

import numpy as np
import pytest
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner
from training.sequence_buffer import EpisodeBuffer

NUM_ACH = len(CRAFTER_ACHIEVEMENTS)


def test_vitals_labels_absolute_in_state_current():
    """Vitals dims 12-15 start at 9.0 and accumulate deltas (plan Stage 1)."""
    T = 5
    buf = EpisodeBuffer()
    buf.add_episode(
        obs=np.zeros((T, 64, 64, 3), dtype=np.uint8),
        actions=np.arange(T),
        rewards=np.zeros(T, dtype=np.float32),
        dones=np.zeros(T, dtype=bool),
        events=np.zeros((T, NUM_ACH), dtype=np.uint8),
        mask_before=np.zeros((T, NUM_ACH), dtype=np.uint8),
        state_delta=np.tile(np.array([-0.5, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.25, 0], dtype=np.float32), (T, 1)),
    )
    ep = buf.episodes[0]
    sc = ep["state_current"]
    # Resource dim 0 (delta -0.5/step): relative count, still deviation-style
    assert sc[0, 0] == 0.0
    assert sc[1, 0] == pytest.approx(-0.5)
    # Vitals dim 12 (health): delta is 0 in this fixture -> stays at the 9.0 baseline
    assert sc[0, 12] == pytest.approx(9.0)
    assert sc[2, 12] == pytest.approx(9.0)
    # Vitals dim 14 (drink, delta -0.25/step): absolute — 9.0 minus accumulated deltas
    assert sc[0, 14] == pytest.approx(9.0)
    assert sc[1, 14] == pytest.approx(9.0 - 0.25)
    assert sc[2, 14] == pytest.approx(9.0 - 0.5)
    assert sc[1, 15] == pytest.approx(9.0)  # energy unchanged (delta 0)


def test_mpc_encodes_with_target_encoder(monkeypatch):
    """Encoder-convention test (plan Stage 1): the MPC planning path must encode with
    use_target=True — the same space as event-head training, belief updates, and bank keys."""
    model = JEPAv3()
    model.eval()
    calls = []

    original_encode = model.encode

    def spy_encode(obs, use_target=False):
        calls.append(use_target)
        return original_encode(obs, use_target=use_target)

    monkeypatch.setattr(model, "encode", spy_encode)

    planner = RecurrentMPCPlanner(
        model=model, horizon=2, num_candidates=4, cem_iterations=1, strict_inference=True,
    )
    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    planner.plan(frame, belief)

    assert calls, "encode was never called"
    assert all(calls), f"MPC must encode with use_target=True, got {calls}"


def test_bank_audit_gate_passes_clean_bank(tmp_path):
    """Audit invariants (plan §4 test 13): a structurally clean bank exits 0."""
    from scripts.audit_phase2_bank import audit

    bank = {
        "version": "phase2_relation_bank_v2",
        "prototypes": {
            "card_a5_enone_c0": {
                "prototype_id": "card_a5_enone_c0", "action": 5, "event_dim": None,
                "needs": [], "produces": [[0, 1.1]], "consumes": [],
                "status": "confirmed", "wilson_lb": 0.7,
                "achievement_delta": [0.0] * 22,
                "precondition_state": [0.0] * 16, "effect_delta": [1.1] + [0.0] * 15,
            },
            "card_a8_e20_c0": {
                "prototype_id": "card_a8_e20_c0", "action": 8, "event_dim": 20,
                "needs": [[0, 1.6]], "produces": [], "consumes": [[0, 1.6]],
                "status": "confirmed", "wilson_lb": 0.75,
                "achievement_delta": [0.0] * 20 + [1.0] + [0.0],
                "precondition_state": [1.6] + [0.0] * 15, "effect_delta": [-1.6] + [0.0] * 15,
            },
        },
        "edges": {
            "card_a8_e20_c0": [
                {"src_id": "card_a5_enone_c0", "dst_id": "card_a8_e20_c0",
                 "kind": "produces_need", "dim": 0, "lift": 1.0, "support": 0, "mean_dt": 0.0}
            ],
        },
    }
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank))
    assert audit(p) == 0


def test_bank_audit_gate_fails_on_vitals_precondition(tmp_path):
    """A bank with a vitals precondition must be rejected (exit 1) — the Stage 2 gate."""
    from scripts.audit_phase2_bank import audit

    bank = {
        "version": "phase2_relation_bank_v2",
        "prototypes": {
            "card_bad": {
                "prototype_id": "card_bad", "action": 8, "event_dim": 20,
                "needs": [[0, 1.6], [15, 0.6]],  # energy precondition — the N3 defect
                "produces": [], "consumes": [[0, 1.6]],
                "status": "confirmed", "wilson_lb": 0.75,
                "achievement_delta": [0.0] * 22,
                "precondition_state": [1.6] + [0.0] * 14 + [0.6],
                "effect_delta": [-1.6] + [0.0] * 15,
            },
        },
        "edges": {},
    }
    p = tmp_path / "bank_bad.json"
    p.write_text(json.dumps(bank))
    assert audit(p) == 1


def test_consolidation_includes_final_transition(tmp_path):
    """N10 regression test: the episode-final step (and its reward) must enter the
    evidence store — transitions_ingested == ep_len, label coverage 1.0."""
    from training.run_phase2_memory_consolidation import consolidate_phase2_memory

    T = 6
    obs = np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8)
    actions = np.arange(T) % 17
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = 1.0  # terminal achievement reward — must reach the MC returns
    dones = np.zeros(T, dtype=bool)
    dones[-1] = True
    events = np.zeros((T, 22), dtype=np.uint8)
    events[-1, 18] = 1  # unlock on the final step
    mask_before = np.zeros((T, 22), dtype=np.uint8)

    shard = tmp_path / "shard_final_test.npz"
    np.savez(
        shard, obs=obs, actions=actions, rewards=rewards, dones=dones,
        events=events, mask_before=mask_before, episode_lengths=np.array([T]),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train_shards": [{"path": str(shard)}]}))

    diag = consolidate_phase2_memory(
        checkpoint_path="checkpoints/jepa_crafter_v3_phase2.pt",
        manifest_path=str(manifest),
        output_bank_path=str(tmp_path / "bank.json"),
        diagnostic_path=str(tmp_path / "diag.json"),
        max_episodes=1,
        min_cluster_size=2,
        device="cpu",
    )
    assert diag["transitions_ingested"] == T  # was T-1 before the N10 fix
    assert diag["label_coverage"] == 1.0
    assert diag["degraded_label_steps"] == 0
