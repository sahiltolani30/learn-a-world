# Phase 2 Causal Memory Redesign — Final Implementation Plan

**Status:** Approved for implementation (multi-agent verified: independent review → plan draft → senior-ML cross-review, 2026-10-28)
**Implementer:** Agy | **Spec source:** this document (the `code-review` Spec axis grades every stage diff against its section here)
**Benchmark context:** Phase 2 memory attached to MPC regressed Crafter 4.31% → 3.50% (`results/screen_v3_ablation_c_active_50ep.json` vs `results/screen_v3_phase2_compositional_50ep.json`, same checkpoint/calibrator/seeds).

**Non-negotiable constraints:**
- ZERO hardcoding: no action names ("act5 harvests"), no item semantics, no recipe amounts in logic. Array index spaces (17 actions, 16 state dims, 22 achievements) are interface constants only.
- Strict inference at playtime: no engine inventory/achievements/position. Playtime unlock labels come from the belief (event head + reward fallback), never `info['achievements']` (protocol: scoring-only).
- Every stage lands as its own commit with green tests and a two-axis review gate (Stage N diff reviewed with fixed point = Stage N−1 commit; Spec axis = this doc).

---

## 1. Root Cause (systematic-debugging Phase 1 artifact — complete; no fixes may skip past it)

**The Iron Law held:** every fix below traces to a verified root cause, not a symptom. The regression is a **triad** — no single fix suffices:

| # | Defect | Evidence (file:line) | Verified by |
|---|--------|----------------------|-------------|
| F1 | Consolidation discards shard `events`/`mask_before` → prototypes achievement-blind | `training/run_phase2_memory_consolidation.py:113-119,173-183`; true unlock = `events[t]·(1−mask_before[t])` (608 unlocks, 588 exact +1.0) | Shard decode, model-free |
| F2 | Spurious preconditions: 0.15-quantile + 0.45 clip copies correlated items into requirements | `planning/causal_memory.py:310-316`; model-free: wood 100% vs 18% base at table unlocks (true cause), sapling 86% vs 83% (coincidence) | Shard statistics |
| F3 | Static lifetime MC returns on one-time achievements | `causal_memory.py:108-145,320` → ranking `compositional_planner.py:82-86` → MPC pull weight `mpc_v3.py:213,449`; place_plant shows 1.65–1.92 for one-time +1.0 | Repro + code path |
| F4 | Confidence poisoned by visual-sim failure matching (air-swings) | `causal_memory.py:266-273,322-333`; act5 avg 517.8 failures, conf 0.081; 0.85-cosine cannot discriminate "facing tree" from "in forest" (6% of random pairs ≥ 0.85) | Repro + measured latents |
| F5 | Flat bag, no edges, 1-step greedy exit + hardcodes | `compositional_planner.py:75` (`action >= 5`), `:100` (READY visual bypass), `:126` (`action == 5` enforcement) | Code + **failing test** |
| N1 | **`tests/test_phase2_compositional_planner.py::test_compositional_backward_chaining` FAILS today** — the :126 hardcode excludes the action-7 producer; the producer card itself then becomes READY via the :100 bypass | Run: `.venv/bin/python -m pytest tests/test_phase2_compositional_planner.py -x -q` | Pytest (canary) |
| F6 | Report's DAG rule hole: place_table only *consumes* wood, no table dim exists → table→pickaxe edge impossible under `Δx>0 ∧ x_pre>0` | 16-d = 6 resources + 6 tools + 4 vitals; engine: `nearby [table]` 3×3 | Engine code |
| F7 | MPC amplification: 50% step-0 injection, per-horizon-step subgoal pull weighted by static `mean_return`, unconditional 0.10 nav pull | `mpc_v3.py:313-337,441-458`; m_bin never passed to `plan_subgoal` (`:204-207`) | Code path |
| F8 | Bank frozen during eval — no cross-game learning (the human-loop blocker) | `register_interaction_outcome` (`causal_memory.py:351-370`) has zero non-test callers; eval loop computes `z_t` (`screen_v3_50eps.py:301`) and `newly_unlocked` (`agent_v3.py:160-177`) and discards both | Code path |
| N3/N8 | Vitals in 16-d labels are **deviation-from-9** (≤ 0 always) → vitals preconditions meaningless; ⚠️ local repro checkpoint `jepa_crafter_v3_phase2.pt` has **no trained state_readout_head** (random init) → local-repro numeric precondition values invalid (mechanism conclusions unaffected; real bank is Drive-only) | `training/sequence_buffer.py:126-131`; checkpoint key listing | Code + torch.load |
| N7 | Belief reward-fallback pins argmax dim permanently (m_bin ratchet) — a guess, not an observation | `agent_v3.py:170-177`; imagination reward requires `m_bin < 0.5` (`mpc_v3.py:419`) | Code path |
| N9 | Three encoder conventions coexist: MPC plans with `use_target=False` (`mpc_v3.py:154`); event/health-head training + belief updates + consolidation use target features | `trainer_v2.py:101,235`, `screen_v3_50eps.py:301`, `run_phase2_memory_consolidation.py:154` | Code path (EMA drift measured small: cos ≈ 0.995 — the problem is convention, not drift) |

**Regression protocol for the future (process rule):** any future screening regression goes through the four debugging phases (reproduce → isolate component boundary with instrumentation → root cause → then fix). Heuristic if-else patches are symptom fixes and are forbidden.

---

## 2. Design: the Causal Card

One card per **(action, unlocked event)**. The card is the atom; the bank = cards + causal graph; the planner reads cards+graph; MPC consumes planner output; the online loop writes cards.

```
CARD: (action a, event_dim e)
  WHEN      visual centroid z_pre (+ visual_dispersion)        — learned
  NEEDS     [(dim, amount)]  surviving dims only               — enrichment-tested
  COSTS     [(dim, amount)]  |negative deltas|                 — learned (budget composition)
  PRODUCES  [(dim, amount)]  positive deltas                   — learned
  UNLOCKS   e; unlock_prob = Laplace-smoothed P(e fires | applied) — event-grounded
  ENABLES   graph edges (produces_need + enabling_lift)        — learned
  EVIDENCE  success/failure counts, offline weight 1.0 / online weight 0.5,
            trust = Wilson lower bound; mean_return = diagnostic ONLY
```

Five properties of a proper relation (the acceptance philosophy): (1) causal, not correlational — enrichment test; (2) quantified — amounts from success quantiles; (3) one-time aware — marginal value → 0 via m_bin; (4) composable — NEEDS+COSTS+ENABLES chains with summed budgets ("3 wood = 2 table + 1 pickaxe"); (5) falsifiable — revised online.

---

## 3. Build Stages

### Stage 0 — Tiny-fix commit (the canary goes green) — `_no dependencies_`

**Files:** `planning/compositional_planner.py`
1. Delete the `p.action == 5 if missing_dim < 6 else True` hardcode (line 126). Producers are any card whose effect covers the deficit — from data, not action identity.
2. Narrow the READY bypass (line 100): visual-free READY only for cards with **zero** requirements; a card with requirements and no visual match is NOT ready (engine needs the affordance context, e.g. table adjacency).

**Gate:** `pytest tests/test_phase2_compositional_planner.py` fully green (N1 regression test).
**Review gate:** two-axis, fixed point = pre-Stage-0 commit. Spec axis checks *only* these two lines changed (f6dd296 — which added the hardcode — is the historical example of scope creep this gate prevents).

### Stage 1 — Foundations (before any re-consolidation)

**Files:** `training/sequence_buffer.py` or label construction, `models/heads.py` retrain script, `training/run_phase2_memory_consolidation.py`
1. **Vitals labels → absolute scale** (offset +9 on dims 12–15, matching `inventory_dict_to_vector`), or ban dims 12–15 from precondition generation. Prefer absolute + a hard bank-audit assertion of zero vitals preconditions.
2. **Encoder unification:** standardize inference on the **target** encoder — change `mpc_v3.py:154` to `use_target=True` (matches event/health-head training and belief updates); **retrain the readout head on target-encoder features** (head-only, frozen base — the head was trained on online features).
3. **Provenance (pytorch-patterns: checkpoint completeness):** every artifact carries `checkpoint_sha256`, encoder flag, readout-gate metrics (Gate-5 MSE), label convention. Mark `scratch/phase2_repro/*` invalid for numeric claims.

**PyTorch rules for this stage (from pytorch-patterns):** `model.eval()` during all consolidation/eval encoding (disables dropout/BN drift); device-agnostic (`cuda`-if-available); seeds set (`torch.manual_seed` + numpy) for deterministic re-consolidation; `weights_only=True` where the pickle allows; `.to(device)` once; atomic artifact writes (tmp + rename).

**Tests:** `test_vitals_labels_absolute` (label construction round-trip); `test_encoder_convention_single` (bank keys, serving queries, x_hat inputs share one encoder flag — grep-level assertion fixture).

### Stage 2 — Re-consolidate + zero-GPU bank audit

Colab with locked checkpoint `jepa_crafter_v3_state_readout.pt` (record SHA). Output `checkpoints/phase2_causal_memory_bank_v2.json` + diagnostic.
**Audit script (no GPU) gates screening:** every place_table-family card `needs ⊆ [[wood, amt]]`, `amt ∈ [0.5, 2.0]`; zero dims from {sapling, health, food, drink, energy} bank-wide; ≥1 action-5 wood-producer card exists with a Wilson-separated wood need; no card both needs and produces the same dim.

### Stage 3 — Card schema v2 + event-grounded consolidation

**Files:** `planning/causal_memory.py`, `training/run_phase2_memory_consolidation.py`

Schema (keep legacy fields for v1-bank load compatibility; `save` v2 / `load` tolerant):
```python
TransitionPrototype: + event_dim: Optional[int]      # 0..21; None = event-free
                     + needs/produces/consumes: List[List[float]]   # [dim, amount] pairs
                     + unlock_prob, wilson_lb, visual_dispersion: float
                     + status: "hypothesis" | "confirmed"            # hypothesis if weighted successes < 3
                     + success_offline/online, failure_offline/online: float   # weighted evidence
EpisodicEvidence:    + event_vec, mask_before_vec: Optional[List[int]]
                     + source: "offline" | "online"
RelationHypothesisBank: + edges, online_evidence (cap 200k), session_attempts, bank_sha256
                        + learn_from_episode(steps), refresh_edges(), producers_for_dim(d)
```

Consolidation v2:
- Ingest `events`/`mask_before`; per-step `u_t = events[t]·(1−mask_before[t])` (missing keys → fallback `u_t = [1]*22 if reward ≥ 0.9 else zeros`, flagged degraded).
- Cluster successes by **(action, fired event)**, sub-cluster by visual cosine ≥ 0.85 → stable IDs `card_a{a}_e{e}_c{k}`; event-free productive steps → `card_a{a}_enone_c{k}`.
- **NEEDS enrichment test** (replaces quantile+clip): for dim d, `q_d` = 15th-pct of successes; keep iff (a) support ≥ 3, (b) Wilson-80% intervals of s_d over successes vs f_d over near-miss failures (matched vs cluster **members**, not bare centroid; |F| < 2 → dim **untestable**, stays hypothesis) do not overlap, (c) gap s_d − f_d ≥ 0.20, (d) BH-FDR q = 0.10 across the 16 dims per card. Count mid-band transitions (0.25 ≤ |Δ| < 0.35, reward ≤ 0) as near-misses — today they fall in neither set. Validated on real statistics: wood 7/7 → LB 0.81 vs 0.18 → **kept**; sapling 6/7 → LB 0.62 vs 0.83 → **killed**.
- Amounts: clip(success-quantile, 0.5, 4.0); **never raised online from single observations** (the F2 trap).
- `effective (k,n)` = offline·1.0 + online·0.5; `confidence := max(0.05, wilson_lb(k,n))`. `mean_return` computed for diagnostics only and **deleted from every consumer**.

### Stage 4 — Causal graph

- `produces_need` edges derived lazily from produces/consumes.
- `enabling_lift` edges from temporal lift: `lift = P(j unlocks | i unlocked earlier) / P(j unlocks) ≥ 1.5`, support ≥ 5, `mean_dt > 0`, **Kahn acyclicity check on insert** (reject + log otherwise). Measured ground truth: wood→table lift 5.58.
- `refresh_edges()` at end of consolidation and after every `learn_from_episode` (O(T), cheap).

### Stage 5 — Planner rewrite

**Files:** `planning/compositional_planner.py` (rewrite `plan_subgoal`; delete dead `compute_action_prior` — zero callers)

```python
def plan_subgoal(self, obs_latent, grounded_state, m_bin=None) -> Optional[dict]
```
- **Top-level targets: event-grounded cards only, still locked per m_bin** (`m_bin[event_dim] < 0.5`). Event-free resource cards are reachable ONLY as chain steps behind a marginal event root — their value comes from downstream demand, never their own productivity (else the farming loop returns).
- Ranking: `score = v_marginal + UCB`, `UCB = min(0.5, 0.3·sqrt(2·ln(1+N_total)/(1+n_p)))` (N_total = bank attempts, grows across games; bonus ≈ 0.5 for untouched, 0.21 at n=5, 0.11 at n=20).
- Multi-step DFS (depth 3) over edges with **cost summation**: `applications(d) = ceil(amount / producer_delta)`.
- `chain_score = v_marginal · Π(wilson_lb of chain) · 0.9^depth / (1 + 0.15·total_cost)`; evaluate top-K=3 (today: top-1).
- READY gate: `sim ≥ sim_threshold OR visual_dispersion < 0.70` (dispersion = the card's own successes had varied visuals → cue empirically uninformative *for that card*; learned, not hardcoded).
- Return None when nothing is marginal → MPC goes silent. Output keeps legacy keys for MPC transition + adds `marginal_value`, `chain`, `total_cost`, `chain_value`, `ucb_bonus`; `expected_return := chain_value` (kills the mean_return aliasing debt).

### Stage 6 — MPC retune

**Files:** `planning/mpc_v3.py`
1. Pass `m_bin_0` into `plan_subgoal` (line 204-207) — prerequisite for everything above.
2. Subgoal shaping: `action_reward = (a_t == prod_act)·0.25·sim_cue·scale`, `nav = 0.05·sim_cue·scale` (was 0.10 unconditional), READY: `0.35·sim_cue·scale`, with `scale = min(chain_value, 1.0)`. Rationale: shaping's job is tie-breaking among near-tied candidates; injection ratio does the heavy lifting.
3. `memory_inject_ratio` 0.50 → **0.25**, gated on `chain_value ≥ 0.3`.
4. **Clip the transition consequence term**: `res_prog ≤ 0.25 total/step` (currently unbounded, can reach ~2.0 and out-shout both Q and shaping — the term most capable of driving harvest spam). Verify the screening checkpoint actually contains a trained `state_delta_head` and gate the term on it (the local one has none).

### Stage 6.5 — 5-episode smoke (cheap de-risk before the 50-ep run)

Champion seeds, `subgoal_trace` logging. Abort if: subgoal active > 60% of steps, or a single action receives > 30% of injections (sleep-loop detector), or score < ablation-C − 1.5.

### Stage 7 — 3-arm screening, same commit (provenance hygiene)

| Arm | Config | Purpose |
|---|---|---|
| A | ablation C (no bank) | baseline, re-run on THIS commit |
| B | bank v2 + `--online_learning` | the candidate |
| C | bank v2, frozen | isolates the online-loop contribution |

Seeds 2050–2099, protocol `strict_visual_jepa_clean_v2` locked. Provenance per run: git_commit, checkpoint/calibrator/bank SHA256s, `online_learning` flag, per-episode `subgoal_trace` + `online_evidence_counts` (guarantees no two result files are ever byte-identical again — the current run2/run1 files are duplicates).

### Stage 8 — Online cross-game learning loop (last: riskiest, depends on stable IDs)

**Files:** new `planning/playtime_recorder.py`, `scripts/screen_v3_50eps.py`, `planning/agent_v3.py`

- Recorder captures per step: `z_t` (already computed at `screen_v3_50eps.py:301` — stop discarding it; encode with `use_target=True` matching consolidation), `x_hat` via `model.predict_state` (do NOT rely on `debug["state_estimate"]` — absent when `use_transition_model=False`), `mask_before` copied **before** `belief.update`, `newly_unlocked` from `belief.history[-1]`.
- `learn_from_episode`: **match-and-update, never re-cluster** (stable card IDs — `card_a{a}_e{e}_o{n}` for online-born): confirmed successes/denials update Wilson stats; needs amounts never updated online.
- **Reward-fallback quarantine (N7):** expose the fallback provenance in `belief.history` (`agent_v3.py`, 3-line change); fallback-pinned dims may create *hypothesis* cards at reduced prior, **never confirm** — confirmation requires event-margin evidence.
- Per episode: `bank.learn_from_episode(...)` → `bank.refresh_edges()` → `bank.save()` (atomic). Bank persists across episodes → game N's lessons shape game N+1 (the human loop: discover → remember → apply → discover deeper).
- CLI flags: `--online_learning` (default off; Arm C is the frozen control), `--online_bank_path`, `--resume_bank`.

### Stage 9 — Acceptance criteria

- **AC1:** bank audit passes (place_table needs wood-only; zero vitals preconditions bank-wide; ≥1 action-5 wood producer).
- **AC2:** no `mean_return` consumers (grep-clean in `compositional_planner.py`, `mpc_v3.py`); harvesting appears in top-3 ranked chains in ≥1 test.
- **AC3:** `collect_wood → place_table` enabling edge present, lift ≥ 3, support ≥ 5.
- **AC4 (primary):** Arm B ≥ 4.31% — non-inferiority (one-sided 95% CI of paired reward diff vs Arm A excludes −0.5). Stretch: ≥ 5.0%.
- **AC5:** ≥1 episode's `subgoal_trace` shows HARVEST(wood) → READY(table) → READY(pickaxe) — depth-3 chaining exercised live.
- **AC6:** Arm C < Arm B on make_wood_pickaxe rate; the 20% pickaxe gate applies **only if** the online bank shows make_wood_pickaxe confirmed-with-event-margin evidence ≥ 5 by episode 25 (evidence-growth curve reported either way).
- **AC7:** MPC silence verified (all targets unlocked → no injection, `active_subgoal_status == ""`).
- **AC8:** no two result files byte-identical; all arms share one `git_commit`; every run records bank SHA.

---

## 4. Test Suite (python-testing-patterns applied)

**Organization:** extend `tests/test_phase2_causal_memory.py` + `tests/test_phase2_compositional_planner.py`; new `tests/test_phase2_cards.py`; shared fixtures in `tests/conftest.py`. All unit tests CPU-only and fast; Colab-only checks marked `@pytest.mark.slow`.

**Naming:** `test_<unit>_<scenario>_<expected>` (e.g. `test_plan_subgoal_with_locked_event_returns_none`).

**conftest fixtures:** `make_card(**overrides)` (one-line card builder with sane defaults), `bank_with(cards)`, `enrichment_case_wood` (7/7 vs f 0.18, |F|=50 — the literal real statistics), `enrichment_case_sapling` (6/7 vs 0.83), `belief_stub` (controllable m_bin + fallback flags), `event_head_mock` (Mock with `side_effect` sequences for fallback-pin tests).

**Updated existing tests:** `test_relation_hypothesis_consolidation` (+`event_vec`, `success_count == approx(3.0)` int→float, `event_dim is not None`); `test_memory_lifecycle_save_and_reload` (+v2 fields round-trip + v1 file still loads); `test_compositional_ready_to_execute` (+locked-m_bin → None scenario + dispersion branch).

**New tests (16 total):** 1 currently-failing backward-chaining goes green (Stage 0 canary) · 2 marginal suppression (`m_bin` locks → None) · 3 UCB ordering (parametrized over n_p: untouched beats weak-confirmed; cap never exceeded) · 4 two-step chain with cost summation (`total_cost == 2.0`) · 5 enrichment on real statistics (wood kept / sapling killed — the F2 tombstone) · 6 incremental learn stability (same card id updated, never re-clustered) · 7 MPC silence when all unlocked · 8 enabling edge from synthetic lift (+ reversed-time → no edge + cyclic triple rejected by Kahn) · 9 Wilson vs naive confidence (1/500 failures still ranks via UCB) · 10 vitals labels absolute · 11 encoder convention single · 12 fallback-provenance exclusion (Mock `side_effect` fallback → hypothesis only, never confirmed) · 13 bank-audit invariants · 14 final-transition inclusion (`range(ep_len)`) · 15 no-top-level-resource-target (bank of only event-free cards → None) · 16 recorder never touches `info['achievements']` (raises if handed one).

Coverage gate: `pytest --cov=planning --cov-fail-under=85` on the touched modules.

---

## 5. Review Gate Protocol (code-review skill, per stage)

At each stage boundary, run the two-axis review on `git diff <prev-stage-commit>...HEAD`:
- **Standards axis:** repo-documented invariants live in module docstrings (strict-inference guards, zero-hardcoding rule above) + smell baseline. Known smell targets for this refactor: *Primitive Obsession* (action ints standing in for affordance concepts — resolved by event_dim on the card), *Data Clumps* (the subgoal dict's producer/target aliasing — resolved by explicit `chain` field), *Duplicated Code* (READY/producer logic in planner + prior).
- **Spec axis:** the stage's section in THIS document is the spec; flag anything implemented but not specified (scope creep — the exact failure mode of commit f6dd296) and anything specified but missing.
Aggregate both axes unmerged; a stage does not land with an open Spec-axis finding.

---

## 6. Risks

1. **Belief-label noise** → online weight 0.5 + confident-firing creation gate + fallback quarantine + Arm C control.
2. **Small samples** (7 table, ~1 pickaxe unlocks offline) → min-support/Wilson/BH parameters (validated in §3 Stage 3); hypothesis status + UCB re-testing; AC6 explicitly evidence-conditioned.
3. **Residual bank defect amplified by MPC** → shaping scaled ≤ 0.30/step, injection 0.25 gated, res_prog clipped, 5-ep smoke aborts.
4. **Needs drift online** → impossible by construction (amounts only re-derived offline).
5. **Enabling-edge cycles** → mean_dt > 0 + Kahn check.
6. **Checkpoint provenance** → SHA pinned everywhere; random-head artifacts quarantined for mechanism testing only.
