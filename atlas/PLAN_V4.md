# Project ATLAS v4 — PRESCRIPTIVE SPEC: Queryable Agent + V3.1 Bottleneck Repair

**Audience:** implementing agent. Follow literally. No redesign, no substitution. If a mechanism
is not specified to the formula level in this document, implement it as DISABLED and report the
gap — never fill it from world knowledge (standing META-RULE from V3).
**Date:** 2026-10-30. Basis, two layers:
1. Three post-mortems (v1 turret, v2.1 starvation, v2.2 monopoly) were each solved by
   *inferring* behavior from indirect artifacts weeks after the fact. Part B makes the agent's
   inner life directly queryable so root causes are lookups, not reconstructions.
2. The V3 empirical campaign report (`docs/ATLAS_V3_EMPIRICAL_RESULTS_AND_BOTTLENECK_REPORT.md`,
   it119→212, 752 episodes) measured four cognitive bottlenecks. Part G fixes them —
   **using learned signals only** (see the Integrity Note below).

**V3 results this plan builds on (measured, do not re-derive):** score 1.28% → **1.76%**
(peak dev 2.38%), HER CE 1.71 → **0.95–1.09**, survival 164.9 → **180.8**, first honest
`place_table` (2.0% screen; champion `it144_seed24407` reward 6.1, 373 steps), material gate
active and 100% learned. Bottlenecks (all measured): B1 arrival-cosine trap (mean cos 0.311,
only 0.1% of steps ≥ 0.85, 18.4% ≥ 0.60 → NAVIGATE times out into WANDER); B2 the 2-vs-3 wood
barrier (3 wood in only 0.2% of 1,704 episodes; table placement consumes the 2 wood, pickaxe
needs 1 more); B3 single-unlock drop-out (first `collect_wood` locks m_soft → GoalBank stops
prioritizing wood before the 2nd/3rd tree); B4 nightfall mortality (75.3% of deaths in steps
148–272; sleep-commitment deaths dominate; worst-case `it184_seed28407`: OptionNet locked
pressing `do` under ranged fire, risk 0.47 just under the 0.50 threshold → death at step 22).

**⚠ Integrity Note on Part G (read before implementing any G-item):** the report's Section 5
suggests constants ("3 wood", "0.40 threshold", "anchor 3×3"). Typing any of those numbers or
concepts as literals is a violation of AGENTS.md §1. Every G-item below re-derives the report's
intent from learned signals: chain costs from `StateDeltaHead` + the lift graph, thresholds
from calibration on the agent's own realized data, anchoring from the existing episode
artifact-view registry. Where the report's fix cannot be expressed learned, it is NOT
implemented in v4 and is listed as deferred.

**Allowed files (NEW unless named):** `atlas/trace.py` (NEW), `atlas/probes.py` (NEW),
`atlas/canaries.py` (NEW), `atlas/replay.py` (NEW), plus EDITS to `atlas/config.py`,
`atlas/arbiter.py`, `atlas/collect.py`, `atlas/selflearn.py`, `atlas/audit_pack.py`,
`atlas/screen.py`, `atlas/colab/run_atlas.py`, `atlas/tests/*` (Part B), and for Part G also
`atlas/goals.py`, `atlas/skills.py`, `atlas/belief.py`, `atlas/safety.py`. Nothing else.

**Standing prohibitions (unchanged):** never train/modify the frozen backbone; never use
achievement/action/item names in decision logic; never change eval seeds (2050–2099 locked,
6150–6159 dev); never edit env behavior; never let diagnostics influence behavior.

**INTEGRITY LAW FOR THIS PLAN (the one new rule):** everything built here is WRITE-ONLY
telemetry and OFFLINE analysis. The strings in `CRAFTER_ACHIEVEMENTS` / `INVENTORY_ITEMS` /
`CRAFTER_ACTION_NAMES` may be used ONLY inside `atlas/probes.py` and `atlas/replay.py` as
*display labels for diagnostics* (same status as journal reason strings), never in
`arbiter.py`, `goals.py`, or any head. Probe results are numbers reported to the operator;
no code branch may read them. An AST/grep test enforces this.

---

## Part A — EXISTING INTERFACES (verified 2026-10-30; do not re-derive, do not change semantics)

- `AtlasAgent.act(obs_uint8, reward_prev, step) -> (action:int, telemetry:dict)` —
  `arbiter.py`. Mode branches inside `act()`: PROBE (`probe_score` [17]), NAVIGATE
  (`option_net.act`), WANDER (softmax sample), FALLBACK (`beta_t * disagreement + prior`).
  Epsilon tail and safety override applied AFTER mode choice. Telemetry dict keys:
  `mode, goal_score, arrival_cos, psi, risk_intended, override_fired, r_int, reason,
  a_intended, a_final, goal_mode, option_budget, abort_reason`.
- `AtlasAgent` episode-local state: `mode, prev_mode, active_goal, goal_score, probe_id,
  probe_newly_unlocked, last_abort_reason, current_goal_mode, current_option_budget,
  material_subgoal_cycles, target_goal_pending, episode_artifact_views, episode_buffer`.
- `AtlasAgent.observe_result(obs, reward, done, discount)` — fills `episode_buffer[-1]`.
- `EpisodeCollector.run_episode(seed, iter_idx)` — `collect.py`. Per step it already extracts:
  telemetry (journal `step_record`), `positions_list` from `info["player_pos"]`, and writes
  shard npz with `obs, actions, rewards, dones, truncated, discounts, events, mask_before,
  inventory, positions, episode_lengths, seeds`. Journal entry keys: `iter, seed, game_id,
  length, total_reward, died, achievements, num_achievements, mode_counts, overrides_count,
  positions, moved_steps, moved_fraction, action_histogram, steps`.
- `AuditPackExporter` (`audit_pack.py`) — `record_decision(...)`, `record_frame(...)`,
  `export(current_iter, git_commit, cfg, cached_episodes, results_dir)`; writes
  `audit/manifest.json`, `latents_last2.npz`, `positions_last2.npz`, `decisions.jsonl`,
  `frames/`, `health_cards/`, 60 MB cap.
- `OptionNet` exposes `act(s, g, greedy)` and `logits(s, g) -> [B,17]` (PLAN.md §5.4).
  If `logits` is absent in `skills.py`, ADD it as the plain trunk forward before argmax —
  this is an allowed edit to `atlas/skills.py` for that purpose ONLY.
- `EpistemicNet.disagreement_all_actions(s) -> [B,17]`; `ContinueNet.risk_all_actions(s) -> [B,17]`;
  `backbone.event_logits_all_actions(s, m) -> [B,17,22]`.
- Shard targets available for probes (training-side ground truth, EXISTING fields, no new
  extraction needed): `positions` [T,2] (player_pos, float), `inventory` [T,16]
  (12 items log1p + 4 vitals /16), `events` [T,22] (bool), `mask_before` [T,22], `rewards`,
  `dones`. Cached latents per shard: `z` [T,128], `h` [T,256] (sidecar `*.latents.npz`).

## Part B — MECHANISMS (implement in this order; each with its verification)

### T1. Per-step decision trace (`atlas/trace.py` NEW; edits: `config.py`, `arbiter.py`, `collect.py`, `selflearn.py`)

**T1a. Config.** Add:
```python
@dataclass
class TraceConfig:
    enabled: bool = True
    dir_name: str = "trace"            # -> atlas_runs/<run_id>/trace/
    max_folder_mb: float = 500.0       # prune oldest trace_it*.jsonl when exceeded
```
Add `trace: TraceConfig = field(default_factory=TraceConfig)` to `AtlasConfig`.

**T1b. Agent exposes the counterfactual (the core edit; `arbiter.py`).**
Add attributes in `__init__` and reset in `reset_episode`:
```python
self.trace_enabled: bool = False       # set True by the collector ONLY
self.last_scores_17: Optional[np.ndarray] = None   # the ranking the mode actually used
self.last_scores_source: str = "none"
self.last_epsilon_fired: bool = False
self.last_override_info: Optional[dict] = None
self.last_arrived: bool = False
```
In `act()`, in EACH mode branch, immediately after the branch computes its ranking and
BEFORE the epsilon tail, set:
- PROBE branch: `self.last_scores_17 = probe_score.detach().cpu().numpy()` (post
  material-suppression values, i.e. exactly what argmax saw);
  `self.last_scores_source = "probe_blend"`.
- NAVIGATE branch: `self.last_scores_17 = self.option_net.logits(s, g_key_t).squeeze(0).detach().cpu().numpy()`;
  `self.last_scores_source = "option_logits"`.
- WANDER branch: `self.last_scores_17 = (self.beta_t * d_all).detach().cpu().numpy()`;
  `self.last_scores_source = "wander_logits"`.
- FALLBACK branch: `self.last_scores_17 = logits.detach().cpu().numpy()`;
  `self.last_scores_source = "fallback_logits"`.
Set `self.last_epsilon_fired` in the epsilon tail (record whether the draw fired, and note:
the replacement is uniform-random — the trace shows intended vs final, that is enough).
Set `self.last_override_info` in the safety block:
```python
self.last_override_info = {"fired": bool(override_fired),
                           "risk_intended": round(risk_intended, 4),
                           "min_risk": round(min_risk_val, 4),
                           "a_intended": int(a_intended), "a_final": int(a_final)}
```
Set `self.last_arrived = True` at every site that transitions into PROBE on arrival
(the NAVIGATE-arrival site and the goal-selection-arrival site), else False.
When `trace_enabled` is False these assignments still run (they are O(1) dict/numpy ops,
no behavior change); only the FILE WRITING is gated by `trace_enabled`.

**T1c. TraceWriter (`atlas/trace.py` NEW).**
```python
class TraceWriter:
    def __init__(self, run_dir, max_folder_mb: float = 500.0)
    def begin_episode(self, iter_idx: int, ep_idx: int, seed: int) -> None
    def write_step(self, agent, step: int, reward_prev: float) -> None   # reads ONLY agent fields
    def end_episode(self, length: int, total_reward: float, died: bool) -> None
```
`write_step` appends ONE json line to `atlas_runs/<run_id>/trace/trace_it<iter>.jsonl`
(file opened in append mode at `begin_episode`, closed at `end_episode`). Exact line schema
(field names are contractual — `atlas/replay.py` and the tests key on them):
```json
{"iter": 121, "ep_idx": 3, "seed": 6234, "t": 143,
 "mode": "NAVIGATE", "prev_mode": "NAVIGATE",
 "goal": {"kind": "event", "dims": [7], "cos": 0.71, "psi": 0.42,
          "budget": 44, "goal_mode": "material_subgoal", "n_try": 3, "n_succ": 1} | null,
 "scores_source": "option_logits", "scores": [17 floats r4],
 "a_intended": 4, "a_final": 4,
 "epsilon_fired": false, "epsilon": 0.12, "beta_t": 0.35,
 "override": {"fired": false, "risk_intended": 0.11, "min_risk": 0.03,
              "a_intended": 4, "a_final": 4},
 "risks": [17 floats r4],
 "event_p_max": [22 floats r3],        // max over actions of event head p(a,i) — the affordance ceiling
 "m_soft": [22 bools], "newly_unlocked": [int],
 "probe_id": 12, "abort_reason": "none", "arrived": false,
 "material_gate": {"suppressed_count": 2} | null,
 "r_prev": 0.0, "r_int": 1.32}
```
Sources, all from existing agent fields at the end of `act()`: mode/prev_mode, `active_goal`
(kind via `GoalEntry.kind`, dims via `event_dims`, cos = telemetry `arrival_cos`, psi =
telemetry `psi`, budget, goal_mode, and `n_try/n_succ` read from the GoalEntry),
`last_scores_17/_source`, `a_intended` (expose: the current code computes it locally — add
`self.last_a_intended` alongside `last_scores_17`), epsilon/beta, `last_override_info`,
`all_risks` (expose as `self.last_risks_17` in the safety block), event p max from
`all_event_probs.max(dim=0)` (expose `self.last_event_p_max`), `belief.m_soft`,
`probe_newly_unlocked` (the CURRENT probe's locks — the same list C2 attributes; do NOT
re-derive), `probe_id`, `last_abort_reason`, `last_arrived`, material gate
`suppressed_count` (expose `self.last_suppressed_count`; None when gate inactive),
`r_int` from telemetry. `write_step` and the trace path must NEVER receive env info — the
collector calls `write_step(agent, step, reward_prev)` after `agent.act` and before
`env.step`; no new agent inputs, signature of `act()` untouched.

**T1d. Collector wiring (`collect.py`).**
`EpisodeCollector.__init__` gains optional `trace_writer: Optional[TraceWriter] = None`.
In `run_episode`: if `trace_writer` is not None → `agent.trace_enabled = True`,
`trace_writer.begin_episode(iter_idx, ep_idx, seed)` before the loop;
`trace_writer.write_step(self.agent, step, reward_prev)` immediately after `agent.act`;
`trace_writer.end_episode(step, sum(rewards), died)` in the journal-writing block.
Save/restore `agent.trace_enabled` around screening exactly like `frozen_memory`
(`screen.py`): screen runs get traces too, tagged by iter_idx = -1 and a `screen: true`
flag added to each line when `agent.current_iter < 0` — implement as
`trace_writer.begin_episode(..., screen=True)` storing the flag on the writer.

**T1e. Selflearn wiring (`selflearn.py`).** Construct one `TraceWriter(run_dir)` when
`cfg.trace.enabled`; pass to the collector; pass `enabled` through `run_atlas.py` as
`--no-trace` (argparse flag; default: tracing ON). Add the trace folder to the
`_enforce_size_cap` scope: `trace.py` implements its own pruning (delete oldest
`trace_it*.jsonl` until under cap) — do not touch audit_pack's cap.

**Verification T1** (new `atlas/tests/test_v4_trace.py`):
1. Run a stub-net agent (existing test fixtures) 30 steps with a TraceWriter on tmp_path:
   file exists, 30 lines, every line parses, every contractual key present
   (exact list asserted), `scores` length 17, `risks` length 17, `event_p_max` length 22.
2. Epsilon=0 stub: `epsilon_fired` false on all lines and `a_intended == a_final`.
3. Stub ContinueNet to force an override on one step: that line has `override.fired == true`
   and `a_intended != a_final`; all other lines false.
4. NAVIGATE line has `scores_source == "option_logits"`; PROBE line
   `"probe_blend"`; WANDER `"wander_logits"`; FALLBACK `"fallback_logits"`.
5. `inspect.signature(AtlasAgent.act)` unchanged — no new parameters.
6. Pruning: create fake trace files totalling > cap → oldest deleted.

### T2. Ground-truth probes — "did it know?" (`atlas/probes.py` NEW; edits: `selflearn.py`, `audit_pack.py`)

Linear probes decode ground truth that ALREADY EXISTS in shards/cached latents. NumPy-only,
deterministic, closed-form — NO torch, NO training loop, NO new randomness:

**T2a. Probe suite.** Features `X` = one of {`z` [T,128], `h` [T,256], `s` [T,384]}, standardized
(mean 0, std 1, std floor 1e-6). Ridge regression closed form:
`W = (XᵀX + λI)⁻¹ Xᵀ y` with `λ = 1e-3` (identity of size D). Fit per target per feature.
Targets (all from shard arrays / cached episodes; names appear ONLY as JSON keys here):
| id | target y | source | type | metric |
|---|---|---|---|---|
| pos_x, pos_y | positions[:,0], positions[:,1] | shard `positions` | regression | R² |
| inv_wood | inventory[:,1] (log1p wood count) | shard `inventory` | regression | R² |
| inv_stone | inventory[:,2] | shard `inventory` | regression | R² |
| vital_health | inventory[:,12] | shard `inventory` | regression | R² |
| vital_food / vital_drink / vital_energy | inventory[:,13..15] | shard `inventory` | regression | R² |
| evt_next5_any | any event fires in t..t+5 | shard `events` | binary | AUROC* |
| evt_<e>_next5 for the 5 most frequent event dims e | events[:,e] max over t..t+5 | shard `events` | binary | AUROC* |
| reward_next5 | sum of rewards t..t+5 | shard `rewards` | regression | R² |

*Binary metric: AUROC computed by rank-count formula on the raw ridge prediction
(implementation: `auc = (Σ_{i∈pos} rank_i − n_pos(n_pos+1)/2) / (n_pos·n_neg)`; ties averaged
by mean-rank). If a target has <20 positive samples in the fit set, report `null` and skip —
never fabricate.

**T2b. Fit protocol.** Once per iteration in `selflearn.py` AFTER latent caching, on the SAME
cached latents used for training: use the most recent ≤8 shards (or all if fewer); split by
EPISODE 80/20 (deterministic: episodes with `index % 5 == 0` → val); report VAL metric only.
Fit 9–14 targets × 3 feature sets. Cost: ≤14 ridge solves of ≤384×384 — milliseconds.

**T2c. Output.** Write `atlas_runs/<run_id>/audit/probes/probes_it<iter>.json`:
```json
{"iter": 121, "git_commit": "abc1234",
 "probes": {"z":  {"pos_x": {"r2": 0.41, "n_val": 1600}, ...},
            "h":  {...}, "s": {...}},
 "notes": "diagnostics only; no code consumes these numbers"}
```
Append one summary line to `train_log.jsonl`: `{"probe_best": {"pos_x_s": 0.41,
"inv_wood_s": 0.77, ...}}` (the s-feature metric per target). `audit_pack.export` copies the
latest probes json into `audit/probes/` (it already has a health_cards pattern to copy).

**Interpretation rubric (for the operator, printed in the json `notes`):** R²/AUROC high on
`s` but low on `z` → info exists but only via recurrent memory (policy problem is learnable);
low on all three → the latent genuinely lacks the info (perception ceiling — the frozen
backbone must be revisited, NOT the policy). Report raw; no gate, no tuning.

**Verification T2** (new `atlas/tests/test_v4_probes.py`):
1. Synthetic: y = X @ w_true + 0.01·noise with n=500, D=32 → R² > 0.95 (harness sanity).
2. Synthetic: y independent of X → R² ≈ 0 (|R²| < 0.05) — guards against optimism.
3. AUROC: perfectly separable synthetic → 1.0; shuffled labels → ≈ 0.5 (±0.05).
4. Fixture shard + fake latents → full suite runs, JSON has all three feature keys,
   val split used every episode exactly once (no leakage: assert fit n_val == 20% of eps).
5. Grep test: `atlas/probes.py` imports nothing from `atlas/arbiter.py` and is imported by
   nothing outside `selflearn.py`/tests — enforcement via AST scan added to
   `test_atlas_isolation.py` (allowed-importer list).

### T3. Canary pre-flight (`atlas/canaries.py` NEW; `python -m atlas.canaries [--quick]`)

Five checks, each with a numeric pass criterion, executed in order, first failure prints
`!!! [CANARY FAIL] <name>: <detail>` and exits nonzero. Total runtime target < 2 min on MPS
(`< 30 s` with `--quick`, which runs CAN1/2/5 only).

- **CAN1 integrity gates**: run the existing isolation AST test + no-cheat greps in-process
  (`pytest atlas/tests/test_atlas_isolation.py -q` via subprocess is acceptable). Pass = exit 0.
- **CAN2 trace round-trip**: build stub nets (reuse fixtures pattern from
  `test_atlas_arbiter.py`), run 50 steps through `AtlasAgent.act` + `TraceWriter` on a temp
  dir, replay with `atlas/replay.py` story renderer. Pass = 50 stories rendered, zero parse
  errors, override/unlock filters each return ≥0 rows without exception.
- **CAN3 probe harness**: the T2 synthetic checks (1)–(3) executed in-process. Pass = all three.
- **CAN4 real-env 1-episode smoke** (skipped with `--quick`): local checkpoint, seed 6150,
  60 steps, tracing on, `backbone.assert_frozen()` before AND after. Pass = episode
  completes, trace has 60 lines, frozen holds, no env-info enters act (signature check).
- **CAN5 no-cheat sweep**: grep `atlas/` for banned structures (`CRAFT_REQUIRED_MATERIALS`,
  achievement-name literals in arbiter/goals/skills/trace) — reuse the existing structural
  test's patterns; import `atlas.replay` and `atlas.probes` and assert they raise no
  import of `envs/`, `eval/`, `models/`, `planning/`, `training/`, `dual_store/`.

### T4. Replay / inspect CLI (`atlas/replay.py` NEW; edits: `tests/*` only otherwise)

Pure log renderer. Reads ONLY `trace/*.jsonl` + `games.jsonl` (the journal) + optional
`audit/probes/*`. No env, no nets, no torch. Subcommands:
```
python -m atlas.replay story   --run-dir RD --iter 121 --ep 3 [--from 100 --to 140]
python -m atlas.replay filter  --run-dir RD [--override] [--abort] [--unlocked] [--mode M] [--iter N]
python -m atlas.replay summary --run-dir RD [--iter N]
python -m atlas.replay goal    --run-dir RD --iter 121 --ep 3      # full goal-lifecycle view
```
- `story`: one human-readable block per step, e.g.
  `t=143 NAVIGATE goal=event[7] cos=0.71 ψ=0.42 | opt: move_down 0.31, do 0.29, move_left 0.27
   | risk 0.11 | ε off | → move_down` and for overrides/unlocks/aborts a `!!!`-prefixed line.
  Action indices rendered via `CRAFTER_ACTION_NAMES` (display only).
- `filter`: prints matching trace lines compactly (t, mode, reason-class) — reason-class =
  `override` | `abort:<reason>` | `unlock:<dims>` | `arrive`.
- `summary`: per-episode rollup — mode step counts, top-3 actions, override count, abort
  reasons histogram, unlock list, plus (if probes json exists) the probe summary line.
- `goal`: for one episode, print every goal cycle: selected goal (kind/dims/score),
  selections by mode (argmax/rotate/frontier/material_subgoal/execute_craft), arrivals,
  probe outcomes, release/abort events — reconstructing the F11/F3 chain the operator cares about.

**Verification T4** (new `atlas/tests/test_v4_replay.py`): golden-trace fixture (20
hand-built lines incl. one override, one abort, one unlock) → each subcommand runs, story
output contains the override marker, filter counts exact, summary rollup matches fixture
counts. Exit codes: 0 on success, 2 on missing files with a clear message.

### T5. Relation snapshots — watch edges form (`audit_pack.py` edit, selflearn hook)

On every `export()` call, additionally write `audit/relations/relations_it<iter>.json`:
```json
{"iter": 121,
 "lift_edges": [[d, e, ci_low, lift], ...],          // only ci_low > 0 edges, as int/float pairs
 "item_event_map": {"0": 7, "1": 0, ...},            // item dim -> event dim or null (V3 F3 map)
 "bank": [{"kind": "event", "event_dims": [7], "n_try": 3, "n_succ": 1,
           "world_frequency": 0.28, "born_iter": 40}, ...]}   // one object per goal, no keys/names
```
Sources: `PredecessorLift` arrays, `agent.item_event_map`, `GoalBank.entries`. The bank dump
MUST NOT include key vectors or any name strings — dims are anonymous ints. Purpose: over a
run's iterations this file shows the wood→table→pickaxe chain ASSEMBLING, and whether F10's
world_frequency is rising — previously only end-state dumps existed.

**Verification T5** (extend `test_v4_trace.py`): fixture agent with a 1-edge lift + 2-goal
bank → json matches schema; assert no string field in `bank` other than `kind`; assert
`item_event_map` values are ints or null.

---

## Part C — IMPLEMENTATION ORDER & DEFINITION OF DONE

Order: **T1 → T2 → T4 → T5 → T3 → boundary 10-iter segment → G1 → G2 → G3 → G4 → v4 60-iter
segment** (T3 last of the instrumentation because it runs the other four as canaries; the
boundary segment collects trace evidence BEFORE any behavior change so G-fixes are validated
against measured baselines, and the G-items land in a fresh segment per the V3 rule that
training-dynamics changes start clean segments).
Test-first per house style: write each verification test, watch it fail, implement, watch it pass.

1. `pytest atlas/tests -q` — all existing + new tests green.
2. `python -m atlas.canaries --quick` — clean, < 30 s.
3. `python -m atlas.canaries` — clean, < 2 min (CAN4 requires the local phase2 checkpoint;
   if absent, it reports `SKIP (no checkpoint)` and does not fail the suite).
4. `python -m atlas.smoke` — clean; smoke now ALSO writes a trace (assert ≥1 line in the
   smoke's temp trace dir) and a relations snapshot.
5. `git diff` touches ONLY the allowed files listed at the top.
6. No-cheat sweep (CAN5) green; `act()` signature byte-identical; backbone untouched.
7. Handback summary: tests added (fail→pass), canary timings, trace size per 100 steps
   measured, and the exact commands for the next Colab segment.

## Part D — RUN PROTOCOL (same run-id `atlas_v2_scratch`; 2,042 episodes are the data)

```bash
# Segment 1 — instrumentation only (T1–T5 active, zero behavior change), 10 iterations:
python -m atlas.colab.run_atlas --phase selflearn --iters 10 --run-id atlas_v2_scratch --minutes 90
# check V-M0: trace complete, G1 calibration logged, baseline unlock-rate recorded
# Segment 2 — G-repairs active (G1–G4), 60 iterations:
python -m atlas.colab.run_atlas --phase selflearn --iters 60 --run-id atlas_v2_scratch --minutes 300
# tracing ON by default; --no-trace to disable; probes + relations written each save_state.
```
Storage budget (measured, not guessed — verify at handback): trace ≈ 250–350 B/step →
≈ 0.3 MB/episode → ≈ 2.5 MB/iteration at 8 episodes → a 100-iteration segment ≈ 250 MB on
Drive, auto-pruned at 500 MB. Audit folder cap unchanged (60 MB). If Drive pressure is real,
`--no-trace` for routine segments and enable for diagnostic segments.

## Part E — WHAT THE OPERATOR CAN NOW ASK (the point of all this)

| Question (previously: days of inference) | Command / file |
|---|---|
| Why did it choose action X at step t? | `replay story --iter N --ep E --from t` |
| Why did it NOT take the obvious action? | same — other 16 scores are in the line |
| Did epsilon / safety / material-gate interfere? | `replay filter --override` / trace field |
| Why do navigations abort? | `replay filter --abort` → reasons histogram in `summary` |
| Is a failure perception or policy? | `audit/probes/probes_itN.json` (s vs z vs h rubric) |
| Is the wood→table edge forming? | `audit/relations/relations_it*.json` diffed over iters |
| Did the goal chain route correctly? | `replay goal --iter N --ep E` |

## Part F — ANTI-ASSUMPTION CHECKLIST (before implementing each T- or G-item)

1. Quote the exact formula/schema line above that you are implementing.
2. Which inputs are learned/mined vs constants? Trace and relations record EXISTING agent
   fields — nothing new is invented, no name/index logic enters any decision path.
   For G-items additionally: G1's bounds (0.60, 0.85) and G4's grid (0.30..0.60) are search
   ranges, not world knowledge; the SELECTED values must always come from the agent's own
   data. If any constant you are about to type encodes game knowledge (item identities,
   action semantics, craft quantities, spatial neighborhoods), STOP, implement that part as
   DISABLED, and report the gap.
3. Achievement/action/item names appear ONLY in `probes.py`/`replay.py` display strings and
   JSON keys — the AST/grep test (CAN1/CAN5, extended to the new files) enforces this.
4. Is there a verification test? Every T- and G-item has one above; write it FIRST.
If you find an ambiguity not resolved above, it is a bug in this plan — report it, do not
resolve it yourself.

---

## Part G — V3.1 BOTTLENECK REPAIRS (learned-only; from the it119→212 campaign)

### Part G0 — ADJUDICATION of the report (read before implementing; do not re-litigate)

The campaign report is measured and mostly trusted, with FOUR corrections that shape the fixes:

- **ADJ1 (report error, B1 diagnosis):** the report claims arrival uses cosine in
  "384-dimensional latent space (z⊕h)". FALSE — `arbiter.py` computes `_cosine(z_np, g_key_np)`
  on the 128-d `z` only. The *rates* (cos ≥0.85 on 0.1% of steps; mean 0.311; 18.4% ≥0.60) are
  accepted as measurements; the stated mechanism is not.
- **ADJ2 (internal contradiction, B1 severity):** "Legitimate PROBE mode is almost never
  reached via arrival" contradicts the report's own count of 11,669 probe steps / 752 episodes
  ≈ 2.5 probes per episode, consistent with PLAN_V3's basis (arrivals ~10/iter, loop open).
  Probes run; the measured problem is navigation EFFICIENCY (most steps far from any keyframe;
  9.7% WANDER), not zero arrivals. G1 therefore calibrates rather than crashes the threshold.
- **ADJ3 (cheat risk in the report's fix #2):** "cost = 2 table + 1 pickaxe = 3 wood" typed as
  a constant is a hand-written recipe — F11-incident class, banned. G2 derives the SAME budget
  by summing `StateDeltaHead` costs along the learned lift chain. No integer 3 anywhere.
- **ADJ4 (cheat risk in the report's fix #3):** "anchor to the 3×3 coordinate neighborhood"
  uses engine coordinates in the decision loop — banned by AGENTS.md §4. G3 anchors to the
  VISUAL tagged view (`episode_artifact_views`, already built in V2.2-F11). No coordinates.
- The report's fix #4 (risk threshold 0.50 → 0.40) is a safety-config constant, allowed (same
  class as the existing 0.50/0.15), but G4 pairs it with measurement so the new value is
  justified, not just copied.

### G1. Calibrated arrival threshold — no more hand-picked 0.85 or 0.65 (`config.py`, `arbiter.py`, `selflearn.py`)

**G1a. Offline calibration (once per dev-screen iteration, in `selflearn.py` after latent
caching; numpy only):** over the most recent ≤8 shards' cached episodes, collect RE-SIGHT
cosines = for every step t, `max over keyframes k in the SAME episode with k.step < t` of
`cos(z_t, z_k)` restricted to pairs where the TopoMap would consider them distinct scenes
(cos < 0.95 — exclude near-identical adjacent frames). This is the distribution of "visually
back at a place I've been". Compute `p75` of this distribution. Then:
```python
arrival_cosine_calibrated = float(np.clip(p75, 0.60, 0.85))   # bounded; never below 0.60
```
Store in state.pt as `arrival_calibration: {"p75": ..., "threshold": ..., "iter": ...}` and
recompute only every 10 iterations (stability; don't chase noise). `arbiter.py` uses
`self.arrival_threshold = calibration value if present else cfg default`. Config adds
`SkillsConfig.arrival_calibrate: bool = True` and `arrival_bounds: Tuple[float,float] =
(0.60, 0.85)`; when `arrival_calibrate=False` the existing fixed `arrival_cosine` is used.
**Why bounds:** below 0.60 the trace (T1) already shows probes firing at unrelated contexts —
precision collapses; the bound is a measured safety margin from the 14-world keyframe audit
(match thresholds 0.90 dedupe / 0.85 world-match), not world knowledge.
**G1b. Arrival-precision telemetry (no behavior change):** T1 trace already logs `arrived`,
`goal`, and `newly_unlocked`. Add to the iteration log row: `arrivals`, and
`arrival_probe_unlock_rate` = probe_unlocks / arrivals (rolling 5-iter mean). GATE for G1:
after G1 activates, `arrival_probe_unlock_rate` must NOT fall by >50% vs the boundary
segment — if it does, the threshold dropped too far; report and restore the previous value
(one-step rollback, logged). This converts the report's "0.1% → ~15%" guess into a measured
precision-guarded knob.

**Verification G1** (new `test_v4_g1_arrival.py`): (1) synthetic episode with a genuine revisit
(cos ≈ 0.9) and a near-miss (cos ≈ 0.7): with calibration computed from a fixture whose p75 =
0.72 → threshold 0.72 → revisit arrives, near-miss does not; (2) clamp: p75 = 0.30 →
threshold 0.60; p75 = 0.99 → 0.85; (3) rollback rule: unlock-rate drop >50% in the stub →
threshold restored and logged.

### G2. Chain-aware material budget — the learned 3-wood (`arbiter.py`, `goals.py`, `selflearn.py`)

Replaces the report's typed "2+1=3" with a sum of LEARNED costs along the LEARNED chain.
Exact algorithm, inserted in the V3-F3 material-subgoal block of `arbiter.py` where the
deficit is computed:

```python
# When target goal's dim e triggers the F3 material path (learned cost vector nonzero AND
# lift predecessor edge ci_low > 1.2 exists), compute the CHAIN budget, not the single-action cost:
chain_dims = [e] + [p for p in range(NUM_ACHIEVEMENTS)
                    if self.lift.ci_low[p, e] > lift_ci_threshold]   # predecessors of e
# budget[i] = sum over d in chain_dims of max(0, -StateDeltaHead(s, a_d)[i])
#   where a_d = argmax_a event_probs[a, d]  (the affordance action for dim d — LEARNED)
# have[i] = Readout(s)[i]  (log1p scale)
# deficit dims: budget[i] > have[i] + material_gate_margin
# ONE deficit dim per cycle (existing V3 rule); cycles = ceil(max deficit), cap max_cycles;
# material subgoal stays EMITTED until have[i] >= budget[i] on every deficit dim
#   (V3 bug being fixed: cycles decremented per probe-completion regardless of holdings).
```
The `it211` failure (table placed → wood 0 → pickaxe goal never re-budgeted) is exactly the
single-action-cost + unlock-dropout interaction: G2 re-plans from CURRENT have vs CHAIN budget
after every completed cycle, so after the table consumes wood, the pickaxe's chain budget
(learned: table-cost + pickaxe-cost) still shows the deficit and re-emits the wood subgoal —
even though `collect_wood` is unlocked (see G2b).
**G2b. Quantity-mode goal scoring (`goals.py`):** add to `GoalEntry` a counter
`material_serves: int = 0` (persisted in the bank npz/json), incremented by the arbiter when a
goal is chosen AS a material subgoal. In `select()`, when a material cycle is ACTIVE (the
arbiter passes `active_material_deficit_dim: Optional[int]`), the scoring term
`w_unlock * max(locked_profiles)` for candidate goals whose `event_dims` contain
`ItemEventMap[deficit_dim]` is computed with `m_soft` for that dim FORCED to 0 (treated as
locked) — i.e., while gathering materials, the already-unlocked state does not zero the score.
This is the learned fix for Bottleneck 3 (single-unlock drop-out): it applies ONLY during an
active material cycle, so general exploration priorities are untouched.
**Verification G2** (new `test_v4_g2_budget.py`): stub delta head with learned costs
[table-action: 2 wood, pickaxe-action: 1 wood] (stub VALUES, not engine values — the test
supplies whatever the stub predicts) → chain budget 3 on the wood item dim; have=1 → deficit 2
→ cycles 2; simulate table unlock consuming wood (have → 0) → re-plan shows deficit 3 → wood
subgoal re-emitted although the wood EVENT dim is unlocked; with `ItemEventMap` link absent →
no substitution (unchanged honesty rule); budget decays correctly when have rises.

### G3. Visual station anchor — stay at what you built (`arbiter.py`, `goals.py`)

All-visual; zero coordinates. In `arbiter.py`, after any place-tier unlock is recorded (the
existing `episode_artifact_views[dim] = z_np` site in the belief-update block):
```python
# If a chain target is pending (target_goal_pending is not None) whose lift predecessor of
# its dim is the JUST-UNLOCKED dim, and the chain's material deficit (G2) is NOT yet
# satisfied: set active_goal = a synthetic GoalEntry(kind="artifact", key=z_np.copy(),
# event_dims=[pending_dim], n_try=0, n_succ=0) — the view AT the placed artifact.
# While an "artifact"-kind goal is active: NAVIGATE arrival check is cos(z_t, key) >=
# arrival_threshold (G1); on arrival, PROBE runs with the material gate ACTIVE (suppress
# unaffordable crafts) and G2 re-plans each cycle. The artifact goal is RELEASED when the
# pending target unlocks OR the deficit is satisfied OR the goal aborts by the normal
# stagnation rule (agent may legitimately leave if materials require it — the wood subgoal
# emission then overrides, per G2).
```
This reproduces the report's "station anchor" intent without its 3×3 coordinate mechanism:
"stay near what you built while the chain needs it, leave when materials call". The tagged
view ALREADY exists (`episode_artifact_views`); G3 only makes it selectable as a goal.
**Verification G3** (new `test_v4_g3_anchor.py`): stub chain (pending pickaxe goal, table just
placed) → active goal becomes artifact-kind with key = tagged z; walking 5 steps with
cos ≥ threshold → PROBE entered (not WANDER); deficit satisfied → artifact goal released and
pending target restored; no pending target at unlock → no anchor (behavior unchanged).

### G4. Calibrated survival repair (`config.py`, `safety.py` training-side, `arbiter.py`)

**G4a. Threshold recalibration from data (once, before the v4 segment; report raw):** from the
last ≤20 shards, compute per-step ContinueNet `risk_all_actions` on cached latents at the
step BEFORE every real death (`dones[t] == True` → risk at t-1) and at a random sample of
survived steps. Choose the new threshold as the value maximizing
`deaths_flagged / total_deaths − 0.5 · false_flag_rate` (a fixed, stated utility; grid over
0.30..0.60 step 0.05). Write the chosen value + the full grid into the plan handback and set
`SafetyConfig.risk_threshold` to it ONLY if deaths_flagged ≥ 60% at that value; else keep
0.50 and report. If the grid picks 0.40 — that CONFIRMS the report's guess; if not, the
measurement wins over the report.
**G4b. Sleep-risk features (training-side only, no new inputs):** no mechanism change —
ContinueNet already trains focal BCE on deaths; the sleep-commitment deaths are simply labels
it must learn. Add ONE diagnostic to the iteration log: `death_context` histogram over the
last segment (death step, mode at death, was-intended-action from the trace T1) so the
operator SEES whether sleeping deaths are being predicted (risk at t-1 from G4a's data)
rather than assuming it. No night/evasion logic in v4 — that stays Phase C per V3-F9.
**Verification G4** (new `test_v4_g4_safety.py`): synthetic risk scores for 10 deaths
(mostly 0.45–0.55) + 1000 survived (mostly 0.05–0.35) → grid selects 0.40, deaths_flagged
≥ 60%; all-survived fixture → threshold unchanged at 0.50, `null` grid row for deaths;
override behavior itself unchanged (existing arbiter tests must stay green).

### G5. DEFERRED (explicitly NOT in v4; listed so nobody "helpfully" adds them)

- Night-evasion policy / sleep scheduling (Phase C; V3-F9 rule: one variable per segment).
- Stone/coal frontier push (depends on the tool chain firing deliberately first).
- Reaching for the 4.31% champion score: v4's success is measured by the G-gates below, not
  by score. Score talk resumes only after M4 (first deliberate pickaxe).

### G6. GATES for Part G (pre-registered; report raw, never tune to them)

| Gate | Metric (source) | Pass |
|---|---|---|
| V-M0 (boundary+10) | trace complete; G1 calibration value logged; baseline arrival_probe_unlock_rate recorded | all present |
| V-M1 (it+20) | episodes reaching chain-budget wood holdings (shard targets, have ≥ G2 budget on wood dim) | ≥ 10% (was 0.2–0.3% at 3 wood) |
| V-M2 (it+30) | `place_table` episodes AND post-table behavior: artifact-anchor entered when chain pending (trace) | table ≥ 10%; anchor entered in ≥ 50% of table episodes |
| V-M3 (it+40) | `make_wood_pickaxe` episodes | ≥ 1 (the report's M4) |
| V-M4 (it+60) | nightfall death share + score trend vs it212 | nightfall share ≤ 65% (was 75.3%); score not worse than 1.76% |
| Ablation | 10 iters with G1 calibration OFF (fixed 0.85) | report both arms' arrivals + unlock-rate |

**Stopping rule (pre-registered):** if V-M1 fails at it+20 — the chain-budget policy still
never holds enough wood — report plainly; the next rung is critic horizon H > 25 / n-step
bootstrap (V3's own escalation), NOT further goal-selection tuning.
