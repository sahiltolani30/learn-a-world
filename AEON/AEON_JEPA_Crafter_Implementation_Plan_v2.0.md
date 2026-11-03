# AEON / JEPA Crafter — Ordered Implementation & Validation Plan

**Version:** 2.0  
**Date:** 3 November 2026  
**Status:** Proposed execution plan; replaces the conflicting stage sequences in v1.3.  
**Repository target:** `AEON/jepa-crafter/`  
**Primary environment:** Crafter (`CrafterReward-v1`; reward-free runs are a separate experiment)  
**Initial hardware target:** one Colab NVIDIA T4 where available  

> **Truth statement:** The repository already contains meaningful JEPA, replay, exploration and CEM-planning scaffolding. Source files being present do not prove that these components work together, that the model learns useful dynamics, or that Crafter relationships are discovered and reused. This plan is a sequence of tests and gates, not a promise that the system will achieve a particular score.

---

## 1. What this plan is trying to prove

The project is not complete when the JEPA loss falls, when a class is implemented, or when the agent gets one achievement. The system must establish each of these claims independently:

1. The environment interface is correct and the learner receives no privileged game state.
2. The from-scratch JEPA predicts future visual latents on genuinely unseen episodes.
3. The future prediction uses the action token and relevant context, rather than merely predicting that the image stays similar.
4. Model representations remain non-collapsed and multi-step imagined rollouts have measurable value.
5. Exploration collects useful experience and persistent model/replay state survives restarts.
6. Anonymous visual-state estimates are stable enough to support evidence about state changes.
7. Relationship hypotheses are generated from observed context/action/effect data, and remain hypotheses until tested.
8. Controlled tests distinguish real conditional effects from correlation, noise, camera movement and tracking errors.
9. Confirmed relationships are saved, retrieved in later episodes/worlds and rejected when their preconditions do not match.
10. Multiple supported relationships can be composed into a closed-loop plan.
11. Frozen transfer and matched ablations show whether relational memory actually helps.
12. Only after these gates pass should the project scale to more workers, more roles or the 1M-step benchmark.

There is no guarantee that JEPA alone will infer named concepts such as “wooden pickaxe” or “stone.” The learner is restricted to RGB observations and anonymous action tokens. Human-readable labels and hidden Crafter state may be used by a separate evaluator to score discoveries, but never as learner input, training labels, action-selection features or relation-memory content.

## 2. Learning boundary — do not change this to make a test pass

### 2.1 Permitted learner inputs

At each environment step, the learner may receive only:

```text
observation_rgb: uint8[64, 64, 3]
action_token:    integer in [0, 16]  # opaque token selected by the agent
next_rgb:        uint8[64, 64, 3]
reward:          one scalar float
terminated:      bool
truncated:       bool
```

Episode ID, timestep, seed and sequence-boundary metadata can be used by storage/sampling code only. They must not enter model tensors.

### 2.2 Forbidden learner inputs

No `info` dictionary; achievement names/flags; inventory dictionaries; map, coordinates, terrain/object/entity IDs; hidden health or crafting state; action names/descriptions; hard-coded recipes; manually authored dependency graphs; scripted action sequences; OCR; or pretrained model weights.

Visible HUD/inventory artwork is part of RGB. The agent may learn visual representations of it. A separate evaluator may inspect `info` or hidden state only to calculate post-hoc metrics and organize diagnostic pairs. The evaluator must have a one-way path to metric artifacts, never to the learner.

### 2.3 Opaque action mapping

- The agent chooses one of 17 integer tokens.
- The environment adapter maps the token to a legal environment action through a run-specific permutation.
- That mapping is stable within one run and is saved in the run manifest/checkpoint.
- Training, evaluation branches, replay and resumed execution must use the exact same mapping whenever they compare the same learned token.
- A new independent run may use a new mapping, but it must not reuse a relation memory trained under a different mapping unless the mapping itself is also restored.
- The learner never receives the permutation or action descriptions.

### 2.4 Evidence labels versus learner information

Evaluator-only labels can answer “did this candidate correspond to the hidden game event?” after the run. They cannot be used to train the JEPA, label relation candidates, set confidence online, guide exploration or select an action. Keep evaluator annotations in a physically and logically separate artifact namespace.

---

## 3. Corrections to v1.3 — the single authoritative order

The v1.3 document grew through several revisions. Its earlier phase list, relational-learning section and final observability section did not all use the same stage order. This v2.0 order resolves those conflicts:

- **Observability and current-baseline correctness come first.** Do not spend 50k or 1M steps on an untrustworthy pipeline.
- **Run a short, instrumented baseline before adding new subsystems.** Otherwise a later improvement or regression cannot be attributed.
- **Validate JEPA and its action conditioning before enabling model-based control.**
- **Test relational hypotheses before promoting them to confirmed persistent knowledge.** The append-only evidence ledger is created with candidate discovery; the durable confirmed-relation memory comes only after controlled validation.
- **Keep low-level latent MPC separate from relation-based compositional planning.** MPC tests whether the learned dynamics model is useful for choosing short-horizon actions. The relation planner tests whether learned precondition/effect links can be composed. A failure in one must not be misreported as a failure in the other.
- **Transfer is tested with frozen model and memory before scaling.** A system that relearns a mechanic in every new world has not demonstrated knowledge transfer.

### Canonical stage map

| Order | Phase | Main output | Gate that unlocks the next dependent phase |
|---:|---|---|---|
| 0 | Repository and runtime audit | Verified source/test inventory and run manifest | `P0_AUDIT_PASS` |
| 1 | Observability foundation | Reconstructable traces and debug artifacts | `P1_TRACE_PASS` |
| 2 | Repair baseline integration | Correct mapping, held-out split, novelty/progress updates, scoring and resume | `P2_BASELINE_INTEGRITY_PASS` |
| 3 | Environment/replay smoke and pilot | Trusted 500-step and 5k–10k pilot | `P3_PIPELINE_PILOT_PASS` |
| 4 | JEPA baseline implementation verification | Correct from-scratch JEPA training path | `P4_JEPA_IMPLEMENTATION_PASS` |
| 5 | JEPA learning/action-conditioning benchmark | Genuine held-out prediction and action-use evidence | `P5_JEPA_LEARNING_PASS` |
| 6 | Single-role exploration | Measurably useful data collection without multiple agents | `P6_EXPLORATION_PASS` |
| 7 | Optional latent MPC | Real-world utility of short-horizon model-based actions | `P7_MPC_UTILITY_PASS` or explicitly `DISABLED` |
| 8 | Anonymous visual state tracker | Stable before/after visual state evidence | `P8_VISUAL_TRACKING_PASS` |
| 9 | Relationship candidates and evidence ledger | Auditable hypotheses with supporting/contradicting evidence | `P9_RELATION_CANDIDATE_PASS` |
| 10 | Controlled conditional testing | Independently tested/confirmed or rejected relations | `P10_RELATION_VALIDATION_PASS` |
| 11 | Persistent relation memory and retrieval | Versioned saved knowledge that can be reused | `P11_RELATION_MEMORY_PASS` |
| 12 | Compositional planning | Closed-loop plans grounded in confirmed relation edges | `P12_COMPOSITION_PASS` |
| 13 | Frozen cross-world transfer and ablations | Evidence that memory helps on unseen worlds | `P13_TRANSFER_PASS` |
| 14 | Scale-up and benchmark | 100k–1M-step studies, optional roles/workers | `P14_SCALEUP_READY` |

A phase may be `PASS`, `FAIL`, `NOT_RUN`, `NOT_VERIFIED`, `BLOCKED` or `DEGRADED`. Only `PASS` unlocks the next dependent phase. An optional module that fails (such as MPC) can remain disabled while independent relation-learning work continues, provided the failure does not violate an earlier prerequisite.

---

## 4. Global rules for every phase

### 4.1 Before coding a phase

1. Read the current implementation, tests and configuration for the exact module.
2. Record whether it is `implemented`, `partial`, `missing` or `not_verified`.
3. Write the component contract and acceptance tests before changing code.
4. Create a new branch/commit or a clean, identifiable change set.
5. Do not rewrite an existing module just because the plan suggests a filename. Preserve working AEON code unless a test demonstrates a defect.

### 4.2 After coding a phase

1. Run targeted unit tests.
2. Run integration tests with adjacent components.
3. Run fault-injection tests for the main failure mode.
4. Run the phase's benchmark using its fixed config and data split.
5. Save test logs, metrics, traces, plots, checkpoint IDs and seed lists.
6. Produce a phase report containing changed files, tests actually run, results, failures, known limitations and the next unlocked phase.
7. Freeze the accepted interface. Later phases may request a change, but it requires a versioned interface decision and regression tests.

### 4.3 Experiment integrity

- One main architectural change per experiment whenever possible.
- Do not tune thresholds after looking at the result and then call the same experiment a pass.
- If a threshold must change, version the threshold/config and run a new experiment.
- A test file existing is not a test pass. A benchmark not executed is `NOT_RUN`.
- Keep failed episodes, ordinary episodes and successful episodes. Do not inspect only highlight videos.
- Preserve previous checkpoints, results and plans; never overwrite the evidence used for an earlier gate.

### 4.4 Proposed thresholds are not scientific constants

All numeric thresholds in this v2.0 document are **initial engineering acceptance targets**. They are not published JEPA constants, not a guarantee of feasibility on free Colab, and not proof that Crafter will be solved. Put them in versioned YAML and pre-register them before each run. A `FAIL` is useful information; it is not permission to silently weaken the gate.

---

## 5. Phase 0 — Repository and runtime audit

**Purpose:** Establish what is actually in the current repository before writing or changing code.

### Inspect

- Git commit SHA, dirty files and branch.
- All modules under `src/`, configs, notebooks, tests, prior experiment results and checkpoints.
- Crafter/Gym/Gymnasium/PyTorch package versions and actual reset/step signatures.
- Current runner arguments and config fields; identify unused or misleading settings.
- Existing source audit in v1.3. Recheck every alleged gap against the current SHA; do not assume the earlier audit is still current.

### Required work

Create:

```text
artifacts/implementation/MASTER_STATUS.md
artifacts/implementation/FILE_AUDIT.csv
artifacts/implementation/DECISIONS.md
artifacts/implementation/RISKS_AND_BLOCKERS.md
artifacts/implementation/run_manifests/preflight.json
```

For every required capability, record path, class/function, state (`implemented/partial/missing/not_verified`), existing test, test status and next action.

### Tests / benchmarks

- Import every current project module.
- Parse every active config and report unknown/unused keys.
- Run the existing full test suite and save exact stdout/stderr.
- Instantiate Crafter and verify RGB shape, dtype and action count.
- Confirm current runner can start and terminate cleanly.
- Confirm that source inspection and runtime test results are recorded separately.

### Pass gate: `P0_AUDIT_PASS`

- Exact SHA and working-tree state captured.
- Every relevant file/config/module accounted for.
- Existing tests run, with pass/fail/not-run status.
- Environment API and versions recorded.
- No claim of learning performance is made at this phase.

**If it fails:** fix environment/import/config/repository blockers only. Do not build new JEPA or relation modules yet.

---

## 6. Phase 1 — Minimal observability foundation

**Purpose:** Make subsequent failures diagnosable. Instrumentation must not change the agent's behaviour.

### Implement

A compact event system, not an unnecessarily large dashboard. Minimum event families:

- `run_started`, `episode_started`, `episode_ended`
- `observation_received`, `action_decision`, `action_executed`, `action_outcome`
- `replay_inserted`, `sequence_sampled`
- `training_update_started`, `training_update_finished`, `ema_updated`
- `checkpoint_saved`, `checkpoint_loaded`
- `failure`, `gate_result`

Every event should include `run_id`, `episode_id`, `timestep`, `event_id`, `event_type`, `code_sha`, `config_hash`, timestamp/step counter and the relevant component version. Keep event payloads bounded; store image/latent arrays in separate referenced artifacts rather than duplicating huge payloads inside JSONL.

For an action, record the observation reference, context reference, all candidate token scores when applicable, random draw/tie-break, selected opaque token, mapping version, model/checkpoint version and the subsequent actual outcome. For learning, record batch/sample IDs, horizon losses, optimizer counter, gradient health and EMA counter. For memory, record every create/update/reject/retrieve/use event with evidence IDs.

### Tests

1. Synthetic episode with at least 10 transitions can be reconstructed from logs.
2. Every action decision has exactly one linked execution and one outcome, including terminal transitions.
3. A crash creates a failure record and leaves the last complete trace readable.
4. Trace writer flush/reopen/checksum round-trip passes.
5. Debug enabled versus disabled produces identical actions and model outputs in a deterministic test harness.
6. Evaluator-only fields inserted into fake `info` never appear in learner, replay, planner or learner-facing logs.

### Pass gate: `P1_TRACE_PASS`

- At least 99.9% event completeness on the synthetic test, and 100% decision-to-outcome linkage.
- No learner/evaluator information leak.
- Deterministic debug-on/debug-off comparison matches exactly at the action level.
- Trace artifacts survive process restart and checksum verification.

**If it fails:** stop before long collection. Debugging a partially logged system is not acceptable evidence.

---

## 7. Phase 2 — Repair and verify current AEON baseline integration

**Purpose:** Correct issues that can make current metrics invalid. Recheck each item against the current SHA; do not blindly apply a fix if the current code has already changed.

### Priority checks and likely integration risks from v1.3

| Area | What must be verified | Required correction if it fails |
|---|---|---|
| Action mapping | The action-conditioning evaluator uses the exact run-specific opaque token permutation used during training | Pass the mapping into every paired test branch; add a regression test |
| Held-out split | Validation episodes are collected independently and are not in training replay | Separate collector/store and split manifest by episode/world seed |
| Novelty updates | Each real observation is inserted exactly once; imagined candidates never enter memory | One explicit post-observation update path; checkpoint or deterministic rebuild |
| Learning progress | Estimator receives actual measured prediction errors | Update from completed real transitions/prediction residuals, not just `score()` calls |
| Candidate normalization | Candidate scores do not depend on candidate evaluation order | Freeze normalizer stats for one candidate set; update only from real observed data |
| Checkpoint resume | Runner restores every required mutable state, not merely writing a checkpoint file | Restore model, EMA, optimizer/scheduler/scaler, replay, mapping, RNG, normalizers, novelty memory and collector boundary |
| Config handling | `initial_random_steps`, worker count and other fields are actually consumed | Honour them or remove/reject them; test every setting |
| Workers | One worker is used for initial diagnostic baseline | Scale only after ordering, provenance and throughput tests |

### Tests

- `test_training_and_evaluator_action_mapping_identical`
- `test_true_heldout_seeds_disjoint_from_replay`
- `test_novelty_insert_once_per_real_observation`
- `test_imagined_candidates_do_not_mutate_novelty_memory`
- `test_learning_progress_updates_from_real_residual`
- `test_candidate_order_does_not_change_scores`
- `test_normalizer_updates_only_after_candidate_ranking`
- `test_config_fields_are_consumed_or_rejected`
- `test_full_checkpoint_resume_restores_state`

### Pass gate: `P2_BASELINE_INTEGRITY_PASS`

- All integrity tests pass.
- Mapping matches across train/eval/resume.
- Training and held-out episode IDs are disjoint; automated assertion confirms no validation sample reaches optimizer batches.
- Candidate ranking is order invariant within a predeclared numerical tolerance (recommended absolute score tolerance `1e-6` for deterministic CPU unit tests).
- Novelty and learning-progress state changes are visible and attributable to real observations/errors.
- Save/reload/continue test restores counters and predictions within declared numerical tolerance.

**If it fails:** do not report a JEPA validation or an exploration improvement. Fix only the failed integration points and rerun Phase 2.

---

## 8. Phase 3 — Environment, replay, checkpoint smoke and short pilot

**Purpose:** Prove the entire collection/data path before expensive model training.

### Implementation scope

- Crafter adapter normalizes supported Gym/Gymnasium signatures.
- Learner receives RGB, opaque token, scalar reward and end flags only.
- Replay stores `uint8` frames and transition metadata, supports wrap-around and contiguous sequence sampling.
- Sampling never crosses terminal/truncated boundaries, ring-buffer overwrite generations or episode reset frames.
- Checkpoint save is atomic and verified by immediate load.

### Run schedule

1. **500-step smoke run:** full trace capture, one environment worker, random actions.
2. **5k–10k pilot:** one worker, same data path, fixed seeds; collect all raw transitions and a disjoint held-out set.
3. Resume once from a checkpoint and continue the pilot.

### Tests / benchmarks

- `test_env_contract`, `test_no_info_leak`, `test_opaque_actions`
- `test_replay_alignment`, `test_no_episode_crossing`, `test_ring_wraparound_generation`
- Generate 10,000 replay sample attempts; require zero invalid boundary crossings and zero stale overwritten-frame references.
- Check raw storage size, memory, disk use and sample throughput.
- Save, terminate the process, reload, and continue. Verify action mapping, replay size/pointer, step/update counters and deterministic inference outputs.
- Confirm the training sampler cannot access held-out storage.

### Pass gate: `P3_PIPELINE_PILOT_PASS`

- Zero contract violations and zero data leakage.
- Zero invalid sequences in the 10,000-sample stress test.
- No unmatched decisions/outcomes in the pilot.
- Resume round-trip passes; all required state restored.
- No unexplained trace gaps or corrupted artifacts.

**If it fails:** do not start the 50k JEPA run.

---

## 9. Phase 4 — Verify/build the JEPA baseline (no relation system yet)

**Purpose:** Get one correct, reproducible from-scratch visual world model.

### Model contract

Keep the v1.3 design as the initial implementation target unless a measured constraint forces a separately documented change:

- Small randomly initialized visual encoder; retain spatial latent tokens.
- Online encoder and frozen-gradient EMA target encoder, initialized identically.
- Anonymous action embedding for 17 tokens.
- Temporal context (GRU in v1.3) reset at episode boundaries.
- Action-conditioned latent predictor; compact predictor ensemble if it fits the T4 memory budget.
- JEPA latent prediction as the primary objective; future target features are stop-gradient and never fed into the predictor.
- Horizons `1, 2, 4, 8` and correct action/observation alignment.
- Scalar reward and termination heads remain auxiliary only.
- No pixel-reconstruction objective as a replacement for JEPA; no recipe, achievement, object-name or inventory-label heads.

Do not increase model size, ensemble members, image resolution or horizon count to hide a failed baseline. First prove the compact version works.

### Required implementation tests

1. Input/latent shapes and dtype.
2. Online encoder receives gradients; target encoder receives none.
3. Target encoder initializes as an exact online copy.
4. EMA changes after successful optimizer steps and does not change after skipped/failed updates.
5. Loss is finite and gradients are finite on synthetic and real batches.
6. Training sequence alignment: action `A[t]` predicts observation `O[t+1]`; horizon 4 targets `O[t+4]`.
7. Sampler never crosses episode end or reset.
8. Predictor does not receive future observations/targets.
9. Optimizer and EMA counters agree.
10. Checkpoint restores model, target, optimizer, scaler and scheduler.

### Pilot

- Begin from random weights.
- Use a fixed 50k environment-step development budget, with at least 10% whole episodes held out from all gradient updates.
- Split by episode/world seeds, not random frames.
- Use one exploration role/policy at this phase; keep relation components disabled.
- Run 3 independent training seeds for development decisions.

### Pass gate: `P4_JEPA_IMPLEMENTATION_PASS`

- All unit/integration tests pass.
- Training updates really occur and counters are correct.
- Loss/gradients are finite, target branch is stop-gradient and EMA updates correctly.
- Checkpoint/resume and data split integrity continue to pass.

This gate means the implementation is functioning. It does **not** yet mean the model learned useful dynamics; that is Phase 5.

---

## 10. Phase 5 — JEPA learning, action conditioning and rollout benchmark

**Purpose:** Decide whether the learned world model is good enough to support any downstream module.

### Evaluation split

- Training episodes: train replay only.
- Validation episodes: independent store, used for declared checkpoint selection/early diagnosis; never added to replay.
- Final held-out episodes: untouched until the selected checkpoint/config is fixed.
- At minimum, use 3 training seeds for this development gate. Use a separate held-out seed list and publish every seed result.

### Baselines

Evaluate on exactly the same held-out transitions:

1. Persistence: predict current latent as future latent.
2. No-action predictor: same state/context but no action conditioning.
3. Shuffled-action control: action sequence permuted across samples while preserving sequence boundaries.
4. Random-initialized predictor sanity baseline.
5. Trained action-conditioned JEPA.

Use the same EMA target encoder snapshot when comparing latent errors for a checkpoint. Report raw cosine distance, cosine similarity, normalized feature MSE, median and 90th-percentile error, by horizon.

### Action-conditioned paired test

- Build paired branches from the same initial visual context/history and run different opaque tokens using the exact saved action mapping.
- Prefer deterministic cloned-state branches if the environment API supports them; verify that support in the installed version.
- If cloning is unavailable, use matched reproducible reset/seed/context trials and clearly label the weaker control. Do not pretend approximate matching is a true counterfactual.
- At least 200 paired action/context cases across at least 20 distinct contexts is the initial target.
- Compare each prediction with the actual outcome for that exact branch. A model merely producing different outputs for different actions does not pass; it must predict the corresponding outcomes better.

### Metrics

- Held-out latent cosine distance at horizons 1, 2, 4, 8.
- Relative improvement over persistence.
- Improvement over no-action and shuffled-action controls on action-sensitive paired cases.
- Teacher-forced versus fully autoregressive rollout error.
- Feature standard deviation per channel, mean pairwise cosine between distinct frames, feature norm distribution and effective rank.
- Reward MAE/correlation and termination AUROC/Brier score where class balance permits. These are auxiliary diagnostics, not proof of mechanics understanding.

### Proposed pass targets

All should be pre-registered in config; targets are engineering heuristics, not universal JEPA constants.

1. Mean held-out one-step cosine distance is at least **5% lower relative to persistence**, averaged across 3 training seeds.
2. On the paired action-sensitive set, action-conditioned prediction error is at least **5% lower relative to both no-action and shuffled-action controls**.
3. At least **3 of 4 horizons (1, 2, 4, 8)** beat persistence on held-out episodes.
4. No collapse alarm: as an initial guardrail, at least 95% of latent channels have non-negligible variance on the fixed held-out batch, effective rank does not fall by more than 50% from the declared healthy reference snapshot, and unrelated-frame similarity does not saturate near 1. Calibrate the exact numeric variance floor in a synthetic test before the run and keep it fixed.
5. Free-running rollout error is reported and does not collapse immediately relative to persistence. A severe teacher-forced/free-running gap blocks long-horizon planning even if one-step passes.
6. Improvement is present on unseen episodes, not merely on training replay.

For stronger public claims, repeat the selected experiment on at least 5 independent training seeds and report paired confidence intervals. A 3-seed development pass is only a gate to continue engineering, not proof of robust generalization.

### Pass gate: `P5_JEPA_LEARNING_PASS`

The required prediction/action/collapse criteria all pass. A failed JEPA gate blocks MPC and any statement that the world model has learned reliable action-conditioned dynamics. Relation instrumentation work can be developed against recorded/synthetic data, but it must not be represented as being grounded in a validated model until this gate passes.

---

## 11. Phase 6 — Single-role exploration and useful data collection

**Purpose:** Check that exploration signals improve the data collected, without yet complicating the system with multiple agents.

### Implement / verify

- One Explorer policy only.
- Latent novelty from real observed latents; insert each real observation once.
- Predictor disagreement across ensemble members; treat it as an uncertainty proxy, not a calibrated probability.
- Learning progress updated from measured real prediction errors over repeated contexts.
- Candidate scoring computes all candidate values against one frozen normalization snapshot. Imagined candidates do not mutate novelty memory or running statistics.
- Retain a random action floor in early experiments; initial suggestion is at least 10%.
- Keep intrinsic, external reward, novelty, disagreement and learning-progress values in separate logs.

### Experiments

Matched environment-step comparison:

A. Uniform random action collection.  
B. Single Explorer using novelty/disagreement/learning-progress signals.  

Same model capacity, action mapping, training budget, seeds, replay capacity and learner updates. Do not compare a larger model against a smaller random baseline.

### Metrics and proposed pass targets

- Unique occupied latent clusters/coverage on a frozen embedding space.
- k-nearest-neighbour coverage distance and state revisit frequency.
- Action-token distribution and entropy.
- JEPA error reduction per 10k environment steps.
- Episode length/survival and achievement metrics as evaluator-only results.

Initial target: Explorer produces at least **10% more occupied latent clusters** than random collection on the fixed frozen embedding/cluster definition, with no more than **5% relative degradation** in held-out JEPA prediction error or task score. If this target is not met, keep random collection or revise the signal in a new experiment. The target is a project heuristic, not a known property of curiosity systems.

### Pass gate: `P6_EXPLORATION_PASS`

Exploration produces a repeatable, measured gain in coverage or data efficiency on 3 development seeds, without integrity failures or severe performance regression. If it fails, use random collection for later discovery studies and label the exploration module `DEGRADED` or disabled.

---

## 12. Phase 7 — Optional latent model-predictive control (MPC)

**Purpose:** Test whether JEPA imagined outcomes are useful for choosing low-level actions in the real environment. Keep this separate from relational planning.

### Implement / verify

- Keep random/Explorer fallback.
- Use CEM or random shooting over opaque token sequences.
- Roll forward the learned latent model; score predicted scalar reward, intrinsic information value, termination risk and rollout uncertainty.
- Execute only the first action, observe the real RGB frame and replan. Never execute the full imagined sequence blindly.
- Log predicted versus actual reward, termination, latent error, selected candidate and imagination gap.

### Tests

- Planner cannot enable before `P5_JEPA_LEARNING_PASS`.
- Identical context and candidate sequence give deterministic scores under deterministic evaluation.
- Changing candidate sequence changes predicted rollout when model evidence supports it.
- CEM probabilities remain finite and no token probability becomes invalid.
- Candidate evaluation does not mutate model, novelty memory or normalizer state.
- Planner replans after every real observation.
- Planner falls back when uncertainty/termination gates are exceeded.

### Benchmark

Compare matched held-out seeds and equal environment-step budgets:

1. Explorer without MPC.
2. Same Explorer with MPC.
3. Random-action baseline for context.

Report aggregate Crafter score, all 22 achievement rates, survival/episode statistics, reward prediction error, imagination gap, planner latency and action count. Use at least 3 development seeds and all episode-level results.

### Pass gate: `P7_MPC_UTILITY_PASS`

Enable MPC by default only when the paired mean primary metric improves and its predeclared confidence interval supports a positive effect, with no material increase in catastrophic termination or model-exploitation failures. Otherwise set MPC to `DISABLED` and continue with the validated non-MPC policy. **MPC failure does not block the separate relation-learning phases.**

---

## 13. Phase 8 — Anonymous visual state tracking

**Purpose:** Convert visual observations into stable, uncertain state snapshots and deltas that a relationship learner can use. This is not a semantic object-labeler.

### Design contract

- Input is RGB only.
- Output is anonymous IDs/slots, visual features, estimated quantities/attributes where visibly supported, confidence, uncertainty and `UNKNOWN` states.
- Track identity through camera/viewport movement, lighting/day-night changes and temporary occlusion.
- Do not map slots to hard-coded names such as wood, stone, pickaxe or workbench in the learner.
- Keep raw visual evidence references alongside derived state estimates.
- Do not claim that every hidden inventory value is observable. If the HUD is unreadable or ambiguous at 64×64, emit `UNKNOWN` rather than guessing.

### Required modules (names can fit current repository conventions)

```text
src/perception/visual_state_tracker.py
src/perception/slot_association.py
src/perception/state_change.py
src/perception/perception_diagnostics.py
```

### Tests

- Same frame twice: stable state estimate.
- Small image perturbation: no unjustified slot replacement.
- Camera/viewport shift: association remains stable where evidence permits.
- Day/night tint: track continuity and confidence degradation are measured.
- Occlusion/reappearance: uncertainty is not mislabelled as destruction.
- Quantity changes: estimates follow visible evidence.
- Ambiguous/low-resolution HUD: tracker abstains (`UNKNOWN`).
- Held-out worlds: no dependence on memorized screen coordinates or a single seed layout.

Use synthetic visual transforms for deterministic component tests. Use evaluator-only annotations on a separate held-out set to score visible changes; labels never enter the tracker or learner.

### Metrics and proposed pass targets

On at least 200 evaluator-annotated held-out visual transitions:

- Visible state-change precision and recall each target **≥0.80** for the subset declared visually identifiable.
- False state-change rate under no-op/camera/light/occlusion controls target **≤0.05**.
- Unknown/abstention rate reported separately; it is not automatically an error.
- Calibration: higher confidence should correspond to higher correctness; report reliability bins and ECE.
- Slot identity switches and missed changes reported by condition.

The 0.80/0.05 targets apply only to a predefined visually identifiable diagnostic subset, not all possible Crafter hidden state. If the tracker fails, limit relation discovery to the reliable state features or fix perception; do not let uncertain visual deltas become strong evidence.

### Pass gate: `P8_VISUAL_TRACKING_PASS`

The tracker passes on held-out diagnostic frames, abstains under ambiguity, and provides complete raw-to-state traceability. This unlocks relationship candidate generation for the supported visual features only.

---

## 14. Phase 9 — Relationship candidate generation and append-only evidence ledger

**Purpose:** Turn observed action-conditioned state changes into testable hypotheses. A candidate is not a confirmed mechanic.

### Candidate representation

Each candidate should minimally store:

```text
relation_id / revision_id
candidate_created_event_id
context_signature_ref
precondition_feature_refs + confidence
opaque_action_token
predicted_effect_feature_refs
observed_effect_feature_refs
quantity/attribute deltas + uncertainty
supporting_evidence_event_ids
contradicting_evidence_event_ids
control_evidence_event_ids
independent_episode_count / world_count
candidate_generator_version
confidence_components + confidence_method_version
status: CANDIDATE | SUPPORTED | PROVISIONAL | REJECTED | STALE | SUPERSEDED
last_tested_event_id
next_experiment_rationale
```

Human-readable names belong only in evaluator annotations. The candidate generator must not import a Crafter recipe table or mechanics reference.

### Evidence ledger

Append-only. Every real observation transition records:

- Source transition/decision ID.
- Before and after state snapshot references and state deltas.
- Action token and action-mapping/model versions.
- Tracker confidence and ambiguity flags.
- Predicted effect versus observed effect.
- Support / contradiction / neutral / inconclusive label from learner-side evidence rules.
- Whether the evidence is independent of previous episodes/worlds.
- Confounders: lighting, camera movement, occlusion, terminal event, uncertain track, concurrent effects, etc.
- Scoring rule/config version.

Never edit old evidence to make a candidate look stronger. Duplicate evidence from the same correlated sequence must not count as independent samples.

### Candidate lifecycle

A conservative initial lifecycle:

`CANDIDATE → SUPPORTED → PROVISIONAL → CONFIRMED`

or `REJECTED`, `STALE`, `SUPERSEDED`.

Only Phase 10 can promote a candidate to `CONFIRMED`. Candidate generation must not label itself as causally certain. Every confidence/status mutation must refer to evidence IDs and the calculation version.

### Tests

- Candidate stores its source transition and action decision.
- Same data replayed twice does not create independent support twice.
- Support and contradiction both remain visible.
- A visual tracker uncertainty flag reduces or blocks evidence strength.
- Context mismatch creates a separate candidate or abstention, not false merging.
- Confidence components reconstruct the reported confidence.
- Evidence ledger can replay a candidate's complete history.
- Fault injection: reversed before/after frame, wrong action token, duplicated transition, and false tracker association are detected or reduce confidence.

### Benchmarks and proposed pass targets

Build a synthetic relation test suite with known abstract transitions and known false correlations. In addition, score candidate quality on a separate evaluator-annotated Crafter diagnostic set containing only effects judged visually identifiable.

Initial targets on the held-out diagnostic subset:

- Candidate precision **≥0.70** and recall **≥0.60** for the predefined relation classes that are actually observable in RGB.
- Duplicate-evidence acceptance **0** in the test suite.
- Every candidate is traceable to at least one real transition.
- No candidate is promoted to confirmed in this phase.

These are engineering targets; if annotation reliability is inadequate, report candidate metrics as not measurable rather than fabricating ground truth.

### Pass gate: `P9_RELATION_CANDIDATE_PASS`

Candidates are traceable, evidence is append-only, contradictions are retained, false correlations are controlled on the diagnostic suite, and all candidates remain explicitly unconfirmed pending Phase 10.

---

## 15. Phase 10 — Controlled conditional testing and relation confirmation

**Purpose:** Distinguish “this action happened before this effect” from “this action has this effect under these visual preconditions.” This phase must occur before confirmed knowledge is committed to long-term relation memory.

### Test design

For a candidate relation, compare:

- Same or closely matched visual context with the candidate action token.
- Same or closely matched visual context with a relevant control action token.
- Contexts where the suspected precondition is present versus absent/uncertain.
- Repeated independent episodes/worlds.

Preferred method: branch from an evaluator-created cloned state if the installed environment supports safe, deterministic state cloning. The clone is used only to construct a test and calculate outcomes; it never provides hidden state to the learner. If cloning is unavailable, use matched reproducible reset/seed trials and explicitly report residual confounding. Do not call approximate matching a perfect counterfactual.

The learner-side test receives RGB/context, opaque action tokens, next RGB and permitted scalar reward/end flags. Evaluator hidden state can score whether the test cases truly matched or whether the effect happened, but it cannot change the model's predictions or candidate confidence during action selection.

### Tests

- Positive-control relation repeatedly produces the expected visible effect when preconditions match.
- Negative-control relation does not pass merely because action and effect co-occurred.
- Same action under changed context predicts different effects when evidence supports conditionality.
- Wrong action mapping is detected.
- Confounders cause abstention or lower confidence.
- Contradictory evidence weakens/splits hypotheses without deleting history.
- Permuting action tokens or context labels breaks the apparent relationship.
- Confidence is calibrated on a separate calibration split.

### Proposed confirmation criteria

For each relation promoted to `CONFIRMED`:

1. At least **20 independent matched contexts** across at least **3 episode/world seeds** (or a documented equivalent sample-size justification).
2. The effect difference between treatment and control has a **95% bootstrap confidence interval excluding zero** on the predeclared effect metric.
3. Multiple-hypothesis correction is applied across the candidate family (initial suggestion: Benjamini–Hochberg FDR `q < 0.05`).
4. Negative-control false confirmation rate is **≤5%** on the declared test suite.
5. Relation confidence is calibrated separately; an unexplained scalar confidence is not sufficient.
6. No unresolved high-confidence contradictory evidence exists.

A sparse mechanic may not reach 20 independent contexts in the development budget. In that case it remains `PROVISIONAL` or `UNKNOWN`. Do not fabricate confirmation or overfit by lowering the evidence count after seeing the result.

### Pass gate: `P10_RELATION_VALIDATION_PASS`

The system can correctly confirm, reject and abstain on synthetic and held-out diagnostic relations; false confirmations are controlled; all confirmed relations have independently testable evidence. This unlocks durable confirmed-relation memory.

---

## 16. Phase 11 — Persistent relation memory and retrieval

**Purpose:** Save tested knowledge and retrieve it in later episodes/worlds. This memory is distinct from JEPA weights, EMA target weights and raw replay.

### Memory components

1. **Parametric world-model memory:** JEPA encoder/predictor/context weights.
2. **Target memory:** EMA target encoder.
3. **Experience memory:** replayed RGB/action/reward transitions.
4. **Relational memory:** versioned relation hypotheses, evidence links, confidence, conditions, outcomes and lifecycle.

Do not describe latent novelty memory as a symbolic relationship memory.

### Implement

- Durable relation store with versioned schemas and stable IDs.
- Index over anonymous context/precondition signatures.
- Retrieval by current visual state/context with confidence and precondition matching.
- Include candidate rejection reasons and stale/conflicting relation handling.
- Save/restore relation memory in checkpoints, with atomic writes and checksums.
- Keep confirmed, provisional and rejected entries distinct. Provisional entries may guide targeted exploration, but must not be treated as facts in a task plan.
- Never force a prior relation onto a context whose match is weak; abstain and explore.

### Tests

- Save/load round trip preserves IDs, revisions, evidence, confidence components and lifecycle.
- Corrupt checkpoint does not overwrite the last valid memory snapshot.
- Identical query retrieves the expected relation deterministically.
- Similar-looking but incompatible preconditions do not trigger the relation.
- Contradictory evidence updates the status while preserving prior revisions.
- Stale relation is not used without a warning/abstention.
- Retrieval trace records included and rejected candidates and the reason for each.
- Relation memory is restored after a process/notebook restart.

### Benchmark and proposed pass targets

Use at least 100 held-out memory queries over previously observed contexts, including positive matches, mismatches, uncertain contexts and contradictions.

- Retrieval precision **≥0.80** on queries with a valid known relation.
- False retrieval rate **≤0.10** on incompatible/mismatch queries.
- Save/load equality: **100%** for IDs, revisions, evidence references, status and stored confidence components.
- No relation status/confidence change without a referenced evidence or explicit maintenance event.

### Pass gate: `P11_RELATION_MEMORY_PASS`

At least one relation that was confirmed in an earlier episode is saved, the process is restarted, and the same relation is retrieved correctly in a later episode. The test must be repeatable and fully traced.

---

## 17. Phase 12 — Compositional planning from confirmed relations

**Purpose:** Compose multiple learned relationships into a plan, execute one step at a time and recover when reality differs from prediction.

### Separate the planning layers

Keep separate logs and code boundaries for:

1. Goal proposal — what observable target/effect to pursue and how it was proposed.
2. Relation retrieval — which confirmed relations match the current context.
3. Dependency composition — how one relation's effect satisfies another relation's precondition.
4. Subgoal ordering — feasibility, resource estimates and uncertainty.
5. Low-level control — JEPA predictions and optional CEM/MPC action-token selection.
6. Execution monitoring — predicted effect versus actual observation, progress and replanning.

### Rules

- Every dependency edge must reference one or more `CONFIRMED` relation IDs/revisions.
- Never hard-code a Crafter crafting tree or recipe fallback.
- If no supported chain exists, return `NO_SUPPORTED_PLAN`, abstain and explore.
- Execute only the first action, observe the real frame, update state and replan.
- If the precondition no longer matches or the predicted effect fails, suspend/revise the plan and append contradiction evidence.
- Goal selection during training must use permitted generic visual-state objectives, intrinsic objectives or scalar environment reward. Named achievements can be used only by the evaluator to score outcomes.

### Tests

- Synthetic micro-world with 3–5 abstract relations and a multi-edge chain.
- Missing relation edge causes `NO_SUPPORTED_PLAN`, not a hard-coded path.
- Unsupported/provisional relation cannot appear as a confirmed plan edge.
- Wrong precondition blocks relation use.
- Contradiction causes replan/abandonment.
- After every real action, execution trace links expected effect, actual visual delta and next plan decision.
- CEM/MPC is tested as a separate low-level controller; relation graph and latent MPC failures are separately attributed.

### Benchmarks and proposed pass targets

- Synthetic micro-world chain completion **≥80% over 100 randomized trials**.
- Unsupported-edge use **0** in tests.
- Every plan edge references a confirmed relation revision and evidence IDs.
- On held-out Crafter diagnostic tasks, report supported-plan rate, subgoal completion, plan steps, relation-use success, invalid-plan rate, fallback frequency and predicted-vs-observed effect accuracy.
- On real Crafter, compare against the same validated exploration policy without relation planning, using matched seeds/budgets. Require a predeclared positive improvement on the primary task metric before making a real-world utility claim.

### Pass gate: `P12_COMPOSITION_PASS`

The planner reliably composes supported relations in synthetic tests and does not use unsupported edges. Real Crafter planning benefit must be reported separately; synthetic composition success does not prove real-world benefit.

---

## 18. Phase 13 — Frozen cross-episode and cross-world transfer

**Purpose:** Prove that previously acquired relationships are used in new worlds, not silently relearned.

### Freeze protocol

After discovery and relation-memory phases:

- Save and hash model weights, EMA, action mapping, visual tracker, relation memory, confidence calibration and config.
- Freeze model weights, relation memory, candidate/evidence updates and all learning statistics during the primary transfer test.
- New observations can be processed for inference. New evidence may be logged in a separate diagnostic file but cannot update the frozen memory or influence the frozen-transfer result.
- Use disjoint world seeds. Keep the same action-token mapping as the discovery run so token meanings remain consistent.

### Matched comparisons

At equal test interaction budgets compare:

A. Full frozen relation memory + retrieval.  
B. Same JEPA with relation memory disabled.  
C. Memory reset/empty.  
D. Shuffled or mismatched relation memory as a negative control.  
E. Optional online-adaptation condition, reported separately and never mixed with frozen transfer.

### Metrics

- Retrieval precision and valid-precondition rate.
- Correct use of a confirmed relation on a new seed.
- Success of the predicted effect after reuse.
- Repeated experiments/rediscovery count.
- Environment steps to first successful reuse / first successful effect.
- Relation-specific and aggregate task success.
- Number of times the agent incorrectly forced a relation into a mismatched context.
- Frozen model/memory hashes before and after test; they must be unchanged.

### Proposed pass targets

- At least **5 discovery training seeds** for a stronger transfer claim, with at least **20 held-out episodes per seed** (100 episodes total) or an equivalent declared design.
- Relation memory enabled improves the predeclared transfer metric by at least **10% relative** over the empty-memory control, or reduces rediscovery interactions by at least **20%**, with a paired confidence interval supporting a positive effect.
- Shuffled/mismatched memory must not reproduce the full-memory gain; otherwise the evaluation may be measuring an unrelated factor.
- Frozen hashes unchanged; zero model or relation-memory updates in the primary transfer partition.

These are targets for a meaningful project claim, not guaranteed outcomes. If confidence intervals overlap zero, report “transfer not demonstrated,” even if selected episodes look successful.

### Pass gate: `P13_TRANSFER_PASS`

The same confirmed relation is retrieved and correctly applied in unseen world seeds under frozen model/memory, and the matched controls show that the effect comes from usable stored knowledge rather than additional steps, lucky seeds or retraining.

---

## 19. Phase 14 — Scale-up and Crafter benchmark

**Purpose:** Only after the correctness and transfer gates, run longer experiments and expand the agent population.

### Scale in this order

1. 100k interaction study with one role/one worker.
2. 200k–300k study with the best validated single-role system.
3. Add additional logical roles (Scientist, Experimenter, Balanced Planner) one at a time, sharing one JEPA model and replay buffer.
4. Compare one role versus multi-role population at equal total environment steps and model capacity.
5. Increase CPU environment workers only after ordering, queue and replay provenance are verified.
6. Run a 1M-step benchmark-style experiment only when checkpoint, storage, throughput and all required gates remain stable.

### Multi-role retention gate

Keep a new role only if matched ablation shows repeatable improvement in held-out model quality, unique-state coverage, data efficiency or achievement breadth. Do not assume more roles/workers means better learning.

### Crafter reporting

- Report all 22 per-achievement success rates.
- Compute the Crafter aggregate score using the benchmark formula defined in v1.3 and report per-run scores before averaging.
- Separate reward-enabled and reward-free results.
- Report training-budget episodes separately from frozen held-out evaluation.
- Include random-action and earlier AEON baseline comparisons where reproducible.
- Report mean, spread and confidence intervals across runs; never select only the best seed.

### Pass gate: `P14_SCALEUP_READY`

No integrity regression; resumable long runs; full trace completeness; reproducible metrics; multi-role benefit demonstrated before retaining the population; and all failed/disabled components explicitly listed. A large benchmark score is not a substitute for relational-transfer evidence.

---

## 20. Standard benchmark report — required for every phase

Every phase report must contain:

```text
phase_id:
phase_name:
code_sha:
config_hash:
model_version:
checkpoint_id:
training_seed_list:
validation_seed_list:
final_test_seed_list:
environment_step_budget_planned / actual:
optimizer_update_budget_planned / actual:
wall_clock_seconds:
hardware_and_package_versions:
data_split_manifest_hash:
tests_executed:
test_results:
primary_metric_and_threshold:
secondary_metrics:
confidence_interval_or_reason_not_available:
trace_completeness:
artifacts_and_plots:
failures_and_first_divergence:
changed_files:
known_limitations:
gate_status:
next_unlocked_phase:
```

### Minimum artifact layout

```text
artifacts/implementation/
  MASTER_STATUS.md
  IMPLEMENTATION_LEDGER.md
  CURRENT_STAGE.md
  DECISIONS.md
  RISKS_AND_BLOCKERS.md
  FILE_AUDIT.csv
  interfaces/
  stages/
    phase-00-audit/
    phase-01-observability/
    phase-02-baseline-integrity/
    phase-03-pipeline-pilot/
    phase-04-jepa-implementation/
    phase-05-jepa-learning/
    phase-06-exploration/
    phase-07-mpc/
    phase-08-visual-tracking/
    phase-09-relation-candidates/
    phase-10-relation-validation/
    phase-11-relation-memory/
    phase-12-composition/
    phase-13-transfer/
    phase-14-scaleup/
  experiments/
  failures/
  reports/
```

Each phase directory should include `PLAN.md`, `CHANGES.md`, `TEST_RESULTS.json`, `METRICS.json`, `GATE.json`, `debug_report.md` and referenced traces/checkpoints as appropriate.

---

## 21. Tests that must exist before claiming the system works

### Environment and data integrity

- `test_env_contract`
- `test_no_info_leak`
- `test_opaque_actions`
- `test_training_and_eval_action_mapping_identical`
- `test_episode_split_disjointness`
- `test_no_episode_crossing`
- `test_ring_wraparound_generation`
- `test_replay_sample_provenance`

### Observability and checkpoint

- `test_decision_outcome_trace_complete`
- `test_debug_logging_does_not_change_policy`
- `test_trace_reopen_checksum`
- `test_checkpoint_contains_all_stateful_components`
- `test_uninterrupted_vs_resumed_equivalence`
- `test_corrupt_checkpoint_preserves_previous_latest`

### JEPA

- `test_jepa_shapes_and_alignment`
- `test_target_encoder_stop_gradient`
- `test_ema_update_matches_optimizer_steps`
- `test_no_future_target_leak`
- `test_heldout_metrics_use_independent_episode_store`
- `test_action_conditioning_paired_branches_use_same_mapping`
- `test_representation_health_report`
- `test_free_running_rollout_metrics`

### Exploration and MPC

- `test_novelty_updates_once_from_real_observation`
- `test_learning_progress_updates_from_real_error`
- `test_candidate_order_invariance`
- `test_imagined_candidates_do_not_mutate_memory`
- `test_mpc_requires_jepa_gate`
- `test_planner_replans_after_real_observation`
- `test_planner_logs_predicted_vs_actual_outcome`

### Perception and relationships

- `test_visual_tracker_abstains_when_ambiguous`
- `test_camera_light_occlusion_stability`
- `test_relation_candidate_has_source_transition`
- `test_evidence_ledger_append_only`
- `test_duplicate_evidence_not_independent`
- `test_contradiction_preserves_history`
- `test_relation_confidence_components_reconstruct`
- `test_controlled_test_negative_controls`
- `test_confirmed_relation_has_independent_evidence`

### Memory, planning and transfer

- `test_relation_memory_roundtrip`
- `test_relation_retrieval_match_and_mismatch`
- `test_stale_relation_not_used_silently`
- `test_plan_edges_reference_confirmed_relations`
- `test_unsupported_plan_abstains`
- `test_plan_replans_after_actual_outcome`
- `test_frozen_transfer_forbids_updates`
- `test_shuffled_memory_negative_control`

For each test, store test ID, code revision, config hash, result, duration, stdout/stderr, failure artifact references and test-run ID. Do not mark unexecuted tests as passed.

---

## 22. Blocking rules — what failure stops what

| Failed gate | Must stop | May continue |
|---|---|---|
| P0 audit | All new implementation beyond resolving audit blockers | Documentation-only cleanup |
| P1 observability | Long collection/training and root-cause claims | Fixing trace writer/tests |
| P2 baseline integrity | All meaningful training/benchmark claims | Fixing integration defects |
| P3 pipeline pilot | JEPA development run | Adapter/replay/checkpoint fixes |
| P4 JEPA implementation | JEPA learning claims and downstream model use | Synthetic model tests/fixes |
| P5 JEPA learning | MPC and claims of learned action-conditioned dynamics | Debugging JEPA; synthetic relation-ledger development clearly labelled |
| P6 exploration | Claim that novelty policy improves data collection | Random collection; relation work can use random data |
| P7 MPC utility | Default MPC use | Non-MPC exploration and relational phases |
| P8 visual tracking | Relationship claims based on unreliable tracked features | Improve tracker; use only proven reliable visual features |
| P9 candidate quality | Controlled promotion of poor-quality candidates | Synthetic ledger tests and candidate-generator debugging |
| P10 controlled validation | Marking relations confirmed | Keep candidates provisional; collect more tests |
| P11 memory/retrieval | Relation-aware planning and transfer claims | Fix persistence/retrieval |
| P12 composition | Claims of multi-step planning utility | Use single-step validated control/exploration |
| P13 transfer | Claims that memory generalizes across worlds | Report within-episode or online-adaptation results separately |
| P14 scale-up | Expand workers/roles/1M run | Continue only the failed gate's debugging experiment |

A phase failure is not hidden by moving to the next phase. A gate can be `DEGRADED` only if the dependent feature is explicitly disabled and the remaining system is still valid.

---

## 23. Implementation-agent handoff protocol

The agent implementing this plan must:

1. Read the entire v2.0 plan and inspect the current SHA before changing files.
2. Start at Phase 0. Do not begin by building the visual tracker or relation learner.
3. Produce a file-by-file audit against actual current source.
4. Recheck earlier reported defects. Treat them as hypotheses until reproduced on the current SHA.
5. Implement only the current phase; no unrequested subsystem expansion.
6. Add or update tests with every module change.
7. Run the acceptance benchmark and save artifacts before marking a gate `PASS`.
8. Never fabricate test output, training curves, benchmark performance or relation evidence.
9. Never use Crafter hidden state, achievement labels, recipes or evaluator annotations in learner code.
10. Never silently change thresholds, action mapping, dataset splits, model architecture or environment budgets.
11. Preserve old plans, checkpoints, experiment outputs and failed runs.
12. At phase completion, report exact changed files, tests run, benchmark results, gate status and next phase.

### Exact next task

**Start Phase 0 only.** Audit the current GitHub commit and the existing AEON implementation, run the tests that already exist, and create the file audit/status report. Do not add the visual state tracker, relationship learner, persistent relation memory or compositional planner yet.

---

## 24. What success must be called — precise vocabulary

Use these phrases consistently:

- **JEPA loss decreased:** optimization metric only.
- **Held-out prediction improved:** validated prediction evidence on episodes excluded from training.
- **Action conditioning demonstrated:** paired action tests show better outcome prediction than no-action/shuffled controls.
- **Visual state tracking validated:** observable visual state changes are tracked within declared accuracy/uncertainty limits.
- **Relationship candidate discovered:** a hypothesis was generated from observed evidence.
- **Relationship supported:** repeated observations support it, but causal/conditional tests are incomplete.
- **Relationship confirmed:** controlled tests and independent evidence pass the declared confirmation gate.
- **Relationship persisted:** it survives process restart/checkpoint reload.
- **Relationship retrieved:** memory returns it for a later matching context.
- **Relationship reused:** the agent acts using it and the expected effect is observed.
- **Cross-world transfer demonstrated:** frozen knowledge is correctly reused on held-out worlds and beats matched no-memory/shuffled-memory controls.
- **Compositional planning demonstrated:** a multi-edge plan uses confirmed relations and executes successfully with closed-loop replanning.

Do not collapse all of these into “the model understands Crafter.”

---

## 25. Final reality check

The 50k-step JEPA pilot, 10k collection pilot, 100k–300k exploration studies and 1M-step benchmark budgets are experiment sizes, not promises of discovery. “Find the table/pickaxe/mine-stone relationship in the first 10 episodes” is not an acceptance guarantee. Measure interaction steps, independent contexts, evidence quality, false confirmations, retrieval and transfer instead of promising an episode count.

A model that predicts images/latents but never learns action effects has failed the action-conditioning gate. A model that learns predictive effects but cannot track visible preconditions has not demonstrated relational discovery. A system that finds candidate relations but never tests contradictions has not demonstrated confirmed knowledge. A memory that stores relations but cannot retrieve them in a new world has not demonstrated transfer. A planner that succeeds only in imagined rollouts has not demonstrated real-world utility.

**The project progresses only when the evidence for the current phase is strong enough to justify the next phase.**
