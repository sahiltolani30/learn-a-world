# JEPA Causal-Chain Learning Plan

## Purpose

Build a continually learning Crafter agent that discovers useful action effects, retains knowledge across games and process restarts, and composes learned requirements into resource-aware plans. Crafter's progression is a branching dependency structure, not a single fixed chain.

The project goal remains a JEPA-based agent that learns world dynamics from experience, plans in latent space, and can later transfer to other games and robotics environments.

This document separates verified facts, hypotheses, permitted learned additions, and privileged or hardcoded assistance. Every phase needs a reproducible experiment and an ablation.

## Current decision and progress tracker

**Updated: 2026-10-27.** This tracker supersedes earlier statements that implementation or smoke-test success meant a phase's scientific gate had passed.

| Work | Status | Evidence / next gate |
|---|---|---|
| Phase 0 instrumentation | Implemented; initial Colab outputs reviewed | Three episodes per ablation identify early bottlenecks; not enough to rank configurations reliably. |
| Phase 1 transition learning | Implemented; basic Colab smoke observed | JEPA stayed active; sampled delta MSE beat persistence. Advanced-chain prediction is not established. |
| Phase 1.5 data consolidation | Implemented; laptop smoke passed | Manifest loader and combined trainer exist; correctness gaps below remain open. |
| Phase 1.5 Colab Run 3 | **Narrow state-delta gate passed reproducibly; broader gate incomplete** | Run 3A and 3B both beat their persistence baselines. Checkpoint is a Phase 1.5 transition-model candidate, not yet a full agent champion. |
| Strict persistent experience/relation memory | Design proposed; not implemented or validated | First prove useful retention after restarting the process. |
| Compositional planning and learned skills | Planned | Build incrementally after state grounding and retention tests. |

### Recorded run evidence

- Earlier Phase 1 collection: 2,000 transitions / 13 episodes, 11 wood events, 5 table events, and no wood-pickaxe/stone/coal/iron/diamond events.
- Earlier Phase 1 training: 1,000 updates; logged JEPA loss 0.5449 -> 0.1352; sampled state-delta MSE 0.0146805 versus zero-delta persistence 0.0168762 (~13% improvement). These are observations, not proof of general causal understanding or absence of representation collapse.
- `state_delta_valid_transitions=4096` in that evaluation counts sampled points, not unique episodes or transitions. Nonzero action separation is not an intervention-accuracy test. Near-perfect termination accuracy needs a majority-class baseline and terminal-event recall.
- Drive Markdown audit reports 860,314 active transitions, 223 wood-pickaxe events, 87 stone events, and 10 coal events, with only 2,000 state-delta-labeled transitions. Those counts are provisional: copied shards across directories and overlapping episodes need content-level verification. The separate 34,000-transition JSON audit describes the laptop, not Drive.
- Prior laptop checks passed: 30 focused tests and a separate 17-test group (overlapping suites), legacy-data one-step smoke, and synthetic labeled smoke. These tested execution paths, not full Colab scale or learning quality.
- Phase 1.5 Colab result: 666,338 transitions / 2,629 episodes, 2,000 valid state-delta labels (0.3001% density), 1,000 steps, horizons 1/5/10/25/50. JEPA loss decreased 1.0143 -> 0.3484, but the state-delta MSE was 0.0100678 versus persistence 0.0073529, so the model was ~36.9% worse than persistence. Only 51 valid delta transitions were sampled in validation; `loss_state_delta` was zero on 998/1,000 logged steps. The causal state-delta gate **FAILED**. No Phase 2 promotion follows from this run.
- Phase 1.5 Colab Run 2: labeled-window sampling was enabled with `labeled_ratio=0.5`; state-delta supervision appeared on all 1,000 training steps, and reported validation sampling reached 6,630 labeled points. JEPA loss decreased 0.9743 -> 0.4479, latent mean standard deviation was 1.325, and collapse was not detected. However, state-delta MSE was 0.0175844 versus persistence 0.0172794: still 1.76% worse, so the causal gate **still FAILED**. The 6,630 value is sampled evaluation points, not 6,630 independent labeled transitions; the dataset still contains 2,000 unique valid labels. Termination accuracy 0.999694 exactly equals its majority baseline, with zero terminal recall/precision and zero true positives; it is not a successful termination model. Reward MAE 0.03132 is reported batch-invariant. No Phase 2 promotion follows from Run 2.
- Phase 1.5 Colab Run 3A/3B: both repeated runs beat the sampled persistence baseline on the reported held-out labeled evaluation: Run 3A `0.0161188 < 0.0164005`; Run 3B `0.0158931 < 0.0159792`. This is a **reproducible narrow state-delta gate pass**. It does not prove full causal understanding: each run still has only 2,000 unique labels across 13 episodes, the reported 6,528 evaluation points are repeated sampled windows, termination accuracy equals the majority baseline with zero recall/precision, counterfactual separation is only output variance (not measured-outcome intervention accuracy), and no gameplay regression is included. Run 3B reports 3,832 episodes after train/validation split while Run 3A/review summaries say 2,629; resolve this provenance inconsistency before treating the manifest audit as final. No full Phase 1.5 champion or Phase 2 promotion follows solely from this result.

### Phase 1.5 output review — Run 3 narrow gate passed; broader review open

The Run 3 JSON artifacts and review report confirm two Colab executions wrote `/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_phase1_5.pt`. The manifest file itself was not present in the local workspace for independent re-verification, so shard-selection claims remain based on the reported totals. Review:

- [PARTIAL] Selected/excluded shards and event totals are reported; the manifest/content identities are not locally available for independent verification.
- [PASS, narrow] Labeled validation sampling reports 6,528 sampled points and state-delta MSE below persistence in both runs; unique valid labels remain 2,000 and independent labeled episodes remain 13 total.
- [PARTIAL] Loss progression, latent variance, and collapse metrics are reported; finite gradients, horizon-specific validation metrics, full checkpoint-loading audit, and the Run 3B episode-count discrepancy remain open.
- [PARTIAL] A validation split is reported, but episode/seed identities and fallback behavior are not independently verified.
- [PARTIAL] Reward MAE is reported batch-invariant. Termination has a majority baseline, but recall/precision are both zero, so the termination head has not passed.
- [COMPLETE, REGRESSION] Gameplay regression across locked seeds 2050–2099 (50 episodes) completed. Score regressed from 4.55% to 3.40% (-1.15%); survival was preserved at 178.34 ± 32.79 vs 181.36 ± 51.63 steps; table placement improved from 52.0% to 56.0%; wood pickaxe regressed from 10.0% to 6.0%; stone mining regressed from 2.0% to 0.0%. This confirms Run 3B must NOT be promoted to agent champion, but remains a transition-model candidate only.

Run 3 passes the narrow state-delta gate, but downstream gameplay regression confirms the model is not ready for champion promotion. Full Phase 1.5 agent gate remains PENDING. Phase 2 remains NOT STARTED.

### Collaboration

Discuss and agree on the architecture first. This assistant supplies research, implementation briefs, acceptance criteria, and reviews. Agy implements code and tests by default. Direct code implementation by this assistant is reserved for rare cases where a precise handoff is insufficient; documentation updates remain part of planning.

## 1. True-JEPA contract

The current core is JEPA-style because it has an online encoder, EMA target encoder, stop-gradient target representations, an action-conditioned latent predictor, latent prediction instead of pixel reconstruction, and anti-collapse monitoring.

Allowed learned additions include reward/termination heads, generic state-delta prediction, event and uncertainty heads, recurrent belief, learned options, episodic memory, and latent-space planning.

The JEPA objective must remain active and separately logged. Auxiliary losses must not replace it.

For strict evaluation, the agent may use raw observations, previous actions, explicitly permitted scalar reward, and internal recurrent/episodic state.

Strict evaluation must not use `info["inventory"]`, `info["achievements"]`, engine coordinates/maps, recipe tables, hardcoded action sequences, or rules such as `if wood >= 2: place_table`.

The Crafter HUD is part of the image and may be read by the visual encoder. Reading the engine inventory dictionary is privileged diagnostic mode.

Every benchmark must separately declare inference access (`strict_visual_jepa` or `privileged_diagnostic`), training provenance (including demonstration-assisted data or pure self-play), and adaptation mode (frozen evaluation or continual learning). Visual-only inference does not erase demonstration or privileged-target training provenance. Scores from different protocols must not be combined.

## 2. Verified repository evidence

- `models/jepa.py` implements the online encoder, EMA target, latent predictor, stop-gradient target, MSE latent loss, and variance regularization.
- `models/jepa_v3.py` adds recurrent state, task heads, value heads, policy prior, and recurrent imagination.
- `planning/semantic_memory.py` stores visual keys, inventory snapshots, actions, inventory deltas, and returns.
- `planning/mpc_v3.py` gates table landmark pull on wood, tracks offsets, adds survival subgoals, and supports a stone landmark.
- `deduce_active_subgoal()` is currently mostly a one-level resource lookup, not recursive graph search.
- `scripts/screen_v3_dual_memory.py` passes inventory/achievements to the writer only in privileged diagnostic mode. Its strict path withholds them, but the existing writer then lacks the positive-effect evidence its inventory-based consolidation expects.

Saved artifacts show the current chain bottleneck:

| Artifact | Tables | Pickaxes | Stone | Score |
|---|---:|---:|---:|---:|
| `results/screen_v3_pure_rl_50ep.json` | 14 | 5 | 0 | 3.5469% |
| `results/screen_v3_tabula_rasa_50ep.json` | 16 | 3 | 0 | 3.5950% |
| `results/screen_v3_champion_clean_50ep.json` | 26 | 4 | 1 | 3.1709% |

These are outcomes, not proof of internal causal understanding. The local `screen_v3_dual_memory_50ep.json` currently contains one episode, so it is not a final 50-episode benchmark.

## 3. Root-cause model

The current system combines JEPA latent prediction, task heads, recurrent belief, primitive-action CEM, episodic nearest-neighbor memory, and engineered controller terms.

The missing component is a learned compositional state-transition model. Planning needs imagined resource quantities, tool possession, object existence/location, proximity requirements, and action consequences.

Verified implementation limitations (not a complete causal diagnosis of gameplay failure):

1. Phase 1 added resource-delta heads, but current-state quantity grounding and their integration into the strict MPC decision path remain incomplete.
2. Backward chaining is not recursive causal graph search.
3. Primitive MPC with `H=10` does not directly cover complete chains; whether terminal values compensate adequately needs measurement.
4. Frontier events are sparse, so direct stone evidence is weak.
5. The existing positive-relation writer depends on environment-state evidence unavailable in strict play.

Hypotheses requiring experiments:

1. Real recurrent-state alignment may be weaker than imagined-state alignment.
2. Dead-reckoning may drift when movement is blocked or the camera changes.
3. The policy prior may imitate common actions without enough subgoal conditioning.
4. The Q signal may be too flat to rank long-term progress.
5. The encoder may not reliably represent small stone/tree objects.

## 4. Phase 0: instrument the failure

Before changing the architecture, log every step: action/reward, actual inventory and achievements in diagnostic mode, predicted state, event probabilities, active target/deficit, retrieved memory, producer action, table/stone offset, score decomposition, and selected action.

Run identical seeds with memory off, memory initially empty, preloaded memory, landmark terms off, privileged inventory access, strict inventory-free access, and policy prior disabled.

The trace must identify the first failed link: wood, table, return to table, pickaxe, stone detection, return to stone, mining, or survival.

**Gate:** do not add another controller bonus until the first failed link is identified for representative episodes.

## 5. Phase 1: learn generic state transitions

Add auxiliary predictions from composite latent state and action for next latent, generic resource/state delta, event, reward/health, termination, and uncertainty.

Crafter inventory may be a training target in a privileged diagnostic experiment. It must not be an input during strict evaluation.

Use `L = L_JEPA + lambda_delta*L_delta + lambda_event*L_event + lambda_reward*L_reward + lambda_done*L_done + lambda_consistency*L_multistep`.

`L_JEPA` must remain active. Enforce predicted state[t+1] = predicted state[t] + predicted delta[t] during imagination.

Train progressively on 1, 5, 10, 25, and 50 steps. Do not claim 25-step learning when only a 4-step loss was trained.

**Gate:** on held-out chain windows, state-delta prediction beats persistence, alternative-action predictions match measured outcomes, rare-event prediction improves, and training/inference provenance is explicit. Action-output variance alone does not pass the intervention gate.

### Phase 1.5: correctness and data gate before scaling

Entry points: `training/phase1_5_data.py`, `scripts/run_phase1_5.py`, and `training/run_phase1_transition_learning.py`. Existing data remains useful for JEPA/event/reward learning even when resource labels are absent. Missing labels must not be treated as observed zero effects.

Open work for Agy, with focused regression tests before a new long run:

- [ ] **Deduplication:** prove content duplication or cumulative inclusion before excluding a shard. Filename family/size alone is insufficient. Detect overlapping episodes across differently packaged shards; retain source provenance and compatible labels.
- [ ] **Reproducible splits:** save explicit episode identities and seeds when available; keep duplicates/known shared seeds together. Report missing seed metadata instead of claiming seed-disjoint validation. Ensure usable labeled validation and report unsupported rare events.
- [ ] **Balanced supervision:** retain broad old experience while explicitly sampling labeled transitions and rare-event windows. Report sampling proportions and unique counts; oversampling does not create new evidence.
- [ ] **Checkpoint integrity:** fail on a missing requested checkpoint rather than silently starting randomly. Preserve EMA target freeze/stop-gradient semantics; record initialization and training state.
- [ ] **Temporal alignment:** test the same observation/previous-action/hidden-state convention in training, acting, and imagination, including episode boundaries and recurrent burn-in.
- [ ] **Rollout consistency:** distinguish teacher-forced sums of delta predictions from free-running imagined state/effect consistency. Introduce longer horizons progressively and evaluate each horizon.
- [ ] **Metric correctness:** reward MAE currently sums errors across batches but divides by one batch's count. A read-only reproduction with constant unit error returned 4.0 for four batches instead of 1.0. Fix accumulation and regression-test batch-count invariance.
- [ ] **Honest evaluation:** remove silent train-set fallback for held-out reports. Report missing-label metrics as unavailable, not successful zero error. Add event support/precision/recall, terminal recall and baseline, and uncertainty calibration. Current uncertainty is trained on reward residuals, not general transition/model uncertainty.
- [ ] **Behavior compatibility:** the trainer changes the encoder without policy/value losses. Measure policy/value regression; choose a controlled representation freeze or joint/replay-based adaptation rather than assuming old heads stay compatible.
- [ ] **Scale and resume:** check host RAM use for full decoded shards and observation tensors, avoid silent buffer eviction, and checkpoint optimizer/RNG/progress for interrupted Colab runs.

**Exit:** corrected metrics and split manifest, validated labeled/rare-event coverage, horizon-specific predictive evidence, and gameplay comparison against the initialization checkpoint. A fixed number of training steps is not an acceptance criterion.

## 6. Phase 2: grounded persistent memory and compositional planning

This is a staged proposal, not an implemented capability. Reuse the JEPA checkpoint rather than restarting from scratch. Empty external memory does not remove knowledge already stored in trained weights.

### 2A. Ground the current belief in observations

Probe whether the existing recurrent representation preserves quantities, tool possession, nearby structures, facing/interaction context, and survival state. Add learned visual/HUD readouts or local spatial features only where probes show a gap. Estimate uncertainty and repeatedly correct state from observations; blindly accumulating predicted deltas drifts.

Ground-truth labels may supervise an explicitly declared training experiment, but strict acting must not receive engine inventory, achievements, positions, maps, or recipes. An achievement having occurred does not establish current resource possession.

### 2B. Persist experience, then consolidate candidate relations

Use two information layers in long-term storage:

1. **Experience evidence:** actual observation sequences, actions, rewards, termination/truncation, observation-derived state estimates/confidence, and model/source/episode identifiers.
2. **Relation hypotheses:** learned success conditions, consumption/production, reusable requirements, duration/risk, supporting examples, counterexamples, and uncertainty.

Keep predicted effects separate from effects inferred from subsequent observations. Neither a single success nor visual similarity establishes a minimum resource requirement or a causal rule. A zero-reward action is not necessarily a failure: repeated resource collection can enable later rewards.

### 2C. Plan with learned requirements and costs

Store transition prototypes containing precondition embedding, action/option, observed effect delta, postcondition embedding, success probability, duration, risk, visual goal key, and return.

Graph edges must come from trajectories. The search algorithm may be engineered; table/pickaxe/stone relations must come from data.

Once supported by trajectories, compose operations while accounting for consumed resources, reusable tools, persistent structures, and spatial access. For example, the agent should discover the total resource budget for a table and a tool by simulating their learned effects, not by receiving a table/tool recipe from us.

Use beam search or A* over learned edges and rank by success probability, duration, survival risk, and uncertainty. Do not select only the highest-return memory because it may be unreachable.

**Gate:** every planned edge has stored trajectory evidence. Start with a probabilistic transition model; introduce an explicit graph only when its abstractions are reliable enough. A hardcoded wood-to-table-to-pickaxe rule fails this phase.

### Memory lifecycle and acceptance experiments

- New lifetime: empty short-term context and empty external long-term memory, with a recorded pretrained model.
- New episode/world: reset hidden/context state and current-world locations/resource beliefs; retain learned skills and relations.
- Process restart: reload long-term memory and training state. Restore short-term context only when resuming the same saved environment state.
- Save model, optimizer, replay/evidence, relation bank, counters, model version, and RNG state. Save updates to existing entries even if bank size has not grown.
- Initially freeze the representation for the memory-only ablation. Later alternate interaction and replay-based consolidation of dynamics and behavior; re-encode evidence or version keys when the encoder changes.

Acceptance order:

1. **Retention:** discover useful wood-acquisition experience, save, restart, and improve acquisition time/success on new maps relative to memory disabled at equal interaction budgets.
2. **Composition:** predict table/tool effects and resource consumption, distinguish resource failures from proximity failures, and execute their composition on new maps.
3. **Revision:** contradictory observations lower confidence or revise a relation instead of reinforcing a false belief.
4. **Longer chains:** extend only after the simpler tests pass, while tracking survival and skill execution failures.

More memory entries are not evidence of improvement. Compare continual-learning curves separately from locked, no-update evaluation. A growing memory must not carry evaluation experience into a supposedly independent benchmark.

## 7. Phase 3: persistent learned options

Develop minimal goal-conditioned execution alongside Phase 2's first retention/composition tests. Do not wait for a large graph before checking whether the agent can execute its chosen subgoals. Broader skill discovery remains this phase's scope.

Primitive CEM should not discover a 50-step task from scratch. Use options containing a goal embedding, initiation condition, action policy, and termination condition.

An option continues until its learned goal is achieved, failure occurs, risk is too high, or timeout is reached.

Train options from successful trajectory segments and relabeled goals, not manually written recipes. Names may be used for debugging, but policies should condition on learned goal/state embeddings.

Use this hierarchy: graph search chooses an option; the option chooses primitive actions; local MPC handles movement and immediate control.

Compare primitive CEM, option policy alone, and option policy plus local MPC on identical seeds and budgets.

## 8. Phase 4: curriculum and data

Label datasets separately as random/self-play, curiosity self-play, demonstration-assisted teacher data, and near-success curriculum data.

Teacher data is useful for bootstrapping, but results trained on it must be labelled demonstration-assisted.

During training only, use saved states such as table-with-zero-wood, table-with-one-wood, table-with-pickaxe, and pickaxe-near-known-stone. Evaluate final agents from ordinary Crafter starts. This is curriculum, not cheating, if injected states never enter the benchmark.

Prioritize rare successes, failures near success, predecessor windows, and complete chain segments. Report independent episode counts, not duplicate samples.

**Gate:** the manifest reports independent counts for table, pickaxe, stone, table-to-pickaxe windows, pickaxe-to-stone windows, and complete-chain windows.

## 9. Phase 5: strict evaluation and transfer

Report P(table), P(pickaxe | table), P(stone | pickaxe), P(stone | table and pickaxe), full-chain completion, and survival to night.

Use locked held-out seeds and do not repeatedly select checkpoints on those same seeds.

The transferable core must not depend on Crafter achievement names, action-number meanings, inventory indices, fixed 2D action deltas, or recipe-specific code. Other environments may provide input, proprioception, action, and goal adapters; JEPA dynamics and learned options should remain reusable.

## 10. Anti-hallucination protocol

Before every claim or checkpoint promotion, record exact checkpoint, script/configuration, dataset sources, teacher-data usage, inventory/achievement usage at test time, seeds, episode count, score, per-achievement results, ablations, and commit/working-tree state.

Use these labels: `VERIFIED` (reproduced from code/artifact), `OBSERVED` (seen once), `HYPOTHESIS` (not tested), `PLANNED` (not executed), and `PRIVILEGED` (uses unavailable environment state).

Never call a relation learned unless a held-out intervention test shows that the model predicts consequences of alternative actions from the same state.

Do not use recipe code, test-time inventory/achievement injection, hardcoded sequences, repeated tuning on held-out seeds, or removal of the JEPA loss.

## 11. Colab campaign order

### Campaign A: diagnosis

Chain trace, memory/landmark/inventory/prior ablations, and a failure-taxonomy JSON.

### Campaign B: state-transition learning

Review the already-produced Phase 1.5 output first. Correct the open Phase 1.5 issues, then validate generic delta/event/reward/done heads, progressively extended rollouts, held-out chain windows, and resumable training state.

### Campaign C: retention, composition, and skills

Ground current state, prove memory retention after process restart, then learn and compose a small set of evidenced transitions with executable goal-conditioned skills. Compare memory off/on and primitive/hierarchical control. Expand an explicit graph only after these gates.

### Campaign D: scale and lock

Balanced self-play, 25-to-50-step objectives, and strict/privileged/demonstration-assisted reports.

Each campaign ends with a saved manifest and reproducible report.

## Final decision rule

The project remains a true JEPA-based agent if JEPA latent prediction remains the world-model foundation; strict planning state is inferred from observations/actions; auxiliary transition heads are generic and learned; memory/graph edges come from experience rather than recipe code; hierarchy is ablated; and privileged diagnostics are clearly labelled.

The immediate next step is **review the available Phase 1.5 Colab artifacts**, then hand Agy a focused correctness-fix brief. Preserve the run as evidence; do not promote its checkpoint or expand memory/planning based solely on a completed status or lower aggregate loss.

## 12. Research basis and developer-only mechanics

Reviewed on 2026-10-27:

- [Crafter benchmark](https://arxiv.org/abs/2109.06780) and [game rules](https://github.com/danijar/crafter/blob/main/crafter/data.yaml): procedural worlds, local visual observations/HUD, survival, and sparse first-unlock achievement rewards. Installed rules checked: table consumes two wood; wood pickaxe consumes one wood near a table; wood pickaxe can collect both stone and coal. These are developer evaluation facts, never supplied recipes for the strict agent.
- [Reward and interaction implementation](https://github.com/danijar/crafter/blob/main/crafter/env.py), [objects](https://github.com/danijar/crafter/blob/main/crafter/objects.py): repeat collection can have zero achievement reward; facing, placement surface, proximity, and survival affect outcomes.
- [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2): supports concurrent world-model/actor/critic learning from replay. Its Minecraft experiment includes structured observations and is not direct evidence for our pixels-only claim.
- [Neural Episodic Control](https://proceedings.mlr.press/v70/pritzel17a.html): supports fast reuse of experience with persistent memory and relatively stable representation keys; it does not by itself identify resource requirements.
- [Director](https://arxiv.org/abs/2206.04114): supports learned latent subgoals and goal-conditioned execution for long horizons; its architecture is not a drop-in JEPA implementation.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985): supports action-conditioned latent prediction and planning, not automatic discovery of arithmetic resource rules from our limited data.

The proposed synthesis is a research hypothesis to test, not a published guarantee of human-like learning speed or diamond mastery.
