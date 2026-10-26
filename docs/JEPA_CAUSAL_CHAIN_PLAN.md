# JEPA Causal-Chain Learning Plan

## Purpose

Make the Crafter agent learn and execute: wood -> table -> pickaxe -> stone -> furnace -> iron.

The project goal remains a JEPA-based agent that learns world dynamics from experience, plans in latent space, and can later transfer to other games and robotics environments.

This document separates verified facts, hypotheses, permitted learned additions, and privileged or hardcoded assistance. Every phase needs a reproducible experiment and an ablation.

## Implementation status

- **Phase 0: implemented.** Run `python3 scripts/run_phase0_chain_diagnostics.py` to produce same-seed traces and controller ablations in `results/phase0_chain_diagnostics.json`.
- **Phase 1: implemented.** Run `python3 training/run_phase1_transition_learning.py --shards_dir <sequence-shards>` to fine-tune an existing v3 checkpoint while retaining and logging the JEPA latent objective.
- **Phase 1.5: implemented.** Run `python3 scripts/run_phase1_5.py --data_root <data-root> --manifest <manifest.json> --audit_only` to create a reproducible deduplicated multi-directory manifest, or omit `--audit_only` to run the combined Phase 1 fine-tune.
- Existing shards remain readable. New state-delta labels are optional; old data still trains latent, event, reward, health, and termination objectives. The Phase 1.5 manifest prefers final cumulative shards, collapses duplicate transition content, and prefers labeled copies when available.
- No result is called a learned causal relation until held-out transition metrics and intervention/ablation evidence are recorded.

## 1. True-JEPA contract

The current core is JEPA-style because it has an online encoder, EMA target encoder, stop-gradient target representations, an action-conditioned latent predictor, latent prediction instead of pixel reconstruction, and anti-collapse monitoring.

Allowed learned additions include reward/termination heads, generic state-delta prediction, event and uncertainty heads, recurrent belief, learned options, episodic memory, and latent-space planning.

The JEPA objective must remain active and separately logged. Auxiliary losses must not replace it.

For strict evaluation, the agent may use raw observations, previous actions, explicitly permitted scalar reward, and internal recurrent/episodic state.

Strict evaluation must not use `info["inventory"]`, `info["achievements"]`, engine coordinates/maps, recipe tables, hardcoded action sequences, or rules such as `if wood >= 2: place_table`.

The Crafter HUD is part of the image and may be read by the visual encoder. Reading the engine inventory dictionary is privileged diagnostic mode.

Every benchmark must declare one track: `strict_visual_jepa`, `privileged_diagnostic`, `demonstration_assisted`, or `pure_self_play`. Scores from different tracks must not be combined.

## 2. Verified repository evidence

- `models/jepa.py` implements the online encoder, EMA target, latent predictor, stop-gradient target, MSE latent loss, and variance regularization.
- `models/jepa_v3.py` adds recurrent state, task heads, value heads, policy prior, and recurrent imagination.
- `planning/semantic_memory.py` stores visual keys, inventory snapshots, actions, inventory deltas, and returns.
- `planning/mpc_v3.py` gates table landmark pull on wood, tracks offsets, adds survival subgoals, and supports a stone landmark.
- `deduce_active_subgoal()` is currently mostly a one-level resource lookup, not recursive graph search.
- `scripts/screen_v3_dual_memory.py` passes inventory and achievement information into the memory/writer; that path is not strict pixels-only evaluation.

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

Confirmed causes:

1. The strict visual path has no explicit learned resource-transition head.
2. Backward chaining is not recursive causal graph search.
3. Primitive MPC with `H=10` is too short for the complete chain.
4. Frontier events are sparse, so direct stone evidence is weak.
5. The dual-memory path is privileged because it reads environment state.

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

**Gate:** on held-out chain windows, state-delta prediction beats a persistence baseline, counterfactual actions produce distinct effects, rare-event prediction improves, and strict/privileged results are separate.

## 6. Phase 2: learn a causal transition graph

Store transition prototypes containing precondition embedding, action/option, observed effect delta, postcondition embedding, success probability, duration, risk, visual goal key, and return.

Graph edges must come from trajectories. The search algorithm may be engineered; table/pickaxe/stone relations must come from data.

For a stone goal, recursively search for producers of missing state: stone requires pickaxe; pickaxe requires table and wood; table requires wood.

Use beam search or A* over learned edges and rank by success probability, duration, survival risk, and uncertainty. Do not select only the highest-return memory because it may be unreachable.

**Gate:** every planned edge has stored trajectory evidence. A hardcoded wood-to-table-to-pickaxe rule fails this phase.

## 7. Phase 3: persistent learned options

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

Generic delta/event/reward/done heads, 1-to-10-step consistency, held-out chain-window evaluation, and Drive-resumable checkpoints.

### Campaign C: graph and options

Graph from successful/failed transitions, goal-conditioned options, and primitive-versus-hierarchical comparison.

### Campaign D: scale and lock

Balanced self-play, 25-to-50-step objectives, and strict/privileged/demonstration-assisted reports.

Each campaign ends with a saved manifest and reproducible report.

## Final decision rule

The project remains a true JEPA-based agent if JEPA latent prediction remains the world-model foundation; strict planning state is inferred from observations/actions; auxiliary transition heads are generic and learned; memory/graph edges come from experience rather than recipe code; hierarchy is ablated; and privileged diagnostics are clearly labelled.

The immediate next step is **Phase 0 instrumentation**, not another reward coefficient or landmark heuristic. First measure the first failed link, then implement the smallest root-cause change and verify it independently.
