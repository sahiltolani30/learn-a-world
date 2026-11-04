# MOSAIC Architecture Decision Ledger

This ledger records the first research cycle's decisions. “Selected” means selected as the current engineering hypothesis, not empirically proven in Crafter. Every provisional decision has a falsification experiment.

## Decision records

### D-001 — Decision-loop information boundary

- **Decision:** the learner receives only `obs_uint8`, previous scalar reward, step/reset information, and its own action history; the public action API is exactly `act(obs_uint8, reward_prev, step)`.
- **Alternatives:** privileged inventory/map inputs; semantic action names; evaluator metadata; hidden achievement arrays.
- **Evidence:** E-001, E-011, E-026; `AGENTS.md`.
- **Arguments:** the mission is discovery from interaction, and the repository's integrity protocol forbids privileged state and hand-authored relations.
- **Objection:** visual inventory/status pixels are still information about state.
- **Resolution:** pixels are allowed by the stated environment interface; structured parses, true arrays, and evaluator `info` are not.
- **Confidence:** high.
- **Gate:** static AST/dataflow audit plus black-box masked-environment execution.

### D-002 — Baseline before complexity

- **Decision:** implement and measure a strong recurrent/object-centric PPO baseline before promoting a world-model, memory, or hierarchy component.
- **Alternatives:** compare only against random/PPO-CNN; start directly from AEON; compare only prediction losses.
- **Evidence:** E-003, E-015, E-016.
- **Arguments:** published Crafter results show that carefully tuned recurrent and object-centric baselines can outperform generic world-model implementations; the local project's prediction gates are not competence gates.
- **Objection:** a baseline delays the proposed research architecture.
- **Resolution:** it is the minimum control experiment required to distinguish architecture value from implementation/tuning variance.
- **Confidence:** high.
- **Gate:** locked seeds, equal step/wall-clock budgets, per-achievement and transfer metrics.

### D-003 — Visual representation and belief state

- **Decision:** use spatial visual tokens plus a recurrent stochastic belief model, anchored by an observation/predictive information objective; do not use pure cosine JEPA prediction as the sole perception loss.
- **Alternatives:** pure AEON JEPA; pixel-level video model; object-centric slots as the only encoder; MuZero-style value-equivalent hidden state.
- **Evidence:** E-004, E-007, E-008, E-024.
- **Arguments:** Crafter needs local geometry and hidden history; recurrent latent models have direct visual-control evidence; pure predictive embeddings can discard visually important static/dynamic details; object-centric models are promising but need a controlled cost/benefit test.
- **Objection:** reconstruction can preserve irrelevant pixels and waste T4 capacity.
- **Resolution:** use masked/lightweight reconstruction or information-floor supervision as a representation anchor, while evaluating planning-relevant latent prediction separately; tune its contribution by held-out control and collapse gates.
- **Confidence:** medium-high for the direction, medium for exact loss composition.
- **Falsification:** scene-swap, object appearance OOD, action-conditioned prediction, reward/value probes, and planner competence ablations.

### D-004 — Dynamics and uncertainty

- **Decision:** maintain an action-conditioned ensemble over the recurrent belief transition and calibrate epistemic disagreement against held-out transition error; model aleatoric uncertainty separately where possible.
- **Alternatives:** single predictor; raw forward prediction error; RND only; literal VIME; deterministic MuZero dynamics.
- **Evidence:** E-006, E-017, E-018, E-023.
- **Arguments:** hidden-mechanic discovery needs uncertainty over competing action effects; ensemble disagreement is a practical approximation to information gain; raw error conflates stochasticity and model failure.
- **Objection:** neural ensemble disagreement can be miscalibrated and correlated.
- **Resolution:** disagreement is an exploration proposal, never a truth oracle; require calibration and real-transition verification.
- **Confidence:** medium-high.
- **Gate:** calibration curve, uncertainty-reduction after interventions, and no benefit claim without downstream competence.

### D-005 — Exploration objective

- **Decision:** use action-conditioned epistemic information gain planned over short horizons, constrained by survival/risk and frontier returnability; use visual novelty/RND/count signals only as secondary frontier proposals.
- **Alternatives:** epsilon/random actions; RND; ICM/forward error; retrospective disagreement; empowerment; DIAYN; reward-free coverage only.
- **Evidence:** E-017, E-018, E-019, E-021, E-023.
- **Arguments:** the objective must distinguish interventions, not only visit rare images; planned novelty is more useful than retrospective novelty; persistent frontier memory prevents detachment.
- **Objection:** planning the information objective can exploit model errors and ignore survival.
- **Resolution:** use robust lower-confidence scoring, short real-action prefixes, frequent verification, and separate intrinsic/extrinsic value heads.
- **Confidence:** medium.
- **Falsification:** compare mechanic-discovery efficiency, uncertainty reduction, and downstream achievements against novelty-only and no-intrinsic baselines.

### D-006 — Persistent memory

- **Decision:** combine a slow recurrent belief, immutable/retrieval episodic atlas, and replay-based semantic/predictive consolidation.
- **Alternatives:** recurrent hidden state only; nearest-neighbor control only; semantic graph only; EWC only; generative replay only.
- **Evidence:** E-019, E-020, memory specialist memo.
- **Arguments:** exact rare discoveries and route segments need fast persistent storage; generalization needs parametric consolidation; replay protects against forgetting; no single memory mechanism covers all regimes.
- **Objection:** memory drift and storage cost can corrupt retrieval.
- **Resolution:** use a frozen/slow retrieval projection, versioned entries, route reliability metadata, and migration only through validated transforms.
- **Confidence:** high for hybrid memory, medium for storage policy.
- **Gate:** reset retention, representation drift, retrieval precision/recall, and return-to-frontier success.

### D-007 — Mechanics/relation learning

- **Decision:** maintain an append-only anonymous hypothesis/evidence ledger over learned visual events and action-conditioned effects; only promote relations after repeated controlled tests and calibrated confidence. No human recipe or affordance table.
- **Alternatives:** no relation memory; hidden-state-only planning; hand-authored symbolic graph; immediate PDDL induction.
- **Evidence:** E-001, E-006, E-023, repository integrity protocol, memory specialist memo.
- **Arguments:** the mission explicitly requires discovering unknown relationships and correcting false beliefs; an evidence ledger makes uncertainty and contradiction explicit; immediate symbolic abstraction is too brittle.
- **Objection:** event extraction from pixels may be unreliable, and anonymous events may not compose.
- **Resolution:** keep relations provisional, use raw trajectory evidence and visual verification, and disable material/progression gating until learned extraction passes its gate.
- **Confidence:** medium-low until tested.
- **Falsification:** held-out context intervention tests, false-discovery control, contradiction recovery, and transfer to unseen visual contexts.

### D-008 — Skill/option acquisition

- **Decision:** learn closed-loop options from validated trajectory segments and promote them only after outcome reliability, termination, and downstream reuse tests.
- **Alternatives:** primitive actions only; Option-Critic from scratch; DIAYN only; DADS only; fixed action chunks; language-generated skills.
- **Evidence:** E-005, E-021, memory specialist memo.
- **Arguments:** long Crafter chains exceed the current primitive horizon; learned skills reduce search depth; diversity alone is not enough, so utility gates are required.
- **Objection:** options can collapse or become non-composable.
- **Resolution:** version skills, measure distinct outcomes and reuse, retain primitive fallback, and train manager/worker on separate schedules.
- **Confidence:** medium.
- **Gate:** option diversity, goal reachability, termination calibration, and long-horizon ablation.

### D-009 — Planner

- **Decision:** plan over skills/retrieved segments with receding-horizon verification; retain primitive actions as a fallback. Defer full learned symbolic/PDDL planning until learned predicates/transitions are calibrated.
- **Alternatives:** current primitive CEM; online MCTS/MuZero; Dreamer actor only; PDDL-first; LLM planner.
- **Evidence:** E-005, E-006, E-016, memory specialist memo.
- **Arguments:** current horizon-8 primitive planning is too short for deep chains; Director supports latent hierarchy; model exploitation requires short execution prefixes; symbolic and LLM planning have weak visual-only evidence.
- **Objection:** learned skills may not be available early.
- **Resolution:** Phase 1 uses a reactive/recurrent baseline and primitive control; skill search is gated and never required for the first world-model hypothesis test.
- **Confidence:** medium.
- **Gate:** option-level search versus primitive CEM versus actor-only under identical compute.

### D-010 — Policy improvement

- **Decision:** use separate extrinsic/task and intrinsic-information value streams, with recurrent actor-critic imagination as the default learned controller; compare against PPO and value-equivalent search.
- **Alternatives:** one mixed reward critic; PPO only; CEM only; MuZero-style search only.
- **Evidence:** E-004, E-018, E-020.
- **Arguments:** intrinsic signals are nonstationary and should not corrupt task value; Dreamer-style imagination addresses long credit assignment; PPO is a strong necessary baseline.
- **Objection:** imagination can amplify model bias.
- **Resolution:** cap rollout horizon by measured model validity, use uncertainty/risk penalties, and verify real transitions frequently.
- **Confidence:** medium-high for separate streams; medium for exact controller.
- **Gate:** real-versus-imagined return gap and policy ablation.

### D-011 — Continual consolidation

- **Decision:** use reservoir + discovery/failure/uncertainty-prioritized sequence replay, historical policy/value distillation, and optional light parameter regularization only if replay is insufficient.
- **Alternatives:** FIFO replay; EWC only; GEM; generative replay; no consolidation.
- **Evidence:** E-020 and memory specialist memo.
- **Arguments:** rare discoveries and failure boundaries must survive later training; recurrent sequence replay is necessary for context; CLEAR supports replay/distillation as a practical stability mechanism.
- **Objection:** replay can be expensive and off-policy bias can grow.
- **Resolution:** measure replay mix and wall-clock cost; maintain provenance and separate old/new policy diagnostics.
- **Confidence:** high for replay-first; medium for distillation mix.

### D-012 — Systems and scaling

- **Decision:** do not set a final worker count or update ratio from the current YAML. Measure environment throughput, learner utilization, p95 action latency, replay bandwidth, and learning progress across candidate configurations on the actual Colab runtime.
- **Alternatives:** accept the current 8-worker T4 setting; maximize parallel workers; run independent learners.
- **Evidence:** E-010, E-025, Craftax systems evidence, RND scaling evidence.
- **Arguments:** more parallel games can increase throughput but reduce temporal freshness and may saturate CPU/GPU; the objective is learning per wall-clock, not games per batch.
- **Objection:** a sweep costs experiments.
- **Resolution:** profile before long training; use short fixed-seed pilots with predetermined stop criteria.
- **Confidence:** high that measurement is required; unknown final winner.
- **Gate:** selected config must dominate or be Pareto-efficient on throughput-adjusted learning progress.

### D-013 — Research stopping rule

- **Decision:** label MOSAIC as a provisional architecture until the staged gates demonstrate representation validity, action-conditioned dynamics, calibrated uncertainty, persistent memory, option utility, and downstream competence.
- **Alternatives:** declare success from prediction loss or a single Crafter score.
- **Evidence:** E-002, E-015, E-016, E-017.
- **Arguments:** prediction accuracy is not task competence; aggregate score can hide breadth/depth failures and random-baseline effects.
- **Confidence:** high.
- **Gate:** all phase exit criteria in the architecture specification.

## Revisions recorded in Cycle 1

1. **Initial attraction to pure JEPA** was rejected as the final representation objective because the repository implementation has no perception anchor, and the broader literature identifies collapse/information-loss risks. It remains an ablation.
2. **Novelty/disagreement as the complete exploration solution** was rejected after Crafter's low unsupervised scores and the noisy-TV/long-horizon literature. It is now one part of active probing plus returnable memory.
3. **Primitive CEM as the long-horizon planner** was rejected as the final abstraction because the current horizon is too short for deep chains. It remains a baseline and fallback.
4. **Pure learned symbolic planning** was rejected as an initial implementation because predicate/event calibration is unresolved. The ledger is retained as a later gated module.
5. **LLM authority** was rejected for the control-critical path because the available evidence assumes skills or text/state abstractions and does not satisfy the strict visual contract.
