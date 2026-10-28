# Phase 2 — Next Iteration Plan (GATED on Colab screening results)

> **Status: DO NOT IMPLEMENT before reading the 3-arm screening results** (Arm A ablation-C on v2 checkpoint / Arm B bank+online / Arm C bank frozen). This plan was produced by a four-stream study (engine mechanics reference, human-learning capability map, adversarial game×implementation analysis, empirical run mining) and the Colab results decide which fixes get greenlit and in what order.
>
> Companion docs: `CRAFTER_MECHANICS_REFERENCE.md` (ground truth), `PHASE2_MEMORY_REDESIGN_PLAN.md` (what is already implemented).

## 0. Results log (updated as runs complete)

**2026-10-28 — Cell 1 + smoke (pre-arms):**
- Retrain v2: Gate 5 PASS on the full Drive corpus (3,832 eps / 666k transitions); held-out MSE 0.0507 vs persistence 1.0000 (**94.9% improvement**; the 0.0323/58.1% figures circulating are the v1 record — the results JSON was overwritten by the v2 run). Pre-train MSE 20.3 → confirms the absolute-vitals label change took effect.
- Consolidation v2: 47,272 transitions / 300 episodes / **100% label coverage** / **932 prototypes** (old bank: 520). Vectorization commit 3c05aab verified: **34/34 phase2 tests pass, zero threshold/constant changes** — semantics preserved.
- Audit gate: **C, E, F, G PASS** (3 trusted wood producers, 3 enabling edges, acyclic, deltas aligned). **A, B, D FAIL** — vitals dims 12–15 entered NEEDS via statistical enrichment: successes occur while healthy, near-misses while dying → the Wilson test correctly detects enrichment, but it is a *confounder* (agent competence), not an action requirement. The engine's requirements are inventory+nearby only.
- **Decision (green-lit): exclude dims 12–15 from NEEDS candidacy** — a registered hypothesis-space restriction ("needs ⊆ inventory"), pre-sanctioned by the redesign plan's Amendment 2 fallback. Vitals remain fully active in the transition/health/maintenance channels. Not a semantic cheat: no item or action names; it restricts the hypothesis class, and vitals-causality is learned elsewhere (F1).
- Smoke (5 eps, bank+online): **3.87%**, 9/22 achievements, place_table 80%, and the designed chain fired live in one episode: collect_wood → place_table → make_wood_sword → defeat_zombie (7 unlocks, reward 6.1). ~20 hypotheses tested per episode, stable. Caveat: n=5, high variance — the arms decide.
- **Next: apply vitals exclusion → re-consolidate (~84s) → re-audit (A–D must pass) → re-read cards → launch 3-arm screening.**

**2026-10-28 — MEMORY READ (real 932-card bank): what the agent learned right vs wrong.**
CORRECT: the notebook's core lessons match reality — wood-producer cards (act5+collect_wood: produces wood ~0.97, no needs), sapling cards (produces sapling ~0.9), 83 event-grounded cards across 8 achievements including **8 defeat_zombie cards, 15 eat_cow cards, 13 collect_wood, 1 place_table, 1 collect_drink**; 3 trusted wood producers; acyclic graph.
THREE MISTAKES (top priority, in order):
1. **Vitals preconditions on 750 cards** ("needs health ≥ 4.0") — root cause: with absolute vitals, successes happen while healthy and near-misses while dying, so health/energy are *genuinely enriched* at successes — the enrichment test detects the competence confounder, not causation. Fix: the already-decided exclusion (needs ⊆ dims 0–11).
2. **NEW — universal-item false positives: 144 cards demand saplings** (incl. a place_table event-free card: needs sapling ≥ 0.597; even movement cards demand saplings). Root cause: sapling is present ~92% of the time (the agent farms it obsessively); relative-only enrichment (Wilson separation + gap ≥ 0.20) passes when the context-shifted failure pool happens to hold slightly fewer. **Fix F11 (new, top priority with #1): absolute failure-absence criterion — a dim is a need only if it is ABSENT in a substantial fraction of failures: f_d ≤ 0.5.** Wood passes (absent in ~82% of failures); sapling dies (present in ~80% of failures — undiscriminating). Registered hyperparameter, item-agnostic.
3. **Edge explosion: 89,624 edges** (mostly vitals-producer × vitals-needer pairs, plus junk event-free movement/noop cards whose readout noise produced spurious deltas). Fixes: #1 removes vitals needs (edges collapse); **extend the event-free breakthrough criterion to require the productive delta on inventory dims 0–11** (kills movement/noop junk cards at the source).
Also noted: most event cards have low Wilson LB (0.02–0.09) because near-miss pools are huge — ranking works on relative order (smoke proved it), but expect compressed trust values; and only ONE event-grounded place_table card exists (second placements within an episode are event-free) — acceptable, but the sapling fix must cover event-free cards too (F11 does).

## 0b. User ruling + the STONE PACKAGE (2026-10-28)

**User ruling (binding):** scripted teacher demonstrations for the deep tiers are **REJECTED** — they violate the true-learning rule, even at training time. The deep-tier discovery paths are limited to: (a) capability-triggered re-probing of previously-silent contexts, (b) analogical hypothesis cards, both zero-hardcode. The "richer corpus" deferred item is struck from the plan.

**Mission: STONE.** Evidence: pickaxe→stone conversion 0/12 — the agent had pickaxes and never reached stone (navigation + survival), while the engine guarantees first-try success once there. The blocking chain is: **survive → walk → commit.**

**Stone package (implementation waves):**
- Wave 1 (small, landing now): **F4** enabler reactivation (stone_pickaxe needs a table — rebuild when absent), **F5** value propagation (stone chain becomes attractive), **F8** uncertainty-aware deficits (no flapping), **F9** unlock-step logging (measure the conversion rate), **F7** = screening config: run the next cycle with `--tte_weight > 0` (no code).
- Wave 2 (next session, fresh context): **F2** chain links as committed options, **F3** learned spatial memory (the reach build), **F1** maintenance head.
- Methodology note: the 3 arms running now measure Phase A code and serve as the **control group** for the stone package; the package gets its own 3-arm cycle after landing.

## 1. Where the agent actually is (empirical, from recorded runs)

- **93% of audited deaths are monsters; 73% at night.** Mean death step 177.7 (of 500-step diagnostic episodes); 0/15 episodes survived to the cap. Starvation/dehydration only 7%.
- **The stone tier is a hard wall:** pickaxe→stone conversion **0/8** (ablation C) and **0/4** (phase2) vs 2/5 in the *weaker* T2 baseline. collect_stone went 4% → 0% under the transition planner. 11 of 22 achievements have never been unlocked in any recorded run.
- **The event heads cannot see the chain that never fires:** recall@0.5 = 0.0 for make_wood_pickaxe, make_wood_sword, place_table, collect_stone, collect_wood, defeat_zombie, wake_up, eat_cow; only collect_drink (0.68) and place_plant (0.91) have usable recall; 11 frontier heads have had zero positive training samples ever. Calibrated macro PR-AUC 0.640 but defeat_zombie recall stays 0.
- **No death/night model in the loop:** `death_risk_factor: 0.0`, `tte_weight: 0.0` (the trained TimeToEvent head is plumbed but zero-weighted).
- **Reward↔length correlation ≈ 0.42** → survival is the multiplier: ~+1.4 achievements per extra ~60 steps of night survival. But the longest episodes still never reached stone → survival-limited *first*, progression-limited *second*.
- **Missing instrument:** recorded runs store no unlock-step timestamps → steps-to-milestone and "wasted tail" are uncomputable. Must be logged in the next screening.

## 2. Loophole adjudication (game truth × our system)

| # | Loophole | Engine fact | Our failure mode | Severity |
|---|---|---|---|---|
| L1 | **Maintenance is untargeted** | thirst decays every 21 steps; drink is free/repeatable; sleep mandatory ~2–3× per 1000-step episode (energy/31 steps awake) | drink/eat/sleep are event-free cards with marginal value 0; vital decay invisible at horizon 10; survival is purely reactive (health_delta signal) | **HIGH** |
| L2 | **Night is invisible** | night = steps 148–272, tint/vignette changes pixels, zombies spawn | TTE head unused; no clock readout; planner goes silent at night partly because latents shift and cue-matching degrades; night centroids absent from mostly-day corpora | **HIGH** |
| L3 | **Distant goals unreachable** | stone often >10 steps away; viewport 9×7 tiles | no spatial memory in strict mode (anchors forbidden); nav = 0.05·cos hill-climb; horizon 10; depth-3 chains silently dropped | **HIGH** |
| L4 | **Table is destructible + reusable** | skeleton arrows convert table→path; agent walks away | m_bin locks place_table → card retired as enabler even when the table is gone/far; silent craft failures | MED (fix designed, unimplemented) |
| L5 | **Threshold vs mean semantics** | needs are counts (≥2 wood); readout estimates continuous + noisy | deficit flapping → harvest/craft oscillation; x_hat has no uncertainty | MED |
| L6 | **Action-5 affordance collision** | `do` = attack/collect/drink | missed attacks are visually-matched "near-misses" that deny combat cards; scene-gist cosine pollution | MED |
| L7 | **eat_plant structurally hard** | plants ripen only while agent stays within L1<18 for ~301 steps | no "nurture/wait" concept; place_plant retires after unlock | LOW (1 achievement) |
| L8 | **Deep-tier exploration** | iron chain needs furnace+coal, 5-6 links, zero data | depth-3 makes iron targets ungroundable → silently skipped (protective) — but also blocks legitimate discovery | LOW |
| L9 | **Silent engine failures** | craft/place silently fail (occupied tile, no tool) | system can't distinguish "did nothing" from "impossible"; retry loops possible | LOW |

## 3. Capability map (human × ours) — the 18-capability audit produced this status summary

PRESENT: working memory (belief+GRU), one-time-vs-repeatable economics (one-sided — see L1).
PARTIAL: single-event learning (needs withheld until 3 weighted successes → chains not chainable early), budgeting (no vital-decay accounting), threat model (reactive only), engagement policy (myopic), curriculum (no difficulty estimate), active perception (planner goes silent when no cue on-screen), experimentation (probe accounting but no probe scheduler), failure recovery (deny lowers trust but nothing chooses "try instead"), credit assignment (0.9^depth kills 5-link chains), calibration (readout has no uncertainty).
ABSENT: spatial memory (strict), route planning, maintenance value, day/night anticipation, analogical transfer (stone pickaxe learns from scratch — no relational transfer), temporal abstraction (chains never held long enough to complete).

## 4. Ranked fix backlog (zero-hardcoding mechanisms only)

| # | Fix | Mechanism (all learned) | Why it moves the score | Cost |
|---|---|---|---|---|
| F1 | **Maintenance value / time-to-critical** | Learn per-dim "time until readout dim d crosses its learned low quantile" from the evidence store's own trajectories (a small head on s_t, targets computed from trajectories — dims indexed, never named). MPC adds a maintenance term = predicted time-to-critical during candidate rollouts (reuse the zero-weighted TTE plumbing pattern). Drink/eat/sleep earn value proportional to how soon the imagined path goes critical. | Survival is the score multiplier (+1.4 achievements per extra ~60 steps); kills the dehydration/starvation tail | M |
| F2 | **Temporal abstraction (options)** | Treat each chain link as a temporally-extended option: commit to (action, cue-key) until the card's `postcondition_key` is matched or a learned-duration timeout (`mean_duration` exists, unused); re-plan only at option boundaries. | The +5–8 achievement band (stone/coal/iron tier) requires 4–6-link chains that horizon-10 per-step replanning can never hold | M–L |
| F3 | **Strict-mode spatial memory** | Learned displacement deltas (regress Δz vs action — replaces the hardcoded movement table) + a slot memory: slots written on high-recurrence latents, storing (latent key, integrated displacement); retrieved by cosine as Q/prior conditioning and a navigation term. | Removes the search bottleneck for water/table/stone; enables F2's long approaches | L |
| F4 | **Enabler reactivation** (designed, unimplemented) | Enabler satisfied iff unlocked AND its learned context is visible now; otherwise the enabler card re-enters the chain as an executable step (achievement value stays 0). Covers table-left-behind **and** arrow destruction (L4). | Unblocks every table-gated craft after walking away / station loss | S |
| F5 | **Value propagation through enabling edges** (designed, unimplemented) | Target value += Σ enabled neighbors' value × trust × lift-discount, cycle-safe. Makes "sword → fight → survive → more achievements" numeric (user's zombie scenario). | Correct tech-tree ordering beyond naive cost | S |
| F6 | **Threat cards** | Generalize the card event dimension to *negative* events: cards keyed (action, large-negative health delta) with the pre-damage visual cluster as WHEN; enabling lift learns "kills need tools"; engagement gated by trust + health context rather than myopic Q. | defeat_zombie + defeat_skeleton = 2/22 achievements; 93% of deaths are monsters | M |
| F7 | **Enable the TTE head** | `tte_weight > 0` (flag only — head is trained and plumbed; consider `standardize_scores` so it competes on scale). Cheapest anticipation win: proximity-to-unlock ordering + implicit day/night rhythm. | One-flag change; orders curriculum; begins dusk preparation | XS |
| F8 | **Uncertainty-aware deficits** | `_deficits` margin scales with the readout's uncertainty output (predict_state already returns it) — stops flapping on noisy counts. | Removes harvest/craft oscillation | XS |
| F9 | **Instrumentation: unlock-step logging** | Log (step, achievement) timestamps per episode in the screening loop; enables steps-to-milestone + wasted-tail metrics for every future run. | Missing instrument — measures whether later fixes actually accelerate progression | XS |
| F10 | **Near-miss refinement for overloaded `do`** | Split near-miss denial by outcome signature (attack-miss: no unlock AND no material delta on the *object* channel vs failed collect) using only learned outcome statistics. | Stops missed swings from denying defeat cards (L6) | S |

## 4a. Zero-hardcoding compliance audit of F1–F10 (binding on implementation)

Rules: no engine facts in agent logic (no item/action/event semantics, no recipe amounts); no privileged info at playtime; everything learnable from data the agent legitimately holds (offline engine labels = sanctioned training-time supervision only). Index spaces, statistical hyperparameters, and magnitude scale-priors are allowed but must be registered.

| Fix | Verdict | Notes |
|---|---|---|
| F1 maintenance value | **COMPLIANT with refinement** | Dims indexed, never named; trajectories are the agent's own. ⚠️ Gray zone: the sketched "low quantile of dim d = critical" embeds a *low-means-dangerous* prior on the 16-d state. **Refinement (mandatory):** anchor criticality on *learned* signals — the value v(d) at which the trained health/transition heads start predicting negative health deltas. Then "critical" is a causal fact the agent learned, not a distributional guess. |
| F2 options | COMPLIANT | Commit/timeout uses learned `mean_duration` + a hyperparameter multiplier; matching is cosine to learned postcondition centroids. No semantics. |
| F3 spatial slots | COMPLIANT | Displacement deltas regressed from observed latent changes (replaces the hardcoded movement table — a hardcode *removed*); slot writes gated by learned recurrence; retrieval by cosine. |
| F4 enabler reactivation | COMPLIANT | "Satisfied" = own belief mask + cosine to learned context; re-entering the chain uses the card's learned COSTS. The word "table" never appears in logic. |
| F5 value propagation | COMPLIANT | Pure graph arithmetic over learned trusts/lifts/values; cycle-safe. |
| F6 threat cards | **COMPLIANT with registered scale-prior** | Keying on (action, large-negative health delta) uses a magnitude threshold on the *learned* health channel — same class as the existing 0.35/0.45 priors; no monster names. Threshold registered as tunable. |
| F7 TTE weight | COMPLIANT | Head already trained on sanctioned offline labels; inference consumes latents only. Zero new logic. |
| F8 uncertainty deficits | COMPLIANT | Uses the readout's own learned uncertainty output. |
| F9 unlock-step logging | **COMPLIANT with implementation guard** | During the episode, log **belief-derived** unlocks only (they may feed decisions). Engine `info['achievements']` timestamps may be recorded solely in the post-episode analysis record — the same scoring-only class as today; must never be read by the memory/planner. |
| F10 near-miss splitting | **COMPLIANT with mechanism pin** | The split MUST stay visual/statistical (cluster-conditional failure pools — missed attacks and failed collects live in different visual contexts, so per-cluster statistics separate them). A rule like "if it was an attack, don't count" would require knowing it was an attack = engine semantics = VIOLATION. |

**Registered scale-priors introduced by this plan** (tunable, non-semantic, re-tune if state scale changes): F1 criticality margin (if quantile fallback retained), F6 damage threshold, F2 timeout multiplier, F5 lift-discount.

## 5. Decision table — what the Colab results greenlight

| Observation in the 3-arm screening | Interpretation | Greenlight |
|---|---|---|
| Arm B ≥ Arm A AND survival ≈ Arm A | Memory no longer harmful; regression fixed | F7 + F9 immediately (cheapest), then F1 |
| Arm B < Arm A on survival | Shaping still interfering with survival instincts | F1 first (maintenance value gives survival its own channel), re-tune injection before anything else |
| Bank audit passes AND table→pickaxe conversion still ≈ 20–30% | Cards correct, execution limited | F2 + F3 (chains can't be held/reached) |
| Arm B ≈ Arm C on pickaxe/sword | Online loop not earning its keep yet | F9 first (instrument), then F1/F2 — hypotheses need more episode exposure before judging |
| place_table card correct but wood→table conversion ≈ 50–60% | Harvest/navigation bottleneck | F3, then F4 |
| defeat_zombie rate unchanged | Combat untouched by memory | F6 + F5 |
| Event-head recall still 0 on table/pickaxe in the v2 run | Supervision/labels fine but head undertrained on rare events | Escalate: consider oversampling rare-unlock frames in head training (training-time only, zero inference change) |
| Night deaths still dominate after F1 | Anticipation missing | F7 tuning, then F2 (shelter preparation emerges from longer planning) |

## 6. Sequencing principle

Cheapest-and-safest first (F7, F9, F8, F4, F5 — all S/XS), then survival (F1), then reach (F2, F3), then combat (F6, F10). Never more than one behavioral mechanism per screening cycle; every cycle = 3-arm screen + bank audit + card read.

## 7. Explicitly deferred (known, consciously not planned now)

- Analogical transfer across tool tiers (relational cards) — high value, high complexity; revisit after F2.
- eat_plant nurture cycles (L7) — structurally hard, 1 achievement.
- Deep-tier (iron/diamond) curriculum — blocked on F2/F3 anyway.
-梦境-style re-training of the event heads with oversampled rare events — training-time only, but changes the locked checkpoint lineage; decide after the v2 screening.
