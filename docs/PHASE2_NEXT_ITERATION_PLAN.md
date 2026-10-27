# Phase 2 — Next Iteration Plan (GATED on Colab screening results)

> **Status: DO NOT IMPLEMENT before reading the 3-arm screening results** (Arm A ablation-C on v2 checkpoint / Arm B bank+online / Arm C bank frozen). This plan was produced by a four-stream study (engine mechanics reference, human-learning capability map, adversarial game×implementation analysis, empirical run mining) and the Colab results decide which fixes get greenlit and in what order.
>
> Companion docs: `CRAFTER_MECHANICS_REFERENCE.md` (ground truth), `PHASE2_MEMORY_REDESIGN_PLAN.md` (what is already implemented).

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
