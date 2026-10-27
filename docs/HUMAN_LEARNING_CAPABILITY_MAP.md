# How a Human Learns Crafter — Capability Map vs Our Architecture

> Research stream 2 of the next-iteration study (see PHASE2_NEXT_ITERATION_PLAN.md). The full analysis behind the plan's "capability map" summary. Game facts cited from CRAFTER_MECHANICS_REFERENCE.md; architecture claims cited file:line.

## Part A — How a real human learns (games 1–10)

**Game 1** (dies ~60 steps, 0–1 achievements). Learns the movement grammar and watches the HUD vitals tick down. Dies of thirst next to water she never drank from — the single most important signal of the session: "that blue tile was drinkable." One-trial causal learning from a lethal outcome.

**Game 2.** Drinks when thirsty (one-trial transfer), experiments near objects. Discovers "face tree + chop → wood" by accident. Sees a zombie at night, health drops near it — a *negative* prototype ("monster + proximity → damage") formed from one encounter.

**Game 3.** First deliberate budget: chop 2–3 wood, experiments with crafting, makes a table. The +1 and the new inventory icon are the signal. She now **remembers where the table is and returns to it** — object permanence.

**Game 4.** Menu exploration at the table yields the wood pickaxe; never re-derived — a memorized procedure keyed to "standing at table with wood." Starts timing nightfall: at dusk she stops ranging and hugs shelter — anticipation from the 300-step rhythm, learned as a *rhythm*, not a number.

**Games 5–6.** Finds stone by visual search; recalls "pickaxe unlocks actions on rock" *by analogy* with the tree — same relational shape, different filler. Dies to skeletons a few times; learns line-of-sight and "fight only with sword and full health" — risk assessment with an internal precondition, not a reflex.

**Games 7–8.** Routines crystallize: drink whenever passing water (free, repeatable — never extinguishes), eat at half food, sleep when energy low. Self-generated curriculum: "stone sword before dark," then coal, then iron. Failure recovery: pickaxe did nothing on iron? Try a different rock — experiment design, not repetition.

**Games 9–10.** Route planning over a memorized map (table↔stone↔water triangle), tool-tier analogy (stone pickaxe should work like wood pickaxe), deliberate night tactics (wait out the night, then hunt zombies *for* the achievement, with a sword). Score climbs toward 8–12/22; expert play (~22) is mostly logistics and risk.

## Part B — 18 capabilities mapped to our architecture

| # | Capability | Status | Where / why | Behavioral consequence |
|---|---|---|---|---|
| 1 | Single-event causal learning | PARTIAL | Online card creation on first confident event (`causal_memory.py:1278-1281`) but needs withheld until 3 weighted successes (`:73`), and needs are never learned online (`:1144,1193`) | First unlock → need-less hypothesis card; the wood→table→pickaxe sequence needs many episodes before it becomes chainable |
| 2 | Object permanence / spatial memory | ABSENT (strict) | Anchor machinery exists (`semantic_memory.py:144`) but setters need semantic_memory (`mpc_v3.py:239-248`); strict landmark branch aborts (`:516-521`) | Forgets where water/table/stone was the moment it scrolls off-screen; dies of thirst near water it drank from |
| 3 | Route planning | ABSENT | No spatial representation; dead-reckoning feeds only the privileged path; strict "navigation" = retinal-similarity gradient (`mpc_v3.py:466-469`) | Visual hill-climbing sticks to walls/water; stone 30+ steps away is beyond horizon 10 — circles locally |
| 4 | Resource budgeting | PARTIAL | produces/consumes deltas (`causal_memory.py:738-741`); chain applications capped 10 (`compositional_planner.py:168-174`); deficits re-checked per step | Plan-time budget on a noisy readout; no vital-decay accounting mid-chain — a correct materials plan can still die in execution |
| 5 | Maintenance routines | ABSENT as value | Vitals only as tiny per-step health_delta (`mpc_v3.py:438-441`); drink is event-free with value 0 (`causal_memory.py:458-460`) | Drinking has literally zero pull; subgoal shaping dominates; agent dehydrates mid-craft — the recurring killer |
| 6 | Day/night anticipation | ABSENT | TimeToEventHead exists (`heads.py:608-648`) but `tte_weight=0.0` (`mpc_v3.py:45`) | No prepare-before-dusk behavior; night encounters are surprises within horizon 10 |
| 7 | Threat model | PARTIAL | Reactive only: lethal-damage shaping, done-prob penalty; near-misses only deny same-action cards | Avoids visible damage but stores nothing about monsters — re-approaches the same skeleton next episode |
| 8 | Risk-conditioned engagement | PARTIAL | Needs-enrichment could learn sword-prereqs; enabling_lift captures ordering; engagement itself is myopic Q | At 3 HP may still swing; uncertainty penalty can suppress combat entirely — kill achievements never fire |
| 9 | Curriculum self-generation | PARTIAL | UCB over attempts, marginal gating, hypothesis floor | No difficulty estimate — can't pick "stone sword next because it's nearest"; deep tiers unreachable at depth 3 |
| 10 | Active perception | PARTIAL | Novelty bonus ‖z−z₀‖·0.02; READY gate needs a visible cue | When no cue is on-screen the planner goes silent — discovery is a random walk, never a deliberate sweep |
| 11 | Experimentation | PARTIAL | `register_interaction_outcome` + UCB | Probe *accounting* exists, no probe *scheduler* — hypotheses clear only by luck |
| 12 | Analogical transfer | ABSENT | Cards keyed (action, event), cosine 0.85; no relational similarity across events | Each tool tier learned from scratch — the human's biggest accelerator is our biggest blind spot |
| 13 | Failure recovery | PARTIAL | Per-step replanning; deny lowers trust | Denial doesn't choose what to try *instead* — CEM can refit to the same futile action |
| 14 | Working memory | PRESENT | Belief masks + GRU h (`agent_v3.py:136-194`) | Functions, but only the 22-d mask is plannable; night/occlusion context in h is opaque to every planner term |
| 15 | One-time vs repeatable economics | PRESENT (one-sided) | Marginal gating (`causal_memory.py:449-469`, `compositional_planner.py:87-94`) | Correctly prevents farming — but assigns repeatable-vital actions zero value (row 5): the ratchet has no inverse |
| 16 | Credit assignment along chains | PARTIAL | Chain trust product + 0.9^depth decay | 5-link chains (wood→table→pickaxe→stone→stone-pickaxe) are beyond depth 3 and silently dropped |
| 17 | Belief calibration | PARTIAL | Wilson LB, hypothesis status, quarantine; but the 16-d readout has no uncertainty | A misread wood count silently corrupts every chain |
| 18 | Temporal abstraction | ABSENT | Chains planned but executed one primitive at a time; `mean_duration` unused; only soft persistence | Horizon 10 ≈ one chain link; the stone/iron band (+5–8 achievements) is effectively out of reach |

## Part C — Top-5 missing capabilities by score impact (current ~4.3% ≈ 1/22; DreamerV3 35% ≈ 7–8/22)

1. **Maintenance value** — survival is the score multiplier (recorded: reward↔length r≈0.42; ~+1.4 achievements per extra ~60 steps). Mechanism: learned per-dim time-to-critical head (targets from the evidence store's own trajectories, dims indexed never named); MPC maintenance term during rollouts, reusing the zero-weighted TTE plumbing pattern.
2. **Temporal abstraction** — the stone/iron band requires 4–6-link chains. Mechanism: chain links as temporally-extended options — commit to (action, cue-key) until the card's `postcondition_key` matches or a learned-duration timeout; re-plan at option boundaries only.
3. **Strict-mode spatial memory** — water/table/stone hubs; converts multi-minute searches into short returns. Mechanism: learned displacement deltas (regress Δz vs action) + slot memory storing (latent key, integrated displacement), retrieved by cosine for Q/prior conditioning and navigation.
4. **Threat model + engagement policy** — defeat_zombie + defeat_skeleton = 2/22 and night survival protects everything else. Mechanism: negative-event cards keyed (action, large-negative health delta) with pre-damage visual clusters as WHEN; enabling lift learns "kills need tools"; engagement planner-gated.
5. **Anticipation via TTE** — cheapest win: the head is trained and plumbed; `tte_weight > 0` orders candidates by proximity-to-unlock and, because night risk correlates with the cycle, begins dusk preparation. Optionally `standardize_scores` so it competes on scale.

Compounding order: #5 and #1 extend episodes cheaply → #2 makes long chains executable → #3 removes the search bottleneck those chains hit → #4 banks the two kill achievements the longer nights make approachable.
