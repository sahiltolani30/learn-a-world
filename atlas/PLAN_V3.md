# Project ATLAS v3 — PRESCRIPTIVE SPEC: Value Learning + the Two-Wood Milestone

**Audience:** implementing agent. Follow literally. No redesign, no substitution. If a mechanism
is not specified to the formula level in this document, implement it as DISABLED and report the
gap — never fill it from world knowledge (F11-incident meta-rule, now standing law).
**Allowed files:** `atlas/config.py`, `atlas/critic.py` (NEW), `atlas/skills.py`, `atlas/goals.py`,
`atlas/train.py`, `atlas/arbiter.py`, `atlas/selflearn.py`, `atlas/health.py`,
`atlas/audit_pack.py`, `atlas/screen.py`, `atlas/colab/run_atlas.py`, `atlas/tests/*`.
Nothing else. All standing prohibitions apply (frozen backbone, no old-data bootstrap, no
name/index/recipe logic anywhere in decisions, locked seeds, learned inputs only).

**Basis (all measured, from the 443MB it118 archive):**
- W1. **Inventory ceiling = 1 item.** Raw shard targets: max log1p inventory ≈ 0.69 (= 1 item),
  item-dim mean 0.006–0.013. `place_table` costs 2 wood (log1p ≈ 1.10) — a state that has
  never existed in his 1,290-episode life. This is the pixel-level root cause of the
  wood(25% of eps) → table(2.5%) → pickaxe(0) collapse.
- W2. Skill CE pinned at ~1.70 for 80 iterations; behavior mix stationary; score flat 1.47% —
  imitation has extracted all it can. No value signal exists anywhere in learning.
- W3. Working and preserved: loop (344+ attributed unlocks, arrivals ~10/iter), rotation
  (32% frontier), concept clouds (100% cross-world for common items), affordances correct at
  every mined context (15/15 patterns), causal memory (21 edges, 0 violations), survival
  (170–196), audit pack, resume integrity, clean build.
- W4. Instrumentation debts: `positions_last2.npz` dummy, unlock attribution inflated ~6×
  (7,264 vs ~834), flatline alarm blind (85 false A3s), health-card trend stuck at 0.0,
  learned material gate never fired (0 records) — present but inert.

---

## Part B — THE FIVE MECHANISMS (in implementation order)

### F1. `CriticNet` — learned long-horizon value (`atlas/critic.py`, NEW)
```python
class CriticNet(nn.Module):
    # trunk: MLP(384 -> 256 -> 256) LayerNorm+SiLU; head: 65-bin two-hot over symlog returns
    # bins span [-5, 10] on symlog(1 + discounted return); EMA target copy (decay 0.995)
    def forward(s) -> logits[65]
    def value(s) -> scalar expectation
def critic_loss(online_logits, target_twohot) -> cross_entropy   # standard two-hot CE
```
Training targets (from replay only, computed in `train.py`):
- Per transition t: `G_t = Σ_{k=0..H} γ^k r_{t+k}` with γ=0.97, H=25 (episode-truncated;
  no bootstrapping in v3.0 — pure Monte-Carlo on own returns, immune to critic-collapse
  feedback loops while the policy is still weak).
- Normalize: symlog scale `v = sign(G)*log(1+|G|)`, then min-max into bin range per batch
  (track running bin edges like ReturnNormalizer; save in state.pt).
Two-hot encode (reuse the pattern in `models/heads.py::two_hot_encode` — reimplement locally
in `atlas/critic.py`, do NOT import models/).
Optimizer: Adam lr 3e-4 (same family as all heads; do not choose another), grad-clip 1.0.
Train 8 epochs/iteration on the same cached (s, r) transitions as the other heads; log
`critic_ce`, `critic_value_mean`, `critic_value_p90` to train_log. Critic optimizer state is
NOT persisted across segments (fresh each resume — Monte-Carlo targets make this harmless).
EMA target update: after EVERY optimizer step (not once per iteration). The running bin-edge
normalizer is a LOCAL reimplementation of the percentile-EMA pattern (5th/95th, decay 0.99)
inside `atlas/critic.py` — do NOT import `models/heads.py`. Its state (low, high, initialized)
is saved inside the critic entry of state.pt and restored on resume.

### F2. Actor-critic shaping of the skill policy (`skills.py`, `train.py`)
Replace pure imitation with **advantage-weighted imitation + value bootstrap blending**:
```python
# per HER positive pair (segment t..t+k, action a_tau):
A_seg = (seg_return_discounted - V(s_t))          # seg_return_discounted uses γ=0.97 (same as F1); V = critic.value at the segment START state
w_adv = exp(clip(A_seg / 3.0, -2.0, 2.0))          # in [e^-2, e^2] ≈ [0.14, 7.4]
w_total = (existing freq × reward × unlock × progress weights) × w_adv
# renormalize mean 1.0, clip 50 (unchanged composition path)
```
This is AWR (Advantage-Weighted Regression) — the safest known bridge from imitation to RL:
actions from segments that beat the critic's expectation get up-weighted; below-expectation
segments get suppressed. No gradient flows into the backbone; the critic is trained only on
realized returns (F1), never on its own bootstrap.
Verification: unit test — with a stub critic, a segment with return +3 vs V=0 gets weight
> 1; return −3 gets < 1; weights renormalize to mean 1.

### F3. Goal value integration — the planner finally optimizes payoff (`goals.py` scoring,
`arbiter.py`)
`select()` scoring adds a learned term and replaces nothing:
```python
score(g) += w_value * V(s) · ψ(s, g)          # w_value = 0.5 (config)
```
PLUS the **quantity-aware subgoal emission** (the direct W1 attack). Trigger condition —
fully learned, NO tier names or tier index lists: for the selected goal's event dim e, let
a* = argmax_a event_probs[a, e] (the affordance action). Fire the material path iff BOTH:
(i) the learned cost vector cost[i] = max(0, -StateDeltaHead(s, a*)[i]) is nonzero for at
least one item dim i in 0..11, AND (ii) the lift graph has ANY predecessor of e with
ci_low > 1.2. Then the planner emits a MATERIAL subgoal before it:
```python
# learned cost: cost[i] = max(0, -StateDeltaHead(s, a*))[i]  for the goal's top-probe action a*
# learned have:  have[i] = Readout(s)[i]  (log1p scale)
# deficit dims:  cost[i] > have[i] + 0.2
# if any deficit: pick the SINGLE largest deficit dim d* = argmax_i (cost[i] - have[i]);
#   find the goal CLOUD = all bank entries whose event_dims contain ItemEventMap^{-1}(d*)
#   (the event dim mined-linked to item dim d*); select within that cloud using the normal
#   select() scoring. expected_cycles = ceil(cost[d*] - have[d*]) (config max_cycles 4).
#   ONE deficit dim per planning cycle; after each completed material cycle, re-plan.
```
The repeat counter: after each arrival+probe on the material goal, decrement remaining
cycles; re-emit until have ≥ cost, then release the original target goal. All quantities
learned (delta head, readout, mined graph). If the lift graph lacks the prerequisite edge,
NO substitution happens (goal pursued as today — honesty over cleverness).

**ItemEventMap — the learned bridge from inventory dims to event dims (MANDATORY, no
exceptions):** the planner must find "the goal cloud for the wood ITEM" but the bank is keyed
by EVENT dims. NEVER link them by name, index convention, or hand-written map (F11-incident
class). Mine the link from training data only, once per iteration in `selflearn.py`:
```python
# over all cached episodes, for each inventory item-dim i (0..11) and event-dim e (0..21):
#   inc[i][t] = inventory[t+1][i] - inventory[t][i] > +0.3   (a real gain of item i)
#   ev[e][t]  = events[t][e]
#   support(i, e) = count of steps where ev[e][t] AND inc[i][t]
#   ItemEventMap[i] = argmax_e support(i, e)  iff support(i, argmax) >= 5, else None (no link)
# save ItemEventMap in state.pt AND restore it in load_state; a dim with no confident
#   link emits NO material subgoal.
```
The material subgoal in F3 uses `ItemEventMap[deficit_dim]` to pick the goal cloud. If the
map has no link for that dim: no substitution, goal pursued as today.
Verification: stub test — goal with learned cost 2 wood, have 0 → selects wood-cloud goal
with cycles=2; after 1 simulated collection (have=1) → cycles=1; have≥cost → original goal.
Test that no substitution occurs when no ci>1.2 edge exists.

### F4. Confidence-gated exploration focus (`epistemic.py` usage in `arbiter.py`)
The measured blind-probe rate is 31%. Change: in PROBE, when
`max(event_probs) < 0.05` (blind context), weight the curiosity term by the state's epistemic
disagreement (already computed) — i.e. keep existing formula but multiply `probe_entropy`
term by `1 + disagreement_all_actions(s).mean()` (config `blind_probe_boost: 1.0`).
Blind probes remain allowed (honest exploration) but prefer the MOST uncertain actions.
Verification: stub test — zero event probs + high disagreement flips ranking toward the
max-disagreement action.

### F6. Stagnation repair — end the 8-step abortion epidemic (implementing agent's Issue 1)
Measured: 87.9% of 7,013 navigations abort into WANDER; 36.6% of ALL steps are wandering;
arrival rate 12.1%. Cause: `stagnation_patience = 8` with ψ-only progress — obstacle
detours legitimately take 5–8 flat-ψ steps. Exact changes (`config.py`, `arbiter.py`):
- a. `stagnation_patience: 8 → 24`; `wander_steps: 10 → 6`.
- b. Progress signal becomes `progress_t = max(ψ(s_t, g), cos(z_t, g))` — track the running
  best of the MAX, not ψ alone (cos-to-goal rises even when ψ misestimates during detours).
- c. Hysteresis: abort only if `best_progress` has not improved by ≥ 0.01 over the window AND
  `cos(z_t, g) < arrival_cosine` (never abort while visually closing in).
- d. Telemetry: log `aborts_by_reason {stagnation, timeout}` per iteration; Part E adds
  arrival-rate gate (see M1a below).
Verification: stub test — 12 flat-ψ steps with rising cos → no abort; 24 steps flat on both
→ abort; wander block exactly 6 steps.

### F7. Vital-goal mining repair — the impossible threshold (Issue 4)
Measured: 0 vital goals selected ever; threshold `vital_spike = 0.15` is mathematically
unreachable (engine regen is +0.038/step). Replace single-step prediction-spike mining with
**windowed realized-vital mining** in `selflearn.py`:
- For each episode, use the inventory targets' vitals dims (12..15, training-side data):
  mine a vital goal at step t when THERE EXISTS a Δ in 1..30 such that
  `vitals[t+Δ] − vitals[t] ≥ 1.0` unit for any vital dim (a drink, a meal, a rest — real
  restorations, not noise), storing the view at t as kind "vital". Keep
  `vital_target_share: 0.15` bank quota (already reserved; today it holds 1/256).
- At play time: if ANY readout vital dim (12..15) estimates below 0.35 (config
  `vital_select_below: 0.35`) AND at least one vital goal exists in the bank, the goal
  draw is REPLACED by the highest-scored vital goal — this check runs BEFORE the rotation
  draw (homeostasis outranks rotation). Otherwise the normal draw runs. Learned thresholds,
  no names.
Verification: synthetic episode with a +1 drink event at t=40 mining a vital goal at the
water view; no goal mined for +0.2 drift; low-vitals stub raises vital-goal score weight.

### F8. Craft-execution step — resolve the F10 paradox (Issue 6)
The analyst is right that F10 blocks deliberate table-building in fresh lives (table views
can't exist before one is built). Resolution — building is an ACTION, not a place:
- Extend F3's planner: when the target goal's event dim e passes F3's learned trigger
  (nonzero cost vector), is NOT yet built this life (`belief.has(e)` is False), materials
  are satisfied (have[i] ≥ cost[i] + 0.2 margin on every item dim with nonzero cost), and
  the affordance action a* = argmax_a event_probs[a, e] has event_probs[a*, e] ≥
  `probe_affordance_min = 0.05` at the CURRENT context: emit **EXECUTE-CRAFT**
  — enter PROBE immediately at the current location with the material gate active (no
  navigation). `place_*`-tier actions need only empty ground (the event head's context
  estimate decides — no terrain rules).
- After a successful place-tier unlock, the built artifact's view is episode-tagged (existing
  F11 registry) — subsequent make-tier goals route to that tagged view (F10-compliant).
- F10 gating itself stays: view-goals for unbuilt artifacts are never navigation targets.
Verification: stub test — table goal, have≥cost, affordance 0.4 → PROBE chosen without
navigation; affordance 0.01 → material subgoal path as in F3; post-place, tagged-view
routing works (existing test) — add the EXECUTE-CRAFT transition test.

### F9. (Unchanged scope note: Issues 2 & 5 are Phase-C per the ladder)
Nightfall mortality (66% die at steps 140–200) and the stone void (0 stone in 1,290 lives)
are NOT addressed by new mechanisms in v3: both depend on first fixing Issues 1/3/4/6
(reach, materials, vitals, craft execution) — a sword and stone both require the
wood→table→tool chain to fire deliberately. Survival economics (sleep policy, night
evasion beyond ContinueNet) remain Phase C. Do NOT bolt them on here (one variable per
segment); the M-gates will show whether the chain fix alone moves survival.

### F5. Material-gate activation + provenance (closes W4's inert gate)
Diagnose then fix: `material_gating_learned` requires `backbone.has_learned_delta_head`.
The it118 checkpoint loads `state_delta_head` (audit line says Loaded 14 — includes it), so
the gate should have fired. Startup behavior in `run_atlas.py`: if
`backbone.has_learned_delta_head` is True → set gating ACTIVE, log `material_gate_active:
true`. If False (e.g. the local `jepa_crafter_v3_phase2.pt` smoke checkpoint, which LACKS
this head — verified in its load audit) → DO NOT crash: disable gating, print a loud
warning, log `material_gate_active: false`; M0 then honestly cannot pass on that
checkpoint and the screen JSON says so. Log `material_gate_active` in every screen JSON. If the gate fires but
suppresses nothing, log per-probe `suppressed_count` (already in the decision record) —
the audit must show nonzero suppressions at wood-less table contexts within 5 iterations.

---

## Part C — INSTRUMENTATION REPAIRS (do before the run; none touch behavior)

- C1. `positions_last2.npz` dummy: write real per-step positions from the collector (same
  arrays as the journal; the audit writer currently writes `dummy` — fix the call).
- C2. Unlock attribution: `unlocked_dims` per decision record = belief newly-locked dims
  **during that probe only** (clear at probe start); today it accumulates per goal-cycle (~6×
  inflation). Add `probe_id` to records so the audit can group correctly.
- C3. Flatline alarm: initialize monitor from the latest dev-screen ach/ep when loading a
  state whose `health_snapshot.best_ach_per_ep == 0.0` but `iter > 10`.
- C4. Health-card trend: compute vs the mean of iters 10–15 baseline (fixed window), report
  raw; the current implementation divides by a zero baseline (produces 0.0).

---

## Part D — IMPLEMENTATION ORDER
F5 → F6 → F7 → C1–C4 → boundary 10-iter segment → F1 → F2 → F3 → F8 → F4 → v3 60-iter segment.
(F8 is implemented AFTER F3: EXECUTE-CRAFT consumes F3's learned have/cost machinery — building
F8 standalone would force improvisation. F1/F2 change training dynamics and must start a fresh
segment; the boundary segment therefore contains F5/F6/F7/C1–C4 only.)

## Part D2 — RUN PROTOCOL

```bash
# boundary segment (repairs only, no new mechanisms): 10 iterations, verify:
python -m atlas.colab.run_atlas --phase selflearn --iters 10 --run-id atlas_v2_scratch --minutes 90
# check: gate active (F5), positions real (C1), attribution sane (C2), no false A3 (C3)
# then the v3 segment:
python -m atlas.colab.run_atlas --phase selflearn --iters 60 --run-id atlas_v2_scratch --minutes 300
```
Continue the SAME run-id (748k transitions of honest experience are the critic's training
data). 30-iteration checkpoints as standing cadence.

## Part E — MILESTONE GATES (the 2-wood ladder; report raw, never tune to them)

| Gate | Metric (source) | Pass |
|---|---|---|
| M0 (it+10) | material gate suppressions at wood-less table contexts | > 0 |
| M1a (it+10) | navigation abort rate + arrivals/iter | aborts < 50% (was 87.9%); arrivals ≥ 15/iter (was ~10) |
| M1 (it+20) | episodes whose max log1p value on the wood item dim ≥ 1.10, measured from **shard targets** (training-side ground truth, NOT the readout); item-dim order is the existing `INVENTORY_ITEMS` extraction order in `collect.py` — evaluation/reporting use only, never a decision input | ≥ 5% of episodes — **the first time in his life he holds 2 wood** |
| M2 (it+30) | place_table episodes | ≥ 10% (was 2.5%) |
| M3 (it+40) | make_wood_pickaxe episodes | ≥ 1 (first deliberate tool) |
| M4 (it+60) | score / ach-per-ep | rising trend vs it118 (≥ +20%); critic_value_p90 increasing |
| Ablation (it+60) | A-critic: rerun 10 iters with `w_value=0, w_adv=1` | report both arms' M1–M3 side by side |

**Stopping rule (the pre-registered escalation, unchanged):** if M1 fails at it+20 — i.e.,
the value-weighted policy still never holds 2 wood — report plainly; the next rung is
extending H beyond 25 / adding n-step bootstrap, NOT more imitation tuning.

## Part F — ANTI-ASSUMPTION CHECKLIST (read before implementing each fix)
Before starting each F-item, answer in one comment at the top of the change:
1. Which EXACT formula in this plan am I implementing? (quote the line)
2. Which of my inputs are learned/mined vs. constants? If any constant encodes world
   knowledge (item names, action identities, costs, terrain rules, dim mappings) — STOP,
   implement that part as DISABLED, and report the gap. This includes: item→event links
   (use ItemEventMap only), costs (StateDeltaHead only), holdings (readout only),
   prerequisites (lift graph only), "what wood is" (nothing — there is no such concept in code).
3. Is there a verification test? If the plan's test cannot pass without the mechanism being
   learned (not constant), write it that way.
Known assume-invitations already closed in this revision: critic optimizer (Adam 3e-4);
EMA update timing (every optimizer step); bin normalizer (local reimplementation, persisted);
item-dim→event-dim mapping (ItemEventMap, mined, persisted on resume); tier vocabulary
(REMOVED — trigger is learned cost vector + lift predecessor, no tier lists); deficit
handling (single largest deficit, one per cycle, re-plan); vital mining window (exists Δ in
1..30); vital selection precedence (replaces the draw, before rotation); behavior on
checkpoints lacking the delta head (graceful disable + loud warning, NO crash); M1
measurement source (shard targets, existing extraction order); F8's dependency on F3
(ordering fixed). If you find another ambiguity, it is a bug in this plan — report it, do
not resolve it yourself.

## Part G — DEFINITION OF DONE
1. All existing + new tests green (`pytest atlas/tests -q`); smoke clean with `prior_weight=0`.
2. No-cheat sweep: zero item/achievement/action names in `atlas/` decision paths
   (grep test in suite); `CRAFT_REQUIRED_MATERIALS` structural test still green.
3. `git diff` limited to allowed files; commit message references this plan.
4. Handback: verification outputs, M0–M4 raw values at each checkpoint, and the exact
   commands run.
