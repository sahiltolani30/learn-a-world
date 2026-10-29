# Project ATLAS v2 — PRESCRIPTIVE Fix Spec (implement exactly as written)

**Audience:** implementing agent. Follow this document literally. Do not redesign, do not
"improve" beyond it, do not skip verification steps. Every place where you might be tempted to
make a judgment call, the decision is already made here. If something is genuinely impossible as
specified, STOP and report back instead of substituting your own approach.

**Allowed files to modify:** `atlas/config.py`, `atlas/backbone.py`, `atlas/skills.py`,
`atlas/train.py`, `atlas/arbiter.py`, `atlas/goals.py`, `atlas/collect.py`, `atlas/screen.py`,
`atlas/selflearn.py`, `atlas/colab/run_atlas.py`, `atlas/smoke.py`, `atlas/tests/*`, and one NEW
file `atlas/health.py`. **Everything else in the repo is untouchable.**

**Never do (hard prohibitions):**
1. Never train, fine-tune, or modify the frozen backbone (`encoder`, `target_encoder`,
   `recurrent_core`, `predictor`, `event_head`, `health_head`, `q_ensemble`, `policy_prior`).
2. Never read Drive replay shards, bootstrap data, or any old training data. The bootstrap phase
   is DELETED, not disabled.
3. Never use achievement names, action names, or item names in decision logic. They exist only in
   display strings (`reason`, journal) and training-target extraction (`collect.py`).
4. Never let the v1 artifacts (`archive_v1_turret/`, old goal banks, old state.pt) be loaded.
5. Never read `docs/CRAFTER_MECHANICS_REFERENCE.md` from code. It is for human verification only.
6. Never change evaluation seeds (2050–2099 locked; 6150–6159 dev).
7. Never rebalance by editing environment rewards, actions, or env behavior — only training
   weights and exploration probabilities may change.
8. Never report a screen result without the provenance fields in §9.

---

## Part A — CONTEXT (why; read once)

v1 (106 iterations) collapsed: the goal-conditioned skill policy (OptionNet) outputs `do` for
98.7% of queries at 94.5% confidence; replayed episodes show ZERO movement, 1 unique tile, 100%
`do`; achievements/episode flatlined at 2.3 for 60 iterations; the goal bank flooded with one
event type (225/256 sapling views); the predecessor-lift matrix produced impossible statistics
(ci_low > lift) from a probability floor bug; reachability AUROC 0.99 was a trivially-easy test
(cross-episode negatives); the behavior-cloned prior from the old checkpoint seeded `do`-heavy
early behavior which HER then amplified. Full evidence in the post-mortem; you don't need it —
this document is self-contained.

Root cause chain to keep in mind while coding:
`majority-action imitation echo (HER) → policy collapses to one action → agent stops moving →
memory floods with the one behavior's unlocks → goals always suggest the same views → repeat`,
with `exploration wired only into a mode that never ran (FALLBACK 0%)` and `prior from old data`
as the initial seed, all hidden by gates that tested the wrong thing.

---

## Part B — CONFIG CHANGES (`atlas/config.py`)

Edit the dataclasses exactly like this (add fields; change defaults; keep everything else):

```python
@dataclass
class BackboneConfig:
    checkpoint: str = field(default_factory=_default_checkpoint)
    device: str = "auto"
    frozen: bool = True
    prior_weight: float = 0.0        # NEW: 0.0 = old behavior-brain has ZERO influence

@dataclass
class SkillsConfig:
    # existing fields unchanged, EXCEPT:
    arrival_cosine: float = 0.85     # CHANGED from 0.93 (0.93 was unreachable; cos plateaus ~0.92)
    # NEW fields:
    wander_steps: int = 10           # forced curiosity walk after option failure
    label_smoothing: float = 0.05    # policy CE label smoothing
    max_action_share: float = 0.25   # not used for subsampling; kept for reference/tests
    weight_clip: float = 50.0        # max per-sample imitation weight

@dataclass
class CollectConfig:
    # existing fields unchanged, EXCEPT:
    epsilon_decay_iters: int = 100   # CHANGED from 40
    epsilon_end: float = 0.10        # CHANGED from 0.05 (floor stays high)
    beta_end: float = 0.30           # CHANGED from 0.15

@dataclass
class GoalsConfig:
    # existing fields unchanged, PLUS:
    max_goals_per_dim: int = 15      # cap per achievement-dimension among "event" goals
    same_kind_dedupe: float = 0.90   # cos threshold to reject near-duplicate same-kind goals
    frontier_min_share: float = 0.20 # bank must keep >= 20% frontier goals
    lift_cap: float = 10.0
    lift_min_base_count: int = 20    # min firings of d to mine edge (d,e)
    lift_min_pair_support: int = 3   # min co-occurrence count k
    lift_ci_threshold: float = 1.2   # ci_low must exceed this to influence scoring

@dataclass
class HealthConfig:
    window_steps: int = 200
    action_share_alarm: float = 0.60
    confidence_alarm: float = 0.85
    flatline_iters: int = 15
    flatline_improvement: float = 0.10
    epsilon_boost: float = 2.0
    boost_steps: int = 100

# AtlasConfig: add  health: HealthConfig = field(default_factory=HealthConfig)
```

---

## Part C — CODE CHANGES, FILE BY FILE (do in this order; run the verification after each)

### Step 1 — `atlas/goals.py`: lift repair + bank caps

**1a. `PredecessorLift.mine` — replace the statistics with exactly this logic:**
```python
# per pair (d, e), d != e, accumulated over episodes as in v1:
#   count_d  = number of steps where event d fires (summed over all episodes)
#   k        = number of e-firings where d fired within `window` steps before
#   n_e      = number of e-firings
#   total_steps = sum of episode lengths
p_d = count_d / total_steps                      # NO floor, no 1e-4 anywhere
if count_d < cfg_lift_min_base_count or k < cfg_lift_min_pair_support or n_e < 1:
    lift[d, e] = 1.0;  ci_low[d, e] = 0.0        # NO EVIDENCE -> edge inactive
else:
    p_cond = k / n_e
    lift[d, e] = float(min(lift_cap, p_cond / p_d))
    # Wilson 95% lower bound on p_cond (z=1.96), then divide by p_d:
    #   denom = 1 + z^2/n_e ; center = (p_cond + z^2/(2 n_e)) / denom
    #   margin = z * sqrt(p_cond*(1-p_cond)/n_e + z^2/(4 n_e^2)) / denom
    #   ci_low[d, e] = max(0.0, (center - margin) / p_d)
    if ci_low[d, e] > lift[d, e] * 1.01 + 1e-6:
        raise RuntimeError("PredecessorLift inconsistent: ci_low > lift — this is a bug, fix it")
```
Add method `assert_consistent()` that checks `ci_low <= lift * 1.01 + 1e-6` elementwise and
raises; call it at the end of `mine()` and in `selflearn.mine_goals_and_lift()`.

**1b. `GoalBank.add` — replace dedupe/evict with:**
- Compute `entry.kind` and `entry.event_dims` (normalize `entry.key` to unit norm FIRST:
  `key = key / max(norm, 1e-8)`; also do this in `GoalEntry.__post_init__`).
- Reject (return None, count in a public `self.rejected_same_kind` counter) if any existing goal
  of the SAME kind has cosine > `same_kind_dedupe` (0.90) with the new key.
- Count per-dim usage: maintain `self._dim_counts: dict[int, int]` over "event" goals. If adding
  this entry would push any dim in `event_dims` above `max_goals_per_dim` (15), reject it.
- Enforce frontier quota: if `kind == "event"` and current frontier share < 0.20 and bank size >
  50, reject event goals until frontier share restored (frontier goals always accepted).
- Eviction when full: evict the entry with lowest `evidence = (n_succ / max(1,n_try)) * (1/(1+0.1*age))`
  among "event"/"vital" kinds; NEVER evict frontier goals below the 20% quota.
- `select()` scoring change: replace the lift term with
  `pred_bonus = 0.25 * min(sum(ci_low[d, e] for d in locked_dims if ci_low[d, e] > 1.2), 4.0)`
  (uses **ci_low**, only edges above 1.2). Where `e` = the best locked dim of this goal's profile.

**Verification 1:** `pytest atlas/tests/test_atlas_memory.py -q` after updating tests:
- rare-event pattern (d fires once, base tiny) → NO edge above 1.2;
- strong common pattern → lift > 2 AND ci_low <= lift;
- inserting 100 same-dim event goals → bank holds ≤ 15 of them;
- all stored keys have norm 1.0 ± 1e-6.

### Step 2 — `atlas/skills.py` + `atlas/train.py`: balanced imitation + h alignment

**2a. `compute_her_loss` — change the policy CE to a frequency-balanced, label-smoothed CE:**
```python
# a_pos: [N] action labels
counts = torch.bincount(a_pos, minlength=17).clamp_min(1)
w = (1.0 / counts[a_pos].float())            # per-sample weight = inverse action frequency
w = w * (len(a_pos) / w.sum())               # normalize mean weight to 1.0
w = w.clamp(max=weight_clip)                 # 50.0, from SkillsConfig
ce_loss = F.cross_entropy(policy_logits, a_pos, weight=None,
                          label_smoothing=label_smoothing, reduction="none")
ce_loss = (ce_loss * w).mean()               # weighted mean
# reachability BCE stays EXACTLY as v1 (unweighted) — do not touch it
```
Do NOT subsample, do NOT duplicate data, do NOT balance the reachability labels.

**2b. `HeadTrainer.cache_latents` — fix the h off-by-one. Replace the per-episode loop with:**
```python
h_t = torch.zeros(1, RECURRENT_DIM, device=self.device)
for t in range(ep_len):
    if t > 0:
        h_t = recurrent_step(z[t:t+1], a[t-1:t], h_t)   # NOTE: a_{t-1}, NOT a_t
    h_arr[start + t] = h_t.cpu().numpy().squeeze(0)
    if t < ep_len - 1:
        z_next_pred = predictor(z[t:t+1], h_t, a[t:t+1])
        target_residuals[start + t] = (z[t+1:t+2] - z_next_pred).cpu().numpy().squeeze(0)
```
This makes cached h identical to play-time h (arbiter uses `h = recurrent_step(z, a_prev, h_prev)`).

**2c. `train_epoch` gates — make them honest:**
- Reachability AUROC: build validation pairs with **two negative types** mixed 50/50:
  (i) cross-episode goals (v1 behavior), (ii) within-episode far goals: `(s_t, g=z_u)` where
  `u > t + 2*max(horizons)` in the SAME episode. Report AUROC on this mixed set as
  `reach_auroc_hard` (this is the gate: pass ≥ 0.65). Keep the easy AUROC logged as
  `reach_auroc_easy` for comparison only.
- Death recall: compute on **validation episodes only** (`val_eps`), pass ≥ 0.40. If val has no
  deaths, report `null` and mark gate `"skipped_no_deaths"` — NOT passed.
- Log `her_action_histogram` (the batch's action distribution) every iteration.

**Verification 2:** new test `test_atlas_skills.py::test_h_alignment`: run 20 random steps through
a stub "play loop" (same recurrence as arbiter) and through `cache_latents` on the same sequence;
assert `np.allclose(h_play, h_cached)`. Test that a synthetic 90%-one-action HER batch yields
per-action total weight mass within [0.5, 2.0]× equal share, and weighted CE of the majority
action contributes ≤ 2× any minority action's contribution.

### Step 3 — `atlas/arbiter.py`: exploration everywhere, WANDER, honest probe credit, prior off

**3a. Epsilon tail in ALL modes.** After `a_intended` is chosen by ANY mode (NAVIGATE, PROBE,
FALLBACK) and BEFORE the safety override:
```python
if np.random.rand() < self.epsilon:
    a_intended = int(np.random.randint(0, NUM_ACTIONS))
```
FALLBACK itself becomes: `logits = beta_t * disagreement_all_actions(s)` (+ nothing else when
`prior_weight == 0`; if `prior_weight > 0`, add `prior_weight * prior_logits`), then argmax.

**3b. WANDER mode.** Add state `self.wander_steps_remaining = 0`. In `reset_episode` zero it.
When a NAVIGATE option times out or stagnates (v1 set mode FALLBACK and immediately reselected):
```python
self.goals.mark_used(key, success=False, ...)   # unchanged
self.active_goal = None
self.mode = "WANDER"
self.wander_steps_remaining = self.cfg.skills.wander_steps   # 10
```
In the mode machine, WANDER takes precedence like PROBE does: while `wander_steps_remaining > 0`:
decrement, set `self.mode = "WANDER"`, and choose the action as
`argmax_a softmax(beta_t * disagreement_all_actions(s))` (sample with temperature 1.0 — i.e.
sample from the softmax, not argmax, to keep it stochastic). After it hits 0, fall through to
normal goal reselection. Telemetry mode string `"WANDER"`; add to `screen.py` mode_counts dict.

**3c. Honest PROBE success credit.** Add `self.probe_unlocked = False` (reset on episode reset and
on probe start). In `act()`, where the belief is updated from the previous step's prediction
(`belief.update(...)` returns `newly_unlocked`): if the previous step's mode was PROBE (track
`self.prev_mode`) and `newly_unlocked` is non-empty, set `self.probe_unlocked = True`.
When a probe ends (the existing `mark_used` site), call
`mark_used(key, success=self.probe_unlocked, ...)`. Never mark success unconditionally.

**3d. Frozen memory.** Add `self.frozen_memory = False`. Gate BOTH:
`goals.mark_used(...)` → `if not self.frozen_memory:` and
`topomap.observe(...)` persistent mirror → when frozen, skip entirely (episode-local observation
may also be skipped; it is not used by decisions).
`screen_agent` in `screen.py` sets `agent.frozen_memory = True` before the run and restores False
after (save/restore pattern like epsilon/beta).

**3e. Prior weight.** Replace every direct `backbone.prior_logits(...)` usage with a call that
multiplies logits by `self.cfg.backbone.prior_weight` (implement as
`backbone.prior_logits_weighted(s, m)` returning `logits * prior_weight`). With the default 0.0
this is a zero vector — FALLBACK is then pure curiosity + exploration tail.

**Verification 3:** update `test_atlas_arbiter.py`:
- stub test: 1000 act() calls with epsilon=0.3 in NAVIGATE mode produce ≥ 3 distinct actions;
- timeout → exactly 10 WANDER telemetry steps → then a goal selection attempt;
- probe with no belief unlock → `mark_used` called with `success=False`;
- `frozen_memory=True` → `goals.entries[*].n_try` unchanged after 50 probes;
- `prior_weight=0` → FALLBACK logits independent of the prior net (stub two priors, identical output).

### Step 4 — NEW FILE `atlas/health.py` + wiring in `selflearn.py`

```python
class HealthMonitor:
    def __init__(self, cfg: HealthConfig)
    def observe_action(self, action_idx: int) -> None        # push into deque(maxlen=window_steps)
    def observe_confidence(self, mean_max_prob: float) -> None   # from val HER batch each iter
    def observe_progress(self, iter_idx: int, ach_per_ep: float) -> None
    def check(self, iter_idx: int) -> dict:
        # returns {"alarms": [str,...], "epsilon_boost_active": bool, "boost_steps_left": int,
        #          "beta_reset": bool}
        # alarms:
        #  A1 action diversity: top action share over window > 0.60
        #     -> engage boost: next 100 collect steps use epsilon * 2.0
        #  A2 confidence: mean_max_prob > 0.85 -> ALARM (log only)
        #  A3 flatline: best ach_per_ep not improved by >=10% within last 15 observations
        #     -> ALARM + beta_reset=True (selflearn sets agent.beta_t = 0.8 for next iter,
        #        then the normal schedule resumes)
        # every alarm: print("!!! [HEALTH ALARM] ...") AND append to train_log.jsonl
```
Wire in `SelfLearnLoop.run_iterations`: every `agent.act` during collection pushes the action
(via the collector); after training, push val confidence; after dev screens, push ach/ep; call
`check(it)` each iteration and apply `epsilon_boost` / `beta_reset` to the collector schedules.
Implement boost by multiplying the collector's current epsilon for the next `boost_steps`
collected steps (a counter, not a permanent change).

**Report cards:** at iterations 30 and 50 write `results/atlas_health_it<N>.json`:
```json
{"top_action_share": ..., "optionnet_confidence": ..., "goal_bank_largest_dim_share": ...,
 "goal_bank_per_dim_counts": {...}, "ach_per_ep_trend_vs_it10": ...,
 "lift_edges_used": ..., "max_ci_low": ..., "alarms_so_far": [...],
 "pass": {"diversity": true/false, "humility": ..., "album_variety": ..., "trend": ..., "lift": ...}}
```
Pass thresholds: top action share ≤ 0.50; confidence ≤ 0.80; largest dim share ≤ 0.40; trend
≥ +10% vs the iteration-10 baseline (use dev-screen ach/ep); all lift edges consistent.
`goal_bank_largest_dim_share` = max over dims of (count of event goals containing that dim) / bank size.
Compute `optionnet_confidence` as mean max-softmax over the val HER positives.

**Verification 4:** unit tests: feed 200 identical actions → A1 fires and boost engages; feed
confidence 0.9 → A2 logs; feed flat progress → A3 fires and beta_reset True; report-card JSON
contains all keys.

### Step 5 — `atlas/collect.py`: journal from step 0 with positions

- Remove any condition that gates journal writing on iteration (v1 added it mid-run).
- In `run_episode`, capture `pos = info.get("player_pos")` after each `env.step` (engine
  diagnostics for AUDIT ONLY — it must never enter `agent.act` or any head; double-check the
  call signature stays `(obs_uint8, reward_prev, step)`).
- Journal entry adds: `"positions": [[x, y], ...]`, `"moved_steps"` (count of position changes),
  `"moved_fraction": moved_steps / len(steps)`, `"action_histogram": {action_name: count}`.
- Keep all existing fields (mode counts, overrides, achievements, per-step reasons).

**Verification 5:** round-trip test: run 30 steps, assert journal line parses, `moved_fraction`
matches the position trace, and `agent.act` was never called with position data (signature check
in test via `inspect.signature`).

### Step 6 — `atlas/screen.py` + `atlas/colab/run_atlas.py`: provenance gate, no bootstrap

- `screen.py` report adds and REQUIRES:
```python
"bootstrap": "none", "prior_weight": cfg.backbone.prior_weight,
"train_from_scratch": True, "run_line": "atlas_v2"
```
  If any of these is missing or `prior_weight != 0.0`, raise and do not write the JSON.
- Set/restore `agent.frozen_memory` around the screening loop (Step 3d).
- `run_atlas.py`: delete the `bootstrap` phase entirely (argparse choice removed; selecting it
  errors). Default `--run-id atlas_v2_scratch`. In `main()`, before creating the SelfLearnLoop:
  if a `state.pt` exists in the target run dir, it belongs to this run — resuming is fine; but
  add an assertion that the run dir is NOT under any `archive*` path.
- Archive step (one-time, do it in code? NO — do it manually in the repo before the run):
  move `atlas_runs/<v1 run dirs>` to `atlas_runs/archive_v1_turret/`. v2 code must never read it.

**Verification 6:** attempt `screen_agent` with `prior_weight=0.5` → raises; with 0.0 → writes
JSON containing the four new fields; `--phase bootstrap` → argparse error.

### Step 7 — `atlas/smoke.py` + tests: prove from-scratch works end to end

- Smoke defaults: empty banks (do NOT load any), `prior_weight=0.0`, 2 episodes × 150 steps,
  assert: journal has positions/moved_fraction; telemetry contains WANDER at least once when
  goals stay unreachable for >25 steps (force with stub goals bank if needed); provenance fields
  present in the mini report; backbone `assert_frozen()` passes after the whole smoke.
- Fix the existing float-precision failure in `test_atlas_belief.py`:
  `assert belief.m_soft[3] == 0.40` → `assert np.isclose(belief.m_soft[3], 0.40)`.
- Final gate for hand-back: `pytest atlas/tests -q` → **all green**, and
  `python -m atlas.smoke` runs clean on MPS or CPU.

---

## Part D — RUN PROTOCOL (after all verifications pass)

```
# Colab, repo at /content/learn-a-world, Drive mounted:
export ATLAS_CHECKPOINT=/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_state_readout_v2.pt
python -m atlas.colab.run_atlas --phase selflearn --iters 100 --run-id atlas_v2_scratch --minutes 300
# repeat the same command until 100 iterations complete (atomic resume every iteration)
# NO --phase bootstrap. NO shard paths. NO checkpoints of heads loaded from anywhere.
```
Dev screens every 5 iters (existing). Health report cards auto-written at iters 30 and 50.

## Part E — INTERPRETATION RULES (for the operator, not tunable by you)

- At iterations 30–50, judge by `atlas_health_it30.json` / `it50.json` PASS flags, NOT by score.
- Expect a LOWER score than v1 early: v1's 2.3 ach/ep was stationary luck-farming. Movement,
  album variety, and humility are the success criteria at this stage.
- If ALL five pass at 50: continue to 100+ before evaluating score.
- If any FAIL: the alarms will already have responded (epsilon boost / beta reset). Do not
  hand-tune mid-run; let the run finish and analyze the journal.
- When reporting behavior, always report ACTION histograms (never "NAVIGATE %" alone — v1's
  "94.9% NAVIGATE" was a cognitive-mode label misread as walking).

## Part F — IMPLEMENTATION ORDER & DEFINITION OF DONE

Implement Steps 1→7 strictly in order, running each step's verification before moving on.
Definition of Done: all tests green (old fixed + new), smoke clean, `git diff` touches ONLY the
allowed files, and a one-screen summary listing: tests added, verification outputs, and the exact
Colab command to launch.
