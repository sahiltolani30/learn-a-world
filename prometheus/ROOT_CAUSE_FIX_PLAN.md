# ROOT_CAUSE_FIX_PLAN.md — Consolidation Redesign for the Crafter JEPA Agent

**Version:** 1.0 — 2026-10-31
**Audience:** implementing agent. Everything below is a spec: follow it literally. Do not redesign, do not add features, do not touch anything in §2 (Do-Not-Touch). If a detail seems ambiguous, prefer the simplest reading that satisfies the acceptance tests in §6.

**Zero-cheat rule (project law, from the original request R2/R3):** the agent may learn ONLY from raw pixels, actions, and environment reward. Never read `info['inventory']`, `info['semantic']`, `info['player_pos']`, or any privileged state in policy, critic, buffer, or reward code. Achievement names in `eval_info` may be used ONLY for logging and for the milestone *flag* (data selection), exactly as today.

---

## 0. Context (why these changes — established by forensic audit, 2026-10-31)

Full evidence in the forensic report and `git log` discussions. Summary of measured facts:

- The agent discovers `place_table` in **15 of 142 training lives (9.9%)** but **0 of 50 greedy eval games**. Discovery works; **consolidation does not**.
- Critic Q-values are a fossil of the early curiosity era: all 17 actions sit at Q ≈ 22 ± 0.25 (flat). Cause: the critic trains on `r_ext + 0.05·r_int` where stored `r_int` was clipped-saturated at 10 early in training (`0.5/step → Q→50 era`) and ≈1.2 late (`→ 6 era`).
- Remaining curiosity (4-head disagreement) fires ONLY on movement (4–18; irreducible procedural-terrain noise) and ≈0 on all interaction actions. Welford ÷std (std = 0.031) amplifies that noise back to O(1).
- ~1,325 place_table presses vs 15 successes → critic/world-model prior is "place_table = no-op".
- One-step TD cannot propagate a +1 backward across the chop→chop→place chain.
- Official 50-game score: 1.40% (below the ~4.6% random baseline) partly because greedy-argmax on a flat Q creates repetitive habits.

**Root cause (one sentence):** the system has no mechanism that converts rare external-reward events into policy. This plan adds that mechanism. It is achievement-agnostic: fixing it once fixes `place_table`, `make_wood_pickaxe`, stone/coal/iron, and everything downstream.

---

## 1. Changes overview

| ID | Change | Fixes | Size |
|----|--------|-------|------|
| C1 | Critic learns external reward only | Q fossil | ~5 lines |
| C2 | Milestone-prioritized replay + trailing-window flags | success drowning | ~60 lines |
| C3 | n-step (5-step) critic targets | chain credit assignment | ~120 lines |
| C4 | Per-action normalized curiosity in actor + novelty-modulated ε (episodic k-NN) | dead/mispointed exploration compass | ~150 lines |
| C5 | Throughput: update interval, batch size, optional action repeat | step budget | ~15 lines |
| C6 | Stochastic + greedy evaluation | misleading eval | ~15 lines |
| C7 | Telemetry | verification | ~30 lines |

Implement in the order C1 → C2 → C5 → C6 → C7 → C4 → C3 (§5). Run the test suite after each change.

---

## 2. DO NOT TOUCH (exonerated by forensics — probes: wood 99.2%, facing 92.0%)

- `src/models/encoder.py` (ObservationEncoder / ResNetTiny)
- `src/models/predictor.py` (ResidualMLPPredictor, ActionEmbedding)
- `src/models/target.py` (TargetEncoder EMA)
- `src/models/losses.py` (VICRegLoss)
- `src/env/wrapper.py` (ZeroCheatWrapper — zero-cheat guarantee lives here)
- `src/agent/actor_critic.py` network architectures (ActorPrior, CriticHead, TwinCritic)
- `src/agent/planner.py` (dead code for now; not part of this plan)
- Observation format (3,64,64) float [0,1]; action space (17); action indices.

**Forbidden designs (would be cheating — do not implement even as options):**
- Action masking from inventory/privileged state.
- Dense per-achievement reward shaping computed from privileged state.
- Per-achievement scripted behaviors or curricula.
- Feeding `eval_info` achievement names anywhere except logging and the milestone flag (existing behavior in `train.py:344-355`).

---

## 3. Detailed changes

### C1 — Critic learns external reward only

**File:** `src/train.py`, method `TrainingPipeline.update_actor_critic()` (currently lines ~429–497).

**Current code (lines ~445–449):**
```python
r_total = self.agent.curiosity_engine.blend_rewards(
    batch["rewards"],
    batch["int_rewards"],
)
```

**Replace with:**
```python
r_ext = batch["rewards"]
r_total = r_ext + self.critic_intrinsic_weight * batch["int_rewards"]
```
where `self.critic_intrinsic_weight` is a new config float, **default 0.0** (`--critic_intrinsic_weight`, also read from dict config in `__init__` next to `self.tier2_ratio`, ~line 192). Keep everything else in the method unchanged (Double-DQN target at lines ~451–457, MSE critic loss, EMA update).

**Why:** removes the curiosity-era fossil from the value function. Curiosity re-enters behavior through C4 (actor term), never through bootstrapped value.

**Acceptance:** unit test — build a synthetic batch where `int_rewards = 100.0` and `rewards = 1.0`; with default config, the computed `target_q` must equal the value obtained with `int_rewards = 0.0`.

---

### C2 — Milestone-prioritized replay + trailing-window flags

**Files:** `src/memory/buffer.py`, `src/train.py`.

**C2.1 — Milestone flag in Tier 1.** `TwoTierReplayBuffer` currently records `is_milestone` only for Tier-2 admission. Add a parallel T1 flag array:
- In `_allocate_tier1`: `self.t1_is_milestone = np.empty(self.tier1_capacity, dtype=np.bool_)`.
- In `add()` (Tier-1 insert, lines ~229–240): store `self.t1_is_milestone[idx] = is_milestone` and reset the slot on wrap: when `t1_ptr` wraps past a slot, the flag is simply overwritten by the next insert (no extra work needed).
- Add method:
```python
def flag_milestone_at(self, index: int) -> None:
    """Mark the T1 slot at `index` as milestone (backward credit window)."""
    if 0 <= index < self.tier1_capacity:
        self.t1_is_milestone[index] = True
```
- Serialize in `state_dict()` / restore in `load_state_dict()` (add `t1_is_milestone` alongside the other arrays, using `self.t1_size` slicing).

**C2.2 — Trailing window in the pipeline.** In `TrainingPipeline`:
- `__init__`: `self.recent_t1_indices = collections.deque(maxlen=30)` and `self.total_milestones = 0`.
- `collect_step()`: `buffer.add(...)` returns nothing today — capture the written slot: after the `self.buffer.add(...)` call, append `(self.buffer.t1_ptr - 1) % self.buffer.tier1_capacity` to `self.recent_t1_indices`.
- On milestone (`is_milestone == True`, computed at lines ~341–355): iterate `self.recent_t1_indices` and call `self.buffer.flag_milestone_at(i)`; increment `self.total_milestones`.

**C2.3 — Milestone-prioritized sampling.** In `TwoTierReplayBuffer.sample()` (lines ~307–405):
- Add constructor params: `milestone_t1_priority: float = 0.25`, `milestone_t2_priority: float = 0.80` (exposed to config as `--milestone_t1_priority`, `--milestone_t2_priority`).
- T1 portion (`actual_n1` draws): with probability `milestone_t1_priority` per draw, replace the uniform draw with a draw from the milestone subset:
```python
ms = np.nonzero(self.t1_is_milestone[:self.t1_size])[0]
# per draw: if len(ms) > 0 and rng.random_sample() < self.milestone_t1_priority:
#     idx = ms[self.rng.randint(0, len(ms))]
# else: idx = self.rng.randint(0, self.t1_size)
```
- T2 portion (`actual_n2` draws): same logic with `t2_is_milestone[:self.t2_size]` and `milestone_t2_priority`.
- Add the milestone flag of each sampled item to the returned dict: `"is_milestone": tensor(bool)` (gather from the respective flag arrays).

**Why:** 15 successes currently float among 8,149 Tier-2 entries (0.18% of T2, ~0.06% of a batch). After this change they dominate the T2 share and get a 30-step trailing window of context so the critic sees the wood-carrying approach states that led to success.

**Note on the existing dual milestone trigger** (`train.py:344-355`: new achievement OR `r_int > 0.5`): leave both in place. The achievement path is scale-free (a boolean on events) and is the load-bearing one for C2/C3 — it is what flags the 15 successes. The `r_int > 0.5` path sits on a drifting Welford scale and only adds extra novelty entries to T2 admission; that drift is acceptable and known (documented here so nobody "fixes" it by reading privileged state).

**Acceptance:** unit test — fill the buffer with 1,000 transitions, flag 10 as milestone, draw 200 batches of 64; the fraction of milestone items must be ≥ 8% (0.25·(T1 share) + 0.80·(T2 share) scaled by actual proportions). Also test `flag_milestone_at` wraps safely and `state_dict` round-trips the new array.

---

### C3 — n-step (5-step) critic targets

**Files:** `src/memory/buffer.py` (data layout + `sample_nstep`), `src/train.py` (`update_actor_critic` uses it), `src/train.py` (`collect_step` passes episode metadata).

**C3.1 — Sequence metadata in the buffer.** Add to Tier 1 (and Tier 2, same layout):
- `_allocate_tier1`: `self.t1_episode_id = np.empty(self.tier1_capacity, dtype=np.int64)`, `self.t1_step_in_ep = np.empty(self.tier1_capacity, dtype=np.int64)`.
- `add()` signature: add optional `episode_id: int = -1, step_in_ep: int = -1` (backward compatible — existing callers/tests keep working; default −1 means "unknown" and disables n-step expansion for that slot).
- Store both arrays on insert; serialize/restore in `state_dict`/`load_state_dict`.

**C3.2 — Pipeline metadata.** `TrainingPipeline.collect_step()` gains params `episode_id: int, step_in_ep: int` and passes them to `self.buffer.add(...)`. The train loop (`train()`, lines ~620–633) passes `self.current_episode` and the current `ep_length` (increment `ep_length` BEFORE the `collect_step` call or pass `ep_length` then increment — pick one, document with the done-reset logic; on `done`, the episode counter increments and `ep_length` resets as today).

**C3.3 — `sample_nstep`.** New method on `TwoTierReplayBuffer`:
```python
def sample_nstep(self, batch_size: int, n: int = 5, gamma: float = 0.99,
                 tier2_ratio: Optional[float] = None, device="cpu") -> Dict[str, torch.Tensor]:
```
Behavior:
1. Draw base indices exactly like `sample()` (same tier2_ratio logic, same milestone priorities from C2).
2. For each base index `i`:
   - If the slot has `episode_id < 0` OR came from Tier 2 → **fallback to 1-step**: `R = rewards[i]`, bootstrap from `next_obs[i]` with `boot_gamma = gamma`, `boot_valid = not done[i]`.
   - Else walk forward `k = 1 .. n`: candidate slot `j = (i + k) % tier1_capacity`. The chain is valid while `t1_episode_id[j] == t1_episode_id[i]` and `t1_step_in_ep[j] == t1_step_in_ep[i] + k` and `t1_dones[(j - 1) % cap] == False`. Accumulate `R += gamma**k * t1_rewards[j]`… note: the base transition's own reward is `t1_rewards[i]` (k=0 term). Stop at the first invalid slot or when a `done` is included in the sum.
   - Output per item: `R` (discounted reward sum including k=0), `boot_index` = last valid slot index `j` (so the bootstrap state is `t1_next_obs[j]`), `boot_gamma = gamma**(K)` where K = number of reward terms summed AFTER the k=0 term (if only the base reward fit, `boot_index = i` and `boot_gamma = gamma`), `boot_valid = (no done inside the window)`.
3. Return the same dict as `sample()` plus: `"nstep_return": (B,) float32`, `"boot_obs": (B, C, H, W) float32 /255`, `"boot_gamma": (B,) float32`, `"boot_valid": (B,) bool`.
4. Ring-wrap guard: never walk into slots that belong to the *not-yet-overwritten future* incorrectly — the `episode_id`/`step_in_ep` checks make stale wrapped slots naturally invalid; assert during testing that `k` never exceeds `t1_size`.

**C3.4 — Critic update.** In `update_actor_critic()`:
- Sample with `batch = self.buffer.sample_nstep(self.batch_size, n=self.nstep, gamma=self.gamma, tier2_ratio=self.tier2_ratio, device=self.device)`.
- Replace the target computation (lines ~441–457) with:
```python
with torch.no_grad():
    z_curr = self.agent.encoder(batch["obs"])
    r_total = batch["rewards"] + self.critic_intrinsic_weight * batch["int_rewards"]  # C1; n-step R replaces batch["rewards"]
    # NOTE: use batch["nstep_return"] + critic_intrinsic_weight * batch["int_rewards"] (intrinsic stays 1-step, k=0 only)
    z_boot = self.agent.target_encoder(batch["boot_obs"])
    q1b, q2b = self.agent.critic(z_boot)
    best_boot_actions = torch.argmax(torch.min(q1b, q2b), dim=-1)
    boot_q = self.agent.critic.target_q_min(z_boot, best_boot_actions)
    target_q = batch["nstep_return"] + batch["boot_gamma"] * batch["boot_valid"].float() * boot_q
```
(Intrinsic reward participates only at k=0 to keep the value function externally anchored; it is 0-weighted by C1 anyway.)

**Why:** a +1 at the place_table step now lands, discounted, on the states 1–5 steps earlier (the wood-carrying approach), instead of requiring one-step generalization to span the chain.

**Config:** `--nstep` int default 5.

**Acceptance:** unit test with a synthetic ring buffer: 10 transitions, one episode, rewards `r=[0,0,0,1,0,...]`, `gamma=0.99`, `n=5` — verify `sample_nstep` returns the exact hand-computed discounted sums and bootstrap indices; verify a `done` truncates the sum and zeroes the bootstrap; verify the wrap-around case (episode crossing `tier1_capacity` boundary) is rejected by the metadata checks rather than silently mixing episodes.

---

### C4 — Exploration: per-action normalized curiosity (actor) + episodic-novelty ε (collection)

The old compass (raw disagreement through the critic) is gone after C1. This replaces it with two unbootstrapped signals that cannot re-create the fossil.

**C4.1 — Per-action disagreement term in the actor loss.**

**File:** `src/train.py`, `update_actor_critic()`.

After computing `z_curr` (detached), compute per-action ensemble disagreement under `torch.no_grad()` (this is an exploration signal, NOT a training signal — no gradients into encoder/predictor):
```python
with torch.no_grad():
    B = z_curr.shape[0]
    z_rep = z_curr.unsqueeze(1).expand(B, self.agent.num_actions, -1).reshape(B * self.agent.num_actions, -1)
    a_rep = torch.arange(self.agent.num_actions, device=z_curr.device).repeat(B)
    _, all_preds = self.agent.predictor(z_rep, a_rep)          # (K, B*17, D)
    disc = self.agent.predictor.compute_epistemic_disagreement(all_preds).view(B, -1)  # (B, 17)
    self.per_action_disc_stats.update(disc)                    # see below
    disc_norm = disc / (self.per_action_disc_stats.std + self.per_action_disc_stats.epsilon)  # ÷std per action column
    disc_norm = torch.clamp(disc_norm, min=0.0, max=5.0)
```
- `self.per_action_disc_stats = RunningMeanStd(shape=(num_actions,))` — reuse `src/curiosity/ensemble.RunningMeanStd` (its `update()` already handles trailing-dim shape `(17,)` with a leading batch dim). Initialize in `__init__`.
- Actor loss (currently lines ~476–490) becomes:
```python
expected_q = (probs * q_all).sum(dim=-1)
expected_int = (probs * disc_norm).sum(dim=-1)
entropy = -(probs * log_probs).sum(dim=-1)
actor_loss = -(expected_q + self.actor_intrinsic_weight * expected_int + 0.05 * entropy).mean()
```
- Config: `--actor_intrinsic_weight` float, **default 0.1** (replaces the old β=0.05 blending, which C1 removed from the critic).

**Why per-action normalization:** movement actions have 100–300× the raw disagreement of interaction actions (terrain noise). Dividing each action column by its own running std makes "surprise" comparable across actions, so `do`-near-a-tree competes fairly with `move_*`.

**Why this cannot fossilize:** the term is one-step, recomputed fresh every update from the current predictor, never stored, never bootstrapped through γ.

**C4.2 — Episodic k-NN novelty with novelty-modulated ε.**

**New file:** `src/curiosity/episodic.py`:
```python
class EpisodicNovelty:
    """Per-episode k-NN novelty in latent space (NGU-style, simplified).

    novelty(z) = mean distance to the k nearest stored keys (float, unnormalized),
    then the key is inserted. Reset at every episode start. Uses float16 storage.
    """
    def __init__(self, latent_dim: int = 256, max_keys: int = 4096, k: int = 10, seed: int = 0): ...
    def reset(self) -> None: ...                       # clear keys
    def novelty_and_insert(self, z: np.ndarray) -> float: ...
        # if count < k: insert and return large sentinel (e.g. 1e3) — early steps are maximally novel
        # dists = sqrt(((keys[:count].astype(np.float32) - z)**2).sum(-1))
        # score = mean of the k smallest dists
        # insert z at ring pointer; return score
```
- Normalize the returned scores with a scalar `RunningMeanStd` (reuse `RunningMeanStd(shape=())`, ÷std, clamp [0, 5]) — keep the normalizer inside `TrainingPipeline`, not the module.
- max_keys = 4096 keeps the distance computation < 1 ms/step on CPU (4096×256 float32 ≈ 1M FLOPs).

**File:** `src/train.py`:
- `__init__`: `self.episodic_novelty = EpisodicNovelty(latent_dim=256)`, `self.episodic_stats = RunningMeanStd(shape=())`, plus config: `--eps_base` 0.05, `--eps_novelty_coef` 0.20, `--eps_max` 0.50.
- `collect_step()`: after computing `z_t` (already computed for curiosity at lines ~332–339), also:
```python
nov_raw = self.episodic_novelty.novelty_and_insert(z_t.detach().squeeze(0).cpu().numpy())
self.episodic_stats.update(nov_raw)
nov = float(np.clip(nov_raw / (float(self.episodic_stats.std) + 1e-4), 0.0, 5.0))
```
  Action selection (currently lines ~316–319) becomes:
```python
if self.current_step < self.warmup_steps:
    action = int(self.env.action_space.sample())
else:
    eps = min(self.eps_max, self.eps_base + self.eps_novelty_coef * nov)
    if float(np.random.random_sample()) < eps:
        action = int(self.env.action_space.sample())
    else:
        action = self.agent.act(obs, eval_mode=False, planning=False)
```
- Reset per episode: in the `if done:` block of `train()` (lines ~628–633) add `self.episodic_novelty.reset()` (the stats normalizer persists).

**Why:** in rarely-seen states (which is exactly the wood≥2-facing-free-tile situation early on), action randomness rises toward 0.5 — the agent explores harder precisely where the unexplored preconditions live, while familiar states follow the learned policy. Within-episode reset means the first tree of every new life pays novelty again — re-motivating interactions each life (this is what fixes the "compass died" problem without touching the critic).

**Acceptance:** unit tests — (a) first call returns sentinel; (b) inserting the same z repeatedly drives novelty toward ~0; (c) `reset()` restores sentinel behavior; (d) a state far from all keys scores higher than one near stored keys; (e) `RunningMeanStd(shape=(17,))` update path handles a `(B, 17)` batch.

---

### C5 — Throughput

**File:** `src/train.py`.

1. **Update interval:** `--update_every` int, default 2. In `train()`, gate the update block (lines ~636–640) with `... and (self.current_step % self.update_every == 0)`. Note in the docstring: this halves gradient updates per env step; raise `--batch_size` to 128 to keep samples-per-update constant (default `--batch_size` changes 64 → 128).
2. **Action repeat (optional flag, default 1):** `--action_repeat` int. When > 1: in `collect_step`, step the env k times with the SAME action, sum rewards, use the final obs as `next_obs`, `done = any(terminated or truncated)`. Store the summed reward as `reward`. Keep k=1 for the 25k verification run; enable k=2 only for the long scale run. (DreamerV3's Crafter setting uses action repeat 2.)
3. **Do NOT implement vectorized envs in this pass** (§7 future work).

**Acceptance:** SPS on the Colab GPU must be ≥ 25 (from ~16) with `update_every=2, batch_size=128, action_repeat=1`, and all 171 existing tests still pass (update any test that asserts per-step update behavior).

---

### C6 — Evaluation protocol: stochastic AND greedy

**Files:** `src/train.py` (`evaluate`), `src/diagnostics.py` (`run_benchmark`).

- `evaluate(self, num_episodes=None, stochastic: bool = False)`; action selection: `self.agent.act(obs, eval_mode=not stochastic, planning=False)`. Record and return BOTH variants per eval cycle: run 3 greedy episodes + 3 stochastic episodes (`eval_episodes` applies to each). Store in `achievement_history[step]` under keys `greedy_*` and `stochastic_*` (keep the existing top-level keys pointing at the greedy numbers so old plots/tests don't break).
- `run_benchmark(..., stochastic: bool = False)` — same one-line change at the action call (currently `diagnostics.py:316`); the printed report gains a line `Policy: greedy|stochastic`.
- During the exploration phase (first ~200k steps), report the **stochastic** number as the primary exploration metric and greedy as the exploitation metric; both go to the same `FORENSIC_DIAGNOSTIC_REPORT.md`.

**Why:** greedy argmax on a still-flat Q produces repetitive habits with blind spots (measured: 1.40% official score vs 7 achievements found by the stochastic policy). This is a measurement fix only — it changes nothing in learning.

---

### C7 — Telemetry (needed to verify the fix worked)

**File:** `src/train.py`.

Add to the metrics dict appended to `self.loss_history` each update:
- `q_spread`: `(q_all.max(dim=-1).values - q_all.min(dim=-1).values).mean()` over the actor batch (tracks the flat-Q healing; expect < 1.0 early, growing steadily).
- `q_mean_ext`: mean of `expected_q` (now external-only).
- `expected_int_mean`: mean of `expected_int` (exploration health).
- `eps_mean`: running mean of the ε used in `collect_step` (exploration health).
Add to per-log_interval print: `MS: {self.total_milestones}` (total milestone-flagged transitions).
Add success counters in `collect_step` (legitimate: derived from reward + action only):
```python
if ext_reward >= 0.5:
    self.success_counts[self.env.action_names[action]] = self.success_counts.get(self.env.action_names[action], 0) + 1
```
Print `self.success_counts` in the final training report. These counters must reproduce the funnel numbers (wood≈70% of lives, table≈10% of lives at current exploration strength — if they collapse after the changes, exploration degraded and C4 weights need revisiting).

Additionally, at every eval cycle call the currently-unused safety check and log it:
```python
from src.models.losses import verify_non_collapse  # already imported at top of train.py
report = verify_non_collapse(z_sample_batch, strict=False)   # z from any sampled batch
# append to the eval summary: report["min_variance"], report["num_collapsed_dims"], report["effective_rank"]
```
This activates the `verify_non_collapse` import that currently sits unused in `train.py` and turns the non-collapse guarantee into a monitored metric instead of a dead check.

---

## 4. New config keys (summary)

| Key | Default | Meaning |
|-----|---------|---------|
| `critic_intrinsic_weight` | 0.0 | β for r_int inside the critic target (C1). Must be 0.0 for the primary runs. |
| `nstep` | 5 | n-step critic targets (C3). |
| `actor_intrinsic_weight` | 0.1 | weight of per-action normalized disagreement in the actor loss (C4.1). |
| `eps_base` / `eps_novelty_coef` / `eps_max` | 0.05 / 0.20 / 0.50 | novelty-modulated ε in collection (C4.2). |
| `milestone_t1_priority` / `milestone_t2_priority` | 0.25 / 0.80 | milestone-prioritized sampling (C2.3). |
| `update_every` | 2 | gradient update interval in env steps (C5). |
| `batch_size` | 128 | (was 64) |
| `action_repeat` | 1 | optional; use 2 for long runs (C5.2). |

All must flow through both `parse_args()` and the dict-based `TrainingPipeline(config)` path (follow the existing pattern used by `tier2_ratio`, `gamma`, etc.).

---

## 5. Implementation order (dependency-respecting)

1. **C1** reward channels + config plumbing.
2. **C2** milestone flags, windows, prioritized sampling (+ state_dict round-trip).
3. **C5** update interval + batch size (immediate SPS win, trivial).
4. **C6** stochastic/greedy eval.
5. **C7** telemetry.
6. **C4** episodic k-NN module + per-action stats + collection ε + actor term.
7. **C3** n-step sampling + critic target rewrite (largest surgery, do last so its acceptance tests run against an already-stabilized loop).

After each step: `python3 -m unittest discover tests` — all existing tests must pass (update only tests that assert the old blended-reward or per-step-update behavior; do not delete tests). Finish with `python3 -m src.test_smoke`.

---

## 6. Verification plan

### 6.1 Unit-level
Every change lists its acceptance test in §3. Put them in `tests/test_consolidation_fix.py`.

### 6.2 Fresh 25k-step smoke run (Colab, ~25 min)
- **Fresh run only** — `no_resume=True` and a NEW checkpoint dir (e.g. `.../checkpoints_v2/`). Do NOT resume the old fossil critic.
- Pass gates after 25k steps:
  - No NaNs; critic loss mean < 0.3 (external rewards are sparse; the old 0.85 floor was intrinsic noise).
  - `q_spread` > 0.5 and growing (flat landscape is healing).
  - Success counters: wood-lives ≥ 60%, table-lives ≥ 5% (exploration must not degrade vs the 70%/10% baseline).
  - Milestone fraction inside sampled batches ≈ C2 targets (instrument `sample()` for one run or add a temporary assert).
  - `collect_table`/`collect_wood` in `success_counts` > 0.

### 6.3 The ONE tracking metric (decides whether consolidation started)
With the existing forensics harness (`AgentForensics.explain_step`), on the constructed scene **(b) wood=2, facing empty grass**: `rank of Q(place_table) among the 17 Q-values ≤ 3`. (Before the fix it ranked 5th, below sleep/place_stone. Cross-scene Q deltas are NOT the metric — that was the misread in the first forensic report.)

### 6.4 Staged scale-up (GPU-budget discipline — do NOT jump to long runs)

- **Stage A — 25k fresh verification** (§6.2 + §6.3 gates). Cost: ~15–25 min. Decision point only.
- **Stage B — resume the SAME clean run to 100k** (only if Stage A passes). Checkpoints every 10k, eval every 10k. Gate: table-lives rate rising above the Stage-A value (consolidation is compounding, not just discovery), collect_wood in stochastic eval > 30%, and ideally first `place_table` appearing in stochastic eval games. Cost: ~1 h.
- **Stage C — commit to 1–2M steps** (only if Stage B passes): enable `action_repeat=2` if SPS < 40, checkpoint every 50k, eval every 10k, report BOTH greedy and stochastic. Expected ladder (indicative, not acceptance-gating): place_table in greedy eval > 30% of games → `make_wood_pickaxe` > 0 (table nearby + wood) → stone/coal follow through the same mechanisms.
- If any stage fails its gate, STOP and report the telemetry (q_spread, success counters, milestone sampling fraction, critic loss) — do not scale a failing run.

---

## 7. Out of scope (future work — do NOT implement now)

- Vectorized/parallel envs (only if SPS < 40 after C5 and 1M-step runs become the bottleneck).
- Frame stacking (`make_crafter_env(frame_stack=3)`): Crafter is a POMDP and k=1 was used throughout the audited run; forensics showed the single-frame encoder reads everything needed (wood 99.2%, facing 92%). Revisit only if the world model's one-step errors stay high at scale. NOTE: k=3 changes the encoder input shape — it interacts with the Do-Not-Touch list, so it needs its own mini-plan, not a drive-by change.
- Wiring `LatentRND` (already implemented in `src/curiosity/rnd.py`, currently unused) — C4.2's episodic novelty covers the exploration need; RND is a cross-episode-lifetime alternative to A/B later.
- CEM planner-in-collection (needs reward/termination heads + longer horizon; separate plan).
- Symlog/two-hot TD targets (DreamerV3-style) — also the designated fallback if critic instability from encoder/EMA representation drift shows up at scale (critic currently trains on online-encoder latents with EMA-encoder targets, unchanged by this plan; evidence says this is second-order, but if critic loss will not settle after C1+C3, apply symlog before touching anything else).
- Critic on target-encoder latents for both input and bootstrap (alternative drift mitigation; requires actor to consume the same latents — bundle with the symlog fallback if ever needed).
- `storage.py` `prune()` return-type bug (returns `str` instead of `List[str]` when `state.pt` exists, lines ~301–303) — harmless; fix opportunistically.

---

## 8. Known pitfalls (read before implementing)

- **Ring-buffer wrap in C3:** the `episode_id`/`step_in_ep` metadata checks are the ONLY thing preventing cross-episode contamination after the ring wraps. Test the wrap case explicitly (§3 C3 acceptance).
- **T2 has no guaranteed sequence:** n-step expansion applies to T1-sampled slots; T2 slots use the 1-step fallback. Do not attempt to walk sequences inside T2.
- **`deque(maxlen=30)` window:** milestone windows flag slots that may later be overwritten by FIFO — that's fine; the flag array is overwritten together with the data.
- **Determinism:** all new sampling uses `self.rng` (the buffer's `np.random.RandomState`) or clearly-seeded objects; do not introduce unseeded `random`/`np.random` global calls.
- **Docstring honesty:** update the module docstrings of `train.py`/`buffer.py` to describe the new reward and sampling semantics; stale docs have already caused confusion in this project (see PROJECT.md vs code drift on frame-stacking and k-NN).

---

## 9. Stage-A report template (send EXACTLY these after the 25k fresh run)

Number the sections the same way. Items marked **[M]** are mandatory; **[O]** optional but valuable.

**[M] 1. Gate checklist** (one line each, PASS/FAIL + number):
1. NaN check + mean `critic_loss` over the last 1,000 updates (gate: < 0.3, no NaNs) — from `loss_history`.
2. `q_spread` at end of run AND trend (mean of first 5k updates vs last 5k updates) — from C7 telemetry (gate: > 0.5 and growing).
3. Exploration health: `success_counts` dict from the final report + total `current_episode`; compute % of lives containing ≥1 `act=5 ∧ rew≥0.5` event (gate: ≥ 60%) and ≥1 `act=8 ∧ rew≥0.5` event (gate: ≥ 5%).
4. `total_milestones` + temporarily instrument `TwoTierReplayBuffer.sample()` for this run only: mean fraction of milestone-flagged items across the last 100 sampled batches (gate: ≥ 8% overall, T2 portion ≈ 80%).
5. Both eval modes at the final eval cycle: unique achievements + mean return for **stochastic** and for **greedy** (from C6's `greedy_*` / `stochastic_*` history keys). Gate: stochastic ≥ greedy in achievements found; cumulative unique ≥ 5.

**[M] 2. Tracking metric** — `AgentForensics.explain_step` on the constructed scene (b) (wood=2, facing empty grass): print the full ranked list of all 17 actions with Q-values and circle the **rank of `place_table`** (gate at Stage B: rank ≤ 3; at Stage A: report the number, any improvement from rank 5 counts as progress). Do NOT compare Q across scenes — within-scene ranking only (the cross-scene delta was the misread in the first forensic report).

**[M] 3. Curves** (screenshots fine): `critic_loss`, `q_spread`, `q_mean_ext`, `expected_int_mean`, `eps_mean`, `wm_loss`, `invariance` over updates; eval history (return/length/achievement count) for both policies over steps.

**[M] 4. Config confirmation**: print the full config dict used, and explicitly confirm these values: `critic_intrinsic_weight=0.0, nstep=5, actor_intrinsic_weight=0.1, eps_base=0.05, eps_novelty_coef=0.20, eps_max=0.50, milestone_t1_priority=0.25, milestone_t2_priority=0.80, update_every=2, batch_size=128, action_repeat=1, no_resume=True`, and a NEW checkpoint directory (must not contain the old fossil `state.pt`).

**[M] 5. Test status**: one line — `python3 -m unittest discover tests` result and `python3 -m src.test_smoke` result.

**[O] 6. Funnel re-run** (same queries as the pre-fix run, for comparability):
- place_table successes (`act=8 ∧ rew≥0.5`) in T1 — expect ≥ the pre-fix 15 if exploration held.
- Lives containing a table success / total episodes (clean count).
- 2–3 success-context traces (last 30 actions before each success).
- NOTE: `act=5 ∧ rew≥0.5` counts ALL do-achievements (wood + drink + cow + sapling + kills) — label it "do-achievements", not "wood chops".

**[O] 7. Decision**: do NOT proceed to Stage B (100k resume) unless report items 1–5 pass. If any fail, stop and send the report as-is.
