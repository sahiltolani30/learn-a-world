# PARALLEL_ENV_PLAN.md — 8-Game Parallel Collection for the Crafter JEPA Agent

**Version:** 1.0 — 2026-10-31
**Audience:** implementing agent. Follow it literally. If a detail seems ambiguous, choose the simplest option that satisfies the acceptance tests in §7.
**Companion to:** `ROOT_CAUSE_FIX_PLAN.md` (that plan's C1–C7 remain untouched — this plan changes ONLY how experience is collected, never how learning works).

**Zero-cheat rule (unchanged):** every game runs the same `ZeroCheatWrapper`. The agent still sees only pixels, actions, and reward. Parallel play changes *how many* games run at once — nothing about what the agent is allowed to know.

---

## 0. Goal and expected numbers

The serial loop collects ~15–17 steps/sec with the GPU idle most of the time (it processes one 64×64 frame per call). This plan collects **8 games simultaneously in the main process and batches all neural-network calls**, so the GPU processes 8 screens per forward instead of 1.

| Metric | Serial (today) | Parallel target (free Colab, 2 vCPUs, N=8) |
|---|---|---|
| Collection speed | ~15–17 SPS | **≥ 40 SPS** (stretch 80) |
| GPU utilization | mostly idle | batched — 8 frames per forward |
| 100k steps wall-clock | ~1.5–2 h | ~25–40 min |
| 1M steps wall-clock | ~17–20 h | ~4–7 h |

On Colab Pro (8+ vCPUs) raise `num_envs` to 16–32 behind the same interface (§8 Phase 2).

**What this plan does NOT do:** no changes to networks, losses, buffer logic, milestone logic, curiosity, or reward semantics. Collection only. Learning results must match the serial run statistically (§7 test protocol).

---

## 1. Design choice: in-process lockstep batching (NOT worker threads, NOT processes in v1)

**Decision:** implement a `ParallelCrafterBatch` class that owns **8 `crafter.Env` + `ZeroCheatWrapper` instances inside the main process** and steps them **sequentially in a loop**, but performs **all neural-network calls batched** (one `encoder(obs_batch)` of shape (8,3,64,64) instead of 8 calls of (1,3,64,64)).

**Why this and not worker processes (yet):**
1. `crafter.Env` is not thread-safe — rules out threads.
2. Worker processes add IPC, pickling, and process-lifetime failure modes — the top source of silent Colab crashes — while the CPU gain on a 2-vCPU free Colab is small (the envs are cheap; the GPU calls were the bottleneck).
3. Sequential env stepping of 8 envs costs ~3–5 ms/iteration; the batched NN call ~10–15 ms. The GPU batching is where the speedup lives, and it needs no processes at all.
4. RAM story is simplest: one process, one buffer, no duplicated runtimes.

**Phase 2 (optional, later, Colab Pro):** swap the internals of `ParallelCrafterBatch` for worker **processes** (one `multiprocessing.Process` per env sending uint8 frames over a `multiprocessing.Pipe`) behind the **same interface** — §8. Do not build Phase 2 until Phase 1 passes its acceptance tests.

---

## 2. RAM budget — the crash-safety contract (read before coding)

Colab free-tier system RAM ≈ **12.7 GB**. Budget for the parallel build at full buffer:

| Component | RAM | Notes |
|---|---|---|
| Tier-1 buffer (50k transitions × (obs + next_obs) × 12,288 B uint8) | **1.23 GB** | the largest consumer |
| Tier-2 buffer (20k × 2 frames) | **0.49 GB** | |
| 8 × crafter.Env (world map + objects + render canvas) | ~0.15–0.25 GB | each env is tiny |
| Python + PyTorch runtime | ~1.5–2.0 GB | |
| Model/optimizer CPU copies | ~0.2–0.4 GB | |
| **Checkpoint-save spike** (contiguous copies of buffer arrays) | **+1.0–1.7 GB transient** | **the real crash risk — see R2** |
| **Total peak** | **≈ 5–6.5 GB** | ~50% headroom — safe **if the rules below are followed** |

**RAM rules (mandatory):**
- **R1 — Buffer never goes on GPU.** Buffer arrays stay `uint8` on CPU. Sampling converts only the drawn batch.
- **R2 — Checkpoint spike control.** `state_dict()` slices like `self.t1_obs[:self.t1_size]` allocate a full copy at save time. To cap the spike:
  - Save **model/optimizer/metadata checkpoints every `save_interval`** (small, ~100–200 MB).
  - Save the **buffer arrays only every 5th checkpoint** (`buffer_save_every: 5`), because the buffer is reproducible — losing it costs replay history, not the agent.
  - After every save: `gc.collect()`, and if CUDA was touched: `torch.cuda.empty_cache()`.
- **R3 — RAM telemetry.** Add `psutil.virtual_memory().percent` to the log line every `log_interval`. **If RAM > 80%: save an emergency checkpoint and abort with a clear message.** Never continue into the OOM zone.
- **R4 — Tier-1 capacity at scale.** For runs > 100k steps, set `tier1_capacity = 50_000` (do not raise it — T1 at 100k capacity would be 2.5 GB). The FIFO eviction is by design.
- **R5 — Action-repeat synergy.** When Stage C enables `action_repeat=2`, buffer fills at half the rate per wall-step — RAM growth halves. Keep in mind, nothing to implement.
- **R6 — One buffer, shared.** All 8 games write into the SAME two-tier buffer. Do not create per-env buffers.

---

## 3. Implementation spec

### 3.1 New class: `ParallelCrafterBatch` (new file `src/env/parallel.py`)

```python
class ParallelCrafterBatch:
    """N Crafter environments stepped in lockstep inside the main process.
    All NN calls happen OUTSIDE this class, batched by the training loop."""
    def __init__(self, num_envs: int = 8, base_seed: int = 42):
        # env i gets seed = base_seed + 1000*i  (deterministic, documented)
        self.envs = [make_crafter_env(seed=base_seed + 1000*i) for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs = np.stack([_to_uint8(self.envs[i].reset()) for i in range(num_envs)])  # (N,3,64,64) uint8
        self.episode_step = [0] * num_envs        # per-env step counter (for episode_id/step_in_ep)
        self.episode_id = [i * (-1) for i in range(num_envs)]  # assign real ids from the pipeline on first step

    def step(self, actions: List[int]):
        """Step all envs. Returns per-env tuples. Auto-resets finished envs."""
        results = []
        for i, env in enumerate(self.envs):
            nxt, rew, term, trunc, info = env.step(actions[i])
            done = term or trunc
            if done:
                nxt = env.reset()           # auto-reset; done=True marks the transition terminal
                self.episode_step[i] = 0
            else:
                self.episode_step[i] += 1
            results.append((nxt, rew, done, info))
        return results
```

Conventions:
- Observations cross the boundary as **uint8 (3,64,64) numpy**; the training loop converts to float32/255 **in the batched tensor call only**.
- `info` passes through `ZeroCheatWrapper` sanitization exactly as today (it is already inside `make_crafter_env`).
- A `done` slot returns the *new* episode's first observation with `done=True` — the training loop stores the transition as terminal and starts the new episode from that obs.

### 3.2 Training-loop changes (`src/train.py`)

1. **Construction:** `self.envs = ParallelCrafterBatch(num_envs=self.num_envs, base_seed=self.seed)`; `num_envs` from config (`--num_envs`, default 8). Keep `self.eval_env` exactly as today (single env, unchanged).
2. **Batched acting (replaces per-step `agent.act`):**
```python
obs_batch = torch.from_numpy(self.envs.obs).to(self.device).float() / 255.0   # (N,3,64,64)
with torch.no_grad():
    z = self.agent.encoder(obs_batch)                                          # ONE forward for N frames
    if self.current_step < self.warmup_steps: action_list = [env.action_space.sample() for _ in range(N)]
    else:
        # ε-greedy per env (C4.2): per-env novelty → per-env eps (keep existing per-env episodic memories)
        probs = self.agent.actor.get_probs(z)
        actions_t = torch.multinomial(probs, 1).squeeze(-1)
        action_list = [int(a) for a in actions_t]
        # then per env: if rand < eps_i: action_i = env.action_space.sample()
```
3. **Batched curiosity:** one `self.agent.predictor(z, actions_tensor)` call of batch N → per-env `r_int` (keep `update_stats=True` semantics).
4. **Buffer writes:** loop the N results and call `self.buffer.add(...)` per env with that env's `episode_id`/`step_in_ep` (C3 metadata) and per-env milestone logic:
   - keep **one `recent_t1_indices` deque (maxlen=30) PER ENV** (8 deques); on a milestone in env i, flag env i's deque only.
   - per-env `episode_id` counters: `self.env_episode_counter[env_i] += 1` on that env's done.
   - per-env achievement tracking (`episode_achievements[env_i]` set), `success_counts` shared, `all_unlocked_achievements` shared.
   - per-env episodic-novelty memory (C4.2): `self.episodic_novelty[env_i]`, reset on that env's done; the Welford *normalizer* stays shared.
5. **Update cadence:** `update_every` now counts **vector steps** (one vector step = N env transitions). With `num_envs=8, update_every=2, batch_size=64`: gradient update every 16 env transitions. Document this in the docstring.
6. **Eval:** unchanged — single env, greedy + stochastic, `eval_interval` in env steps as today.
7. **Telemetry:** log `SPS` as **N × vector-steps/sec** (label it "env SPS"), plus vector-steps/sec separately; log `psutil` RAM% (rule R3) and GPU memory if easy.
8. **Checkpoint/resume:** component names unchanged (`agent`, `buffer`, optimizers) — the parallel pipeline **loads the serial-run checkpoint directly** (`CheckpointManager.restore`, `no_resume=False`). Add `num_envs` to checkpoint metadata; a checkpoint saved under different `num_envs` is still loadable (env count is not part of any state dict).

### 3.3 Config keys (new)

| Key | Default | Meaning |
|---|---|---|
| `num_envs` | 8 | parallel games; set to 1 to fall back to (almost) serial behavior for debugging |
| `buffer_save_every` | 5 | save buffer arrays only every Nth checkpoint (RAM rule R2) |
| `ram_abort_percent` | 80 | emergency-save + abort threshold (rule R3) |

Everything else identical to the Stage-B config (nstep=5, milestone 0.25/0.80, eps 0.05/0.20/0.50, actor_intrinsic_weight=0.1, critic_intrinsic_weight=0.0, batch_size=64, update_every=2, action_repeat=1).

### 3.4 What must NOT change
- Networks, losses, VICReg, curiosity engine, buffer logic, milestone logic, reward semantics, ZeroCheatWrapper.
- The checkpoint format (so serial↔parallel checkpoints are interchangeable).
- Eval protocol and achievement tracking.
- No new reward terms, no per-env reward differences, no seed tricks beyond documented env seeding.

---

## 4. Known pitfalls (read before implementing)

1. **Per-env state is where bugs live.** Eight episode counters, eight achievement sets, eight novelty memories, eight trailing-window deques. Any of them shared by mistake = silent learning corruption. Write per-env unit tests (§7).
2. **`done` slots return the NEW episode's obs.** The stored transition must keep the OLD obs as `next_obs` and mark `done=True`; the new obs becomes the next transition's `obs`. Get this wrong and you leak terminal-state credits across episodes.
3. **Do not batch across a `done`.** If env i finished this vector step, its slot in the next batched forward is the reset obs — fine — but its n-step/milestone bookkeeping must not mix episodes.
4. **`eval_info` is per-env:** aggregate achievement discovery prints across envs; `all_unlocked_achievements` stays global (unique set).
5. **Warmup:** 100 warmup *vector steps* = 800 env transitions of random play (fine — document the change).
6. **Drive save stall:** with saves every 1000 vector steps (= 8000 env steps) the save cadence is rare enough; keep an eye on SPS dips (expected, harmless).
7. **Determinism:** results will NOT bitwise-match the serial run (batched forwards, different env seeding). That is expected — §7 compares distributions, not exact values.

---

## 5. Acceptance tests (add to `tests/test_parallel_env.py`)

1. **Shape/wiring:** one `ParallelCrafterBatch(num_envs=8).step([...])` returns 8 valid tuples; a done env auto-resets and reports `done=True` exactly once at the boundary.
2. **Terminal integrity:** craft a scripted env sequence where env i ends; assert the stored transition's `next_obs` equals the pre-reset frame and the following transition's `obs` equals the reset frame.
3. **Per-env bookkeeping:** two envs finish episodes at different times → their `episode_id`s advance independently; milestone flagging in env 0 does not touch env 1's window.
4. **Batched == serial (math):** with identical weights and forced-greedy actions, one batched `encoder(obs_batch)` output must equal 8 single `encoder(obs_i)` outputs within float tolerance (allclose atol=1e-5).
5. **RAM rule check:** after a simulated save with a full 50k buffer, peak extra RAM (measured via `psutil` around the save call) ≤ 2.0 GB; save with `buffer_save_every=5` skips buffer arrays on 4 of 5 saves.
6. **Resume interop:** save a checkpoint from the serial pipeline, restore into the parallel pipeline (`CheckpointManager.restore`) — keys all match, training proceeds.
7. **Full suite:** `python3 -m unittest discover tests` — all existing tests pass (update only tests that assume a single-env pipeline).

---

## 6. Rollout protocol

1. Implement → run unit tests → run `python3 -m src.test_smoke`.
2. **25k-step validation run** with `num_envs=8` resuming the Stage-B checkpoint (`no_resume=False`): compare against the serial Stage-B report on DISTRIBUTION metrics: unique achievements (±1), table-lives rate (±5 pp), critic-loss range, q_spread range, no NaN, SPS ≥ 40.
3. Only after (2) passes → Stage C long run on the parallel pipeline (1M+, `action_repeat=2` if SPS < 60).
4. If (2) fails: the bug is in this plan's plumbing, not in the agent — do NOT change learning code to compensate; report the diff.

---

## 7. Phase 2 (optional, later — Colab Pro / 8+ vCPUs)

Replace the internals of `ParallelCrafterBatch.step` with one `multiprocessing.Process` per env (worker loop: `recv actions → env.step → send uint8 frame + scalars` over `multiprocessing.Pipe`; recreate env on `done` inside the worker). Interface and main-loop code unchanged. Only do this if Phase 1 measures SPS < 40 and CPU is saturated — measure first.

---

## 8. Do NOT implement in this pass
- Dream training / reward+termination heads / imagination rollouts (separate plan, after Stage B report).
- Spatial latents / masked prediction (separate plan).
- Changes to milestone thresholds, ε schedule, or any C1–C7 parameter.
- Distributed/multi-GPU anything.
