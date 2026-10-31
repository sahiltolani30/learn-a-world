# UPGRADE_1_2_3_SPEC.md — MAP + MEMORY + STRONGER VALUES (+ Blocking Fixes)

**Version:** 1.0 — 2026-10-31
**Audience:** implementing agent. Follow literally; choose the simplest option satisfying the acceptance tests. **Naming is frozen** — these are the "Upgrades 1–3" of `ROADMAP.md` (read it first for order and gates).
**Companion docs:** `ROOT_CAUSE_FIX_PLAN.md` (C1–C7, done), `PARALLEL_ENV_PLAN.md` (8-env pipeline, done & running), `ROADMAP.md` (master order).

**Project law (unchanged):** agent learns only from raw pixels, actions, environment reward. No inventory access, no action masking from privileged state, no reward hand-editing, no per-achievement scripts.

---

## 0. Context (why — measured evidence)

Stage B (100k, 8-parallel) verified: 12/22 achievements, place_table in eval (79k), pickaxe ×92, survival-300 first crossings, imagination cosine 0.99@1→0.96@5 (drift retreating at H≤5). Remaining measured blockers:
1. **Buffer data loss at resume** (blocking bug — see P0.1).
2. **Travel gate failed** — `collect_stone` never; radius ≤12 tiles → ε raise (P0.5).
3. **World-model drift at H>5** — flat 256-d latent, full-vector MSE lets errors smear globally → U1.
4. **Off-screen amnesia** — table forgotten when scrolled out (87.5% drop-off history) → U2.
5. **Consistency gap** — skills at 2–14% in eval → U3 (c)+(d).
6. Instrument debt: probe verdict logic, success-count contamination, benchmark N, H=10 gap, loss_history cap → P0 fixes.

---

## 1. P0 — Blocking fixes (implement FIRST, before anything else)

### P0.1 Buffer-resume data loss (BLOCKING)
**Evidence:** log at step 44,800 showed `Buf: 982 (T1:800, T2:182)` = exactly 100 vec-steps × 8 envs since the step-44,000 resume → the buffer was **empty at resume**; ~44k transitions of prior experience were silently lost.
**Fix (all three):**
1. `state.pt` (the auto-resume file) **must always include the full buffer arrays.** Remove `buffer_save_every` gating for `state.pt` only; the numbered `checkpoint_step_N.pt` files may keep the every-5th rule for size.
2. **Resume assertion:** in the resume path, after restore: `if not no_resume and buffer.t1_size == 0 and metadata["step"] > 5_000: raise RuntimeError("Resumed step X but buffer is empty — state.pt is missing buffer arrays. Aborting to protect the lineage.")`
3. **Roundtrip test:** create buffer with 10,000 known transitions → save via CheckpointManager → restore into a fresh instance → assert `t1_size`, `t2_size`, `t1_ptr`, and a checksum of `t1_obs[:100]` all match.
**Also:** `buffer_save_every` semantics must guarantee the **numbered** checkpoint at every multiple of `buffer_save_every × save_interval` contains buffer arrays.

### P0.2 Probe verdict logic (diagnostics)
In `diagnostics.py` counterfactual section: **delete the ΔQ verdict** (cross-scene absolute comparison is invalid — it has produced backwards conclusions twice). Verdict = **within-scene comparison only**: report `rank of place_table` and `policy prob` for scene (b) and scene (c) separately; PASS if `rank(b) ≤ 3 AND prob(b) ≥ 0.5 AND rank(c) > rank(b) + 3`.

### P0.3 Achievement-verified success counting
Replace the `ext_reward ≥ 0.5` counter (it attributes wake_up's automatic +1 to whatever action was in flight — fake entries like `make_iron_sword: 45`). New rule in `collect_step`: when the `eval_info.unlocked_achievements` set contains an achievement **not seen earlier in this episode**, increment `success_counts[achievement_name]` and record `success_actions[achievement_name] = action_name` of the current step. Successes are counted **per achievement name**, never per action+reward.

### P0.4 Benchmark standardization
Both policies run **N=50 games**; report both scores side by side; state N in the header (the "20 vs 50" mismatch must not recur).

### P0.5 Travel lever (pre-registered)
`eps_base: 0.05 → 0.10` (keep `eps_novelty_coef=0.2`, `eps_max=0.5`). Travel gate failed at Stage B (`collect_stone` never). This is the only exploration change.

### P0.6 Telemetry
`loss_history` appended **every update** (fix the 500-entry cap); drift test extended to **H=10** in the standard report; keep GPS + cause-of-death + counterfactual probes; benchmark N=50 per P0.4; **add the stage-wise policy probe** (fresh-spawn / wood=2 / facing-table / pickaxe+facing-stone — the four canonical crafting stages) to the standing report: it tracks the wood-tech chain end-to-end and proved at 100k that the mining affordance is pre-wired (Stage 4: `do` 76.8% facing stone) while `collect_stone`=0 — isolating travel as the sole stone blocker.

---

## 2. Data policy: what carries over, what restarts

- **Network weights: RESTART.** U1 changes the encoder output (256-d vector → 64-token grid); old weights cannot load. Fresh initialization, documented.
- **Replay buffer: CARRY OVER.** Frames are architecture-independent. Load `checkpoint_step_100000.pt`'s buffer (T1 50,000 + T2 ~13,074) into the new pipeline **after P0.1's roundtrip test passes**. If verification fails → fresh empty buffer is accepted (log it clearly); at ~66 env-SPS refilling 63k transitions costs ~16 minutes — the loss is recoverable, but carrying preserves the rare-event milestones (tables, pickaxes), which are the valuable part.
- **Zero-cheat state:** unchanged; buffer contents are raw frames + scalars only.
- Per-env transient state (GRU hidden states, episodic novelty memories, episode counters) **resets at resume** — documented, acceptable.

---

## 3. U1 — MAP (spatial tokens + masked prediction)

**The design in one sentence:** the encoder stops flattening its internal 8×8 grid; the world model predicts *masked tiles of the next frame's grid*, and everything downstream (policy, critic, curiosity) reads a pooled view so the rest of the codebase is untouched.

### 3.1 Encoder change (`src/models/encoder.py`)
- Keep the trunk through `stage3` (B,64,8,8) — unchanged.
- **Replace** `spatial_conv → flatten → proj → LayerNorm` with:
  - `token_proj: Conv2d(64, 64, kernel_size=1)` → tokens `(B, 64, 64)` (64 tokens × dim 64), each token = one tile's description. **No flatten.**
  - `pool: nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)` with a learned query (1×64) → pooled vector (B,64); then `pooled_proj: Linear(64, 256)` + LayerNorm → the **same 256-d interface** the rest of the codebase consumes.
- **Interface contract (do not break):** `encoder(x)` still returns `(B, 256)` pooled; add `encoder.forward_tokens(x)` returning `(B, 64, 64)` tokens + `(B, 256)` pooled. Everything downstream (actor, critic, curiosity, planner, diagnostics) keeps working on the pooled 256-d vector.

### 3.2 Target encoder
- Same architecture (EMA copy, τ=0.99 — unchanged mechanism). `forward_tokens` on the target encoder provides the **answer key**: the true next-frame tokens.

### 3.3 Predictor change (`src/models/predictor.py`)
- New predictor: input = current tokens (B,64,64) + action embedding (32) + memory vector (U2, below).
- Architecture: **2-layer transformer encoder** (d_model=64, 4 heads, ff=256, pre-LN) over the 64 token positions, with the action embedding + memory injected as **two extra input tokens** (67 total positions). Output head: per-position Linear(64→64) → predicted next-frame tokens.
- **Masking:** per sample, uniformly mask **35% of the 64 positions** (≈22 tokens, resampled every update). Loss only on masked positions: MSE between predicted and EMA-target tokens of the next frame, **averaged over masked positions** (normalize per-sample, then batch-mean).
- Keep the **4-head ensemble**: 4 independent output heads (as today) → disagreement computed on the **pooled** prediction of masked positions (mean over masked positions → 64-d → variance across 4 heads) — preserves the existing `EpistemicCuriosityEngine` semantics (it consumes `(K, B, D)` all_preds; feed it the pooled masked-position prediction, D=64).
- Param budget: ≤ 500k new params. Keep total model < 3M params.

### 3.4 Losses (world-model optimizer)
- `L_tokens` = masked-prediction MSE (U1, weight 1.0).
- `L_pool` = existing VICReg on the pooled 256-d of predicted-vs-target next frame (weight 1.0 — unchanged anti-collapse duty).
- `L_reward`, `L_done` = U3 heads (below).
- Total WM loss = 1.0·L_tokens + 1.0·L_pool + 0.5·L_reward + 0.5·L_done + (existing variance/covariance terms on the pooled slot, unchanged).
- Curiosity: **unchanged mechanism** — pooled 4-head disagreement → Welford → ε and actor term. Nothing new to tune.

### 3.5 Why masked prediction (for the record)
Local per-tile errors cannot smear across the scene (the drift-wall fix); the action-conditioned hidden-tile game forces object-level rules (tree+do→grass+wood); the HUD tokens force inventory-count features to survive (wood/stone counters get their own slots).

---

## 4. U2 — MEMORY (the notebook)

- **GRU core:** `nn.GRUCell(input_size=256, hidden_size=256)`. Input = pooled latent z_t; hidden state h_t carried per env.
- **Injection:** h_t enters the predictor as the extra memory token (§3.3) AND is concatenated to the pooled vector before policy/critic readout: `readout = concat(pooled_256, h_256)` → 512-d → a single Linear(512→256)+LayerNorm restores the 256-d interface for ActorPrior/TwinCritic/curiosity. (Downstream code untouched.)
- **Per-env hidden states:** 8 GRU hidden vectors (one per parallel env), updated every vector step: `h_i = GRUCell(z_i, h_i)`. **Reset h_i to zeros on that env's `done`.** Reset all at resume (documented — transient state).
- This is the minimal mechanism that lets "my table is 4 tiles behind me" persist while the tile is off-screen.

---

## 5. U3 — STRONGER VALUES

### 5.1 Reward + termination heads
- Two small MLP heads on the pooled readout (input 256, hidden 128): `reward_head` → scalar prediction of next-step external reward (MSE, masked to real reward); `done_head` → termination logit (BCE). Both trained on the WM optimizer every update. Auxiliary weight 0.5 each. They shape the representation toward reward-relevant features now and are the prerequisites for U4 later.

### 5.2 Symlog value targets (DreamerV3 trick, MSE version)
- Critic targets transformed: `symlog(x) = sign(x) · ln(1 + |x|)`. The critic predicts `symlog(Q)`; MSE against `symlog(target_q)`; invert (`sign(y)·(e^|y| − 1)`) only when Q values are consumed by the actor/planner/eval.
- Two-hot bucketing: **optional stretch goal** (only if validation shows value instability with symlog-MSE).

### 5.3 Self-imitation on milestone windows
- For batch items where `is_milestone=True` AND the transition has verified achievement reward (P0.3 counting): add `L_SI = −mean(log π(a|z))` over those items, weight **0.1**, added to the actor loss. Cap: skip if a batch has > 32 milestone items (avoid dominating). This converts the 29-tables/92-pickaxes history into direct policy pressure.

### 5.4 Exploration
- `eps_base = 0.10` (P0.5). Everything else unchanged.

---

## 6. New config keys

| Key | Default | Meaning |
|---|---|---|
| `token_dim` | 64 | per-tile latent dimension (U1) |
| `mask_ratio` | 0.35 | fraction of next-frame tokens hidden per sample (U1) |
| `mem_hidden` | 256 | GRU hidden size (U2) |
| `si_weight` | 0.1 | self-imitation loss weight (U3) |
| `symlog_values` | True | symlog critic targets (U3) |
| `eps_base` | **0.10** | raised (P0.5) |
| `buffer_save_every` | removed for state.pt | P0.1 |

All must flow through `parse_args()` and the dict-config path.

---

## 7. Known pitfalls

1. **Interface freeze:** ActorPrior, TwinCritic, curiosity engine, planner, and `diagnostics.py` all consume a 256-d pooled vector — the pooled readout MUST keep returning 256-d or the whole codebase breaks. The token grid is internal to the world model.
2. **Mask resampling:** masks must be resampled every update (fixed masks → the model only learns fixed tiles).
3. **Done-slot handling in the parallel loop** (from PARALLEL_ENV_PLAN §4) — unchanged, still the #1 silent-corruption risk; the GRU reset must key off the same per-env `done`.
4. **Buffer carry-over must pass P0.1's roundtrip test before training starts** — otherwise run with a fresh buffer and log it.
5. **Checkpoint incompatibility:** fresh weights by design; the buffer is the only carried component. Version the checkpoint (`arch: "v3-spatial"`) and refuse to load v2 weights silently.
6. **Curiosity scale shift:** pooled disagreement now comes from token predictions (D=64 pooled) — expect different raw magnitudes; the Welford normalizer adapts, but watch the first 5k steps for runaway ε.
7. **Do not** add pixel decoders, do not raise mask_ratio above 0.5, do not touch reward semantics.

---

## 8. Acceptance tests (`tests/test_upgrade_123.py`)

1. **Interface:** `encoder(x)` returns (B,256); `forward_tokens` returns (B,64,64)+(B,256); old consumers run unchanged.
2. **Masked loss:** with a fixed random seed, masking produces exactly ⌊0.35·64⌋ masked positions; loss computed only on them.
3. **Memory:** GRU reset on done; two envs' hidden states stay independent in the parallel loop.
4. **Symlog roundtrip:** `inv(symlog(x)) == x` for x ∈ {−0.1, 0, +1, −3.7, 50}.
5. **Self-imitation:** milestone items contribute the BC term; non-milestone items contribute zero; batch cap works.
6. **Success counting:** a synthetic step where `wake_up` fires while action=`make_iron_sword` counts `wake_up += 1` and `make_iron_sword += 0`.
7. **P0.1 roundtrip:** full save/load preserves `t1_size`, `t2_size`, pointer, checksum (per P0.1).
8. **Resume assertion:** resuming step >5,000 with an empty buffer raises the RuntimeError.
9. Full suite: `python3 -m unittest discover tests` green; `python3 -m src.test_smoke` green.

---

## 9. Rollout protocol

1. **P0 fixes first** → unit tests → smoke test.
2. **Fresh training run** (`no_resume=True`, new dir `checkpoints_v3`), **buffer carried from `checkpoint_step_100000.pt`** (after roundtrip verification), `num_envs=8`, `eps_base=0.10`.
3. **25k validation:** the standard gates + drift test H=1..10 + travel watch (`collect_stone` headline).
4. **Gate check:** if travel opened (stone collected) and drift cosine at H=5 ≥ 0.9 → Stage C 1M+ immediately (this model). If drift at H=5 < 0.8 → U4 stays locked; report.
5. Stage C → re-measure drift → U4 decision per ROADMAP Phase 5.

## 10. Do NOT implement in this pass
- U4 dreaming (gated). Pixel decoders. SIGReg. Search/MCTS. Multi-agent. Worldgen changes. Reward edits. Any privileged-state feature.
