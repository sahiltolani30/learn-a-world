# SAPNA — Spatial Anticipatory Predictive Neural Agent
## Complete A-to-Z Implementation Plan (v1, target: official Crafter score ≥ 10–15)

**सपना** = Hindi for *"dream."* SAPNA learns to act by dreaming — training its policy inside
imagined rollouts of its own learned world model, the one organ shared by every top Crafter agent.

---

## 0. MISSION & HARD CONSTRAINTS

**Mission.** Build a NEW model, `sapna/`, as a self-contained package inside this repo. Target:
**official Crafter geometric score ≥ 10% (stretch 15%) at 1.1M env steps** — the same budget as
DreamerV3's official Crafter run (which scores 19.4 in its fixed paper version; 14.5 in v1).
Current v3 lineage plateaus at ~3% — SAPNA must break that by adding the three missing champion
organs: **sequence-trained memory, imagination-based policy training, two-hot value regression**.

**Non-negotiable constraints (violating any = implementation rejected):**
1. **ZERO CHEAT / ZERO HARDCODING.** The agent sees pixels only. No privileged state
   (inventory/semantic map/player pos) may reach the encoder, critic, actor, buffer, or reward.
   No per-achievement logic, no scripted behaviors, no action masking from state.
   The ONLY gray zone allowed (already established practice in this repo): milestone flags for
   *replay data selection* read from `eval_info.achievement_count` — logging/data-selection only.
2. **DO NOT TOUCH the existing system.** No edits to `src/` (the v3-spatial run is in flight),
   no edits to `notebooks/train_colab.ipynb`, no deletion of `checkpoints_v2`, `checkpoints_v3`,
   or the Drive folders of other models (`learn-a-world/prometheus/`, etc.).
   SAPNA is a NEW parallel package: `sapna/` + `notebooks/train_sapna_colab.ipynb`.
3. **NO IMPROVISATION.** Every hyperparameter, shape, formula, gate, and failure response is in
   this document. If something is genuinely unspecified or a gate fails in a way not covered by
   §13 (failure playbook), STOP and write a report. Do not invent config changes.
4. Document precedence inside this repo: `SAPNA_PLAN.md` governs `sapna/`; `ROADMAP.md` governs
   `src/`. They must not cross-reference.

**Authoritative sources used for this design** (the implementing agent does not need to re-read
them; they are listed for provenance):
- DreamerV3 official config (github.com/danijar/dreamerv3, `dreamerv3/configs.yaml`, crafter
  section: steps 1.1e6, envs 1, train_ratio 512, batch 16×64, imag_length 15, symexp_twohot
  bins 255, unimix 0.01, free_nats 1.0, loss_scales rew/con 1.0, imag_loss {lam 0.95,
  actent 3e-4}, shared lr 4e-5) and DreamerV3 paper v2 (arXiv 2301.04104: Crafter 19.4 @1M).
- EMERALD, "Accurate and Efficient World Modeling with Masked Latent Prediction"
  (Burchi & Timofte, ICML 2025, arXiv 2507.04075): current Crafter SOTA, 58.1 @10M, first to
  beat human (50.5). Paper Table 10: B=16, T=64, 16 envs, LR WM 1e-4 / actor-critic 3e-5,
  H=15, λ=0.95, γ=0.997, entropy 3e-4, critic EMA 0.98, spatial latent 4×4×32×32, TSSM 4×512,
  MaskGIT 2×256, S=3 decode steps; symlog two-hot reward head; Bernoulli continue head.
- Δ-IRIS (arXiv 2406.19320, Crafter Table 9: 9.3 @1M, 42.5 @10M; imagination H=15, REINFORCE
  + value baseline over λ-returns, imagined:collected ratio 64).
- IRIS (Micheli et al., ICLR 2023) — the exception that proves the rule: behavior-cloned
  transformer policy, no imagination.
- STORM (Zhang et al., NeurIPS 2023) — stochastic transformer world model, Dreamer-style
  imagination training, parameter sharing.

---

## 1. PATTERN ANALYSIS — what the top 4 champions share

| Organ | DreamerV3 | EMERALD (SOTA) | Δ-IRIS | SAPNA takes it? |
|---|---|---|---|---|
| Spatial/token latents (not flat vector) | ✗ (flat RSSM) | ✓ 4×4×32×32 | ✓ tokens | ✓ ours (v3-spatial tokens) |
| Temporal memory **trained on sequences** | ✓ BPTT | ✓ TBTT, T=64 | ✓ | ✓ **NEW for us (T=16)** |
| Reward + termination heads in latent space | ✓ | ✓ | ✓ | ✓ (v3 has them; kept) |
| **symlog two-hot** regression (reward AND value) | ✓ 255 bins | ✓ 255 bins | — | ✓ **NEW for us** |
| **Policy trained in imagination** (H=15, λ-returns) | ✓ REINFORCE | ✓ REINFORCE | ✓ REINFORCE | ✓ **NEW for us — the key organ** |
| Parallel envs + high train ratio | ✓ ratio 512 | ✓ 16 envs | ✓ ratio 64 | ✓ 8 envs (Colab budget) |

**The single universal organ is imagination training.** IRIS, the one top agent without it,
trains a return-conditioned behavior transformer instead — a different policy organ, not more
model-free TD. Every model-based Crafter champion since DreamerV2 trains the policy on dreamed
latent rollouts with λ-returns. Our v3 lineage has NONE of this — `grep imagination src/train.py`
returns nothing; the critic is n-step TD on real replay. That is the structural reason for the
3% plateau: no mechanism bootstraps value across the 60–90-step crafting chains, and one real
experience yields exactly one gradient lesson instead of ~15.

**What SAPNA deliberately does NOT copy:**
- Categorical/VAE latents (DreamerV2/V3/EMERALD all use them). Our continuous-token JEPA
  perception with VICReg + EMA target is **already validated** by our own probes (wood count
  99–100%, facing 92.8%) and our masked-prediction loss trains stably. Switching latent type is
  a full surgery with uncertain payoff at our scale. SAPNA keeps continuous tokens. Documented
  risk: continuous latents are less battle-tested for imagination — mitigated by §13 gates.
- Pixel decoder / reconstruction (EMERALD and Δ-IRIS both have one). Anti-JEPA by definition;
  efficiency ablations attribute the wins to the latent-space agent, not the decoder.
- MCTS/reanalysis (EfficientZero). Highest ceiling, unaffordable on Colab.
- DreamerV3's RSSM KL machinery (free bits, KL balancing) — N/A without stochastic latents.

**SAPNA's own identity** (why it is a synthesis, not a copy): JEPA token perception (ours) +
champion training recipe (DreamerV3/EMERALD pattern) + milestone-prioritized sequence replay
(ours, Curious-Replay-flavored) + per-frame spatial masked prediction (ours, v3) instead of
MaskGIT-refined categorical decoding.

---

## 2. ARCHITECTURE SPEC (exact shapes — implement verbatim)

Total parameter budget: **1.5M–4M** (asserted in smoke test). Input pixels: `(B,3,64,64)`
float32 in [0,1] (same normalization as `src/env/wrapper.py` — copy that wrapper verbatim).

### 2.1 Perception — `sapna/encoder.py`
Copy **verbatim** the v3 encoder (from `src/models/encoder.py`, class `ObservationEncoder`,
current main branch) into `sapna/encoder.py`, changing only the module/import path:
- Conv stem: 3× (stride-2 conv 4×4 + ResBlock), channels (32, 64, 64) → feature map (B,64,8,8).
- `token_proj`: 1×1 conv 64→`token_dim=64` → **64 spatial tokens of dim 64**, shape (B,64,64).
- Pooling (W1a-3, as the current code actually is): **mean-pool over the 64 tokens** →
  `pooled_proj: Linear(64,256)` → `LayerNorm(256)` → **pooled z (B,256)**. (NOT attention
  pooling — the W1A scene-collapse fix replaced it; copy the class as it exists today.)
- Public interface (frozen, matches the copied class exactly):
  - `forward_tokens(x) -> (tokens (B,64,64), pooled (B,256))`
  - `forward(x) -> (B,256)`
  - `pool_tokens(tokens (B,64,64)) -> (B,256)`  (used on *predicted* grids in L_pool and in
    imagination)
  - `decode_tokens(tokens (B,64,64)) -> (B,3,64,64)`  (the W1a-0 recon anchor's light decoder
    — comes with the copy; see the WM loss block)
- `TargetEncoder`: `copy.deepcopy(encoder)`, EMA `tau=0.99` per optimizer step, permanent
  `.eval()` + `torch.no_grad()` (identical to v3 `src/models/target.py` semantics).

### 2.2 Dynamics & heads — `sapna/dynamics.py`
- `ActionEmbedding(num_actions=17, action_dim=32)` — copy verbatim from
  `src/models/predictor.py`.
- `TemporalCore`: `nn.GRUCell(input_size=288, hidden_size=256)`.
  Recurrence (single source of truth, used identically in training / collection / imagination):
  ```
  h_{t+1} = GRUCell( concat[z_t (256), a_emb_t (32)] , h_t )     # 288 → 256
  ```
  `h` is the memory BEFORE the transition; heads and policy read `concat[z_t, h_t]` (512-d).
- `SpatialPredictor` — copy verbatim v3's `SpatialTransformerPredictor`
  (`src/models/predictor.py`) and use its ACTUAL interface (do not redesign it):
  - `forward_tokens(tokens (B,64,64), a (B,) int64, h (B,256) | None) -> (mean_pred (B,64,64),
    all_preds (4,B,64,64))`. Pass **discrete action ids**, never pre-embedded — the class
    owns its `ActionEmbedding`/`action_proj` internally. Always pass the real `h` (the
    class's `h=None` fallback to learned `default_mem` must never be used by SAPNA).
  - The class ALWAYS returns full-grid predictions; masking exists only inside
    `compute_masked_loss(all_pred_tokens, target_tokens, mask, position_weights=None)`.
    Training calls it with `mask = sample_token_mask(batch, num_tokens=64, mask_ratio=0.35)`
    → exactly `floor(0.35·64)=22` masked positions, resampled per optimizer update as
    **per-row iid masks** over the `B*(T−1)` flattened steps (shape `(B*(T−1), 64)` —
    matching `sample_token_mask(batch_size=…, …)` and the §4 block exactly).
  - `position_weights=None` in SAPNA v1 (v3's W1a-2 dynamic-tile weighting is deliberately
    not carried over; the optional argument stays available in the copied class).
  - Imagination uses the SAME `forward_tokens` call — the full-grid `mean_pred` is the
    one-shot decode (EMERALD's 3-step MaskGIT refinement is explicitly deferred).
  - `compute_epistemic_disagreement(all_preds (4,B,64,64), mask (B,64)) -> (B,)` comes with
    the copy and produces the stored-but-unrewarded `r_int`.
- `RewardHead`: `Linear(512,256) → ELU → Linear(256,255)`. Input = the **ARRIVAL state**
  `concat[pool_tokens(mean_pred_t), h_{t+1}]` (post-action: the predicted next pooled state
  + the post-step memory). It predicts the distribution of `r_{t+1}` — the reward of
  arriving at that state. Symlog two-hot (§5.3). This is DreamerV3's actual semantics and it
  makes immediate reward ACTION-CONDITIONED: different `a_t` → different predicted arrival
  state → different priced reward. (v1.0 read the head on the pre-action `concat[z_t, h_t]`
  with no action input — that prices all 17 actions identically and blunts the key organ;
  fixed in v1.2 after independent review.)
- `DoneHead`: `Linear(512,256) → ELU → Linear(256,1)`. Same ARRIVAL-state input; predicts
  Bernoulli termination of arriving at `(z_{t+1}, h_{t+1})`.
- `Actor` (`sapna/policy.py`): `Linear(512,256) → ELU → Linear(256,256) → ELU → Linear(256,17)`;
  orthogonal init, final layer gain **0.01** (≈uniform at init = built-in exploration warmup).
- `Critic` (`sapna/policy.py`): `Linear(512,256) → ELU → Linear(256,256) → ELU → Linear(256,255)`
  (distributional, §5.3).

### 2.3 Posterior / prior split (the RSSM-like core, in one paragraph)
On real data, `z_{t+1}` comes from the **online encoder on the real next frame** ("posterior").
In imagination, `z_{t+1}` is the **pooled full-token prediction** ("prior"):
`z_pred_{t+1} = encoder.pool_tokens(mean_pred)`, where `mean_pred` is the first return of
`SpatialPredictor.forward_tokens(tokens_t, a_t, h_t)`. The same `pool_tokens` is used in the
training `L_pool` (per-head) and in imagination — no new parameters. This posterior/prior
split is the exact
role DreamerV3's RSSM plays; ours is built from JEPA tokens instead of categorical latents.

### 2.4 Memory-state convention (read twice — this is where implementations go wrong)
- `h` is per-env, `torch.zeros(256)`, reset on `done=True` and at episode start.
- TRAINING (sequence unroll, T=16): `h_0 = 0`; for t = 0..T−1:
  `h_{t+1} = GRUCell(concat[z_t, a_emb_t], h_t)` where `z_t` is the online-encoder pooled
  vector of the REAL frame at t. Gradients flow through the whole unroll (TBPTT within the
  window; no detach inside the window — this is what trains the memory).
- COLLECTION: maintain `env_hidden_states` tensor `(num_envs, 256)` exactly like v3
  (`train.py` env_hidden_states); update with the same recurrence using real z each step;
  reset the slot on done. **Eval must do the same per-episode** (see §9 — Hotfix-4 lesson).
- IMAGINATION: identical recurrence, but `z` is the predicted pooled vector (prior, §2.3).

### 2.5 Deliberate deviations from the copied v3 code (read before touching anything)
1. **Reward/Done head wiring differs from v3 on purpose.** v3 reads them on the predicted
   next pooled state alone, with plain MSE (`train.py:1052-1055`). SAPNA reads them on the
   **ARRIVAL state** `concat[pool_tokens(mean_pred), h_next]` — DreamerV3's actual
   arrival-state semantics, which makes immediate reward action-conditioned — and upgrades
   reward to symexp_twohot CE. (v1.0–v1.1 of this plan wrongly specified pre-action inputs
   `concat[z_t, h_t]`; corrected in v1.2 after independent review.) Do NOT copy v3's head
   wiring either.
2. `position_weights=None` (v3's W1a-2 dynamic tile weighting is not carried over).
3. Everything else in the WM loss block — masking, per-head VICReg, token variance floor
   (0.05, weight 1.0), 0.3 recon anchor, var/cov hinge — is copied to mirror
   `train.py:1025-1067` exactly.
4. Sequence training (T=16 BPTT) replaces v3's 1-step batch; the §4 block is the sequence
   form of the same losses. GRU h starts at 0 per window (TBPTT-stateless, like
   DreamerV3's sequence resets) — a documented approximation.

---

## 3. EXACT HYPERPARAMETERS (single source of truth — `sapna/config.py`)

Tag legend: **[D3]** = DreamerV3 official config, **[EM]** = EMERALD paper Table 10,
**[V3]** = our verified v3 value, **[OURS]** = our choice with reason.

| Name | Value | Tag | Reason |
|---|---|---|---|
| num_envs | 8 | OURS | user requirement; Colab free tier ~2 vCPU safe |
| action_repeat | 2 | OURS | halves collection cost + buffer growth; DreamerV3-official uses 1 but with train_ratio 512 on A100 — we are compute-poor |
| frame_stack | 1 | [D3][EM] | none of the champions stack |
| seq_len T | 16 | OURS | EMERALD uses 64; 16 is what a T4 affords; ≥8 required for memory BPTT to matter |
| batch_seqs B | 8 | OURS | B×T = 128 frames per encoder pass |
| token_dim | 64 | [V3] | v3 validated |
| pooled_dim | 256 | [V3] | interface freeze |
| mask_ratio | 0.35 | [V3] | 22 of 64 tokens |
| ema_tau | 0.99 | [V3] | per optimizer step |
| vicreg sim / var / cov | 1.0 / 1.0 / 0.04 | [V3] | var hinge γ=1.0, ε=1e-4 in sqrt |
| reward_bins | 255 | [D3][EM] | symexp_twohot |
| reward_bins_range | ±20 (symlog space) | [D3] | clamp symlog inputs to ±20 |
| imag_horizon H | 15 | [D3][EM] | official champions' value |
| lambda (λ-returns) | 0.95 | [D3][EM] | |
| gamma | 0.997 | [D3][EM] | |
| entropy coeff (actent) | 3e-4 | [D3][EM] | added INSIDE the actor objective (§5.4) |
| unimix (action probs) | 0.01 | [D3] | probs = 0.99·π + 0.01/17, always, env + imagination |
| lr world model | 1e-4 | [EM] | Adam |
| lr actor | 3e-5 | [EM] | Adam |
| lr critic | 3e-5 | [EM] | Adam |
| grad clip | 1.0 (global norm) | [V3] | known-good in this repo |
| update_every | **1** (vec step = 16 raw env steps → train ratio 8 raw / 16 stored) | [AMEND v1.2] | doubles gradient contact per life (v1.0's ratio 4 was mis-derived); pre-registered fallback if SPS < 20 at G1: `update_every=2` + `batch_seqs=16, seq_len=8` (same 128 frames, 256 imagination starts from 16 episodes) |
| warmup_steps | 200 vec-steps random-uniform | OURS | buffer needs valid ≥16-length windows first |
| buffer capacity | 100,000 transitions | OURS | 1.23 GB uint8 — Colab RAM contract (§7) |
| replay sampling | 70% uniform / 30% milestone-window | [V3 C2] | validated mechanism, carried over |
| eps_base (ε-greedy) | **0.0** | [D3][EM] | champions use none; entropy + unimix only. Pre-registered lever: 0→0.05 at G3 branch (§13) |
| curiosity (disagreement) weight in actor | **0.0** (off) | [OURS] | champions show entropy suffices; disagreement still computed + logged (diagnostics + milestone admission). Do not enable without a plan revision |
| self-imitation | none in v1 | OURS | revisit only after G4 report |
| checkpoint interval | 5,000 env steps | [V3] | state.pt ALWAYS includes buffer (P0.1 lesson) |
| eval_interval | 5,000 env steps; 3+3 episodes dual policy | [V3] | |
| benchmark | N=50 both policies | [V3 P0.4] | |

---

## 4. TRAINING LOOP (pseudocode — implement exactly this)

```
init: envs = ParallelCrafterBatch(num_envs=8)   # copied verbatim into sapna/envs.py
      modules per §2; optimizers: opt_wm  Adam(lr 1e-4) on [encoder, SpatialPredictor,
        TemporalCore, mem_proj, RewardHead, DoneHead]   (TargetEncoder excluded)
      opt_actor Adam(3e-5) on [Actor];  opt_critic Adam(3e-5) on [Critic]
      h = zeros(8, 256); warmup: 200 vec-steps uniform-random actions

loop over vec-steps (one vec-step = one action per env, action_repeat applied inside envs):
  collect: probs = unimix(Actor(concat[z_t, h_t]));  a ~ probs;  envs.step(a)  (repeat 2)
           z_{t+1} = Encoder(x_next);  h = GRUCell(concat[z_t, emb(a)], h);  reset slot on done
           buffer.add(obs, a, r_sum_over_repeat, done, episode_id, step_in_ep, milestone_flag)
           milestone_flag = new achievement this episode (per-env set) OR r_int_norm > 0.5
           r_int = predictor.compute_epistemic_disagreement(all_preds_t, mask_t)  # stored, NOT rewarded
  if step < warmup or buffer.valid_windows < 64: continue
  if step % update_every != 0: continue

  # ---- 1) WORLD-MODEL UPDATE (one gradient step; mirrors train.py:1025-1067 exactly) ----
  seq = buffer.sample_sequences(B=8, T=16)          # §7
  # encoder pass: treat (8,16) as 128 independent frames, then reshape back
  toks, z = Encoder(seq.obs.flatten(0,1))           # (128,64,64), (128,256) → (8,16,·)
  h_seq = unroll TemporalCore over T (inputs z, emb(a))   # BPTT, no detach inside window
  mask = sample_token_mask(batch_size=B*(T-1), num_tokens=64, mask_ratio=0.35)  # per update
  for t in 0..T-2:                                  # accumulate over t
      mean_pred_t, all_preds_t = SpatialPredictor.forward_tokens(toks[t], a[t], h[t])
      L_tokens += compute_masked_loss(all_preds_t, TargetEnc.tokens(obs[t+1]), mask[8t:8t+8])
      L_pool   += VICReg(stack([Enc.pool_tokens(all_preds_t[k]) for k in 4 heads]),
                         TargetEnc.pooled(obs[t+1]))["loss"]     # per-head VICReg, v3 wiring
      arr_t     = concat[Enc.pool_tokens(mean_pred_t), h[t+1]]   # ARRIVAL state (post-action)
      L_reward += twohot_ce(RewardHead(arr_t), symlog(clamp(r[t+1], ±20)))
      L_done   += BCEWithLogits(DoneHead(arr_t), done[t+1])
  tok_var    = toks.flatten(0,1).var(dim=0)         # (64,64) per position-feature
  L_tokfloor = relu(0.05 - tok_var).mean()          # W1a-4 token variance floor, weight 1.0
  L_recon    = MSE(Encoder.decode_tokens(toks.flatten(0,1)), seq.obs.flatten(0,1))  # W1a-0
  L_varcov  = var_coeff(1.0)·var_z + cov_coeff(0.04)·cov_z from
               VICReg._compute_var_cov(z.flatten(0,1).unsqueeze(0))   # v3 anti-collapse hinge
  loss_wm = 1.0·mean_t(L_tokens) + 1.0·mean_t(L_pool) + 1.0·L_tokfloor
          + 0.3·L_recon + 1.0·mean_t(L_reward) + 1.0·mean_t(L_done) + L_varcov
  opt_wm.step();  TargetEncoder.ema_update()

  # ---- 2) IMAGINATION UPDATE (one gradient step) ----
  # CRITICAL (DreamerV3 semantics): the TRAJECTORY is computed under torch.no_grad() so no
  # imagination gradient ever reaches opt_wm; actor/critic are then evaluated WITH grad on
  # top of the detached states.
  with torch.no_grad():
      starts = all (z[t], h[t]) posterior states from the SAME batch → N = 8*16 = 128
      tokens_0 = the real frame's token grid of each start (posterior tokens)
      for k in 0..H-1 (H=15):
          probs_k = unimix(Actor(concat[z_k, h_k]));  a_k ~ probs_k
          mean_pred = SpatialPredictor.forward_tokens(tokens_k, a_k, h_k)[0]  # full decode
          tokens_{k+1} = mean_pred;  z_{k+1} = Encoder.pool_tokens(mean_pred)
          h_{k+1} = GRUCell(concat[z_k, emb(a_k)], h_k)
          r_k = symexp(score(RewardHead(concat[z_{k+1}, h_{k+1}])))  # ARRIVAL-state pricing
          d_k = sigmoid(DoneHead(concat[z_{k+1}, h_{k+1}]))          # detached (reward_grad=False)
          v_k = symexp(score(Critic(concat[z_k, h_k])))       # detached (for λ-returns only)
  G = lambda_returns(r, d, v, v_bootstrap; γ=0.997, λ=0.95)   # §5.2, all detached inputs
  # recompute WITH grad on the detached states:
  probs, logprob, entropy = Actor(concat[z_k, h_k])           # z_k, h_k detached tensors
  v_grad = symexp(score(Critic(concat[z_k, h_k])))
  v_bootstrap = symexp(score(Critic(concat[z_H, h_H])))
  advantage = sg(G − v_grad)                                  # sg = detach; see §5.4
  loss_actor  = −mean( logprob(a_k) · advantage_k ) − 3e-4 · mean(entropy)
  loss_critic = twohot_ce(Critic(concat[z_k, h_k]), symlog(clamp(G, ±20)))   # G detached
  opt_actor.step();  opt_critic.step()

  every 5k env steps: checkpoint (§11); every 5k: eval (§9); log telemetry every update (§10)
```

Indexing convention (memorize): `RewardHead(concat[z_t, h_t])` is trained against
`r[t+1]` — the reward received after action `a_t`. The same input predicts the reward of the
imagined action `a_k` taken in state `(z_k, h_k)`. DoneHead identical with `done[t+1]`.
Critic/Actor always read `concat[z, h]` of the state they act FROM.

---

## 5. LOSS FORMULAS (verbatim)

**5.1 Symlog / symexp** (already implemented in `src/agent/actor_critic.py:21-40` — copy):
`symlog(x) = sign(x)·ln(1+|x|)`;  `symexp(y) = sign(y)·(exp(|y|)−1)`.

**5.2 λ-returns** (H=15, γ=0.997, λ=0.95, bootstrap at step H):
```
G_H = v_bootstrap
G_k = r_k + γ·(1−d_k)·[ (1−λ)·v_{k+1} + λ·G_{k+1} ]     for k = H−1 .. 0
```
`r_k, d_k` are the detached **arrival-state** head predictions (the reward / termination of
arriving at imagined state k+1, §2.2); `v_{k+1}` = critic's scalar value at the NEXT
imagined state (detached inside the return computation; gradients reach the critic only
through `loss_critic`). Shapes: all `(N,15)` with `N=128`; bootstrap `v_H` shape `(N,)`.

**5.3 symexp_twohot** (DreamerV3 / EMERALD):
- bins: 255 scalar centers `b_i = symexp(linspace(-20, 20, 255))`.
- target encode: `y = clamp(symlog(x), −20, 20)`; find adjacent bins `i, i+1` around y;
  two-hot weights `(b_{i+1}−y, y−b_i)/(b_{i+1}−b_i)` → target distribution over 255.
- loss: `CrossEntropy(logits, twohot_target)` (i.e., −Σ target·log_softmax(logits)).
- scalar readout: `symexp( Σ softmax(logits)·linspace(−20,20,255) )`.

**5.4 Actor objective (DreamerV3 form):**
`advantage_k = sg(G_k − v_k)` (both sides detached inside the advantage);
`loss_actor = −mean_k( logprob(a_k | probs_k) · advantage_k ) − 3e-4 · mean_k(entropy(probs_k))`.
The entropy term is a SEPARATE additive term with its own gradient — it must NOT be placed
inside the advantage (that would kill its gradient). REINFORCE with value baseline.
No PPO clipping, no DPG — copy the champion form exactly.

**5.5 VICReg** (copy v3 `src/models/losses.py` semantics): sim = MSE on normalized vectors,
coeff 1.0; var hinge coeff 1.0 (γ=1.0, ε=1e-4 inside sqrt); cov coeff 0.04; reduction
`mean_heads`; plus the v3 extra var/cov hinge on the online pooled z. `verify_non_collapse`
(min var > 0.1) MUST be wired as a logged metric and a run-abort check (v3 wires it as a
logged check with strict=False at `train.py:1335`; SAPNA promotes it to a hard abort).

---

## 6. REPLAY BUFFER — `sapna/buffer.py`

Ring buffer, `capacity=100_000` transitions, arrays (pre-allocated, uint8 obs):
`obs (100k,3,64,64) uint8` [1.23 GB], `action int8`, `reward float32` (sum over the 2 repeated
frames), `done bool`, `episode_id int64`, `step_in_ep int64`, `milestone bool`,
`r_int_norm float32` — produced by copying v3's `WelfordNormalizer`
(`src/curiosity/ensemble.py`; disagreement is Welford-normalized online). Milestone admission
(C2 dual trigger): new achievement in this episode's per-env set OR `r_int_norm > 1.0`
(v1.2: raised from 0.5 — a 0.5 z-score threshold flags ~31% of ALL transitions under a
Welford scale and dilutes the milestone pool to near-uniform). **RAM contract:** peak budget = buffer 1.23 GB + T2 none
(SAPNA has NO two-tier system — one pool, the 70/30 priority is at sampling time) + model/opt
~0.1 GB + 8 envs ~0.2 GB + checkpoint-save spike ≤ 0.6 GB → **≤ 2.5 GB** of Colab's 12.7 GB.
Do NOT raise capacity above 100k for v1.

`sample_sequences(B=8, T=16)`: pick 8 start indices; 70% uniform over valid windows, 30% from
windows containing ≥1 milestone transition (fallback to uniform after 100 failed tries). Valid
window: all 16 positions share one `episode_id` AND `step_in_ep` strictly contiguous
(`step_in_ep[t+k] == step_in_ep[t]+k`). Collect per-env episode lengths on add; maintain a
per-episode index (`episode_id → list of start positions with len ≥ T`). If fewer than 64 valid
windows exist, use `T=8` for that batch (same code path, flag `seq_len_actual` in telemetry).
Return dict of stacked tensors on CPU; caller moves to device.

`save_state()` / `load_state()`: ALL arrays included in every checkpoint (P0.1 lesson — the
v2 lineage lost its diary twice; SAPNA will not lose its third). On load: assert
`len(buffer) == expected` and roundtrip-verify 3 random transitions byte-identical, else raise
`RuntimeError("SAPNA buffer roundtrip failed — aborting to protect the diary")`.

---

## 7. COLLECTION & PARALLEL ENVS — `sapna/envs.py`

Copy **verbatim** from `src/env/parallel.py` (`ParallelCrafterBatch`) and `src/env/wrapper.py`
(`ZeroCheatWrapper`, `make_crafter_env`) into `sapna/envs.py`, changing only imports.
`action_repeat=2` is implemented by a thin loop AROUND the copied class (the copied
`ParallelCrafterBatch.step` has no repeat logic — write exactly this, per env slot):
```
def step_with_repeat(i, a):
    r_sum, done, obs, info = 0.0, False, None, None
    for _ in range(action_repeat):                 # 2
        obs, r, done, info = envs.step_single(i, a)   # copied class, un-repeated step
        r_sum += float(r)
        if done: break
    return obs, r_sum, done, info                  # LAST sub-step frame/info; reward summed
```
Keep the copied class's terminal-observation handling untouched; integrate repeat at the
slot level. Note: `step_single(i, a)` is a small, documented ADDITION to the copied class —
the class only exposes the lockstep `step(actions)`; extract/wrap its per-slot step.
The "verbatim" rule applies to everything else in the file. Re-run the existing zero-cheat
leak tests (§11 T11) against the copied wrapper.
8 envs, seeds 0..7, `base_seed=42` semantics as in the copied file.

**Collection policy:** sample from unimixed actor probs. NO ε-greedy in v1, NO curiosity bonus.
Early exploration comes from the near-uniform init (actor final gain 0.01) + entropy reg.
The pre-registered ε lever (0→0.05) exists ONLY in the §13 decision tree.

---

## 8. IMAGINATION MODULE — `sapna/imagination.py`

Functions (pure, testable):
- `twohot_encode(x: (N,) float) -> (N,255) float` (§5.3, clamps inside)
- `twohot_score(logits: (N,255)) -> (N,) float` (scalar readout §5.3)
- `lambda_returns(rewards (N,15), conts (N,15), values (N,15), bootstrap (N,), gamma, lam)
   -> returns (N,15)` — backward recursion per §5.2.
- `imagine_batch(z0 (N,256), h0 (N,256), actor, critic, predictor, temporal_core,
   action_embedding, pool_fn, H=15) -> dict` with keys `actions (N,15) int`,
   `logprobs (N,15)`, `entropies (N,15)`, `values (N,15)`, `bootstrap (N,)`,
   `rewards (N,15)`, `conts (N,15)`, `final_z (N,256)`, `final_h (N,256)`.
   All sampling with `torch.distributions.Categorical` over unimixed probs. Actions stored as
   ints for logprob recomputation consistency (recompute logprob from stored probs — no
   re-sampling). N=128 default. No gradient through reward/cont heads (detach at read).

---

## 9. EVALUATION — `sapna/train.py::evaluate` + `sapna/benchmark.py`

- Every `eval_interval`: 3 greedy + 3 stochastic episodes, each capped at 1000 env steps
  (v3 rule — eval cost grows with survival; the cap counts RAW env steps including action
  repeats). Greedy = argmax of unimixed probs (argmax is
  unchanged by unimix). Stochastic = sample.
- **`SAPNAAgent` facade** (thin class in `sapna/policy.py`): bundles all modules; exposes
  `act(obs, h, greedy=False) -> (action, h_next, z)` and `reset_h()`. Both `evaluate()` and
  `benchmark.py` consume ONLY this facade — one single eval code path (this is the
  structural prevention of a Hotfix-4 repeat).
- **MANDATORY (Hotfix-4 lesson):** maintain `eval_h` per running episode — update with the
  §2.4 recurrence using the real observation each step, reset on done, and pass it to BOTH
  greedy and stochastic action selection. Never call act with default/zero h mid-episode.
  Acceptance test T9 covers this.
- `sapna/benchmark.py`: `run_benchmark(agent, n=50, policy in {greedy, stochastic})` — copy
  `compute_crafter_score` **verbatim** from `src/diagnostics.py:67-77`
  (`score = exp( (1/22)·Σ ln(1+S_i) ) − 1`, S_i in percent). Records per-achievement rates,
  returns, lengths, cause-of-death, movement% (fraction of actions in {move_left, move_right,
  move_up, move_down}, from the action histogram), and unique-achievement counts. NO position
  instrumentation — the crafter engine exposes no player-position attribute and privileged
  reads are banned. Saves JSON to `reports/`.
- Telemetry logged EVERY update (P0.6 lesson — v3 lost trend data to a 500-entry cap):
  `wm_loss, L_tokens, L_pool, L_reward, L_done, var_hinge, cov_term, reward_ce, done_bce,
  actor_loss, critic_loss, q_spread_equivalent (= std of v over states), entropy_mean,
  advantage_std, imagined_reward_mean, real_reward_mean, imagination_reward_corr (every 500
  updates: Pearson r between imagined cumulative and real cumulative rewards on 32 freshly
  sampled real windows — no holdout set needed), milestone_fraction, sps, ram_percent`. Keep full history in the checkpoint
  (no 500-entry cap).

---

## 10. FILES TO CREATE (exact list — nothing else)

```
sapna/
  __init__.py          # version "sapna-v1"
  config.py            # dataclass SAPNAConfig — every row of §3 as a field with that default
  encoder.py           # TokenEncoder + TargetEncoder (§2.1, copied from src/models/encoder.py)
  dynamics.py          # ActionEmbedding, TemporalCore, SpatialPredictor, RewardHead, DoneHead
  policy.py            # Actor, Critic, unimix(), symlog, symexp (copied), SAPNAAgent facade (§9)
  imagination.py       # §8 functions
  buffer.py            # SAPNAReplayBuffer (§6)
  envs.py              # ParallelCrafterBatch + wrapper (copied verbatim, §7)
  train.py             # loop per §4, evaluate per §9, checkpoint/resume per §11, CLI per below
  benchmark.py         # §9
tests/
  test_sapna.py        # §12 acceptance tests (one file, numbered T1..T12)
notebooks/
  train_sapna_colab.ipynb   # §14
SAPNA_PLAN.md          # this document (already at repo root)
```
CLI of `sapna.train` (argparse, all with §3 defaults): `--device --total_steps
--checkpoint_dir --num_envs --batch_seqs --seq_len --action_repeat --update_every
--lr_wm --lr_actor --lr_critic --imag_horizon --mask_ratio --unimix --entropy_coeff
--gamma --lam --buffer_capacity --warmup_steps --eval_interval --save_interval
--milestone_frac --eps_base (default 0.0) --no_resume --seed` .

Checkpoint format: `state.pt` = `{step, config dict, arch: "sapna-v1", module state_dicts
(encoder, target_encoder, predictor, temporal_core, heads, actor, critic), optimizer states,
buffer arrays, Welford/telemetry history}`. Save atomic (tmp file + rename). Sanitize any
`PosixPath` → `str` recursively (Hotfix-3 lesson). Load with `torch.load(...,
weights_only=False)` — own trusted file (Hotfix-1 lesson). Resume fails LOUD on arch mismatch
or buffer roundtrip failure (Hotfix-2 lesson).

---

## 11. ACCEPTANCE TESTS (`tests/test_sapna.py`) — all must pass before any GPU run

- **T1 shapes + identities:** encoder `(2,3,64,64)` → `forward_tokens` returns
  `((2,64,64),(2,256))`; identity `forward(x) == pool_tokens(forward_tokens(x)[0])`
  (allclose 1e-6); predictor `forward_tokens(tokens,(2,),h)` → `mean_pred (2,64,64)`,
  `all_preds (4,2,64,64)`; `compute_masked_loss` returns scalar ≥ 0; heads `(2,255)`/`(2,1)`;
  actor `(2,17)`.
- **T2 param budget:** `1.5e6 < total < 4e6`.
- **T3 symlog/two-hot roundtrip:** `twohot_encode→twohot_score` reconstructs x within 1% for
  x ∈ {−1, −0.1, 0, 0.1, 1, 5, 10}; target rows sum to 1; no NaN at x=±20 (clamped).
- **T4 λ-returns hand example:** rewards `[0,0,1,0,0]`, conts all 1, v=0, bootstrap v=0,
  γ=0.997 λ=0.95 → G_0 ≈ 1·γ²·λ² + γ³·λ³ + ... (compute expected in the test with the same
  formula, assert allclose 1e-5); cont=0 at step 2 cuts all later credit.
- **T5 GRU reset independence:** two envs' h stay independent after 10 steps with different
  actions; reset on done → zeros.
- **T6 sequence sampler validity:** 200 sampled windows from a synthetic buffer with short
  episodes — 100% satisfy same-episode + contiguous `step_in_ep`; milestone-pool windows all
  contain ≥1 milestone.
- **T7 imagination shapes + determinism:** `imagine_batch` with fixed seed → same outputs
  twice; shapes per §8; rewards/conts detached (`.requires_grad == False`), values carry grad.
- **T8 two-hot critic reduces:** 200 gradient steps of Critic on synthetic fixed G targets →
  CE drops >5×; scalar readout error < 0.05; reward-head action-conditioning sanity: two
  different arrival-state inputs (`concat[pool(mean_pred_a), h_a]` vs `…_b`) produce
  different predicted reward distributions.
- **T9 eval-h maintenance (Hotfix 4):** run 20 eval steps — `eval_h` changes across steps,
  resets on done, and is actually passed to act (assert via monkeypatched spy).
- **T10 checkpoint roundtrip:** save → mutate → load → all arrays byte-identical, config
  restored; arch tag mismatch raises.
- **T11 zero-cheat leak test:** copy of the existing wrapper-leak tests — no key from
  `{inventory, achievements, semantic, player_pos}` reachable in obs/step outputs used by the
  agent path; milestone flag consumed only from `eval_info` count.
- **T12 smoke end-to-end (CPU, 200 steps, tiny config):** runs to completion, telemetry has
  all §9 keys, imagination-reward-corr computed at least once, `verify_non_collapse` passes.

---

## 12. COLAB NOTEBOOK — `notebooks/train_sapna_colab.ipynb` (cell-by-cell)

Mirror the existing notebook's style (markdown header per cell, prints, paths from
`pathlib`). Drive root for SAPNA: **`/content/drive/MyDrive/learn-a-world/sapna/`** with
subfolders `checkpoints_v1/`, `reports/`, `plots/` (auto-created, Cell 2). Same dependency
list as the existing Cell 3 (crafter, gymnasium, imageio, matplotlib, psutil).

1. **Cell 1 — Hardware verify** (torch version, CUDA, GPU name, RAM, cpu_count).
2. **Cell 2 — Drive mount + folder creation** (`learn-a-world/sapna/{checkpoints_v1,reports,plots}`).
3. **Cell 3 — Pip installs** (same deps as existing notebook).
4. **Cell 4 — Repo clone + path** — clone `github.com/sahiltolani30/learn-a-world`
   (pull latest), `sys.path.insert` repo root; print `sapna.__version__ == "sapna-v1"`.
5. **Cell 5 — CPU smoke** — run T12 config for 200 steps on CPU; print PASS/FAIL per §11.
6. **Cell 6 — GPU training** — `python -m sapna.train --device cuda
   --checkpoint_dir /content/drive/MyDrive/learn-a-world/sapna/checkpoints_v1
   --total_steps 25000 ...` for the FIRST validation run (extend to 1_100_000 only after G1
   passes, per §13). Resume by re-running the same cell (no `--no_resume`).
7. **Cell 7 — Plots** — loss curves from checkpoint telemetry (wm, reward_ce, done_bce,
   actor, critic, entropy, imagination_reward_corr).
8. **Cell 8 — Gate checklist** — auto-evaluates the §13 gate conditions from telemetry +
   a fresh 25-game benchmark, prints PASS/FAIL per gate with values.
9. **Cell 9 — Stage probe** — the 4 canonical scenes (fresh spawn / wood=2 no table / facing
   table / facing stone): actor probs + critic value per scene (adapted to `concat[z,h]`
   input; scenes constructed with privileged state for DIAGNOSIS ONLY — never fed to training).
10. **Cell 10 — Official benchmark N=50 dual policy** — prints official score both policies,
    per-achievement table, saves JSON to `reports/benchmark_step_<n>.json`.
11. **Cell 11 — Drift & dream forensics** — rollout drift cosine + L2 at H=1..10 vs real
    futures (same methodology as the v3 senior-forensics Item 9, actual taken actions), plus
    imagination-vs-real reward correlation on 32 held-out sequences.

---

## 13. RUN PROTOCOL, GATES, DECISION TREE, FAILURE PLAYBOOK

**Stage G0 — Smoke (CPU, 200 steps).** All §11 tests green. → proceed.
**Stage G1 — Validation run, 25k steps** (`--total_steps 25000`, fresh `checkpoints_v1`,
`--no_resume`). Gates: no NaN; `wm_loss` declining; `reward_ce < 1.5` (init ≈ ln 255 ≈ 5.5);
`done_bce < 0.3`; stochastic eval mean achievements ≥ 2.5; SPS ≥ 20; imagination reward corr
≥ 0.3 — with the v1.2 caveat: if fewer than 10 of the 32 sampled windows contain nonzero
cumulative reward, log "insufficient reward variance" and PASS-with-caveat instead of
stopping; `var_hinge > 0.1` (no collapse). **PASS → extend the SAME run** to 1.1M
(`--total_steps 1100000`, resume). **FAIL → stop, send telemetry, do not tune.**
**Stage G2 — checkpoint @100k (running gate).** Official stochastic score ≥ 2.5%; unique ≥ 9;
collect_wood ≥ 30%; drift cosine @H=5 ≥ 0.8; place_table occurring in training
(milestone counter). Continue regardless to 1.1M unless ≥2 gates red → stop + report.
Drift lever (pre-authorized, v1.2): if drift cosine @H=5 < 0.8 but all other gates are
green → apply the single lever `--imag_horizon 8` and resume (the one authorized response
to deterministic-decode drift; document in the report).
**Stage G3 — checkpoint @300k.** Score ≥ 5%; unique ≥ 11; **collect_stone appeared**.
Branch: if score ≥ 5% but collect_stone = 0 AND benchmark movement% < 30% (mean over the
50 games) → apply the ONE pre-authorized lever `--eps_base 0.05` and resume (document in
report). Anything else → continue.
**Stage G4 — 1.1M (TARGET).** At G4 also run ONE N=50 benchmark with NO 1000-step episode
cap (official-protocol parity — DreamerV3's 19.4 is measured uncapped; v1.2 addition).
Score ≥ 10% = mission accomplished (stretch 15%). Score 8–10%
→ continue same run to 3M (one command). Score < 8% with all gates healthy → continue to 3M
and report at 2M. Score < 5% → STOP, full forensic report (§9 instruments), no config changes.
**Stage G5 — beyond.** Post-G4 upgrades (MaskGIT refinement S=3, T=32, categorical latents
A/B, self-imitation) are OUT OF SCOPE for this plan; require a new revision of this document.

**Failure playbook (symptom → cause → action; nothing else is authorized):**
- NaN in first 5k → check two-hot clamp ±20 present, grad clip 1.0 present, LRs as §3 →
  if all present, stop + report.
- `var_hinge → 0` (collapse) → `verify_non_collapse` abort fires → stop + report. Never
  retune VICReg coefficients.
- Eval return exactly −0.90 with 0 achievements while training progresses → eval-h bug
  (Hotfix-4 fingerprint) → run T9, fix only that, report.
- SPS < 20 → reduce `--seq_len 8` (and only that) → if still < 20, `--update_every 4` →
  report. NEVER shrink encoder/batch below spec, never shrink buffer.
- imagination_reward_corr < 0.1 at G1 → drift pathology → stop + report (do NOT enable
  curiosity, do NOT change H).
- milestone pool empty at 25k → check flag wiring via T6 → report.
- Actor entropy < 0.5 nats before 50k → check unimix actually applied (T7 covers probs) →
  report.
- RAM > 80% → emergency save + abort (psutil check in the loop, v3 R3 rule).

---

## 14. EXPECTATIONS & BUDGET MATH (set the bar honestly)

- DreamerV3 official Crafter: **19.4 @1.1M** (fixed v2 paper; 14.5 in v1) with train_ratio 512
  on A100-class hardware. Δ-IRIS: 9.3 @1M. SAPNA is the compute-poor replica: train_ratio
  ~8 raw (16 stored transitions) per update (v1.2 corrected derivation), ~1.7–2.5M params, T4
  — targeting **10–15% at 1.1M** means "DreamerV3-v1 territory at DreamerV3's own budget".
  Achievable; not guaranteed; gates decide. Single-seed Crafter variance is ±2–4 score
  points — the G4 8%-vs-10% branch sits inside that noise band; when in doubt, continue to
  3M rather than stop.
- Worked example of what score 10 needs: per-achievement rates {wake_up 90, sapling 70,
  place_plant 60, collect_wood 50, collect_drink 40, eat_cow 30, place_table 20,
  collect_stone 10, make_wood_pickaxe 5} → Σln(1+S_i)=2.98 → score = exp(2.98/22)−1 ≈ **14.5%**.
  Nine solid early/mid achievements — no diamond required.
- Wall-clock: at SPS 25–40 (update_every=1 spends more time on gradients; SPS gate is 20),
  1.1M steps ≈ 8–12 h of pure env time; with evals/saves budget ~12–16 h → one Colab
  session with auto-resume, or 2–3 sessions (checkpoint every 5k makes preemption cheap).
  25k validation ≈ 20–30 min.

---

## 15. IMPLEMENTATION ORDER (for the implementing agent — do in this order, test as you go)

1. `sapna/envs.py` (copy) + T11 → 2. `sapna/encoder.py` (copy) + T1/T2 →
3. `sapna/dynamics.py` + `sapna/policy.py` + T1/T3/T5 → 4. `sapna/imagination.py` + T3/T4/T7 →
5. `sapna/buffer.py` + T6 → 6. `sapna/config.py`, `sapna/train.py`, `sapna/benchmark.py` +
T8/T9/T10/T12 → 7. notebook §14 → 8. run G0 → G1 → report.
The whole pass is ~1.5–2 implementation days. When any step is ambiguous, §0.3 applies:
stop and report — never guess.

---

*Design lineage: SAPNA = our JEPA token perception (validated in the v3 lineage) + the
training recipe extracted from the four champion Crafter agents (DreamerV3, EMERALD, Δ-IRIS,
STORM) + milestone-prioritized replay (our C2). No single model was copied wholesale; every
borrowed organ is cited in §0. Zero-cheat discipline per the project's standing constraint.*
