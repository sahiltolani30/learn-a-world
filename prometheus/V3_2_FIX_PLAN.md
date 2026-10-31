# V3_2_FIX_PLAN.md — All Outstanding Issues in the v3.1 Agent + Exact Fixes

**Version:** 1.0 — 2026-10-31
**Scope:** every issue found across the v3.1-W1a run (25k), the diagnostic quad, the P1 pickaxe-chain diagnostics, the mind viewer, and the math audit — consolidated with exact fixes. Nothing here is a guess: each issue cites its measurement; each fix targets the measured mechanism.
**Relation to other documents:** `UPGRADE_1_2_3_SPEC.md` (W1a architecture — implemented & verified), `MATH_AUDIT.md` (equation verification + the GRU finding), `V3_FIX_AND_RUN_PLAN.md` (hotfixes — implemented), `W1A_SCENE_COLLAPSE_FIX.md` (scene-collapse fix — implemented, partially successful), `SAPNA_PLAN.md` (the separate next-generation rebuild — independent, do not mix codebases).
**Project law (unchanged):** pixels, actions, environment reward. No privileged state, no reward hand-editing, no per-achievement scripts.

---

## 0. THE COMPLETE ISSUE LEDGER (all measured, ranked by impact × cost)

| # | Issue | Evidence (measured) | Fix | Cost |
|---|---|---|---|---|
| I1 | Scene-collapse residual: token mean cosine 0.9917 (target ≤0.90); per-token std 0.0143 (6× better than pre-W1a 0.0024 — insufficient); dead tokens (min std 0.0000) | V2 probe on the v5 45k checkpoint | **Escalation**: token_var_floor 0.05→0.10, l_recon 0.3→1.0 (the pre-planned lever) | Config |
| I2 | Predictor action-blindness persists despite W1a-1: 17-action disagreement spread 0.0002 (quad D); near-table probe flat 1.28–1.33 | Quad D + P1.3 | **FiLM action conditioning + disagreement on dynamic positions** (§3.1) | ~30 lines |
| I3 | **GRU memory core never trains** — zero gradients by construction (every GRU call under no_grad; losses consume stored batch["h"]) | MATH_AUDIT §1 (call-site trace) | **Auxiliary memory loss**: h_next = GRU(z, h); predict the boot state's pooled latent (§3.2) | ~20 lines |
| I4 | The actor's exploration term is a CONSTANT: disc_norm pinned at the 5.0 clamp all run (measured flat) | v4 telemetry plot + code (clamp max=5.0) | Recalibrate: remove the hard clamp for the actor term, rely on per-action Welford + a wide safety clamp at 50 (§3.3) | ~3 lines |
| I5 | Untried (state, action) pairs carry zero exploration pressure: `make_wood_pickaxe` tried at a table would be maximally novel, but the novelty key is state-only | Code (EpisodicNovelty keys on readout) + measured make-prob 1.49% at the table with 0 successes in 43k steps | **(S,A)-augmented novelty key** = concat(readout, one-hot(action)) (§3.4) | ~5 lines |
| I6 | Workstation amnesia (partial): 21.8% immediate walk-off; crafting tried in 61% of windows but success needs wood≥1 which exists in only 9.6% of lives | Corrected de-interleaved scan (V1) | Upstream: I1–I5 + steps. No direct fix — wood scarcity is a scene/behavior consequence | — |
| I7 | Greedy eval = statue (50/50 games −0.90; 100% failed-craft spam): argmax over a flat young critic locks onto a permanently-failing action (make_wood_pickaxe with wood=0, seen in the mind viewer) | Mind viewer + 50-game bench | Protocol: judge behavior by stochastic eval until Q differentiates; keep greedy as a long-term tracker. No code change | — |
| I8 | Travel/boredom layer absent: nothing pays for leaving the local zone across episodes; RND implemented but never wired | Cross-run: stone locked in every run ever; 40k steps zero progress in v5 | **RND wiring + BoredomMonitor** (§3.5) | ~80 lines |
| I9 | Minor: eps-plot label ambiguity; SI gate admits incidental-reward milestones; unscaled covariance plot; boot-argmax on target-encoder latents (standard approx) | Various | Batched small fixes (§5) | ~15 lines |

**Explicitly NOT issues (verified):** VICReg forms, symlog round-trip consistency, Double-DQN structure, n-step math, masked-loss weighting math, recon tile assembly, Welford, k-NN novelty, ε schedule, EMA, zero-cheat surface, Hotfix-6 h consistency (all verified in MATH_AUDIT.md).

---

## 1. IMPLEMENTATION ORDER (each step: implement → unit test → full suite → smoke)

**Step 1 (I3):** GRU auxiliary memory loss — MATH_AUDIT.md §1 has the exact spec. ~20 lines.
**Step 2 (I4):** exploration-term recalibration — §3.3. ~3 lines.
**Step 3 (I5):** (S,A)-novelty key — §3.4. ~5 lines.
**Step 4 (I2):** FiLM action conditioning + dynamic-position disagreement — §3.1. ~30 lines.
**Step 5 (I1):** escalation config — token_var_floor 0.10, l_recon weight 1.0 (config only).
**Step 6 (I8):** RND wiring + BoredomMonitor — §3.5. ~80 lines.
**Step 7 (I9):** minor batch — §5.

Then the 25k validation battery (§6) → decision tree (§7) → Stage C 1M (§8).

---

## 3. NEW FIX SPECIFICATIONS

### 3.1 I2 — Make the predictor actually use the action (FiLM + dynamic-position disagreement)

**Why W1a-1 was insufficient:** the additive broadcast `tokens + a_proj(a_emb)` gives every token the SAME constant offset — attention layers can learn to ignore a constant. The disagreement is computed as the variance across heads of the **mean over masked positions** — diluting the few action-sensitive tiles 22-fold.

**FiLM conditioning** (per-token scale+shift — the action modulates every token multiplicatively, which attention cannot ignore):
```python
# in SpatialTransformerPredictor.__init__:
self.film = nn.Linear(action_dim, 2 * token_dim)
# in forward_tokens, after a_proj broadcast:
film = self.film(a_emb)                                  # (B, 2*token_dim)
gamma_f, beta_f = film.chunk(2, dim=-1)                  # (B, token_dim) each
tokens = tokens * (1.0 + gamma_f).unsqueeze(1) + beta_f.unsqueeze(1)
```
**Dynamic-position disagreement:** in `compute_epistemic_disagreement`, when a per-sample dynamic-weight vector is supplied, pool the heads' predictions with those weights (replacing the uniform mean over masked positions):
```python
if dyn_w is not None:   # (B, 64) weights from W1a-2's delta ranking
    w = dyn_w.unsqueeze(0).unsqueeze(-1)                 # (1, B, 64, 1)
    pooled = (all_preds * w).sum(dim=2) / w.sum(dim=2).clamp(min=1e-6)
```
and in the collection/curiosity path, pass the same dynamic weights (computed from the target-vs-current delta, as in W1a-2).

**Acceptance:** with two different actions on the same frame, the per-token prediction delta must be nonzero on **≥ 16 of 64 positions** (was: effectively uniform), and the 17-action disagreement spread on a fixed scene must reach **max/min ≥ 3** (measured 0.0002 spread before).

**⚠️ Architecture bump (mandatory):** FiLM adds new parameters (`film`, and W1a-6's `mem_to_pooled` does too) → **old v3.1-w1a checkpoints cannot load** → fresh weights by design. Bump the checkpoint metadata tag to `"arch": "v3.2-film"` and refuse to load anything else silently (same pattern as the v3.1 handler: legacy detected → weights restart, message printed). **Buffer policy: do NOT carry the v5 buffer** — it is the looping-policy's diary (stationary spam, low value); a fresh buffer is cleaner and refills in ~15 min at 30+ SPS.

**Scope note (where the dynamic-position weighting applies):** the W1a-2 delta ranking needs the NEXT frame's tokens — available in `update_world_model`/`update_actor_critic` (the batch has `next_obs`/`boot_obs`) but NOT at collection time (the prediction happens before stepping). Therefore: the dynamic-position weighting applies to **(a) the WM masked loss and (b) the update-time disagreement** (both have next frames); the **collection-time curiosity keeps uniform pooling** over masked positions (unchanged). Do not invent a next-frame lookup at collection.

### 3.2 I3 — GRU auxiliary memory loss (from MATH_AUDIT §1 — restated exactly)

In `update_actor_critic` (the n-step batch already carries `batch["h"]`, `batch["boot_obs"]`, `batch["boot_valid"]`):
```python
boot_tokens, boot_pooled_tgt = self.agent.target_encoder.forward_tokens(batch["boot_obs"])
h_next = self.agent.gru(z_curr.detach(), batch["h"].float())          # differentiable in GRU weights only
pred_next_pooled = self.agent.mem_to_pooled(h_next)                   # Linear(256, 256), NEW
l_mem = F.mse_loss(pred_next_pooled, boot_pooled_tgt.detach())
l_mem = (l_mem * batch["boot_valid"].float()).mean()
wm_loss = wm_loss + 0.3 * l_mem
```
Add `self.agent.mem_to_pooled = nn.Linear(256, 256)` to the agent + the WM optimizer. **Acceptance:** `agent.gru.weight_ih.grad` non-zero after 200 steps; held-out next-pooled prediction improves vs initialization; no NaN.

### 3.3 I4 — Un-dead the exploration term
`disc_norm = clamp(disc/(std+ε), 0, 50)` — the Welford per-action normalization already bounds the scale statistically; the hard 5.0 clamp was pinning the whole term at its ceiling (measured flat 5.0). Keep a wide safety clamp at 50, monitor `expected_int_mean` — if it trends (not pinned), the term is alive.

### 3.4 I5 — (S,A)-augmented novelty key
In the collection loops (both parallel and serial):
```python
key = np.concatenate([r_i, np.eye(17, dtype=np.float32)[act]])        # 273-d
nov_raw = self.episodic_novelties[env_i].novelty_and_insert(key)      # EpisodicNovelty(latent_dim=273)
```
Every untried (state, action) pair becomes maximally novel → ε spikes → `make_*` gets tried at tables. ~5 lines + `EpisodicNovelty(latent_dim=273)` construction change.

### 3.5 I8 — RND wiring + BoredomMonitor
- **RND:** instantiate `LatentRND(latent_dim=256)`; train its predictor on sampled latents each update (`l_rnd = rnd.compute_loss(z_curr.detach())`, weight 0.2 in the WM loss).
- **⚠️ RND integration point (corrected):** RND novelty is a STATE-level signal — a state-constant added to the actor's `expected_int = Σ_a π(a)·int(a)` contributes **zero gradient** (the constant factors out of the sum over actions). Therefore RND enters through the **ε/behavior channel**, exactly like the episodic novelty:
```python
nov_global = self.rnd_stats.normalize(rnd.compute_novelty(z_curr.detach()), center=False)
eps = min(self.eps_max, self.eps_base
          + self.eps_novelty_coef * nov_episodic
          + 0.10 * nov_global)                     # new lifelong component, weight 0.10
```
  With its own RunningMeanStd (`rnd_stats`). The BoredomMonitor's L2 escalation multiplies the RND contribution (not the actor term). Never put RND into the critic or the env reward (fossil risk — the v2 lesson).
- **BoredomMonitor:** as designed in UPGRADE_1_2_3_SPEC (BOREDOM MODULE section) — BORED when zero first-time achievements for 3 consecutive eval windows; escalation ladder L1–L4; doubles as the representation-disease canary (auto-runs the scene-cosine probe when it fires).
- **Do NOT:** put either signal into the critic or the env reward (fossil risk).

---

## 5. MINOR BATCH (I9)
1. Benchmark plot label: print the policy name and N in the header (done in the 50-game version — keep).
2. SI gate: tighten to `is_milestone & (rewards >= 0.5)` (achievement-verified only).
3. Covariance plot: scale by the loss coefficient (cosmetic).
4. Boot-argmax on target-encoder latents: accepted approximation — document in MATH_AUDIT (done).

---

## 6. THE 25K VALIDATION BATTERY (after Steps 1–5)

Run the master forensic suite on the fresh 25k checkpoint. The gates:

| Gate | Target | Was |
|---|---|---|
| **Scene-cosines (token grid, 4 scenes)** | mean ≤ 0.90 (escalated) | 0.9917 |
| **17-action disagreement spread** | max/min ≥ 3 | 2× |
| **Drift H=1..10** | ≥ 0.8 at H=5–10 (re-earned with sight) | 0.86 @10 (partly vacuous) |
| **Stage probe: scene-sensitivity** | the 4 stage distributions must DIFFER (do% spread ≥ 15pp across scenes) | identical ±3% |
| **place_table rank in scene (b)** | ≤ 5 | #13 |
| **place_table-lives** | ≥ 10% and rising | 2.7% |
| **make_wood_pickaxe successes** | > 0 by 25k | 0 in v5 |
| Dual benchmark N=50 | stochastic ≥ 2%, greedy ≥ 0.5% | 1.39 / 0.00 |
| Hygiene | no NaN, RAM ≤ 40%, SPS ≥ 25 | ✓ |

---

## 6.1 Exact run config for the post-fix 25k validation
```python
training_config = {
    "device": device,
    "total_steps": 25_000,                    # validation first
    "checkpoint_dir": ".../prometheus/checkpoints_v6",   # NEW dir — v5 is the archived statue lineage
    "no_resume": True,                        # fresh weights (arch v3.2-film breaks compat) + FRESH BUFFER (see below)
    "allow_fresh_buffer": True,               # required for this one fresh launch; FLIP TO False after the first checkpoint
    "num_envs": 8, "batch_size": 64, "update_every": 2,
    "eps_base": 0.10, "eps_novelty_coef": 0.20, "eps_max": 0.50,
    "token_var_floor": 0.10,                  # I1 escalation
    "l_recon_weight": 1.0,                    # I1 escalation (recon weight as config)
    "mem_loss_weight": 0.3,                   # I3
    "si_weight": 0.1, "nstep": 5, "symlog_values": True,
    "eval_interval": 1_000, "eval_episodes": 3, "save_interval": 1_000,
    "num_envs_note": "8 parallel; buffer_save_every 5; RAM guard 80%",
}
```
**Buffer policy: FRESH, deliberately.** The v5 buffer is the looping-policy's diary (stationary spam + failed-craft attempts) — carrying it would teach the new brain the old loop. Fresh collection at 30+ SPS refills 25k in ~15 min. The BoredomMonitor state starts clean; `allow_fresh_buffer` flips to False once `checkpoint_step_1000.pt` exists (the run's own diary is then protected by the P0.1 assertion).

## 7. DECISION TREE AT 25K

- **All gates green** → Stage C: extend the SAME lineage to 1M+ (`action_repeat=2` if SPS < 40). The ladder: pickaxe consistency → collect_stone → stone tier.
- **Scene-cosines green but place_table still rank > 10** → the representation sees; the value side lags → run 50k more before judging (check SI counts).
- **Scene-cosines still > 0.95** → the collapse survived the escalation → this is the SAPNA trigger: stop patching, build the next-generation per SAPNA_PLAN.md (sequence-trained memory + imagination training are the organs v3.1 structurally lacks).
- **Training instability** (NaN, sustained critic spikes) → stop, report, reduce l_recon to 0.5.

---

## 8. WHAT NOT TO DO
- No reward hand-editing, no action masking from state, no per-achievement scripts.
- No U4 dreaming before the drift gate is re-earned WITH sight.
- No deletion of checkpoints_v2/v3/v4/v5 (lineage archive).
- No changes to SAPNA_PLAN.md or the sapna/ package (independent track).
- Do not run Stage C before the gates — a 1M-step statue is a week of waste.
