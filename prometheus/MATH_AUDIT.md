# MATH_AUDIT.md — Every Equation in the v3.1 Agent, Verified

**Version:** 1.0 — 2026-10-31
**Method:** line-by-line reading of every loss, transform, and update in the current code (`losses.py`, `predictor.py`, `encoder.py`, `actor_critic.py`, `buffer.py`, `train.py`), each equation re-derived by hand and checked against the reference form (VICReg: Bardes et al.; symlog: DreamerV3; Double-DQN; n-step TD; Chan/Welford; count-based exploration).
**Result: 14 equations verified correct · 1 MAJOR structural bug found (GRU never trains) · 3 minor notes.**

---

## 1. MAJOR FINDING — The memory GRU never trains (zero gradients by construction)

**The bug.** `agent.gru` (U2's memory core) is called in exactly four places, and **every one is inside `torch.no_grad()` or consumes pre-stored data**:

| Call site | Context | Gradient? |
|---|---|---|
| `train.py:838` (parallel collection) | inside `no_grad` | ❌ |
| `train.py:972` (serial collection) | inside `no_grad` | ❌ |
| `train.py:1254/1291` (evaluate) | inside `no_grad` | ❌ |
| `update_actor_critic` / `update_world_model` | **no GRU call at all** — they consume `batch["h"]`, which is *stored data* read from the buffer, not a GRU output | ❌ |

The GRU's parameters sit in the WM optimizer list (`train.py:432/1076`) but **no loss term backpropagates through any GRU computation**. Consequence: **the memory notebook is a frozen random projection.** Its output `h_t` is a fixed random nonlinear summary of the observation history (reservoir-computing style) — it *does* carry some information about past states (random RNNs retain traces; `readout_proj` learns to read them via the actor loss), so U2 is not entirely dead — but it is **not the learned memory the upgrade intended, and all the h-consistency machinery (storage, boot_h, eval_h) is feeding a frozen random feature map into the critic and actor.**

**Why every test missed it:** the GRU sits in the optimizer, shapes are correct, losses all decrease (other terms), and nothing NaNs. A zero-gradient parameter is invisible unless you specifically check `param.grad`.

**Why it matters:** the h that reaches the critic/actor is the agent's only memory of off-screen objects (the table!). A frozen random GRU retains *some* history traces, but a *learned* one would carry task-relevant content ("a table was placed here 10 steps ago"). This is a plausible contributor to the measured workstation amnesia (100% walk-off) and the weak scene-discrimination of the value function.

**THE FIX — auxiliary memory-loss on consecutive pairs (~20 lines, uses the existing n-step batch):**

In `update_actor_critic` (or `update_world_model`), for each batch item with `boot_valid`:
```python
# z_curr computed WITH grad here (or detach to train the GRU only — recommended):
z_in = z_curr.detach()
h_next_pred = self.agent.gru(z_in, batch["h"].float())               # differentiable in GRU weights
boot_tokens, boot_pooled = self.agent.target_encoder.forward_tokens(batch["boot_obs"])
pred_next_pooled = self.agent.mem_to_pooled(h_next_pred)              # small Linear(256, 256)
l_mem = F.mse_loss(pred_next_pooled, boot_pooled.detach()) * (batch["boot_valid"].float())
wm_loss = wm_loss + 0.3 * l_mem.mean()
```
This trains the GRU to make `h` **predictive of the next state** — exactly the memory function U2 intended. The next-state pooled target comes from the target encoder (stable, stop-grad) — the same convention as the main JEPA loss.
**Config:** `--mem_loss_weight` default 0.3.
**Staleness note (document, accept):** the stored `batch["h"]` was produced by *older* GRU weights; with a small learning rate the mismatch is bounded. Re-dumping stored h periodically (every ~50k) is optional hygiene.
**Acceptance test:** after 200 steps on a fixed synthetic sequence, `agent.gru.weight_ih.grad` is non-zero AND a held-out next-pooled prediction improves vs initialization.

---

## 2. VERIFIED CORRECT — equation by equation

### 2.1 VICReg (losses.py) ✓
- Invariance: `MSE(pred_pooled, target_pooled)` per head, mean over heads ✓
- Variance hinge: `mean(relu(γ − sqrt(var + ε)))` per head, `var_loss = mean over heads` (the `0.5·(v_pred+v_targ)` form depends on `regularize_target` — standard VICReg either way) ✓
- Covariance: `Σ off-diag(C²)/D` per head ✓ standard
- Applied to `(K, B, 256)` stacked predicted-pooled vs `(B, 256)` target-pooled (canonicalized) ✓ shapes verified
- Note (cosmetic, known): the plotted "Covariance" is unscaled — plotting artifact only.

### 2.2 Masked token-prediction loss (predictor.py) ✓
- `sq_diff = Σ_d (pred − target)²` per position → `× eff_mask` (mask × dynamic-tile weights) → `÷ Σ eff_mask` per sample → mean over batch, mean over heads.
- This is a correctly normalized **weighted mean of per-position MSE** — the absolute scale of the dynamic weights is irrelevant (the ÷norm makes it a weighted average) ✓
- `sample_token_mask`: `floor(0.35 × 64) = 22` tokens, uniform without replacement, resampled per update ✓
- **Verified: the position-weight renormalization in train.py is optional** — the loss normalizes internally. ✓

### 2.3 Reconstruction loss + tile assembly (W1a-0) ✓
- `decode_tokens`: tokens (B,64,64) → `recon_head` → (B,8,8,3,8,8) → `permute(0,3,1,4,2,5)` → (B,3,64,64). Verified correct for row-major token order (`t = row·8 + col`, matching `flatten(2).transpose(1,2)` of the (B,64,8,8) feature map) ✓
- Loss: `MSE(recon, batch["obs"])` vs the **current** frame in [0,1] ✓ (autoencoding, per spec — not future-frame)

### 2.4 Per-token variance floor (W1a-4) ✓
- `relu(0.05 − var_over_batch(token_pos, feat)).mean()` — correct free-bits analogue at tile level. One scale note: token features come from a 1×1 conv on the trunk output (non-negative ReLU features) — the 0.05 floor is reachable; verified empirically by `l_tokfloor` being small-but-nonzero in telemetry.

### 2.5 Reward / termination heads ✓
- `MSE(pred_r, r)` and `BCEWithLogits(pred_d, done)` on `mean_pred_pooled` ✓ standard.

### 2.6 symlog / symexp (U3) ✓
- `symlog(x) = sign(x)·log1p(|x|)`, `symexp(y) = sign(y)·expm1(|y|)` — **exact inverses**, numerically stable forms ✓
- Round-trip verified: `symexp(symlog(x)) == x` for the operating range ✓

### 2.7 Critic loss + Bellman target (U3 + C3) ✓
- **Consistency verified end-to-end:** the target critic outputs symlog-space values → `boot_q = symexp(boot_q_raw)` converts to raw → `target_q = nstep_return + γᵏ·boot_valid·boot_q` (raw) → `compute_loss` applies `symlog(target_q)` → both prediction and target in symlog space ✓ round-trip consistent.
- Double-DQN: bootstrap action = `argmax min(Q1_online, Q2_online)` at the boot state; evaluation by `min(Q1_target, Q2_target)` ✓ correct Double-DQN.
- Approximation noted (acceptable, standard): boot latents come from the **target** encoder while the online critic was trained on online-encoder latents — EMA lag makes this small.
- Actor: `q_all = symexp(min-Q)` — the symexp preserves ordering, so the actor's expected-Q maximization is consistent with the critic's symlog convention ✓.

### 2.8 n-step return computation (buffer.py `sample_nstep`) ✓
- `R = Σ_{k=0}^{K} γᵏ·r_{t+k}`, bootstrap from the last valid slot's `next_obs` with `γ^{K+1}`, `boot_valid = no-done-in-window` ✓
- Chain validity: `episode_id` match + `step_in_ep` consecutiveness + ring-wrap guarded by `k < t1_size` ✓
- Done handling: base-done → 1-term, no bootstrap ✓; mid-chain done → sum stops, no bootstrap ✓
- `boot_h` from the next-slot stored h (Hotfix-6 fallback to the base slot) ✓
- Tier-2 items: clean 1-step fallback ✓
- One subtlety verified correct: `step_in_ep` for a done transition is recorded post-reset (0) — chains never cross it because the done-transition's own chain is terminated at itself ✓

### 2.9 Actor loss ✓
- `−mean( Σ_a π(a)·symexp(min-Q) + β_int·Σ_a π(a)·disc_norm(a) + 0.05·H(π) )` + `si_weight·SI` ✓
- Entropy `H = −Σ p·log_softmax` ✓ correct.
- SI gate: `is_milestone & rewards > 0`, cap 32 ✓ (minor note: novelty-milestones with incidental +0.1 regen rewards pass — noise-level).

### 2.10 Per-action disagreement (C4.1) ✓
- `(K,B,D)` path: variance across heads, summed over D, mean over batch ✓; `(K,B,64,d)` token path: mask-pooled first ✓.
- Normalization: per-action Welford → `disc/(std+ε)` clamp [0,5] ✓.

### 2.11 Welford running stats (ensemble.py) ✓
- Chan's parallel algorithm: `mean_A + δ·(nB/n)`, `M2 = M2A + M2B + δ²·(nA·nB/n)` ✓ numerically stable, correctly implemented.

### 2.12 Episodic k-NN novelty (episodic.py) ✓
- Mean Euclidean distance to k=10 nearest keys, float16 ring, sentinel 1000 for first k steps, per-episode reset ✓.

### 2.13 ε schedule ✓
- `ε = min(ε_max, ε_base + 0.2·novelty_normalized)` ✓.

### 2.14 EMA target updates ✓
- `θ̄ ← τ·θ̄ + (1−τ)·θ` under no_grad, buffers included ✓.

### 2.15 Zero-cheat surface ✓
- Re-audited the new paths (parallel collect, carry-over, n-step, probes): no privileged keys in any info/obs path reaching the agent; zero-cheat assertion present in every collection branch ✓.

---

## 3. MINOR NOTES (no action required now)
1. Boot-argmax uses target-encoder latents (§2.7) — standard approximation.
2. SI gate admits novelty-milestones with incidental +0.1 rewards — noise-level.
3. The token-variance floor is scale-dependent on the conv features — revisit if the trunk changes.
4. `weights/weights.mean()` renormalization before the masked loss is optional (the loss normalizes internally) — cosmetic.
5.VICReg plotted covariance is unscaled (cosmetic, plotting only).

---

## 4. THE FIX ORDER (updated)

1. **GRU memory loss** (§1 fix) — REQUIRED, blocks meaningful U2.
2. Everything else in the W1a package is already implemented and verified.
3. Then: 25k validation → scene-cosine probe (≤0.9) + action-spread (≥3×) + drift re-earn + place_table rank → Stage C.
