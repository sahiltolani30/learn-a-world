# W1B_LAST_CHANCE_PLAN.md — The Three Unpulled Levers + The Gate

**Version:** 1.0 — 2026-11-02
**Context:** the v6 25k run applied the W1a escalation (variance floor 0.10, recon 1.0) and the canonical scene probe STILL read 0.998–1.00. The pre-registered decision tree says SAPNA takes over. This document is the one remaining fix round on the v3 lineage — **three levers that were identified by measurement but never pulled** — with a hard gate and an honest handover if it fails.

**Project law (unchanged):** pixels, actions, environment reward. No privileged state, no reward edits, no per-achievement scripts.

---

## 1. WHY THIS IS A REAL LAST CHANCE, NOT A STALL

The v6 run measured something important that the failure narrative missed:

- **Per-token scene std: max 0.21** (was 0.035 pre-W1a, 0.0143 at v5). Roughly **8 token positions now carry STRONG scene signal** — the encoding is no longer fully collapsed.
- **But the pooled/readout cosines are still ~0.998** — because mean pooling averages those 8 strong tokens with **56 static grass tokens**, drowning the signal 7:1.

**So the information EXISTS in the token grid and is destroyed at the READOUT.** That is a pooling problem — and pooling was never fixed: W1a-3 replaced a collapsed learned query with plain mean, which trades one dilution for another. Three levers target the measured chain, and none has ever been tried on this architecture:

| Lever | Targets | Precedent |
|---|---|---|
| **L-A: Content-addressed pooling** | The measured 8-vs-56 drowning | Content-addressed attention (the query = the data itself, not a learned vector) |
| **L-B: Deconv reconstruction decoder** | The weak linear anchor (blurry ghosts — the mind viewer showed it) | EMERALD's autoencoder-grounded latents (the Crafter SOTA's core choice); DreamerV3's kept decoder |
| **L-C: GRU sequence training (BPTT-lite)** | The frozen memory notebook (MATH_AUDIT: zero gradients by construction — the aux loss trains it stepwise but never through time) | Dreamer's sequence training over imagined/real trajectories |

---

## 2. THE THREE LEVERS — EXACT SPECS

### L-A — Content-addressed pooling (replaces mean pooling)
The pooling weight for each token = **its own dynamic content**, not a learned query and not a uniform average:
```python
# tokens: (B, 64, token_dim)
attn_scores = tokens.norm(dim=-1) / math.sqrt(token_dim)        # (B, 64) — dynamic tiles have higher norms
attn_w = torch.softmax(attn_scores, dim=-1)                     # (B, 64)
pooled_token = (attn_w.unsqueeze(-1) * tokens).sum(dim=1)       # (B, token_dim) content-weighted
pooled = self.pooled_proj(pooled_token)                         # → LayerNorm → 256-d (unchanged)
```
**Why this and not the learned query (v1) or mean (W1a-3):** the learned query collapsed onto static regions (v1 failure); mean drowns dynamic tiles 56:7 (measured). The token **norm** is a data-driven, training-free attention score — dynamic tiles (player, tree, HUD digits, new objects) measurably carry higher-norm features (std 0.21 vs 0.01 — the norm IS the dynamic signal we measured). The strongest tiles get the weight by construction.
**Acceptance:** after 200 updates, the pooled cosine between two DIFFERENT scenes (one with table in view, one without) must drop below the token-grid cosine (i.e., pooling must not add invariance). Plus: the pooled vector of a frame WITH a table must differ from WITHOUT by ≥ the token-grid difference (pooling preserves, not dilutes).

### L-B — Deconv reconstruction decoder (replaces the linear recon head)
The linear per-token decoder (64 → 192) can only produce blurry patch averages — the mind viewer's ghost pictures ARE its ceiling. A real decoder forces content:
```python
# in the encoder:
self.recon_decoder = nn.Sequential(
    nn.Conv2d(token_dim, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),
)   # tokens (B,64,64) -> reshape (B,64,8,8) -> Conv2d operates per-tile -> PixelShuffle-free assembly
```
Assembly identical to the current `decode_tokens` (per-tile convs are position-shared — valid). Loss: `MSE(recon, batch["obs"])`, weight **1.0** (unchanged). Params: ~50k.
**Why it matters:** the linear recon was the best *linear* readout of the tokens — the mind viewer's ghosts are its ceiling. A conv decoder reconstructs sprite shapes and digit strokes, forcing the tokens to carry them.
**Acceptance:** the V1 mind-viewer decode must show the player sprite and ≥1 HUD item recognizable at 25k (human eyeball pass, saved as PNG).

### L-C — GRU BPTT-lite (the notebook learns through time)
The stepwise aux loss (h predicts the next pooled) trains the GRU one step at a time on stale stored h. The real fix: train **through sampled sequences**:
- In `update_world_model`: for each batch item with valid chain metadata, walk up to **8 consecutive slots** (the `sample_nstep` chain-walk logic, reused), run `z_i = encoder(obs_i)` (no_grad, cached) and `h_i = GRU(z_i, h_{i-1})` **with grad**, predicting each next pooled: `L_seq = Σ_t MSE(mem_to_pooled(h_{t+1}), pooled_target_{t+1}.detach())` over the chain, normalized per chain.
- Chain discovery: reuse the `sample_nstep` validity walk (episode_id + step_in_ep consecutive, done-terminated). Batch: 8 chains × 8 steps per update.
- **Acceptance:** `gru.weight_hh.grad` non-zero; the h-trajectory smoothness stays ≥ 0.95 while `h`'s mutual information proxy (next-pooled prediction R² on held-out chains) improves vs the frozen baseline.

---

## 3. WHAT STAYS UNCHANGED
Everything else in v3.2-film: FiLM conditioning, (S,A)-novelty key 273-d, boredom ladder + canary + persistence, milestone replay, symlog, n-step, RND distillation, the memory loss from MATH_AUDIT (subsumed by L-C — keep one, remove the stepwise duplicate to avoid double-counting), ε schedule, dual eval, all instruments.

---

## 4. THE GATE — pre-registered, no moving target

**At 25k on the W1B run:**
| Gate | Threshold | Verdict if failed |
|---|---|---|
| **Canonical scene-cosines (4 rendered scenes, token grid + pooled)** | **mean ≤ 0.95** (stretch 0.90) | The architecture is done — SAPNA takes over, per its own plan |
| **place_table-lives** | ≥ 10% and repeating (success_counts) | — |
| **make_wood_pickaxe successes** | ≥ 3 in the run | — |
| **Mind viewer V1** | player + ≥1 HUD item recognizable | — |
| **Drift H=10** | ≥ 0.60 (re-earned partially is acceptable; full ≥0.80 unlocks U4 later) | — |

**If the canonical gate fails: SAPNA takes over.** The handover package: this entire document set (ROADMAP, MATH_AUDIT, MIND_VIEWER, all plans), the verified harness (parallel envs, buffer safety, milestone replay, boredom thermostat, instruments), the complete failure map (every generation's measured cause), and the design requirement SAPNA must not repeat: **the token representation must be anchored by a perception loss strong enough to survive a static-dominated world** — reconstruction-grounded latents (EMERALD-style) are the proven form.

---

## 5. RUN PROTOCOL
1. Implement L-A + L-B + L-C (smallest diff: pooling swap, decoder swap, one chain-training block).
2. Bump arch tag: `"v3.2-w1b"`. Fresh weights, **fresh buffer** (the v5/v6 diary is the old policy's loop — low value), new dir `checkpoints_v7`.
3. 25k validation → the gate table above → report with the mind viewer images.
4. If green → extend to 100k (same lineage), then Stage C protocol.
5. If the canonical gate fails at 25k → hand over to SAPNA with everything documented.

**The honest odds, stated once:** L-A and L-B target the two measured failure points directly (the drowning and the weak anchor), and both are the field-proven patterns on this exact benchmark. That makes this a real chance — not a coin flip. But it is the last one on this architecture: if the eyes still read 0.95+ after the anchors are strengthened, the conclusion is that the ResNetTiny trunk itself is the limit, and SAPNA's architecture is the answer. The data will decide, and the gate is written down before the run.
