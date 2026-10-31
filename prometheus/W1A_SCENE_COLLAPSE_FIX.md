# W1A_SCENE_COLLAPSE_FIX.md — The Encoder Scene-Collapse Fix + Complete Training Guide

**Version:** 1.0 — 2026-10-31
**Audience:** implementing agent. Everything below is exact — no assumptions needed. If any detail seems ambiguous, choose the simplest option that passes the acceptance tests in §7.
**Supersedes:** the "W1a REVISION ADDENDUM" and "W1a-RE" sections of `UPGRADE_1_2_3_SPEC.md` (same fixes, now fully specified). `ROADMAP.md` remains the master order; this document is Phase-3's fix + the complete post-fix training protocol.
**Project law (unchanged):** pixels, actions, environment reward. No inventory access, no action masking from privileged state, no reward hand-editing, no per-achievement scripts.

---

## 1. THE ISSUE — what is broken, measured, and why

### 1.1 The measured evidence (25k diagnostic quad, verified)

| Measurement | Result | Healthy reference (v2) |
|---|---|---|
| Token-grid cosine between 4 visibly-different scenes (empty grass / wood=2 / facing table / facing stone) | **0.9954–0.9995** | scenes must be separable (≤0.9 target) |
| Per-token std across those scenes | **0.0024 mean, 0.035 max** (only the player-center tokens vary) | substantial variation across scene content |
| Predictor disagreement across all 17 actions, one scene | **0.000187–0.000395, spread 0.0002** | v2 varied ~100× (movement 4–18 vs `do` 0.05) |
| Behavioral gradient | scene-*independent* skills at 99% (sapling 148/149, place_plant 148/149); scene-*dependent* skills collapsed (wood 18%, place_table 2.7%, pickaxe 0%) | v2: wood 70%+, place_table 27% |
| Greedy evaluation | **50/50 games at exactly −0.90, 0 achievements; GPS: 1 square, movement 0%; action pie 100% failed-crafting spam** | v2 greedy: 1.45–1.90% |

### 1.2 The diagnosis (one paragraph)

**The U1 world model is content-free.** The hide-and-guess (masked tile prediction) objective on a grass-dominated screen is ~90% static tiles. The cheapest way to minimize it: the encoder maps *every* frame to nearly the same token set ("average field"), and the transformer predicts "tiles stay roughly the same" while **ignoring the action token** (it is 1 of 66 inputs — attention has no gradient pressure to use it). Both were measured: scene-invariant tokens (quad B) and action-blind predictions (quad D). The EMA target encoder then locks this collapsed representation in as the answer key — a self-consistent, content-free loop. Everything downstream inherits it: the policy and critic read a constant map → they cannot distinguish scenes → the wood chain collapsed (place_table 2.7% vs v2's 27%, pickaxe locked) → greedy evaluation becomes a stationary crafting-spam statue.

**In the literature this is called representation collapse (or posterior collapse).** Every top world-model team hit it and solved it the same way: **separate perception from prediction** — a perception loss (reconstruction / information floor) *forces* the latent to retain scene content; the dynamics loss is then free to compress. Our U1 trained the encoder *only* through prediction — that is the architectural root cause.

### 1.3 What this issue is NOT

- Not the memory notebook (Hotfix 6 verified implemented: `t1_h` stored, actor/critic train with `batch["h"]` — quad A).
- Not the config (verified: eps 0.10, mask 0.35, token_dim 64, mem_hidden 256 — quad C).
- Not the parallel pipeline, buffer, or eval plumbing (all verified).
- Not capacity: the player-center tokens DO vary (std 0.035) — tokenization works. It is a **gradient-pressure** problem, not a size problem.
- Note: the H=10 drift PASS (0.8595) is **partly vacuous** — a content-free model trivially predicts a static field. The gate must be re-earned after this fix.

---

## 2. THE FIX — five changes, exact specifications

All changes are in `src/models/encoder.py`, `src/models/predictor.py`, `src/agent/agent.py`, `src/train.py`. Nothing else changes. All self-supervised (own pixels/actions/rewards) — zero cheat.

### 2.1 W1a-0 — Reconstruction auxiliary head (THE KEYSTONE)

**Purpose:** the perception anchor. You cannot reconstruct the player/table/stone from tokens that don't encode them — this loss makes scene-collapse impossible.

**Add to the encoder** (`ObservationEncoder`):
```python
self.recon_head = nn.Linear(token_dim, 3 * 8 * 8)   # each token decodes its own 8x8 patch
```
**Add method:**
```python
def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
    """tokens (B, 64, token_dim) -> reconstructed frame (B, 3, 64, 64) in [0,1]-scale."""
    B = tokens.size(0)
    patches = self.recon_head(tokens).view(B, 8, 8, 3, 8, 8)
    frame = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 64, 64)
    return frame
```
(Token order is row-major: token t = grid position (t//8, t%8) — matching the existing `flatten(2).transpose(1,2)`.)

**Loss (in `update_world_model`):**
```python
recon = self.agent.encoder.decode_tokens(curr_tokens)          # (B, 3, 64, 64)
l_recon = F.mse_loss(recon, batch["obs"])                       # batch["obs"] is float [0,1]
```
`l_recon` weight **0.3** in the WM loss (§2.6). Params: 64×192 ≈ 12k — negligible.

### 2.2 W1a-1 — Action conditioning on EVERY token

**In `SpatialTransformerPredictor`:** add
```python
self.a_proj = nn.Linear(action_dim, token_dim)
```
In `forward_tokens`, before the transformer: inject the action into every position:
```python
a_tok = self.a_proj(self.action_embedding(a))            # (B, token_dim)
tokens = tokens + a_tok.unsqueeze(1)                     # (B, 64, token_dim) += broadcast
```
Keep the existing extra action+memory tokens as well. The action now modulates all 66 positions.

### 2.3 W1a-2 — Dynamic-tile loss weighting (the missing scene-information gradient)

**In `update_world_model`,** compute per-position change magnitude and upweight the tiles that actually carry dynamics:
```python
with torch.no_grad():
    delta = (target_tokens - curr_tokens).norm(dim=-1)          # (B, 64)
    weights = torch.ones_like(delta)
    topk = torch.topk(delta, k=8, dim=-1).indices                # the 8 most-changing tiles
    weights.scatter_(1, topk, 5.0)
    weights = weights / weights.mean(dim=-1, keepdim=True)       # mean weight = 1 per sample
```
`compute_masked_loss` gains an optional `position_weights` argument: per-position MSE × weights, then mean over masked positions (renormalized). Static-grass tiles keep weight ~1; the 8 dynamic tiles get ~5 — the model is *forced* to learn trees, the player, the HUD, and stone.

### 2.4 W1a-3 — Mean pooling (replace learned-query attention)

In the encoder, replace the `MultiheadAttention`-query pooling with:
```python
pooled_token = tokens.mean(dim=1)                                # (B, token_dim)
```
Then `pooled_proj` + LayerNorm exactly as now. Rationale: the learned query can attend to static regions and wash out the scene; mean pooling preserves every region equally (v2's flatten did this implicitly and its representation was scene-rich). Add `pool_tokens` accordingly (mean-based). Delete or bypass the attention module — do not keep both paths.

### 2.5 W1a-4 — Per-token variance floor (free-bits analogue)

**In `update_world_model`:**
```python
tok_var = curr_tokens.var(dim=0)                                 # (64, token_dim) batch variance per position-feature
l_tokfloor = F.relu(self.token_var_floor - tok_var).mean()       # floor 0.05, weight 1.0
```
Add to the WM loss with weight **1.0**. This is the Dreamer "free bits" analogue at tile level: no token position may collapse to a constant. Config: `--token_var_floor` default 0.05.

### 2.6 W1a-5 — Reward-relevant pressure bump
`l_reward` and `l_done` weights: **0.5 → 1.0** each. They are the only reward-relevant scene pressure the encoder receives; doubled until the representation is healthy.

### 2.7 Updated WM loss (complete)
```python
wm_loss = (1.0 * l_tokens            # masked tile prediction (with W1a-2 weights)
         + 1.0 * l_pool              # VICReg on pooled (unchanged)
         + 0.3 * l_recon             # W1a-0 reconstruction anchor
         + 1.0 * l_tokfloor          # W1a-4 per-token variance floor
         + 1.0 * l_reward            # U3 (bumped)
         + 1.0 * l_done              # U3 (bumped)
         + var_coeff * var_z + cov_coeff * cov_z)   # pooled anti-collapse (unchanged)
```

---

## 3. THE OTHER BLOCKING FIXES (from V3_FIX_AND_RUN_PLAN.md — verify, don't skip)

1. **Hotfix 4 — eval memory:** `evaluate()` must maintain `eval_h` (reset per episode, GRU-updated per step) and pass it to BOTH greedy and stochastic `act()` calls. **Verify it is actually in `train.py::evaluate`** — the 25k run's eval pattern suggests it may still be missing; if missing, evals will read falsely dead.
2. **Hotfix 1 — carry-over loader:** `weights_only=False` for own checkpoints. Needed only when resuming with a carry source; verify present.
3. **Hotfix 2 — fail-loud carry-over:** implemented (verified). Keep `--allow_fresh_buffer` default False.
4. **P0.1** — `state.pt` always includes buffer (implemented, verified). Note the save-spike caution in V3_FIX_AND_RUN_PLAN §3b.

---

## 4. ACCEPTANCE TESTS (add to `tests/test_upgrade_123.py` or `test_w1a.py`)

1. **Recon shape:** `decode_tokens(tokens)` returns (B,3,64,64); a token grid of zeros decodes to a constant frame.
2. **Recon learns:** 200 optimizer steps on a single fixed frame must drive `l_recon` below 0.01 (proves the head + token path can carry pixel detail).
3. **Action-everywhere:** changing the action changes EVERY token position's prediction (per-position output delta > 0 for all 64 positions after a forward with two different actions).
4. **Dynamic-tile weights:** synthetic batch where one tile changes → that tile's loss weight ≈ 5, others ≈ 1.
5. **Per-token floor:** a collapsed token position (constant across batch) produces a positive `l_tokfloor`; a varying one produces ~0.
6. **Mean pooling:** pooled = mean of tokens before `pooled_proj` (verify by direct computation).
7. **Eval memory (Hotfix 4):** during `evaluate()`, the eval hidden state changes across steps and resets on `done` (assert via a probe hook or by checking two consecutive act() calls with identical obs yield identical readouts only when h is reset).
8. **Success counting:** unchanged behavior (achievement-verified).
9. Full suite green; smoke test exit 0.

---

## 5. WHAT TO RUN ON COLAB — the complete training guide (in order)

### 5.1 Pre-flight (before any GPU time)
1. Implement §2 + §3 fixes; run the acceptance tests; run the full suite + smoke test.
2. **Bump the architecture tag** in `save_checkpoint` metadata: `"arch": "v3.1-w1a"` (old v3-spatial checkpoints cannot load — fresh weights by design).
3. **Delete `checkpoints_v3`** on Drive (the statue-lineage buffer is deliberately discarded — its experience is stationary spam; fresh collection with the fixed model is cleaner and cheap at ~30 SPS).
4. **Keep `checkpoints_v2` archived** (weights-only historical record).
5. **Flip `allow_fresh_buffer` back to `False`** — the fresh run's own `state.pt` will carry its diary (P0.1); from now on any failed resume must fail loud.
6. Config for the validation run:
```python
training_config = {
    "device": device,
    "total_steps": 25_000,               # VALIDATION FIRST
    "checkpoint_dir": ".../prometheus/checkpoints_v4",
    "num_envs": 8, "batch_size": 64, "update_every": 2,
    "eps_base": 0.10,                    # VERIFY it prints/applies (P0.5)
    "eval_interval": 1_000, "eval_episodes": 3,
    "save_interval": 1_000,
    "no_resume": True,
    # token_dim 64, mask_ratio 0.35, mem_hidden 256, si_weight 0.1, symlog_values True,
    # nstep 5, milestone 0.25/0.80, critic_intrinsic_weight 0.0, actor_intrinsic_weight 0.1
}
```

### 5.2 The 25k validation run — what to watch DURING
- `WM Loss` starts ~2–3 (the puzzle is now harder: dynamic tiles weighted 5×) and declines steadily.
- `Critic:` small (symlog).
- RAM ≤ 40%; SPS ≥ 25 (the recon head + per-token floor add minor cost).
- Training discoveries on pace: wood chain (table/sword) inside the first ~2k.

### 5.3 At 25k — the validation battery (run ALL, send the full report)
1. **Scene-collapse probe (THE gate):** token-grid cosines between the 4 canonical scenes — **target ≤ 0.9** (was 0.995). If still > 0.95: escalate the per-token floor before anything else.
2. **Action-sensitivity probe:** 17-action disagreement spread on one scene — **target max/min ≥ 3×** (was 2×).
3. **Drift test H=1..10** — re-earn the gate (≥ 0.8 at H=5–10) *with a model that actually sees the scene* (the previous pass was partly vacuous).
4. **Stage probe:** place_table rank ≤ 3 AND prob ≥ 0.5 in scene (b); scene distributions must now DIFFER across stages.
5. **Dual benchmark N=50** both policies.
6. **Behavioral:** place_table-lives ≥ 10% and rising (recovering toward v2's 27%), wood-lives ≥ 60%, first `collect_stone` = the travel headline, GPS movement > 0.
7. Standard: no NaN, RAM ≤ 40%, SPS ≥ 25, eval achievements > 0 in at least some evals (Hotfix-4 check).

### 5.4 The 25k decision tree

| Result | Action |
|---|---|
| Scene-cosines ≤ 0.9 AND place_table rank ≤ 3 AND table-lives ≥ 10% | **Continue the same run to 1M+** (Stage C). The ladder resumes. |
| Scene-cosines fixed but place_table still rank > 10 | The representation sees; the *value* side lags → run 50k more before judging; check SI loss counts. |
| Scene-cosines still > 0.95 | Stop. The collapse survived the recon anchor → raise `token_var_floor` to 0.1 and `l_recon` weight to 1.0, restart 25k. |
| Training discoveries slower than the 5k scorecard | Check eps_base actually 0.10 and the dynamic-tile weights are live. |

### 5.5 Stage C (1M+) — the ladder protocol
- Resume the SAME lineage to `total_steps: 1_000_000` (state.pt carries the diary; P0.1). `action_repeat=2` if SPS < 60. `save_interval` 2,500.
- Expected ladder (indicative): collect_stone → place_stone/furnace → stone_pickaxe → coal → iron → iron tools. Survival past 300 becomes common.
- **U4 (dreaming) gate:** re-measure drift H=5–10 at 250k/500k/1M checkpoints. When cosine ≥ 0.8 holds on the *fixed* model → write the U4 spec and unlock dreaming.
- Keep: milestone replay, ε schedule, achievement-verified counting, dual eval N=50, forensic suite every 25k.

---

## 6. WHAT NOT TO DO
- No pixel-reconstruction of *future* frames, no video generation — the recon head is current-frame autoencoding only.
- No changes to reward semantics, ε schedule (beyond 0.10), milestone logic, or buffer logic.
- No U4 dreaming before the gate. No SIGReg. No search/MCTS. No architecture growth (token count/dim stay fixed).
- Do not carry the statue buffer; do not delete `checkpoints_v2`; do not skip the falsification protocol in §5.1.

## 7. Success criteria for the WHOLE document
A v3.1 agent that, at 25k steps: sees scenes (cosines ≤0.9), uses actions (spread ≥3×), crafts tables in ≥10% of lives, and passes the drift gate ≥0.8 @ H=10 — measured, reported, no assumptions.
