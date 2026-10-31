# V3_FIX_AND_RUN_PLAN.md — Hotfix for Diary Carry-over + Colab Run Protocol

**Version:** 1.0 — 2026-10-31
**Audience:** implementing agent. Follow literally. This plan is small and surgical — it does NOT change the model, the learning rules, or `UPGRADE_1_2_3_SPEC.md` (U1–U3 are already implemented and verified). It fixes the broken **diary carry-over**, hardens it against silent failure, and defines the exact Colab run sequence.
**Read with:** `ROADMAP.md` (master order), `UPGRADE_1_2_3_SPEC.md` (U1–U3 architecture — already implemented).

**Project law (unchanged):** pixels, actions, environment reward — nothing else.

---

## 0. What happened (evidence from the v3 startup log)

The v3-spatial run started correctly — the puzzle (masked tile prediction) is training (WM Loss 2.67→1.54 declining; transformer predictor live; symlog critic confirmed by tiny losses ~0.003–0.06). But the **diary carry-over failed**:

```
[!] Warning: Could not load buffer from .../checkpoints_v2/checkpoint_step_100000.pt:
    WeightsUnpickler error: Unsupported global: GLOBAL pathlib._local.PosixPath ...
[*] Starting Stage C with a fresh replay buffer.
```

**Root cause:** PyTorch 2.6 changed `torch.load` to default `weights_only=True`. The carry-over code calls `torch.load` without overriding it, and the checkpoint payload contains a `pathlib.PosixPath` object (saved by the checkpoint manager's metadata). The safe loader refuses it → the exception was caught → the run continued with an **empty buffer** → the 63,074-transition Stage-B diary was discarded. (The main `CheckpointManager.load` path already uses `weights_only=False` — only the new carry-over code path has this bug.)

**Consequence if unfixed:** every restart silently starts with an empty diary — the exact "silent loss" failure class this project has now hit twice.

**Also observed (not blocking, monitored):** greedy evaluation returns exactly −0.90 with 0 achievements in every eval — the known fresh-critic greedy stagnation (return = −9 health/10, zero achievements). The *training* policy is healthy: 9 achievements by step 12k with a fresh brain and empty buffer (place_table @960, make_wood_sword @3,447 — 3× faster than v2). Watch, don't fix: if evals still show 0 achievements past step ~30k, inspect the eval path then.

---

## 1. HOTFIX 1 — Fix the carry-over loader (2 lines, blocking)

**File:** the Stage-C carry-over code in `src/train.py` (the section logging `[*] Checking for Stage B buffer carry-over from:`).

Replace every `torch.load(path, ...)` in that section with:

```python
buffer_state = torch.load(path, map_location=self.device, weights_only=False)
```

Rationale: the file is our own checkpoint from our own Drive — a trusted source; `weights_only=False` is what the main `CheckpointManager.load` already uses (it has worked across every resume so far).

**Acceptance:** loading `checkpoints_v2/checkpoint_step_100000.pt` yields a buffer state with `t1_size == 50000` and `t2_size == 13074` (±small drift from the last save point).

---

## 1b. HOTFIX 4 — Evaluation forgets to use the memory notebook (BLOCKING for meaningful eval numbers)

**Found while investigating "why does v3 eval get 0 achievements while v2's fresh run got them at step 500":** in `src/train.py::evaluate()` (line ~1009), the greedy call is:

```python
action = self.agent.act(obs, eval_mode=True, planning=False)   # ← h is NOT passed
```

`agent.act` then calls `get_readout(z, h=None)` which defaults the memory to **zeros and never updates it** — so during ALL evaluation episodes the memory notebook is frozen at zero, while the critic was trained on readouts where the memory *evolves* during every episode. The eval policy therefore acts on off-distribution inputs (a state the critic almost never saw: "step-0 memory, mid-episode scene") → degenerate passive behavior → the exact −0.90/0-achievement fingerprint. v2 never had this bug because it had no memory component — its eval path was correct by construction.

**Fix (in `evaluate()`, both the greedy and stochastic arms):**
```python
eval_h = torch.zeros(1, self.mem_hidden, device=self.device)   # per episode
# per step:
action = self.agent.act(obs, eval_mode=(not stochastic), planning=False, h=eval_h)
# after stepping (and on episode reset):
if done: eval_h = torch.zeros(1, self.mem_hidden, device=self.device)
else:     eval_h = self.agent.gru(pooled_z, eval_h)   # same update rule as training
```
(Match the exact update rule used in `collect_step` — read the pooled latent the same way.)

**Acceptance test:** run 3 greedy eval episodes at ~zero training and assert the eval loop updates `eval_h` (it changes across steps) and resets on `done`; plus a unit test that `get_readout(z, h)` with evolving h differs from h≡0 after a few updates.

**Note:** this bug affects **evaluation numbers only** — training collection passes `h` correctly (verified), so learning was never harmed. It explains the −0.90/0-achievement eval fingerprint on top of the fresh-critic passivity.

---

## 1c. HOTFIX 5 — Benchmark hygiene (from the same report)
- The stochastic benchmark ran 50 games while the header said 20 and greedy ran 20 — standardize: **both policies N=50**, N printed in the header and the summary computed over the same games.
- `success_counts` reporting: label the wake_up-coincidence issue as fixed by P0.3's achievement-verified counting (already implemented — verify the new counts contain no entries absent from the unlocked list).

## 1d. HOTFIX 6 — Make the memory state `h` a first-class part of the stored transition (CRITICAL — found in full-code audit)

**Three input-distribution mismatches found by line-by-line audit (all one root cause):**

1. **Critic input mismatch (CRITICAL).** The critic is trained on the RAW pooled latent: `z_curr = encoder(obs)` → `compute_loss(z_curr, ...)` (train.py:970/982). But every greedy action and every evaluation feeds the critic the READOUT: `act()` eval-mode → `critic(get_readout(z, h))` (agent.py:367-369). The critic is queried on inputs from a distribution it was never trained on (a Linear warp of its training inputs, plus an h-half it never saw). This corrupts every argmax the eval policy makes.
2. **Actor input mismatch (CRITICAL).** The actor is trained on `get_readout(z_curr, h=None → zeros)` (train.py:1008) but acts during collection on `get_readout(z, h_evolved)` (train.py:541/618). The h-half of `readout_proj`'s weights is never trained (h≡0 during training ⇒ the h-columns only add their bias), so from step 1 of every episode the actor acts on inputs shifted by an untrained transform.
3. **Predictor mismatch (MEDIUM).** World-model training calls the predictor WITHOUT h (train.py:886), but collection-time curiosity calls it WITH h (train.py:641). `mem_proj` receives no gradient during training (h never passed) → at collection the memory token is a random projection injecting noise into the disagreement signal.

**Root cause:** the memory state `h_t` depends on the episode's history, but the replay buffer stores only (obs, action, reward, done) — so h cannot be reconstructed for sampled transitions, and the implementer trained everything h-free while acting h-full.

**Fix — store h per transition (cheap and complete):**
- Buffer: add `t1_h` / `t2_h` arrays, float16 (N, 256) → ~25 MB at 50k. `add(..., h=None)` stores the h that was CURRENT when the transition was taken (pre-update, matching what the actor actually used to act).
- `sample`/`sample_nstep` return `batch["h"]` (float32 converted) for base transitions; `sample_nstep` also returns `boot_h` read from the boot slot's stored h (fallback: base h).
- **Critic training:** `readout_curr = get_readout(z_curr, batch["h"])` → critic trains on readouts — SAME distribution as acting. Bellman bootstrap: compute target Q on `get_readout(z_boot, boot_h)` (target encoder pooled + stored boot h).
- **Actor training:** `readout_curr = get_readout(z_curr, batch["h"])` — actor trains on h-bearing readouts, matching collection.
- **Predictor training:** `forward_tokens(curr_tokens, actions, h=batch["h"])` — mem_proj finally trains; collection curiosity becomes in-distribution.
- **Eval:** Hotfix 4's `eval_h` (unchanged) now matches training semantics.
- GRU updates at collection: wrap in `torch.no_grad()` (they currently build autograd graphs — train.py:716/848).

**Acceptance tests:** (a) train 200 updates with h-bearing batches — assert `readout_proj` h-half weights receive nonzero gradients; (b) critic loss on stored-h readouts converges below the h=0 baseline on the same data; (c) buffer roundtrip includes `t1_h` checksum.

**Stopgap (if hotfix must ship same-day):** 1-line alternative — make `act()` eval-mode feed the critic the RAW pooled z instead of the readout (fixes mismatch #1 only; #2/#3 remain). Not recommended as the end state.

**Severity without the fix:** the policy/critic act on inputs they never trained on whenever h≠0 (i.e., always after each episode's first step) — this is plausibly a contributor to weak greedy behavior beyond the Hotfix-4 eval bug.

## 2. HOTFIX 2 — Fail loud: no more silent fresh buffers

The carry-over must never silently degrade to an empty buffer. New behavior in the same section:

```python
CARRY_SOURCE = os.path.join(CARRY_DIR, "checkpoint_step_100000.pt")   # or state.pt fallback
try:
    buffer_state = torch.load(CARRY_SOURCE, map_location="cpu", weights_only=False)
    ...restore into self.buffer...
    print(f"[+] Buffer carry-over OK: {self.buffer.size} transitions restored from {CARRY_SOURCE}")
except Exception as e:
    if not self.allow_fresh_buffer:
        raise RuntimeError(
            f"Diary carry-over FAILED ({e}). Refusing to start with an empty buffer. "
            f"Fix the source checkpoint or pass --allow_fresh_buffer to explicitly accept the loss."
        ) from e
    print(f"[!] Buffer carry-over failed ({e}) — starting FRESH by explicit flag.")
```

- New config flag: `--allow_fresh_buffer` (default **False**). Silent loss is now impossible: either the diary loads, or the run stops and says why.
- Print the restored transition count in the banner — the operator must see `63,074` (or close) at startup.

**Acceptance tests (add to `tests/test_upgrade_123.py`):**
1. Carry-over from a valid v2-format checkpoint restores `t1_size`/`t2_size` exactly.
2. Carry-over from a corrupted/missing file raises `RuntimeError` when `allow_fresh_buffer=False`, and proceeds with an empty buffer only when the flag is set.

---

## 3. HOTFIX 3 — Sanitize checkpoints (so safe-loading works forever)

The `PosixPath` in the payload is what triggered the safe-loader refusal. In `CheckpointManager.save`/`save_atomic_checkpoint`, sanitize the payload before `torch.save`: recursively convert every `pathlib.PurePath` to `str` (one small helper applied to the metadata/config dict). Keep the `weights_only=False` fix from Hotfix 1 regardless (belt and suspenders).

**Acceptance:** a saved checkpoint loads successfully with `weights_only=True` (test: save → `torch.load(path, weights_only=True)` → no exception).

---

## 4. Colab run protocol (exact sequence)

> ⚠️ **Folder rule: delete ONLY `checkpoints_v3`. NEVER delete `checkpoints_v2`** — it contains `checkpoint_step_100000.pt`, the sole source of the 63,074-transition diary and the entire Stage A/B archive.

1. **Push the hotfixed code to GitHub** (Colab pulls from the repo — verify the cell-4 `git pull` shows the hotfix commit).
2. **Delete `checkpoints_v3`** on Drive (user is doing this) — v3 starts fresh.
3. **Run cell 6** with:
   ```python
   training_config = {
       "device": device,
       "total_steps": 25_000,          # VALIDATION FIRST — extend to 1,000,000 only after the 25k report
       "checkpoint_dir": ".../prometheus/checkpoints_v3",
       "num_envs": 8,
       "batch_size": 64,
       "update_every": 2,
       "eps_base": 0.10,               # travel lever (pre-registered)
       "eval_interval": 1_000,
       "eval_episodes": 3,
       "save_interval": 1_000,
       "no_resume": False,             # v3 folder is empty → fresh weights, and the carry-over pulls the v2 diary
       # everything else per UPGRADE_1_2_3_SPEC defaults (token_dim 64, mask_ratio 0.35, mem_hidden 256,
       # si_weight 0.1, symlog_values True, milestone 0.25/0.80, nstep 5, critic_intrinsic_weight 0.0)
   }
   ```
4. **Startup verification (all four must be true before walking away):**
   - `[+] Buffer carry-over OK: ~63,074 transitions restored` (Hotfix 2 banner)
   - First log line: `Buf:` ≈ **63,000** (T1 ≈ 50,000, T2 ≈ 13,000)
   - `WM Loss` starts high (~2–3) and declines — the puzzle is training
   - `Critic:` loss small (~0.003–0.06) — symlog is live
   - If instead you see `[*] Starting Stage C with a fresh replay buffer.` → **STOP**, the hotfix didn't reach Colab (check the git pull).
5. **At 25k:** run the master forensic suite (`senior_forensics`) → send the report. It must include: drift test H=1..**10**, the stage-wise policy probe (fresh spawn / wood=2 / facing-table / pickaxe+stone), dual benchmark N=50, GPS, cause-of-death.
6. **25k decision tree:**
   - `collect_stone` appeared AND drift cosine @H=5 ≥ 0.9 → extend the same run to **1,000,000 steps** (Stage C — the stone/coal era). Do not change anything else.
   - Travel still closed (no stone) → keep training to 50k with ε=0.10 before touching anything; if still closed at 50k, report back (next lever is pre-registered but not automatic).
   - Eval achievements still 0 at 30k → send the report; we inspect the eval path for a v3-specific bug.
7. **Resume note:** from now on, mid-run resumes are safe — `state.pt` always carries the diary (P0.1), and Hotfix 2 makes any carry-over failure loud instead of silent.

---

## 5. Expected signatures while it runs (know what "healthy" looks like)

| Signal | Healthy v3 signature |
|---|---|
| WM Loss | starts ~2–3, declining through the 20s of thousands (harder puzzle than v2's 0.4–0.7 — that's correct) |
| Critic loss | ~0.003–0.06 (symlog-compressed — tiny by design) |
| Eval return early | may stay −0.90 (fresh-critic greedy passivity) — **watch, don't panic before 30k** |
| Training discoveries | wood chain within the first ~5k (v3 already proved: sword @3.4k) |
| MS/T1 ratio | ~10% early (flood absent in v3 — better than v2's 98%) |
| RAM | ≤ 40% |
| SPS | 40–120 env-SPS |

---

## 6. Do NOT do in this pass
- No changes to U1–U3 architecture (they're implemented and verified).
- No dreaming (U4) — gated on the H=5–10 drift measurement from the 25k report.
- No reward edits, no action masking, no worldgen changes.
- Do not delete or "clean" `checkpoints_v2` — it is the only copy of the Stage A/B diary and the pretrained-lineage archive.
