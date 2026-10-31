# ROADMAP.md — Master Order of Operations (FROZEN)

**Version:** 1.0 — 2026-10-31
**This is the single authoritative roadmap.** Older documents (`ROOT_CAUSE_FIX_PLAN.md`, `PARALLEL_ENV_PLAN.md`) remain valid as *implementation specs for steps already taken/in progress*. Whenever any message, report, or document disagrees with this file on order or naming — **this file wins**.

**Project law (never violated):** the agent learns only from raw pixels, actions, and environment reward. No inventory access, no action masking from privileged state, no hand-edited reward values, no per-achievement scripts.

---

## 0. Where we are (status snapshot, 2026-10-31)

| Item | Status |
|---|---|
| Root-cause fixes (clean critic, milestone replay, n-step, exploration ε, map... no — see Upgrades) | ✅ Done & verified (Stage A: 10 achievements, table-lives 19.3%, critic 0.14, place_table 0%→14% official eval) |
| Stage B — 100k steps on the 8-parallel pipeline | 🔄 RUNNING |
| Parallel-env validation vs serial | ⏳ with the 100k report |
| Stage A forensic instruments (GPS, cause-of-death, explain_step, probes) | ✅ Keep forever |
| Known open problems | travel (homebody), survival-300 (0/50), greedy stagnation, world-model drift (H=10 cosine 0.34), milestone-flood residue |

**Verified lesson that shapes everything below:** the team's earlier imagination builds (v1: decision-time CEM, 0.00%; v3: recurrent imagination, plateau 2.76–3.35%) plus our own drift test (cosine 0.34 @ H=10) prove the current world model hallucinates past ~5 imagined steps. Therefore: **map & memory FIRST, dreaming LAST — gated on a measured test.**

---

## 1. The 4 upgrades (frozen names — no renames)

| # | Name | One-line meaning | Measured problem it fixes |
|---|---|---|---|
| **U1 — MAP** | Keep the encoder's 8×8 grid as 64 spatial tokens; train by hiding tiles of the next map and predicting them | World-model drift (hallucination past ~5 steps); blur representation |
| **U2 — MEMORY** | Small recurrent memory so off-screen objects survive | Table amnesia (87.5% post-table drop-off) |
| **U3 — STRONGER VALUES** | (a) reward + termination prediction heads, (b) symlog/two-hot value targets, (c) self-imitation on milestone windows | Skills stuck at 2–14%; prepares U4 |
| **U4 — DREAMING** | Actor-critic trained on imagined rollouts (H=5, λ-returns) | Long chains (iron, diamond) — **GATED** |

**Background tracks (never paused):** MORE LIVES (Stage B → Stage C 1M+) and OPTIONAL BONUS (passive video pretraining of the U1 encoder — only if W1-style representation learning proves slow).

---

## 2. Order of operations

### Phase 0 — NOW: finish Stage B
- 100k steps running on `checkpoints_v2` (8-parallel, batch 64, update_every 2).
- **Exit:** run completes → produce the report (§4 checklist) → send for analysis.

### Phase 1 — Stage B decision point
Report answers four questions:
1. Table-lives ≥ 19.3% (or rising)?
2. Travel: any `collect_stone`/`place_stone`?
3. Survival: any life past step 300?
4. Greedy benchmark moved off 1.45?
**Decisions triggered:** if still spawn-bound → add `eps_base` 0.05→0.10 for Phase 3. In all cases → write the **U1–U3 spec** (`UPGRADE_1_2_3_SPEC.md`).

### Phase 2 — Implement Upgrades 1–3 (ONE spec, ONE implementation pass)
- **U1 MAP:** encoder outputs 64 tokens (8×8 grid, d≈64, no flatten); predictor = small masked-prediction transformer over visible tokens + action embedding → predicts masked token positions of the NEXT frame's target latents; EMA target encoder outputs the grid; policy/critic read out via attention pooling; VICReg kept on the pooled slot.
- **U2 MEMORY:** GRU over token grids (or persistent object slots) — table survives off-screen.
- **U3 STRONGER VALUES:** reward + termination heads on latents; symlog/two-hot critic targets; self-imitation loss on milestone-window successes.
- **Checkpoint reality:** encoder output shape changes → **fresh network weights; replay buffer (raw frames) carries over** — all collected experience is preserved.
- Implementation effort: ~1–2 days.

### Phase 3 — Validation of the new brain
- **Drift test (the U4 gate):** held-out rollout cosine similarity — target ≥ 0.8 at H=5–10.
- Standard 25k gates: table-lives, wood-lives, critic loss, q_spread trend, dual eval.
- Parallel-vs-architecture sanity: counters within distribution of Stage B.

### Phase 4 — Stage C: 1M+ steps (the stone/coal era)
- On the parallel pipeline, resuming from Phase 3's checkpoint (architecture-independent buffer carries everything).
- `action_repeat=2` if SPS < 60. Checkpoints every 50k, eval every 10k, both policies.
- **Expected ladder:** collect_stone → place_stone/furnace → stone_pickaxe → coal → iron → iron tools. Survival past 300 becomes common.
- Watch: travel, milestone-flood residue (T2 hits capacity → evictions), greedy score.

### Phase 5 — U4 gate → DREAMING
- Re-run the drift test on the Stage C model. **Pass (cosine ≥ 0.8 @ H=5–10) → write U4 spec and run it** (actor-critic on imagined rollouts, H=5, λ-returns — the Zephyr design finally tested on a healthy foundation). Fail → return to U1/U2 refinement.
- Realistic era: this is where iron tools → diamond become reachable.

### Phase 6 — Frontier (unscheduled, research territory)
- Value-guided planning (shaping the JEPA space with learned values), EfficientZero-style reanalysis/search, hierarchical imagination (H-JEPA concept), SIGReg as VICReg successor if collapse ever appears, encoder scaling.

---

## 3. Gates & instruments (the numbers that decide everything)

| Gate | Threshold | Phase |
|---|---|---|
| Critic loss | < 0.30, no NaN | every phase |
| Table-lives | ≥ 19.3% or rising | Phase 1, 3, 4 |
| Travel | first `collect_stone` | Phase 4 headline |
| Survival | first life > 300 steps | Phase 4 |
| Greedy benchmark | moving off 1.45% | Phase 4 |
| q_spread trend | measurable (log every update) and rising | every phase |
| **U4 fidelity gate** | **drift cosine ≥ 0.8 @ H=5–10** | Phase 5 |
| RAM guard | < 80% at all times (emergency save + abort above) | always |

---

## 4. Deliberately NOT done (with trigger to reconsider)

| Item | Why not now | Reconsider when |
|---|---|---|
| Decision-time MPC/CEM planning | team's v1 data: 0.00%; ~500ms/step | never, unless imagination training succeeds first |
| U4 without the fidelity gate | v1/v3 lesson: dreams hallucinate past ~5 steps | — |
| Action masking / privileged shaping / reward edits | project law | never |
| SIGReg replacing VICReg | VICReg works, non-collapse verified | collapse at scale |
| Vectorized worker processes (Phase 2 of parallel plan) | in-process batching sufficient on 2 vCPUs | SPS < 40 after Phase 1 |
| Multi-agent, world-model pretraining at V-JEPA scale | compute/benefit mismatch | Tier frontier |

---

## 5. Document registry

| File | Role | Status |
|---|---|---|
| `ROADMAP.md` (this file) | master order — wins all conflicts | current |
| `ROOT_CAUSE_FIX_PLAN.md` | C1–C7 fixes (critic hygiene, replay, n-step, ε) | ✅ implemented & verified |
| `PARALLEL_ENV_PLAN.md` | 8-game parallel collection + RAM safety | 🔄 implemented, in use |
| `UPGRADE_1_2_3_SPEC.md` | MAP + MEMORY + STRONGER VALUES implementation spec | ⏳ to be written at Phase 1 |
| `CRAFTER_MECHANICS_REFERENCE.md` | ground-truth game rules (human reference) | reference |
