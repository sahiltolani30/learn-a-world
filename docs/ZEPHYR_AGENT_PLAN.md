# Project Zephyr — True-JEPA Actor-Critic World Model Agent

**Status: Stage-0 slice IMPLEMENTED and smoke-tested (2026-10-28). Full `zephyr/` package: world
model, heads, losses, vec env, replay, actor-critic, exploration, planner (Modes A/B), locked
screen, Stage-0 trainer with ZG1/2/3/5 gate evaluation, Stage-1 joint trainer with resume, and
30 passing unit tests including an isolation test (zero imports from the previous line; zero
modifications to any pre-existing file). Next: first Colab session (`zephyr/colab/run_s0.py`)
to pass ZG1–ZG3/ZG5 at full scale.**

**Status history: PLAN (2026-10-27, approved architecture). Created by the ZCode agent.**

Zephyr is a *separate* agent built inside this repository. It shares nothing mutable with the
existing v1/v2/v3 ("T2") line: no file is edited, no module is imported from `models/`,
`planning/`, or `training/`. If Zephyr needs code the old line already has, it is either imported
read-only (two whitelisted modules, below) or rewritten fresh under the `zephyr/` namespace.

## 1. Identity and collision rules

Everything Zephyr creates lives under these names. Nothing else in the repo uses them, so the
other AI's line ("JEPAv3", "T2", `jepa_crafter_*`, protocol `strict_visual_jepa_clean_v*`) and
Zephyr can never overwrite each other.

| Namespace | Value |
|---|---|
| Code package | `zephyr/` (new top-level directory) |
| Module/class prefix | `zephyr_` / `Zephyr` (e.g. `ZephyrWM`, `ZephyrAC`) |
| Checkpoints | `checkpoints/zephyr_*.pt` (local), Drive `learn-a-world/zephyr/checkpoints/` |
| Results | `results/zephyr_*.json`, Drive `learn-a-world/zephyr/reports/` |
| Training data | Drive `learn-a-world/zephyr/data/` |
| Protocol id | `strict_visual_zephyr_clean_v1` |
| Gates | `ZG0` … `ZG9` |
| Tests | `tests/test_zephyr_*.py` |
| Seeds | training 7000+, dev screens 6150–6159, locked benchmark 2050–2099 (same as T2 for comparability), locked final 3100–3199 (unused by T2's 3000–3099) |

**Read-only imports (whitelist — the only old code Zephyr touches):**
1. `envs/` — Crafter wrapper with the determinism patch, `snapshot_state()`/`restore_state()`
   (later used for curriculum; importing it registers `Crafter-v0`).
2. `eval/metrics.py` — `compute_crafter_score()` (official Crafter geometric-mean score) and
   `CRAFTER_ACHIEVEMENTS`, so scores are computed by the identical function as T2's.

Everything else (encoder, GRU, predictor, heads, actor, critic, buffers, collectors, trainer,
planner, screening harness) is new code in `zephyr/`. No import of `models.*`, `planning.*`,
`training.*` is allowed; a unit test enforces this (AST scan of the `zephyr/` package).

## 2. Mission and reference numbers

Beat the T2 line on the **locked Zephyr protocol** (50 episodes, seeds 2050–2099, ≤1000 steps,
strict pixels-only inference, `compute_crafter_score`), then climb toward published
world-model SOTA on Crafter. Two protocols exist and are never merged:

**Zephyr/T2 locked protocol (1000-step episodes)** — the comparability arena:

| Agent | Score | Note |
|---|---:|---|
| Random (v2-era, 100 eps) | 1.77% | reference floor |
| v2_pure champion | 2.16% | old promoted champion |
| T2 checkpoint, clean re-screen | 2.79% / 3.18% | protocol v1 / v2 schema (superseded) |
| T2 paired report | 4.55% | historical, non-clean protocol; not a baseline |
| **T2 + active-consequence scoring + StateReadout** | **4.31%** | `results/screen_v3_ablation_c_active_50ep.json` — VERIFIED: strict v2 protocol, reward 4.30, survival 184.9, 5.22 ach/ep, 10/22. **The honest baseline Zephyr must beat.** |
| **Zephyr target ladder** | 6% → 8.6% (2×) → 10% → 15% | gates ZG6–ZG8, rebased 2026-10-27 |

*Design validation (OBSERVED): the T2 line's jump from 3.18% → 4.31% came from exactly the two
mechanisms Zephyr bets on — a learned pixel-grounded inventory readout (Zephyr's ZG3 head) and
transition-head consequence scoring actually wired into planning with an ablation proof (Zephyr's
ZG4 requirement). This de-risks Zephyr's design; it does not shrink its differentiators: the T2
planner still runs H=10 CEM with engineered scoring constants (`harvest_bonus`, `vital_weight`,
persistence bonuses), has no exploration mechanism (stone essentially undiscovered), and no
actor-critic imagination training. Zephyr's remaining edge is dense-label joint training,
learned exploration, and 15-step latent imagination replacing hand-tuned scoring.*

**Official Crafter protocol (10M frames, 10k-step episodes; EMERALD, ICML 2025, Table 2)** —
directional only, different episode budget:

| Agent | Crafter score |
|---|---:|
| DreamerV3 (M, 37M params) | 34.9% |
| DreamerV3 (XL, 200M params) | 39.6% |
| Δ-IRIS (25M params) | 42.5% |
| Human experts | 50.5% |
| EMERALD (30M params) | 58.1% (first >human at 10M; unlocks all 22) |

Interpretation: our 1000-step budget caps achievable score well below the 10k-step numbers, so
"beat DreamerV3-class" is operationalized as *matching its behavior*, not its absolute number:
(1) consistent multi-hop tool chains (wood → table → pickaxe → stone), (2) ≥10 of 22
achievements discovered, (3) no hand-coded rules anywhere.

## 3. Why a new architecture instead of patching T2

Zephyr is designed against the five verified failure modes of the old line:

1. **Transition heads were provably unused by planning** (T2's head-invariance test: deleting them
   left actions bitwise identical). In Zephyr, *every* decision pathway runs through the learned
   world model, and any auxiliary-head usage carries a mandatory ablation test (ZG4).
2. **Policy/critic broke when the encoder was retrained without them** (latent drift 7.05, policy
   top-1 agreement 80.9%, gameplay 4.55→3.40%). Zephyr trains actor, critic, and world model
   *jointly from replay* (Dreamer-style), so the policy is never stale relative to the encoder.
   The frozen-then-retrain recipe that caused the drift is not used at all.
3. **State-delta supervision was 0.3% dense (2,000 labels/666k transitions)** because labels
   existed only for retro-labeled windows. Zephyr's collector logs engine inventory every step in
   diagnostic mode → **100% of training transitions carry supervision labels** (targets only).
4. **Termination head had zero recall** (majority-class collapse). Zephyr uses class-weighted
   focal BCE and gates on recall ≥ 0.5 *and* precision ≥ 0.5 vs the majority baseline.
5. **No exploration mechanism for sparse frontier events** (stone never discovered except by
   accident). Zephyr runs intrinsic motivation from day one: ensemble latent disagreement +
   episodic novelty, both fully learned, decayed as extrinsic reward takes over.

Design principles carried over from the T2 plan's contract (unchanged, re-declared):
JEPA latent prediction is the world-model foundation and is never disabled; strict inference uses
only raw pixels, previous actions, scalar reward, and internal recurrent state; engine
inventory/achievements/coordinates/semantic maps are training-time *targets* and evaluation-time
*scoring* only, never action inputs; no recipe tables, no `if wood >= 2` rules, no hardcoded
action sequences, no landmark anchors, no frontier-index thresholds (T2's MPC contains all of
these in non-strict mode — Zephyr contains none in any mode); every claim labeled
`VERIFIED / OBSERVED / HYPOTHESIS / PLANNED / PRIVILEGED`; every phase ships an ablation.

## 4. Architecture

### 4.1 ZephyrWM — the JEPA world model

```
obs (64,64,3) ──► ConvEncoder ──► z_t (512)
                  (4×stride-2 conv, 32→64→128→256, SiLU, LayerNorm, → MLP 512)

belief:   h_t = GRU(h_{t-1}, [z_t ; emb(a_{t-1})])          (GRU 512)

JEPA:     ẑ_{t+1} = Predictor([z_t ; emb(a_t) ; h_t])
          target  = sg( EMA-Encoder(o_{t+1}) )                (decay 0.996→0.999)
          L_JEPA  = MSE(ẑ_{t+1}, target)
                    + λ_v·VarReg(ẑ) + λ_c·CovReg(ẑ)           (VICReg-lite, anti-collapse)

multi-step: predictor+GRU unrolled H_wm ∈ {1..15}, horizon-weighted schedule
```

Auxiliary heads (all read only `z_t, h_t`; all supervised by diagnostic targets; none is an
inference *input*):

| Head | Output | Loss | Used by |
|---|---|---|---|
| `reward` | symlog twohot over 65 bins | cross-entropy | Mode-A actor, Mode-B CEM |
| `continue` | P(done) | weighted focal BCE | imagination bootstrapping, CEM |
| `readout` | 16-d self-state: 12 backpack counts (log1p) + health/food/drink/energy | MSE on log1p | strict-play self-state estimate, logging, Mode-B features |
| `effect` | Δ16-d inventory per action | masked MSE | **Mode-B only** + relation bank (Phase R); banned in Mode A |
| `critic` | symlog twohot λ-return | cross-entropy | actor objective, CEM terminal value |

The readout head is Zephyr's state grounding: the agent estimates "how much wood do I carry"
from pixels, trained on 100%-dense labels, and *beats a persistence baseline on held-out
episodes* before it is trusted (ZG3). This is learned grounding, not privileged input — the same
distinction the T2 plan already established for its Section-6 readout.

### 4.2 ZephyrAC — actor-critic in imagination (primary controller)

DreamerV3-lite recipe, proven on Crafter (DreamerV3-M 34.9%, EMERALD 58.1% at 10M frames):

- **Actor**: categorical over 17 actions, unimix 1%, from `[z_t; h_t]`; trained on imagined
  λ-returns (H=15, λ=0.95, γ=0.997) with entropy regularization; gradients flow through the
  learned dynamics into the critic objective but not back into the world-model loss
  (DreamerV3 stop-gradient discipline).
- **Critic**: symlog twohot EMA-target critic, regressed on λ-returns of imagined rollouts.
- **Replay**: episode buffer, uniform 90% + achievement-prioritized 10% (a transition that
  precedes any achievement within 16 steps is oversampled — a *learned-data* balancing rule, not
  a reward hack; ablated).
- **Joint updates**: every train step updates WM + actor + critic from the same batch. This
  structurally eliminates T2's encoder-drift-vs-policy failure.

### 4.3 Zephyr-X — exploration (how stone/coal get discovered)

Both signals are learned and hand-off automatically:

1. **Ensemble disagreement**: K=5 bootstrap dynamics heads predict ẑ_{t+1}; intrinsic reward =
   std across their predictions (Plan2Explore-style epistemic uncertainty).
2. **Episodic novelty**: kNN distance of h_t against a within-episode key memory (E3B-style),
   cheap and per-episode.

`r_total = r_ext + β_t · (w₁·r_ens + w₂·r_episodic)`; β scheduled high in early training, decayed
once extrinsic progress saturates. No hand-written curiosity targets ("find stone") — the
mechanism must *discover* stone, and ZG7 measures that it does.

### 4.3b Weight provenance decision: trained FROM SCRATCH

Zephyr trains a **new model with fresh random initialization**. No `jepa_crafter_*.pt`
checkpoint is loaded, copied, or fine-tuned. Old checkpoints stay on Drive untouched, used only
as *baseline comparisons* in reports.

Why not warm-start from the T2 checkpoint:

1. **Provenance cleanliness** — any Zephyr score must be attributable to Zephyr's recipe alone;
   a warm-started score would need T2's training history as a confound label.
2. **Contaminated priors** — T2's weights encode the 3.5%-plateau behavior (behavior-cloned
   policy prior, CEM-era Q-ensemble) that Zephyr is designed to replace.
3. **The compatibility problem returns** — Zephyr's encoder co-adapts with actor/critic from
   step 0. Grafting a frozen or warm encoder trained under different objectives reintroduces
   exactly the drift failure that killed T2's promotion (4.55% → 3.40%).
4. **Compute is sufficient** — DreamerV3-class agents learn Crafter from scratch within the
   plan's 10M-frame budget; there is no evidence warm-starting shortens this materially.

Optional labeled ablation only (ABL-W): encoder-only warm start from T2's frozen Phase-1.5
checkpoint if S1 progress is slow — always reported as `warm_start`, never as primary.

### 4.4 Planning modes (ablated, never silently mixed)

- **Mode A (primary)**: actor-greedy action at every step. The actor *is* the plan — it was
  trained on 15-step latent imagination, so multi-hop credit already lives in the policy.
- **Mode B (ablation, "Zephyr-MPC")**: CEM over latent imagination (H=15) scoring
  `Σγᵗr̂_t + γ^H·V(s_H)` with continue-head discounting, optionally biased by the effect-head
  prior. Exists to answer T2's open question — does explicit consequence scoring help once the
  heads are actually wired in? — with a clean A/B (ablation A4/A5). Mode B requires the ZG4
  ranking-shift proof before any report claims its effect.

## 5. How Zephyr "understands" wood → table → pickaxe → stone

Relations are never written. They are (a) represented, (b) verified, (c) later, banked:

1. **Represented**: the effect head learns, from 100%-dense diagnostic labels, that
   `(state-context, action=do_place_table)` correlates with `Δwood = −2`; the readout learns to
   perceive wood; the reward head learns `place_table → +1`. Composition (a table and a pickaxe
   cost 3 wood total) is *derivable* by the model, not memorized from a recipe file.
2. **Verified**: ZG4 intervention testing — from held-out states, predict outcomes under
   alternative actions, compare against measured outcomes (T2's Gate-9 protocol, stricter
   threshold: ≥85% + action-ranking shift test if effect head touches any planner path).
3. **Banked (Phase R)**: mine replay into an evidence-backed relation bank —
   `(precondition-readout-context, action) → {N_pos, N_neg, mean Δ, episode evidence ids,
   Wilson CI}`. Edges exist only with trajectory evidence (T2 Phase-2C gate, adopted). Used as a
   Mode-B prior. **Retention gate ZG9**: save bank → restart process → new-world acquisition time
   improves vs empty bank at equal interaction budget.

Survival (hunger/health/night) needs no special machinery: it is dense-ish extrinsic reward the
critic learns; the readout gives the actor's belief the vital signals.

## 6. Training campaign (Colab) and gates

Config style: dataclasses in `zephyr/config.py` (single source of truth; the T2 line's
argparse-default duplication is deliberately not copied). All Colab runs are **resumable**:
every 30 min, `{model, optimizers, replay index, RNG, collector state, config, git commit}` →
`learn-a-world/zephyr/runs/<run_id>/`; a resume manifest pattern like T2's phase-1.5 work.

| Stage | What happens | Colab budget (T4 est.) | Gate |
|---|---|---|---|
| **S0 ignition** | SubprocVectorEnv (8–16 workers) random collection, 250k frames, 100% label density → train ZephyrWM ~150k steps | 3–5 h, 1 session | ZG1, ZG2, ZG3, ZG5 |
| **S1 loop** | Joint WM+AC+exploration collect/train loop, 2M frames total, resumable; dev screens every 500k on seeds 6150–6159 | 12–24 h, 3–6 sessions | ZG4, ZG6 |
| **S2 scale** | Continue to 5–10M frames; β decay; Mode-B ablation; achievement-priority replay analysis | 2–5 GPU-days, N sessions | ZG7, ZG8 |
| **R relations** | Relation-bank mining + retention-after-restart experiment | 1 session | ZG9 |
| **C curriculum** | Snapshot/restore near-success start states (agent-found only), labeled curriculum-assisted | 1 session | optional booster |

**Gates (promotions require: locked 50-ep screen + ablation + provenance JSON):**

- **ZG0** protocol lock: `zephyr/screen.py` emits `strict_visual_zephyr_clean_v1` metadata
  (git commit, checkpoint SHA-256, strict flags, provenance labels).
- **ZG1** JEPA health: L_JEPA↓, no collapse (latent std ≥ 1.0 on train+val), L_JEPA never
  disabled in any config.
- **ZG2** free-running rollouts beat persistence at horizons 1/5/10/15 on held-out episodes.
- **ZG3** readout beats persistence readout by ≥30% (held-out, log1p MSE).
- **ZG4** intervention accuracy ≥ 85% (held-out counterfactual action→outcome match); mandatory
  ranking-shift ablation for any head used in planning.
- **ZG5** continue head recall ≥ 0.5 AND precision ≥ 0.5 AND AUC ≥ 0.9 (majority baseline
  reported alongside).
- **ZG6** gameplay ≥ **8.6%** (2× honest baseline 4.31%) on seeds 2050–2099, survival ≥ 185
  steps, `place_table` ≥ 60%.
- **ZG7** discovery without curriculum: ≥ 8/22 achievements found in training, `collect_stone`
  ≥ 1 event, pickaxe ≥ 30% of dev-screen episodes.
- **ZG8** ≥ **10%** locked protocol (15% stretch).
- **ZG9** retention: post-restart acquisition improvement, memory on vs off, equal budget.

**Planned ablations** (each phase ships one): A1 L_JEPA off (expected collapse → proves the
contract), A2 exploration off (stone discovery dies), A3 twohot vs MSE reward head, A4 Mode A vs
Mode B, A5 effect-head prior on/off, A6 joint training vs T2-style freeze-then-retrain (expected
to reproduce the drift failure, confirming root cause).

## 7. Code layout (all new)

```
zephyr/
  __init__.py          config.py          vec_env.py         buffer.py
  wm.py                heads.py           losses.py          ac.py
  explore.py           planner.py         screen.py          relbank.py
  train_wm.py          train_loop.py      colab/run_s0.py    colab/run_s1.py
tests/test_zephyr_*.py    (isolation-from-old-modules test, twohot, focal, gates smoke)
```

Reused read-only: `envs/` (deterministic wrapper + snapshots), `eval/metrics.py` (official
score). Rejected T2 patterns: hardcoded frontier indices, landmark anchors/dead-reckoning,
inventory-gated MPC bonuses, semantic-memory subgoal deduction, behavior-cloned policy prior as
the only actor, freeze-then-retrain promotion.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| T4 compute ceiling (DreamerV3-class needs ~10M frames) | Lite config (~15M params), fp16, resumable multi-session runs; honest reporting of frames trained; L4/A100 config knob if available |
| 1000-step episode budget caps score | Primary targets defined *within* our locked protocol; official-protocol run deferred to S2 stretch |
| Exploration never finds metallurgy chain | ZG7 measured early; fallback: achievement-prioritized replay + snapshot-restore curriculum (labeled, training-only) |
| Collapse / dead latents | EMA target + VICReg-lite terms + ZG1 monitors from step 0 |
| Colab disconnects | 30-min checkpointing, Drive-resumable, run manifests |
| Actor ignores rare actions | unimix categorical + entropy reg + action-usage metrics in dev screens |
| accidentally importing old modules | AST isolation unit test in CI/test suite |

## 9. What Zephyr needs from the operator

Colab GPU sessions when a stage is ready to run (T4 works; L4/A100 roughly halves wall-clock).
Local MPS laptop is used only for unit tests and short smoke runs. All heavy artifacts live in
`learn-a-world/zephyr/` on Drive; the repo receives only code, tests, and report JSONs.

**Immediate next step (PLANNED, not executed): implement `zephyr/` S0 slice**
(`wm.py`, `heads.py`, `losses.py`, `vec_env.py`, `buffer.py`, `train_wm.py`, `screen.py`,
configs + unit tests), laptop smoke test, then hand over `colab/run_s0.py` for the first Colab
session.
