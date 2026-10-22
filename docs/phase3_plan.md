# learn-a-world v3 — Phase 3 Plan2Explore Curiosity Collection

> **Created:** October 23, 2026  
> **Authorization:** Muse Phase 3 conditional green light (via Sahil Oct 23 2026)  
> **Parent Plan:** [docs/v3_implementation_plan.md](docs/v3_implementation_plan.md)  
> **This document covers:** Implementation + Mac smoke test + Colab execution for Phase 3 Tranche T1 only.  
> T2+ require separate authorization after T1 gate review.

---

## The Honest Baseline

**Same Seeds (2050–2099), Same Protocol (50 eps, max 1000 steps/ep, H=10, C=64, CEM=2):**

| Model | Crafter Score | Mean Reward | Survival | Unique Ach. | Committed File |
|:---|:---:|:---:|:---:|:---:|:---|
| **v3 Phase 2** | **2.31%** | 2.50 ± 0.94 | 171.8 ± 55.5 steps | 9 / 22 | `results/screen_v3_50eps_results.json` |
| **v2 Champion** | **2.74%** | 2.52 ± 1.47 | 171.1 ± 56.7 steps | 10 / 22 | `results/screen_v2_50eps_results.json` |

**v3 Phase 2 currently trails v2 by 0.43 pp on identical seeds. Phase 3 goal: close and surpass this gap.**  
T1 Gate: v3 post-T1 screening (seeds 2050–2099, 50 eps) must beat **2.31%** (pre-Phase-3 v3 baseline).  
Phase 4 gate: must also beat v2's **2.74%** AND reach **3.0%** on locked seeds 3000–3099.

> Cross-protocol comparisons (100-ep locked seeds vs 50-ep screening) are always labelled explicitly and never merged in prose.

---

## Mandatory Corrections (Done — Committed `2960a29`)

- [x] `results/colab_run1_summary.json` — Corrected notes: `initial_metrics` are Step-0 untrained v3 values, not v2 baselines. True cross-model diagnostics documented inline.
- [x] `docs/v3_implementation_plan.md` — Plan with gates committed to repo.
- [x] `results/screen_v2_50eps_results.json` — v2 champion screened seeds 2050–2099, score **2.74%**, committed.

---

## Seed Discipline (Fixed)

| Seed Range | Purpose | Status |
|:---|:---|:---|
| `5000+` | Phase 3 collection | Reserved — use in order starting 5000 |
| `2050–2099` | Screening / model selection (50 ep) | In use |
| `4000–4049` | Post-Phase-3 diagnostic re-runs (D1, D3) | Reserved |
| `3000–3099` | **Locked final only** | Untouched — off limits |
| `2000–2049` | Retired (diagnostics used part) | Done |

---

## Phase 3 Architecture: Plan2Explore Curiosity

**Curiosity = mean per-dimension std across 3 independent latent dynamics predictors:**

$$r_\text{cur}(z_t, h_t, a_t) = \frac{1}{d}\sum_{j=1}^d \text{std}_{k \in \{1,2,3\}}\!\left[\hat{z}^k_{t+1}[j]\right]$$

Each predictor $f^k_\phi$ has the **same architecture as `RecurrentPredictor`** (z + h + a → delta_z → next_z) but **independently initialized** weights and **independent Adam optimizer**.

This is **NOT Q-ensemble disagreement**. Q measures value uncertainty; predictor ensemble measures dynamics-model uncertainty — novelty to the world model's own imagination.

**Collection policy:** `RecurrentMPCPlanner` with augmented reward:
$$r_\text{total} = \lambda_\text{task} \cdot r_\text{task} + \lambda_\text{cur} \cdot r_\text{cur}, \quad \lambda_\text{task}=0.3, \;\lambda_\text{cur}=1.0$$

---

## Files to Build

### 1. `models/curiosity.py` — Add `Plan2ExploreCuriosity` (keep old `CuriosityModule`)

```python
class Plan2ExploreCuriosity(nn.Module):
    """
    Plan2Explore: 3-predictor ensemble disagreement over imagined next latents.
    Curiosity bonus = mean per-dim std across K predictors.
    Each predictor is a RecurrentPredictor (same arch, independent weights).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_actions: int = 17,
        num_predictors: int = 3,
    ): ...

    def forward(self, z, h, action) -> torch.Tensor:
        # Returns [K, B, latent_dim] — per-predictor next-z predictions

    def curiosity_bonus(self, z, h, action) -> torch.Tensor:
        # Returns [B] scalar bonus per sample
        # = mean over latent dims of std over K predictors

    def predictor_loss(self, z, h, action, z_target) -> torch.Tensor:
        # MSE loss for each predictor against stop-gradient target, averaged over K

    def optimizers(self, lr=3e-4):
        # Returns list of K independent Adam optimizers (one per predictor)
```

**Training:** Predictors update online during collection (every 100 collected steps, 20 gradient steps on 32-sequence batches from the **full** accumulated buffer).

---

### 2. `training/collector_v3.py` — New File

Drive-resumable curiosity-guided episode collection.

**Shard naming:** `shard_curiosity_v3_t{tranche:02d}_{shard_idx:04d}.npz`

**Manifest:** `dataset_index.json` updated atomically after every shard. Each shard entry records:
```json
{
  "shard_name": "shard_curiosity_v3_t01_0000.npz",
  "tranche": 1,
  "source_policy": "curiosity_v3",
  "seed_range": [5000, 5019],
  "transition_count": 2000,
  "per_achievement_event_counts": {"collect_stone": 12, ...}
}
```

**Resume logic:** On startup, read manifest, count total transitions already collected, resume from next uncollected seed.

**Full checkpoint save:** Every 1,000 collection steps save:
`{ model_state_dict, curiosity_state_dict, optimizer_state_dicts, step, rng_state_np, rng_state_torch, config_hash }` → Drive.

**Interface:**
```python
def collect_tranche(
    model_checkpoint: str,         # jepa_crafter_v3_phase2.pt
    curiosity_checkpoint: str,     # None on first run
    drive_dir: str,                # Drive output dir
    target_new_transitions: int,   # 160_000 for T1
    seed_start: int,               # 5000
    tranche_id: int,               # 1
    lambda_task: float = 0.3,
    lambda_cur: float = 1.0,
    max_steps_per_ep: int = 1000,
    device: str = 'cuda',
) -> dict:                         # manifest summary dict
```

---

### 3. `training/run_v3_colab_run2.py` — New Colab Campaign Script

Combined Colab script:
1. Mount Drive, read manifest for existing transitions.
2. Call `collect_tranche()` for T1 (160k transitions, seeds 5000+).
3. After collection: call incremental retrain on ALL data (old 8k shards + new 160k curiosity shards).
4. Save full training checkpoint + step results JSON to Drive every 500 steps; keep last + best by val loss.
5. Post-retrain: run 50-ep screening on seeds 2050–2099 → `results/screen_v3_t1_results.json`.
6. Inline T1 gate check report.

---

### 4. `tests/test_phase3_curiosity.py` — 6 Smoke Tests

All 6 must pass on Mac CPU before Colab:

| # | Test Name | What It Checks |
|:---|:---|:---|
| 1 | `test_plan2explore_forward_shapes` | K=3 preds return `[K, B, latent_dim]` |
| 2 | `test_plan2explore_bonus_positive` | Bonus > 0 with independent random inits |
| 3 | `test_plan2explore_predictor_loss_gradient` | Loss differentiable, `.grad` not None |
| 4 | `test_collector_v3_smoke` | 2 env steps collected, shard written, manifest updated |
| 5 | `test_manifest_resume_logic` | Load existing manifest, correct seed skip |
| 6 | `test_curiosity_guided_plan_smoke` | 1 planning call with augmented reward, action ∈ [0, 16] |

---

## Mac Smoke Test Protocol

### Run Tests
```bash
cd /Users/harshitatolani/Sahil/ML
python3 -m pytest tests/test_phase3_curiosity.py -v
```

### Manual Collection Smoke Run (300 transitions, CPU, write to /tmp)
```bash
python3 training/collector_v3.py \
    --model_checkpoint checkpoints/jepa_crafter_v3_phase2.pt \
    --drive_dir /tmp/smoke_collect_p3 \
    --target_transitions 300 \
    --seed_start 5000 \
    --tranche_id 1 \
    --device cpu

# Verify manifest
python3 -c "import json; d=json.load(open('/tmp/smoke_collect_p3/dataset_index.json')); print(json.dumps(d, indent=2))"

# Verify shard structure
python3 -c "import numpy as np; d=np.load('/tmp/smoke_collect_p3/shard_curiosity_v3_t01_0000.npz',allow_pickle=True); print({k: d[k].shape for k in d})"
```

**Mac gate PASSED when:** 6/6 tests green + shard loads + manifest shows correct transition count.

---

## Colab Run 2 Execution Plan (T1)

### Setup Cell
```python
from google.colab import drive
drive.mount('/content/drive')

# Pull latest code
import subprocess
result = subprocess.run(['git', 'pull', 'origin', 'main'],
                       capture_output=True, text=True,
                       cwd='/content/learn-a-world')
print(result.stdout)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.cuda.is_available()}")

# Copy v3 checkpoint from Drive
import shutil, os
drive_ckpt = '/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_phase2.pt'
local_ckpt = '/content/learn-a-world/checkpoints/jepa_crafter_v3_phase2.pt'
os.makedirs('/content/learn-a-world/checkpoints', exist_ok=True)
if not os.path.exists(local_ckpt):
    shutil.copy(drive_ckpt, local_ckpt)
```

### Stage 1: T1 Collection (~90–120 min on T4 GPU)
```python
import sys
sys.path.insert(0, '/content/learn-a-world')
from training.run_v3_colab_run2 import collect_t1

manifest = collect_t1(
    model_checkpoint='/content/learn-a-world/checkpoints/jepa_crafter_v3_phase2.pt',
    curiosity_checkpoint=None,  # or path if resuming
    drive_dir='/content/drive/MyDrive/learn-a-world',
    target_new_transitions=160_000,
    seed_start=5000,
    device='cuda',
)
print(f"T1 collected: {manifest['new_transitions']} transitions")
print(f"Frontier counts: stone={manifest['event_counts']['collect_stone']}, "
      f"coal={manifest['event_counts']['collect_coal']}, "
      f"iron={manifest['event_counts']['collect_iron']}")
```

### Stage 2: Incremental Retrain (~45–60 min on T4 GPU)
```python
from training.run_v3_colab_run2 import retrain_t1

ckpt_path = retrain_t1(
    base_checkpoint='/content/learn-a-world/checkpoints/jepa_crafter_v3_phase2.pt',
    drive_dir='/content/drive/MyDrive/learn-a-world',
    num_steps=5_000,       # retrain steps on ALL data
    save_every=500,
    device='cuda',
)
print(f"T1 retrained checkpoint: {ckpt_path}")
```

### Stage 3: T1 Screening (50 eps, seeds 2050–2099)
```bash
# Run from terminal in Colab
!python3 scripts/screen_v3_50eps.py \
    --checkpoint $T1_CKPT_PATH \
    --num_episodes 50 \
    --seed_start 2050 \
    --output_json results/screen_v3_t1_results.json

# Copy result to Drive BEFORE any prose
!cp results/screen_v3_t1_results.json /content/drive/MyDrive/learn-a-world/results/

# T1 Gate check
!python3 -c "
import json
v3_pre = json.load(open('results/screen_v3_50eps_results.json'))['crafter_score']
v3_t1  = json.load(open('results/screen_v3_t1_results.json'))['crafter_score']
gate = v3_t1 > v3_pre
print(f'T1 Gate: {v3_t1:.4%} > {v3_pre:.4%} (pre-Phase-3) => {\"PASSED\" if gate else \"FAILED — STOP\"}')"
```

### Colab Logs Expected
Every 10 episodes during collection:
```
[T1-Ep 010/800] Seed 5009 | Steps: 284 | r_task: 2.10 | r_cur: 0.043 | 
                Ach (3): collect_wood, collect_sapling, wake_up
[T1-Ep 010] Curiosity loss: 0.0821 | Ensemble std: 0.0031
```
Every 1,000 collection steps:
```
[CHECKPOINT] Saved full state to Drive: step_001000.pt (model + curiosity + opt + rng)
```

---

## T1 Gate Summary

| Gate | Metric | Requirement | What to Do if FAILED |
|:---|:---|:---|:---|
| **T1 Screening** | v3_t1 score, seeds 2050–2099 | > **2.31%** | STOP; write diagnosis note — do not build actor |
| **T1 Frontier sanity** | `collect_stone` count from manifest | > 0 (sanity) | Log plainly; curiosity may not be reaching stone terrain |

Full final gates (all tranches done, before Phase 4):

| Gate | Metric | Requirement |
|:---|:---|:---|
| **Frontier** | Cumulative manifest | `collect_stone` ≥ 5,000, `collect_coal` ≥ 500, `collect_iron` > 0 |
| **D1 re-run** | Pearson r (seeds 4000–4049) | Watch: must move above 0.30 |
| **D3 re-run** | Head audit on planner data | Compute on new data, not old 8k slice |
| **Actor ablation** | actor+CEM vs CEM-only (seeds 2050–2099) | Actor ships if neutral-positive |
| **Locked final** | 100 eps, seeds 3000–3099 | v3 > v2 AND v3 ≥ **3.0%** |

---

## Commit Protocol (Non-Negotiable)

```
RULE: Push all result files BEFORE writing any prose summary.
RULE: Every number in prose must exist in a pushed file.
RULE: If not run → write "not run yet" — never estimate.
RULE: Screening (50 eps, 2050–2099) ≠ Locked (100 eps, 3000–3099) — always label.
```

---

## Implementation Order Checklist

- [x] **Mac:** Add `Plan2ExploreCuriosity` to `models/curiosity.py`
- [x] **Mac:** Write `tests/test_phase3_curiosity.py` (6 tests)
- [x] **Mac:** `python3 -m pytest tests/test_phase3_curiosity.py -v` → 6/6 green
- [x] **Mac:** Write `training/collector_v3.py`
- [x] **Mac:** Write `training/run_v3_colab_run2.py`
- [x] **Mac:** Manual smoke run (300 transitions → /tmp/smoke_collect_p3/)
- [x] **Mac:** Verify shard + manifest structure
- [x] **Mac:** Verify resume logic in practice (seed skip, shard indexing, cumulative episodes)
- [ ] **Mac:** `git add -A && git commit -m "feat: phase3 curiosity + collector + smoke tests"`
- [ ] **Mac:** `git push origin main`
- [ ] **Colab:** Pull latest, copy v3 checkpoint
- [ ] **Colab:** Stage 1 — T1 collection (160k transitions, seeds 5000+)
- [ ] **Colab:** Shards + manifest saved to Drive
- [ ] **Colab:** Stage 2 — Incremental retrain on ALL data
- [ ] **Colab:** Stage 3 — T1 screening, save to Drive
- [ ] **Colab:** `git push` results before writing any prose
- [ ] **Colab:** T1 Gate check (score > 2.31%?)
- [ ] **Post-T1:** Download `screen_v3_t1_results.json` + `dataset_index.json`
- [ ] **Post-T1:** Report T1 results (honest numbers, no estimates)
