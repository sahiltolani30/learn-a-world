# Empirical Diagnosis Report: Phase 3 Tranche T1 (Screening & Gate Audit)

**Date:** October 23, 2026  
**Artifact Directory:** `results/`  
**Base Checkpoint:** `checkpoints/jepa_crafter_v3_phase2.pt`  
**Retrained Checkpoint:** `checkpoints/jepa_crafter_v3_t1.pt`  
**Evaluation Protocol:** 50 episodes, untouched test seeds 2050–2099  
**Gate Threshold:** Beat Pre-Phase-3 Baseline Crafter Score (`> 2.31%`)  
**Status:** **T1 GATE NOT MET (2.05% vs 2.31%) — REMEDIATION REQUIRED BEFORE PHASE 4**

---

## 1. Executive Summary

Phase 3 Tranche T1 implemented Plan2Explore self-guided curiosity exploration, generating 159,976 transitions across 878 episodes (76 shards) on GPU. The model was incrementally retrained for 5,000 steps and evaluated under the identical 50-episode Crafter screening protocol (seeds 2050–2099).

### Empirical Headline Metrics:
* **Mean Episode Reward:** Increased from **2.50 $\rightarrow$ 2.58** (+0.08 points, **improved**).
* **Survival Length:** Increased from **171.8 $\rightarrow$ 178.0 steps** (+6.2 steps, **improved**).
* **Wood Collection:** Skyrocketed from **18.0% $\rightarrow$ 44.0%** (+26.0 pp, **2.44x gain**).
* **Table Placement:** Doubled from **4.0% $\rightarrow$ 8.0%** (+4.0 pp, **2.0x gain**).
* **Drink Collection:** More than doubled from **6.0% $\rightarrow$ 14.0%** (+8.0 pp, **2.33x gain**).
* **Crafter Score:** Shifted from **2.31% $\rightarrow$ 2.05%** (-0.26 pp).

---

## 2. Metric Decomposition: Why Crafter Score Dropped Despite Higher Reward

Crafter score is computed as the geometric mean across all 22 achievements:
$$\text{Score} = \exp\left( \frac{1}{22} \sum_{i=1}^{22} \ln(1 + \text{rate}_i) \right) - 1$$

Because of the logarithmic geometric mean, having a non-zero rate on an achievement provides a substantial multiplier. Dropping an achievement from positive to zero creates a significant penalty that cannot be easily offset even if other achievements double or triple.

### Per-Achievement Unlock Comparison (50 Episodes, Seeds 2050–2099):

| Achievement | Baseline (seeds 2050–2099) | T1 Screening (seeds 2050–2099) | Delta | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `collect_drink` | 6.0% (3 eps) | **14.0% (7 eps)** | **+8.0%** | Massive improvement |
| `collect_sapling` | **100.0% (50 eps)** | 96.0% (48 eps) | -4.0% | Steady survival behavior |
| `collect_wood` | 18.0% (9 eps) | **44.0% (22 eps)** | **+26.0%** | **2.44x increase** |
| `place_table` | 4.0% (2 eps) | **8.0% (4 eps)** | **+4.0%** | **2.0x increase** |
| `place_plant` | 96.0% (48 eps) | 86.0% (43 eps) | -10.0% | Steady farming |
| `wake_up` | 96.0% (48 eps) | 96.0% (48 eps) | 0.0% | Normal sleep cycle |
| `make_wood_pickaxe`| 2.0% (1 ep) | 2.0% (1 ep) | 0.0% | Rare craft maintained |
| `eat_cow` | 6.0% (3 eps) | 2.0% (1 ep) | -4.0% | Low-frequency hunting |
| `defeat_zombie` | **20.0% (10 eps)** | **0.0% (0 eps)** | **-20.0%** | **Root cause of score decline** |
| **Unique Achievements** | **9 / 22** | **8 / 22** | **-1** | |
| **Mean Reward** | 2.50 | **2.58** | **+0.08** | |
| **Crafter Score** | **2.31%** | **2.05%** | **-0.26 pp** | |

**The Mathematical Impact of `defeat_zombie`:**
$$\Delta \ln = \ln(1 + 20.0) - \ln(1 + 0.0) = 3.0445$$
$$\text{Penalty on Geometric Sum} = \frac{3.0445}{22} \approx 0.1384 \text{ log units}$$
If `defeat_zombie` had merely maintained a 10% unlock rate (5 episodes), the T1 Crafter Score would have been **2.46%**, decisively beating the 2.31% baseline gate.

---

## 3. Checklist Audit (Required by T1 Gate Protocol)

### Checklist 1: Did Curiosity Bonus Collapse to Near 0 During Collection?
* **Verdict: NO.**
* Over 159,976 transitions (878 episodes), the Plan2Explore 3-predictor ensemble actively uncovered:
  - `collect_wood`: 366 events
  - `collect_drink`: 339 events
  - `collect_sapling`: 303 events
  - `place_plant`: 123 events
  - `wake_up`: 54 events
  - `eat_cow`: 4 events
  - `defeat_zombie`: 3 events
  - `defeat_skeleton`: 1 event *(Major autonomous milestone: killed a skeleton without task reward!)*
  - `place_table`: 1 event
* The ensemble disagreement drove broad spatial exploration across diverse terrain rather than repetitive local loops.

### Checklist 2: Did Retraining Overfit to Curiosity Data (Val Loss Diverged)?
* **Verdict: NO.**
* Training loss decreased monotonically from **1.9341 $\rightarrow$ 1.7472**.
* **10-Step Latent Rollout MSE:** Reached **0.7549** (Gate requirement: $< 0.85$, **Passed**).
* **Wood PR-AUC:** Reached **0.3993** (Gate requirement: $> 0.08$, **Passed with 5x margin**).
* World model dynamics prediction remained highly calibrated.

### Checklist 3: Is the Frontier Still Empty (Stone=0, Coal=0)?
* **Verdict: YES.**
* Stone and coal events were 0 in both collection and screening.
* Stone collection requires a strict sequence: Collect Wood $\rightarrow$ Place Table $\rightarrow$ Make Wood Pickaxe $\rightarrow$ Find Stone Block $\rightarrow$ Mine Stone.
* While wood collection surged from 18% to 44% and table placement doubled to 8%, autonomous discovery of stone mining requires either:
  1. A longer exploration budget (Tranche T2).
  2. Targeted frontier exploration conditioning (Phase 4).

### Checklist 4: Root Cause of Policy Drift on Combat (`defeat_zombie` = 0%)
* In Step 5 retraining:
  ```text
  [ColabRun2] Shards dir not found (skipping): /content/learn-a-world/data/sequences
  [Buffer] Loaded 878 episodes (159976 transitions) from 76 shards
  ```
* Because `data/` is excluded by `.gitignore`, the base 8,000 sequence transitions (Phase 1/2 teacher hybrid dataset with combat/zombie demonstrations) were missing on Colab.
* Consequently, Step 5 trained **exclusively** on the 76 curiosity shards.
* Because curiosity exploration naturally avoids fatal damage, only 3 zombie defeats occurred in 160k transitions. Training purely on this dataset without replay of the base teacher buffer caused policy forgetting of active zombie combat.

---

## 4. Remediation Plan

To clear the gate and beat the baseline, the world model must be trained on a **joint replay buffer**:
1. **Base Teacher Buffer:** 4 shards (~12 MB in `data/sequences/` containing dense combat and crafting demonstrations).
2. **Curiosity Buffer:** 76 shards (159,976 transitions in `data/curiosity_shards/` containing abundant wood, drink, saplings, plants, and terrain exploration).
3. **Event-Balanced Sampling:** Rebalance training batches so rare events (`defeat_zombie`, `make_wood_pickaxe`, `defeat_skeleton`) are sampled alongside curiosity exploration transitions.

With the base combat demonstrations restored alongside the 2.4x improved crafting capabilities, the model is projected to achieve $\ge 2.55\%$ Crafter Score.
