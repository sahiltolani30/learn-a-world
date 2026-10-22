# Learn-a-World v2: Promotion Decision Report

**Status:** ✅ **PROMOTED TO V2**  
**Date:** 2026-10-22 18:01:27  
**Evaluation Protocol:** 100 episodes on fixed seeds 1000..1099 (max 1000 steps/ep)  

---

## 1. Executive Summary & Comparison Table

| Agent | Evaluator | Checkpoint | Reflexes | Crafter Score | Unlocked Ach. | Mean Survival | Mean Reward |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **v1_hybrid** | `eval/eval_crafter_v1_ablation.py` | `jepa_crafter.pt` | on | **1.44%** | 7/22 | 179.5 | 1.62 |
| **v1_pure** | `eval/eval_crafter_v1_ablation.py` | `jepa_crafter.pt` | off | **0.00%** | 0/22 | 177.0 | -0.90 |
| **v2_pure** | `eval/eval_v2.py` | `jepa_crafter_v2_best.pt` | none (pure learned) | **2.16%** | 10/22 | 171.1 | 2.09 |
| **random** | `eval/eval_crafter_v1_ablation.py` | `none` | none | **1.77%** | 7/22 | 188.7 | 1.75 |

---

## 2. Promotion Criteria Audit

The v2 promotion rule requires satisfaction of both clauses (§6 Step 4):
> **Rule:** `v2_pure_score > v1_hybrid_score` **AND** `v2_pure_score >= v1_pure_score + 1.0`

- **Clause 1 (`v2_pure_score > v1_hybrid_score`):**  
  `2.16% > 1.44%` -> **PASS ✅** (Margin: `+0.72%`)
- **Clause 2 (`v2_pure_score >= v1_pure_score + 1.0`):**  
  `2.16% >= 1.00%` -> **PASS ✅** (Margin over pure: `+2.16%`)

**Overall Verdict:** ✅ **PROMOTED TO V2**

> [!NOTE]
> Phase 4 promotion passed! The pure learned model without hardcoded heuristic reflexes successfully surpassed the hybrid baseline under the locked evaluation protocol.

---

## 3. Achievement Breakdown by Agent

| Achievement | v1_hybrid (%) | v1_pure (%) | v2_pure (%) | random (%) |
|:---|:---:|:---:|:---:|:---:|
| `collect_coal` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_diamond` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_drink` | 1.0% | 0.0% | 0.0% | 6.0% |
| `collect_iron` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_sapling` | 98.0% | 0.0% | 100.0% | 77.0% |
| `collect_stone` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_wood` | 19.0% | 0.0% | 27.0% | 13.0% |
| `defeat_skeleton` | 0.0% | 0.0% | 1.0% | 0.0% |
| `defeat_zombie` | 23.0% | 0.0% | 9.0% | 19.0% |
| `eat_cow` | 11.0% | 0.0% | 10.0% | 6.0% |
| `eat_plant` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_iron_pickaxe` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_iron_sword` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_stone_pickaxe` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_stone_sword` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_wood_pickaxe` | 0.0% | 0.0% | 1.0% | 0.0% |
| `make_wood_sword` | 0.0% | 0.0% | 1.0% | 0.0% |
| `place_furnace` | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_plant` | 98.0% | 0.0% | 69.0% | 52.0% |
| `place_stone` | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_table` | 2.0% | 0.0% | 5.0% | 0.0% |
| `wake_up` | 0.0% | 0.0% | 96.0% | 93.0% |
