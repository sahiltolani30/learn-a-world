# Learn-a-World v2: Promotion Decision Report

**Status:** ❌ **PROMOTION FAILED**  
**Date:** 2026-10-22 10:58:38  
**Evaluation Protocol:** 2 episodes on fixed seeds 1000..1001 (max 15 steps/ep)  

---

## 1. Executive Summary & Comparison Table

| Agent | Evaluator | Checkpoint | Reflexes | Crafter Score | Unlocked Ach. | Mean Survival | Mean Reward |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **v1_hybrid** | `eval/eval_crafter_v1_ablation.py` | `jepa_crafter.pt` | on | **0.52%** | 2/22 | 15.0 | 1.90 |
| **v1_pure** | `eval/eval_crafter_v1_ablation.py` | `jepa_crafter.pt` | off | **0.00%** | 0/22 | 15.0 | 0.00 |
| **v2_pure** | `eval/eval_v2.py` | `jepa_crafter_v2.pt` | none (pure learned) | **0.00%** | 0/22 | 15.0 | 0.00 |
| **random** | `eval/eval_crafter_v1_ablation.py` | `none` | none | **0.43%** | 2/22 | 15.0 | 1.00 |

---

## 2. Promotion Criteria Audit

The v2 promotion rule requires satisfaction of both clauses (§6 Step 4):
> **Rule:** `v2_pure_score > v1_hybrid_score` **AND** `v2_pure_score >= v1_pure_score + 1.0`

- **Clause 1 (`v2_pure_score > v1_hybrid_score`):**  
  `0.00% > 0.52%` -> **FAIL ❌** (Margin: `-0.52%`)
- **Clause 2 (`v2_pure_score >= v1_pure_score + 1.0`):**  
  `0.00% >= 1.00%` -> **FAIL ❌** (Margin over pure: `+0.00%`)

**Overall Verdict:** ❌ **PROMOTION FAILED**

> [!WARNING]
> Phase 4 promotion did not trigger. Per the v2 implementation plan, the next step is to activate **Phase 5: Hierarchical Achievement Layer** to mine the achievement dependency graph and guide the planner with explicit subgoal unlocking bonuses.

---

## 3. Achievement Breakdown by Agent

| Achievement | v1_hybrid (%) | v1_pure (%) | v2_pure (%) | random (%) |
|:---|:---:|:---:|:---:|:---:|
| `collect_coal` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_diamond` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_drink` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_iron` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_sapling` | 100.0% | 0.0% | 0.0% | 50.0% |
| `collect_stone` | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_wood` | 0.0% | 0.0% | 0.0% | 0.0% |
| `defeat_skeleton` | 0.0% | 0.0% | 0.0% | 0.0% |
| `defeat_zombie` | 0.0% | 0.0% | 0.0% | 0.0% |
| `eat_cow` | 0.0% | 0.0% | 0.0% | 0.0% |
| `eat_plant` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_iron_pickaxe` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_iron_sword` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_stone_pickaxe` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_stone_sword` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_wood_pickaxe` | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_wood_sword` | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_furnace` | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_plant` | 100.0% | 0.0% | 0.0% | 50.0% |
| `place_stone` | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_table` | 0.0% | 0.0% | 0.0% | 0.0% |
| `wake_up` | 0.0% | 0.0% | 0.0% | 0.0% |
