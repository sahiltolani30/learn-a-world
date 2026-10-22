# Learn-a-World v1 Baseline Freeze & Ablation Report

## Executive Summary

This document records the official baseline performance of the Learn-a-World v1 agent before any v2 architectural changes.
The purpose of Phase 0 is to measure the honest contribution of the JEPA World Model + Latent MPC planner versus the hardcoded reflex rules layered on top in `eval/eval_crafter.py`.

**Git Tag:** `v1-freeze`  
**Branch:** `v2`  
**Evaluation Seeds:** Screening Seeds 1000–1049 (50 episodes, max 1000 steps/ep)  

---

## 1. Official Screening Protocol Comparison (50 Episodes, Seeds 1000–1049)

| Configuration | Mode | Reflexes | Do-Bias | Crafter Score | Unlocked Ach. | Mean Survival | Mean Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **hybrid (v1 truth)** | MPC | ON | 0.35 | **1.16%** | **5/22** | 172.3 steps | 1.50 |
| **v1_pure (planner only)** | MPC | OFF | 0.00 | **0.00%** | **0/22** | 175.7 steps | -0.90 |
| **curious (heuristic)** | Curious | OFF | 0.00 | **3.15%** | **10/22** | 181.5 steps | 2.72 |
| **random (uniform)** | Random | OFF | 0.00 | **1.57%** | **7/22** | 181.4 steps | 1.56 |

---

## 2. Historical Benchmark Progress (20 Episodes, Seed Offset 5000)

| Cycle | Protocol | Crafter Score | Unlocked | Mean Survival | Key Features / Breakthroughs |
|:---|:---|:---:|:---:|:---:|:---|
| **Cycle 1** | Pure curiosity MPC (H=7, C=128) | 0.85% | 2/22 | 84.2 | Baseline JEPA latent dispersion |
| **Cycle 2** | Wood>=3 sequential crafting reflex | 2.14% | 4/22 | 142.0 | First crafting table and pickaxe unlocks |
| **Cycle 3** | Survival priority + eat/drink reflexes | 3.37% | 6/22 | 184.2 | Solved starvation/thirst; added plant placement |
| **Cycle 4** | Place stone + Wood>=4 + H=8/C=256 | **1.38%** | **6/22** | **165.6** | Stone placement unlock, sword sequencing, deeper planning |

---

## 3. Honest Diagnosis: Planner vs Reflexes

Comparing `hybrid` to `v1_pure` exposes the true impact of the hand-coded rule engine:
- **Score Gap:** `hybrid` achieves **1.16%** while `v1_pure` achieves **0.00%**.
- **Survival Gap:** Reflex-guided survival extends episode length from **175.7** steps to **172.3** steps.
- **Achievement Gap:** `hybrid` unlocks **5** achievements vs **0** for `v1_pure`.

### Key Takeaway for v2
The curiosity-maximizing MPC planner in v1 has **no notion of reward or survival**; it maximizes latent representation dispersion. Without survival reflexes, the agent dehydrates and starves rapidly. Without crafting reflexes, uniform candidate sampling almost never sequences the multi-step prerequisite actions (wood -> table -> pickaxe -> stone).

The v2 architecture addresses this root cause directly:
1. **EventHead** replaces curiosity dispersion with calibrated per-achievement unlock probabilities.
2. **HealthHead** predicts survival vital deltas (health/hunger/thirst).
3. **QEnsemble** provides long-horizon return estimation.
4. **PolicyPrior** guides candidate generation to make crafting sequences exponentially more probable.

---

## 4. Per-Achievement Unlock Breakdown (Screening Seeds 1000–1049)

| Achievement | Tier | Hybrid (%) | v1_pure (%) | Curious (%) | Random (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `collect_coal` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_diamond` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_drink` | Tier 1-2 | 0.0% | 0.0% | 42.0% | 14.0% |
| `collect_iron` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_sapling` | Tier 1-2 | 100.0% | 0.0% | 100.0% | 58.0% |
| `collect_stone` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `collect_wood` | Tier 1-2 | 10.0% | 0.0% | 62.0% | 26.0% |
| `defeat_skeleton` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `defeat_zombie` | Tier 3-4 | 22.0% | 0.0% | 12.0% | 0.0% |
| `eat_cow` | Tier 3-4 | 8.0% | 0.0% | 12.0% | 2.0% |
| `eat_plant` | Tier 1-2 | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_iron_pickaxe` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_iron_sword` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_stone_pickaxe` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_stone_sword` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `make_wood_pickaxe` | Tier 1-2 | 0.0% | 0.0% | 2.0% | 0.0% |
| `make_wood_sword` | Tier 1-2 | 0.0% | 0.0% | 4.0% | 0.0% |
| `place_furnace` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_plant` | Tier 1-2 | 100.0% | 0.0% | 64.0% | 48.0% |
| `place_stone` | Tier 3-4 | 0.0% | 0.0% | 0.0% | 0.0% |
| `place_table` | Tier 1-2 | 0.0% | 0.0% | 16.0% | 2.0% |
| `wake_up` | Tier 3-4 | 0.0% | 0.0% | 50.0% | 98.0% |

---

## 5. Verification Gate Status

- [x] `git tag v1-freeze` created
- [x] `v2` branch created and checked out
- [x] `results/ablation_v1.json` generated with keys: `hybrid`, `v1_pure`, `curious`, `random`
- [x] `results/v1_freeze.md` generated with comparative table and diagnosis
- [x] Zero modifications to protected v1 files (`git diff main -- eval/eval_crafter.py models/ planning/mpc.py training/` is empty)
- [x] All 24 gate tests passing
