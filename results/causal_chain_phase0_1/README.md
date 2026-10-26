# Causal Chain Learning: Phase 0 & Phase 1 Execution Artifacts

This directory contains the verified execution logs, diagnostic summaries, and validation metrics from the Colab screening run executed on October 27, 2026.

---

## Directory Manifest

| File | Cell | Description |
| :--- | :---: | :--- |
| `cell2_phase0_chain_diagnostics.json` | 2 | Complete step-level failure trace across 5 controller ablations (4.1 MB) |
| `cell3_collector_log.txt` | 3 | Full console output from curiosity-guided trajectory collection |
| `cell3_collector_v3_summary.json` | 3 | Parsed JSON summary of 2,000 transitions & milestone counts |
| `cell4_phase1_training_log.txt` | 4 | Step-by-step training convergence log (1,000 steps, $H \in \{1, 5, 10\}$) |
| `cell5_phase1_verification_metrics.json` | 5 | Final verification metrics and pass/fail gate checks |

---

## 1. Phase 0 Diagnostic Findings (Cell 2)

**Setup**: 3 episodes across 5 controller configurations on identical seeds (2050, 2051, 2052), max 300 steps.

### Failure Taxonomy Across Ablations

| Ablation Configuration | Mean Reward | Primary Broken Link | Detailed Breakdown |
| :--- | :---: | :---: | :--- |
| `strict_no_memory` | 2.43 | `table_execution` | 3/3 episodes failed at `table_execution` |
| `strict_empty_memory` | 3.43 | Mixed | 1 $\times$ `table_execution`, 1 $\times$ `pickaxe_execution`, 1 $\times$ `wood_acquisition` |
| `strict_preloaded_memory` | 3.10 | `table_execution` | 2 $\times$ `table_execution`, 1 $\times$ `wood_acquisition` |
| `strict_landmarks_off` | 1.77 | Mixed | 1 $\times$ `table_execution`, 1 $\times$ `wood_acquisition`, 1 $\times$ `pickaxe_execution` |
| `strict_policy_prior_off` | 2.43 | `table_execution` | 2 $\times$ `table_execution`, 1 $\times$ `wood_acquisition` |

### Key Takeaways:
- **60% of failures (9/15 episodes)** occur at **`table_execution`**: the agent gathers wood but fails to place the table or face it properly.
- **27% of failures (4/15 episodes)** occur at **`wood_acquisition`**: failing to gather sufficient wood before night or death.
- **13% of failures (2/15 episodes)** occur at **`pickaxe_execution`**: table is placed, but craft action is not triggered.
- **0% reach Stone**: Stone is never reached because the chain consistently breaks at Table or Pickaxe.

---

## 2. Transition Data Collection (Cell 3)

- **Transitions Collected**: 2,000 transitions across 13 episodes (Seeds 5000–5012).
- **Tranche**: T1 curiosity-guided exploration.
- **Shard Written**: `shard_curiosity_v3_t01_0000.npz`.
- **Milestones Captured**:
  - `collect_wood`: 11
  - `place_table`: 5
  - `place_plant`: 6
  - `collect_sapling`: 8
  - `collect_drink`: 5
  - `wake_up`: 12

---

## 3. Phase 1 Transition Learning (Cell 4 & 5)

- **Model**: `JEPAv3` with auxiliary `StateDeltaHead`.
- **Horizons Supervised**: 1, 5, 10 steps.
- **Loss Convergence**:
  - Initial Loss: $1.4144$ (JEPA: $0.5449$, Delta: $0.0148$)
  - Final Loss: $-0.2398$ (JEPA: $0.1352$, Delta: $0.0050$)

### Gate Verification Sign-Off

| Gate Requirement | Threshold | Measured Result | Verdict |
| :--- | :---: | :---: | :---: |
| **1. JEPA Objective Active** | Latent loss $> 0$ (no collapse) | $0.5449 \to 0.1352$ | **PASS** |
| **2. State Delta vs Persistence** | $\text{MSE} < \text{Persistence}$ | **$0.01468 < 0.01688$** (13% error reduction) | **PASS** |
| **3. Counterfactual Separation** | $\text{Separation} > 0$ | **$0.00048$** | **PASS** |
| **4. Termination Prediction** | Accuracy $> 99\%$ | **$99.98\%$** | **PASS** |
| **5. Valid Transitions** | $\ge 1,000$ transitions | **$4,096$ evaluated** | **PASS** |

**Conclusion**: All 5 verification gates have passed. The foundation is ready for Phase 2 causal graph search.
