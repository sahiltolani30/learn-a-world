# Phase 1.5 Colab Execution Review & Diagnostic Audit

**Date**: October 27, 2026  
**Artifact**: `checkpoints/jepa_crafter_v3_phase1_5.pt`  
**Command Executed**:
```bash
python3 scripts/run_phase1_5.py \
  --data_root /content/drive/MyDrive/learn-a-world/data \
  --manifest /content/drive/MyDrive/learn-a-world/results/phase1_5_manifest.json \
  --checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_t2_final.pt \
  --output_checkpoint /content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_phase1_5.pt \
  --output_json /content/drive/MyDrive/learn-a-world/results/phase1_5_training.json \
  --steps 1000 \
  --batch_size 16 \
  --seq_len 51 \
  --horizons 1 5 10 25 50 \
  --split_seed 17 \
  --device cuda
```

---

## 1. Verified Execution Summary

| Metric / Property | Result | Assessment |
| :--- | :---: | :--- |
| **Deduplicated Dataset** | **666,338 transitions** (2,629 episodes) | Successfully loaded via `phase1_5_manifest.json` |
| **Long Horizon Supervised** | $H \in \{1, 5, 10, 25, 50\}$ | Multi-step consistency up to 50 steps |
| **JEPA Latent Loss** | $1.0143 \to 0.3484$ | **PASSED** (Stable convergence, no representation collapse) |
| **Termination Accuracy** | **99.98%** | **PASSED** (Done signals highly calibrated) |
| **Counterfactual Separation** | **0.000547** | **PASSED** (Actions produce distinct latent trajectories) |
| **State-Delta MSE vs Persistence** | $0.01006$ vs $0.00735$ | **UNDERPERFORMED** (Did not beat persistence baseline) |
| **State-Delta Labeled Valid Count** | 51 validation transitions | Extreme sparsity in validation set |

---

## 2. Root Cause Analysis: Label Sparsity Dilution

In the earlier Phase 1 smoke test (Cell 4 & 5), the model trained exclusively on `data/phase1_transitions/` (where 100% of transitions had valid state-delta labels). In that test, `state_delta_mse` (0.01468) beat persistence (0.01688) by 13%.

In Phase 1.5, when pooling the entire 666,338 transition dataset:
1. **0.30% Label Density**: Only 2,000 out of 666,338 transitions have `state_delta_valid = True`. The remaining 664,338 transitions are legacy shards without state-delta annotations.
2. **Gradient Starvation**: Because uniform random sampling was used, batches containing state-delta labels were drawn **only twice in 1,000 steps** (at Step 400 with `delta=0.0152`, and Step 800 with `delta=0.0121`). For 998 out of 1,000 steps, `loss_state_delta` was literally `0.0000`.
3. **Sparse Validation**: In the 10% held-out validation split, only 51 transitions had valid state deltas.

---

## 3. Recommended Actions for Omni

To satisfy the Phase 1 Gate ($\text{MSE}_{\text{delta}} < \text{MSE}_{\text{persistence}}$) across the full dataset:

1. **Balanced Batch Sampling (Recommended)**:
   In `training/phase1_5_data.py` / `training/sequence_buffer.py`, sample 50% of each batch from shards with `has_state_delta = True`, and 50% from the general dataset. This ensures every gradient step updates both the JEPA latent core and the `StateDeltaHead`.
2. **Collect Additional Labeled Transitions**:
   Run `collector_v3.py` for 20,000 to 50,000 transitions so state-delta annotations reach at least 5% to 10% of total transitions.
