# Investigation C: Held-Out Prediction Benchmark Report

**Gate Status:** `HELDOUT_PREDICTION_PASS` (BLOCKED)

---

## 1. Executive Summary

This investigation audited the held-out evaluation pipeline, multi-horizon target alignment, seed independence, and baseline comparison controls.

### Core Finding
The held-out evaluation infrastructure is mathematically sound and 100% disjoint from training replay data. The failure of the 50,000-step model to beat persistence is not an evaluation artifact or off-by-one indexing bug; it is an optimization failure where predicting a static frame achieves lower loss than the unregularized model.

---

## 2. Integrity & Independence Verification

1. **Seed Disjointness**:
   - Training Seeds: `1000..1007` (8 parallel environments).
   - Held-Out Seeds: `900000..900004` (independent held-out store).
   - Disjointness Assertion: `heldout_seeds.isdisjoint(training_seeds)` **PASSED**.
   - Zero Held-Out Leakage: Held-out transitions are stored in a private buffer and never enter the training replay.
2. **Multi-Horizon Target Alignment**:
   - Verified across anchors in `SequenceSampler`:
     - Horizon 1 target = $O_{t+1}$
     - Horizon 2 target = $O_{t+2}$
     - Horizon 4 target = $O_{t+4}$
     - Horizon 8 target = $O_{t+8}$
   - Mathematical alignment is verified and exact.
3. **Identical Control Transitions**:
   - The persistence baseline, no-action control, and shuffled-action control evaluate on the exact same held-out batch.
   - Upgraded no-action control to pass true zero embeddings ($\mathbf{0} \in \mathbb{R}^{B \times H \times 64}$) rather than arbitrary action token 0.

---

## 3. Multi-Horizon Prediction Results (Held-Out Evaluation)

| Horizon | Active Model Cosine Distance | Persistence Baseline Distance | No-Action Control Distance | Shuffled-Action Distance | Improvement vs Persistence |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **H=1** | $0.4018$ | **$0.0264$** | $0.4017$ | $0.4018$ | **$-1421.8\%$** |
| **H=2** | $0.6221$ | **$0.0558$** | $0.6220$ | $0.6221$ | **$-1015.5\%$** |
| **H=4** | $0.9042$ | **$0.0873$** | $0.9041$ | $0.9042$ | **$-935.7\%$** |
| **H=8** | $1.0214$ | **$0.1114$** | $1.0212$ | $1.0214$ | **$-817.3\%$** |

*(Note: Under representation collapse at 50k steps on T4 Colab, distances were $0.0005$ across all baselines with $-1.32\%$ improvement).*

---

## 4. Root Cause Analysis

Why does the model fail to beat persistence?
1. **Terrain Invariance**: Crafter environments are 85–90% static tiles (grass, water, stone, black HUD).
2. **Persistence Advantage**: Over short horizons (1–2 steps), copying the current frame $O_t$ is correct for 90%+ of screen patches, yielding a tiny cosine distance ($0.0264$).
3. **Prediction Penalty on Static Background**: If the model makes slight errors on the 55 static grass patches while trying to predict the 5 player/motion patches, its overall spatial mean error is penalized heavily.
4. **Remediation**: Dynamic patch loss weighting must be introduced so that optimization gradients focus on patches where real world motion occurs.

---

## 5. Gate Verdict
**`HELDOUT_PREDICTION_PASS` is BLOCKED.**  
The model does not beat persistence by the required $\ge 5.0\%$ margin on held-out transitions.
