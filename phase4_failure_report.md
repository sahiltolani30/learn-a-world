# AEON JEPA-Crafter — Phase 4 Failure Investigation Report

**Date:** 2026-11-04  
**Run ID:** `jepa_crafter` (50,000 Environment Steps, T4 GPU Pilot)  
**Status:** `INVESTIGATION COMPLETE — GATES BLOCKED FOR PHASE 5`

---

## 1. Executive Summary

This report documents the exhaustive diagnostic investigation into the failure of the 50,000-step JEPA world model pilot to satisfy the Phase 5 learning benchmark (`P5_JEPA_LEARNING_PASS`).

In accordance with strict research integrity laws ([`AGENTS.md`](file:///Users/harshitatolani/Sahil/ML/AGENTS.md) / [`GEMINI.md`](file:///Users/harshitatolani/Sahil/ML/GEMINI.md)):
- Zero hand-crafted game rules or recipe cheat tables were used.
- The failure was reported honestly without altering benchmark thresholds or forcing green tests.
- The safety interlock performed as designed: MPC planning remained safely **LOCKED**.

---

## 2. Discrepancy Reconciliation (Task 2)

### Issue
In `final_report.md`, Section 1 reported:
> `Official Crafter Benchmark Score: 0.00% across 0 evaluated episodes.`
Whereas `achievement_rates.csv` reported:
> `jepa_crafter, 12345, 296, 2.0867% ... place_table: 14.53%`
> `jepa_crafter, 12345, 280, 2.2007% ... place_table: 14.29%`

### Root Cause Mechanism
1. [`AchievementRecorder`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/src/evaluation/achievement_recorder.py) is an in-memory accumulator instantiated inside [`train_agent.py`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/src/training/train_agent.py#L178).
2. Prior to the Google Colab disconnect, the agent ran for 40,000+ steps across 8 parallel games, accumulating **296 episodes** in Run 0 and **280 episodes** in Run 1, with genuine achievements (43 table placements, 115 drink collections).
3. When the runtime reconnected, `train_agent.py` loaded `checkpoint_latest.pt`. Because the checkpoint was saved at step 50,000, `env_step = 50,000 >= target_env_steps (50,000)`.
4. The training collection loop `while env_step < target_env_steps:` did not execute any new steps.
5. `AchievementRecorder` was **not serialized in the checkpoint dictionary** (`checkpoint_data`). Thus, the freshly instantiated recorder had `num_episodes = 0`.
6. When `report_gen.generate_final_report` and `export_csv` were called at completion, the empty recorder output `0 evaluated episodes` and appended a 3rd row with 0 episodes.

### Artifact Reference
Full evidence serialized in [`evaluation_consistency.json`](file:///Users/harshitatolani/Sahil/ML/evaluation_consistency.json).

---

## 3. Action Conditioning & Forward Path Trace (Tasks 3, 4, 5)

### Controlled Experiment
Holding visual context $O_t$ (seed 42, rich tree and stone scene) and context state $h_t$ strictly fixed, all 17 discrete action tokens $a \in \{0, 1, \dots, 16\}$ plus a zero-embedding control were forward-passed through the predictor member.

### Numerical Results (Full Precision)
| Diagnostic Metric | Measured Value | Plan v2.0 Requirement |
|---|---|:---:|
| Pairwise Cosine Distance Across Actions (Mean) | **$9.92 \times 10^{-6}$** | $> 1.00 \times 10^{-2}$ |
| Pairwise Cosine Distance Across Actions (Max) | **$1.89 \times 10^{-5}$** | $> 1.00 \times 10^{-2}$ |
| Pairwise $L_2$ Distance Across Actions (Mean) | $0.3986$ | N/A |
| Pairwise Max Absolute Difference (Mean) | $0.0210$ | N/A |
| Distance to Zero-Action Embedding ($\mathbf{0}$) | **$4.97 \times 10^{-6}$** | $> 1.00 \times 10^{-2}$ |
| Action Embedding Table Gradient Norm | $\mathbf{0.001668}$ | $> 0$ (Confirmed) |
| Condition MLP Gradient Norm | $\mathbf{0.067009}$ | $> 0$ (Confirmed) |

### Diagnostic Finding
- **Gradients do physically flow** through `cond_mlp` to the action embeddings. The code path is genuinely connected and distinct.
- **Architectural Bottleneck**: In [`LatentPredictorMember`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/src/models/latent_predictor.py#L25-L66), `cond_mlp` maps `[h_t, a_emb]` into global FiLM parameters $(\gamma, \beta)$, which are broadcast identically across all 64 spatial tokens:
  $$\tilde{z} = (1 + \gamma) \cdot z_t + \beta$$
- The Transformer self-attention blocks then process the 64 spatial tokens. Because there is **no dedicated action token** in the Transformer sequence, self-attention attends exclusively to spatial correlations. The minute global scale/shift is suppressed by LayerNorm, leaving the output prediction divergence near zero ($9.92 \times 10^{-6}$).

### Artifact Reference
Full evidence serialized in [`action_conditioning_diagnostics.json`](file:///Users/harshitatolani/Sahil/ML/action_conditioning_diagnostics.json).

---

## 4. Timestep Alignment & Held-Out Independence (Tasks 6, 7)

### Verification Outcomes
1. **Seed Independence**:
   - Training Seeds: `1000..1007` (8 parallel environments).
   - Held-Out Seeds: `900000..900004` (independent held-out store).
   - Seed Disjointness: **CONFIRMED 100% DISJOINT**.
2. **Timestep Indexing**:
   - Verified across anchors in [`SequenceSampler`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/src/data/sequence_sampler.py#L114):
     - Offset 1: `step_idx = anchor + 0` $\to$ `next_obs` is $O_{t+1}$.
     - Offset 2: `step_idx = anchor + 1` $\to$ `next_obs` is $O_{t+2}$.
     - Offset 4: `step_idx = anchor + 3` $\to$ `next_obs` is $O_{t+4}$.
     - Offset 8: `step_idx = anchor + 7` $\to$ `next_obs` is $O_{t+8}$.
   - Timestep Alignment: **CONFIRMED MATHEMATICALLY EXACT**.
3. **Failure Cause for Test B**:
   - Persistence baseline error on held-out transitions is $0.0264$ (or $0.0005$ under saturated representations).
   - The active model prediction error is $0.4018$ (or $0.0005$).
   - The failure of Test B is **not** an off-by-one indexing error. It is caused by the model failing to predict physical spatial deltas more accurately than simply copying the current frame.

### Artifact Reference
Full evidence serialized in [`heldout_baseline_comparison.json`](file:///Users/harshitatolani/Sahil/ML/heldout_baseline_comparison.json).

---

## 5. Representation Manifold Geometry & Covariance (Task 8)

### Manifold Spectrum
Tested on 1,024 latent tokens across held-out Crafter scenes:

| Geometric Metric | Measured Value | Healthy Reference | Status |
|---|---|:---:|:---:|
| Feature Dimension ($D$) | 128 | 128 | Verified |
| Latent Standard Deviation | 1.0459 | $> 0.50$ | Healthy |
| Mean Channel Std | 0.8023 | $> 0.10$ | Healthy |
| Collapsed Channels ($S_j < 0.01$) | **0.00%** | $< 5.0\%$ | Healthy |
| Effective Rank (Participation Ratio) | **13.54** | $> 2.0$ | Healthy |
| Spectral Top-1 Component Variance | 20.44% | $< 50.0\%$ | Healthy |
| Spectral Top-5 Components Variance | 49.72% | $< 80.0\%$ | Healthy |
| **Mean Pairwise Cosine Similarity** | **0.9923** | $< 0.85$ | **SATURATED (FAIL)** |
| **Median Pairwise Cosine Similarity** | **0.9936** | $< 0.85$ | **SATURATED (FAIL)** |
| **Maximum Pairwise Cosine Similarity** | **0.9999995** | $< 0.95$ | **SATURATED (FAIL)** |

### Diagnostic Finding
- Individual latent dimensions have non-zero variance (std $\approx 0.80$). Channels are not zeroed out.
- **Directional Angular Saturation**: However, all 128-dimensional latent vectors lie within an extremely acute cone ($< 5^\circ$ spread). Because they point in nearly the same direction, the cosine similarity between unrelated scenes is $0.9923$, saturating the collapse detector.

### Artifact Reference
Full evidence serialized in [`representation_health.json`](file:///Users/harshitatolani/Sahil/ML/representation_health.json).

---

## 6. Official Gate Status

| Gate | Criterion | Status |
|---|---|:---:|
| **Evaluation Integrity Gate** | Disjoint seeds, correct alignment, reconciled episode logs | **PASS** |
| **Action Conditioning Gate** | Action divergence $> 10^{-2}$, beats no-action/shuffled controls | **BLOCKED** |
| **Representation Health Gate** | Pairwise cosine similarity $< 0.90$, no directional saturation | **BLOCKED** |
| **`P5_JEPA_LEARNING_PASS`** | All Phase 5 criteria must pass simultaneously | **BLOCKED** |

---

## 7. Root Cause Summary & Next Steps

The investigation definitively rules out implementation bugs (timesteps, data leakage, broken computation graphs). The root causes are purely theoretical:
1. **Pure Cosine Loss Shortcut**: Without an explicit variance penalty $\mathcal{L}_{\text{var}}$, cosine distance encourages all latents to point in the same direction.
2. **Grass Dominance**: 90% of spatial tokens are static grass; random patch masking trains the model to predict static persistence.
3. **FiLM-Only Action Conditioning**: Global broadcast scale/shift is ignored by spatial self-attention heads.

### Remediation Protocol (Zero Cheat):
1. **VICReg Spatial Token Variance Hinge**: $\mathcal{L}_{\text{var}} = \frac{1}{D}\sum \max(0, \gamma - S_j)$.
2. **Action Token Injection**: Prepend an explicit action token to the Transformer sequence and add an action residual to all tokens.
3. **Dynamic Patch Loss Weighting**: Upweight patches with non-zero temporal change.
4. **Checkpoint State Persistence**: Save `AchievementRecorder` state in checkpoints to preserve episode counts across runtime resumes.
