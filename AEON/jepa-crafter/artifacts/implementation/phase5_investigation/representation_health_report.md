# Investigation D: Representation Health Report

**Gate Status:** `REPRESENTATION_HEALTH_PASS` (BLOCKED)

---

## 1. Executive Summary

This investigation analyzed the representation manifold geometry, covariance spectrum, channel variance, and pairwise angular distributions across 1,024 latent tokens from held-out Crafter scenes.

### Core Finding
The representation does **not** suffer from channel collapse (collapsed channel fraction is 0.00%, mean channel std is 0.8023). However, it suffers from severe **Directional Angular Saturation**: all 128-dimensional latent vectors are compressed into a narrow directional cone ($< 5^\circ$ spread), causing mean pairwise cosine similarity between unrelated scenes to saturate at **0.9923 – 0.9991**.

---

## 2. Geometric & Spectral Diagnostics

| Metric | Measured Value | Pre-Declared Health Standard | Diagnostic Evaluation |
|---|---|:---:|:---:|
| **Latent Standard Deviation** | **1.0459** | $> 0.50$ | Healthy magnitude |
| **Mean Channel Std** | **0.8023** | $> 0.10$ | No dead channels |
| **Min Channel Std** | **0.1198** | $> 0.01$ | Healthy channel floors |
| **Collapsed Channel Fraction ($< 0.01$)** | **0.00%** | $< 5.0\%$ | 0 of 128 channels collapsed |
| **Effective Rank (Participation Ratio)** | **13.54** | $> 2.0$ | Distributed covariance |
| **Top-1 Spectral Component Variance** | **20.44%** | $< 50.0\%$ | No single-axis dominance |
| **Top-5 Spectral Components Variance** | **49.72%** | $< 80.0\%$ | Multi-dimensional variance |
| **Mean Pairwise Cosine Similarity** | **0.9923** | $< 0.85$ | **SATURATED (COLLAPSED CONE)** |
| **Median Pairwise Cosine Similarity** | **0.9936** | $< 0.85$ | **SATURATED (COLLAPSED CONE)** |
| **Maximum Pairwise Cosine Similarity** | **0.9999995** | $< 0.95$ | **SATURATED (COLLAPSED CONE)** |
| **Collapse Warning** | **ACTIVE** | `NONE` | Active due to Cosine Sim $> 0.98$ |

---

## 3. Mathematical Mechanism: Why Pure Cosine Distance Induces Cone Collapse

In self-supervised JEPA with stop-gradient target encoder:
$$\mathcal{L}_{\text{cos}}(\hat{z}, \bar{z}) = 1 - \frac{\hat{z} \cdot \bar{z}}{\|\hat{z}\|_2 \|\bar{z}\|_2}$$

1. **Cosine Normalization Invariance**: Cosine distance only measures angles on the unit sphere $\mathbb{S}^{127}$. It is completely invariant to feature magnitude $\|\hat{z}\|_2$.
2. **Degenerate Minimum**: If the online encoder maps all visual inputs to a constant ray $v \in \mathbb{R}^{128}$ ($z = c \cdot v$ for $c > 0$), then the normalized vector is always $\frac{v}{\|v\|}$. The cosine similarity is identically $1.0$, and the loss is identically $0.0$.
3. **Absence of Variance Regularization**: In contrast to VICReg (Bardes et al., 2022) or contrastive frameworks (SimCLR), the baseline loss contained **no variance penalty** across the batch. The network minimized cosine loss by collapsing the angular spread across spatial patches.
4. **Empirical Consequence**: Because every patch points in direction $v$, any two frames have cosine similarity $> 0.99$, and predicting $v$ trivially achieves near-zero cosine distance without learning any environment mechanics.

---

## 4. Remediation Protocol

To break the cone and establish healthy representation geometry:
1. **VICReg Variance Hinge**:
   $$\mathcal{L}_{\text{var}} = \frac{1}{D} \sum_{j=1}^D \max\left(0, \gamma - \sqrt{\text{Var}(Z_{:, j}) + \epsilon}\right)$$
   where $\gamma = 0.50$ enforces a hard standard deviation floor per channel across spatial tokens.
2. **Dynamic Patch Masking**: Sample tokens with probability proportional to their temporal delta $\| \bar{z}_{t+1} - z_t \|_2$, forcing capacity onto physical world changes.

---

## 5. Gate Verdict
**`REPRESENTATION_HEALTH_PASS` is BLOCKED.**  
Pairwise cosine similarity ($0.9923$) exceeds the pre-declared threshold ($0.90$), and the collapse warning remains active.
