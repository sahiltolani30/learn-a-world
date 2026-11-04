# Phase 5 Recovery: Next Action Roadmap

## 1. Executive Decision on Retraining
**Decision: Retraining is fully justified.**

### Why Retraining is Necessary:
The investigation definitively showed that:
1. The evaluation pipeline itself was successfully audited and repaired (`EVALUATION_INTEGRITY_PASS` is PASSED).
2. The failure to pass `P5_JEPA_LEARNING_PASS` is not an evaluation artifact, off-by-one error, or data leakage bug.
3. The root cause is a fundamental property of pure cosine loss on static-heavy environments: gradient descent converged into a zero-loss directional cone ($0.9923 - 0.9991$ cosine similarity) where action divergence vanished ($9.92 \times 10^{-6}$).
4. Simply running more steps on the current collapsed checkpoint would only entrench the degenerate cone. The optimization objective itself must include anti-collapse variance regularization and direct action token injection.

---

## 2. Targeted Remediation Protocol (Zero Cheat, Zero Handcrafted Rules)

In the upcoming recovery turn, implement the 3 surgical enhancements:

### Enhancement 1: Spatial Token Variance Hinge (VICReg Regularization)
- **File:** `src/learning/jepa_losses.py`
- **Mechanism:** Enforce channel-wise standard deviation floor ($\gamma = 0.50$) across spatial tokens in each batch:
  $$\mathcal{L}_{\text{var}} = \frac{1}{D} \sum_{j=1}^D \max\left(0, \gamma - \sqrt{\text{Var}(Z_{:, j}) + \epsilon}\right)$$
- **Effect:** Forces visual features to spread out across the hypersphere, breaking the $0.9923$ saturation cone and enabling distinct representations for trees, stone, and terrain.

### Enhancement 2: Action Token Injection & Residual Projection
- **File:** `src/models/latent_predictor.py`
- **Mechanism:**
  1. Add a direct linear projection of the action embedding as a residual to every spatial token: $\tilde{z}_i = z_i + W_a(a_t)$.
  2. Prepend an explicit action token as the 65th input token to the Transformer blocks: $[a_{\text{tok}}, \tilde{z}_1, \dots, \tilde{z}_{64}]$.
- **Effect:** Self-attention heads are mathematically forced to attend to the action token, raising action divergence from $10^{-6}$ to $> 10^{-2}$.

### Enhancement 3: Dynamic-Tile Loss Weighting
- **File:** `src/learning/jepa_losses.py`
- **Mechanism:** Compute per-patch temporal deltas $\Delta_i = \| \bar{z}_{t+1, i} - z_{t, i} \|_2$ and upweight the top-$K$ changing patches ($3.0\times$ weight).
- **Effect:** Prevents the loss from being dominated by static grass tiles, forcing capacity onto the player, tools, and world mechanics.

---

## 3. Immediate Action Plan

1. **Commit and Push Audit Deliverables**:
   Commit all 12 investigation artifacts and tests to `main`.
2. **Implement Phase 5 Recovery Enhancements**:
   Apply Enhancements 1, 2, and 3 in `AEON/jepa-crafter/`.
3. **Verify Regression Suite**:
   Run full test suite locally (ensuring 100% green tests).
4. **Execute Clean Retraining on Colab**:
   Launch a fresh training run from step 0 with the regularized objective.
5. **Monitor Phase 5 Gate Progression**:
   - Verify `Test B (Held-out beat persistence)` turns and stays `True`.
   - Verify `Test C (Action-conditioned)` turns `True` (divergence $> 0.01$).
   - Verify `Test D (No collapse)` turns `True` (pairwise cosine $< 0.90$).
   - Unlock `P5_JEPA_LEARNING_PASS` and proceed to Phase 6.
