# Project AEON: Risks, Blockers & Mitigations

## Risk 1: Free Colab T4 Session Preemption
- **Severity:** High
- **Description:** Google Colab free-tier GPU instances can disconnect or terminate after 12 hours or during idle periods.
- **Mitigation:**
  - Automated Google Drive mounting at `/content/drive/MyDrive/learn-a-world/aeon/`.
  - Atomic checkpoint persistence every 10,000 steps.
  - Resume loader restores replay pointer, model weights, optimizer/scaler state, and RNG seed.

## Risk 2: Latent Representation Collapse in JEPA
- **Severity:** High
- **Description:** Without negative pairs (like SimCLR) or pixel decoders (like MAE/Dreamer), self-supervised latent prediction can collapse to constant vectors.
- **Mitigation:**
  - Asymmetric EMA target network with slow momentum update ($0.996 \to 0.999$).
  - Target branch is strictly stop-gradient (`detach()`).
  - Spatial token masking (50% random patch dropout on target).
  - Continuous feature diagnostics measuring channel variance, participation ratio (effective rank), and pairwise cosine distance across frames.

## Risk 3: Premature Latent MPC Rollout Exploitation
- **Severity:** Medium
- **Description:** If CEM optimizes against an inaccurate dynamics model, it can select catastrophic actions that cause early deaths.
- **Mitigation:**
  - Plan v2.0 mandates `P5_JEPA_LEARNING_PASS` before enabling MPC.
  - If Phase 7 benchmark shows no statistically significant improvement over single-role Explorer, MPC remains explicitly `DISABLED` without blocking subsequent relation-learning phases.

## Risk 4: False Causal Correlation in Relational Discovery
- **Severity:** Medium
- **Description:** Actions taken prior to background events (e.g. nightfall or animal movement) could be misclassified as causal mechanics.
- **Mitigation:**
  - Append-only evidence ledger in Phase 9 separates `CANDIDATE` from `CONFIRMED`.
  - Phase 10 controlled tests require $\ge 20$ independent contexts and Benjamini-Hochberg False Discovery Rate control ($q < 0.05$).
