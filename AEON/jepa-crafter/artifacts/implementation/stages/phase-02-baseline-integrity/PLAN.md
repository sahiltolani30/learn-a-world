# Phase 2 — Repair and Verify Baseline Integration Plan

## Objective
Correct and verify the 8 critical integration risks identified in Plan v2.0 §7:
1. Ensure the action-conditioning evaluator (`ActionConditioningTester`) strictly uses the run's exact `OpaqueActionMapping`.
2. Decouple validation evaluation from the training replay buffer using a completely disjoint `HeldoutEpisodeStore` (seeds $\ge 900,000$).
3. Ensure real observations are inserted into `NoveltyEstimator` / `LatentMemory` exactly once.
4. Prevent imagined rollout candidates from mutating novelty memory or running normalizer statistics.
5. Update `LearningProgressEstimator` only from real measured transition residuals, keeping `score()` strictly read-only.
6. Enforce candidate sequence score order-invariance within numerical tolerance $< 10^{-6}$.
7. Verify candidate normalizers update only after candidate ranking from real observed transitions.
8. Enforce strict configuration validation, rejecting unknown top-level sections or unrecognized arguments.
9. Verify that checkpoint resume restores the full mutable state (weights, EMA target, optimizer, scaler, replay, action mapping, novelty memory, running normalizers, and scheduler).

## Acceptance Criteria
- [x] All 9 Phase 2 unit tests in `tests/test_baseline_integrity.py` pass.
- [x] Automated assertion confirms training and held-out seeds share zero overlap.
- [x] Candidate ranking proven order-invariant within $< 10^{-6}$.
- [x] Checkpoint save/restore proven byte-and-tensor equivalent.
- [x] `P2_BASELINE_INTEGRITY_PASS` gate achieved.
