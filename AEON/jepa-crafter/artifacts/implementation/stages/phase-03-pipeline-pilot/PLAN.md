# Phase 3 — Environment, Replay, Checkpoint Smoke and Pipeline Pilot Plan

## Objective
Validate the end-to-end data pipeline, ring-buffer wraparound, and multi-horizon sequence extraction prior to large-scale JEPA model training:
1. Normalize Crafter environment contract (`uint8[64, 64, 3]`, scalar float reward, 17 discrete actions).
2. Guarantee circular ring-buffer overwrite tracking with absolute generation tags to prevent stale cross-generation sequence sampling.
3. Perform a 10,000-sample stress test over multiple wraparounds to guarantee zero invalid boundary crossings.
4. Execute a 500-step smoke run with full trace capture and 100% decision-outcome linkage.
5. Execute a 5,000-step collection pilot collecting raw transitions and a disjoint held-out validation set.
6. Verify atomic checkpoint saving and corruption resilience (preserves previous valid latest).
7. Verify uninterrupted vs save/resume/continue collection equivalence.

## Acceptance Criteria
- [x] Zero contract violations and zero data leakage.
- [x] Zero invalid sequences in 10,000-sample stress test.
- [x] 100% decision-to-outcome linkage in 500-step smoke run.
- [x] Complete 5,000-step pilot data collection and trace verification.
- [x] Checkpoint resume roundtrip restores exact buffer, mapping, and counters.
- [x] `P3_PIPELINE_PILOT_PASS` gate achieved.
