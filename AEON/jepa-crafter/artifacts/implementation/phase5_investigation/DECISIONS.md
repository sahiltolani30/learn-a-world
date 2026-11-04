# Project AEON: Phase 5 Investigation Decisions

## Decision 001: Evidence-Based Audit Without Architecture Drift
- **Context:** The 50,000-step checkpoint failed Phase 5 gates (action divergence ~0, pairwise cosine ~0.999, negative improvement over persistence).
- **Decision:** Do not increase training budget or rewrite the architecture as a first response. Dissect the failure into 6 independent investigations (A: Evaluation Integrity, B: Action Conditioning, C: Held-out Prediction, D: Representation Health, E: Training Dynamics, F: Planner).
- **Date:** 2026-11-04
- **Status:** ACTIVE

## Decision 002: Isolation of Vectorized Evaluator Taps
- **Context:** `VectorCrafterAdapter` shared a single mutable `EvaluatorTap` across all parallel games, cross-contaminating reward, length, and achievement tracking upon individual episode terminations.
- **Decision:** Isolate evaluator accumulation per environment worker. Each parallel worker accumulates statistics in its own private state, committed atomically on termination.
- **Date:** 2026-11-04
- **Status:** ACTIVE

## Decision 003: Checkpoint Persistence of Evaluation History
- **Context:** Resuming from a checkpoint where `env_step >= target_env_steps` created a fresh `AchievementRecorder` reporting 0 episodes in the summary, creating an apparent contradiction with CSV rows.
- **Decision:** Serialize `achievement_recorder` state in the checkpoint dictionary. If evaluation is run with 0 episodes, explicitly report `NOT_RUN` / `INVALID` rather than a valid measured zero.
- **Date:** 2026-11-04
- **Status:** ACTIVE

## Decision 004: Preservation of Historical Experiment Artifacts
- **Context:** Re-running evaluations must not overwrite historical artifacts or erase negative results.
- **Decision:** Store baseline investigation data in dedicated versioned directories with explicit timestamps and SHA-256 traces.
- **Date:** 2026-11-04
- **Status:** ACTIVE
