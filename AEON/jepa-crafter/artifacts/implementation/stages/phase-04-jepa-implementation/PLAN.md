# Phase 4 — Verify/Build the JEPA Baseline

## Purpose
Get one correct, reproducible from-scratch visual world model before downstream exploration or relationship learning.

## Model Contract Verification
- Small randomly initialized visual encoder; spatial tokens `(B, 64, 128)`.
- Online encoder and frozen-gradient EMA target encoder initialized identically.
- Anonymous action embedding for 17 tokens (`0..16`).
- Temporal context reset at episode boundaries.
- Action-conditioned latent predictor ensemble (3 members).
- JEPA latent prediction primary objective; stop-gradient on target features.
- Multi-horizon prediction `(1, 2, 4, 8)`.
- Scalar reward and termination heads remain auxiliary only.
- Zero pixel-reconstruction heads, zero recipe tables, zero cheating shortcuts.

## Acceptance Criteria
- [x] Test 1: Input/latent shapes and dtype.
- [x] Test 2: Online encoder receives gradients; target encoder receives none.
- [x] Test 3: Target encoder initializes as an exact online copy.
- [x] Test 4: EMA changes after successful optimizer steps and does not change after skipped updates.
- [x] Test 5: Loss is finite and gradients are finite on synthetic and real batches.
- [x] Test 6: Training sequence alignment ($A_t \to O_{t+1}$, Horizon 4 $\to O_{t+4}$).
- [x] Test 7: Sampler never crosses episode end or reset.
- [x] Test 8: Predictor does not receive future observations/targets.
- [x] Test 9: Optimizer and EMA counters agree.
- [x] Test 10: Checkpoint restores model, target, optimizer, scaler, and scheduler.
