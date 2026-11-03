# Phase 1 — Minimal Observability Foundation Plan

## Objective
Implement a lightweight, append-only event tracing system to record the full lifecycle, action decisions, environment outcomes, training updates, and failure modes with cryptographic checksum verification and zero information leakage.

## Architecture
- `src/observability/events.py`: Event types, schema, bounded payloads, and leak prevention.
- `src/observability/trace_writer.py`: Thread-safe append-only JSONL writer with SHA-256 companion checksums.
- `src/observability/tracer.py`: `TraceManager` providing clean API with 1-to-1 decision-outcome linkage.

## Acceptance Criteria
- [x] Synthetic episode with 10+ transitions can be reconstructed from trace.
- [x] 100% decision-to-outcome linkage (`action_decision` $\to$ `action_executed` $\to$ `action_outcome`).
- [x] Crash/failure handler records stack trace and leaves trace readable.
- [x] Reopen/append checksum verification roundtrip passes.
- [x] Debug-on vs debug-off policy determinism matches bit-for-bit.
- [x] Strict filter raises ValueError if any privileged game state leaks into logs.
- [x] `P1_TRACE_PASS` gate achieved.
