# Phase 1 — Code Changes & Additions

## New Modules
1. `src/observability/events.py`:
   - `EventType` enumeration with all 14 lifecycle and operational event families.
   - `Event` dataclass with automatic leak-prevention checking (`validate_no_leakage`) against `FORBIDDEN_LEAK_KEYS`.
   - Compact bounded JSON serialization.
2. `src/observability/trace_writer.py`:
   - `TraceWriter` with streaming append-only JSONL writes, `fsync` flushing, and companion `.sha256` checksum verification.
3. `src/observability/tracer.py`:
   - `TraceManager` providing clean API for event emission, decision tracking, and 1-to-1 linkage across decisions, executions, and outcomes.
4. `src/observability/__init__.py`:
   - Clean exports for external consumption.

## Modified Modules
1. `src/training/train_agent.py`:
   - Added `TraceManager` initialization.
   - Wired tracing for run lifecycle, action decisions, outcomes, training updates, checkpoints, and validation gates.
   - Added automatic trace finalization and SHA-256 output.

## Test Suite
1. `tests/test_observability.py`:
   - 5 comprehensive tests verifying 100% linkage, crash safety, checksum roundtrips, debug determinism, and zero info leakage.
