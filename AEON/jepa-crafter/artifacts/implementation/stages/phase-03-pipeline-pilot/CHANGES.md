# Phase 3 — Code Changes & Additions

## Modified Modules
1. `src/data/replay_buffer.py`:
   - Added `self.transition_ids = np.zeros(capacity, dtype=np.int64)` tracking absolute monotonic write IDs.
   - Updated `add()` to write `self.transition_ids[t_idx] = self.total_added`.
   - Updated `get_state()` and `set_state()` to serialize and restore `transition_ids`.
2. `src/data/sequence_sampler.py`:
   - Enhanced `is_valid_anchor()` with generation verification:
     `self.replay.transition_ids[idx] == start_tid + k`
   - Guarantees zero sequence extraction across circular buffer wraparound boundaries or stale overwritten generations.

## New Tests
1. `tests/test_pipeline_pilot.py`:
   - `test_ring_wraparound_generation`: Multi-overwrite wraparound tracking.
   - `test_10k_sample_stress_test`: 10,000 sequence samples over 3.5x buffer capacity with 0 boundary crossings.
   - `test_replay_sample_provenance`: Full transition ID and timestep traceability.
   - `test_corrupt_checkpoint_preserves_previous_latest`: Atomic file saving resilience.
   - `test_uninterrupted_vs_resumed_equivalence`: Uninterrupted vs save/resume/continue collection equivalence.
   - `test_pilot_500_steps_single_worker`: 500-step smoke run with 100% decision-outcome linkage and SHA-256 trace verification.
