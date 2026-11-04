# Phase 2 — Code Changes & Additions

## Modified Modules
1. `src/evaluation/action_conditioning_tests.py`:
   - Updated `ActionConditioningTester.__init__` to accept `opaque_mapping: Optional[OpaqueActionMapping] = None`.
   - Passed `opaque_mapping` into all paired root and counterfactual branch adapters (`CrafterAdapter(seed=seed, opaque_mapping=self.opaque_mapping)`).
2. `src/exploration/novelty.py`:
   - Added `update: bool = False` flag to `RunningNormalizer.normalize()` to support frozen evaluation.
   - Added `get_state()` and `set_state()` to `RunningNormalizer` and `NoveltyEstimator`.
   - Added `insert_observation()` for explicit real observation ingestion and updated `evaluate()` to default to `update_memory=False, update_normalizer=False`.
3. `src/exploration/latent_memory.py`:
   - Added `get_state()` and `set_state()` methods for full snapshot restoration.
4. `src/exploration/disagreement.py`:
   - Added `update_from_observation()`, `update_from_residual()`, `get_state()`, and `set_state()`.
   - Made `score()` strictly read-only without updating normalizer or moving averages.
5. `src/planning/candidate_scorer.py`:
   - Enforced frozen normalizer evaluation in candidate rollout scoring (`update_normalizer=False`), guaranteeing order invariance.
6. `src/utils/config.py`:
   - Added top-level section validation to `Config.from_dict()`, raising `ValueError` on unknown sections.
7. `src/training/resume.py`:
   - Extended `restore_checkpoint()` to restore `ema_state_dict`, `novelty_estimator`, `disagreement_estimator`, `progress_estimator`, and `update_scheduler`.
8. `src/training/train_agent.py`:
   - Integrated `HeldoutEpisodeStore` with disjoint seed assertion (`verify_disjoint_from`).
   - Wired heldout store batches into validation evaluation and final report generation.
   - Serialized full novelty, disagreement, progress, and scheduler states into checkpoints.

## New Modules
1. `src/data/heldout_store.py`:
   - Implemented `HeldoutEpisodeStore` managing validation episodes from disjoint seeds (`heldout_seed_start: 900000`).

## New Tests
1. `tests/test_baseline_integrity.py`:
   - 9 comprehensive tests covering all 8 defect areas.
