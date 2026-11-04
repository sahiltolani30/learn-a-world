# Project AEON: Root-Cause Investigation Register

This register documents every anomaly, defect, and failure mechanism discovered during the Phase 4/Phase 5 failure investigation.

---

## Issue RC-001: EvaluatorTap State Collision in Vectorized Environments
- **Status:** `FIXED_AND_TESTED`
- **Affected File & Function:** `src/env/vector_adapter.py:VectorCrafterAdapter.__init__`, `src/env/evaluator_tap.py:EvaluatorTap.record_step`, `src/env/evaluator_tap.py:EvaluatorTap.reset_episode`
- **Observed Symptom:** Vectorized collection with $N=8$ environments produced corrupted reward, episode length, and achievement associations.
- **Reproduction Procedure:** Step two environments alternately with different rewards and terminate one while the other is in-flight.
- **Evidence:** Prior implementation shared a single `EvaluatorTap` instance across all 8 `CrafterAdapter` instances. Terminating any single environment wiped the shared accumulator (`self._current_reward = 0.0`), corrupting the other 7 running episodes.
- **Root Cause:** Lack of environment worker isolation (`env_id`) in `EvaluatorTap`.
- **Proposed Minimal Fix:** Add `worker_id` to `CrafterAdapter` and index all internal accumulators by `env_id` inside `EvaluatorTap`.
- **Regression Test:** [`tests/test_investigation_a_evaluation_integrity.py::test_interleaved_environments_remain_independent`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/tests/test_investigation_a_evaluation_integrity.py) (PASSED).

---

## Issue RC-002: AchievementRecorder Reset on Checkpoint Resume
- **Status:** `FIXED_AND_TESTED`
- **Affected File & Function:** `src/training/train_agent.py`, `src/evaluation/achievement_recorder.py:AchievementRecorder`
- **Observed Symptom:** Executive summary reported `0.00% across 0 evaluated episodes` despite CSV containing 296 and 280 episodes with ~2.1-2.2% score.
- **Reproduction Procedure:** Resume a run from a checkpoint at `env_step >= target_env_steps`.
- **Evidence:** `AchievementRecorder` was an in-memory accumulator not serialized in `checkpoint_data`. The resumed process started at step 50,000, skipped the collection loop, and reported `0 evaluated episodes` from the empty instance.
- **Root Cause:** Checkpoint dictionary omitted `achievement_recorder_state`.
- **Proposed Minimal Fix:** Add `get_state()` and `load_state()` to `AchievementRecorder` and serialize in `CheckpointStore`.
- **Regression Test:** [`tests/test_investigation_a_evaluation_integrity.py::test_checkpoint_state_persistence_restores_episode_history`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/tests/test_investigation_a_evaluation_integrity.py) (PASSED).

---

## Issue RC-003: Misleading Zero Score Presentation in Final Report
- **Status:** `FIXED_AND_TESTED`
- **Affected File & Function:** `src/evaluation/report.py:ReportGenerator.generate_final_report`
- **Observed Symptom:** A run with 0 evaluation episodes formatted the result as `Official Crafter Benchmark Score: 0.00% across 0 evaluated episodes`, masking that the evaluation was not run.
- **Reproduction Procedure:** Call `generate_final_report` with `num_episodes: 0`.
- **Evidence:** `report.py` formatted `None` or 0 as `0.00%` without checking validity.
- **Root Cause:** Absence of valid status branching in markdown template.
- **Proposed Minimal Fix:** Check `num_episodes == 0` and explicitly print `NOT_RUN / INVALID`.
- **Regression Test:** [`tests/test_investigation_a_evaluation_integrity.py::test_report_with_zero_evaluation_episodes_marked_invalid`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/tests/test_investigation_a_evaluation_integrity.py) (PASSED).

---

## Issue RC-004: Lack of Checkpoint ID in Benchmark CSV Rows
- **Status:** `FIXED_AND_TESTED`
- **Affected File & Function:** `src/evaluation/achievement_recorder.py:AchievementRecorder.export_csv`
- **Observed Symptom:** Multiple runs and intermediate checkpoints appended rows with identical `run_name` without distinguishing which checkpoint produced which row.
- **Reproduction Procedure:** Call `export_csv` twice from different checkpoints.
- **Evidence:** CSV header was `["run_name", "seed", ...]`, lacking `checkpoint_id`.
- **Root Cause:** Missing metadata column in CSV schema.
- **Proposed Minimal Fix:** Add `checkpoint_id` to header and row dictionary.
- **Regression Test:** [`tests/test_investigation_a_evaluation_integrity.py::test_benchmark_csv_distinguishes_runs`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/tests/test_investigation_a_evaluation_integrity.py) (PASSED).

---

## Issue RC-005: Pseudo No-Action Baseline Using Arbitrary Action Token 0
- **Status:** `FIXED_AND_TESTED`
- **Affected File & Function:** `src/evaluation/heldout_prediction.py:HeldoutPredictorEvaluator.evaluate_batch`
- **Observed Symptom:** The no-action baseline in heldout prediction was identical to active prediction or shuffled prediction.
- **Reproduction Procedure:** Inspect `HeldoutPredictorEvaluator` line 62.
- **Evidence:** `no_action_tokens = torch.zeros_like(actions_future)` fed token ID `0` into the embedding table. Token `0` is an active environment action (e.g. `move_left`), not a zero ablation.
- **Root Cause:** Failure to pass zero-embedding tensor directly to rollout function.
- **Proposed Minimal Fix:** Allow `autoregressive_rollout` to accept `future_action_embeddings` and pass true zero vectors `torch.zeros(B, H, action_dim)`.
- **Regression Test:** [`tests/test_phase4_jepa_contract.py`](file:///Users/harshitatolani/Sahil/ML/AEON/jepa-crafter/tests/test_phase4_jepa_contract.py) & `test_investigation_b_action_conditioning.py` (PASSED).

---

## Issue RC-006: FiLM Broadcast Modulation Suppressed by LayerNorm
- **Status:** `CONFIRMED`
- **Affected File & Function:** `src/models/latent_predictor.py:LatentPredictorMember.forward`
- **Observed Symptom:** Action prediction divergence across all 17 actions is vanishingly small ($9.92 \times 10^{-6}$), causing Test C to fail.
- **Reproduction Procedure:** Execute forward pass on fixed context $O_t$ across all 17 action tokens and measure pairwise cosine distance.
- **Evidence:** `cond_mlp` produces a single global scale $\gamma$ and shift $\beta$ applied uniformly across 64 spatial tokens. Subsequent `TransformerEncoder` self-attention and `LayerNorm` center and normalize features, canceling uniform channel shifts and damping action differentiation.
- **Root Cause:** Lack of direct action token injection in the Transformer self-attention sequence.
- **Proposed Minimal Fix (Phase 5 Recovery):** Add action residual projection directly to token inputs and prepend an explicit action token into the Transformer sequence.
- **Status:** Gate `ACTION_CONDITIONING_PASS` remains `BLOCKED` pending training recovery.

---

## Issue RC-007: Directional Representation Cone & Saturated Manifold
- **Status:** `CONFIRMED`
- **Affected File & Function:** `src/learning/jepa_losses.py:cosine_feature_loss`, `src/learning/feature_diagnostics.py`
- **Observed Symptom:** Pairwise cosine similarity across distinct frames is $0.9923 - 0.9991$. Test D collapse warning active. Model prediction distance ($0.0005$) ties persistence ($0.0005$).
- **Reproduction Procedure:** Measure pairwise cosine similarity matrix on held-out visual tokens.
- **Evidence:** Pure cosine distance loss possesses a degenerate trivial minimum: mapping all visual tokens to a single directional cone. Individual channels have non-zero variance ($S_j \approx 0.80$), but the angular spread between tokens is $< 5^\circ$.
- **Root Cause:** Absence of spatial token variance regularization (VICReg hinge) and 90% static terrain dominance.
- **Proposed Minimal Fix (Phase 5 Recovery):** Introduce spatial token variance hinge $\mathcal{L}_{\text{var}} = \frac{1}{D}\sum \max(0, \gamma - S_j)$ and dynamic-tile loss weighting.
- **Status:** Gate `REPRESENTATION_HEALTH_PASS` remains `BLOCKED` pending training recovery.
