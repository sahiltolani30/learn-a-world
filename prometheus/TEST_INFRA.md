# E2E Test Infra: Crafter Tabula Rasa JEPA

## Test Philosophy
- Opaque-box, requirement-driven verification derived strictly from `ORIGINAL_REQUEST.md`.
- No internal mock reliance; validates actual tensor dimensions, gradients, execution latencies, and representation properties.
- Dual testing requirements:
  1. Ultra-fast CPU smoke test (`python -m src.test_smoke`) passing in < 60 seconds.
  2. Multi-tier comprehensive test suite (`python -m tests.run_all_tests`) verifying feature correctness, boundaries, non-collapse, zero-cheat sanitization, and end-to-end integration.

## Feature Inventory & Test Mapping
| # | Feature | Milestone | Tier 1 (Features) | Tier 2 (Boundaries) | Tier 3 (Cross-Feature) | Tier 4 (E2E Scenarios) |
|---|---------|-----------|:-----------------:|:-------------------:|:----------------------:|:----------------------:|
| 1 | ResNetTiny Visual Encoder | M1 | 5 tests | 5 tests | ✓ | ✓ |
| 2 | Action Embedding Module | M1 | 5 tests | 5 tests | ✓ | ✓ |
| 3 | 4-Head Latent Predictor | M1 | 5 tests | 5 tests | ✓ | ✓ |
| 4 | EMA Target Encoder | M1 | 5 tests | 5 tests | ✓ | ✓ |
| 5 | Anti-Collapse VICReg Loss | M1 | 5 tests | 5 tests | ✓ | ✓ |
| 6 | Zero-Cheat Crafter Wrapper | M2 | 5 tests | 5 tests | ✓ | ✓ |
| 7 | Generalizable Env Interface | M2 | 5 tests | 5 tests | ✓ | ✓ |
| 8 | Epistemic Curiosity Engine | M3 | 5 tests | 5 tests | ✓ | ✓ |
| 9 | Episodic k-NN Novelty | M3 | 5 tests | 5 tests | ✓ | ✓ |
| 10 | Two-Tier Replay Buffer | M3 | 5 tests | 5 tests | ✓ | ✓ |
| 11 | Actor-Critic Policy Prior | M4 | 5 tests | 5 tests | ✓ | ✓ |
| 12 | Categorical Latent CEM Planner | M4 | 5 tests | 5 tests | ✓ | ✓ |
| 13 | Dual-Mode Agent | M4 | 5 tests | 5 tests | ✓ | ✓ |
| 14 | Ultra-Fast Smoke Test | M5 | 5 tests | 5 tests | ✓ | ✓ |
| 15 | Colab & Drive Notebook | M5 | 5 tests | 5 tests | ✓ | ✓ |
| 16 | Integrated Training Loop | M5 | 5 tests | 5 tests | ✓ | ✓ |

## Test Architecture
- Test Runner: `python3 -m unittest discover -s tests -p 'test_*.py'` or `python3 tests/run_all_tests.py`
- Location: `tests/`
- Pass/Fail Semantics: Exit code 0 on all tests passed; non-zero on assertion failure.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Expected Outcome |
|---|----------|--------------------|------------------|
| 1 | Tabula Rasa Cold Start | Env wrapper + Encoder + Random Action + Replay push | Smooth step execution, no privileged leak, variance > 0.1 |
| 2 | Pure Latent Dynamics Update | World Model batch update (Encoder + Predictor + Target + VICReg) | Loss decreases, target EMA updates, non-collapse variance maintained |
| 3 | Epistemic Exploration Discovery | Agent stepping under curiosity guidance vs random | Agent visits novel states, heads show high variance on new tiles |
| 4 | Hybrid CEM Planning in Hazard | Agent facing low health/vital changes | CEM evaluates candidates with Actor prior, selects valid action in <1ms |
| 5 | Checkpoint Persistence & Resume | Two-Tier buffer + Weights save/load cycle | Weights match exactly, buffer resumes without data corruption |
| 6 | CPU Smoke Test Execution | Full 10-step sequence under CPU mode | Completes in < 60 seconds (target < 5 seconds) |
