# Phase 0 — Repository and Runtime Audit Plan

## Objective
Establish the definitive baseline of the existing codebase under `AEON/jepa-crafter/`, verify runtime dependencies and Crafter environment contracts, confirm all existing unit tests pass, and generate preflight manifests prior to any Phase 1 observability instrumentation.

## Scope
1. **Repository State**: Commit `8829fe08285513d803621dfd813739bf872b842d` on branch `main`.
2. **Runtime Verification**:
   - Python: 3.9.6
   - PyTorch: 2.8.0
   - Crafter: 1.8.3
3. **Module Auditing**: Clean import test for all 46 modules under `src/`.
4. **Configuration Auditing**: Validation of `configs/crafter_t4.yaml`, `configs/crafter_t4_reward_free.yaml`, and `configs/evaluation.yaml`.
5. **Contract Checks**: Verification of Crafter observation shape `(64, 64, 3)`, dtype `uint8`, 17 discrete actions, and evaluator isolation.
6. **Existing Test Suite**: Full run of all 19 pytest test cases.

## Acceptance Criteria
- [x] Exact commit SHA recorded.
- [x] All 46 modules import without errors.
- [x] All 3 config files load and parse successfully.
- [x] Crafter step/reset API satisfies strict learner contract.
- [x] 19/19 tests pass cleanly.
- [x] `P0_AUDIT_PASS` gate achieved.
