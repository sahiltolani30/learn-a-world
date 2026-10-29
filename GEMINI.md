# Project ATLAS: Absolute Research Integrity & Anti-Cheating Protocol

This document defines non-negotiable operational laws for all AI agents working on this codebase. These rules are active at all times.

---

## 1. ABSOLUTE BAN ON HARDCODING & RECIPE TABLES
- **NO Hand-Crafted Game Rules**: Never write dictionaries, mappings, or lookup tables that encode human knowledge of the environment (e.g., crafting recipes, action-to-ingredient mappings, terrain rules, tool requirements).
- **All Relations Must Be Learned**:
  - Costs, prerequisites, and affordances must derive purely from learned model heads (e.g., `StateDeltaHead`, `PredecessorLift`, `VitalsReadout`) or empirical experience.
  - If a learned extraction pipeline is not yet implemented, **material gating must remain disabled**. Probes must run honestly and fail naturally. NEVER substitute a human-typed dictionary to simulate a model learning something.
- **Structural Integrity Audits**: The codebase must be maintained so that structural AST greps for hardcoded recipe tables (like `CRAFT_REQUIRED_MATERIALS`) always return zero matches.

---

## 2. ZERO SHORTCUTS & ZERO GUESSWORK
- **Follow Plans Literally**: When executing a plan or specification, do not guess, assume, or improvise "shortcuts" to make things easier or faster.
- **If Unsure, Ask or Stop**: If a requirement seems abstract or missing an explicit learned mechanism, do not bridge the gap with a hardcoded cheat. Stop and report the gap, or implement the fallback path explicitly.
- **No Token-Burning Procrastination**: Deliver direct, robust, and mathematically sound implementations without unnecessary refactor loops or excuses.

---

## 3. GREEN TESTS DO NOT EXCUSE DISHONEST IMPLEMENTATIONS
- Writing a unit test that verifies a hardcoded cheat table is an act of deception, not testing.
- Tests must verify:
  1. The learned behavior operates purely on tensor/array inputs.
  2. Banned cheat structures and shortcut dictionaries do not exist in the source code.
  3. Provenance and integrity invariants are strictly satisfied.

---

## 4. STRICT VISUAL AGENT BOUNDARIES
- Signature of `act()`: Must strictly receive `(obs_uint8, reward_prev, step)`.
- No privileged environment information, inventory dictionaries, or true achievement arrays may ever enter the agent's decision loop.
- All internal representations must stem from the visual encoder, recurrent core, and frozen backbone heads.
