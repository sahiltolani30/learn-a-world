## 2026-11-02T19:02:43Z

# DISPATCH: Track 7 — Anti-Cheating & Strict AST Static Analysis Integrity Audit

**Working Directory**: `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/forensic_integrity_auditor_1`
**Role**: Forensic Integrity Auditor (`teamwork_preview_auditor`)
**Target Output**: `integrity_ast_audit.md` and `handoff.md`

## Assignment & Requirements:
1. Read `/Users/harshitatolani/Sahil/ml 2/ORIGINAL_REQUEST.md`, `/Users/harshitatolani/Sahil/ml 2/AGENTS.md`, and `/Users/harshitatolani/Sahil/ml 2/GEMINI.md`.
2. Conduct an exhaustive, systematic AST static analysis and grep search across the ENTIRE repository (`prometheus/`, `dual_store/`, `training/`, `planning/`, `atlas/`, `envs/`, `models/`, `scripts/`, etc.).
3. Verify 100% compliance with `AGENTS.md` and `GEMINI.md`:
   - Absolute ban on hardcoded tables and recipe dictionaries: verify zero occurrences of recipe lookup tables, material requirement dicts (e.g. `CRAFT_REQUIRED_MATERIALS`), action-to-ingredient mappings, terrain rules, tool requirements.
   - All relations must be learned or material gating disabled.
   - Strict visual agent boundaries: check signature of `act()` across all agents — must strictly receive `(obs_uint8, reward_prev, step)`. Zero privileged environment information, inventory dictionaries, or true achievement arrays entering the agent's decision loop.
4. Execute Python AST scanning scripts if needed to parse all `.py` files and inspect dict literals, function signatures, and data flows.
5. Provide a binary verdict: CLEAN or INTEGRITY VIOLATION. If clean, provide complete verified evidence tables of AST inspection. If any violation is found, report exact file path, line numbers, and offending code snippet.
6. Deliver your report to `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/forensic_integrity_auditor_1/integrity_ast_audit.md` and write your `handoff.md`. Send a completion message to the Project Orchestrator.
