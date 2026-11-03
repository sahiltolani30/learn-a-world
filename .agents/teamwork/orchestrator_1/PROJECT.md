# Project: Prometheus JEPA Forensic Investigation & Root-Cause Blueprint

## Architecture & Investigation Framework
An exhaustive, zero-guesswork forensic investigation and root-cause analysis conducted by an 8–9 agent community of specialized theorists and empirical auditors to determine the precise mathematical, architectural, and optimization bottlenecks of the Prometheus JEPA world model on the Crafter benchmark.

### Core Architecture Under Investigation
- **Visual Encoder**: Translates $64 \times 64 \times 3$ RGB observations to spatial latent tokens $Z_t \in \mathbb{R}^{8 \times 8 \times 64}$.
- **Spatial Transformer Predictor**: Autoregressively predicts next spatial latent tokens given action conditioning and cross-attention.
- **Recurrent Core (GRU)**: Aggregates temporal context and latent trajectories for multi-step credit assignment.
- **Twin Critics & Actor**: Learns state-action values under $n=5$ step TD returns and outputs actions under entropy regularization.
- **Dual-Store Buffer**: Tier-1 (online rolling buffer) + Tier-2 (milestone / achievement prioritized buffer) for sample replay.

## Feature Inventory & Requirement Mapping
| # | Requirement / Topic | Assigned Track & Agent | Deliverable Artifact | Source |
|---|---|---|---|---|
| R1.1 | Sample Complexity & MDP DAG Dynamics (Crafter 22 achievements, 25k step bounds) | Track 1: `theorist_sample_complexity_1` | `sample_complexity_audit.md` | R1 Spec |
| R1.2 | Spatial Geometry & $8 \times 8 \times 64$ Token Grid Capacity ($1 \times 1$ interactables, pooling) | Track 2: `theorist_spatial_geometry_1` | `spatial_geometry_audit.md` | R1 Spec |
| R1.3 | Bellman Propagation & $n=5$ TD Convergence (rare crafting credit assignment) | Track 3: `theorist_bellman_td_1` | `bellman_td_audit.md` | R1 Spec |
| R2.1 | Encoder & Spatial Transformer Gradient Flow Dissection (dead zones, objective clash) | Track 4: `codebase_gradient_trunk_1` | `encoder_predictor_gradient_audit.md` | R2 Spec |
| R2.2 | GRU Recurrent Core & Twin Critics Optimization Mechanics (interference, UTD, LR) | Track 5: `codebase_memory_critics_1` | `memory_critics_audit.md` | R2 Spec |
| R2.3 | Tier-1 vs Tier-2 Buffer Distribution Diagnostics (milestone echo chambers vs affordances) | Track 6: `buffer_distribution_auditor_1` | `buffer_distribution_audit.md` | R2 Spec |
| R3.1 | Anti-Cheating & Strict AST Static Analysis Integrity Audit (AGENTS.md / GEMINI.md) | Track 7: `forensic_integrity_auditor_1` | `integrity_ast_audit.md` | R3 Spec |
| R3.2 | Empirical Counter-Verification & Probe Checkpoint Execution (variance, loss, norms) | Track 8: `empirical_probe_challenger_1` | `empirical_probe_audit.md` | R3 Spec |
| R4.1 | Adversarial Cross-Examination & Scorecard Consensus Synthesis | Track 9: `adversarial_critic_synthesizer_1` | `adversarial_cross_exam.md` | R4 Spec |
| R4.2 | Definitive Root-Cause Blueprint & Feasibility Verdict (Consensus signed by community) | Project Orchestrator | `ROOT_CAUSE_BLUEPRINT_VERDICT.md` | R4 Spec |

## Milestones & Investigation Phases
| Phase | Focus | Status |
|---|---|---|
| Phase 1: Survey & Decomposition | Establish community structure, working directories, and precise specifications | DONE |
| Phase 2: Parallel Theoretical & Codebase Audits | Dispatch Tracks 1–8 in parallel to conduct deep proofs, code dissection, AST checks, and empirical probe runs | IN_PROGRESS |
| Phase 3: Adversarial Cross-Examination & Empirical Validation | Cross-examine findings between theorists and empirical auditors; resolve discrepancies | PLANNED |
| Phase 4: Definitive Root-Cause Classification & Final Blueprint | Synthesize R1–R4 into finalized consensus scorecard and actionable remediation blueprint | PLANNED |

## Interface Contracts & Evidence Requirements
1. **Mathematical Proofs**: Must include formal definitions of state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel $\mathcal{P}$, metric spaces, contraction factors, or information-theoretic entropy bounds. No hand-waving.
2. **Codebase References**: Must cite exact file paths (`prometheus/src/...`), class names, function names, and line numbers.
3. **Integrity Enforcement**: Binary veto if any agent detects hardcoded lookup tables, recipe dictionaries, or privileged state leaks.
4. **Empirical Grounding**: Any empirical claims must be supported by log data, checkpoint weight inspections, or diagnostic execution traces.
