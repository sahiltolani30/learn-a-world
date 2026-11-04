# MOSAIC

**MOSAIC — Persistent Visual Skill Atlas** is the next isolated visual-learning project for Crafter in this repository.

## Status

This directory contains the **P0/P1 executable boundary and implementation plan**. The research specification is complete; later world-model, memory, skill, and planner phases remain locked behind empirical gates.

Start here:

1. [`PLAN.md`](PLAN.md) — phase-by-phase implementation plan and promotion gates.
2. [`STATUS.md`](STATUS.md) — current implementation status and exact next run.
3. [`colab/README.md`](colab/README.md) — when and how to run the first Colab job.
4. [`notebooks/MOSAIC_P0_P1_Colab.ipynb`](notebooks/MOSAIC_P0_P1_Colab.ipynb) — the project Colab notebook.
5. [`../research/MOSAIC_Architecture_Specification.md`](../research/MOSAIC_Architecture_Specification.md) — architecture, equations, evidence, and red-team decisions.
6. [`../research/MOSAIC_Evidence_Ledger.md`](../research/MOSAIC_Evidence_Ledger.md) — source evidence and repository audit.
7. [`../research/MOSAIC_Decision_Ledger.md`](../research/MOSAIC_Decision_Ledger.md) — debated alternatives and falsification gates.

## Non-negotiable control boundary

The public single-environment policy method is exactly:

```python
act(obs_uint8, reward_prev, step) -> int
```

The policy receives pixels, the previous scalar reward, the public step/reset signal, and its own private learned state. It never receives environment metadata, structured inventory, semantic action names, evaluator annotations, or hand-authored game rules. The environment adapter owns the external opaque-token permutation.

## Planned layout

```text
mosaic/
├── README.md
├── PLAN.md
├── colab/
│   └── README.md
├── notebooks/
│   └── MOSAIC_P0_P1_Colab.ipynb
├── configs/             # added with the first executable phase
├── src/                 # implementation added phase by phase
├── tests/               # contract and phase gates added before code authority
└── artifacts/           # manifests and lightweight reports only
```

Large checkpoints, replay shards, and run logs belong in the model-specific Google Drive directory, not Git.

## Colab status

Do **not** start a training run from this bootstrap commit. Colab becomes appropriate after the P0 contract tests and the P1 baseline runner exist and pass their local smoke checks. The first Colab job is a short validation/profile run, not the full training budget.
