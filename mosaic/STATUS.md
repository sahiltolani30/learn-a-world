# MOSAIC Implementation Status

- **Current tranche:** P0/P1 executable boundary and baseline runner
- **Last local verification:** 2026-11-04
- **Training authority:** P0 passed; P1 profile is ready; P2–P8 are locked

## Passed locally

- Exact single-policy signature test
- uint8 pixel, scalar reward, and step validation
- Crafter boundary smoke with metadata discarded
- Opaque action permutation bijection and provenance hash
- Atomic runtime checkpoint roundtrip
- Source scan for banned rule-table symbols in `mosaic/src`
- Gymnasium facade returns empty `info`
- `PYTHONPATH=. .venv/bin/pytest mosaic/tests -q`: 8 passed
- P0 phase runner smoke
- Random P1 collection smoke
- Short CPU PPO P1 smoke
- Colab notebook JSON parse and phase-runner help

## Ready to run in Colab

Run the notebook:

```text
mosaic/notebooks/MOSAIC_P0_P1_Colab.ipynb
```

Run cells through the P0 gate first. Then inspect the P0 manifest in:

```text
/content/drive/MyDrive/learn-a-world/mosaic/results/p0_contract_colab/
```

Only after P0 passes, set `RUN_PPO_PROFILE = True` in the notebook and run the short P1 profile. This profile measures the baseline; it is not the final MOSAIC training run.

## Still locked

The following are planned but not yet allowed to influence actions:

- stochastic predictive belief and P2 world-model comparison;
- calibrated uncertainty and P3 active probes;
- P4 persistent atlas;
- P5 anonymous relation validation;
- P6 learned skills;
- P7 planner authority;
- P8 final T4 scale and transfer.
