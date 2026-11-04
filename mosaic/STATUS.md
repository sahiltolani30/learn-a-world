# MOSAIC Implementation Status

- **Current tranche:** P0/P1/P2 verified; P3 implemented & ready for Colab execution
- **Last local verification:** 2026-11-04
- **Training authority:** P0 passed; P1 profile verified in Colab; P2 world model passed all empirical gates in Colab; P3 implemented & tested locally; P4–P8 are locked

## Passed locally

- Exact single-policy signature test
- uint8 pixel, scalar reward, and step validation
- Crafter boundary smoke with metadata discarded
- Opaque action permutation bijection and provenance hash
- Atomic runtime checkpoint roundtrip
- Source scan for banned rule-table symbols in `mosaic/src`
- Gymnasium facade returns empty `info`
- P0 phase runner smoke
- Random P1 collection smoke
- Short CPU PPO P1 smoke
- Colab notebook JSON parse and phase-runner help
- P2 Spatial Visual Encoder & Perceptual Anchor Decoder shapes and reconstruction
- P2 Recurrent Stochastic Belief model step, prior/posterior distribution, and free-bits KL
- P2 Dynamics ensemble ($M=3$) transition predictions and epistemic disagreement
- P2 Action conditioning sensitivity and paired-action divergence test
- P2 Multi-term world model loss computation and gradient backpropagation
- P2 Target encoder EMA updates
- P2 Representation collapse diagnostic metrics (effective rank and variance check)
- P2 Atomic world model checkpoint saving and loading
- **P3 Uncertainty calibration curve binned quantile computation & Spearman rank correlation**
- **P3 Dual value streams ($V_{\text{task}}$ and $V_{\text{info}}$) parameter isolation and gradient independence**
- **P3 Active probe scorer survival risk penalty verification ($\rho \widehat{\text{Risk}}$)**
- **P3 Active probe policy adherence to strict `act(obs, reward, step)` visual contract**
- **P3 Atomic active probe runtime checkpoint saving and loading**
- `PYTHONPATH=. .venv/bin/pytest mosaic/tests -v`: 21 passed

## Passed in Colab

- `p0_contract_colab`: PASS (Clean-room contract, zero metadata leakage, bijection preserved)
- `p1_random_smoke`: PASS (`mean_reward`: 2.1, `mapping_sha256`: `df121011...`)
- `p1_profile_001`: PASS (10,000-step PPO baseline profile logged and checkpointed)
- `p2_world_model_001`: **PASS** (All P2 Gates cleared on NVIDIA GPU):
  - **Action conditioning passed**: `is_action_sensitive = True` (belief divergence: `3.162`, latent divergence: `1.631`, reward divergence: `0.138`)
  - **Beats persistence**: `model_error` ($0.000129$) is less than half of `persistence_error` ($0.000299$) with positive $\Delta = +0.000171$
  - **Collapse avoided**: `effective_rank = 5.001`, covariance off-diagonal energy = $1.75 \times 10^{-7}$, `is_collapsed = False`

## Ready to run in Colab

Run the notebook:

```text
mosaic/notebooks/MOSAIC_P0_P1_Colab.ipynb
```

Execute **Cell 14** (`p3_active_probe_001`) to run Phase P3:
1. Loads the trained P2 world model checkpoint automatically from Drive.
2. Evaluates the 3 exploration arms (`random`, `disagreement_only`, `active_probe`).
3. Computes the calibration curve and rank correlation between predicted uncertainty and true transition error.

## Still locked

- P4 persistent atlas;
- P5 anonymous relation validation;
- P6 learned skills;
- P7 planner authority;
- P8 final T4 scale and transfer.
