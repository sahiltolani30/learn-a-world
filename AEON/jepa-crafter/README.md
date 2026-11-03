# JEPA Crafter Agent — From-Scratch Self-Supervised World Model

Implementation of the complete engineering plan specified in `AEON JEPA_Crafter_Implementation_Plan.md`.

## System Overview
- **Visual Encoder:** 3-stage convolutional residual network with GroupNorm and GELU, yielding an 8×8 spatial grid of 128-dimensional tokens (64 tokens total) and an attention-pooled global latent.
- **EMA Target Encoder:** Exponential Moving Average stop-gradient target network with momentum ramp from 0.996 to 0.999.
- **Action Embedding:** Learns an embedding for 17 anonymous opaque action tokens.
- **Temporal Context:** GRU summarizing recent observations and actions.
- **Latent Predictor Ensemble:** 3 independent transformer-based predictors for multi-horizon autoregressive dynamics and epistemic uncertainty.
- **Replay Memory:** Sequence-aware circular buffer preserving raw uint8 RGB observations and preventing sequence sampling across episode boundaries.
- **Latent Planning:** Cross-Entropy Method (CEM) planner evaluating candidate sequences in imagined latent space.
- **Zero Information Leakage:** Zero game mechanics, crafting recipe tables, or privileged environment inputs. Strict quarantine isolates `info` to metrics evaluation only.

---

## Directory Structure
```
jepa-crafter/
├── README.md
├── requirements.txt
├── configs/
│   ├── crafter_t4.yaml
│   ├── crafter_t4_reward_free.yaml
│   └── evaluation.yaml
├── notebooks/
│   ├── 00_environment_smoke_test.ipynb
│   ├── 01_collect_random_data.ipynb
│   ├── 02_train_jepa.ipynb
│   ├── 03_train_agent.ipynb
│   └── 04_evaluate_and_ablate.ipynb
├── src/
│   ├── env/
│   │   ├── crafter_adapter.py
│   │   ├── opaque_actions.py
│   │   ├── learner_observation_filter.py
│   │   └── evaluator_tap.py
│   ├── data/
│   │   ├── replay_buffer.py
│   │   ├── sequence_sampler.py
│   │   ├── episode_store.py
│   │   └── checkpoint_store.py
│   ├── models/
│   │   ├── visual_encoder.py
│   │   ├── ema_encoder.py
│   │   ├── temporal_context.py
│   │   ├── action_embedding.py
│   │   ├── latent_predictor.py
│   │   ├── predictor_ensemble.py
│   │   ├── auxiliary_heads.py
│   │   └── jepa_world_model.py
│   ├── learning/
│   │   ├── jepa_losses.py
│   │   ├── learner.py
│   │   ├── optimizer.py
│   │   ├── feature_diagnostics.py
│   │   └── update_scheduler.py
│   ├── exploration/
│   │   ├── latent_memory.py
│   │   ├── novelty.py
│   │   ├── disagreement.py
│   │   ├── role_definitions.py
│   │   └── action_selector.py
│   ├── planning/
│   │   ├── latent_rollout.py
│   │   ├── cem_planner.py
│   │   ├── candidate_scorer.py
│   │   └── planner_diagnostics.py
│   ├── evaluation/
│   │   ├── achievement_recorder.py
│   │   ├── crafter_score.py
│   │   ├── heldout_prediction.py
│   │   ├── action_conditioning_tests.py
│   │   ├── relation_diagnostics.py
│   │   └── report.py
│   ├── training/
│   │   ├── collect.py
│   │   ├── train_world_model.py
│   │   ├── train_agent.py
│   │   └── resume.py
│   └── utils/
│       ├── config.py
│       ├── seeding.py
│       ├── logging.py
│       ├── device.py
│       └── safety_checks.py
└── tests/
    ├── test_env_contract.py
    ├── test_no_info_leak.py
    ├── test_opaque_actions.py
    ├── test_replay_sequences.py
    ├── test_ema_update.py
    ├── test_jepa_shapes.py
    ├── test_no_episode_crossing.py
    ├── test_checkpoint_resume.py
    ├── test_planner_replans.py
    └── test_evaluator_isolation.py
```

---

## Setup & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running Automated Test Suite
All 10 required acceptance tests:
```bash
pytest -q tests/test_env_contract.py
pytest -q tests/test_no_info_leak.py
pytest -q tests/test_opaque_actions.py
pytest -q tests/test_replay_sequences.py
pytest -q tests/test_ema_update.py
pytest -q tests/test_jepa_shapes.py
pytest -q tests/test_no_episode_crossing.py
pytest -q tests/test_checkpoint_resume.py
pytest -q tests/test_planner_replans.py
pytest -q tests/test_evaluator_isolation.py
```
Expected result: **All 10 test modules pass cleanly with zero failures.**

### 3. Training
To train the agent using the standard configuration:
```python
from src.utils.config import load_config
from src.training.train_agent import run_training

config = load_config("configs/crafter_t4.yaml")
results = run_training(config)
```

To run reward-free intrinsic exploration:
```python
config = load_config("configs/crafter_t4_reward_free.yaml")
results = run_training(config)
```

---

## Benchmark Metrics & Artifacts
The training and evaluation pipeline automatically produces:
1. `final_report.md`: Markdown summary of architecture, validation gates, per-achievement stats, and limitations.
2. `achievement_rates.csv`: Success rates for all 22 Crafter achievements and the geometric-mean Crafter score.
3. `heldout_prediction.json`: Multi-horizon (1, 2, 4, 8) future-latent prediction comparisons against persistence and ablations.
4. `action_conditioning.json`: Paired action test outcomes validating genuine action-conditioned dynamics.
5. `metrics.jsonl` / `metrics.csv`: Real-time training loss and anti-collapse geometry metrics.
