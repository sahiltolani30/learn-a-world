# MOSAIC Implementation Plan

- **Model:** MOSAIC — Persistent Visual Skill Atlas
- **Repository package:** `mosaic/`
- **Drive package:** `learn-a-world/mosaic/`
- **Status:** P0/P1 executable tranche implemented; P2–P8 remain empirically gated

The current implementation status and exact Colab handoff are recorded in [`STATUS.md`](STATUS.md).

## 1. Why this project structure

The existing model projects establish several useful conventions:

- **Prometheus** uses an isolated model package, README, source modules, tests, and a Colab notebook.
- **Zephyr** uses an isolated package, explicit module contracts, Colab setup/run scripts, checkpoint manifests, and smoke/e2e tests.
- **ATLAS** uses a prescriptive plan, phase gates, a strict import boundary, frozen-component provenance, resumable self-learning iterations, and diagnostic artifacts.
- The repository Drive workflow uses a model-specific folder with checkpoints, logs, shards, and results so different model families cannot overwrite one another.

MOSAIC adopts these operational patterns while retaining its own visual-only boundary and learned-only relation policy. Existing `atlas/`, `zephyr/`, `prometheus/`, `true_jepa/`, and `AEON/jepa-crafter/` code remains outside the MOSAIC package unless a later phase explicitly audits and reuses a component.

## 2. Locked research decisions

1. Measure a strong recurrent and object-centric PPO baseline before adding hierarchy.
2. Begin from the verified AEON visual/predictive baseline, but test a stochastic belief and an information anchor rather than assuming pure JEPA is sufficient.
3. Use action-conditioned uncertainty only after calibration against held-out transitions.
4. Keep task value and information value in separate streams.
5. Store anonymous visual evidence and controlled interventions, never a human-authored rule table.
6. Promote memory, skills, and planning only after shadow-mode gates pass.
7. Preserve primitive-action fallback at every phase.
8. Save atomic, resumable checkpoints to `learn-a-world/mosaic/` on Drive.
9. Never put model weights, replay shards, or large generated artifacts in Git.

## 3. Project and artifact layout

### 3.1 Git layout

```text
mosaic/
├── README.md
├── PLAN.md
├── colab/
│   ├── README.md
│   ├── setup_colab.sh       # Phase P0/P1 deliverable
│   └── run_phase.py         # Phase P1 deliverable
├── notebooks/
│   └── MOSAIC_P0_P1_Colab.ipynb
├── configs/
│   ├── p0_contract.yaml     # Phase P0 deliverable
│   ├── p1_baseline.yaml     # Phase P1 deliverable
│   └── mosaic_t4.yaml       # added only after profiling
├── src/
│   ├── api.py               # exact visual policy boundary
│   ├── env_boundary.py      # sanitized environment adapter
│   ├── baselines/           # recurrent/object-centric controls
│   ├── models/              # encoder, belief, dynamics, heads
│   ├── data/                # sequence replay and provenance
│   ├── exploration/         # calibrated uncertainty and probes
│   ├── memory/              # atlas and anonymous evidence
│   ├── skills/              # validated learned options
│   ├── planning/            # receding-horizon controller
│   ├── training/            # phase-specific learners
│   └── evaluation/          # offline metrics and reports
└── tests/
    ├── test_contract.py
    ├── test_no_leak.py
    ├── test_permutation.py
    ├── test_replay_provenance.py
    └── test_phase_gates.py
```

The listed implementation paths are planned interfaces, not files claimed to exist in this bootstrap commit.

### 3.2 Google Drive layout

```text
/content/drive/MyDrive/learn-a-world/mosaic/
├── checkpoints/
├── logs/
├── shards/
├── results/
├── manifests/
├── caches/
└── notebooks/
```

Each run uses a subdirectory under `results/` or `checkpoints/` containing the code SHA, configuration hash, seed partition, opaque mapping hash, and hardware manifest.

## 4. Phase-by-phase implementation

### P0 — Clean-room control contract

**Purpose:** make the policy boundary executable before learning anything.

**Implement:**

- `src/api.py` with exact `act(obs_uint8, reward_prev, step) -> int` signature;
- reset-on-`step == 0` behavior;
- uint8 shape/dtype validation and opaque-token output validation;
- environment adapter that strips metadata before learner access;
- changed-token-permutation integration test;
- runtime leak inspection and source/data-flow audit;
- checkpoint roundtrip for recurrent state and RNG.

**Gate:** all contract, reset, permutation, and no-leak tests pass. No model score is reported before this gate.

**Execution:** local smoke only. No long Colab run.

### P1 — Strong baseline ladder

**Purpose:** establish the control experiment that MOSAIC must beat.

**Implement/measure:**

- random policy;
- feedforward PPO control;
- recurrent PPO control;
- recurrent/object-centric PPO control if the implementation is available under the same boundary;
- identical train/evaluation seed partitions and environment-step budget;
- throughput, latency, memory, survival, reward, breadth, depth, and per-outcome metrics.

**Gate:** a reproducible baseline report exists and the strongest baseline is selected using a pre-registered primary metric. A weak baseline cannot unlock MOSAIC complexity.

**Execution:** first short Colab profile after P0 is complete. Still no full 1M-step training run.

### P2 — Predictive visual belief

**Purpose:** test whether a predictive model adds useful state and long-horizon credit.

**Implement:**

- AEON encoder/grid baseline adapter;
- recurrent stochastic belief with posterior/prior distinction;
- action-conditioned dynamics ensemble;
- reward and continuation heads;
- masked visual/information anchor as an ablation;
- held-out prediction, action-conditioning, collapse, scene-swap, and real/imagination-gap diagnostics.

**Gate:** declared-horizon prediction beats persistence, action conditioning is demonstrated, representation does not collapse, and the result is measured against P1. Prediction alone does not unlock planning.

**Execution:** Colab only after P1 baseline results are saved and the P2 smoke tests pass.

### P3 — Calibrated uncertainty and active probing

**Purpose:** convert uncertainty into controlled experiments rather than an unexplained curiosity bonus.

**Implement:**

- ensemble disagreement and aleatoric uncertainty logging;
- held-out calibration curves and selective-risk metrics;
- information-progress measurement;
- active probe scorer with survival/risk and returnability terms;
- random, novelty-only, disagreement-only, raw-error, task-only, and active controls;
- shadow-mode probe traces.

**Gate:** uncertainty predicts held-out transition error, verified interventions reduce uncertainty, and probing improves a downstream metric without unacceptable task regression.

**Execution:** a dedicated short Colab segment. Do not start multi-hour training until the calibration report is written.

### P4 — Persistent visual atlas

**Purpose:** retain useful discoveries across resets and representation updates.

**Implement:**

- versioned visual keys and context embeddings;
- raw trajectory references and reliability metadata;
- closed-loop return policy references;
- immutable/slow retrieval projection;
- reservoir, recent, discovery, failure, and uncertainty replay streams;
- reset-retention and held-out visual-appearance tests.

**Gate:** hybrid memory beats recurrent-only and single-memory controls on retrieval, reset retention, returnability, and transfer.

**Execution:** Colab self-learning segment with atomic Drive checkpoints. Atlas remains observational until the gate passes.

### P5 — Anonymous relation evidence

**Purpose:** learn visual action-effect relations without inserting human rules.

**Implement:**

- append-only anonymous evidence records;
- predicted-versus-observed visual deltas;
- repeated controlled interventions and negative trials;
- held-out contexts and contradiction handling;
- declared multiple-testing/FDR procedure;
- offline relation reports.

**Gate:** only relations that pass controlled and held-out tests may influence behavior. Material/progression gating stays disabled until this gate passes.

**Execution:** Colab collection and offline validation. Every candidate remains provisional until promoted.

### P6 — Learned options and skills

**Purpose:** shorten long action chains with reusable closed-loop controllers.

**Implement:**

- option initiation and termination heads;
- learned visual/outcome goals;
- trajectory mining and candidate option training;
- primitive fallback;
- shadow execution and reliability tracking.

**Gate:** held-out outcome reliability, termination calibration, distinctness, reuse, and matched-compute improvement over primitive control.

**Execution:** Colab after P5; never promote an option from diversity alone.

### P7 — Receding-horizon planning

**Purpose:** use calibrated dynamics, atlas entries, and options without trusting imagination blindly.

**Implement:**

- option/trajectory retrieval;
- robust ensemble rollout scoring;
- task/information/risk/model-gap terms logged independently;
- one-action or short-prefix execution followed by real verification;
- actor-only and primitive CEM controls;
- planner shadow mode and authority switch.

**Gate:** planner improves competence or discovery at matched compute, with bounded real/imagination gap and acceptable p95 action latency.

**Execution:** Colab only after P7 shadow diagnostics pass. This is the first point at which full MOSAIC planning authority may be enabled.

### P8 — T4 scale, transfer, and final ablations

**Purpose:** evaluate the complete promoted system under one locked protocol.

**Implement/run:**

- T4 worker/batch/update/precision/planner profile;
- independent training and held-out seed partitions;
- final model and every promoted predecessor;
- memory, option, planner, uncertainty, and anchor ablations;
- full manifest, raw metrics, checkpoints, and failure reports.

**Gate:** statistically supported improvement over the strongest baseline and complete integrity/transfer report. If a module does not help, remove or disable it.

## 5. Test and artifact policy

Every phase follows:

1. Write the contract test.
2. Run it and observe the expected failure.
3. Implement the smallest compliant change.
4. Run focused tests, then the phase suite.
5. Save a phase manifest and gate result.
6. Only then allow the next module to influence actions.

Each phase writes:

- `manifests/<phase>_<run_id>.json`;
- `logs/metrics.jsonl`;
- `results/<phase>_<run_id>/summary.md`;
- raw machine-readable metrics;
- checkpoint hashes when training is involved;
- code SHA and working-tree status.

Large generated artifacts remain on Drive. Git stores code, tests, configs, plans, lightweight reports, and hashes.

## 6. When to run Colab

### Do not run now

The current commit is a planning/bootstrap commit. The exact MOSAIC adapter and baseline runner do not exist yet. Running a long training job now would produce an unvalidated result and would not test the planned architecture.

### First Colab run

Run Colab after these local conditions are true:

- P0 files exist;
- P0 tests pass;
- a one-episode masked-environment smoke works;
- the repository is pushed at a clean SHA;
- the Drive `mosaic/` folders exist;
- a phase manifest is ready.

The first Colab run should be a short P0/P1 profile and baseline smoke, not full training.

### Planned Colab bootstrap cells

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!rm -rf /content/learn-a-world
!git clone https://github.com/sahiltolani30/learn-a-world.git /content/learn-a-world
%cd /content/learn-a-world
!python -m pip install --upgrade pip
!python -m pip install -r AEON/jepa-crafter/requirements.txt
```

```python
from pathlib import Path

drive_root = Path('/content/drive/MyDrive/learn-a-world/mosaic')
for name in ('checkpoints', 'logs', 'shards', 'results', 'manifests', 'caches', 'notebooks'):
    (drive_root / name).mkdir(parents=True, exist_ok=True)
print(drive_root)
```

The eventual runner will use a command of this form after it is implemented:

```bash
python -m mosaic.colab.run_phase \
  --phase p1_baseline \
  --drive-root /content/drive/MyDrive/learn-a-world/mosaic \
  --config mosaic/configs/p1_baseline.yaml \
  --run-id p1_profile_001
```

Do not substitute a full training command until the corresponding phase gate is explicitly marked ready.

### Resuming after disconnect

The runner will atomically write a checkpoint to the MOSAIC Drive folder containing model state, optimizer/scaler state, replay provenance, atlas/option state when present, mapping hash, scheduler state, and RNG states. A new Colab session mounts Drive, pulls the same Git SHA, uses the same run ID, and resumes from the latest valid checkpoint.

## 7. Definition of done

MOSAIC is not considered complete because it has a high training reward or a single successful trajectory. Completion requires:

- exact visual-only API and clean information boundary;
- reproducible strong baseline;
- validated representation and calibrated dynamics;
- measured memory/reset/transfer value;
- controlled anonymous relation evidence;
- reliable skills and planner under matched compute;
- full T4 manifest and reproducibility artifacts;
- failed ablations and limitations reported;
- no hand-authored game-rule shortcut anywhere in the decision loop.
