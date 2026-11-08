# ORBIT — Phase 3 Master Plan & Operational Specification
**Continuous Real-Game Self-Play and Online Replay Augmentation**
**Project ATLAS / ORBIT Research Program**

---

## 1. Controlling Specification & Baseline Context

This document establishes the binding operational specification for **Phase 3** of Project ORBIT, following the completion and review of Gates 0 through 6.

### Current Project Baseline & Model Preservation
* **Gate 6 Review Outcome:** Recorded as **`PARTIAL PASS — Policy learning demonstrated, overall gameplay improvement not yet demonstrated.`**
* **Provisional Policy Champion:** Gate 6 $H=4$ actor-critic checkpoint ([`results/gate6/policy_h4.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate6/policy_h4.pt)).
* **Primary World Model Baseline:** Gate 4 Formulation C residual change-weighted checkpoint ([`results/gate4/formulation_c_residual_change-weighted.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate4/formulation_c_residual_change-weighted.pt)).
* **Preserved Comparison Baselines:** Gate 6 $H=1, 2, 8$ checkpoints ([`results/gate6/`](file:///Users/harshitatolani/Sahil/ML/results/gate6/)) and Gate 5 temporal training regimes.
* **Matched Development Reference (Unmodified Crafter):**
  - Matched Random Baseline: Mean Return = **1.10 – 1.33**, Crafter Score = **7.20% – 8.40%**, Mean Survival = **161.9 – 181.1 steps**.
  - Gate 6 $H=4$ Baseline: Mean Return = **1.00 – 1.33**, Crafter Score = **6.47% – 7.73%**, Mean Survival = **153.1 – 155.4 steps**.

---

## 2. Core Objective & Fundamental Operating Invariants

### Core Objective
Transform ORBIT from an offline predictive world model into an **autonomous agent that plays unmodified Crafter, accumulates real visual transitions, updates its internal dynamics from empirical outcomes, retrains its policy in latent imagination, and demonstrates validated gameplay improvement against the matched random baseline and prior champions.**

```
       +-------------------------------------------------------------+
       |                                                             |
       v                                                             |
[1. Real Play (H=4)] ---> [2. Verified Replay] ---> [3. Update WM] --+
(Pure Pixels, tau=1.2)    (Contiguous Shards)       (Held-Out Gated) |
                                                                     v
                                                            [4. Train Policy]
                                                            (Imagination H=4)
                                                                     |
                                                                     v
                                                            [5. 30-Seed Eval]
                                                            (Random vs H4 vs New)
                                                                     |
                                             PASS Benchmark? --------+
                                            /                \
                                        [YES]                [NO]
                                          |                    |
                                  Promote Champion       Retain H=4 &
                                                         Diagnose Gap
```

### Non-Negotiable Operational Laws ([AGENTS.md](file:///Users/harshitatolani/Sahil/ML/AGENTS.md) / [GEMINI.md](file:///Users/harshitatolani/Sahil/ML/GEMINI.md))
1. **Absolute Ban on Hardcoded Recipes & Knowledge Tables:**
   - No dictionaries, lookup tables, or mappings encoding human knowledge of Crafter rules, recipes, or affordances (AST greps for banned structures must yield exactly zero hits).
   - Costs, prerequisites, and affordances must derive purely from learned model heads or empirical experience.
2. **Strict Visual Boundary:**
   - Policy decision signature is strictly [`VisualJEPAAgent.act(obs_uint8, reward_prev, step)`](file:///Users/harshitatolani/Sahil/ML/true_jepa/agent.py#L75-L100).
   - Zero access to privileged environment internals, inventories, coordinate maps, or achievement sets in the decision loop.
   - Evaluator-only metadata (`info["achievements"]`, `info["inventory"]`) is read only after action submission for logging.
3. **Execution Environment Protocol:**
   - All heavy model training, multi-step fine-tuning, and multi-seed evaluations must run on **GPU (Google Colab / CUDA)** via [`notebooks/15_orbit_phase3_selfplay_colab.ipynb`](file:///Users/harshitatolani/Sahil/ML/notebooks/15_orbit_phase3_selfplay_colab.ipynb) to avoid local CPU bottlenecks. Local runs are restricted to fast unit tests (< 10s) and static audits.

---

## 3. Sequential Execution Order (Step-by-Step)

To guarantee complete attribution and prevent confounding multiple architectural updates, execution must proceed through five isolated, uncoupled stages.

---

### Step 1 (Phase 3A): Continuous Real-Game Gameplay Collection
**Objective:** Validate that the agent can reliably play unmodified Crafter, navigate terrain, and log pristine transition tuples without updating any weights.

* **Acting Policy:** Frozen Gate 6 $H=4$ policy candidate ([`results/gate6/policy_h4.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate6/policy_h4.pt)).
* **Exploration Mechanism:** Controlled stochastic sampling with temperature $\tau = 1.2$ to break spatial immobility and mode collapse.
* **Data Scale:** 30 complete episodes, maximum 300 steps per episode.
* **Transition Tuple Schema:**
  Every transition record must explicitly preserve:
  ```python
  {
      "obs": obs_uint8,             # [64, 64, 3] uint8 visual observation
      "action": int,                # Discrete action ID [0..16]
      "reward": float,              # Received scalar reward from Crafter
      "next_obs": next_obs_uint8,   # [64, 64, 3] uint8 resulting observation
      "terminated": bool,           # Health exhaustion / natural episode death
      "truncated": bool,            # Max episode steps reached (300)
      "episode_id": int,            # Monotonic episode index
      "step": int,                  # Monotonic step counter within episode
      "policy_version": str,        # "gate6_h4"
      "wm_version": str,            # "gate4_formulation_c"
      "data_source": str,           # "real_crafter_selfplay"
  }
  ```
* **Boundary & State Management:**
  - Strict recurrent hidden state reset (`agent.reset()`) at episode boundaries.
  - Correct 1-step temporal alignment: $(o_t, a_t, r_t, o_{t+1})$.
* **Acceptance Criteria (Phase 3A Gate):**
  - Zero weight updates executed during Phase 3A.
  - Zero privileged info leaks into stored transitions.
  - Complete episodes reconstructible from disk storage.
  - Serialization to `results/phase3/data/selfplay_h4_replay.npz`.

---

### Step 2 (Phase 3B): Replay Buffer Integration & Sequence Verification
**Objective:** Integrate collected real-game experience into the versioned replay system while guaranteeing sequence contiguity for recurrent belief training.

* **Data Partitioning & Sharding:**
  - **Canonical Baseline Shard:** 54 episodes, 10,054 transitions (`data/gate3/train_replay_10k.npz`).
  - **Self-Play Online Shard:** 30 episodes, ~4,500 transitions (`results/phase3/data/selfplay_h4_replay.npz`).
  - **Combined Augmented Replay:** 84 episodes, ~14,500 transitions in memory.
* **Recurrent Sequence Integrity:**
  - Preserve contiguous trajectory order for recurrent GRU burn-in ($T_{\text{prefix}}=4$).
  - Respect episode boundary masks: trajectory sampling must never cross episode transitions.
  - Separate training and held-out evaluation datasets by **episode seeds** (never randomly slice individual frames from the same episode).
* **Token Caching Synchronization:**
  - Extract $8 \times 8 \times 192$ DINOv2 latent patch tokens for all newly collected transitions.
  - Synchronize buffer indices so that FIFO eviction or capacity caps never desynchronize raw frames from cached tokens.
* **Acceptance Criteria (Phase 3B Gate):**
  - Automated tests verify trajectory continuity and replay loading without shape or index mismatch.
  - Verified token cache saved to `results/phase3/data/selfplay_token_cache.pt`.

---

### Step 3 (Phase 3C): World Model Learning from Real Experience
**Objective:** Fine-tune the world model transition predictor and recurrent core on augmented experience, validating predictive improvements on held-out seeds.

* **Freezing & Training Bounds:**
  - Pretrained DINOv2 ViT-S/14 backbone: **Strictly Frozen**.
  - Visual spatial adapter: **Fixed** during initial online fine-tuning.
  - Trainable parameters: Transition predictor MLP/Transformer, recurrent GRU belief core, scalar reward head, continuation head.
* **Training Formulation:** Autoregressive rollout ($H=4$) with change-weighted token loss (Formulation C).
* **Held-Out Diagnostic Evaluation (`data/gate3/eval_heldout_seeds.pt`):**
  Compare candidate against baseline Gate 4 Formulation C:
  1. 1-step overall relative prediction error: $\text{MSE}_{\text{model}} / \text{MSE}_{\text{persist}}$.
  2. Changed-token relative prediction error (tokens with $\|\Delta z\| > 0.1$).
  3. Static-token relative prediction error.
  4. Shuffled-action sensitivity gap ($\text{MSE}_{\text{shuffled}} - \text{MSE}_{\text{true}}$).
  5. Multi-step teacher-forced and free-running prediction errors across $H \in \{1, 2, 4, 8\}$.
  6. Scalar reward head MAE and continuation calibration.
* **Acceptance Criteria (Phase 3C Gate):**
  - Candidate must demonstrate a lower relative prediction error than the Gate 4 baseline on held-out data.
  - Candidate must maintain positive shuffled-action sensitivity.
  - Selected world model is saved to `results/phase3/world_model_phase3.pt` and **strictly frozen** (`requires_grad_(False)`).

---

### Step 4 (Phase 3D): Policy Retraining via Latent Imagination
**Objective:** Train an updated actor-critic policy within the latent rollouts of the validated, frozen world model.

* **Architecture & Hyperparameters:**
  - Rollout Horizon: $H=4$.
  - Return Formulation: Generalized Advantage Estimation (GAE $\lambda=0.95, \gamma=0.99$).
  - Entropy Regularization: Calibrated annealing ($\beta = 0.03 \to 0.005$) to balance broad movement exploration with targeted harvesting density.
  - Value Target Normalization: Symlog value transformation / Huber clipping to prevent negative value divergence.
* **Anti-Exploitation Protections:**
  - Strict world model parameter freeze (`wm.requires_grad_(False)`).
  - Reward clipping to $[-1.0, 1.0]$.
  - Transition uncertainty penalty based on ensemble or predictor residual variance.
  - Online telemetry tracking: actor loss, value loss, mean imagined reward, mean imagined value, and action entropy.
* **Acceptance Criteria (Phase 3D Gate):**
  - Stable value progression without runaway drift.
  - Checkpoint saved to `results/phase3/policy_phase3.pt`.

---

### Step 5 (Phase 3E): Rigorous Real-Game Crafter Benchmark
**Objective:** Evaluate real gameplay performance in unmodified Crafter across 30 identical pinned seeds.

* **Evaluation Protocol:**
  - Environment: Unmodified Crafter (`crafter.Env()`).
  - Evaluation Seeds: Pinned 30 seeds (`5000..5029`), max 300 steps per episode.
  - Evaluated Arms:
    1. **Matched Random Baseline**
    2. **Gate 6 Provisional Champion ($H=4$)**
    3. **Newly Trained Phase 3 Policy Candidate**
* **Reporting Metrics:**
  - Crafter Score (%) and Geometric Progression.
  - Mean Return $\pm$ Standard Deviation and Median Return.
  - Survival Duration (Mean, Median, Min, Max steps).
  - Full 22 Achievement Unlock Rates (Evaluator-only).
  - Action Distribution and Shannon Entropy across all 17 actions.
  - Complete 30-seed per-seed comparison table.
* **Promotion Decision Matrix:**
  - **PROMOTE TO CHAMPION:** If and only if the candidate outperforms both the Matched Random Baseline and the Gate 6 Champion on mean return and Crafter score across the 30 seeds without violating integrity invariants.
  - **RETAIN GATE 6 CHAMPION:** If performance does not surpass both baselines, retain [`results/gate6/policy_h4.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate6/policy_h4.pt) as champion, archive candidate metrics, and report failure analysis honestly.

---

## 4. Phase 3F — Continuous Improvement Loop Architecture

Once all five stages are validated independently, they connect into an automated, versioned learning loop:

```
[Collection Loop] ---> [Replay Cache] ---> [World Model Training]
       ^                                            |
       |                                            v
[Deployment Gate] <--- [30-Seed Benchmark] <--- [Policy Imagination]
 (Strict Promotion Rule)
```

1. **Automated Safety Invariant:** The system must never automatically overwrite or replace a superior policy with an inferior candidate.
2. **Traceable Lineage Manifest:** Every trained checkpoint must be accompanied by a JSON manifest documenting the exact data shards, world model version, training hyperparameters, and seed-level evaluation returns.

---

## 5. Implementation Artifacts and Codebase Integration

All functionality reuses existing cleanroom modules without code duplication:

| Operational Component | Existing Module Reused / Extended | Output Path |
|:---|:---|:---|
| Visual Agent Interface | [`true_jepa/agent.py`](file:///Users/harshitatolani/Sahil/ML/true_jepa/agent.py) (`VisualJEPAAgent`) | In-memory agent loop |
| Replay Buffer & Cache | [`true_jepa/replay.py`](file:///Users/harshitatolani/Sahil/ML/true_jepa/replay.py) (`PixelReplay`, `CachedDINOv2Replay`) | `results/phase3/data/` |
| World Model Dynamics | [`true_jepa/model.py`](file:///Users/harshitatolani/Sahil/ML/true_jepa/model.py) (`TrueJEPA`) | `results/phase3/world_model_phase3.pt` |
| Latent Imagination & GAE | [`true_jepa/objectives.py`](file:///Users/harshitatolani/Sahil/ML/true_jepa/objectives.py) (`imagination_actor_critic_gae_step`) | In-memory policy trainer |
| Verification Test Suite | [`tests/test_orbit_phase3_selfplay.py`](file:///Users/harshitatolani/Sahil/ML/tests/test_orbit_phase3_selfplay.py) | Pytest pass logs |
| Colab GPU Runner | [`notebooks/15_orbit_phase3_selfplay_colab.ipynb`](file:///Users/harshitatolani/Sahil/ML/notebooks/15_orbit_phase3_selfplay_colab.ipynb) | Google Colab runtime |
| End-to-End Orchestrator | [`experiments/run_phase3_iterative_selfplay.py`](file:///Users/harshitatolani/Sahil/ML/experiments/run_phase3_iterative_selfplay.py) | Stage execution engine |
| Experiment Telemetry | [`results/phase3/phase3_experiment_metrics.json`](file:///Users/harshitatolani/Sahil/ML/results/phase3/phase3_experiment_metrics.json) | Complete metrics JSON |
| Comparative Plots | [`results/phase3/phase3_comparison_plots.png`](file:///Users/harshitatolani/Sahil/ML/results/phase3/phase3_comparison_plots.png) | 4-panel figure |

---

## 6. Phase 3A Execution Contract

Per user instruction:
> *"Start with Phase 3A only. Don't let the implementation jump into online training until we know that ORBIT can reliably play, collect, save, and reload real experience."*

The immediate operational priority is to verify and freeze **Phase 3A Data Collection & Replay Serialization**:
1. Confirm 30-episode rollout execution using frozen Gate 6 $H=4$ under $\tau=1.2$.
2. Verify schema compliance and temporal alignment: $(o_t, a_t, r_t, o_{t+1})$.
3. Verify reloadability from disk and token cache synchronization.
4. Issue formal Phase 3A verification report before authorizing Phase 3B/C.
