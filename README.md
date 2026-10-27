# Learn-a-World

**A JEPA-style latent world model that learns by playing.**

An agent plays simulated worlds (a custom maze → Push-T → the game Crafter). Every step it takes is stored in a capped replay buffer. A world model (encoder + action-conditioned latent predictor, custom-trained in PyTorch — no LLM APIs anywhere) learns the world's rules from that self-generated experience. A planner (MPC) dreams candidate action sequences through the model and acts on the best one — so play improves, generating richer experience: **Play → Remember → Learn → Imagine → Play better**.

Headline experiment: when the world's dynamics change at test time, the model keeps learning from its own prediction errors (test-time adaptation) and recovers — while a frozen model and a PPO baseline degrade.

---

## Status & Build Phases

- [x] **Phase 0 — Toolchain & Environment Setup** (PyTorch MPS/CUDA, Crafter, SB3, Project Scaffolding)
- [x] **Phase 1 — Custom Maze + Self-Play Loop v1** (GridMazeEnv, ReplayBuffer, JEPA Model, VICReg Anti-Collapse, 99.09% loss reduction)
- [x] **Phase 2 — Planner + PPO Baseline** (MPC/CEM Planner + Curiosity Module + Comparative Results Table)
- [x] **Phase 3 — Push-T + Crafter Full Loop** (Push-T Physics + Crafter Achievement Score + Drive Resumability)
- [x] **Phase 4 — Dynamics Change & Test-Time Adaptation** (Headline Table: Frozen vs Adaptive vs PPO)
- [x] **Phase 1.5 — Causal State-Delta Learning Gate** (Narrow state-delta gate passed; Run 3B preserved as candidate transition model)
- [x] **v2 Promotion — Pure Learned World Model & Locked Evaluation** (**2.16%** on Crafter benchmark across 100 locked seeds — **PROMOTED TO V2 ✅**)
- [ ] **Phase 5 — Hierarchical Subgoal Layer & Full Release** (Autonomous Causal Precedence DAG Mining, Hugging Face Checkpoints & Model Card)

*(Note: The original v1 hybrid milestone remains permanently archived and accessible at tag [`v1-freeze`](https://github.com/sahiltolani30/learn-a-world/tree/v1-freeze)).*

Full design blueprint: [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md)  
Comprehensive results & per-achievement breakdown: [`RESULTS.md`](RESULTS.md)

---

## Phase 1.5: Causal State-Delta Gate

Phase 1.5 trains auxiliary transition heads on the JEPA representation and requires the
predicted state-change MSE to beat a zero-change persistence baseline on held-out data.
Both independent Colab runs passed this narrow gate; **Run 3B is preserved as a candidate
transition model**, while `jepa_crafter_v3_t2_final.pt` remains the standing agent champion.

| Run | State-Delta MSE | Persistence MSE | Gate | Validation Summary |
|:---|---:|---:|:---:|:---|
| Run 3A (`2b952a4`) | **0.0161188** | 0.0164005 | **PASS (narrow)** | 315 unique held-out labels; 6,528 sampled validation positions |
| Run 3B (`67bff02`) | **0.0158931** | 0.0159792 | **PASS (narrow)** | 315 unique held-out labels; 6,528 sampled validation positions |

The reported 6,528 validation positions are sampled sequence positions, not 6,528
independent labels (dataset contains 2,000 unique valid labels across 13 episodes). The termination head remained at the majority baseline with zero recall/precision.
Gameplay regression on seeds 2050–2099 scored 3.398% vs the 4.55% T2 reference baseline, so full Phase 1.5 agent promotion remains pending.

### Strict Visual Crafter Screen

The Phase 1.5 checkpoint was evaluated without privileged inventory inputs on 50 fixed
episodes (`2050..2099`). It reached **3.398% Crafter Score**, improving on the clean
3.171% screen, but it does not replace the existing 4.55% T2 reference benchmark.

| Checkpoint | Episodes | Crafter Score | Mean Survival | Mean Reward | Achievements |
|:---|---:|---:|---:|---:|---:|
| Phase 1.5 candidate | 50 | **3.398%** | 178.34 | 3.42 | 4.32 / ep, 10 / 22 |
| Clean T2 champion screen | 50 | 3.171% | 172.96 | 3.34 | 4.24 / ep, 10 / 22 |
| T2 reference benchmark | 50 | **4.55%** | 181.36 | 4.12 | 5.02 / ep, 12 / 22 |

Artifacts: [`Run 3A`](results/phase1_5_training_run3a.json) ·
[`Run 3B`](results/phase1_5_training_run3b.json) ·
[`Candidate screen`](results/screen_v3_phase1_5_candidate.json) ·
[`Phase 1.5 audit`](results/phase1_5_review.md)

---

## Headline Experiment: Test-Time Dynamics Adaptation (Phase 4)

When world dynamics secretly shift at test time (e.g. controls inverted), the **Adaptive JEPA World Model** detects prediction errors and recovers online purely from self-supervised visual observations with **ZERO environment reward**, while the Frozen World Model collapses and PPO fails.

| World Dynamics | Method | Success Rate (%) | Recovery Steps | Reward Needed? |
| :--- | :--- | :---: | :---: | :---: |
| Normal Physics | **Frozen WM + MPC** | **73.3%** | — | No |
| Normal Physics | **Adaptive WM + MPC** | **60.0%** | — | No |
| Normal Physics | **PPO Baseline** | 0.0% | — | Yes |
| Changed Physics (Inverted) | **Frozen WM + MPC** | 0.0% | — | No |
| Changed Physics (Inverted) | **Adaptive WM + MPC** | **46.7%** | **~125 steps** | **No (Zero)** |
| Changed Physics (Inverted) | **PPO Baseline** | 0.0% | Failed | Yes (Retrain) |

*All 24 gate tests across Phases 0, 1, 2, 3, and 4 passing with 100% parity (`pytest -v tests/`).*

---


## Crafter Flagship Benchmark: v2 Promotion Results

> **Protocol & Budget:** **2.16%** Crafter Score · 100 episodes on fixed held-out seeds `1000..1099` (max 1,000 steps/ep) · 390,000 transitions collected (300k starting dataset + 90k self-play) · **Pure learned, zero scripted rules**.

| Agent | Architecture & Planning Mode | Reflexes / Rules | Crafter Score | Achievements Unlocked | Mean Survival | Mean Reward | Promotion Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`v2_pure` (Ours)** | **JEPA v2 + Learned RewardValuePlanner** | **None (Pure Learned)** | **2.16%** | **10 / 22** | 171.1 steps | **+2.09** | **PROMOTED ✅** |
| **`v1_hybrid`** | Latent JEPA World Model + MPC | **Hybrid (Scripted Reflexes ON)** | **1.44%** | 7 / 22 | 179.5 steps | +1.62 | Baseline to beat |
| **`random`** | Uniform Random Action Policy | None | **1.77%** | 7 / 22 | 188.7 steps | +1.75 | Baseline |
| **`v1_pure`** | Latent JEPA World Model + MPC | **Scripted Reflexes OFF** | **0.00%** | 0 / 22 | 177.0 steps | -0.90 | Control (Ablation) |

*Full 22-achievement breakdown, training curves, single source of truth, and honest caveats (random survival baseline, tool crafting frequency, and planning horizon):* 👉 **[`RESULTS.md`](RESULTS.md)**  
*Promotion decision audit & gate report:* 👉 **[`results/promotion_decision.md`](results/promotion_decision.md)**

---


## Phase 2 Benchmark Results (Latent Planning & Exploration)

Comparison evaluated across 20 test episodes on `GridMaze-v0`:

| Agent Method | Success Rate (%) | Avg Steps to Goal | Env Interaction Budget | Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Random Agent** | 26.7% | 20.7 | 0 steps | Wanders randomly; high step count |
| **World Model + MPC** | **73.3%** | **9.1** | 4,000 steps | Latent CEM planning cuts steps by >55% |
| **WM + MPC + Curiosity** | **66.7%** | **10.3** | 4,474 steps | **High success & fast paths** via self-play fine-tuning |
| **PPO Baseline** | 0.0% | 100.0 | 5,000 steps | Fails on sparse reward without world model |

*All 14 gate tests across Phases 0, 1, and 2 passing with 100% parity (`pytest -v tests/`).*

---

## Phase 1 Results (Custom Grid Maze)

| Metric | Untrained Baseline (Step 0) | Trained JEPA Model (Step 1,000) | Change |
| :--- | :---: | :---: | :---: |
| **Validation Loss** | `1.9211` | **`0.0174`** | **-99.09% error reduction** |
| **Latent Space Std-Dev ($\sigma_z$)** | `0.0238` | **`1.3742`** | **Healthy distribution (No collapse)** |
| **All Gate Tests (`pytest`)** | — | **14 passed** | **100% Pass** |

---

## Tech Stack
- **Frameworks:** Python 3.11 · PyTorch 2.x (Apple Silicon MPS / NVIDIA CUDA) · Gymnasium 1.x
- **Environments:** Custom Grid Maze (`GridMaze-v0`) · Push-T · Crafter (Survival Benchmark)
- **RL & Baselines:** Stable-Baselines3 (PPO Baseline)
- **Deployment & Demo:** Hugging Face Hub · Gradio · Google Colab (Free T4 GPU)

---

## Quickstart

### 1. Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Verification & Gate Tests
```bash
python scripts/verify_phase0.py
pytest -v tests/
```

### 3. Visualize Environments
```bash
# Visualize custom GridMaze environment & agent:
python scripts/visualize_maze.py

# Visualize Crafter game world:
python scripts/visualize_world.py
```

### 4. Train JEPA World Model on Maze
```bash
# Local Mac (Apple Silicon MPS or CPU):
python -m training.trainer --config experiments/configs/maze.yaml

# Google Colab (Free T4 GPU with Drive checkpointing):
# Open notebooks/01_train_colab.ipynb and run all cells
```
