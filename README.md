# Learn-a-World

**A JEPA-style latent world model that learns by playing.**

An agent plays simulated worlds (a custom maze → Push-T → the game Crafter). Every step it takes is stored in a capped replay buffer. A world model (encoder + action-conditioned latent predictor, custom-trained in PyTorch — no LLM APIs anywhere) learns the world's rules from that self-generated experience. A planner (MPC) dreams candidate action sequences through the model and acts on the best one — so play improves, generating richer experience: **Play → Remember → Learn → Imagine → Play better**.

Headline experiment: when the world's dynamics change at test time, the model keeps learning from its own prediction errors (test-time adaptation) and recovers — while a frozen model and a PPO baseline degrade.

---

## Status & Build Phases

- [x] **Phase 0 — Toolchain & Environment Setup** (PyTorch MPS/CUDA, Crafter, SB3, Project Scaffolding)
- [x] **Phase 1 — Custom Maze + Self-Play Loop v1** (GridMazeEnv, ReplayBuffer, JEPA Model, VICReg Anti-Collapse, 99.09% loss reduction)
- [ ] **Phase 2 — Planner + PPO Baseline** (MPC/CEM Planner + Curiosity Module + Comparative Results Table)
- [ ] **Phase 3 — Push-T + Crafter Full Loop** (Crafter Achievement Score vs Baselines on Held-Out Seeds)
- [ ] **Phase 4 — Dynamics Change & Test-Time Adaptation** (Headline Table: Frozen vs Adaptive vs PPO)
- [ ] **Phase 5 — Release** (Hugging Face Checkpoints + Model Card + Gradio Live Demo + 1-Min Video)

Full design blueprint: [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md)  
Architecture & reference notes: [`docs/REFERENCE_STUDY.md`](docs/REFERENCE_STUDY.md)

---

## Phase 1 Results (Custom Grid Maze)

| Metric | Untrained Baseline (Step 0) | Trained JEPA Model (Step 1,000) | Change |
| :--- | :---: | :---: | :---: |
| **Validation Loss** | `1.9211` | **`0.0174`** | **-99.09% error reduction** |
| **Latent Space Std-Dev ($\sigma_z$)** | `0.0238` | **`1.3742`** | **Healthy distribution (No collapse)** |
| **All Gate Tests (`pytest`)** | — | **9 passed in 78s** | **100% Pass** |

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
