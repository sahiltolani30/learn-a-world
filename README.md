# Learn-a-World

**A JEPA-style latent world model that learns by playing.**

An agent plays simulated worlds (a custom maze → Push-T → the game Crafter). Every step it takes is stored in a capped replay buffer. A world model (encoder + action-conditioned latent predictor, custom-trained in PyTorch — no LLM APIs anywhere) learns the world's rules from that self-generated experience. A planner (MPC) dreams candidate action sequences through the model and acts on the best one — so play improves, generating richer experience: **Play → Remember → Learn → Imagine → Play better**.

Headline experiment: when the world's dynamics change at test time, the model keeps learning from its own prediction errors (test-time adaptation) and recovers — while a frozen model and a PPO baseline degrade.

---

## Status & Build Phases

- [x] **Phase 0 — Toolchain & Environment Setup** (PyTorch MPS/CUDA, Crafter, SB3, Project Scaffolding)
- [ ] **Phase 1 — Custom Maze + Self-Play Loop v1** (Buffer + JEPA Encoder/Predictor + First Prediction Errors)
- [ ] **Phase 2 — Planner + PPO Baseline** (MPC/CEM Planner + Curiosity Module + Comparative Results Table)
- [ ] **Phase 3 — Push-T + Crafter Full Loop** (Crafter Achievement Score vs Baselines on Held-Out Seeds)
- [ ] **Phase 4 — Dynamics Change & Test-Time Adaptation** (Headline Table: Frozen vs Adaptive vs PPO)
- [ ] **Phase 5 — Release** (Hugging Face Checkpoints + Model Card + Gradio Live Demo + 1-Min Video)

Full design blueprint: [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md)  
Architecture and reference notes: [`docs/REFERENCE_STUDY.md`](docs/REFERENCE_STUDY.md)

---

## Tech Stack
- **Frameworks:** Python 3.11 · PyTorch 2.x (Apple Silicon MPS / NVIDIA CUDA) · Gymnasium 1.x
- **Environments:** Custom Grid Maze · Push-T · Crafter (Survival Benchmark)
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

### 2. Run Phase 0 Verification
```bash
python scripts/verify_phase0.py
pytest -v tests/test_phase0.py
```

### 3. Colab & Drive Testing
Open `notebooks/00_colab_drive_test.ipynb` on Google Colab with GPU runtime to verify Drive mounting and compressed `.npz` replay chunk persistence.
