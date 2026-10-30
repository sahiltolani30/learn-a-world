# Crafter Tabula Rasa JEPA World Model

An autonomous, self-learning AI agent powered by a **true Joint Embedding Predictive Architecture (JEPA)** that learns physics, affordances, and environment dynamics in [Crafter](https://github.com/danijar/crafter) (and generalizable to Atari) purely from raw pixels without decoders, hardcoded rules, or cheats.

---

## 🌟 Key Architectural Features

1. **True JEPA World Model (Pixel-Free Latent Dynamics)**:
   * **Visual Context Encoder** $E_\theta(x_t) \to z_t \in \mathbb{R}^{256}$: ResNetTiny with LayerNorm mapping 64×64×3 raw observations to latent representations.
   * **Action-Conditioned Latent Predictor** $P_\phi(z_t, a_t) \to \hat{z}_{t+1}$: 4-head ensemble residual MLP forecasting future latent states without pixel reconstruction.
   * **Polyak EMA Target Encoder** $E_{\bar{\theta}}(x_{t+1}) \to z_{t+1}$: Updated via Exponential Moving Average ($\tau=0.99$) with strict stop-gradient isolation.
   * **VICReg Anti-Collapse Regularization**: Combined Invariance ($L_2$), Variance Hinge ($\text{Var}(z_j) > 1$), and Covariance Decorrelation losses to mathematically guarantee non-collapse without negative pairs.

2. **Zero-Cheat Tabula Rasa Learning**:
   * Agent begins with **zero prior domain knowledge** (no hardcoded recipes, no crafting trees, no privileged state access).
   * Strict `CrafterWrapper` strips all internal state (`inventory`, `player_pos`, `semantic`, internal recipes) from observations and info dicts.
   * Learns causal mechanics (e.g. chopping trees yields wood; combining wood at table unlocks tools) purely from sensorimotor visual interaction.

3. **Epistemic Curiosity & Discovery**:
   * **4-Head Predictor Disagreement**: Measures genuine epistemic uncertainty (what the model hasn't mastered yet) rather than aleatoric noise, completely avoiding the classic "noisy TV" trap.
   * **Welford RunningMeanStd**: Online running mean/std normalizer keeping exploration rewards stably calibrated.

4. **Two-Tier Persistent Memory Buffer**:
   * **Tier 1 (70% Rolling FIFO)**: Tracks recent interactions for active on-policy updates.
   * **Tier 2 (30% Curated Milestone Archive)**: Retains rare, high-curiosity breakthroughs (e.g. first wood harvested, crafting table built).
   * **Cross-Episode Persistence**: Automatically serializes replay buffer and model weights so past learnings carry over into subsequent games.

5. **Hybrid Planning & Execution**:
   * **Fast Policy Reflex Mode**: Direct Actor inference (~0.01 ms CPU latency).
   * **Deliberative Latent Planning Mode**: Cross-Entropy Method (CEM) trajectory optimization rolling out candidates purely in latent space evaluated by TwinCritic (~0.65 ms CPU latency).

---

## 🚀 Quickstart & Smoke Testing

### 1. Run the Local CPU Smoke Test (< 1 Second)
Verify the entire 13-stage pipeline (Crafter environment, zero-cheat sanitization, encoder, predictor ensemble, curiosity, VICReg, autograd backward pass, non-collapse metrics) on CPU:

```bash
python3 -m src.test_smoke
```

Expected output:
```
================================================================================
 C R A F T E R   J E P A   C P U   S M O K E   T E S T
================================================================================
 [PASS] Stage 1: Environment initialized with seed 42
 [PASS] Stage 2: Zero privileged cheat state confirmed - 0 leaks
 [PASS] Stage 3: Dual-Mode Stepping (Fast & CEM Planning)
 [PASS] Stage 4: Visual Latent Encoding (3, 64, 64) -> R^256
 [PASS] Stage 5: 4-Head Predictor Ensemble Forward Pass
 [PASS] Stage 6: Epistemic Curiosity Engine
 [PASS] Stage 7: Polyak EMA Target Encoder update verified
 [PASS] Stage 8: Two-Tier Buffer Mixed Sampling (70% T1, 30% T2)
 [PASS] Stage 9: VICReg Loss (Inv, Var, Cov)
 [PASS] Stage 10: Backpropagation & Non-Zero Gradients Verified
 [PASS] Stage 11: Non-Collapse Verified (Var(z) > 0.1 on all dims)
 [PASS] Stage 12: Execution Time Budget (<0.8s on CPU)
================================================================================
 * Status: ALL 13 STAGES PASSED [PASS]
```

### 2. Run the Full Test Suite (171 Tests)
```bash
python3 -m unittest discover tests
```

---

## ☁️ Google Colab & Google Drive Training Pipeline

The repository includes a ready-to-run Jupyter notebook configured for Google Colab with GPU acceleration and persistent storage on Google Drive:
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)

### Workflow:
1. **Push your code to a public GitHub repo**:
   ```bash
   git remote add origin https://github.com/<YOUR_USERNAME>/crafter-jepa.git
   git branch -M main
   git push -u origin main
   ```
2. **Open in Google Colab**:
   * Upload `notebooks/train_colab.ipynb` to Google Colab.
   * Enable GPU runtime (`Runtime -> Change runtime type -> T4 GPU`).
3. **Mount Google Drive**:
   * Cell 2 automatically mounts `/content/drive/MyDrive/crafter_jepa/checkpoints`.
   * Checkpoints and training metrics persist directly to your Google Drive across session disconnects or preemptions.
4. **Train**:
   * Cell 6 launches full GPU training with live Crafter achievement tracking.
   * Cell 7 plots world model loss curves, VICReg variance dynamics, and Crafter milestone unlock curves.

---

## 📁 Project Structure

```
.
├── notebooks/
│   └── train_colab.ipynb       # End-to-end Colab notebook with Drive sync
├── src/
│   ├── models/
│   │   ├── encoder.py          # ResNetTiny visual encoder (x -> z)
│   │   ├── predictor.py        # 4-head action-conditioned latent predictor
│   │   ├── target.py           # Polyak EMA target encoder with stop-gradient
│   │   └── losses.py           # VICReg loss engine & variance metrics
│   ├── env/
│   │   ├── wrapper.py          # Zero-cheat CrafterWrapper & FrameStack
│   │   └── atari_wrapper.py    # Atari ALE wrapper (generalization interface)
│   ├── curiosity/
│   │   ├── ensemble.py         # Epistemic curiosity engine & Welford normalizer
│   │   └── rnd.py              # Latent Random Network Distillation novelty
│   ├── memory/
│   │   ├── buffer.py           # Two-tier replay buffer (70% FIFO, 30% Curated)
│   │   └── storage.py          # CheckpointManager with disk persistence
│   ├── agent/
│   │   ├── actor_critic.py     # ActorPrior and TwinCritic networks
│   │   ├── planner.py          # Categorical CEM latent trajectory optimizer
│   │   └── agent.py            # Unified JEPAAgent (Reflex + Deliberative)
│   ├── test_smoke.py           # Ultra-fast CPU smoke test script (<1s)
│   └── train.py                # Complete training loop & evaluation engine
├── tests/                      # 171 comprehensive unit and adversarial tests
└── README.md
```
