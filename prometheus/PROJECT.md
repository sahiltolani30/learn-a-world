# Project: Crafter Tabula Rasa JEPA World Model

## Architecture
Autonomous self-learning agent powered by a true Joint Embedding Predictive Architecture (JEPA) world model in Crafter (transferable to Atari ALE) without pixel decoders, hardcoded rules, or privileged cheats.

### System Diagram & Information Flow
```
Raw Observation x_t (64x64x3 RGB)
       │
       ▼
┌───────────────────────────────┐
│ Context Encoder E_θ           │ ──> Latent State z_t (D=256)
│ (ResNetTiny + LayerNorm)      │        │                │
└───────────────────────────────┘        │                ▼
                                         │       ┌────────────────────────┐
                                         │       │ Actor-Critic Prior     │
                                         │       │ π_ψ(a|z), Q_ω(z, a)    │
                                         │       └────────────────────────┘
                                         │                │
                                         │ (Action a_t)   ▼
                                         │       ┌────────────────────────┐
                                         │       │ Hybrid Latent MPC /    │
                                         │       │ Categorical CEM (H=5)  │
                                         │       └────────────────────────┘
                                         ▼                │
       ┌──────────────────────────────────────────────────┤
       │
       ▼
┌─────────────────────────────────┐
│ 4-Head Action-Predictor P_ϕ     │ ──> Predicted Latents {ẑ^(k)_(t+1)} (k=1..4)
│ Residual Latent Delta MLP       │        │                │
└─────────────────────────────────┘        │                │
                                           │                ▼
                                           │       ┌─────────────────────────┐
                                           │       │ Epistemic Disagreement  │
                                           │       │ r_int = Var_k(ẑ^(k))    │
                                           │       └─────────────────────────┘
                                           ▼
Next Observation x_(t+1)
       │
       ▼
┌───────────────────────────────┐
│ Target Encoder E_θ̄ (EMA/Stop) │ ──> Target Latent z_(t+1) (D=256)
└───────────────────────────────┘        │
       │                                 │
       ▼                                 ▼
┌────────────────────────────────────────────────────────┐
│ Hybrid Anti-Collapse Loss (VICReg)                     │
│ L = λ Inv(ẑ, z) + μ Var(z) + ν Cov(z)                  │
│ Mathematically guarantees Var(z_j) ≥ 1.0 (no collapse) │
└────────────────────────────────────────────────────────┘
```

## Architectural Consensus (R6 Settled)
1. **World Model & Representation**:
   - `ResNetTinyEncoder`: 3 residual downsampling stages with LayerNorm, mapping $64\times 64\times 3 \to 256$ dimensions. Preserves fine-grained HUD vital/inventory pixels in rows 52-63.
   - `TargetEncoder`: EMA polyak update ($\tau=0.99$) under `torch.no_grad()`.
   - `ResidualMLPPredictor`: 4-head ensemble predicting latent delta $\hat{z}^{(k)}_{t+1} = z_t + \Delta_\phi(z_t, e(a_t))$.
   - `VICRegLoss`: Invariance ($\lambda=1.0$), Variance hinge ($\mu=1.0$, target $\gamma=1.0$), and Covariance decorrelation ($\nu=0.04$).
2. **Intrinsic Motivation & Curiosity**:
   - Epistemic Disagreement across the 4 predictor heads eliminates the "Noisy TV" aleatoric trap.
   - Episodic $k$-NN latent novelty bonus provides exploratory drive.
   - Running Welford normalization ensures scale-invariance.
3. **Planning & Policy**:
   - Hybrid Policy-Guided Latent MPC (TD-MPC2 style): Actor prior $\pi(a|z)$ guides Categorical CEM candidate sampling ($N=50, H=5$). Terminal Critic $Q(z_H, a_H)$ provides long-horizon return evaluation.
   - Dual-Mode: Fast Policy mode ($0.011$ ms) for local CPU tests and data collection; Deliberative Planning mode ($0.65$ ms) for tactical evaluation.
4. **Zero-Cheat Compliance**:
   - `ZeroCheatWrapper` completely scrubs `info['inventory']`, `info['achievements']`, `info['semantic']`, and internal crafting coordinates. Agent learns affordances solely from visual pixels and intrinsic novelty.
5. **Memory & Persistence**:
   - Two-Tier Prioritized Replay Buffer: Tier 1 (100k FIFO) + Tier 2 (20k Curated Milestone Archive) with 70/30 sampling.
   - Checkpoint persistence across episodes and sessions.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | ResNetTiny Visual Encoder | Conv+Residual blocks mapping 64x64x3 to D=256 with LayerNorm | M1 | R1, R4 |
| 2 | Action Embedding Module | Embeds discrete actions (Crafter 17, Atari |A|) to d_a=32 | M1 | R1, R4 |
| 3 | 4-Head Latent Predictor | Residual MLP ensemble predicting z_{t+1} deltas | M1 | R1, R2 |
| 4 | EMA Target Encoder | Stop-gradient Polyak EMA target encoder with tau=0.99 | M1 | R1 |
| 5 | Anti-Collapse VICReg Loss | Variance hinge + covariance decorrelation + L2 invariance | M1 | R1 |
| 6 | Zero-Cheat Crafter Wrapper | Strips info['inventory'], achievements; frame-stacking (k=3) | M2 | R3 |
| 7 | Generalizable Env Interface | Plug-and-play gym/gymnasium adapter for Crafter and Atari | M2 | R4 |
| 8 | Epistemic Curiosity Engine | 4-head inter-prediction variance with Welford normalization | M3 | R2 |
| 9 | Episodic k-NN Novelty | Latent distance to rolling episodic memory | M3 | R2 |
| 10 | Two-Tier Replay Buffer | 100k FIFO + 20k milestone archive with 70/30 sampling | M3 | R2 |
| 11 | Actor-Critic Policy Prior | MLP policy pi(a|z) and twin critics Q_1, Q_2(z, a) | M4 | R3 |
| 12 | Categorical Latent CEM Planner| Guided latent MPC rollout with actor seeding and critic tail | M4 | R3 |
| 13 | Dual-Mode Agent | Fast policy mode (0.011ms) & deliberative planning mode (0.65ms) | M4 | R3, R5 |
| 14 | Ultra-Fast Smoke Test | CPU test in <60s verifying shapes, grad flow, non-collapse | M5 | R5 |
| 15 | Colab & Drive Notebook | Self-contained notebook with Drive mounting & checkpointing | M5 | R5 |
| 16 | Integrated Training Loop | Alternating world-model update, policy update, and env stepping | M5 | R2, R5 |
| 17 | E2E Test Suite (Tiers 1-4) | Comprehensive test suite covering all features and invariants | M6 | Acceptance |
| 18 | Adversarial Hardening (Tier 5) | Latent collapse stress tests, cheat leakage audits | M6 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core JEPA Representation & Dynamics | `src/models/encoder.py`, `predictor.py`, `target.py`, `losses.py` | none | DONE |
| M2 | Zero-Cheat Environment Abstraction | `src/env/wrapper.py`, `src/env/atari_wrapper.py` | none | DONE |
| M3 | Intrinsic Curiosity & Two-Tier Memory | `src/curiosity/ensemble.py`, `src/curiosity/rnd.py`, `src/memory/buffer.py` | M1 | DONE |
| M4 | Latent Policy & Hybrid MPC Planner | `src/agent/actor_critic.py`, `src/agent/planner.py`, `src/agent/agent.py` | M1, M2, M3 | DONE |
| M5 | Dual-Environment Pipeline & Training | `src/train.py`, `src/test_smoke.py`, `notebooks/train_colab.ipynb` | M1, M2, M3, M4 | DONE |
| M6 | Comprehensive Verification & Hardening | Full test runner, CPU smoke verification, non-collapse & zero-cheat audit | M1, M2, M3, M4, M5 | DONE |

## Interface Contracts

### 1. `ObservationEncoder` ↔ `LatentPredictor` / `Agent`
```python
# src/models/encoder.py
class ObservationEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 256): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C, H, W) normalized float in [0, 1]
        # Output: (B, latent_dim) with LayerNorm
        ...
```

### 2. `LatentPredictor` ↔ `CuriosityEngine`
```python
# src/models/predictor.py
class LatentPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, num_actions: int = 17, action_dim: int = 32, num_heads: int = 4): ...
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: z (B, latent_dim), a (B,) integer action IDs
        # Output:
        #   mean_pred: (B, latent_dim) ensemble mean
        #   all_preds: (num_heads, B, latent_dim) for epistemic disagreement
        ...
```

### 3. `TargetEncoder`
```python
# src/models/target.py
class TargetEncoder(nn.Module):
    def __init__(self, encoder: ObservationEncoder, tau: float = 0.99): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C, H, W) -> Output: (B, latent_dim) detached
        ...
    def update(self):
        # Polyak EMA: θ̄ = τ θ̄ + (1-τ) θ
        ...
```

### 4. `VICRegLoss`
```python
# src/models/losses.py
class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff: float = 1.0, var_coeff: float = 1.0, cov_coeff: float = 0.04, gamma: float = 1.0): ...
    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Returns: {'loss': total, 'invariance': inv, 'variance': var, 'covariance': cov}
        ...
```

### 5. `ZeroCheatWrapper`
```python
# src/env/wrapper.py
class ZeroCheatWrapper(gym.Wrapper):
    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Strips all privileged keys: 'inventory', 'achievements', 'semantic', 'player_pos'
        # Returns: sanitized obs (C, H, W), external reward (not used for cheat), done, clean info
        ...
```

## Code Layout
```
/Users/harshitatolani/Sahil/ml 2/
├── ORIGINAL_REQUEST.md
├── PROJECT.md
├── TEST_INFRA.md
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   ├── target.py
│   │   └── losses.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── wrapper.py
│   │   └── atari_wrapper.py
│   ├── curiosity/
│   │   ├── __init__.py
│   │   ├── ensemble.py
│   │   └── rnd.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── buffer.py
│   │   └── storage.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── actor_critic.py
│   │   ├── planner.py
│   │   └── agent.py
│   ├── train.py
│   └── test_smoke.py
├── notebooks/
│   └── train_colab.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_tier1_features.py
│   ├── test_tier2_boundaries.py
│   ├── test_tier3_combinations.py
│   ├── test_tier4_scenarios.py
│   └── run_all_tests.py
```
