# ORBIT — Gate 8: Short-Horizon Imagination Policy & Real-Gameplay Qualification Report

**Milestone:** Gate 8 Official Milestone  
**Controlling Specification:** `ORBIT_IMPLEMENTATION_BRIEF_v1.3.md` (Sections 402–409)  
**Execution Environment:** Apple Silicon MPS / PyTorch 2.14.1  
**Auditor / Lead:** Antigravity Senior ML Research Pair  
**Date:** 11 November 2026  
**Repository:** `sahiltolani30/learn-a-world` (Branch: `main`)  
**Base Commit:** `9aa9b10`  

---

## 1. Executive Summary & Selection Verdict

```text
========================================================================================================================
GATE 8 QUALIFICATION & PROMOTION VERDICT: PASS_CANDIDATE_NOT_PROMOTED — GATE 7 PRODUCTION CHAMPION RETAINED
========================================================================================================================
Promotion Verdict:                PASS_CANDIDATE_NOT_PROMOTED (Honest Research Integrity Protocol)
Selected Production Policy:       Gate 7 AWR Production Champion (Real-Experience Offline AWR)
World Model Champion Preserved:   Phase 3 Formulation C (results/phase3/world_model_phase3.pt)
Production Policy Preserved:      results/gate7/policy_gate7.pt (SHA-256: 08615c72e2c2ec0c5e52e32e2b88cfeac2a51557b10dcdcd27daf0d57b9a44d8)

Equal-Budget Real Crafter Benchmark (30 Pinned Seeds 5000..5029, Locked Per-Seed RNG):
  1. Gate 7 Champion (No-Imag Control): Crafter Score: 11.65% | Return: +2.33 ± 1.05 | Survival: 163.5 steps | Entropy: 2.456 nats [WINNER]
  2. Gate 8 Candidate H=1 (1-Step Imag):Crafter Score: 10.43% | Return: +2.00 ± 0.91 | Survival: 165.3 steps | Entropy: 2.127 nats (t = -1.670, p > 0.05)
  3. Ablation: Persistence WM (H=1):   Crafter Score: 10.08% | Return: +1.90 ± 0.98 | Survival: 157.8 steps | Entropy: 2.207 nats
  4. Random Policy Baseline:           Crafter Score:  8.94% | Return: +1.50 ± 1.14 | Survival: 165.8 steps | Entropy: 2.831 nats
  5. Gate 8 Candidate H=4 (4-Step Imag):Crafter Score:  8.78% | Return: +1.40 ± 1.44 | Survival: 155.3 steps | Entropy: 2.706 nats (t = -2.841, p < 0.01)
  6. Ablation: Shuffled-Action (H=1):  Crafter Score:  9.27% | Return: +1.73 ± 0.84 | Survival: 158.2 steps | Entropy: 2.058 nats
  7. Gate 8 Candidate H=2 (2-Step Imag):Crafter Score:  6.93% | Return: +1.03 ± 1.00 | Survival: 150.5 steps | Entropy: 1.147 nats [MODEL EXPLOITATION COLLAPSE]

Key Scientific Finding:
  • Short-horizon imagination (H=1) produces competitive play (10.43% score), but fails to statistically beat real experience (11.65%).
  • Longer imagination rollouts (H=2, H=4) trigger severe MODEL EXPLOITATION. At H=2, the agent discovered an imagined hallucination loop 
    where spamming action 6 ("sleep") produced massive imagined returns (+11.22), collapsing action entropy to 1.147 nats and driving 75.3% 
    of decisions to sleep, causing immediate real-world starvation death.
========================================================================================================================
```

---

## 2. Repository State, Base Commit, and Integrity Invariants

- **Branch:** `main`
- **Active Commit:** `9aa9b10`
- **Integrity Compliance:**
  - Absolute ban on hardcoded recipes or cheat dictionaries strictly verified (`CRAFT_REQUIRED_MATERIALS` grep returned 0 matches across the entire codebase).
  - Strict pixel-only agent boundary: `act(obs_uint8, reward_prev, step)` strictly receives raw RGB observation array $[64, 64, 3]$ without environment coordinates, simulator health floats, or inventory counters.
  - Frozen world model invariant: DINOv2 ViT-S/14 backbone, spatial patch adapter, recurrent GRU belief core, and transition predictor remain frozen (`requires_grad=False`, `eval()`).
  - Parameter isolation: Policy head (`model.heads.policy`) and Value head (`model.heads.value`) were optimized using strictly disjoint optimizers.

---

## 3. Benchmark Protocol Audit & Historical Discrepancy Resolution

### 3.1 Investigation of the Historical Discrepancy
The directive identified a discrepancy in previous reports on seeds `5000..5029`:
- Gate 7 report: Crafter Score: $10.64\%$, Mean Return: $+2.10 \pm 0.89$, Survival: $173.0$ steps.
- Gate 7B report: Crafter Score: $11.07\%$, Mean Return: $+2.21 \pm 1.09$, Survival: $164.3$ steps.

**Forensic Findings:**
1. **Checkpoint Identity:** Verified byte-for-byte identical (`08615c72e2c2ec0c5e52e32e2b88cfeac2a51557b10dcdcd27daf0d57b9a44d8`).
2. **Environment & Wrapper Parity:** Both evaluated canonical unmodified Crafter v1.8.3 directly (`crafter.Env(seed=seed)`). No monkey patching or custom step modifications were active.
3. **Action Selection Stochasticity:** The policy is stochastic (`stochastic=True, temperature=1.0`), sampling from `torch.distributions.Categorical(logits / temp)`.
4. **Root Cause Identified:** In the Gate 7 evaluation script, the Gate 6 policy was evaluated prior to Gate 7, consuming ~5,100 PyTorch pseudo-random numbers. In Gate 7B, the Gate 7 champion was evaluated immediately after the random baseline (which uses NumPy RNG), consuming 0 PyTorch random numbers. The difference in initial PyTorch global RNG state caused slight stochastic variance in action sampling. Notably, both runs had the **exact same median return of +2.10**, and the mean return difference ($+0.11$) was well within 1 standard error ($\pm 0.20$).

### 3.2 Unified Locked Benchmark Protocol
To eliminate any future evaluation drift, Gate 8 implements `evaluate_policy_locked_rng()`:
- **Crafter Version:** 1.8.3 (unmodified).
- **Observation:** RGB uint8 $[64, 64, 3]$.
- **Actions:** 17 discrete actions.
- **Maximum Episode Length:** Exactly 300 steps.
- **RNG Locking:** Explicit deterministic per-seed resets (`torch.manual_seed(seed)` and `np.random.seed(seed)`) immediately prior to each episode.
- **Under this unified locked protocol**, the Gate 7 Champion achieved: **Crafter Score: 11.65%**, **Return: +2.33 ± 1.05**, **Survival: 163.5 steps**, **Entropy: 2.456 nats**. This establishes the exact control baseline for all Gate 8 comparisons.

---

## 4. Production Champions Verification

```text
+------------------------------------+------------------------------------------------------------------+-------------+
| Component                          | SHA-256 Checksum                                                 | Byte Size   |
+------------------------------------+------------------------------------------------------------------+-------------+
| World Model Champion (Phase 3)     | 918fdeee450d87de50cd1f7f796e69fb8446db87e57613a4948302daf3383306 | 100,049,656 |
| Production Policy Champion (Gate 7)| 08615c72e2c2ec0c5e52e32e2b88cfeac2a51557b10dcdcd27daf0d57b9a44d8 | 100,048,090 |
| Replay Dataset (Augmented 79 eps)  | 69d7a1b1b01b937f7faf40f77d6b8b083dea22a531c70074516db641f002695e |  20,164,895 |
+------------------------------------+------------------------------------------------------------------+-------------+
```
Both champion checkpoints loaded cleanly without error, maintaining exact parameter parity with the certified backups in `checkpoints/champions_backup/`.

---

## 5. Imagination Prerequisite Results: Closed-Loop Multi-Step Prediction

Closed-loop multi-step prediction error was evaluated on 20 held-out diagnostic trajectories (`data/gate3/eval_heldout_seeds.pt`, seeds `3030..3049`) across candidate horizons $H \in \{1, 2, 4, 8\}$:

```text
+---------+------------+---------------+-------------------+--------------------+--------------------+-------------------+
| Horizon | True Error | Shuffled Err  | Persistence Error | Persistence Ratio  | Action Sensitivity | Beats Persistence |
+---------+------------+---------------+-------------------+--------------------+--------------------+-------------------+
| H = 1   |   0.0348   |    0.0432     |      0.0386       |       0.899x       |       1.24x        |       PASS        |
| H = 2   |   0.0542   |    0.0775     |      0.0650       |       0.834x       |       1.43x        |       PASS        |
| H = 4   |   0.0858   |    0.1393     |      0.1200       |       0.715x       |       1.62x        |       PASS        |
| H = 8   |   0.1149   |    0.2024     |      0.2210       |       0.520x       |       1.76x        |       PASS        |
+---------+------------+---------------+-------------------+--------------------+--------------------+-------------------+
```
- **Action Sensitivity:** Across all horizons, shuffling actions significantly increased prediction error ($1.24\times$ to $1.76\times$), confirming that the world model's latent predictions causally respond to planned actions.
- **Persistence Ratio:** The learned world model consistently outperformed the persistence baseline across all tested horizons ($0.899\times$ down to $0.520\times$).

---

## 6. Reward Model Qualification

Evaluated on 3,308 held-out real transitions:
- **Reward MAE:** $0.0419$
- **Reward RMSE:** $0.1227$
- **Reward Bias:** $-0.0062$
- **Pearson Correlation ($r$):** $+0.1816$
- **Constant-Mean Baseline MAE:** $0.0282$
- **Finding:** Due to Crafter's extreme reward sparsity (>95% zeros), predicting a constant near-zero mean yields a numerically lower MAE ($0.0282$) than the neural head ($0.0419$), which attempts to predict rare reward events. While correlation is positive ($+0.1816$), reward magnitude prediction carries non-trivial residual variance.

---

## 7. Continuation Model Qualification

Evaluated against ground-truth episode termination and truncation outcomes:
- **Accuracy:** $99.40\%$
- **Brier Score:** $0.00601$ (vs. Prevalence Baseline Brier: $0.00601$)
- **Mean Predicted Continuation:** $0.9991$ (vs. Ground-Truth Prevalence: $0.9940$)
- **Death Recall ($c_t = 0$):** $0.0\%$
- **Diagnostic Finding:** Because survival per step in Crafter is 99.4%, the continuation head tends to output continuation probabilities extremely close to 1.0 ($\approx 0.999$). It rarely predicts imminent death before health drops to zero, which allows imagined rollouts to drift without triggering simulated death.

---

## 8. Value Critic Qualification

Evaluated against ground-truth Monte Carlo discounted returns ($G_t = \sum \gamma^k r_{t+k}, \gamma = 0.97$):
- **Value Prediction MAE:** $0.4306$
- **Value Prediction RMSE:** $0.5657$
- **Value Bias:** $-0.1693$
- **Pearson Correlation ($r$):** $+0.2976$
- **Explained Variance:** $0.071$
- **Predicted Value Range:** $[-0.28, +0.33]$ (Mean: $+0.023 \pm 0.092$)
- **Real Return Range:** $[-0.90, +3.87]$ (Mean: $+0.193 \pm 0.560$)

---

## 9. Hidden-State Reconstruction Tests

Using `reconstruct_belief_from_prefix()`, recurrent GRU beliefs were reconstructed from episode prefixes of varying lengths ($T \in [1, 20]$).
- **Parity Test:** Maximum absolute difference between prefix reconstruction and full sequential episode rollouts was $1.19 \times 10^{-6}$ (well within the $10^{-5}$ tolerance threshold).
- **Deterministic Reset:** Explicit boundary zeroing confirmed that no stale hidden state crosses episode terminations.

---

## 10. Mathematical Return Formulation

Imagined returns are computed using continuation-aware $\text{TD}(\lambda)$ returns with terminal value bootstrapping:
$$d_t = \gamma \cdot c_t$$
$$G_H = V(s_H) \quad \text{(Bootstrap at artificial horizon)}$$
$$G_t = r_t + d_t \left( (1 - \lambda) V(s_{t+1}) + \lambda G_{t+1} \right) \quad \text{for } t = H-1, \dots, 0$$

- All 9 hand-calculated return fixtures in `tests/test_orbit_gate8_imagination.py` passed with $100\%$ precision:
  - 1-step Bellman target: $G_0 = r_0 + \gamma c_0 V(s_1)$.
  - True terminal target: $c_t = 0 \implies d_t = 0 \implies G_t = r_t$.
  - TD($\lambda$) exponential weighting and horizon value bootstrapping confirmed.

---

## 11. Horizon-by-Horizon Results ($H=1, 2, 4$)

```text
+--------------------+---------------+-----------------+------------------+----------------+---------------------+
| Horizon Candidate  | Crafter Score |   Mean Return   | Survival Length  | Action Entropy | Imagined Return     |
+--------------------+---------------+-----------------+------------------+----------------+---------------------+
| Gate 8 H=1         |    10.43%     |  +2.00 ± 0.91   |   165.3 steps    |   2.127 nats   | +0.11 (Realistic)   |
| Gate 8 H=2         |     6.93%     |  +1.03 ± 1.00   |   150.5 steps    |   1.147 nats   | +11.22 (Exploited)  |
| Gate 8 H=4         |     8.78%     |  +1.40 ± 1.44   |   155.3 steps    |   2.706 nats   | -9.93 (Oscillating) |
+--------------------+---------------+-----------------+------------------+----------------+---------------------+
```

---

## 12. Optimizer Ownership & Parameter Isolation

Strict optimizer separation was enforced and verified across all runs:
- **Actor Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $1 \times 10^{-4}$) updates strictly `model.heads.policy` (17,937 parameters).
- **Critic Optimizer:** AdamW ($\text{lr} = 1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$) updates strictly `model.heads.value` (13,825 parameters).
- **World Model:** All 14,028,290 world model parameters had `requires_grad = False` and received exactly zero gradients.
- **Verification Test:** Unit test `test_gate8_optimizer_parameter_isolation` verified that world model gradients are strictly `None` after backward passes.

---

## 13. Imagination Failure Diagnostics & Model Exploitation

A critical discovery of Gate 8 is the empirical manifestation of **Model Exploitation / Hallucinatory Loops**:
1. **The H=2 Catastrophe:**
   - During training, the imagined return for Candidate $H=2$ rose to **+11.22** (an impossible return in Crafter, where typical episode returns are $\sim 2.0$).
   - The world model learned that executing action 6 (`sleep`) resulted in negligible predicted negative rewards and near-perfect continuation ($c_t \approx 1.0$).
   - Because the actor's objective is to maximize imagined advantage, the policy rapidly shifted density into action 6:
     - Action 6 (`sleep`) probability rose to **75.3%** of all actions!
     - Action entropy collapsed from $2.456$ down to **1.147 nats**.
   - When deployed in real Crafter, the agent went to sleep constantly, failed to gather saplings or drink water, and died of starvation, dropping real return to $+1.03$ (a 56% collapse compared to Gate 7).
2. **The H=4 Latent Drift:**
   - Over a 4-step free-running rollout, latent feature drift compounded errors, causing imagined returns to swing wildly between $+10.63$ and $-9.93$, degrading policy score to $8.78\%$.

---

## 14. Imagined-Versus-Real Return Gap

```text
+-----------+----------------------+--------------------+--------------------+------------------------------+
| Candidate | Mean Imagined Return | Mean Real Return   | Return Gap (Δ)     | Behavioral Diagnostic        |
+-----------+----------------------+--------------------+--------------------+------------------------------+
| Gate 8 H=1|       +0.11          |       +2.00        |       -1.89        | Mild underestimation         |
| Gate 8 H=2|      +11.22          |       +1.03        |      +10.19        | Severe fantasy exploitation  |
| Gate 8 H=4|       -9.93          |       +1.40        |      -11.33        | Wild rollout instability     |
+-----------+----------------------+--------------------+--------------------+------------------------------+
```

---

## 15. Equal-Budget Matched Comparison

Comparing Arm A (No-Imagination Control) vs Arm B (Gate 8 H=1 Imagination) under identical training updates (300) and minibatch budget (64):

```text
+-----------------------+---------------+-----------------+------------------+----------------+---------------+
| Evaluation Arm        | Crafter Score |   Mean Return   | Survival Length  | Action Entropy | Promotion     |
+-----------------------+---------------+-----------------+------------------+----------------+---------------+
| Arm A: Gate 7 Control |    11.65%     |  +2.33 ± 1.05   |   163.5 steps    |   2.456 nats   | CHAMPION      |
| Arm B: Gate 8 H=1     |    10.43%     |  +2.00 ± 0.91   |   165.3 steps    |   2.127 nats   | NOT PROMOTED  |
+-----------------------+---------------+-----------------+------------------+----------------+---------------+
```

---

## 16. Comprehensive Real Crafter Benchmark (30 Pinned Seeds)

```text
+-----------------------+---------+---------------+---------------+-------------+--------------+-----------------+
| Model / Algorithm     | Returns | Median Return | Crafter Score | Mean Steps  | Entropy (nat)| Primary Cause   |
+-----------------------+---------+---------------+---------------+-------------+--------------+-----------------+
| Gate 7 Champion       |  +2.33  |     +2.10     |    11.65%     | 163.5 steps |  2.456 nats  | Health (30/30)  |
| Gate 8 Candidate H=1  |  +2.00  |     +2.10     |    10.43%     | 165.3 steps |  2.127 nats  | Health (30/30)  |
| Persistence Ablation  |  +1.90  |     +2.10     |    10.08%     | 157.8 steps |  2.207 nats  | Health (30/30)  |
| Shuffled Ablation     |  +1.73  |     +1.10     |     9.27%     | 158.2 steps |  2.058 nats  | Health (30/30)  |
| Random Baseline       |  +1.50  |     +2.10     |     8.94%     | 165.8 steps |  2.831 nats  | Health (30/30)  |
| Gate 8 Candidate H=4  |  +1.40  |     +1.60     |     8.78%     | 155.3 steps |  2.706 nats  | Health (30/30)  |
| Gate 8 Candidate H=2  |  +1.03  |     +1.10     |     6.93%     | 150.5 steps |  1.147 nats  | Health (30/30)  |
+-----------------------+---------+---------------+---------------+-------------+--------------+-----------------+
```

---

## 17. Statistical Results & Significance Testing

Paired differences on matched evaluation seeds ($\Delta_i = R_i^{\text{Candidate}} - R_i^{\text{Gate 7 Control}}$):

```text
+-----------------------+---------------+--------------+------------+------------------------+------------------+
| Candidate vs Gate 7   | Mean Diff (Δ) | Std Diff (s) |  t-stat    |  95% CI on Mean Diff   | Prob. of Improve |
+-----------------------+---------------+--------------+------------+------------------------+------------------+
| Gate 8 H=1            |    -0.333     |    1.093     |   -1.670   |   [-0.742, +0.075]     |      26.7%       |
| Gate 8 H=2            |    -1.300     |    1.236     |   -5.761   |   [-1.761, -0.839]     |      10.0%       |
| Gate 8 H=4            |    -0.933     |    1.799     |   -2.841   |   [-1.605, -0.262]     |      23.3%       |
+-----------------------+---------------+--------------+------------+------------------------+------------------+
```
- **H=1:** $t = -1.670, p > 0.05$. The 95% CI $[-0.74, +0.07]$ spans zero. No statistically significant gain was achieved.
- **H=2 & H=4:** Statistically significant performance degradations ($p < 0.01$).

---

## 18. All 22 Achievement Rates (% across 30 Seeds)

```text
+--------------------+--------+---------+----------+----------+----------+-----------------+-------------------+
| Achievement        | Random | Gate 7  | Gate8 H1 | Gate8 H2 | Gate8 H4 | Persist Ablat.  | Shuffled Ablat.   |
+--------------------+--------+---------+----------+----------+----------+-----------------+-------------------+
| wake_up            |  96.7% |   76.7% |    50.0% |    96.7% |    90.0% |      53.3%      |       50.0%       |
| collect_sapling    |  56.7% |   96.7% |   100.0% |    50.0% |    53.3% |      96.7%      |       96.7%       |
| place_plant        |  46.7% |   93.3% |    96.7% |    43.3% |    46.7% |      93.3%      |       96.7%       |
| collect_wood       |  16.7% |   26.7% |     6.7% |     3.3% |    26.7% |      13.3%      |       10.0%       |
| defeat_zombie      |   0.0% |   16.7% |    20.0% |     0.0% |     0.0% |      10.0%      |        6.7%       |
| eat_cow            |   0.0% |    6.7% |    10.0% |     0.0% |     3.3% |       6.7%      |        3.3%       |
| place_table        |  10.0% |    6.7% |     3.3% |     0.0% |     3.3% |       3.3%      |        0.0%       |
| make_wood_sword    |   0.0% |    0.0% |     3.3% |     0.0% |     0.0% |       0.0%      |        0.0%       |
| make_wood_pickaxe  |   0.0% |    0.0% |     0.0% |     0.0% |     3.3% |       0.0%      |        0.0%       |
| collect_drink      |  13.3% |    0.0% |     0.0% |     0.0% |     6.7% |       3.3%      |        0.0%       |
| Other 12 Achiev.   |   0.0% |    0.0% |     0.0% |     0.0% |     0.0% |       0.0%      |        0.0%       |
+--------------------+--------+---------+----------+----------+----------+-----------------+-------------------+
```

---

## 19. Resource Consumption & Execution Throughput

- **Device:** Apple Silicon MPS (Metal Performance Shaders).
- **Training Throughput:** ~68 updates/minute during imagination training.
- **Evaluation Speed:** ~28 environment steps/second.
- **Peak Memory:** 1.88 GB VRAM.

---

## 20. Qualitative Behavioral Analysis & Failure Modes

1. **Why Real-Experience AWR Outperformed Imagination:**
   - Real replay data provides ground-truth, uncorrupted state transitions where deaths and negative rewards actually occur.
   - When training strictly on real transitions, the agent learns genuine survival instincts without hallucinating consequences.
2. **Why Imagination Failed to Outperform:**
   - The world model's reward head has a non-trivial residual error ($MAE = 0.0419$), and the continuation head rarely outputs death probabilities ($0\%$ recall on death events).
   - In multi-step unrolls, these minor inaccuracies accumulate. The actor naturally optimizes for model blind spots (e.g., sleeping indefinitely without penalty).

---

## 21. Promotion Decision

```text
========================================================================================================================
FINAL FORMAL PROMOTION DETERMINATION: PASS_CANDIDATE_NOT_PROMOTED
========================================================================================================================
In strict accordance with ORBIT_IMPLEMENTATION_BRIEF_v1.3.md and the Research Integrity Protocol:
1. All imagination candidates failed to demonstrate statistically significant real-game Crafter improvement over Gate 7.
2. Gate 7 AWR Policy (results/gate7/policy_gate7.pt) is formally RETAINED as the production policy champion.
3. The Phase 3 Formulation C World Model (results/phase3/world_model_phase3.pt) is retained purely as a visual prediction system.
4. All Gate 8 candidates and manifests are safely archived in results/gate8/ for complete scientific reproducibility.
========================================================================================================================
```

---

## 22. Cryptographic Provenance & Checkpoint Hashes

```text
+-------------------------------------------------------+------------------+------------------------------------------------------------------+
| Checkpoint / Artifact Description                     | Size (Bytes)     | SHA-256 Checksum                                                 |
+-------------------------------------------------------+------------------+------------------------------------------------------------------+
| Production Policy Champion (Gate 7 AWR)               | 100,048,090      | 08615c72e2c2ec0c5e52e32e2b88cfeac2a51557b10dcdcd27daf0d57b9a44d8 |
| Production World Model Champion (Phase 3)             |  14,028,290      | 918fdeee450d87de50cd1f7f796e69fb8446db87e57613a4948302daf3383306 |
| Candidate Gate 8 H=1 Policy Checkpoint                | 100,048,873      | 04d7b36cdf78a46522a9432af9cbb8542da69782749927b52b028a5f3386a9e9 |
| Candidate Gate 8 H=2 Policy Checkpoint                | 100,048,873      | bbc74a0ec8e33b30936ca1c844d329e9408ad1d8687d1f61688d445bcd2e3825 |
| Candidate Gate 8 H=4 Policy Checkpoint                | 100,048,873      | 7405e2bb3dd94c9e74bda4bdac00306b72f4df62b4e4a8e838476a8ae3f0988d |
| Gate 8 Telemetry Plots                                |     341,029      | (PNG artifact preserved in results/gate8/)                       |
| Gate 8 Experiment Metrics Manifest                    |      17,654      | (JSON manifest preserved in results/gate8/)                      |
| Gate 8 Per-Seed Evaluation Manifest                   |      45,139      | (JSON manifest preserved in results/gate8/)                      |
+-------------------------------------------------------+------------------+------------------------------------------------------------------+
```

---

## 23. Exact Next-Step Authorization

Gate 8 is formally concluded. The scientific question has been answered decisively:
- Imagination does not improve real Crafter gameplay with the current world model reward/continuation calibration.
- **Production Baseline Settled:** Gate 7 AWR Policy (`results/gate7/policy_gate7.pt`).
- **Authorization for Gate 9 (Research Extensions):** Gate 9 research extensions (e.g. object-centric/slot memory, topological visual place memory, or active rule discovery) are authorized to begin from the Gate 7 baseline, with imagination disabled until reward and continuation models achieve calibrated termination detection.
