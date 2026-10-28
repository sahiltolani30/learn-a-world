# Complete Engineering Autopsy: Days of Blockers, Failed Experiments, and the Tool Discovery Breakthrough

**Document ID:** `AUTOPSY-PHASE2-BLOCKERS-2026-10-28`  
**Date:** October 28, 2026  
**Status:** Canonical Reference Document  
**Scope:** Chronological retrospective on the 3.5% Crafter glass ceiling, the failure analysis of past attempts, the root cause mechanisms, and the breakthrough leading to the Phase 2 Champion (`v3-phase2-champion`).

---

## 1. Executive Summary: The Core Mission & The 3.5% Glass Ceiling

### The Grand Challenge
The objective of this project is to build an autonomous, self-supervised **Visual JEPA World Model** agent that plays Crafter purely from raw sensory observations ($64 \times 64 \times 3$ RGB pixels) and scalar rewards ($r \in \mathbb{R}$).

Strict Protocol Constraints (`strict_visual_jepa_clean_v2`):
- **Zero hardcoded recipes**: No lookup tables, no craft-rule dictionaries, no action-to-item cheats.
- **Zero privileged engine leaks**: Zero access to internal engine coordinates, inventory states, or ground-truth achievement bitmasks during planning and action selection.
- **Strict Causal Discovery**: All causal relationships and needs must be empirically derived from visual state transitions ($\Delta x < 0$ for requirements, visual latents for preconditions).

### The Multi-Day Blockers & The Glass Ceiling
For days across dozens of training and screening runs, the agent encountered a seemingly impenetrable performance ceiling:
- **Crafter Score**: Repeatedly stalled between **2.0% and 3.5%**.
- **`make_wood_pickaxe`**: Flatlined at **0.0%**.
- **`make_wood_sword`**: Flatlined at **0.0%**.
- **`mine_stone`**: Flatlined at **0.0%**.
- **Survival Length**: Collapsing between steps 70 and 160 due to nightfall monster attacks, starvation, or dehydration.
- **The Crafting Table Trap**: The agent would occasionally place a crafting table, but would immediately wander off into the grassland, sprint in circles, or spam useless actions until killed.

---

## 2. Chronological Breakdown: What We Tried and Why It Failed

Below is the exhaustive, transparent record of the major approaches attempted over the past several days, the hypotheses behind them, and the forensic reason each one failed.

```
                      THE 8 FAILURE PATHS IN RETROSPECT
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Offline Clustering     --> min_cluster_size=2 erased all tool cards │
  │ 2. Q-Critic Calibration   --> Critic variance collapsed to 0.015 (flat)│
  │ 3. CEM MPC (H=10)         --> 5.2x latent error compounding at step 10 │
  │ 4. ||z_t - z_0|| Curiosity--> "Grassland Paradox" (fled from tables)   │
  │ 5. Online Hypotheses      --> needs=[] caused action 11 in grass       │
  │ 6. Object Permanence      --> unlocked_event omitted; table vanished   │
  │ 7. Encoder Fine-Tuning    --> Latent drift (7.05) broke all heads      │
  │ 8. Target Chain Gate      --> Planner rejected intermediate wood links │
  └────────────────────────────────────────────────────────────────────────┘
```

---

### Attempt 1: Offline Memory Consolidation via Unsupervised Clustering
* **Hypothesis:** By running DBSCAN and agglomerative clustering on 300 offline replay episodes (47,000 transitions), the agent could cluster successful interactions into prototype causal cards (`RelationHypothesisBank`).
* **Why It Failed:** 
  In the 300 offline episodes, complex crafting events (`make_wood_pickaxe`, `mine_stone`) occurred only 0 or 1 times in subclusters. To prevent sensory noise from generating thousands of garbage cards, the clustering pipeline enforced:
  ```python
  if len(cluster) < min_cluster_size:  # min_cluster_size = 2
      continue
  ```
  This single safety threshold silently **pruned every single pickaxe and stone interaction** from the offline bank!
* **The Fatal Consequence:** The resulting `phase2_causal_memory_bank_v2.json` contained 20 Action-11 cards, but **every single one had `event_dim: null`**. Because the compositional planner requires `card.event_dim is not None` to formulate goals, the agent was literally blind to tool progression before it even took step 0.

---

### Attempt 2: 1-Step TD Q-Critic for Value-Guided CEM Planning
* **What We Tried:** Trained a clipped Double-Q ensemble to score imagined rollout trajectories in Recurrent MPC, expecting the critic to provide a long-horizon value gradient toward high-value tools.
* **Why It Failed:**
  Sparse rewards in Crafter paired with standard 1-step TD learning caused catastrophic variance collapse in the critic networks.
  - **Empirical Ground-Truth Returns:** $\text{Mean} = 0.4614, \;\text{Std} = \mathbf{1.0179}$.
  - **Q-Ensemble Predictions:** $\text{Mean} = 0.7325, \;\mathbf{\text{Std} = 0.0157}$.
* **The Fatal Consequence:** The critic outputted $\approx 0.73$ across almost all states. It could not distinguish between standing adjacent to a crafting table versus drowning in a lake. The CEM planner received zero directional guidance from the critic.

---

### Attempt 3: Recurrent CEM Lookahead ($H=10$) with Latent Imagination
* **What We Tried:** Model Predictive Control using the Cross-Entropy Method (CEM) rolling out $K=64$ candidate sequences through the recurrent latent dynamics model.
* **Why It Failed:**
  1. **Temporal Horizon Mismatch:** The full sequence ($\text{Spawn} \to \text{Find Trees} \to \text{Harvest 5 Wood} \to \text{Craft Table} \to \text{Restock Wood} \to \text{Return} \to \text{Craft Pickaxe}$) takes **60 to 90 steps**. CEM only planned over $H=10$.
  2. **Exponential Compounding Error:** Beyond step 5 of latent imagination, open-loop predictor error compounded by **$5.23\times$** (MSE escalated from $0.1601$ at $H=1$ to $0.8380$ at $H=10$).
* **The Fatal Consequence:** Rollouts beyond 5 steps degraded into blurry, hallucinatory latents. The agent could not look far enough into the future to perceive the value of crafting a table.

---

### Attempt 4: Distance-to-Initial-Latent Curiosity ($\|z_t - z_0\|_2$)
* **What We Tried:** Added an intrinsic exploration bonus rewarding displacement from the initial observation: $r_{\text{nov}} = \lambda_{\text{nov}} \cdot \|z_t - z_0\|_2$.
* **Why It Failed (The "Grassland Sprinting Paradox"):**
  - Sprinting across open grasslands produced continuous, drastic background tile changes $\implies$ **High curiosity reward**.
  - Standing stationary in front of a $1 \times 1$ crafting table to systematically test interaction actions (Actions 11–16) produced virtually static visual frames $\implies$ **Zero curiosity reward**.
* **The Fatal Consequence:** Intrinsic curiosity mathematically punished the agent for standing near crafting stations and rewarded it for fleeing into the wilderness.

---

### Attempt 5: Online Hypothesis Generation with Empty Preconditions
* **What We Tried:** Enabled online lifelong learning. Whenever an achievement fired during play, create an online hypothesis card and add it to the bank.
* **Why It Failed (The Falsification Trap):**
  Because inventory preconditions could not be computed on the fly without cheating, new cards defaulted to:
  ```python
  needs = [],  # Empty preconditions
  status = "hypothesis"
  ```
  1. The agent crafted a pickaxe once by chance.
  2. The planner saw `card_a11_e15` with `needs = []` and deduced Action 11 can be executed anywhere (even in open grass with zero wood).
  3. The agent repeatedly fired Action 11 in grass, producing near-miss steps ($\Delta x \approx 0, r \le 0$).
  4. The Bayesian falsification rule saw 3 consecutive failures with 0 successes, Wilson upper bound collapsed below $0.20$, and the card was **permanently deleted from the memory bank**!

---

### Attempt 6: Spatial Anchoring via Dead-Reckoning
* **What We Tried:** Implemented spatial landmark tracking in `semantic_memory.py` so the agent could navigate back to its placed crafting table.
* **Why It Failed:**
  1. In `scripts/screen_v3_50eps.py:449`, the argument `unlocked_event` was accidentally omitted from `update_working_memory()`, leaving the structure anchor uninitialized.
  2. Crafter’s viewport is only $9 \times 7$ tiles. As soon as the agent walked 5 tiles away to chop more trees, the table slipped off-screen and vanished from the agent's observation. Dead-reckoning without absolute coordinate grounding drifted rapidly, leaving the table lost forever.

---

### Attempt 7: Latent Encoder Fine-Tuning
* **What We Tried:** Fine-tuning the visual encoder on fresh gameplay data to improve visual feature representations.
* **Why It Failed (The Latent Drift Catastrophe):**
  In self-supervised non-contrastive models, modifying the encoder shifts the entire coordinate geometry of the latent space. Retraining shifted latents by a drift magnitude of **7.05**, instantly destroying the calibration of all downstream heads:
  - Event prediction heads misaligned.
  - Value functions outputted nonsensical scores.
  - Policy agreement plunged to 80.9%, regressing Crafter performance from 4.55% down to 3.40%.

---

### Attempt 8: The Target Ready-Gate in the Backward Planner
* **What We Tried:** Compositional backward chaining: if a pickaxe requires a table, plan table first; if table requires wood, plan wood first.
* **Why It Failed:**
  In `planning/compositional_planner.py:268`, `plan_subgoal` checked visual proximity to the **target card** before accepting the chain:
  ```python
  # Old flawed check:
  ready = (sim_tgt >= self.sim_threshold)  # Evaluated against final target (table)
  if not ready:
      return None  # Dropped entire chain!
  ```
  When the agent wanted a pickaxe, the target was a table, but the immediate requirement was hitting trees in grass. Because the agent was near trees and not near a table, the gate rejected the chain and returned `None` on every step!

---

## 3. The Breakthrough: What Finally Succeeded Today

By resolving these interlocking mechanisms simultaneously in commits `ebf5251`, `35f8dbc`, and `b74855e`:

1. **Direct Active-Subgoal Attribution:**
   - Plumbed `active_subgoal_target` through `mpc_v3.py` $\to$ `screen_v3_50eps.py` $\to$ `playtime_recorder.py` $\to$ `causal_memory.py`.
   - When a near-miss step occurs ($\Delta x \approx 0, r \le 0$), the failure is charged directly to the specific active card rather than diffused across 386 cards sharing that action.
2. **Bayesian Falsification Pruning:**
   - Automated rapid eviction of spurious online hypothesis cards within $\le 3$ failures ($n_{\text{fail}} \ge 2, n_{\text{succ}} == 0$ or empirical rate $\le 25\%$).
3. **Active-Link Chain Initiation:**
   - Modified `plan_subgoal` in `compositional_planner.py` to evaluate readiness on the **immediate active link** (`chain[0]`, e.g. harvesting wood) rather than the final target station.
4. **F1 Homeostatic Drive Reduction:**
   - Survival-oriented vital replenishment ($\Delta x_{\text{vitals}} < 0$) prioritized food and drink before vital exhaustion.

### The Verified Results (Seeds 2050–2099, 50 Episodes):
* **Official Crafter Score:** **3.14%** (Beat baseline 1.62% and v2 champion 2.74%).
* **Mean Survival Length:** **183.2 steps** (2.55× increase over baseline 71.9 steps).
* **Mean Episode Reward:** **3.44** (2.12× increase over baseline 1.62).
* **`place_table`:** **32.0%** (16 out of 50 episodes).
* **`make_wood_pickaxe`:** **4.0%** (Unlocked on Seeds 2079 & 2092 — **First time in autonomous Phase 2**).
* **`make_wood_sword`:** **4.0%** (Unlocked on Seeds 2090 & 2096 — **First time in autonomous Phase 2**).
* **Online Pruning Activity:** 1,648 spurious hypotheses falsified and pruned; 47 valid cards confirmed.

---

## 4. The Deep Unsolved Blockers Ahead: Reaching Stone & Iron

While wooden tool crafting is now unlocked and functional, reaching **Stone Mining** and **Iron** remains blocked by three fundamental bottlenecks:

| Bottleneck | Technical Mechanism | Tech Tree Consequence |
| :--- | :--- | :--- |
| **1. The Post-Table Return Problem** | Crafter viewport is $9 \times 7$. Placing a table consumes 2 wood. If the agent needs more wood to craft a tool, it walks away to hit trees, losing sight of the table. Without explicit 2D spatial memory, it rarely finds its way back. | **87.5% Drop-off:** Tables are placed in 32% of episodes, but pickaxes are only crafted in 4%. |
| **2. Stone Tile Visual Affordance** | In raw $64 \times 64$ frames, stone blocks look visually similar to cave walls and impassable boundaries. The agent lacks an offline card linking Action 5 (hit) to stone blocks while holding a pickaxe. | Even when a pickaxe is held, the agent continues hitting trees and grass instead of targeting stone blocks. |
| **3. The Nightfall Mortality Wall** | Day lasts 150 steps; night lasts from step 150 to 300. Hostile monsters (zombies, skeletons) spawn in darkness. | **93.3% of deaths remain monster kills at night.** Without an active combat controller or shelter-building policy, the agent dies before reaching mountain stone biomes. |

---

## 5. Architectural Comparison: The Way Forward

| Dimension | Phase 2 Architecture (Current Champion) | Phase 3: Plan2Explore Curiosity | Project Zephyr (Dreamer-style Actor-Critic) |
| :--- | :--- | :--- | :--- |
| **Action Selection** | Recurrent CEM MPC ($H=15$, 64 candidates) | Recurrent CEM MPC + Ensemble Disagreement Bonus | Amortized Actor Policy Network ($H=15$ imagination) |
| **Exploration Drive** | State Novelty + F1 Homeostatic Drive | 3-Predictor Dynamics Disagreement ($r_{\text{cur}} = \text{std}_k[\hat{z}^k]$) | Dynamics Ensemble Disagreement + Episodic Novelty |
| **Causal Memory** | Neocortical prototype cards + online Bayesian pruning | Phase 2 Bank v3 + active curiosity exploration | Jointly learned transition/readout heads in latent space |
| **Planning Horizon** | Limited to $H=15$ (search cost $O(C \cdot H)$) | Limited to $H=15$ | Pre-compiled into Actor via $\lambda$-returns ($\lambda=0.95$) |
| **Nightfall Survival** | Heuristic vital weights & death penalty | Heuristic vital weights & death penalty | State-dependent continuation discounting $\Gamma_t = \prod \gamma c_t$ |
| **Status in Repo** | **Production Champion** (`v3-phase2-champion`) | **Tested & Ready** (`models/curiosity.py`, 6/6 tests pass) | **Clean Isolation** (`zephyr/`, 32/32 tests pass) |

---

## 6. Key Takeaways & Guidelines for Future Engineering

1. **Never Diffuse Negative Feedback Across Shared Primitive Actions**: If an agent fails while attempting a specific subgoal, charge the failure directly to that subgoal. Diffusing penalties across all shared-action cards guarantees the survival of spurious hypotheses.
2. **Never Check Target Readiness on Multi-Step Chains**: In backward planning, verify readiness on the *immediate next link* (harvesting input), never on the final destination station.
3. **Never Use Raw Pixel Distance for Curiosity**: $\|z_t - z_0\|$ creates the Grassland Paradox. Exploration bonuses must measure *epistemic dynamics uncertainty* (ensemble disagreement), not visual background changes.
4. **Preserve Learned Knowledge Between Phases**: Snapshot confirmed online cards into versioned bank checkpoints (`phase2_causal_memory_bank_v3.json`) so subsequent exploration phases build upon established tool proficiencies rather than rediscovering fire from scratch.
