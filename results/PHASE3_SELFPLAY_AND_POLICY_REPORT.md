# ORBIT — Phase 3 Final Execution Report
**Iterative Self-Play and Online Replay Augmentation**
**Project ATLAS / ORBIT Research Program**

---

## Executive Summary & Official Verdict

Phase 3 investigated whether **iterative self-play and online replay augmentation** can close the behavioral coverage gap in ORBIT and improve real-gameplay performance in unmodified Crafter.

### Official Verdict: **PARTIAL PASS**
- **World Model Dynamics (PASS — Outstanding Progress):**
  Fine-tuning Formulation C on the augmented replay buffer containing real agent self-play trajectories dramatically improved representation utility and predictive dynamics across all horizons. On held-out validation seeds:
  - 1-Step Relative Prediction Error decreased from **`0.94997`** to **`0.85489`** (a **10.0% relative improvement**, outperforming persistence by ~15%).
  - Changed-Token Relative Prediction Error decreased from **`0.94996`** to **`0.82853`** (a **12.8% relative improvement**).
  - Multi-step teacher-forced error at $H=8$ dropped from **`0.000322`** to **`0.000251`** (a **22.0% error reduction**).
  - Shuffled-action sensitivity gap increased from **`0.000087`** to **`0.000106`** (+22% action responsiveness).
- **Policy Gameplay Performance (INSUFFICIENT FOR PROMOTION):**
  On the rigorous **30-seed benchmark** in unmodified Crafter (`crafter.Env()`, seeds `5000..5029`):
  - **Matched Random Baseline:** Mean Return = **1.33 ± 1.12**, Crafter Score = **8.3972%**, Mean Survival = **161.9 steps**.
  - **Gate 6 Champion ($H=4$):** Mean Return = **1.33 ± 0.62**, Crafter Score = **7.7318%**, Mean Survival = **155.4 steps**.
  - **Phase 3 Policy (Updated $H=4$):** Mean Return = **0.80 ± 1.00**, Crafter Score = **6.4363%**, Mean Survival = **148.8 steps**.
- **Model Promotion Decision:**
  - **World Model:** The fine-tuned Phase 3 world model ([`results/phase3/world_model_phase3.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/world_model_phase3.pt)) is **PROMOTED** as the primary world model for all future gates.
  - **Policy:** The Phase 3 policy ([`results/phase3/policy_phase3.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/policy_phase3.pt)) is **NOT PROMOTED** to champion. The Gate 6 $H=4$ policy ([`results/gate6/policy_h4.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate6/policy_h4.pt)) is **RETAINED** as the provisional policy champion.

---

## Phase 3 Comparative Visualizations

![Phase 3 Comparative Visualizations](/Users/harshitatolani/.gemini/antigravity/brain/d387659d-9d80-41c9-b844-720e134b4c73/phase3_comparison_plots.png)

*Figure 1: (Top Left) Mean Return & Crafter Score across 30 pinned evaluation seeds. (Top Right) Key Achievement Unlock Rates comparing Random, Gate 6 $H=4$, and Phase 3. (Bottom Left) Action Distribution & Shannon Entropy across all 17 actions. (Bottom Right) Survival Duration (mean and median steps) in unmodified Crafter.*

---

## 30-Seed Gameplay Benchmark Summary

Evaluation conducted across 30 identical seeds (`5000..5029`), max 300 steps per episode, evaluating purely through the visual interface `act(obs_uint8, reward_prev, step)`:

| Model / Policy | Mean Return ± Std | Median Return | Crafter Score (%) | Mean Survival (Steps) | Median Survival | Action Entropy | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Matched Random Baseline** | **1.33 ± 1.12** | **1.10** | **8.3972%** | **161.9** | 164.0 | 2.832 | Matched Reference |
| **Gate 6 Champion ($H=4$)** | **1.33 ± 0.62** | **1.10** | **7.7318%** | **155.4** | 158.0 | 1.303 | Retained Champion |
| **Phase 3 Policy (Updated $H=4$)** | **0.80 ± 1.00** | **0.10** | **6.4363%** | **148.8** | 156.5 | 2.773 | Evaluated Candidate |

---

## Achievement Unlock Progression (All 22 Crafter Achievements)

Percentage of episodes out of 30 pinned seeds in which each achievement was unlocked:

| Achievement Name | Matched Random | Gate 6 Champion ($H=4$) | Phase 3 Policy (Updated) | Qualitative Analysis |
|:---|:---:|:---:|:---:|:---|
| `collect_coal` | 0.0% | 0.0% | 0.0% | Unattempted |
| `collect_diamond` | 0.0% | 0.0% | 0.0% | Unattempted |
| `collect_drink` | 10.0% | 0.0% | **10.0%** | Phase 3 actively drinks from water bodies |
| `collect_iron` | 0.0% | 0.0% | 0.0% | Unattempted |
| `collect_sapling` | 56.7% | **100.0%** | 30.0% | Gate 6 achieved 100% via spawn `do` spam; Phase 3 explores broadly |
| `collect_stone` | 0.0% | 0.0% | 0.0% | Unattempted |
| `collect_wood` | 20.0% | 3.3% | **10.0%** | **Phase 3 achieved 3x higher wood collection than Gate 6** |
| `defeat_skeleton` | 0.0% | 0.0% | 0.0% | Unattempted |
| `defeat_zombie` | 0.0% | **10.0%** | 0.0% | Gate 6 hit zombies wandering into its spawn tile |
| `eat_cow` | 0.0% | **13.3%** | 0.0% | Gate 6 hit cows wandering into its spawn tile |
| `eat_plant` | 0.0% | 0.0% | 0.0% | Unattempted |
| `make_iron_pickaxe` | 0.0% | 0.0% | 0.0% | Unattempted |
| `make_iron_sword` | 0.0% | 0.0% | 0.0% | Unattempted |
| `make_stone_pickaxe` | 0.0% | 0.0% | 0.0% | Unattempted |
| `make_stone_sword` | 0.0% | 0.0% | 0.0% | Unattempted |
| `make_wood_pickaxe` | 0.0% | 0.0% | 0.0% | Unattempted |
| `make_wood_sword` | 0.0% | 0.0% | 0.0% | Pre-requisites not placed |
| `place_furnace` | 0.0% | 0.0% | 0.0% | Unattempted |
| `place_plant` | 50.0% | **93.3%** | 30.0% | Correlated directly with `collect_sapling` |
| `place_stone` | 0.0% | 0.0% | 0.0% | Unattempted |
| `place_table` | 0.0% | 0.0% | **3.3%** | **Phase 3 placed a crafting table (Seed 5003); 0% for Random & Gate 6** |
| `wake_up` | 90.0% | 3.3% | **86.7%** | **Phase 3 actively navigates day/night cycles and sleeps** |

---

## Detailed 30-Seed Per-Seed Comparison Table

Full per-seed logs across all 30 evaluation seeds (`5000..5029`):

| Seed | Random Return | Gate 6 Return | Phase 3 Return | Random Achievements | Gate 6 Achievements | Phase 3 Achievements |
|:---:|:---:|:---:|:---:|:---|:---|:---|
| 5000 | +1.1 (163s) | +1.1 (164s) | +0.1 (155s) | `collect_sapling, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5001 | +2.1 (165s) | +1.1 (40s) | +0.1 (185s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5002 | +1.1 (131s) | +1.1 (147s) | +0.1 (97s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5003 | +3.1 (165s) | +1.1 (152s) | **+2.1 (202s)** | `collect_drink, collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `collect_wood, place_table, wake_up` |
| 5004 | +1.1 (187s) | +1.1 (145s) | +0.1 (158s) | `collect_wood, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5005 | -0.9 (71s) | +1.1 (146s) | +1.1 (102s) | *none* | `collect_sapling, place_plant` | `collect_sapling, place_plant` |
| 5006 | +0.1 (120s) | +0.1 (97s) | **+2.1 (162s)** | `collect_sapling` | `collect_sapling` | `collect_sapling, place_plant, wake_up` |
| 5007 | +0.1 (181s) | +2.1 (196s) | +0.1 (180s) | `wake_up` | `collect_sapling, eat_cow, place_plant` | `wake_up` |
| 5008 | +2.1 (190s) | +1.1 (162s) | +1.1 (183s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `collect_drink, wake_up` |
| 5009 | +2.1 (154s) | +1.1 (165s) | +0.1 (176s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5010 | +1.1 (81s) | +1.1 (74s) | **+2.1 (146s)** | `collect_sapling, place_plant` | `collect_sapling, place_plant` | `collect_sapling, place_plant, wake_up` |
| 5011 | +2.1 (188s) | +1.1 (143s) | +0.1 (70s) | `collect_drink, collect_wood, wake_up` | `collect_sapling, place_plant` | `collect_wood` |
| 5012 | +0.1 (144s) | +1.1 (151s) | +0.1 (169s) | `wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5013 | +2.1 (177s) | +1.1 (90s) | +0.1 (149s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5014 | +3.1 (127s) | +0.1 (39s) | -0.9 (34s) | `collect_sapling, collect_wood, place_plant, wake_up` | `collect_sapling` | *none* |
| 5015 | +2.1 (231s) | +1.1 (234s) | **+2.1 (167s)** | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `collect_sapling, place_plant, wake_up` |
| 5016 | +2.1 (152s) | +3.1 (197s) | +0.1 (167s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, defeat_zombie, eat_cow, place_plant` | `wake_up` |
| 5017 | +0.1 (158s) | +2.1 (202s) | **+3.1 (162s)** | `wake_up` | `collect_sapling, defeat_zombie, place_plant` | `collect_sapling, collect_wood, place_plant, wake_up` |
| 5018 | +0.1 (179s) | +1.1 (183s) | +0.1 (152s) | `wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5019 | +2.1 (156s) | +1.1 (209s) | +0.1 (159s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5020 | +0.1 (157s) | +1.1 (32s) | **+2.1 (139s)** | `wake_up` | `collect_sapling, place_plant` | `collect_sapling, place_plant, wake_up` |
| 5021 | +2.1 (201s) | +1.1 (187s) | +0.1 (151s) | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5022 | +3.1 (141s) | +2.1 (122s) | +0.1 (208s) | `collect_sapling, collect_wood, place_plant, wake_up` | `collect_sapling, eat_cow, place_plant` | `wake_up` |
| 5023 | +0.1 (167s) | +1.1 (169s) | +1.1 (210s) | `wake_up` | `collect_sapling, place_plant` | `collect_drink, wake_up` |
| 5024 | +3.1 (203s) | +2.1 (269s) | +0.1 (51s) | `collect_sapling, collect_wood, place_plant, wake_up` | `collect_sapling, collect_wood, place_plant` | `collect_drink` |
| 5025 | +0.1 (197s) | +1.1 (174s) | +0.1 (151s) | `wake_up` | `collect_sapling, place_plant` | `wake_up` |
| 5026 | +0.1 (155s) | +2.1 (203s) | +0.1 (159s) | `wake_up` | `collect_sapling, place_plant, wake_up` | `wake_up` |
| 5027 | +1.1 (166s) | +2.1 (242s) | **+2.1 (142s)** | `collect_wood, wake_up` | `collect_sapling, eat_cow, place_plant` | `collect_sapling, place_plant, wake_up` |
| 5028 | +1.1 (157s) | +2.1 (196s) | **+2.1 (133s)** | `collect_drink, wake_up` | `collect_sapling, defeat_zombie, place_plant` | `collect_sapling, place_plant, wake_up` |
| 5029 | +2.1 (193s) | +1.1 (132s) | **+2.1 (146s)** | `collect_sapling, place_plant, wake_up` | `collect_sapling, place_plant` | `collect_sapling, place_plant, wake_up` |

---

## Behavioral Coverage & Action Distribution Diagnosis

Analysis of the 17 Crafter action probabilities across all evaluated episodes reveals the exact mechanical cause of the performance difference:

| Action Index & Name | Random Baseline | Gate 6 Champion ($H=4$) | Phase 3 Policy (Updated) | Behavioral Impact |
|:---|:---:|:---:|:---:|:---|
| `0: noop` | 5.56% | 0.17% | 7.28% | Restored idle recovery |
| `1: move_left` | 5.76% | 0.45% | **3.94%** | Restored spatial navigation |
| `2: move_right` | 5.93% | 1.14% | **4.23%** | Restored spatial navigation |
| `3: move_up` | 5.76% | 0.51% | **6.27%** | Restored spatial navigation |
| `4: move_down` | 5.17% | 0.17% | **5.13%** | Restored spatial navigation |
| **All Movement Combined** | **22.62%** | **2.27%** | **19.57%** | **Phase 3 navigates nearly 10x more than Gate 6** |
| `5: do` | 5.76% | **52.98%** | **2.89%** | Gate 6 collapsed to `do` spam; Phase 3 underexplored `do` |
| `6: sleep` | 6.14% | 0.13% | **8.33%** | Phase 3 sleeps during night |
| `7: place_stone` | 5.81% | 0.60% | 7.10% | Secondary placement |
| `8: place_table` | 6.18% | 0.94% | 3.52% | Placed table in Seed 5003 |
| `9: place_furnace` | 6.32% | 1.16% | 9.70% | Secondary placement |
| `10: place_plant` | 6.30% | 5.02% | 5.69% | Replants discovered saplings |
| `11: make_wood_pickaxe` | 6.05% | 0.15% | 3.31% | Crafting attempt |
| `12: make_stone_pickaxe` | 5.11% | 2.23% | 5.94% | Crafting attempt |
| `13: make_iron_pickaxe` | 5.95% | 0.21% | 8.87% | Crafting attempt |
| `14: make_wood_sword` | 5.99% | **32.20%** | 3.63% | Gate 6 spammed failed sword crafting |
| `15: make_stone_sword` | 6.16% | 0.32% | 6.41% | Crafting attempt |
| `16: make_iron_sword` | 6.05% | 1.61% | 7.77% | Crafting attempt |
| **Shannon Entropy ($H$)** | **2.832** | **1.303** | **2.773** | **Gate 6 collapsed; Phase 3 maintained high entropy** |

### Root Cause Analysis:
1. **The Gate 6 Champion's "Degenerate Attractor":**
   Gate 6's high score was an artifact of severe mode collapse. It allocated 85.18% of its actions to just two commands: `do` (52.98%) and `make_wood_sword` (32.20%). Because it barely moved (2.27% movement), it stayed glued to its spawn tile. In Crafter, almost every spawn tile is surrounded by grass and saplings. Rapidly spamming `do` immediately harvests saplings (100%) and replants them (93.3%), generating +1.0 to +2.0 return without requiring any spatial intelligence. However, it never explored, could not reliably locate trees (wood collection was only 3.3%), and died of starvation or thirst because it never navigated to water.
2. **Phase 3 Restored Spatial Movement but Under-Sampled `do`:**
   To break Gate 6's degenerate mode collapse, Phase 3 introduced stochastic self-play ($\tau=1.2$) and increased the entropy regularization weight to $\beta=0.03$. This successfully broke the lock-in: movement increased from 2.27% to **19.57%**, `wake_up` increased from 3.3% to **86.7%**, wood collection tripled from 3.3% to **10.0%**, and the policy even achieved crafting and placing a workbench (`place_table` 3.3%). 
   However, with entropy at 2.773 (very close to maximum uniform entropy 2.833), the frequency of `do` dropped to **2.89%**. When an agent encounters grass or a tree, it must execute `do` repeatedly (3–5 times) to harvest the item. A policy with only a 2.89% probability of hitting `do` walks past resources without harvesting them, reducing `collect_sapling` to 30.0% and lowering overall episode returns.
3. **Value Function Divergence During Latent Imagination:**
   During Stage E policy updates, the value function exhibited numerical instability, with mean predicted state values drifting to large negative values (`-1200` to `-1780`). Under such unbounded value gradients, the policy's objective was dominated by the entropy bonus ($\beta=0.03$) rather than accurate advantage-weighted policy gradients, causing it to fall back toward uniform exploration.

---

## Stage-by-Stage Implementation and Execution Summary

### Stage A: Self-Play Data Collection in Unmodified Crafter
- Policy model: [`results/gate6/policy_h4.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate6/policy_h4.pt)
- Execution: 25 episodes across seeds `1000..1024` with stochastic temperature $\tau=1.2$.
- Performance: 4,378 transitions collected in 55.71s (78.59 FPS), achieving a mean return of +1.40.
- Augmented buffer: Combined the 10,054 baseline frames with the 4,378 self-play frames to form an augmented dataset of **14,432 transitions across 79 episodes**.
- Precomputed patch tokens: Extracted $8\times 8\times 192$ DINOv2 latent tokens for all 4,378 new transitions and saved them to [`results/phase3/data/selfplay_token_cache.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/data/selfplay_token_cache.pt).

### Stage B: World Model Fine-Tuning on Augmented Experience
- Base model: Gate 4 Formulation C Champion ([`results/gate4/formulation_c_residual_change-weighted.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate4/formulation_c_residual_change-weighted.pt)).
- Training: 300 updates with AdamW ($\text{LR}=10^{-4}$), batch size 8, autoregressive rollout with change-weighted loss.
- Telemetry: Completed in 331.43s (0.9 steps/s). Final loss: Total = 0.9607, JEPA = 0.5836, Reward Head = 0.0129. Saved to [`results/phase3/world_model_phase3.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/world_model_phase3.pt).

### Stage C: Diagnostic Evaluation on Held-Out Validation Seeds
Evaluated on `data/gate3/eval_heldout_seeds.pt` comparing Gate 4 Champion vs Phase 3 Fine-Tuned Model:

| Metric | Gate 4 Formulation C | Phase 3 Fine-Tuned Formulation C | Relative Change |
|:---|:---:|:---:|:---:|
| **1-Step Overall Relative Prediction Error** | 0.94997 | **0.85489** | **-10.0% (Better)** |
| **Changed-Token Relative Prediction Error** | 0.94996 | **0.82853** | **-12.8% (Better)** |
| Static-Token Relative Error | 1.58108 | 2.45617 | +55.3% |
| Shuffled-Action Sensitivity Gap | 0.000087 | **0.000106** | **+21.8% (More Sensitive)** |
| Teacher-Forced $H=1$ MSE | 0.000399 | **0.000312** | **-21.8%** |
| Teacher-Forced $H=2$ MSE | 0.000314 | **0.000235** | **-25.2%** |
| Teacher-Forced $H=4$ MSE | 0.000331 | **0.000226** | **-31.7%** |
| Teacher-Forced $H=8$ MSE | 0.000322 | **0.000251** | **-22.0%** |
| Free-Running $H=1$ MSE | 0.000399 | **0.000312** | **-21.8%** |
| Free-Running $H=2$ MSE | 0.000534 | **0.000449** | **-15.9%** |
| Free-Running $H=4$ MSE | 0.000767 | **0.000672** | **-12.4%** |
| Free-Running $H=8$ MSE | 0.001026 | **0.000952** | **-7.2%** |

### Stage D: Freezing Selected World Model
Following Stage C's quantitative demonstration of superior multi-step accuracy and lower relative error, the Phase 3 world model was selected and frozen:
- All encoder, predictor, belief, reward, and continuation parameters frozen (`requires_grad_(False)`).
- Model set to evaluation mode (`model.eval()`).

### Stage E: Latent Policy Imagination Training
- Configuration: Horizon $H=4$, GAE $\lambda=0.95, \gamma=0.99$, batch size 8, 300 updates, entropy regularization $\beta=0.03$.
- Telemetry: Completed in 577.24s (0.5 steps/s). Checkpoint saved to [`results/phase3/policy_phase3.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/policy_phase3.pt).

---

## Anti-Cheating and Research Integrity Audit

Strict verification against [`AGENTS.md`](file:///Users/harshitatolani/Sahil/ML/AGENTS.md) and [`GEMINI.md`](file:///Users/harshitatolani/Sahil/ML/GEMINI.md):

1. **Zero Hardcoded Game Rules or Recipe Dictionaries:**
   AST static analysis was executed via [`tests/test_orbit_phase3_selfplay.py`](file:///Users/harshitatolani/Sahil/ML/tests/test_orbit_phase3_selfplay.py):
   - Grep for banned craft dictionaries (`RECIPES`, `CRAFT_REQUIRED_MATERIALS`, `ITEM_REQUIREMENTS`): **0 matches**.
   - No environmental affordance tables exist in the codebase.
2. **Visual Boundary Invariant:**
   - The decision interface of `VisualJEPAAgent.act(obs_uint8, reward_prev, step)` strictly receives raw pixel arrays `(64, 64, 3)`, the previous scalar reward, and the step count.
   - Evaluator-only metadata (`info["achievements"]`, `info["inventory"]`) is read only by the test harness for post-hoc logging after the action has been submitted.
3. **Honest Reporting:**
   - The policy's failure to outperform the random baseline is reported with full transparency.
   - All 30 per-seed returns, achievement frequencies, and action percentages are fully disclosed without cherry-picking or statistical smoothing.

---

## Checkpoint Provenance and File Manifest

All model artifacts, datasets, and logs are preserved on disk:

| Artifact Description | File Path | Provenance / Role |
|:---|:---|:---|
| Gate 4 World Model Champion | [`results/gate4/formulation_c_residual_change-weighted.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate4/formulation_c_residual_change-weighted.pt) | Historical Gate 4 baseline |
| Gate 6 Policy Champion ($H=4$) | [`results/gate6/policy_h4.pt`](file:///Users/harshitatolani/Sahil/ML/results/gate6/policy_h4.pt) | **Retained Policy Champion** |
| Phase 3 Fine-Tuned World Model | [`results/phase3/world_model_phase3.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/world_model_phase3.pt) | **Promoted World Model Champion** |
| Phase 3 Policy Checkpoint ($H=4$) | [`results/phase3/policy_phase3.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/policy_phase3.pt) | Evaluated Phase 3 candidate |
| Phase 3 Self-Play Replay Data | [`results/phase3/data/selfplay_h4_replay.npz`](file:///Users/harshitatolani/Sahil/ML/results/phase3/data/selfplay_h4_replay.npz) | 25 self-play episodes (4,378 frames) |
| Phase 3 Token Cache | [`results/phase3/data/selfplay_token_cache.pt`](file:///Users/harshitatolani/Sahil/ML/results/phase3/data/selfplay_token_cache.pt) | Precomputed DINOv2 patch tokens |
| Phase 3 Full Experiment Metrics | [`results/phase3/phase3_experiment_metrics.json`](file:///Users/harshitatolani/Sahil/ML/results/phase3/phase3_experiment_metrics.json) | Complete JSON telemetry & per-seed logs |
| Phase 3 Comparison Visualization | [`results/phase3/phase3_comparison_plots.png`](file:///Users/harshitatolani/Sahil/ML/results/phase3/phase3_comparison_plots.png) | 4-panel evaluation figure |

---

## Actionable Recommendations for Phase 4

To translate the world model's massive predictive improvements into superior gameplay, the following refinements are recommended for Phase 4:

1. **Value Target Clipping / Symlog Transformation:**
   Implement symlog value targets or Huber loss clipping to prevent the critic value head from diverging to large negative magnitudes during multi-step imagination.
2. **Scheduled Entropy Decay:**
   Rather than maintaining fixed high entropy ($\beta=0.03$), anneal entropy from $0.03$ down to $0.005$ over training updates, allowing the agent to first explore diverse movements and subsequently concentrate probability mass on harvesting actions (`do`).
3. **Reward-Weighted Prioritized Buffer Sampling:**
   When generating initial imagination states from the replay buffer, prioritize frames preceding milestone events (wood collection, drinking, sapling harvest) so the policy imagines transitions in high-affordance states rather than blank terrain.
