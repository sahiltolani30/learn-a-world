# 🚀 Can a World Model Learn to Play and Plan With ZERO Human Rules?

Most game-playing AI agents look impressive until you look under the hood: they often rely on hand-coded "cheat reflexes" (e.g., *if health < 5, run away; if tree in front, chop*). Remove those human-written heuristics, and the model completely collapses.

That was our exact problem in **Learn-a-World v1**. When we tested our latent MPC world model without its handcrafted reflexes, its Crafter score was **0.00%**. The neural network had great eyes, but zero internal understanding of value, health, or goals.

Today, we are promoting **Learn-a-World v2** to `main`: **a pure, autonomous JEPA-style world model that learns to play, survive, and craft with zero hardcoded rules.**

---

### 📊 The Locked Evaluation Numbers (100 Episodes · Seeds 1000..1099)

We ran a locked 100-episode comparison across four agents under a single, unified test harness on the official **Crafter** benchmark:

* 🤖 **v2_pure (Our New Model):** **2.16%** · **10 / 22 Achievements** · Mean Reward: **+2.09**
* 🛠️ **v1_hybrid (With Human Cheat Rules):** **1.44%** · 7 / 22 Achievements · Mean Reward: +1.62
* 🎲 **Random Action Baseline:** **1.77%** · 7 / 22 Achievements · Mean Reward: +1.75
* ❌ **v1_pure (v1 Without Cheat Rules):** **0.00%** · 0 / 22 Achievements · Mean Reward: -0.90

> **Verdict:** `v2_pure` officially beat `v1_hybrid` (+0.72% margin) and crushed `v1_pure` (+2.16% margin), passing the project's strict promotion gate with **zero human intervention**.

---

### 🧠 What Changed Inside the Architecture?

Instead of generating raw pixels (which wastes millions of compute FLOPs rendering grass and tree textures), Learn-a-World follows a **JEPA (Joint Embedding Predictive Architecture)** philosophy:

1. **Learned Reward & Event Heads:** An `EventHead` trained with Focal Loss predicts probabilities across all 22 Crafter achievements directly from latent embeddings $z \in \mathbb{R}^{128}$.
2. **Distributional Value Ensemble (`QEnsemble`):** An ensemble of 3 Two-Hot categorical value heads predicts future returns while penalizing trajectories where the ensemble disagrees (epistemic uncertainty penalty).
3. **Multi-Step Imagination Consistency:** The latent dynamics model unrolls 4 steps into the future, enforcing temporal consistency across imagined trajectories.
4. **Self-Play Collect–Retrain Loop:** The agent plays using its own planner, collects its successes and mistakes, and fine-tunes itself across 3 iterative rounds (peaking at **2.68%** during training).

---

### 🔨 What the Model Autonomously Discovered

* **Tool Crafting Unlocked for the First Time:** The agent learned to collect wood, place crafting tables (5.0%), and craft **wooden pickaxes (1.0%)** and **wooden swords (1.0%)**. In v1 and random play, tool crafting was 0.0%.
* **Autonomous Sleep / Circadian Rhythm:** Nights in Crafter are lethal. The model autonomously learned to sleep through the night (**`wake_up: 96%`**), regenerating health and avoiding monster spawns.
* **10 / 22 Achievements:** Matching the achievement count of DeepMind’s **DreamerV2**!

---

### 🔍 Radical Transparency: The Honest Caveats

We believe in reporting RL results with complete scientific honesty:

1. **Why does Random score 1.77%?** In Crafter, randomly pressing buttons causes the agent to sleep through nights and passively stumble upon saplings and cows. But random play **never crafts tables and never crafts tools (0.0%)**.
2. **The 1% Tool Bottleneck:** While the agent unlocked Tier 3 tools, it does so in 1% of games. Why? Because our planning horizon is **8 steps ($H=8$)**. Chaining a 20-step crafting recipe with an 8-step planner is right at the boundary of what short-horizon MPC can achieve.
3. **Compute Budget:** SOTA models like DreamerV3 and Director train on 1,000,000 to 5,000,000 steps on Google TPU pods. We achieved this on **390,000 transitions on a single free Google Colab GPU in 2.5 hours**.

---

### 🔮 What’s Next: Phase 5

To break past PPO (~4.2%) and DreamerV2 (~10.0%), we are implementing **Autonomous Causal Precedence DAG Mining** and a **Manager–Worker Latent Hierarchy** to extend planning from 8 steps to 100+ steps.

* Full Per-Achievement Scorecard & Caveats: [RESULTS.md](https://github.com/sahiltolani30/learn-a-world/blob/main/RESULTS.md)  
* Complete Open-Source Codebase: https://github.com/sahiltolani30/learn-a-world  
*(v1 remains permanently accessible at tag `v1-freeze`)*.

#MachineLearning #DeepLearning #WorldModels #ReinforcementLearning #ArtificialIntelligence #PyTorch #JEPA #AutonomousAgents
