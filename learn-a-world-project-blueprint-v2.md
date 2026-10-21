# Learn-a-World — Complete Project Blueprint (v2)

**A JEPA-style latent world model that learns by playing: an agent plays a game world, learns its rules from its own experience, plans by imagination — and keeps learning when the world changes.**

- **Builder:** Sahil Kumar Tolani — B.Tech CSE (AI), fresher, Noida, India
- **Purpose:** Portfolio flagship project targeting AI/ML Engineer roles in the 2027 hiring market
- **Timeline:** ~5 weeks, a few hours/day, solo build
- **Budget:** ₹0 (Colab free GPU + Google Drive from Sahil's Jio Pro Google plan, 1 TB)
- **Status:** Blueprint v2 — supersedes v1. Changes from v1: (1) learning is a SELF-PLAY LOOP (agent generates its own experience; no external recordings needed), (2) Crafter is the Stage-3 flagship world, (3) curiosity-driven exploration added, (4) concrete storage/compute plan, (5) new-environment behaviour specified and answered as a design question.
- **For review by:** Claude (external review) — review questions in Section 16.

---

## 1. One-line summary

An agent plays simulated worlds (custom maze → physics pushing task → the real game Crafter). Every step it takes is stored in a capped replay buffer. A JEPA-style world model (encoder + action-conditioned latent predictor, custom-trained in PyTorch) learns the world's dynamics from that self-generated experience. A sampling-based planner (MPC) uses the model to imagine action sequences and act on the best one, so play improves, generating richer experience — a continuous Play → Remember → Learn → Imagine → Play-better loop. Headline experiment: when the world's dynamics change at test time, the model keeps learning from its own prediction errors (test-time adaptation) and recovers, while a frozen model and a PPO baseline degrade.

## 2. Why this project, why now (market context)

- World models are the post-LLM frontier receiving the largest 2026 funding: AMI Labs (Yann LeCun, ex-Meta) ~$1.03B seed (Mar 2026, ~$3.5B valuation); Ineffable Intelligence (UK) $1.1B for autonomous RL-style learning; Recursive Superintelligence (UK) $650M for self-improving AI; World Labs (Fei-Fei Li) $1B+ raised, ~$5B valuation, product "Marble" already commercial; Google DeepMind's Genie 3 (first real-time interactive world model); NVIDIA Cosmos 2M+ downloads; Wayve (world-model driving) $1.2B at $8.6B (Feb 2026). Total 2026 world-model funding > $5B. Use cases being funded: self-driving, warehouse/factory robotics, game & 3D-content generation, and synthetic worlds for training other AIs.
- October 2026 research this project directly builds on:
  - **H-JEPA** (arXiv:2610.06805, 5 Oct 2026): hierarchical JEPA planning; AntMaze success 18% → 73%; open code (on LeWM + stable-worldmodel), 84 checkpoints on Hugging Face.
  - **AdaJEPA** (NYU, Oct 2026): JEPA world model that adapts at test time from the consequences of its actions — inspiration for Stage 4.
  - **JEPA-Anything** (Oct 2026, Apache-2.0): one JEPA recipe across 7 domains.
- Precedent for the self-play design: DeepMind's Dreamer line learned Crafter from scratch through its own play (and was the first agent to collect diamonds in Minecraft) with no human gameplay data.
- Hiring logic: India fresher AI hiring is saturated with RAG/LangChain projects; world-model/JEPA + deep PyTorch experience at fresher level is near-zero supply, pointed at where the funding (and therefore 2027 hiring) is going.

## 3. Problem statement & research questions

Model-free RL memorises behaviour for one reward: sample-hungry, task-locked, brittle when dynamics change. A world model learns dynamics by prediction and can plan for any goal in a world it understands. This project measures, at small scale:

1. How well does a small JEPA model learn environment dynamics purely from an agent's own play?
2. Does latent-space planning beat a PPO baseline on success/achievement score and sample efficiency?
3. Under changed dynamics, how much does a frozen world model degrade, and how much does test-time adaptation recover vs the RL baseline?
4. How far does curiosity-driven exploration extend what the self-play loop can discover (vs random play alone)?

## 4. The learning loop (core design)

```
        ┌──────────────────────────────────────────────────┐
        ▼                                                  │
  1. PLAY — agent acts in the world (early: near-random;
     later: planner-chosen actions + curiosity drive)
        │
  2. REMEMBER — every (observation, action, next observation,
     reward/achievement flags) is auto-saved to the replay
     buffer (fixed cap; oldest overwritten, CCTV-style)
        │
  3. LEARN — world model trains on sampled memories
     (predict next latent from current latent + action)
        │
  4. IMAGINE & PLAN — planner dreams candidate action
     sequences through the model, picks the best, acts
        └────► better play → deeper exploration → richer
               memories → better model → back to 1
```

- No external datasets of gameplay are required at any stage. The agent's own experience is the only training signal for the world model (DROID in Stage 3 is an optional offline extra, not a dependency).
- **Curiosity module:** an intrinsic bonus for visiting states whose outcomes the model predicts poorly / that are novel in latent space, so the young agent explores instead of repeating — the standard fix for the "dies at night forever, never learns shelter" failure of pure random play.
- Early bootstrapping: an initial short phase of random + light scripted play seeds the buffer so the first model version has something to learn from; the loop then runs autonomously.

## 5. Model architecture

### 5.1 Encoder ("eyes")
- Input: observation image (maze top-down view / Push-T frame / Crafter 64×64 frame).
- Output: latent embedding z (128–512 dims).
- Small CNN trained with the predictor (Stages 1–2 and Crafter default). Option for Stage 3 experiments: frozen DINOv2-small features as an encoder variant to compare.

### 5.2 Predictor ("imagination") — core custom-trained model
- Input: z_t + action embedding → Output: predicted ẑ_{t+1}.
- Small transformer/MLP predictor (a few million params), action-conditioned, purely latent (never predicts pixels) — following the LeWM/H-JEPA pattern.
- Loss: latent prediction error vs the encoder's actual z_{t+1}, with JEPA anti-collapse machinery (EMA target encoder / stop-gradient + variance monitoring), per the reference implementations.

### 5.3 Planner (no training)
- MPC with CEM/random-shooting: sample N action sequences of horizon H, roll through the Predictor, score by distance to goal latent (goal = encoded goal observation; in Crafter, goal = achievement-seeking targets such as resource/survival states), execute first k actions, replan.

### 5.4 Adaptation module (Stage 4 headline)
- Online, during play: after each real step, compare predicted vs actual next latent; small bounded gradient updates to the Predictor from a recent-experience window (simplified AdaJEPA loop). Frozen copy always retained as control; drift monitored on a held-out probe set.

### 5.5 Baselines
- **PPO** (stable-baselines3), same environments and interaction budget: the RL rival.
- **Frozen world model** (adaptation off): the control for Stage 4.
- **Published Crafter reference scores** (Dreamer-line results) as external context for the Crafter stage.

## 6. Worlds (small → big)

A "world" = a program exposing state, actions, a step rule (physics), and a rendered observation. The model NEVER sees the step-rule code; it must infer the rules from experience alone.

- **Stage 1–2 — Custom maze world (we write it, ~1 day):** Gymnasium env we code ourselves: grid, walls, agent, goal, plus controllable special zones (e.g., slippery floor). Because we own the rules, we can change them secretly for the Stage-4 adaptation test.
- **Stage 3a — Push-T (imported research env):** 2D visual physics — push a T-block onto a target; sliding/rotation dynamics to learn.
- **Stage 3b — Crafter (flagship, real game):** 2D survival game (tiny Minecraft-like): procedurally generated world each episode (same rules, new layout), 64×64 pixel observations, 22 built-in achievements (collect wood, build table, defeat zombie, …) providing a recognised score. Chosen because: richer interacting rules than any maze; published world-model baselines exist for comparison; runs fast on CPU.
- Optional stretch: Atari (Breakout) — known to be much harder for planning; not committed scope.

## 7. New-environment behaviour (design answer, pre-registered)

- **New layout, same rules** (e.g., unseen Crafter map): expected to work zero-shot — the model learned rules, not a map. Tested explicitly with held-out seeds.
- **New task, same world** (new goal): planner re-aims with no retraining; PPO must retrain per reward — measured as a contrast where feasible.
- **New rules/dynamics** (changed physics): frozen model expected to degrade; adaptation module is the project's answer — it needs only raw experience (no rewards) to re-learn, vs PPO's full reward-driven retraining. Stage 4 measures all three conditions on identical episode sets/seeds.
- **Completely new game:** out of scope honestly — would need fresh self-play data collection (cheap in interaction terms, no labels/rewards needed) plus training; stated as future work, not claimed.

## 8. Data & storage plan (concrete)

- We store FRAMES, never videos. Crafter frame = 64×64×3 uint8 ≈ 12 KB raw; maze/Push-T frames comparable or smaller.
- Replay buffer cap: ~300k transitions ≈ **3–4 GB** on disk (compressed array format, e.g., chunked npz/HDF5-style shards). Buffer overwrites oldest data when full (FIFO) — storage cannot grow beyond the cap.
- Checkpoints: a few hundred MB per stage.
- **Total project footprint: ~15–25 GB** on Sahil's Google Drive (1 TB available via Jio Pro Google plan) — Drive is the single home for dataset + checkpoints + code backups.
- Training loads only small batches (~64 sequences) into RAM/GPU at a time; the full buffer never sits in memory.

## 9. Compute plan (Mac + Colab)

- **Mac (development + light stages):** run environments, self-play data collection (Crafter is CPU-light), coding, Stage 1–2 training (Apple Silicon MPS if available).
- **Google Colab (heavy training):** free T4 GPU for Stage 3 (Crafter/Push-T) training runs; Drive mounted for dataset + checkpoint persistence across Colab sessions; checkpoint frequently (sessions are time-limited); every experiment sized to finish or safely resume within free-tier sessions.
- Deliberate constraint: all model/data sizes chosen so the entire project is reproducible on free compute.

## 10. Training procedure (per stage)

1. Seed buffer (random + scripted play) → 2. Train encoder+predictor on buffer samples (track validation latent-prediction error + latent-variance collapse check) → 3. Enable planner; run loop (interleaved play/train cycles) → 4. Freeze snapshots for evaluation → 5. Log configs/seeds/results. PPO baseline trained separately with matched interaction budget.

## 11. Evaluation protocol (honest measurement, pre-registered)

1. **Prediction quality:** 1-step and multi-step latent prediction error on held-out transitions.
2. **Task performance:** maze — goal success % on held-out start–goal pairs; Crafter — official achievement score (geometric mean across 22 achievements, the benchmark's standard metric) + survival time; Push-T — target coverage/success.
3. **Sample efficiency:** environment steps to reach reported performance, world-model system vs PPO.
4. **Adaptation table (headline):** normal world vs changed world × {frozen WM, adaptive WM, PPO} on identical seeds/episodes.
5. **Generalisation:** held-out map seeds (unseen layouts) performance.
6. **Reporting rules:** fixed seeds; failures and negative results reported; no number goes on the resume unless measured under this protocol.

## 12. Stages & timeline (~5 weeks)

- **Stage 0 (days 1–3):** public GitHub repo; toolchain (PyTorch, Gymnasium, stable-worldmodel/LeWM study); Crafter running locally.
- **Stage 1 (week 1):** custom maze env; self-play loop v1 (buffer + training); first prediction-error numbers.
- **Stage 2 (week 2):** planner live in maze; PPO baseline; first comparison table; curiosity module on/off comparison.
- **Stage 3 (weeks 3):** Push-T, then Crafter full loop; achievement-score measurement vs published reference context.
- **Stage 4 (week 4):** dynamics-change testbed (maze physics change + Crafter-side perturbation where supported); frozen vs adaptive headline table.
- **Stage 5 (week 5):** release — cleaned repo + README with tables and 1-min demo video, HF model + card, Gradio live demo, write-up/LinkedIn post, community feedback.

## 13. Tech stack

Python · PyTorch · Gymnasium · Crafter (official env) · stable-worldmodel / LeWM (reference + envs) · H-JEPA repo (design reference) · stable-baselines3 (PPO) · Hugging Face (encoders, model hosting) · Google Colab + Google Drive (compute/storage) · Gradio (demo) · Git/GitHub.
Explicitly NOT used: any LLM API in the system; LangChain/RAG; paid compute; external gameplay datasets (except optional DROID offline check).

## 14. Deliverables

Public repo; trained checkpoints + HF model card; live Gradio demo; results tables (Sections 11.2/11.4); demo video; technical write-up; resume bullets (Section 15) written only from measured results.

## 15. Intended resume bullets (written AFTER measurement)

- "Built a self-play world-model agent (PyTorch): a JEPA latent dynamics model learned the Crafter game world purely from the agent's own experience (capped replay buffer, curiosity-driven exploration); latent-space MPC planner reached a Crafter achievement score of X vs Y for a PPO baseline on a matched interaction budget (measured)."
- "Implemented test-time adaptation — online predictor updates driven by prediction error — recovering task success from A% → B% under changed environment dynamics, where the frozen model and RL baseline degraded (fixed-seed protocol)."
- "Custom-trained the full system on free compute (Colab + Drive); released code, trained model (Hugging Face) and a live planning demo publicly."

## 16. Risks & mitigations

- **JEPA collapse/instability** → follow reference recipe exactly first (EMA target, variance checks); tiny maze stage makes debugging cheap.
- **Exploration failure in Crafter** (agent never discovers deep achievements) → curiosity bonus (committed), scripted seeding, achievement-progress logging to detect stalls early; scope fallback: report honestly on the achievements reached.
- **Planning cost in Crafter** → latent planning, capped horizon/candidates; latency reported, not hidden.
- **Colab session limits** → checkpointing + Drive persistence; resumable training by design.
- **Crafter dynamics can't be perturbed for Stage 4** → adaptation headline lives in the custom maze (we own its physics) + environment-side perturbations (action/observation noise, layout shifts) in Crafter; pre-declared fallback, not a post-hoc excuse.
- **Scope creep** → hierarchy (H-JEPA multi-level) and Atari are stretch-only.

## 17. Stretch goals

Two-level hierarchical predictor; DINOv2-encoder variant comparison; DROID offline planning check; Atari Breakout probe.

## 18. References

H-JEPA (arXiv:2610.06805, Oct 2026, MIT code, on LeWM + stable-worldmodel) · AdaJEPA (NYU CDS, Oct 2026) · JEPA-Anything (Oct 2026) · Hafner, Crafter benchmark (2021) + DreamerV3 (2023) — self-play world-model precedent and published Crafter reference results · LeCun (2022) JEPA position paper · Meta V-JEPA 2 · DeepMind Genie 3 · World Labs Marble · NVIDIA Cosmos · Funding context: AMI Labs ~$1.03B seed; Ineffable $1.1B; Recursive $650M; Wayve $1.2B Series D · Ha & Schmidhuber (2018) — original world-models paper (the "2–3 years ago" generation this design explicitly differs from: pixel-generative → latent JEPA; offline demos → self-play loop + test-time adaptation; lab-only → open code/free compute).

## 19. Review questions (for the external reviewer)

1. Is the scope realistic for a solo fresher in ~5 weeks on free compute (Mac + Colab free GPU)?
2. Is the self-play loop + curiosity design sound for Crafter, and is the exploration risk adequately handled?
3. Is the evaluation protocol sufficient to support the resume claims in Section 15?
4. Any technical errors in the architecture, training, storage, or compute descriptions?
5. What single change would most strengthen this as a hiring signal for 2027 AI/ML roles?

*End of Blueprint v2.*
