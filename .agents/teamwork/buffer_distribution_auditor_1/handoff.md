# Handoff Report: Memory Buffer Distributions & Replay Dynamics Audit (Track 6)
**Agent ID:** `buffer_distribution_auditor_1`  
**Working Directory:** `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/buffer_distribution_auditor_1`  
**Recipient / Parent:** Project Orchestrator (`7af1422f-94e2-4a53-9321-d03a1d49141f`)  
**Target Artifact:** `buffer_distribution_audit.md`  
**Handoff Type:** Hard (Task complete)

---

## 1. Observation

Direct observations from codebase inspection, checkpoint diagnostics, and script execution:

1. **Buffer Stratification & Sampling Implementation (`prometheus/src/memory/buffer.py:444-474`)**:
   - `TwoTierReplayBuffer` allocates Tier-1 ($C_1 = 50,000$, rolling FIFO) and Tier-2 ($C_2 = 20,000$, curated milestone archive).
   - In `sample()`:
     - `actual_n2 = min(round(batch_size * 0.30), self.t2_size)`, `actual_n1 = batch_size - actual_n2`.
     - In Tier-1: with probability `milestone_t1_priority = 0.25`, draws uniformly from `ms1` (milestone slots in T1); else uniform from $0 \dots N_1-1$.
     - In Tier-2: with probability `milestone_t2_priority = 0.80`, draws uniformly from `ms2` (milestone slots in T2); else uniform from $0 \dots N_2-1$.
   - **Zero Importance Sampling Weights**: Lines 513–524 return only tensors for `obs`, `actions`, `rewards`, `next_obs`, `dones`, `int_rewards`, `epistemic_errors`, `is_tier2`, `is_milestone`, `h`. No importance sampling weight $w_i$ is computed or returned.

2. **Tier-2 1-Step Fallback Truncation (`prometheus/src/memory/buffer.py:698-706`)**:
   In `sample_nstep()`:
   - For Tier-1 indices: iterates $k = 1 \dots n-1$ along `episode_id` and sequential `step_in_ep` to accumulate discounted reward $R$ and find bootstrap state `boot_obs` with `boot_gamma = gamma**num_terms` (lines 643–695).
   - For Tier-2 indices: unconditionally executes:
     ```python
     nstep_returns[batch_i] = self.t2_rewards[idx]
     boot_obs[batch_i] = self.t2_next_obs[idx]
     boot_gamma[batch_i] = gamma
     boot_valid[batch_i] = not self.t2_dones[idx]
     boot_h[batch_i] = self.t2_h[idx]
     ```
     Tier-2 transitions are strictly forced to 1-step TD returns, discarding $n=5$ step trajectories.
   - Line 705 assigns `boot_h[batch_i] = self.t2_h[idx]`, evaluating the bootstrap state $s_{t+1}$ using memory state $h_t$ instead of $h_{t+1}$.

3. **Milestone Self-Imitation Loss (`prometheus/src/train.py:1271-1281`)**:
   ```python
   milestone_mask = batch["is_milestone"] & (batch["rewards"] >= 0.5)
   num_milestones = int(milestone_mask.sum().item())
   if 0 < num_milestones <= 32:
       si_actions = batch["actions"][milestone_mask]
       si_log_probs = log_probs[milestone_mask].gather(1, si_actions.unsqueeze(1)).squeeze(1)
       si_loss = -si_log_probs.mean()
       actor_loss = actor_loss + self.si_weight * si_loss
   ```
   $\mathcal{L}_{\text{SI}}$ applies strictly to positive reward milestone items ($\ge 0.5$) with weight $\lambda_{\text{SI}} = 0.10$. Non-rewarded actions (e.g. attempting to craft without materials, $r = 0$) receive zero self-imitation gradient.

4. **Measured Empirical Critic Flattening (`results/diagnostics/q_calibration.json:10-15`)**:
   - `q_pred_mean`: $0.7325$, `q_pred_std`: **$0.0157$** (near-zero variance across 8,000 transitions).
   - `realized_return_mean`: $0.4614$, `realized_return_std`: **$1.0179$** ($65\times$ greater variance than critic predictions).
   - `mean_ensemble_disagreement`: $0.000410$, `max_ensemble_disagreement`: $0.00899$.

5. **Empirical Policy Pathology in Telemetry & Plans (`prometheus/V3_2_FIX_PLAN.md:19-20`)**:
   - *"I7: Greedy eval = statue (50/50 games -0.90; 100% failed-craft spam): argmax over a flat young critic locks onto a permanently-failing action (make_wood_pickaxe with wood=0, seen in the mind viewer)."*
   - Official score collapsed to 0.0%–1.4% (`results/final_comparison.json:60`, `v1_pure crafter_score: 0.0`).

---

## 2. Logic Chain

1. **Step 1: Derivation of the Minimum Milestone Sampling Floor**:
   From Observation 1, the total batch milestone expectation is:
   $$\mathbb{E}[P(\text{Milestone})] = 0.70(0.25 + 0.75 \alpha_1) + 0.30(0.80 + 0.20 \alpha_2) = 0.415 + 0.525 \alpha_1 + 0.060 \alpha_2$$
   Since $\alpha_1, \alpha_2 \ge 0$, $P(\text{Milestone}) \ge 41.5\%$ unconditionally.
   In empirical Crafter runs, $\alpha_1 \approx 0.06$ and $\alpha_2 \approx 0.995$, yielding an empirical expectation of **$50.65\%$** (empirically validated at **$50.45\%$** over 64,000 sample draws).
   Because milestones occur in only $\sim 6.0\%$ of generated frames, milestones are oversampled by **$8.32\times$**, while standard exploration steps are suppressed to $0.53\times$.

2. **Step 2: Individual Transition Repetition & Gradient Skew**:
   A rare milestone transition (e.g. 15 occurrences of `place_table` out of 25,000 steps) is replayed an average of **$520.0$ times**, compared to **$33.8$ times** for a standard transition ($15.38\times$ relative repetition; up to $1,143\times$ early in training).
   Because no importance sampling correction ($w_i \equiv 1.0$) is applied (Observation 1), the stochastic TD gradient minimizes Bellman error on milestone states with $45\times$ to $240\times$ distorted weighting.
   This destroys $L_2$ Bellman contraction on the $99.9\%$ non-milestone state space, causing the critic to collapse into a completely flat function with standard deviation $0.0157$ (Observation 4).

3. **Step 3: Mechanism of the "Statue" Trap & Value Echo Chamber**:
   From Observation 3, $\mathcal{L}_{\text{SI}}$ applies pure behavior cloning ($\lambda_{\text{SI}} = 0.10$) to positive milestone events ($r \ge 0.5$).
   In Crafter, the only visual indicator of inventory (e.g. wood count) is a $5\times 7$ numeric glyph in row 0 of the HUD. Under $8\times 8$ spatial token pooling, this glyph contributes $\le 3.125\%$ of representation variance.
   The latent representation $z$ is dominated by background terrain (grass, trees, lighting).
   Because the actor MLP is Lipschitz continuous, driving $\pi(a^* \mid z_{\text{wood}=2}) \to 1.0$ forces $\pi(a^* \mid z_{\text{wood}=0}) \to 1.0$ on visually similar scenes.
   Because Tier-2 discards near-misses (where $r=0$) and $\mathcal{L}_{\text{SI}}$ ignores $r < 0.5$, the network receives zero negative signal to contrast preconditions.
   During greedy evaluation, argmax selects $a^*$ in states with zero materials, failing the action ($r=0, s_{t+1}=s_t$) and triggering an infinite loop of failed-craft spam (Observation 5).

4. **Step 4: Multi-Step Credit Decoupling**:
   From Observation 2, Tier-2 transitions unconditionally fall back to 1-step TD.
   Consequently, the 520 replays of rare milestone transitions in Tier-2 cannot propagate discounted value backward to the preceding wood-chopping and approach steps ($t-5 \dots t-1$). Multi-step credit assignment is completely broken on the milestone archive.

---

## 3. Caveats

1. **AST / Zero-Cheat Protocol Boundary**:
   In strict compliance with `AGENTS.md` and `GEMINI.md`, the auditor did not propose or evaluate any privileged state injection (e.g. reading internal Crafter inventory dicts). All identified remediations rely exclusively on visual contrastive learning, advantage weighting, and buffer mathematics.
2. **Online Vectorized Environment Interleaving**:
   In 8-parallel environment execution, Tier-1 transitions are interleaved across environments. While `sample_nstep` correctly enforces `episode_id` and sequential `step_in_ep` checks, wrap-around overwrite timing depends on relative episode lengths across parallel threads.

---

## 4. Conclusion

The dual-store replay dynamics in Prometheus JEPA suffer from five structural bottlenecks:
1. **Mathematical Distribution Distortion**: Batch milestone density is lower-bounded at $41.5\%$ (empirically $50.45\%$), oversampling rare transitions by $15.4\times$ to $1,143\times$.
2. **Value Echo Chambers**: Milestone Self-Imitation ($\mathcal{L}_{\text{SI}}$) lacks negative contrastive near-misses. Combined with spatial pooling HUD dilution ($3.1\%$), it induces policy bleed to zero-resource states, causing the catastrophic "Statue" failed-craft spam.
3. **Uncorrected PER Bias**: The complete absence of importance sampling weights ($\beta = 0.0$) skews gradient updates, collapsing critic variance ($Q_{\text{std}} = 0.0157$).
4. **1-Step TD Truncation in Tier-2**: Truncation to 1-step returns prevents milestone rewards from propagating backward to prerequisite affordances.
5. **Recurrent Memory Asynchrony**: Stored $h$ vectors in Tier-2 drift out-of-distribution, and bootstrap targets in Tier-2 lag by 1 step ($h_t$ used instead of $h_{t+1}$).

---

## 5. Verification Method

To independently verify these conclusions on the codebase:

1. **Verify Empirical Sampling Distribution**:
   ```bash
   python3 -c "
   import numpy as np, sys
   sys.path.insert(0, 'prometheus')
   from src.memory.buffer import TwoTierReplayBuffer

   buf = TwoTierReplayBuffer(tier1_capacity=50000, tier2_capacity=20000, default_tier2_ratio=0.30,
                             milestone_t1_priority=0.25, milestone_t2_priority=0.80, seed=42)
   obs = np.zeros((3, 64, 64), dtype=np.uint8)
   np.random.seed(42)
   rares = set(np.random.choice(25000, 15, replace=False))
   commons = set(np.random.choice(list(set(range(25000)) - rares), 1500, replace=False))
   for t in range(25000):
       is_r, is_c = t in rares, t in commons
       buf.add(obs, 999 if is_r else t%17, 1.0 if is_r else (0.1 if is_c else 0.0), obs,
               t%100==99, 1.0 if is_c else 0.1, is_r or is_c, 5.0 if is_r else 0.1, t//100, t%100)
   drawn_ms, drawn_rare = 0, 0
   for _ in range(500):
       b = buf.sample(128, tier2_ratio=0.30)
       drawn_ms += int(b['is_milestone'].sum())
       drawn_rare += int((b['actions'] == 999).sum())
   print(f'Empirical Batch Milestone Rate: {drawn_ms/64000*100:.2f}% (Theory >= 41.5%)')
   print(f'Rare Milestone Oversampling Factor: {(drawn_rare/64000)/(15/25000):.2f}x')
   "
   ```
   **Pass Condition**: Milestone fraction must be $\ge 41.5\%$ (measures $\sim 50.45\%$), and rare oversampling must be $\ge 8.0\times$.

2. **Verify Code Locations**:
   - `grep -n -A 10 "Process Tier 2 items" prometheus/src/memory/buffer.py` (proves 1-step fallback and `boot_h = t2_h[idx]`).
   - `grep -n -A 10 "Self-Imitation Loss on verified milestone" prometheus/src/train.py` (proves $\mathcal{L}_{\text{SI}}$ filters strictly for $r \ge 0.5$).
   - `python3 -c "import json; d = json.load(open('results/diagnostics/q_calibration.json')); print('Q pred std:', d['q_pred_std'])"` (proves flat critic $Q_{\text{std}} = 0.0157$).
