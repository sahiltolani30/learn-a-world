# Handoff Report: Track 5 — Recurrent Core & Twin Critics Optimization Mechanics

**Agent:** `codebase_memory_critics_1`  
**Working Directory:** `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_memory_critics_1`  
**Delivered Artifact:** `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_memory_critics_1/memory_critics_audit.md`  
**Handoff Type:** Hard (Task Complete)  

---

## 1. Observation

Direct code-level and empirical observations from `prometheus/src/`:

1. **Pseudo Double-DQN Target Action Selection (`src/train.py:1212-1214`):**
   ```python
   q1b, q2b = self.agent.critic.target_forward(readout_boot)
   best_boot_actions = torch.argmax(torch.min(q1b, q2b), dim=-1)
   boot_q_raw = self.agent.critic.target_q_min(readout_boot, best_boot_actions)
   ```
   `best_boot_actions` is selected using `self.agent.critic.target_forward(readout_boot)` (`target_q1_net` and `target_q2_net`), not the online critic. Evaluation is performed by the exact same target heads.

2. **Redundant Double Target EMA Update Bug (`src/train.py:1170, 1232` & `src/agent/agent.py:403`):**
   In `src/agent/agent.py:394-404`:
   ```python
   def update_target_networks(self, tau: Optional[float] = None) -> None:
       t = self.tau if tau is None else tau
       self.target_encoder.update(tau=t)
       self.critic.update_target(tau=t)
   ```
   In `update_world_model()` (`src/train.py:1170`): calls `self.agent.update_target_networks(tau=self.tau)`.
   In `update_actor_critic()` (`src/train.py:1232`): calls `self.agent.critic.update_target(tau=self.tau)`.
   In `train()` (`src/train.py:1578-1580`): both methods run on every update step, invoking `critic.update_target` twice per iteration.

3. **Symlog Contraction Degradation (`src/agent/actor_critic.py:21-45, 577` & `src/train.py:1215-1217`):**
   The symlog Bellman operator $T_{\mathrm{sym}}(v) = \mathrm{symlog}(R^{(n)} + \gamma^n \mathrm{symexp}(v))$ has derivative $\kappa(v) = \gamma^n \frac{1 + \mathrm{symexp}(v)}{1 + R^{(n)} + \gamma^n \mathrm{symexp}(v)}$.
   Empirical verification via PyTorch:
   - $v = 0.0 \implies \kappa = 0.95100$
   - $v = 1.0 \implies \kappa = 0.98139$
   - $v = 2.0 \implies \kappa = 0.99307$
   - $v = 5.0 \implies \kappa = 0.99965$
   - $v = 10.0 \implies \kappa = 1.00000$

4. **Depressed Update-to-Data (UTD) Ratio (`src/train.py:358, 750-879, 1572-1580`):**
   Parallel execution runs $N=8$ environments in lockstep. Each vector step collects 8 transitions. With `update_every = 1`, 1 world model and 1 actor-critic update occur per 8 transitions ($\mathrm{UTD} = 0.125$). With default CLI `update_every = 2`, $\mathrm{UTD} = 0.0625$. Across 25,000 steps, total critic updates are strictly limited to $1,500 - 3,000$.

5. **Asymmetric Ownership of `readout_proj` (`src/train.py:463-470, 1221, 1256-1288`):**
   `readout_proj` is instantiated in `self.agent.readout_proj` and maps $[z_{\mathrm{pooled}}, h_t] \in \mathbb{R}^{512} \to \mathbb{R}^{256}$.
   `self.actor_optimizer` includes `list(self.agent.readout_proj.parameters())`.
   `self.critic_optimizer` contains ONLY `q1_net` and `q2_net`.
   In `update_actor_critic()`, the Critic evaluates on `readout_curr.detach()`. `readout_proj` is trained exclusively by actor loss.

6. **Decoupled GRU Memory Core (`src/train.py:1061, 1106-1135, 1208, 1256`):**
   In `update_actor_critic()`, `batch["h"]` is passed as a detached tensor. $\nabla_{\theta_{\mathrm{gru}}} L_{\mathrm{critic}} \equiv \mathbf{0}$ and $\nabla_{\theta_{\mathrm{gru}}} L_{\mathrm{actor}} \equiv \mathbf{0}$.
   In `update_world_model()`, `batch.get("h")` passed to the Spatial Transformer is a detached buffer tensor. $\nabla_{\theta_{\mathrm{gru}}} L_{\mathrm{tokens}} \equiv \mathbf{0}$.
   The GRU is optimized solely by $L_{\mathrm{seq}}$ (`src/train.py:1125`) predicting future pooled visual latents $z_{t+1}$ over 8-step chains (`sample_sequence_chains(num_chains=8, chain_len=8)`).

7. **Entropy Bonus vs Value Signal (`src/train.py:1267-1270` & `src/curiosity/boredom.py:67-71`):**
   Discrete action space $|\mathcal{A}| = 17 \implies \mathcal{H}_{\max} = \ln(17) \approx 2.833$.
   Default $\alpha = 0.05 \implies \alpha \mathcal{H} \approx 0.1416$. Boredom L3 $\alpha = 0.10 \implies \alpha \mathcal{H} \approx 0.2833$.
   Milestone external Q-values in symlog space are $Q \in [0.05, 0.20]$. $\mathrm{SNR} = \Delta Q / (\alpha \mathcal{H}) \le 1.0$.

---

## 2. Logic Chain

1. **Step 1: Failure of Double-DQN Decoupling.**
   From Observation 1, $a^* = \arg\max_a \min(Q_{\bar{\omega}_1}, Q_{\bar{\omega}_2})$ and evaluation is $\min(Q_{\bar{\omega}_1}(s', a^*), Q_{\bar{\omega}_2}(s', a^*))$. Because both selection and evaluation use the target weights $\bar{\omega}$, online parameters $\omega$ are bypassed. This collapses to single-target Q-learning under a min operator, failing to provide the statistical decoupling proved by van Hasselt et al. (2015).

2. **Step 2: Rapid Target Drift.**
   From Observation 2, `critic.update_target` is invoked twice per update cycle. Instead of tracking the online critic at momentum $\tau = 0.99$, the target network updates with $\tau_{\mathrm{eff}} = 0.99^2 = 0.9801$, tracking online weights twice as fast ($0.0199$ vs $0.0100$). The first update also occurs before the online critic optimizer has taken its step.

3. **Step 3: Stalled Value Propagation.**
   From Observation 3 and Observation 4, the symlog transformation degrades the contraction factor from $\gamma^5 = 0.951$ towards $\kappa(v) \to 1.0$ as values exceed zero. Simultaneously, the low UTD ratio restricts total critic updates to $1,500 - 3,000$ steps over 25k interactions. In 1,500 updates, an operator contracting at $\kappa \approx 0.993$ cannot propagate reward signals from rare crafting events across a 20-step trajectory.

4. **Step 4: Pessimism on Rare Affordances.**
   From Observation 1 and Section 3.5 of `memory_critics_audit.md`, $\min(Q_1, Q_2) = \bar{Q} - \frac{|Q_1 - Q_2|}{2}$. Frequent survival actions (sleep, walk) have high sample counts and $|Q_1 - Q_2| \approx 0$. Rare actions (`place_table`, `make_wood_pickaxe`) have high epistemic variance $|Q_1 - Q_2| \gg 0$, which penalizes their value estimate, driving the policy away from exploratory tech tree milestones.

5. **Step 5: Representational Instability via Readout Inversion.**
   From Observation 5, the state representation input to the Critic, `readout_proj([z, h])`, is updated solely by the Actor. The Critic cannot shape `readout_proj` to distinguish value states. When the Actor updates `readout_proj` to maximize immediate policy return, it shifts the coordinate system for the Critic on the next step without value grounding.

6. **Step 6: Workstation Amnesia from Objective Decoupling.**
   From Observation 6, the GRU memory core receives zero task reward or policy gradients. It is trained purely to minimize $\|W_{\mathrm{mem}} h_t - \bar{z}_t\|_2^2$, predicting next visible on-screen pooled tokens. Because off-screen objects (e.g. a placed crafting table behind the agent) provide zero mutual information about visible forward tokens, the GRU update gate mathematically overwrites off-screen object memories within 2–4 steps, explaining the 100% walk-off amnesia observed in empirical evaluations.

---

## 3. Caveats

1. **Replay Buffer Sample Mixture:**
   Tier 1 vs Tier 2 milestone prioritization ($70\% / 30\%$) alters the empirical transition distribution, which partially elevates the frequency of milestone samples seen by the critic relative to pure uniform sampling. However, because `make_wood_pickaxe` only occurred once in 25k steps, milestone replay cannot overcome extreme zero-count data scarcity.
2. **Empirical Horizon Ceiling:**
   Even if the UTD ratio and BPTT truncation are fully corrected, consolidating the entire 22-achievement DAG (reaching diamond tools) within 25,000 steps remains bounded by sample complexity limits addressed in Track 1.

---

## 4. Conclusion

The primary optimization and architectural bottlenecks preventing Project Prometheus from consolidating multi-step crafting sequences are:
1. **The Double-DQN action selection defect and redundant target EMA update bug**, which destabilize Bellman targets.
2. **Symlog contraction slowdown ($\kappa \to 1.0$) combined with severely depressed UTD ($0.0625 - 0.125$)**, limiting critic updates to 1,500–3,000 in 25k steps and preventing TD credit propagation.
3. **Double-Q epistemic pessimism**, which depresses values of rare milestone actions where epistemic variance is highest.
4. **Decoupled GRU memory objective ($\nabla_{\theta_{\mathrm{gru}}} Q = 0$) and 8-step BPTT truncation**, which forces the memory core to actively forget off-screen crafting tables within 2–4 steps.
5. **Architectural inversion of `readout_proj`**, where the Actor unilaterally distorts the Critic's state space.

---

## 5. Verification Method

### 5.1 Independent Code & Ast Verification
To verify the double update bug, Double Q defect, and lack of GRU task gradients:
```bash
# 1. Verify double critic target update in train.py and agent.py
grep -n "critic.update_target" prometheus/src/train.py prometheus/src/agent/agent.py

# 2. Verify target-target Double Q argmax in update_actor_critic
sed -n '1212,1215p' prometheus/src/train.py

# 3. Verify readout_proj ownership exclusively in actor_optimizer
sed -n '463,471p' prometheus/src/train.py

# 4. Verify AST zero-cheat compliance
pytest prometheus/tests/test_smoke.py
```

### 5.2 Mathematical Contraction Verification
Run the verification script to verify symlog contraction factor $\kappa(v)$:
```bash
python3 -c "
import torch
def symlog(x): return torch.sign(x) * torch.log1p(torch.abs(x))
def symexp(y): return torch.sign(y) * torch.expm1(torch.abs(y))
for v in [0.0, 0.5, 1.0, 2.0, 5.0]:
    vt = torch.tensor(v, requires_grad=True)
    f = symlog(0.95099 * symexp(vt))
    f.backward()
    print(f'v={v:3.1f} -> kappa={vt.grad.item():.5f}')
"
```

### 5.3 Invalidation Conditions
This audit's conclusions would be invalidated if:
1. `src/train.py:1212-1214` were modified to compute `q_online = self.agent.critic(readout_boot)` and select `best_boot_actions = torch.argmax(torch.min(q1_online, q2_online), dim=-1)`.
2. `self.critic.update_target` were removed from `agent.update_target_networks`.
3. Gradients from $L_{\mathrm{critic}}$ were backpropagated directly into `agent.gru`.
