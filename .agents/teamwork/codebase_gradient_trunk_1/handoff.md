# Handoff Report: Track 4 — Visual Trunk & Spatial Transformer Gradient Flow

**Agent**: `codebase_gradient_trunk_1`  
**Role**: Codebase & Gradient Flow Auditor (Trunk & Predictor)  
**Deliverable Artifact**: `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/codebase_gradient_trunk_1/encoder_predictor_gradient_audit.md`  
**Target Recipient**: Project Orchestrator (`7af1422f-94e2-4a53-9321-d03a1d49141f`) & Synthesis Council  

---

## 1. Observation

Direct code inspections and empirical tensor evaluations were conducted on `prometheus/src/models/encoder.py`, `prometheus/src/models/predictor.py`, `prometheus/src/models/target.py`, `prometheus/src/models/losses.py`, and `prometheus/src/train.py`, as well as checkpoint `learn-a-world/sapna/checkpoints_v1/state_step_25008.pt` (25,008 steps on Crafter):

1. **Destructive Gradient Antagonism between Dynamics and Reconstruction**:
   - In `train.py:1051-1156`, both `l_tokens` (masked spatial prediction) and `l_recon` (pixel autoencoding) backpropagate simultaneously into the shared 3-stage convolutional trunk (`stage1`, `stage2`, `stage3`, `token_proj`).
   - Evaluating individual parameter gradients on the shared trunk:
     - $\|\nabla_{\theta_{\text{trunk}}} \mathcal{L}_{\text{tokens}}\| = \mathbf{8.446443}$
     - $\|\nabla_{\theta_{\text{trunk}}} \mathcal{L}_{\text{recon}}\| = \mathbf{0.023970}$
     - Ratio: Dynamics gradient is **$352.4\times$ larger** than the reconstruction anchor.
   - Cosine similarity between gradient vectors across all 270,368 shared trunk parameters:
     - $\cos(\nabla \mathcal{L}_{\text{tokens}}, \nabla \mathcal{L}_{\text{recon}}) = \mathbf{-0.9169}$ (near-180° anti-parallel).
   - In addition, variance and covariance penalties on the current pooled latent directly fight each other:
     - $\cos(\nabla \text{Var}(z_t), \nabla \text{Cov}(z_t)) = \mathbf{-0.9999}$ (exact 180° antiparallel opposition).

2. **Residual Skip-Connection Action Corruption**:
   - In `predictor.py:187-211`:
     ```python
     tokens = tokens + a_broadcast.unsqueeze(1)                        # Line 187
     tokens = tokens * (1.0 + gamma_f).unsqueeze(1) + beta_f.unsqueeze(1) # Line 190
     ...
     all_preds = torch.stack([tokens + head(spatial_out) for head in self.heads], dim=0) # Line 211
     ```
   - In Crafter transitions, the ground-truth token change has magnitude $\|s_{t+1} - s_t\|_2 = \mathbf{0.0404}$.
   - The action perturbation injected into the skip connection has magnitude $\|a_{\text{broadcast}}\|_2 = \mathbf{0.8692}$, $\|\beta\|_2 = \mathbf{0.8713}$, and $\|\gamma\|_2 = \mathbf{0.8358}$ ($\sim 1.23$, which is **$30.4\times$ larger** than the true transition delta).
   - Evaluating prediction error on `state_step_25008.pt`:
     - Trivial persistence baseline ($\hat{s}_{t+1} = s_t$): $\text{MSE} = \mathbf{6.981 \times 10^{-5}}$.
     - Trained Spatial Transformer Predictor: $\text{MSE} = \mathbf{6.121 \times 10^{-4}}$ (**$8.77\times$ worse** than persistence).

3. **Content-Addressed Pooling Jacobian Ill-Conditioning**:
   - In `encoder.py:181-184`:
     ```python
     attn_scores = tokens.norm(dim=-1) / math.sqrt(token_dim) # (B, 64)
     attn_w = torch.softmax(attn_scores, dim=-1)              # (B, 64)
     pooled_token = (attn_w.unsqueeze(-1) * tokens).sum(dim=1)
     z = self.ln(self.pooled_proj(pooled_token))
     ```
   - On Crafter frames, token norms span $[0.6233, 2.7766]$, producing attention weights in $[0.0133, 0.0174]$ (max-to-min ratio only $1.29\times$, near-uniform average).
   - SVD spectrum of the input-output Jacobian $J_{\text{pool}} = \frac{\partial z}{\partial S} \in \mathbb{R}^{256 \times 4096}$:
     - Singular values: $s_{\max} = 1.2651$, $s_{\min} = 0.000000$, median singular value = **$0.0000$**.
     - Condition number: $\kappa = \mathbf{3.57 \times 10^8}$.
     - Effective rank (Participation Ratio): **$59.25$ out of $256$** ($76.9\%$ of dimensions are dead null spaces).

4. **Receptive Field Over-Spread and Spatial Token Redundancy**:
   - In `encoder.py:77-95`, 3 sequential stages with ResBlocks produce a theoretical receptive field of **$78 \times 78$ pixels** on a $64 \times 64$ canvas.
   - Input pixel gradient support for individual corner tokens covers **1,772 out of 4,096 pixels ($43.3\%$)**, spreading $27.7\times$ beyond the nominal $8\times 8$ tile area.
   - On `state_step_25008.pt`, adjacent token cosine similarity is **$0.9936$** (horizontal) and **$0.9820$** (vertical), confirming that spatial tokens are copies of a global average rather than localized features.

5. **Epistemic Diversity Collapse via Mean Pooled Supervision**:
   - In `train.py:1086-1102`, `pred_r` and `pred_d_logits` are supervised on `mean_pred_pooled = encoder.pool_tokens(all_pred_tokens.mean(dim=0))`.
   - Gradient norms on all 4 ensemble heads under $\mathcal{L}_{\text{reward}}$:
     - `heads.0.weight` through `heads.3.weight`: $\|\nabla\| = \mathbf{0.829856}$ (exact match).
     - `heads.0.bias` through `heads.3.bias`: $\|\nabla\| = \mathbf{0.144148}$ (exact match).

6. **Mask Dropping 65% of Dynamic Affordances**:
   - In `train.py:1074-1082`, uniform 35% masking is combined with dynamic-tile weights: $\text{eff\_mask} = \text{mask} \times \text{weights}$.
   - For changing tiles not captured in the random mask, $\text{eff\_mask} = 0.0 \times 5.0 = 0.0$.
   - On average, $\mathbb{E}[|D \setminus M|] = 8 \times 0.65 = \mathbf{5.2 \text{ out of 8 dynamic tiles}}$ receive zero prediction loss gradient on every training step.

---

## 2. Logic Chain

1. **Step 1 (From Observation 1)**: The dynamics prediction loss $\mathcal{L}_{\text{tokens}}$ seeks to minimize prediction error with respect to the Polyak target $y_{t+1} = E_{\theta_{\text{target}}}(x_{t+1})$. Because dynamics in Crafter are complex and stochastic, the easiest mathematical solution to $\min \|\hat{s}_{t+1} - s_{t+1}^{\text{target}}\|^2$ is the trivial representation where all observations map to a constant vector $c$, making next-step prediction trivially zero.
2. **Step 2 (From Observation 1)**: The pixel reconstruction loss $\mathcal{L}_{\text{recon}}$ seeks to preserve pixel contrasts, textures, and spatial diversity. Therefore, its gradient directly opposes the representation-smoothing gradient of $\mathcal{L}_{\text{tokens}}$, resulting in $\cos(\nabla \mathcal{L}_{\text{tokens}}, \nabla \mathcal{L}_{\text{recon}}) = -0.9169$.
3. **Step 3 (From Observation 1)**: Because $\|\nabla \mathcal{L}_{\text{tokens}}\| = 8.446$ is $352.4\times$ larger than $\|\nabla \mathcal{L}_{\text{recon}}\| = 0.024$, the net gradient step updates the encoder in the direction dictated by dynamics smoothing. The reconstruction anchor is completely overwhelmed, causing the visual representation to collapse into a scene-invariant field (as measured in the 25k diagnostic quad: scene cosine $0.9954-0.9995$).
4. **Step 4 (From Observation 2)**: In `SpatialTransformerPredictor`, overwriting `tokens` with action broadcast and FiLM before the residual connection forces the heads to predict the inverse of the action perturbation. Because the perturbation ($\sim 1.23$) is $30\times$ larger than the actual environment transition delta ($0.0404$), the heads fail to cancel it across all 62 static tiles, resulting in prediction MSE that is $8.77\times$ worse than simple persistence.
5. **Step 5 (From Observation 3)**: Content-addressed pooling has an effective rank of $59.25$ and condition number $3.57 \times 10^8$. Any value or policy signal backpropagating from the critic or actor is blocked across $77\%$ of the 256 latent dimensions, severely restricting policy learning capacity.
6. **Step 6 (From Observation 5 & 6)**: Ensemble head averaging forces all 4 heads into the same trajectory (collapsing epistemic disagreement to $0.0002$), while random masking drops $65\%$ of dynamic transitions. The agent cannot explore via curiosity and cannot learn crafting affordances.

---

## 3. Caveats

- **Scope Boundary**: This audit investigated the visual trunk, predictor, target encoder, and JEPA loss functions. It did not audit the internal hidden state transitions of the GRU memory core across long sequences or the dual-store replay buffer sampling distributions (these belong to Tracks 5 and 6).
- **Architecture Lineage**: Empirical measurements were executed on the authoritative 25k checkpoint `learn-a-world/sapna/checkpoints_v1/state_step_25008.pt` and verified against the current `v3.2-w1b` repository codebase. While `state_step_25008.pt` used `recon_head` (linear) instead of `recon_decoder` (conv), both were verified to exhibit the same gradient magnitude suppression ($< 0.065$ vs $8.45$).

---

## 4. Conclusion

The primary failure of the Prometheus JEPA world model on Crafter is **NOT** a lack of model capacity, but a set of **destructive optimization conflicts and residual wiring flaws**:
1. **Perception-Dynamics Entanglement**: Backpropagating the dynamics prediction loss into the visual encoder trunk destroys spatial representations due to a $352:1$ gradient dominance with $\cos = -0.9169$ anti-parallel interference against the reconstruction anchor.
2. **Corrupted Residual Skip Connection**: Reassigning `tokens` to FiLM-modulated values in `predictor.py:187-211` destroys the residual identity property, degrading dynamics MSE to $8.77\times$ worse than persistence.
3. **Pooling Rank Bottleneck**: Content-addressed pooling nullifies $77\%$ of the latent space ($\kappa = 3.57 \times 10^8$, rank $59 / 256$).
4. **Receptive Field Over-Spread**: $78\times 78$ pixel receptive fields cause adjacent tokens to be $0.9936$ identical, eliminating spatial locality.

These findings directly inform the actionable remediation blueprint specified in Section 6 of `encoder_predictor_gradient_audit.md`.

---

## 5. Verification Method

To independently reproduce and verify all quantitative findings:

1. **Verify Gradient Interference & Norm Hierarchy**:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'prometheus')
   import torch, torch.nn.functional as F, crafter, numpy as np
   from src.models.encoder import ObservationEncoder
   from src.models.predictor import SpatialTransformerPredictor, sample_token_mask
   from src.models.target import TargetEncoder

   ckpt = torch.load('learn-a-world/sapna/checkpoints_v1/state_step_25008.pt', map_location='cpu', weights_only=False)
   encoder = ObservationEncoder(); encoder.load_state_dict(ckpt['encoder'], strict=False)
   predictor = SpatialTransformerPredictor(); predictor.load_state_dict(ckpt['predictor'], strict=True)
   target_encoder = TargetEncoder(encoder, tau=0.99)

   env = crafter.Env(seed=42)
   obs = env.reset(); obs_list, next_list, a_list = [], [], []
   for _ in range(16):
       a = np.random.randint(0, 17); n_obs, r, d, _ = env.step(a)
       obs_list.append(torch.from_numpy(obs).permute(2,0,1).float()/255.0)
       next_list.append(torch.from_numpy(n_obs).permute(2,0,1).float()/255.0)
       a_list.append(a); obs = env.reset() if d else n_obs
   b_obs, b_next, b_a = torch.stack(obs_list), torch.stack(next_list), torch.tensor(a_list)

   tokens, _ = encoder.forward_tokens(b_obs)
   with torch.no_grad(): t_tokens, _ = target_encoder.forward_tokens(b_next)
   recon = encoder.decode_tokens(tokens); l_recon = F.mse_loss(recon, b_obs)
   _, all_preds = predictor.forward_tokens(tokens, b_a)
   l_tokens = predictor.compute_masked_loss(all_preds, t_tokens, sample_token_mask(16, 64, 0.35))

   trunk_params = [p for n, p in encoder.named_parameters() if any(s in n for s in ['stage1','stage2','stage3','token_proj'])]
   encoder.zero_grad(); l_tokens.backward(retain_graph=True); g_tok = torch.cat([p.grad.view(-1) for p in trunk_params])
   encoder.zero_grad(); l_recon.backward(); g_rec = torch.cat([p.grad.view(-1) for p in trunk_params])
   print(f'l_tokens trunk grad norm: {g_tok.norm().item():.4f}')
   print(f'l_recon trunk grad norm:  {g_rec.norm().item():.4f}')
   print(f'Cosine similarity:        {F.cosine_similarity(g_tok.unsqueeze(0), g_rec.unsqueeze(0)).item():.4f}')
   "
   ```
   *Expected Output*: `l_tokens trunk grad norm ~ 8.45`, `l_recon trunk grad norm ~ 0.024`, `Cosine similarity ~ -0.917`.

2. **Verify Predictor MSE vs Persistence Ratio**:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'prometheus')
   import torch, crafter, numpy as np
   from src.models.encoder import ObservationEncoder
   from src.models.predictor import SpatialTransformerPredictor
   from src.models.target import TargetEncoder

   ckpt = torch.load('learn-a-world/sapna/checkpoints_v1/state_step_25008.pt', map_location='cpu', weights_only=False)
   encoder = ObservationEncoder(); encoder.load_state_dict(ckpt['encoder'], strict=False)
   predictor = SpatialTransformerPredictor(); predictor.load_state_dict(ckpt['predictor'], strict=True)
   target_encoder = TargetEncoder(encoder, tau=0.99)

   env = crafter.Env(seed=42); obs = env.reset(); a = np.random.randint(0, 17); n_obs, _, _, _ = env.step(a)
   x = torch.from_numpy(obs).permute(2,0,1).unsqueeze(0).float()/255.0
   x_n = torch.from_numpy(n_obs).permute(2,0,1).unsqueeze(0).float()/255.0
   with torch.no_grad():
       tok, _ = encoder.forward_tokens(x); t_tok, _ = target_encoder.forward_tokens(x_n)
       pred, _ = predictor.forward_tokens(tok, torch.tensor([a]))
   mse_persistence = ((tok - t_tok)**2).mean().item()
   mse_pred = ((pred - t_tok)**2).mean().item()
   print(f'Persistence MSE: {mse_persistence:.6e}, Predictor MSE: {mse_pred:.6e}, Ratio: {mse_pred / mse_persistence:.2f}x')
   "
   ```
   *Expected Output*: `Persistence MSE ~ 6.98e-05`, `Predictor MSE ~ 6.12e-04`, `Ratio ~ 8.77x`.

3. **Invalidation Conditions**:
   - The primary conclusion is invalidated if $\cos(\nabla \mathcal{L}_{\text{tokens}}, \nabla \mathcal{L}_{\text{recon}}) \ge 0.0$.
   - The predictor skip-corruption finding is invalidated if the trained predictor MSE is strictly lower than persistence MSE on held-out Crafter steps.
   - The pooling ill-conditioning finding is invalidated if the Jacobian condition number is $< 100.0$.
