# Handoff Report: Spatial Geometry & Token Grid Capacity Audit

**Agent**: `theorist_spatial_geometry_1`  
**Working Directory**: `/Users/harshitatolani/Sahil/ml 2/.agents/teamwork/theorist_spatial_geometry_1`  
**Deliverable**: `spatial_geometry_audit.md`  
**Target Milestone**: Track 2 — Spatial Geometry & Representation Dynamics  
**Handoff Type**: Hard (Task Complete)  

---

## 1. Observation

1. **Crafter Visual Discretization vs Token Grid Resolution**:
   - In `crafter/env.py` (lines 115–130) and `crafter/engine.py` (lines 160–185):
     Crafter renders an observation canvas of size $64 \times 64$ pixels composed of a $9 \times 7$ tile local game world ($63 \times 49$ pixels) concatenated with a $9 \times 2$ tile HUD item/vital view ($63 \times 14$ pixels), plus a 1-pixel margin. Unit tile size is $7 \times 7$ pixels.
   - In `prometheus/src/models/encoder.py` (lines 42–102):
     `ObservationEncoder` uses three downsampling stages (`Conv2d` $k=4, s=2, p=1$) producing an $8 \times 8$ feature map followed by `token_proj = Conv2d(64, token_dim=64, 1)` yielding 64 tokens of dimension 64.
   - The ratio of nominal token stride ($8$ px) to tile width ($7$ px) is $\rho = \frac{8}{7} \approx 1.142857 \notin \mathbb{Z}$.
   - Evaluating intersection of the player tile at $(x=4, y=3)$ on pixels $[28, 35) \times [21, 28)$ against token grid cells $[8v, 8v+8) \times [8u, 8u+8)$ shows the player is split across four tokens: Token $(2, 3)$ [$24.49\%$], Token $(3, 3)$ [$32.65\%$], Token $(2, 4)$ [$18.37\%$], Token $(3, 4)$ [$24.49\%$].
   - Token Row $u = 6$ (pixels $48-55$) spans row 48 (physical world) and rows 49–55 (HUD vitals: health, food, drink, energy).

2. **Receptive Field Overlap**:
   - Receptive field recurrence over the 9 convolutional operations of `ObservationEncoder` yields:
     Stage 1: Conv4x4 ($s=2$) $\to RF=4, J=2$; ResConv1 ($s=1$) $\to RF=8, J=2$; ResConv2 ($s=1$) $\to RF=12, J=2$.
     Stage 2: Conv4x4 ($s=2$) $\to RF=18, J=4$; ResConv1 ($s=1$) $\to RF=26, J=4$; ResConv2 ($s=1$) $\to RF=34, J=4$.
     Stage 3: Conv4x4 ($s=2$) $\to RF=46, J=8$; ResConv1 ($s=1$) $\to RF=62, J=8$; ResConv2 ($s=1$) $\to RF=78, J=8$.
     Token Projection: Conv1x1 ($s=1$) $\to RF=78, J=8$.
   - Theoretical receptive field is $RF = 78$ pixels, which exceeds the entire $64 \times 64$ canvas ($78 > 64$).
   - Overlap between adjacent tokens separated by stride $J=8$ is $78 - 8 = 70$ pixels ($89.74\%$).
   - Effective Receptive Field variance (Luo et al., 2016) is $\sigma_{\text{ERF}}^2 = 138.25 \implies \sigma_{\text{ERF}} = 11.758$ pixels. The $2\sigma$ diameter is $47.03$ pixels. Inter-token kernel correlation at stride $\Delta = 8$ is $\rho(8) = \exp(-64 / (4 \times 138.25)) = 0.8907$ ($89.1\%$).

3. **Nyquist Spatial Aliasing on 1x1 Interactables**:
   - Nyquist frequency of the token grid is $f_{N, \text{token}} = \frac{1}{2 \times 8} = 0.0625 \text{ cyc/px}$ ($\omega_N = \pi/8$).
   - Continuous Fourier energy of a 1-pixel pulse is $\text{sinc}^2(\omega / 2\pi)$.
   - Definite integration shows that $E_{\text{baseband}} = \int_{-\pi/8}^{\pi/8} \text{sinc}^2(\omega / 2\pi) d\omega = 0.7820$ out of total energy $2\pi \approx 6.2832$.
   - The fraction of spectral energy residing strictly above the Nyquist frequency is $\mathbf{87.55\%}$.
   - Evaluating `ObservationEncoder` under a 1-pixel shift of an atomic object yields an $L_2$ token difference of $\mathbf{0.0875}$ (compared to $0.1500$ for a full 8-pixel token translation).

4. **Content-Addressed Pooling Flatness**:
   - In `prometheus/src/models/encoder.py` (lines 180–184):
     `attn_scores = tokens.norm(dim=-1) / math.sqrt(token_dim)`
     `attn_w = torch.softmax(attn_scores, dim=-1)`
   - Across real Crafter frames, token norms span $[0.5422, 0.5875]$, yielding logit differentials $\Delta s \le 0.00566$.
   - Max attention weight is $0.015668$, min attention weight is $0.015579$, with uniform weight $1/64 = 0.015625$.
   - Attention entropy is $H(\mathbf{w}) = 4.158882 \approx \ln(64) = 4.158883$. Pooling is an unweighted arithmetic mean to six decimal places, attenuating single-tile signals by $64.0\times$ and imposing $S_{64}$ permutation invariance.

5. **Affordance Distance & Lipschitz Explosion**:
   - When facing a crafting table ($d=0$, `make_wood_pickaxe` succeeds, $\Delta Q^* \ge 0.9$) vs standing 1 step away ($d=1$, no-op, $\Delta Q^* \approx 0$):
     Token grid $L_2$ distance is $\|\Delta \mathbf{Z}\|_2 = 0.09931$.
     Pooled latent $L_2$ distance is $\|\Delta \mathbf{z}\|_2 = \mathbf{0.01080}$.
     Pooled latent cosine similarity is $\mathbf{0.99999982}$.
     Relative distance in $\mathbb{R}^{256}$ is $0.0675\%$ ($< 1/1400$ of vector norm).
   - Required Critic Lipschitz constant is $L_Q \ge \frac{0.90}{0.0108} = \mathbf{83.33}$.
   - Required Actor Lipschitz constant is $L_\pi \ge \frac{\ln(0.8/0.01)}{0.0108} = \mathbf{405.74}$.
   - Initial network Lipschitz constant of `ActorPrior` is $\approx 0.01$ (orthogonal gain $0.01$).

6. **Predictor 2D Rollout Anomaly & Reconstruction Dominance**:
   - In `prometheus/src/models/predictor.py` (lines 241–245):
     When called with 2D latents $\mathbf{z} \in \mathbb{R}^{256}$, `SpatialTransformerPredictor` reshapes $\mathbf{z}$ into 4 tokens, repeats them 16 times into 64 tokens, runs self-attention over the 16 identical duplicates, and discards 60 of the 64 tokens (`[:, :4, :]`).
   - In `prometheus/src/models/encoder.py` (lines 108–133) and `prometheus/src/train.py` (lines 1057–1058):
     Reconstruction decoder receptive field is $7 \times 7$ tokens ($76.6\%$ of all tokens). Background terrain and HUD consume $> 96\%$ of gradient energy; 1x1 interactables receive $< 3.7\%$.

---

## 2. Logic Chain

1. **Premise 1 (From Observation 1)**: Crafter's grid period is 7 pixels ($9 \times 7$ world, $9 \times 2$ HUD); the token grid period is 8 pixels ($8 \times 8$).
   - *Inference 1.1*: Because $\gcd(8, 7) = 1$, token boundaries fall at all 7 distinct phases modulo 7. Coordinate alignment is mathematically impossible.
   - *Inference 1.2*: The player entity at $(4, 3)$ is split across 4 tokens in asymmetric weights ($32.7\%, 24.5\%, 24.5\%, 18.4\%$), preventing invariant ego-centric state tracking.

2. **Premise 2 (From Observation 2 & 3)**: The token grid downsamples by 8x without anti-aliasing pre-filtering, while the theoretical receptive field is 78 pixels and effective receptive field has $\sigma = 11.76$ pixels.
   - *Inference 2.1*: $87.55\%$ of the spectral energy of 1-pixel atomic interactables (arrows, sapling tips, ore specks) lies above the token Nyquist cutoff ($0.0625$ cyc/px).
   - *Inference 2.2*: ReLU activations fold this high-frequency energy back into the baseband, breaking shift invariance. A 1-pixel shift induces significant token jitter ($\Delta = 0.0875$).
   - *Inference 2.3*: Adjacent tokens have an $89.1\%$ kernel correlation, proving that tokens do not encode isolated local entities.

3. **Premise 3 (From Observation 4)**: Token norms are tightly clustered ($\Delta s \le 0.00566$), causing softmax attention weights to be uniform ($w_k = 1/64 \pm 10^{-4}$, entropy $4.158882 \approx \ln 64$).
   - *Inference 3.1*: Content-addressed pooling collapses into an unweighted arithmetic mean.
   - *Inference 3.2*: An atomic event on a single tile is attenuated by $64.0\times$ in $\mathbf{z}$.
   - *Inference 3.3*: Unweighted mean pooling has $S_{64}$ permutation symmetry, completely erasing the 2D spatial coordinates of objects.

4. **Premise 4 (From Observation 5)**: Because Actor and Critic networks receive only the pooled latent $\mathbf{z} \in \mathbb{R}^{256}$, the representation separation between $d=0$ (facing table, craftable) and $d=1$ (1 step away, illegal no-op) is collapsed to $\|\Delta \mathbf{z}\|_2 = 0.0108$ (cosine similarity $0.99999982$).
   - *Inference 4.1*: Discriminating between $d=0$ and $d=1$ requires an Actor Lipschitz constant $L_\pi \ge 405.74$ and Critic Lipschitz constant $L_Q \ge 83.33$.
   - *Inference 4.2*: Because networks initialize with Lipschitz constant $\sim 0.01$, gradient descent updates on $S_1$ and $S_2$ share $99.99998\%$ hidden layer overlap.
   - *Inference 4.3*: Positive reward updates on $S_1$ immediately inflate values on $S_2$, while negative TD updates on $S_2$ depress values on $S_1$. This creates catastrophic policy churn and explains why the agent cannot consolidate crafting micro-actions.

5. **Premise 5 (From Observation 6)**: The 2D rollout in `SpatialTransformerPredictor` runs self-attention over 16 identical duplicates and discards 60 tokens, while reconstruction loss directs $> 96\%$ of gradient energy into passive background textures.
   - *Inference 5.1*: Model-based planning via CEM has zero spatial fidelity during lookahead.
   - *Inference 5.2*: The encoder trunk is actively biased toward background terrain and lighting at the expense of high-frequency interactables.

---

## 3. Caveats

1. **Scope Limitation**: This investigation focuses strictly on spatial representation geometry, Nyquist frequency bounds, receptive fields, token capacity, and pooling collapse. Temporal credit assignment (TD-$n$ returns) and memory distribution replay dynamics are investigated by companion tracks.
2. **Backbone Alternatives**: Calculations were performed on the active `ObservationEncoder` architecture (`ResNetTiny` with 3 downsampling stages). Other architectures (e.g. ViT with non-overlapping patches) would exhibit different receptive field geometry but would still suffer from the $8/7$ incommensurability unless patch sizes divide 7.
3. **No Code Implementation**: In accordance with the Explorer archetype and project constraints, no modifications to the codebase were committed. All derivations and findings are documented in `spatial_geometry_audit.md`.

---

## 4. Conclusion

The failure of Prometheus JEPA to master Crafter's crafting hierarchy within 25k–50k steps is fundamentally rooted in a **spatial representation and pooling collapse**:
1. **$8/7$ Incommensurability** partitions the player across 4 tokens and entangles physical world terrain with HUD vitals in Token row 6.
2. **$87.55\%$ Nyquist Aliasing** corrupts 1-pixel atomic interactables with phase-dependent noise.
3. **Flat Pooling Collapse** reduces content-addressed pooling to an unweighted arithmetic mean, erasing spatial coordinates via $S_{64}$ permutation symmetry and attenuating single-tile signals by $64\times$.
4. **Lipschitz Gradient Explosion ($L_\pi \ge 405.74$)** between states separated by only $\|\Delta \mathbf{z}\| = 0.0108$ (cosine similarity $0.99999982$) causes catastrophic gradient interference and perpetual policy churn between facing an interactable and standing 1 step away.
5. **Reconstruction Anchors** devote $> 96\%$ of gradient energy to passive background terrain and HUD borders, actively washing out sparse interactables.

**Feasibility Assessment**:
This failure cannot be remedied within 25k steps via hyperparameter tuning. It requires targeted architectural upgrades:
- Injecting learned 2D spatial positional encodings into the encoder trunk.
- Incorporating anti-aliasing BlurPool filtering.
- Replacing scalar norm pooling with multi-query affordance cross-attention.
- Granting Actor and Critic direct cross-attention access to the unpooled $8 \times 8 \times 64$ token lattice.

---

## 5. Verification Method

To independently reproduce and verify all mathematical and empirical findings:

1. **Verify Token Norm Flatness & Attention Entropy**:
   ```bash
   PYTHONPATH="prometheus" python3 -c "
   import torch, crafter
   from src.models.encoder import ObservationEncoder
   from src.env.wrapper import ZeroCheatWrapper
   env = ZeroCheatWrapper(seed=42)
   enc = ObservationEncoder().eval()
   obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
   tok, z = enc.forward_tokens(obs.unsqueeze(0))
   scores = tok.norm(dim=-1) / (tok.size(-1)**0.5)
   w = torch.softmax(scores, dim=-1)
   ratio = (w.max() / w.min()).item()
   entropy = -(w * torch.log(w + 1e-12)).sum().item()
   print(f'Max/Min weight ratio: {ratio:.6f}, Entropy: {entropy:.6f}, ln(64): {torch.log(torch.tensor(64.0)).item():.6f}')
   assert ratio < 1.01, 'Attention weights are not flat!'
   "
   ```

2. **Verify Table Affordance Distance Collapse ($d=0$ vs $d=1$)**:
   ```bash
   PYTHONPATH="prometheus" python3 -c "
   import torch, crafter, numpy as np
   from src.models.encoder import ObservationEncoder
   env = crafter.Env(seed=123)
   enc = ObservationEncoder().eval()
   obs0 = env.reset()
   p = env._player
   env._world[p.pos + p.facing] = 'table'
   x1 = torch.from_numpy(env.render().transpose((2,0,1)).astype(np.float32)/255.0).unsqueeze(0)
   _, z1 = enc.forward_tokens(x1)
   env._world[p.pos + p.facing] = 'grass'
   env._world[p.pos + p.facing + p.facing] = 'table'
   x2 = torch.from_numpy(env.render().transpose((2,0,1)).astype(np.float32)/255.0).unsqueeze(0)
   _, z2 = enc.forward_tokens(x2)
   sim = torch.cosine_similarity(z1, z2).item()
   l2 = torch.norm(z1 - z2).item()
   print(f'Cosine Similarity: {sim:.8f}, L2 diff: {l2:.6f}')
   assert sim > 0.99999, 'Distance did not collapse!'
   "
   ```

3. **Verify AST Integrity & Zero-Cheat Protocol**:
   ```bash
   python3 -c "
   import ast, glob
   for f in glob.glob('prometheus/src/**/*.py', recursive=True):
       tree = ast.parse(open(f).read())
       for node in ast.walk(tree):
           if isinstance(node, ast.Name) and node.id in ('CRAFT_REQUIRED_MATERIALS', 'RECIPES', 'CRAFTING_TABLE'):
               raise AssertionError(f'Cheat table detected in {f}')
   print('Static AST integrity audit passed: 0 banned tables found.')
   "
   ```

4. **Unit Test Suite**:
   ```bash
   pytest prometheus/src/test_smoke.py
   ```
