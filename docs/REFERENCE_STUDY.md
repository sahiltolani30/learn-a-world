# Reference Architecture & Implementation Study

This document details the architectural foundation and reference designs for **Learn-a-World**, synthesized from H-JEPA (arXiv:2610.06805), AdaJEPA (NYU, Oct 2026), and LeWM / stable-worldmodel.

---

## 1. Why JEPA (Joint-Embedding Predictive Architecture)?

Traditional world models (e.g. DreamerV1-V3, Ha & Schmidhuber 2018) generate pixel-level future frames:
$$\hat{x}_{t+1} = \text{Decoder}(z_{t+1})$$
Generating pixels wastes capacity on irrelevant high-frequency visual noise (grass moving, clouds, textures) and suffers from blurring or hallucination.

**JEPA predicts exclusively in abstract representation space:**
$$\hat{z}_{t+1} = \text{Predictor}(z_t, a_t)$$
where $z_t = \text{Encoder}(x_t)$. No pixel reconstruction is ever performed.

---

## 2. Preventing Representation Collapse (The Core Challenge)

Because there is no pixel reconstruction loss, a naive network will collapse to a trivial constant representation:
$$\forall x, \quad \text{Encoder}(x) = \mathbf{c} \implies \text{Loss} = \|\mathbf{c} - \mathbf{c}\|^2 = 0$$

### The Triad Defense Pattern:
1. **Target Encoder with Stop-Gradient:**
   - Online encoder parameterized by $\theta$.
   - Target encoder parameterized by $\xi$, updated via Exponential Moving Average (EMA):
     $$\xi \leftarrow \tau \xi + (1 - \tau) \theta, \quad \tau \in [0.99, 0.999]$$
   - Target output $\bar{z}_{t+1} = \text{sg}(\text{TargetEncoder}_\xi(x_{t+1}))$. Gradients flow only through the online branch.

2. **$L_2$ Normalization / LayerNorm:**
   - Latent vectors are normalized: $z = \frac{z}{\|z\|_2}$ or passed through LayerNorm to prevent scale explosion.

3. **Latent Variance Monitoring (VICReg-style metric):**
   - For a batch $Z \in \mathbb{R}^{B \times D}$, compute per-dimension standard deviation:
     $$\sigma_j = \sqrt{\text{Var}(Z_{:, j}) + \epsilon}$$
   - Collapse Check: Ensure $\frac{1}{D} \sum_{j=1}^D \sigma_j > \text{threshold}$ (typically $> 0.05$).
   - If variance falls below threshold, learning has collapsed to a point.

---

## 3. Predictor Architecture

Following the LeWM / H-JEPA pattern:
- **Input:** $z_t \in \mathbb{R}^{D}$ concatenated with an action embedding $e(a_t) \in \mathbb{R}^{D_a}$.
- **Core:** 2-4 layer MLP with LayerNorm and GELU activations, or small 2-layer Transformer block.
- **Output:** Residual prediction $\Delta \hat{z}$, so $\hat{z}_{t+1} = z_t + \Delta \hat{z}$. Residual connection provides inductive bias that actions produce perturbations to state.
- **Loss:**
  $$\mathcal{L}_{\text{pred}} = \frac{1}{B} \sum_{i=1}^B \|\hat{z}_{t+1}^{(i)} - \bar{z}_{t+1}^{(i)}\|_2^2$$

---

## 4. Sampling-Based Latent Planning (MPC / CEM)

Planning operates directly in latent space without environment interactions:
1. Encode current observation $x_t \to z_t$.
2. Encode goal observation $x_{\text{goal}} \to z_{\text{goal}}$.
3. Sample $N$ action sequences of horizon $H$: $\mathbf{a}^{(i)} = (a_0^{(i)}, \dots, a_{H-1}^{(i)})$.
4. Roll forward iteratively through Predictor:
   $$\hat{z}_{k+1}^{(i)} = \text{Predictor}(\hat{z}_k^{(i)}, a_k^{(i)})$$
5. Score each candidate sequence by proximity to goal:
   $$S(\mathbf{a}^{(i)}) = -\|\hat{z}_H^{(i)} - z_{\text{goal}}\|_2$$
6. Select top elite sequences to update distribution (CEM) or take the argmax action sequence.
7. Execute first action $a_0^*$, receive new observation $x_{t+1}$, replan (MPC).

---

## 5. Test-Time Adaptation (AdaJEPA Pattern)

When environment dynamics change at test time:
1. Agent takes action $a_t$, receives next frame $x_{t+1}$.
2. Compares prediction $\hat{z}_{t+1}$ with actual $\bar{z}_{t+1}$.
3. Runs 1-5 gradient steps on Predictor parameters $\phi$ using a small sliding replay window (e.g., last 64 transitions):
   $$\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}_{\text{pred}}$$
4. **Key property:** Zero reward signal is needed to adapt; the model adapts purely from physics prediction errors.
