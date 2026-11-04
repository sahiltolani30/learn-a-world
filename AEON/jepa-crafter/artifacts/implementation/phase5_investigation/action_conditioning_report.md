# Investigation B: Action-Conditioning Pathway Report

**Gate Status:** `ACTION_CONDITIONING_PASS` (BLOCKED)

---

## 1. Executive Summary

This investigation traced the propagation of discrete action tokens from environment interaction to future latent state prediction:
$$\text{Action Token } a_t \in \{0..16\} \longrightarrow \text{Embedding Table } E(a_t) \longrightarrow \text{FiLM MLP } \longrightarrow \text{Transformer Blocks } \longrightarrow \hat{z}_{t+1}$$

### Core Finding
The action-conditioning code path is functionally connected, non-trivial, and receives finite backpropagation gradients. However, the architectural design of global FiLM broadcast across 64 spatial tokens followed by LayerNorm dampens the action signal by several orders of magnitude, causing action prediction divergence to collapse to $9.92 \times 10^{-6}$.

---

## 2. Hypothesis Testing Matrix

| Hypothesis | Test Method | Result | Evidence |
|---|---|:---:|---|
| **H1: Action embeddings do not reach predictor** | Forward tensor trace | **REJECTED** | `cond_mlp` takes `torch.cat([h_t, a_emb], dim=-1)` directly. |
| **H2: Action conditioning is bypassed** | Forward inspection & ablation | **REJECTED** | FiLM modulation modulates tokens: `(1 + gamma) * z + beta`. |
| **H3: Gradients do not flow to action table** | PyTorch backward pass on loss | **REJECTED** | Embedding table gradient norm = $0.001668 > 0$; `cond_mlp` = $0.067009 > 0$. |
| **H4: Replay actions and next_obs misaligned** | Index-by-index replay trace | **REJECTED** | `SequenceSampler` anchor $t$ pairs $a_t$ with $O_t$ and target $O_{t+1}$ exactly. |
| **H5: Control branches share computation** | Paired counterfactual isolation | **REJECTED** | Active, zero-embedding, and shuffled branches compute independently. |
| **H6: Metric cannot detect differences** | Full-precision $L_2$ & max diff | **CONFIRMED** | Cosine distance across actions is $9.92 \times 10^{-6}$, rounding to $0.000000$ in display. |
| **H7: Global FiLM suppressed by LayerNorm** | Architectural sensitivity sweep | **CONFIRMED** | LayerNorm centers and rescales uniform channel biases across tokens. |

---

## 3. Empirical Action Sensitivity Sweep (Full Precision)

Tested on fixed context $O_t$ (seed 42) across all 17 actions:

```text
Pairwise comparisons count: 136 pairs
• Pairwise Cosine Distance (Mean): 9.9187e-06
• Pairwise Cosine Distance (Max) : 1.8869e-05
• Pairwise L2 Distance (Mean)    : 0.3986
• Pairwise Max Abs Diff (Mean)   : 0.0210
• Distance to Zero-Action Control: 4.9702e-06
```

---

## 4. Architectural Bottleneck & Remediation

### The LayerNorm Invariance Trap
In `LatentPredictorMember`:
```python
cond_input = torch.cat([context_state, action_emb], dim=-1)  # [B, 320]
film_params = self.cond_mlp(cond_input)                       # [B, 256]
gamma, beta = film_params.chunk(2, dim=-1)                   # [B, 128], [B, 128]

# Broad-cast across all 64 spatial tokens
modulated = (1.0 + gamma.unsqueeze(1)) * current_tokens + beta.unsqueeze(1)
out = self.final_norm(self.transformer(modulated))
```

1. **Spatial Invariance**: In Crafter, an action changes only 1 or 2 spatial tiles (e.g. player movement, tree chopped). Broadcasting $(\gamma, \beta)$ uniformly across all 64 patches cannot communicate *where* spatial deltas occur.
2. **LayerNorm Normalization**: Because `beta` is identical across all tokens, `nn.LayerNorm(128)` centers the features along the channel dimension, neutralizing constant bias shifts.
3. **No Sequence Action Token**: Because the Transformer sequence contains only the 64 spatial tokens, multi-head self-attention has no dedicated action token to attend to.

### Proposed Architecture Upgrade:
1. **Direct Action Residual**: Add action projection to every token: $\tilde{z}_i = z_i + W_a(a_t)$.
2. **Sequence Action Token**: Prepend an explicit action token as the 65th input token to the Transformer blocks.

---

## 5. Gate Verdict
**`ACTION_CONDITIONING_PASS` is BLOCKED.**  
While code-level connectivity is verified, empirical held-out predictive benefit over control baselines is not yet achieved on the current checkpoint.
