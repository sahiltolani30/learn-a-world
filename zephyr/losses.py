"""Zephyr loss primitives: symlog two-hot coding, lambda-returns, regularizers.

Pure torch utilities shared by the world model, heads, and actor-critic
training. No model or training-loop code lives here.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "symlog",
    "inv_symlog",
    "TwoHotCoder",
    "lambda_returns",
    "vicreg_var_loss",
    "vicreg_cov_loss",
    "focal_bce_with_logits",
    "masked_mse",
]


def symlog(x: Tensor) -> Tensor:
    """sign(x) * log(1 + |x|): symmetric log squash used by two-hot targets."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inv_symlog(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotCoder:
    """Symlog-space two-hot distribution coder (DreamerV3-style).

    Bin centers are uniformly spaced in symlog space over
    ``[-symlog_range, symlog_range]``. A scalar is encoded by splitting its
    probability mass linearly (in symlog space) between the two nearest centers,
    so interpolating the centers reproduces the symlog value exactly. Values
    whose symlog falls outside the range are clamped to the edge bins. This is
    intentionally not an ``nn.Module``: it holds no learnable state, and its
    centers are moved to the input device/dtype on the fly.
    """

    def __init__(self, symlog_range: float = 20.0, num_bins: int = 65) -> None:
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if symlog_range <= 0.0:
            raise ValueError(f"symlog_range must be > 0, got {symlog_range}")
        self.symlog_range = float(symlog_range)
        self.num_bins = int(num_bins)
        self.bin_centers = torch.linspace(
            -self.symlog_range, self.symlog_range, self.num_bins
        )

    def _centers(self, like: Tensor) -> Tensor:
        """Bin centers moved to the device/dtype of ``like``."""
        return self.bin_centers.to(device=like.device, dtype=like.dtype)

    def encode(self, values: Tensor) -> Tensor:
        """[...] real values -> [..., num_bins] two-hot distribution (sums to 1)."""
        centers = self._centers(values)
        step = centers[1] - centers[0]
        x = symlog(values.to(centers.dtype)).clamp(-self.symlog_range, self.symlog_range)
        pos = (x - centers[0]) / step
        low_f = pos.floor().clamp(0.0, float(self.num_bins - 1))
        frac = (pos - low_f).clamp(0.0, 1.0)
        low = low_f.long()
        high = (low + 1).clamp(max=self.num_bins - 1)
        probs = torch.zeros(
            (*values.shape, self.num_bins), device=values.device, dtype=centers.dtype
        )
        # scatter_add handles exact bin hits (low == high) by summing to 1.0.
        probs.scatter_add_(-1, low.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
        probs.scatter_add_(-1, high.unsqueeze(-1), frac.unsqueeze(-1))
        return probs

    def loss(self, logits: Tensor, values: Tensor) -> Tensor:
        """Mean cross-entropy between softmax(logits) and encode(values). Scalar."""
        target = self.encode(values.to(logits.dtype))
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()

    def decode(self, logits: Tensor) -> Tensor:
        """[..., num_bins] logits -> [...] real values via softmax-weighted centers."""
        centers = self._centers(logits)
        expected_symlog = (torch.softmax(logits, dim=-1) * centers).sum(dim=-1)
        return inv_symlog(expected_symlog)


def lambda_returns(
    rewards: Tensor,
    continues: Tensor,
    values: Tensor,
    bootstrap: Tensor,
    gamma: float = 0.997,
    td_lambda: float = 0.95,
) -> Tensor:
    """Dreamer lambda-returns, computed backward over [T, B] tensors.

    returns[t] = r_t + gamma * c_t * ((1 - td_lambda) * V_{t+1}
                                      + td_lambda * returns[t+1])
    where V_{t<T} comes from ``values``, the terminal V_T is ``bootstrap`` [B],
    and ``continues`` c_t are per-step continuation factors in [0, 1].
    td_lambda=1 gives discounted Monte-Carlo returns; td_lambda=0 gives
    one-step TD targets.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [T, B], got shape {tuple(rewards.shape)}")
    if values.shape != rewards.shape or continues.shape != rewards.shape:
        raise ValueError("rewards/continues/values must all be [T, B]")
    next_values = torch.cat([values[1:], bootstrap[None].to(values.dtype)], dim=0)
    discount = gamma * continues
    inputs = rewards + discount * next_values * (1.0 - td_lambda)
    returns: List[Tensor] = []
    agg = bootstrap.to(values.dtype)
    for t in reversed(range(rewards.shape[0])):
        agg = inputs[t] + discount[t] * td_lambda * agg
        returns.append(agg)
    returns.reverse()
    return torch.stack(returns, dim=0)


def vicreg_var_loss(z: Tensor, gamma: float = 1.0) -> Tensor:
    """Anti-collapse variance hinge: mean over dims of relu(gamma - std_d).

    Uses the biased std (unbiased=False) so a single-row batch yields 0 instead
    of nan. ``z`` is flattened to [N, D] first.
    """
    flat = z.reshape(-1, z.shape[-1])
    std = flat.std(dim=0, unbiased=False)
    return F.relu(gamma - std).mean()


def vicreg_cov_loss(z: Tensor) -> Tensor:
    """Anti-collapse covariance term: mean squared off-diagonal covariance.

    ``z`` is flattened to [N, D]; the covariance is normalized by N - 1 and the
    mean runs over the D * (D - 1) off-diagonal entries.
    """
    flat = z.reshape(-1, z.shape[-1])
    n, d = flat.shape
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = centered.t() @ centered / max(n - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / max(d * (d - 1), 1)


def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    pos_weight: float = 25.0,
    gamma: float = 2.0,
) -> Tensor:
    """Class-weighted focal BCE-with-logits, reduced to a scalar.

    Per-element loss is the pos_weight-weighted BCE modulated by (1 - p_t)^gamma
    where p_t = sigmoid(logits) for positives and 1 - sigmoid(logits) for
    negatives. ``targets`` are float {0, 1}, any shape broadcastable to logits.
    """
    targets = targets.to(dtype=logits.dtype)
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    modulation = (1.0 - p_t).pow(gamma)
    return (modulation * bce).mean()


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over elements where ``mask`` is truthy.

    ``mask`` must broadcast to ``pred``'s shape. Returns a 0.0 tensor (never
    nan) when no element is valid.
    """
    diff2 = (pred - target).pow(2)
    m = mask
    while m.dim() < diff2.dim():
        m = m.unsqueeze(-1)
    m = torch.broadcast_to(m, diff2.shape).to(dtype=diff2.dtype)
    denom = m.sum().clamp(min=1.0)
    return (diff2 * m).sum() / denom


def roc_auc(scores: Tensor, targets: Tensor) -> float:
    """Rank-based ROC-AUC with tie handling; 0.5 when one class is absent."""
    s = scores.detach().float().reshape(-1)
    y = targets.detach().float().reshape(-1)
    pos_mask = y > 0.5
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(s)
    sorted_s = s[order]
    ranks = torch.empty(sorted_s.shape[0], dtype=torch.float32)
    i = 0
    n = sorted_s.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[i : j + 1] = (i + j + 2) / 2.0  # 1-based average rank of the tie group
        i = j + 1
    # ranks are in sorted order; map back: rank of element at order[k] is ranks[k]
    sorted_y = y[order]
    sum_pos_ranks = float(ranks[sorted_y > 0.5].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
