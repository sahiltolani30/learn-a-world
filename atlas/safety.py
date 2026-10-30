"""ContinueNet: reactive survival / risk estimator.

Estimates P(death) under candidate action a given composite state s.
Trained on replay with class-weighted focal BCE (deaths are ~2% of steps).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.config import NUM_ACTIONS, STATE_DIM, SafetyConfig


class ContinueNet(nn.Module):
    """Predicts death risk (P(death)) for state-action pairs (s, a).

    MLP(384 + 32 -> 256 -> 256 -> 1).
    """

    def __init__(self, cfg: Optional[SafetyConfig] = None):
        super().__init__()
        self.cfg = cfg or SafetyConfig()
        h_dim = self.cfg.hidden_dim
        action_emb_dim = 32

        self.action_emb = nn.Embedding(NUM_ACTIONS, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + action_emb_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def risk_logits(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Raw death logits of shape [B]."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if a.ndim == 0:
            a = a.unsqueeze(0)

        a_emb = self.action_emb(a)
        feat = torch.cat([s, a_emb], dim=-1)
        return self.net(feat).squeeze(-1)

    def risk(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """P(death) in [0, 1], shape [B]."""
        return torch.sigmoid(self.risk_logits(s, a))

    def alive_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """P(alive) in [0, 1], shape [B]."""
        return 1.0 - self.risk(s, a)

    def risk_all_actions(self, s: torch.Tensor) -> torch.Tensor:
        """Death risk for every action: shape [B, 17]."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]
        a_all = torch.arange(NUM_ACTIONS, device=s.device).unsqueeze(0).expand(B, NUM_ACTIONS)
        s_exp = s.unsqueeze(1).expand(B, NUM_ACTIONS, STATE_DIM)

        flat_s = s_exp.reshape(-1, STATE_DIM)
        flat_a = a_all.reshape(-1)

        r_flat = self.risk(flat_s, flat_a)
        return r_flat.reshape(B, NUM_ACTIONS)


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: float = 25.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal loss for binary death prediction.

    Args:
        logits: unnormalized risk logits [B].
        targets: binary ground-truth (1.0 for real death, 0.0 for survival) [B].
        weight: positive class multiplier on done / death.
        gamma: focusing parameter reducing relative loss for well-classified examples.
    """
    targets = targets.float().view_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = weight * targets + 1.0 * (1.0 - targets)
    focal_weight = alpha_t * ((1.0 - p_t) ** gamma)

    return (focal_weight * bce).mean()


def calibrate_risk_threshold(
    death_risks: Any,
    survived_risks: Any,
    grid: Optional[Any] = None,
) -> Tuple[float, Dict[str, Any]]:
    """G4a: Threshold recalibration from data maximizing deaths_flagged/total - 0.5 * false_flag_rate.
    Grid over 0.30..0.60 step 0.05. Updates threshold only if deaths_flagged >= 60%, else keeps 0.50.
    """
    import numpy as np

    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.30, 0.65, 0.05)]

    d_risks = np.asarray(death_risks, dtype=np.float32)
    s_risks = np.asarray(survived_risks, dtype=np.float32)

    total_deaths = len(d_risks)
    if total_deaths == 0:
        return 0.50, {"grid": None, "chosen": 0.50, "deaths_flagged_rate": None}

    grid_results = {}
    best_utility = -1e9
    best_tau = 0.50

    for tau in grid:
        tau_val = float(tau)
        flagged = int(np.sum(d_risks > tau_val))
        tpr = flagged / total_deaths
        false_flags = int(np.sum(s_risks > tau_val)) if len(s_risks) > 0 else 0
        fpr = false_flags / max(1, len(s_risks))
        utility = tpr - 0.5 * fpr
        grid_results[str(tau_val)] = {
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "utility": round(utility, 4),
            "deaths_flagged": flagged,
        }
        if utility > best_utility:
            best_utility = utility
            best_tau = tau_val

    flagged_share = float(np.sum(d_risks > best_tau) / total_deaths)
    if flagged_share >= 0.60:
        chosen_threshold = best_tau
    else:
        chosen_threshold = 0.50

    summary = {
        "grid": grid_results,
        "best_tau": best_tau,
        "chosen": chosen_threshold,
        "deaths_flagged_rate": round(flagged_share, 4),
    }
    return chosen_threshold, summary
