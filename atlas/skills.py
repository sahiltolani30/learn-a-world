"""OptionNet (goal-conditioned policy) + ReachabilityNet + HER loss.

Sub-goal reaching via closed-loop latent guidance.
Trained offline / on shards using Hindsight Experience Replay (HER).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.config import (
    LATENT_DIM,
    NUM_ACTIONS,
    RECURRENT_DIM,
    STATE_DIM,
    SkillsConfig,
)


class OptionNet(nn.Module):
    """Goal-conditioned policy network: pi(a | s, g).

    Encodes goal latent g (128) and conditions MLP on composite state s (384).
    """

    def __init__(self, cfg: Optional[SkillsConfig] = None):
        super().__init__()
        self.cfg = cfg or SkillsConfig()
        g_dim = self.cfg.goal_emb_dim
        h_dim = self.cfg.hidden_dim

        self.goal_encoder = nn.Sequential(
            nn.Linear(LATENT_DIM, g_dim),
            nn.LayerNorm(g_dim),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM + g_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, NUM_ACTIONS),
        )

    def logits(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns action logits of shape [..., NUM_ACTIONS]."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if g.ndim == 1:
            g = g.unsqueeze(0)
        g_emb = self.goal_encoder(g)
        feat = torch.cat([s, g_emb], dim=-1)
        return self.trunk(feat)

    def act(self, s: torch.Tensor, g: torch.Tensor, greedy: bool = True) -> torch.Tensor:
        """Selects action index [B] or scalar int if 1D."""
        logits = self.logits(s, g)
        if greedy:
            actions = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        return actions.squeeze(0) if s.ndim == 1 else actions


class ReachabilityNet(nn.Module):
    """Reachability estimator: psi(s, g) in [0, 1].

    Predicts whether goal view g is reachable from state s within the option horizon.
    """

    def __init__(self, cfg: Optional[SkillsConfig] = None):
        super().__init__()
        self.cfg = cfg or SkillsConfig()
        h_dim = self.cfg.hidden_dim

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + LATENT_DIM, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def raw_logits(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid logits of shape [...] (useful for BCEWithLogitsLoss)."""
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if g.ndim == 1:
            g = g.unsqueeze(0)
        feat = torch.cat([s, g], dim=-1)
        out = self.net(feat).squeeze(-1)
        return out

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Reachability probability in [0, 1]."""
        logits = self.raw_logits(s, g)
        return torch.sigmoid(logits)


def build_her_batch(
    episodes: Sequence[Dict[str, Union[np.ndarray, torch.Tensor]]],
    horizons: Sequence[int] = (5, 10, 20),
    negatives_per_pos: int = 1,
    device: Optional[torch.device | str] = None,
) -> Dict[str, torch.Tensor]:
    """Constructs Hindsight Experience Replay (HER) training pairs.

    Args:
        episodes: list of episode dicts containing keys:
            'z' [T, 128], 'h' [T, 256], 'actions' (or 'a') [T]
        horizons: look-ahead steps for positive future goals.
        negatives_per_pos: number of cross-episode negative goals per positive.
        device: destination torch device.

    Returns:
        dict with:
            's_pos': [N_pos, 384]
            'g_pos': [N_pos, 128]
            'actions_pos': [N_pos] int64
            's_reach': [N_reach, 384]
            'g_reach': [N_reach, 128]
            'reach_labels': [N_reach] float32
    """
    if not episodes:
        raise ValueError("Cannot build HER batch from empty episodes list")

    parsed_eps = []
    for ep in episodes:
        z = ep["z"]
        h = ep["h"]
        a = ep["actions"] if "actions" in ep else ep["a"]
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        T = len(z)
        s = np.concatenate([z, h], axis=-1)
        parsed_eps.append({"z": z, "s": s, "a": a, "T": T})

    s_pos_list: List[np.ndarray] = []
    g_pos_list: List[np.ndarray] = []
    a_pos_list: List[int] = []

    s_reach_list: List[np.ndarray] = []
    g_reach_list: List[np.ndarray] = []
    reach_label_list: List[float] = []

    num_eps = len(parsed_eps)

    for ep_idx, ep in enumerate(parsed_eps):
        T = ep["T"]
        if T < 2:
            continue
        for k in horizons:
            if k >= T:
                continue
            for t in range(0, T - k):
                goal = ep["z"][t + k]
                for tau in range(t, t + k):
                    s_pos_list.append(ep["s"][tau])
                    g_pos_list.append(goal)
                    a_pos_list.append(int(ep["a"][tau]))

                    # Reachability positive
                    s_reach_list.append(ep["s"][tau])
                    g_reach_list.append(goal)
                    reach_label_list.append(1.0)

                    # Reachability negative (cross-episode or distant temporal shift)
                    for _ in range(negatives_per_pos):
                        if num_eps > 1:
                            other_idx = (ep_idx + np.random.randint(1, num_eps)) % num_eps
                            other_ep = parsed_eps[other_idx]
                            neg_u = np.random.randint(0, other_ep["T"])
                            neg_g = other_ep["z"][neg_u]
                        else:
                            # Single-episode fallback: sample far-away or rolled latent
                            neg_u = (tau + T // 2) % T
                            neg_g = ep["z"][neg_u]

                        s_reach_list.append(ep["s"][tau])
                        g_reach_list.append(neg_g)
                        reach_label_list.append(0.0)

    if not s_pos_list:
        # Fallback empty tensors
        return {
            "s_pos": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_pos": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "actions_pos": torch.zeros((0,), dtype=torch.int64, device=device),
            "s_reach": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_reach": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "reach_labels": torch.zeros((0,), dtype=torch.float32, device=device),
        }

    out = {
        "s_pos": torch.from_numpy(np.stack(s_pos_list, axis=0)).float().to(device),
        "g_pos": torch.from_numpy(np.stack(g_pos_list, axis=0)).float().to(device),
        "actions_pos": torch.from_numpy(np.array(a_pos_list, dtype=np.int64)).to(device),
        "s_reach": torch.from_numpy(np.stack(s_reach_list, axis=0)).float().to(device),
        "g_reach": torch.from_numpy(np.stack(g_reach_list, axis=0)).float().to(device),
        "reach_labels": torch.from_numpy(np.array(reach_label_list, dtype=np.float32)).to(device),
    }
    return out


def compute_her_loss(
    option_net: OptionNet,
    reach_net: ReachabilityNet,
    her_batch_data: Dict[str, torch.Tensor],
    ce_weight: float = 1.0,
    bce_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Joint policy imitation + reachability discrimination loss.

    Loss = ce_weight * CE(policy_logits, actions_pos) + bce_weight * BCE(reach_logits, reach_labels)
    """
    s_pos = her_batch_data["s_pos"]
    g_pos = her_batch_data["g_pos"]
    a_pos = her_batch_data["actions_pos"]

    s_reach = her_batch_data["s_reach"]
    g_reach = her_batch_data["g_reach"]
    labels_reach = her_batch_data["reach_labels"]

    if s_pos.shape[0] == 0:
        zero = torch.tensor(0.0, device=s_pos.device, requires_grad=True)
        return zero, {"ce_loss": 0.0, "bce_loss": 0.0, "total_loss": 0.0}

    policy_logits = option_net.logits(s_pos, g_pos)
    ce_loss = F.cross_entropy(policy_logits, a_pos)

    reach_logits = reach_net.raw_logits(s_reach, g_reach)
    bce_loss = F.binary_cross_entropy_with_logits(reach_logits, labels_reach)

    total_loss = ce_weight * ce_loss + bce_weight * bce_loss
    metrics = {
        "ce_loss": float(ce_loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
    return total_loss, metrics
