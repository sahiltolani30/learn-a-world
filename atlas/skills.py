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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


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
    horizons: Sequence[int] = (5, 10),
    negatives_per_pos: int = 1,
    max_samples: int = 32_768,
    her_reward_weight_max: float = 4.0,
    her_unlock_weight: float = 2.0,
    progress_weight_gain: float = 3.0,
    critic: Optional[Any] = None,
    critic_gamma: float = 0.97,
    device: Optional[torch.device | str] = None,
) -> Dict[str, torch.Tensor]:
    """Constructs Hindsight Experience Replay (HER) training pairs.

    Args:
        episodes: list of episode dicts containing keys:
            'z' [T, 128], 'h' [T, 256], 'actions' (or 'a') [T],
            'rewards' (or 'r') [T], 'events' [T, 22]
        horizons: look-ahead steps for positive future goals.
        negatives_per_pos: number of cross-episode negative goals per positive.
        max_samples: maximum number of transitions to build when episode count is large.
        her_reward_weight_max: cap on return-based imitation weight.
        her_unlock_weight: multiplier for segments containing event unlocks.
        device: destination torch device.

    Returns:
        dict with:
            's_pos': [N_pos, 384]
            'g_pos': [N_pos, 128]
            'actions_pos': [N_pos] int64
            'weights_pos': [N_pos] float32
            'is_unlock_pos': [N_pos] bool
            's_reach': [N_reach, 384]
            'g_reach': [N_reach, 128]
            'reach_labels': [N_reach] float32
    """
    if not episodes:
        raise ValueError("Cannot build HER batch from empty episodes list")

    parsed_eps = []
    for ep in episodes:
        assert ("rewards" in ep or "r" in ep), "episodes must contain rewards ('rewards' or 'r')"
        assert "events" in ep, "episodes must contain events"

        z = ep["z"]
        h = ep["h"]
        a = ep["actions"] if "actions" in ep else ep["a"]
        r = ep["rewards"] if "rewards" in ep else ep["r"]
        events = ep["events"]

        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().numpy()
        if isinstance(events, torch.Tensor):
            events = events.detach().cpu().numpy()

        T = len(z)
        if T >= 2:
            s = np.concatenate([z, h], axis=-1)
            parsed_eps.append({"z": z, "s": s, "a": a, "r": r, "events": events, "T": T})

    s_pos_list: List[np.ndarray] = []
    g_pos_list: List[np.ndarray] = []
    a_pos_list: List[int] = []
    w_pos_list: List[float] = []
    unlock_pos_list: List[bool] = []

    s_reach_list: List[np.ndarray] = []
    g_reach_list: List[np.ndarray] = []
    reach_label_list: List[float] = []

    num_eps = len(parsed_eps)
    max_h = max(horizons)
    far_threshold = 2 * max_h

    if num_eps > 50:
        mean_k = max(1, int(np.mean(horizons)))
        samples_per_ep = max(2, min(50, (max_samples * 2) // (num_eps * mean_k)))
        for ep_idx, ep in enumerate(parsed_eps):
            T = ep["T"]
            valid_ks = [k for k in horizons if k < T]
            if not valid_ks:
                continue
            for _ in range(samples_per_ep):
                k = int(np.random.choice(valid_ks))
                t = int(np.random.randint(0, T - k))
                goal = ep["z"][t + k]

                # Progress-weighted HER (V2.2 F4)
                cos_start = _cosine(ep["z"][t], goal)
                cos_end = 1.0  # g = ep["z"][t+k], so cos_end = 1.0
                progress = cos_end - cos_start
                w_progress = 1.0 + progress_weight_gain * max(0.0, progress)

                seg_return = float(np.sum(ep["r"][t:t + k]))
                w_reward_base = (
                    (1.0 + min(max(seg_return, 0.0), her_reward_weight_max))
                    * (her_unlock_weight if np.any(ep["events"][t:t + k]) else 1.0)
                    * w_progress
                )
                w_adv = 1.0
                if critic is not None:
                    discounts = critic_gamma ** np.arange(k)
                    seg_return_discounted = float(np.sum(ep["r"][t:t + k] * discounts))
                    with torch.no_grad():
                        s_t_tensor = torch.from_numpy(ep["s"][t]).float().to(device if device else "cpu")
                        v_st = float(critic.value(s_t_tensor))
                    A_seg = seg_return_discounted - v_st
                    w_adv = float(np.exp(np.clip(A_seg / 3.0, -2.0, 2.0)))
                w_reward = w_reward_base * w_adv
                is_unlock = bool(np.any(ep["events"][t:t + k]))

                for tau in range(t, t + k):
                    s_pos_list.append(ep["s"][tau])
                    g_pos_list.append(goal)
                    a_pos_list.append(int(ep["a"][tau]))
                    w_pos_list.append(w_reward)
                    unlock_pos_list.append(is_unlock)

                    # Reachability positive
                    s_reach_list.append(ep["s"][tau])
                    g_reach_list.append(goal)
                    reach_label_list.append(1.0)

                    for _ in range(negatives_per_pos):
                        other_idx = (ep_idx + int(np.random.randint(1, num_eps))) % num_eps
                        other_ep = parsed_eps[other_idx]
                        neg_u = int(np.random.randint(0, other_ep["T"]))
                        neg_g = other_ep["z"][neg_u]
                        s_reach_list.append(ep["s"][tau])
                        g_reach_list.append(neg_g)
                        reach_label_list.append(0.0)
    else:
        # Small episode batches (e.g. unit tests or local smoke)
        for ep_idx, ep in enumerate(parsed_eps):
            T = ep["T"]
            for k in horizons:
                if k >= T:
                    continue
                for t in range(0, T - k):
                    goal = ep["z"][t + k]

                    # Progress-weighted HER (V2.2 F4)
                    cos_start = _cosine(ep["z"][t], goal)
                    cos_end = 1.0  # g = ep["z"][t+k], so cos_end = 1.0
                    progress = cos_end - cos_start
                    w_progress = 1.0 + progress_weight_gain * max(0.0, progress)

                    seg_return = float(np.sum(ep["r"][t:t + k]))
                    w_reward_base = (
                        (1.0 + min(max(seg_return, 0.0), her_reward_weight_max))
                        * (her_unlock_weight if np.any(ep["events"][t:t + k]) else 1.0)
                        * w_progress
                    )
                    w_adv = 1.0
                    if critic is not None:
                        discounts = critic_gamma ** np.arange(k)
                        seg_return_discounted = float(np.sum(ep["r"][t:t + k] * discounts))
                        with torch.no_grad():
                            s_t_tensor = torch.from_numpy(ep["s"][t]).float().to(device if device else "cpu")
                            v_st = float(critic.value(s_t_tensor))
                        A_seg = seg_return_discounted - v_st
                        w_adv = float(np.exp(np.clip(A_seg / 3.0, -2.0, 2.0)))
                    w_reward = w_reward_base * w_adv
                    is_unlock = bool(np.any(ep["events"][t:t + k]))

                    for tau in range(t, t + k):
                        s_pos_list.append(ep["s"][tau])
                        g_pos_list.append(goal)
                        a_pos_list.append(int(ep["a"][tau]))
                        w_pos_list.append(w_reward)
                        unlock_pos_list.append(is_unlock)

                        # Reachability positive
                        s_reach_list.append(ep["s"][tau])
                        g_reach_list.append(goal)
                        reach_label_list.append(1.0)

                        for _ in range(negatives_per_pos):
                            if num_eps > 1:
                                other_idx = (ep_idx + int(np.random.randint(1, num_eps))) % num_eps
                                other_ep = parsed_eps[other_idx]
                                neg_u = int(np.random.randint(0, other_ep["T"]))
                                neg_g = other_ep["z"][neg_u]
                            else:
                                neg_u = (tau + T // 2) % T
                                neg_g = ep["z"][neg_u]
                            s_reach_list.append(ep["s"][tau])
                            g_reach_list.append(neg_g)
                            reach_label_list.append(0.0)

    n_pairs = len(s_pos_list)
    reach_stride = 1 + negatives_per_pos
    if n_pairs > max_samples:
        keep_idx = np.random.choice(n_pairs, size=max_samples, replace=False)
        s_pos_list = [s_pos_list[i] for i in keep_idx]
        g_pos_list = [g_pos_list[i] for i in keep_idx]
        a_pos_list = [a_pos_list[i] for i in keep_idx]
        w_pos_list = [w_pos_list[i] for i in keep_idx]
        unlock_pos_list = [unlock_pos_list[i] for i in keep_idx]

        reach_keep_idx = []
        for i in keep_idx:
            for j in range(reach_stride):
                reach_keep_idx.append(reach_stride * i + j)
        s_reach_list = [s_reach_list[j] for j in reach_keep_idx]
        g_reach_list = [g_reach_list[j] for j in reach_keep_idx]
        reach_label_list = [reach_label_list[j] for j in reach_keep_idx]

    if not s_pos_list:
        # Fallback empty tensors
        return {
            "s_pos": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_pos": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "actions_pos": torch.zeros((0,), dtype=torch.int64, device=device),
            "weights_pos": torch.zeros((0,), dtype=torch.float32, device=device),
            "is_unlock_pos": torch.zeros((0,), dtype=torch.bool, device=device),
            "s_reach": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_reach": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "reach_labels": torch.zeros((0,), dtype=torch.float32, device=device),
        }

    out = {
        "s_pos": torch.from_numpy(np.stack(s_pos_list, axis=0)).float().to(device),
        "g_pos": torch.from_numpy(np.stack(g_pos_list, axis=0)).float().to(device),
        "actions_pos": torch.from_numpy(np.array(a_pos_list, dtype=np.int64)).to(device),
        "weights_pos": torch.from_numpy(np.array(w_pos_list, dtype=np.float32)).to(device),
        "is_unlock_pos": torch.from_numpy(np.array(unlock_pos_list, dtype=bool)).to(device),
        "s_reach": torch.from_numpy(np.stack(s_reach_list, axis=0)).float().to(device),
        "g_reach": torch.from_numpy(np.stack(g_reach_list, axis=0)).float().to(device),
        "reach_labels": torch.from_numpy(np.array(reach_label_list, dtype=np.float32)).to(device),
    }
    return out


def build_validation_reach_pairs(
    val_eps: Sequence[Dict[str, Union[np.ndarray, torch.Tensor]]],
    horizons: Sequence[int] = (5, 10),
    max_samples: int = 4096,
    device: Optional[torch.device | str] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Builds validation reachability pairs for easy and hard (50/50 mixed) AUROC evaluation.

    Returns:
        (easy_dict, hard_dict) each with 's_reach', 'g_reach', 'reach_labels'.
    """
    if not val_eps:
        empty = {
            "s_reach": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_reach": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "reach_labels": torch.zeros((0,), dtype=torch.float32, device=device),
        }
        return empty, empty

    parsed_eps = []
    for ep in val_eps:
        z = ep["z"]
        h = ep["h"]
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        T = len(z)
        if T >= 2:
            s = np.concatenate([z, h], axis=-1)
            parsed_eps.append({"z": z, "s": s, "T": T})

    if not parsed_eps:
        empty = {
            "s_reach": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_reach": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "reach_labels": torch.zeros((0,), dtype=torch.float32, device=device),
        }
        return empty, empty

    num_eps = len(parsed_eps)
    max_h = max(horizons)
    far_threshold = 2 * max_h

    s_pos: List[np.ndarray] = []
    g_pos: List[np.ndarray] = []
    s_neg_easy: List[np.ndarray] = []
    g_neg_easy: List[np.ndarray] = []
    s_neg_hard: List[np.ndarray] = []
    g_neg_hard: List[np.ndarray] = []

    samples_collected = 0
    for ep_idx, ep in enumerate(parsed_eps):
        T = ep["T"]
        valid_ks = [k for k in horizons if k < T]
        if not valid_ks:
            continue
        # Sample positive points
        n_picks = max(1, min(20, max_samples // max(1, num_eps)))
        for _ in range(n_picks):
            k = int(np.random.choice(valid_ks))
            t = int(np.random.randint(0, T - k))
            s_t = ep["s"][t]
            g_t = ep["z"][t + k]

            s_pos.append(s_t)
            g_pos.append(g_t)

            # Easy negative: cross-episode goal
            other_idx = (ep_idx + int(np.random.randint(1, max(2, num_eps)))) % num_eps
            other_ep = parsed_eps[other_idx]
            u_easy = int(np.random.randint(0, other_ep["T"]))
            s_neg_easy.append(s_t)
            g_neg_easy.append(other_ep["z"][u_easy])

            # Hard negative: within-episode far goal (u > t + 2*max(horizons))
            if t + far_threshold < T:
                u_hard = int(np.random.randint(t + far_threshold, T))
                neg_g_hard = ep["z"][u_hard]
            else:
                neg_g_hard = other_ep["z"][u_easy]
            s_neg_hard.append(s_t)
            g_neg_hard.append(neg_g_hard)

            samples_collected += 1
            if samples_collected >= max_samples:
                break
        if samples_collected >= max_samples:
            break

    if not s_pos:
        empty = {
            "s_reach": torch.zeros((0, STATE_DIM), dtype=torch.float32, device=device),
            "g_reach": torch.zeros((0, LATENT_DIM), dtype=torch.float32, device=device),
            "reach_labels": torch.zeros((0,), dtype=torch.float32, device=device),
        }
        return empty, empty

    # Easy batch: all positives + all easy negatives
    s_easy = s_pos + s_neg_easy
    g_easy = g_pos + g_neg_easy
    lbl_easy = [1.0] * len(s_pos) + [0.0] * len(s_neg_easy)

    # Hard batch: all positives + 50/50 mix of easy and hard negatives
    n_neg = len(s_pos)
    mixed_neg_s = []
    mixed_neg_g = []
    for i in range(n_neg):
        if i % 2 == 0:
            mixed_neg_s.append(s_neg_easy[i])
            mixed_neg_g.append(g_neg_easy[i])
        else:
            mixed_neg_s.append(s_neg_hard[i])
            mixed_neg_g.append(g_neg_hard[i])

    s_hard = s_pos + mixed_neg_s
    g_hard = g_pos + mixed_neg_g
    lbl_hard = [1.0] * len(s_pos) + [0.0] * len(mixed_neg_s)

    easy_dict = {
        "s_reach": torch.from_numpy(np.stack(s_easy, axis=0)).float().to(device),
        "g_reach": torch.from_numpy(np.stack(g_easy, axis=0)).float().to(device),
        "reach_labels": torch.from_numpy(np.array(lbl_easy, dtype=np.float32)).to(device),
    }
    hard_dict = {
        "s_reach": torch.from_numpy(np.stack(s_hard, axis=0)).float().to(device),
        "g_reach": torch.from_numpy(np.stack(g_hard, axis=0)).float().to(device),
        "reach_labels": torch.from_numpy(np.array(lbl_hard, dtype=np.float32)).to(device),
    }
    return easy_dict, hard_dict


def compute_her_loss(
    option_net: OptionNet,
    reach_net: ReachabilityNet,
    her_batch_data: Dict[str, torch.Tensor],
    ce_weight: float = 1.0,
    bce_weight: float = 0.5,
    label_smoothing: Optional[float] = None,
    weight_clip: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Joint policy imitation + reachability discrimination loss.

    Uses frequency-balanced, label-smoothed cross entropy for OptionNet.
    Loss = ce_weight * CE_balanced(policy_logits, actions_pos) + bce_weight * BCE(reach_logits, reach_labels)
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

    cfg_smoothing = getattr(option_net.cfg, "label_smoothing", 0.05) if label_smoothing is None else label_smoothing
    cfg_clip = getattr(option_net.cfg, "weight_clip", 50.0) if weight_clip is None else weight_clip

    policy_logits = option_net.logits(s_pos, g_pos)
    counts = torch.bincount(a_pos, minlength=NUM_ACTIONS).clamp_min(1)
    freq_weight = (1.0 / counts[a_pos].float())

    if "weights_pos" in her_batch_data and her_batch_data["weights_pos"].shape[0] == a_pos.shape[0]:
        w_reward = her_batch_data["weights_pos"].float()
        w = freq_weight * w_reward
    else:
        w = freq_weight

    # Renormalize to mean 1.0
    w = w * (len(a_pos) / max(w.sum(), 1e-8))
    # Clip at cfg_clip (50.0)
    w = w.clamp(max=cfg_clip)

    per_sample_ce = F.cross_entropy(
        policy_logits, a_pos, weight=None, label_smoothing=cfg_smoothing, reduction="none"
    )
    ce_loss = (per_sample_ce * w).mean()

    reach_logits = reach_net.raw_logits(s_reach, g_reach)
    bce_loss = F.binary_cross_entropy_with_logits(reach_logits, labels_reach)

    total_loss = ce_weight * ce_loss + bce_weight * bce_loss

    w_mean = float(w.mean().detach().item())
    w_p95 = float(np.percentile(w.detach().cpu().numpy(), 95))
    if "is_unlock_pos" in her_batch_data and len(her_batch_data["is_unlock_pos"]) > 0:
        unlock_frac = float(her_batch_data["is_unlock_pos"].float().mean().item())
    else:
        unlock_frac = 0.0

    metrics = {
        "ce_loss": float(ce_loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "her_weight_mean": w_mean,
        "her_weight_p95": w_p95,
        "her_unlock_pair_fraction": unlock_frac,
    }
    return total_loss, metrics
