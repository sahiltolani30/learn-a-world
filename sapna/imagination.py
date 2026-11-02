"""
sapna/imagination.py

Imagination rollouts and two-hot distributional math per SAPNA_PLAN.md §5.2, §5.3, §8.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sapna.policy import symlog, symexp, unimix


def twohot_encode(
    x: Union[torch.Tensor, float],
    num_bins: int = 255,
    low: float = -20.0,
    high: float = 20.0,
    is_symlog: bool = False,
) -> torch.Tensor:
    """
    Two-hot target encoding per SAPNA_PLAN.md §5.3.
    Clamps symlog(x) to [low, high] and computes adjacent linear weights.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    y = x if is_symlog else symlog(x)
    y = torch.clamp(y, low, high)

    step = (high - low) / float(num_bins - 1)
    c = torch.linspace(low, high, num_bins, device=y.device, dtype=y.dtype)

    idx = ((y - low) / step).floor().long().clamp(0, num_bins - 2)
    c_below = c[idx]

    w_above = ((y - c_below) / step).clamp(0.0, 1.0)
    w_below = 1.0 - w_above

    probs = torch.zeros(*y.shape, num_bins, device=y.device, dtype=y.dtype)
    probs.scatter_(-1, idx.unsqueeze(-1), w_below.unsqueeze(-1))
    probs.scatter_add_(-1, (idx + 1).unsqueeze(-1), w_above.unsqueeze(-1))
    return probs.clamp(min=0.0)


def twohot_score(
    logits: torch.Tensor,
    num_bins: int = 255,
    low: float = -20.0,
    high: float = 20.0,
) -> torch.Tensor:
    """
    Scalar readout per SAPNA_PLAN.md §5.3:
    symexp( sum(softmax(logits) * linspace(-20, 20, 255)) )
    """
    probs = torch.softmax(logits, dim=-1)
    c = torch.linspace(low, high, num_bins, device=probs.device, dtype=probs.dtype)
    y = (probs * c).sum(dim=-1)
    return symexp(y)


def twohot_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 255,
    low: float = -20.0,
    high: float = 20.0,
    is_symlog: bool = True,
) -> torch.Tensor:
    """
    Cross-entropy loss against two-hot target distribution: -sum(target * log_softmax(logits)).
    Accepts target as either:
    1. Pre-encoded distribution (B, num_bins)
    2. Scalar targets (B,) [symlogged if is_symlog=True, or raw otherwise]
    """
    if target.dim() == 2 and target.size(-1) == logits.size(-1):
        target_dist = target
    else:
        target_dist = twohot_encode(
            target.squeeze(-1) if target.dim() > 1 else target,
            num_bins=num_bins,
            low=low,
            high=high,
            is_symlog=is_symlog,
        )

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_dist * log_probs).sum(dim=-1).mean()
    return loss


def lambda_returns(
    rewards: torch.Tensor,
    conts: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Compute lambda-returns via backward recursion per SAPNA_PLAN.md §5.2.
    G_H = v_bootstrap
    G_k = r_k + gamma * cont_k * [ (1 - lam) * v_{k+1} + lam * G_{k+1} ]
    
    Args:
        rewards: (N, H)
        conts: (N, H)
        values: (N, H)
        bootstrap: (N,)
        gamma: float
        lam: float
    Returns:
        returns: (N, H)
    """
    N, H = rewards.shape
    returns = torch.zeros_like(rewards)
    next_v = bootstrap
    next_G = bootstrap

    for k in reversed(range(H)):
        r_k = rewards[:, k]
        c_k = conts[:, k]
        if k + 1 < H:
            v_next = values[:, k + 1]
        else:
            v_next = next_v

        G_k = r_k + gamma * c_k * ((1.0 - lam) * v_next + lam * next_G)
        returns[:, k] = G_k
        next_G = G_k

    return returns


def imagine_batch(
    tokens0: Optional[torch.Tensor] = None,
    z0: Optional[torch.Tensor] = None,
    h0: Optional[torch.Tensor] = None,
    actor: Optional[nn.Module] = None,
    critic: Optional[nn.Module] = None,
    predictor: Optional[nn.Module] = None,
    temporal_core: Optional[nn.Module] = None,
    action_embedding: Optional[nn.Module] = None,
    pool_fn: Optional[callable] = None,
    reward_head: Optional[nn.Module] = None,
    done_head: Optional[nn.Module] = None,
    slow_critic: Optional[nn.Module] = None,
    H: int = 15,
    unimix_ratio: float = 0.01,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Imagined latent trajectory rollout per SAPNA_PLAN.md §4 & §8.
    
    Supports flexible argument ordering:
    imagine_batch(z0, h0, actor, critic, predictor, temporal_core, action_embedding, pool_fn, H=15)
    with optional tokens0, reward_head, done_head, slow_critic.
    """
    # Handle positional variations where z0 was passed as first arg
    if tokens0 is not None and tokens0.dim() == 2 and z0 is not None and z0.dim() == 2:
        # User passed (z0, h0, actor, critic, ...) without tokens0
        # Realign arguments
        actual_z0 = tokens0
        actual_h0 = z0
        actual_actor = h0
        actual_critic = actor
        actual_pred = critic
        actual_core = predictor
        actual_emb = temporal_core
        actual_pool = action_embedding
        H_val = pool_fn if isinstance(pool_fn, int) else H

        actor = actual_actor
        critic = actual_critic
        predictor = actual_pred
        temporal_core = actual_core
        action_embedding = actual_emb
        pool_fn = actual_pool
        z0 = actual_z0
        h0 = actual_h0
        H = H_val
        tokens0 = kwargs.get("tokens0", None)

    reward_head = reward_head or kwargs.get("reward_head", None)
    done_head = done_head or kwargs.get("done_head", None)
    slow_critic = slow_critic or kwargs.get("slow_critic", None)

    N = z0.size(0)
    device = z0.device

    if tokens0 is None:
        tokens_k = z0.view(N, 4, -1).repeat(1, 16, 1)
    else:
        tokens_k = tokens0

    z_k = z0
    h_k = h0

    actions_list = []
    logprobs_list = []
    entropies_list = []
    values_list = []
    rewards_list = []
    conts_list = []
    states_list = []

    for k in range(H):
        state_k = torch.cat([z_k.detach(), h_k.detach()], dim=-1)
        states_list.append(state_k)

        # Policy forward
        logits_k = actor(state_k)
        raw_probs_k = torch.softmax(logits_k, dim=-1)
        probs_k = unimix(raw_probs_k, unimix_ratio=unimix_ratio)
        dist_k = Categorical(probs=probs_k)
        a_k = dist_k.sample()

        logprob_k = dist_k.log_prob(a_k)
        entropy_k = dist_k.entropy()

        # Critic value: critic if provided, evaluated with gradients
        critic_to_use = critic if critic is not None else slow_critic
        if critic_to_use is not None:
            v_logits = critic_to_use(state_k)
            v_k = twohot_score(v_logits)
        else:
            v_k = torch.zeros(N, device=device)

        # World model step (under no_grad)
        with torch.no_grad():
            if hasattr(predictor, "forward_tokens_clean"):
                mean_pred = predictor.forward_tokens_clean(tokens_k, a_k, h_k)[0]
            else:
                mean_pred = predictor.forward_tokens(tokens_k, a_k, h_k)[0]

            tokens_next = mean_pred
            if pool_fn is not None:
                z_next = pool_fn(mean_pred)
            else:
                z_next = mean_pred.mean(dim=1)

            a_emb = action_embedding(a_k)
            h_next = temporal_core(z_k, h_k, a_emb=a_emb)

            # Arrival state
            arr_k = torch.cat([z_next, h_next], dim=-1)

            if reward_head is not None:
                r_logits = reward_head(arr_k)
                r_k = twohot_score(r_logits).detach()
            else:
                r_k = torch.zeros(N, device=device)

            if done_head is not None:
                d_logits = done_head(arr_k)
                d_k = torch.sigmoid(d_logits).squeeze(-1).detach()
                c_k = (1.0 - d_k).detach()
            else:
                c_k = torch.ones(N, device=device)

        actions_list.append(a_k)
        logprobs_list.append(logprob_k)
        entropies_list.append(entropy_k)
        values_list.append(v_k)
        rewards_list.append(r_k)
        conts_list.append(c_k)

        tokens_k = tokens_next
        z_k = z_next
        h_k = h_next

    # Bootstrap value at step H
    state_H = torch.cat([z_k.detach(), h_k.detach()], dim=-1)
    critic_to_use = critic if critic is not None else slow_critic
    if critic_to_use is not None:
        v_H_logits = critic_to_use(state_H)
        bootstrap = twohot_score(v_H_logits)
    else:
        bootstrap = torch.zeros(N, device=device)

    return {
        "actions": torch.stack(actions_list, dim=1),
        "logprobs": torch.stack(logprobs_list, dim=1),
        "entropies": torch.stack(entropies_list, dim=1),
        "values": torch.stack(values_list, dim=1),
        "bootstrap": bootstrap,
        "rewards": torch.stack(rewards_list, dim=1),
        "conts": torch.stack(conts_list, dim=1),
        "final_z": z_k,
        "final_h": h_k,
        "states": torch.stack(states_list, dim=1),
    }
