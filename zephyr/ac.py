"""Zephyr actor-critic: latent imagination rollout + DreamerV3-style losses.

Recurrence convention (used consistently by wm, ac, planner and trainers):

    h_t      = belief BEFORE processing z_t;  s_t = concat(z_t, h_t)
    a_t      ~ Actor(s_t)
    z_hat_{t+1} = predict(z_t, a_t, h_t)                    (JEPA prior)
    h_{t+1}  = belief_step(z_{t+1}, a_t, h_t)             (posterior: z from encoder;
                                                           imagination: z from predictor)

During imagination there are no observations, so the predicted latent feeds the
belief — the closed-loop prior. The same equations drive the free-running JEPA
evaluation (gate ZG2).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from zephyr.config import ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.losses import lambda_returns
from zephyr.wm import ZephyrWM


def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """Categorical probabilities mixed with a small uniform component."""
    probs = torch.softmax(logits.float(), dim=-1)
    n = probs.shape[-1]
    return (1.0 - unimix) * probs + unimix / n


def _dist(probs: Tensor) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(probs=probs)


def ensemble_disagreement(preds: Tensor) -> Tensor:
    """Plan2Explore-style intrinsic reward: mean per-dim std across members.

    preds [K, ..., D] -> [...] scalar disagreement.
    """
    return preds.float().std(dim=0).mean(dim=-1)


def imagine_trajectory(
    wm: ZephyrWM,
    heads: ZephyrHeads,
    z0: Tensor,
    h0: Tensor,
    horizon: int,
    cfg: ZephyrConfig,
    deterministic: bool = False,
    with_ensemble: bool = True,
) -> Dict[str, Tensor]:
    """Closed-loop latent rollout under the current actor.

    Returns dict:
        z          [H, B, latent]   predicted latents (steps 1..H)
        h          [H, B, belief]   beliefs after each prior update
        s          [H, B, latent+belief]
        actions    [H, B] int64
        log_prob   [H, B]
        entropy    [H, B]
        reward_logits [H, B, bins], reward_values [H, B]
        continue_logits [H, B], continue_probs [H, B]
        value_logits [H+1, B, bins]  critic at s_0..s_H (s_H = bootstrap state)
        intrinsic  [H, B]           ensemble disagreement (zeros if disabled)
    """
    device = z0.device
    batch = z0.shape[0]
    z = z0
    h = h0
    s = heads.state_input(z, h)

    critic_logits_0 = heads.critic(s)
    value_logits = [critic_logits_0]

    zs, hs, ss, acts, logps, ents = [], [], [], [], [], []
    rew_logits, cont_logits, intr = [], [], []
    for _ in range(horizon):
        actor_logits = heads.actor(s)
        if deterministic:
            action = torch.argmax(actor_logits, dim=-1)
            probs = unimix_probs(actor_logits, cfg.ac.actor_unimix)
            log_p = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
            ent = torch.zeros_like(log_p)
        else:
            dist = _dist(unimix_probs(actor_logits, cfg.ac.actor_unimix))
            action = dist.sample()
            log_p = dist.log_prob(action)
            ent = dist.entropy()

        z_next = wm.predict(z, action, h)
        h_next = wm.belief_step(z_next, action, h)

        if with_ensemble and cfg.explore.disagreement_weight > 0.0:
            preds = heads.ensemble(z, action, h)
            r_int = ensemble_disagreement(preds)
        else:
            r_int = torch.zeros(batch, device=device)

        s_next = heads.state_input(z_next, h_next)

        zs.append(z_next)
        hs.append(h_next)
        ss.append(s_next)
        acts.append(action)
        logps.append(log_p)
        ents.append(ent)
        rew_logits.append(heads.reward(s_next))
        cont_logits.append(heads.contin(s_next))
        intr.append(r_int)
        value_logits.append(heads.critic(s_next))

        z, h, s = z_next, h_next, s_next

    reward_logits_t = torch.stack(rew_logits, dim=0)
    cont_logits_t = torch.stack(cont_logits, dim=0)
    coder = heads.reward.coder
    reward_values = coder.decode(reward_logits_t)
    cont_probs = 1.0 - torch.sigmoid(cont_logits_t)

    return {
        "z": torch.stack(zs, dim=0),
        "h": torch.stack(hs, dim=0),
        "s": torch.stack(ss, dim=0),
        "actions": torch.stack(acts, dim=0),
        "log_prob": torch.stack(logps, dim=0),
        "entropy": torch.stack(ents, dim=0),
        "reward_logits": reward_logits_t,
        "reward_values": reward_values,
        "continue_logits": cont_logits_t,
        "continue_probs": cont_probs,
        "value_logits": torch.stack(value_logits, dim=0),
        "intrinsic": torch.stack(intr, dim=0),
    }


def trajectory_lambda_returns(
    imagination: Dict[str, Tensor],
    critic_coder,  # TwoHotCoder of the critic head
    cfg: ZephyrConfig,
    intrinsic_weight: float = 0.0,
    slow_value_logits: Optional[Tensor] = None,
) -> Tensor:
    """λ-returns of an imagined trajectory -> [H, B].

    Rewards are predicted env reward (+ optional intrinsic, used only during
    training-time exploration); continuation uses predicted continue probs.
    ``slow_value_logits`` [B, bins] provides the terminal bootstrap when given
    (slow target critic), else the imagined critic's own final value.
    """
    rewards = imagination["reward_values"]
    if intrinsic_weight > 0.0:
        rewards = rewards + intrinsic_weight * imagination["intrinsic"]
    continues = imagination["continue_probs"]
    values = critic_coder.decode(imagination["value_logits"][:-1])   # [H, B]
    if slow_value_logits is not None:
        bootstrap = critic_coder.decode(slow_value_logits)           # [B]
    else:
        bootstrap = critic_coder.decode(imagination["value_logits"][-1])
    return lambda_returns(
        rewards,
        continues,
        values,
        bootstrap,
        gamma=cfg.ac.gamma,
        td_lambda=cfg.ac.td_lambda,
    )


def normalize_returns(lam_ret: Tensor) -> Tensor:
    """Per-timestep standardization (DreamerV3-style advantage scaling)."""
    mean = lam_ret.mean(dim=1, keepdim=True)
    std = lam_ret.std(dim=1, keepdim=True)
    return (lam_ret - mean) / (std + 1e-8)


def actor_lambda_return_loss(
    lam_ret: Tensor,
    log_prob: Tensor,
    entropy: Tensor,
    cfg: ZephyrConfig,
) -> Tensor:
    """REINFORCE through imagined λ-returns + entropy bonus. All scalar."""
    advantage = normalize_returns(lam_ret).detach()
    pg = -(advantage * log_prob).mean()
    ent = -cfg.ac.actor_entropy_weight * entropy.mean()
    return cfg.ac.actor_loss_weight * (pg + ent)


def critic_twohot_loss(critic_logits: Tensor, lam_ret: Tensor, coder) -> Tensor:
    """Critic regresses the imagined λ-returns (returns treated as targets)."""
    return coder.loss(critic_logits, lam_ret.detach())
