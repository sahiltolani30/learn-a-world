"""Zephyr world-model training (Stage 0) and gate evaluation (ZG1/ZG2/ZG3/ZG5).

Single batch flow (convention from zephyr/ac.py):

    posterior pass (teacher forcing):  s_t = [z_t, h_t] with
        h_{t+1} = belief_step(z_{t+1}^{real}, a_t, h_t)          -> head inputs
    prior pass (free-running JEPA):
        z_hat_{t+1} = predict(z_hat_t, a_t, h_hat_t)
        h_hat_{t+1} = belief_step(z_hat_{t+1}, a_t, h_hat_t)     -> JEPA loss vs
                                                      ema_encode(obs_{t+1})

L = L_JEPA + w_r*L_reward + w_c*L_continue + w_inv*L_readout + w_eff*L_effect + w_ens*L_ensemble.
The JEPA term is never disabled (project contract); anti-collapse var/cov terms
ride inside L_JEPA. Inventory/effect labels are privileged training targets —
never inference inputs.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from zephyr.buffer import ZephyrReplay
from zephyr.config import ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.losses import (
    focal_bce_with_logits,
    masked_mse,
    roc_auc,
    vicreg_cov_loss,
    vicreg_var_loss,
)
from zephyr.wm import ZephyrWM


def _batch_to(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _posterior_states(
    wm: ZephyrWM, z_now: Tensor, z_next_real: Tensor, actions: Tensor
) -> List[Tensor]:
    """Teacher-forced beliefs: returns [h_0, ..., h_{T-1}] (pre-observation)."""
    b = z_now.shape[0]
    h = wm.init_state(b, z_now.device)
    h_pre: List[Tensor] = []
    for t in range(actions.shape[1]):
        h_pre.append(h)
        h = wm.belief_step(z_next_real[:, t], actions[:, t], h)
    return h_pre


def _prior_rollout(
    wm: ZephyrWM, z0: Tensor, actions: Tensor, grad: bool = True
) -> Tuple[List[Tensor], List[Tensor]]:
    """Free-running closed-loop latent rollout from (z0, h=0).

    Returns (z_hat sequence: T entries for steps 1..T, h_hat sequence).
    """
    b = z0.shape[0]
    h = wm.init_state(b, z0.device)
    z = z0
    zs: List[Tensor] = []
    hs: List[Tensor] = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for t in range(actions.shape[1]):
            a_t = actions[:, t]
            z_next = wm.predict(z, a_t, h)
            h = wm.belief_step(z_next, a_t, h)
            zs.append(z_next)
            hs.append(h)
            z = z_next
    return zs, hs


def _encode_windows(
    wm: ZephyrWM, batch: Dict[str, Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Online z for [B, T] frames and [B, T] next frames + EMA targets."""
    obs = batch["obs"]
    obs_next = batch["obs_next"]
    b, t = actions_shape(batch)
    z_now = wm.encode(obs.reshape((b * t,) + obs.shape[2:])).reshape(b, t, -1)
    z_next_real = wm.encode(obs_next.reshape((b * t,) + obs.shape[2:])).reshape(b, t, -1)
    z_target = wm.ema_encode(obs_next.reshape((b * t,) + obs.shape[2:])).reshape(b, t, -1)
    return z_now, z_next_real, z_target


def actions_shape(batch: Dict[str, Tensor]) -> Tuple[int, int]:
    return int(batch["actions"].shape[0]), int(batch["actions"].shape[1])


def wm_joint_losses(
    wm: ZephyrWM,
    heads: ZephyrHeads,
    batch: Dict[str, Tensor],
    cfg: ZephyrConfig,
    device: torch.device,
    return_states: bool = False,
):
    """Full Stage-0 objective. Returns (total_loss, metrics[, states]).

    ``states`` (when ``return_states``) carries the teacher-forced posterior
    tensors the AC trainer reuses for imagination starts:
    z_now [B,T,D], h_pre [T,B,H], s_all [B,T,s_dim] — all with gradients.
    """
    batch = _batch_to(batch, device)
    valid = batch["valid"]
    done = batch["done"]
    rewards = batch["rewards"]
    inventory = batch["inventory"]
    inventory_next = batch["inventory_next"]
    b, t = actions_shape(batch)

    z_now, z_next_real, z_target = _encode_windows(wm, batch)
    h_pre = _posterior_states(wm, z_now, z_next_real, batch["actions"])
    s_all = torch.stack(
        [heads.state_input(z_now[:, i], h_pre[i]) for i in range(t)], dim=1
    )  # [B, T, s_dim]

    # ---- JEPA (free-running, horizon-decay weighted) ----
    zs_prior, _ = _prior_rollout(wm, z_now[:, 0], batch["actions"], grad=True)
    z_prior = torch.stack(zs_prior, dim=1)  # [B, T, D]
    je_mask = (valid & ~done).to(z_prior.dtype)
    weights = torch.tensor(
        [cfg.world_model.horizon_decay ** i for i in range(t)],
        device=device, dtype=z_prior.dtype,
    )
    per_step = ((z_prior - z_target) ** 2).mean(dim=-1)  # [B, T]
    denom = (weights * je_mask.mean(dim=0)).sum().clamp(min=1e-8)
    l_jepa = (per_step * weights * je_mask).sum() / denom
    l_var_pred = vicreg_var_loss(z_prior, gamma=cfg.world_model.var_gamma)
    l_cov_pred = vicreg_cov_loss(z_prior)
    l_var_online = vicreg_var_loss(z_now, gamma=cfg.world_model.var_gamma)
    l_jepa_total = (
        l_jepa
        + cfg.world_model.var_weight_pred * l_var_pred
        + cfg.world_model.cov_weight_pred * l_cov_pred
        + cfg.world_model.var_weight_online * l_var_online
    )

    # ---- heads on posterior states ----
    v_mask = valid
    d_mask = valid & ~done  # exclude bootstrap-padded steps from effect targets

    s_flat = s_all.reshape(b * t, -1)
    sel = v_mask.reshape(-1)
    if bool(sel.any()):
        l_reward = heads.reward.coder.loss(
            heads.reward(s_flat[sel]), rewards.reshape(b * t)[sel]
        )
        cont_logits_sel = heads.contin(s_flat[sel])
        l_continue = focal_bce_with_logits(
            cont_logits_sel,
            done.reshape(b * t)[sel].float(),
            pos_weight=cfg.heads.continue_pos_weight,
            gamma=cfg.heads.continue_focal_gamma,
        )
    else:
        l_reward = s_flat.sum() * 0.0
        l_continue = s_flat.sum() * 0.0
    readout_pred = heads.readout(s_flat).reshape(b, t, -1)
    l_readout = masked_mse(readout_pred, inventory, valid)

    effect_pred = heads.effect(
        s_flat, batch["actions"].reshape(b * t)
    ).reshape(b, t, -1)
    effect_target = inventory_next - inventory
    l_effect = masked_mse(effect_pred, effect_target, d_mask)

    h_stack = torch.stack(h_pre, dim=1)
    ens_preds = heads.ensemble(z_now, batch["actions"], h_stack)
    target_ens = z_next_real.detach()
    if bool(d_mask.any()):
        l_ensemble = sum(
            masked_mse(ens_preds[k], target_ens, d_mask)
            for k in range(ens_preds.shape[0])
        ) / ens_preds.shape[0]
    else:
        l_ensemble = ens_preds.sum() * 0.0

    total = (
        l_jepa_total
        + cfg.heads.reward_loss_weight * l_reward
        + cfg.heads.continue_loss_weight * l_continue
        + cfg.heads.readout_loss_weight * l_readout
        + cfg.heads.effect_loss_weight * l_effect
        + cfg.heads.ensemble_loss_weight * l_ensemble
    )

    metrics = {
        "loss": float(total.detach()),
        "loss_jepa_mse": float(l_jepa.detach()),
        "loss_var_pred": float(l_var_pred.detach()),
        "loss_cov_pred": float(l_cov_pred.detach()),
        "loss_var_online": float(l_var_online.detach()),
        "loss_reward": float(l_reward.detach()),
        "loss_continue": float(l_continue.detach()),
        "loss_readout": float(l_readout.detach()),
        "loss_effect": float(l_effect.detach()),
        "loss_ensemble": float(l_ensemble.detach()),
        "latent_std_online": float(z_now.std(dim=(0, 1)).mean().detach()),
        "latent_std_prior": float(z_prior.std(dim=(0, 1)).mean().detach()),
        "done_fraction": float(done.float().mean()),
    }
    if return_states:
        states = {
            "z_now": z_now,
            "h_pre": torch.stack(h_pre, dim=0),
            "s_all": s_all,
        }
        return total, metrics, states
    return total, metrics


def build_optimizers(
    wm: ZephyrWM, heads: ZephyrHeads, cfg: ZephyrConfig, include_ac: bool = False
) -> Dict[str, torch.optim.Optimizer]:
    """Parameter-grouped AdamW: world / heads / (actor / critic)."""
    o = cfg.optim
    world_params = (
        list(wm.encoder.parameters())
        + list(wm.gru.parameters())
        + list(wm.predictor.parameters())
        + list(heads.ensemble.parameters())
    )
    head_params = (
        list(heads.reward.parameters())
        + list(heads.contin.parameters())
        + list(heads.readout.parameters())
        + list(heads.effect.parameters())
    )
    optimizers: Dict[str, torch.optim.Optimizer] = {
        "world": torch.optim.AdamW(
            world_params, lr=o.lr_world, weight_decay=o.weight_decay, eps=o.adam_eps
        ),
        "heads": torch.optim.AdamW(
            head_params, lr=o.lr_heads, weight_decay=o.weight_decay, eps=o.adam_eps
        ),
    }
    if include_ac:
        optimizers["actor"] = torch.optim.AdamW(
            list(heads.actor.parameters()),
            lr=o.lr_actor, weight_decay=o.weight_decay, eps=o.adam_eps,
        )
        optimizers["critic"] = torch.optim.AdamW(
            list(heads.critic.parameters()),
            lr=o.lr_critic, weight_decay=o.weight_decay, eps=o.adam_eps,
        )
    return optimizers


def _clip_and_step(
    optimizers: Dict[str, torch.optim.Optimizer], grad_clip: float
) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for name, opt in optimizers.items():
        norms[name] = float(
            torch.nn.utils.clip_grad_norm_(
                [p for group in opt.param_groups for p in group["params"]], grad_clip
            )
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
    return norms


def _save_pair(
    wm: ZephyrWM,
    heads: ZephyrHeads,
    step: int,
    ckpt_path: str,
    heads_ckpt_path: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    heads_path = heads_ckpt_path or ckpt_path.replace(".pt", "_heads.pt")
    wm_tmp, heads_tmp = ckpt_path + ".tmp", heads_path + ".tmp"
    wm.save_checkpoint(wm_tmp, step=step)
    heads.save_checkpoint(heads_tmp, step=step)
    os.replace(wm_tmp, ckpt_path)
    os.replace(heads_tmp, heads_path)


def _save_sidecar(
    sidecar_path: str,
    step: int,
    optimizers: Dict[str, torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
) -> None:
    """Trainer resume state: optimizer moments, AMP scaler, both RNG streams."""
    import random as _random

    payload = {
        "format": "zephyr_trainer_v1",
        "step": int(step),
        "opt_states": {k: v.state_dict() for k, v in optimizers.items()},
        "scaler_state": scaler.state_dict(),
        "rng_np": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "rng_python": _random.getstate(),
    }
    tmp = sidecar_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, sidecar_path)


def _load_sidecar(
    sidecar_path: str,
    expect_step: int,
    optimizers: Dict[str, torch.optim.Optimizer],
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
) -> bool:
    """Restore trainer state iff the sidecar matches the model checkpoint step.

    Returns True when everything was restored; a missing/stale sidecar returns
    False and training proceeds with fresh optimizer state (safe fallback).
    """
    import random as _random

    if not os.path.exists(sidecar_path):
        return False
    try:
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "zephyr_trainer_v1":
            return False
        if int(payload["step"]) != int(expect_step):
            return False
        for name, state in payload["opt_states"].items():
            if name in optimizers:
                optimizers[name].load_state_dict(state)
        scaler.load_state_dict(payload["scaler_state"])
        rng.bit_generator.state = payload["rng_np"]
        torch.set_rng_state(payload["rng_torch"])
        _random.setstate(payload["rng_python"])
        return True
    except Exception:
        return False


def train_world_model(
    wm: ZephyrWM,
    heads: ZephyrHeads,
    replay: ZephyrReplay,
    cfg: ZephyrConfig,
    device: str = "cpu",
    steps: int = 50000,
    log_every: int = 500,
    ckpt_path: Optional[str] = None,
    heads_ckpt_path: Optional[str] = None,
    ckpt_every: int = 5000,
    ckpt_every_seconds: float = 900.0,
    on_log=None,
    initial_step: int = 0,
) -> Dict[str, float]:
    """Stage-0 training loop. Returns the final metrics dict.

    Crash-safe: when ``ckpt_path`` is set, a trainer sidecar (optimizer, AMP
    scaler, RNG streams) is written atomically next to the model checkpoints,
    at least every ``ckpt_every`` steps OR ``ckpt_every_seconds`` seconds —
    whichever comes first — so an interrupted run resumes to within ~15 min
    of progress. On resume, pass ``initial_step`` = the checkpoint's step and
    the sidecar restores the exact optimizer/RNG state when it matches.
    """
    dev = torch.device(device)
    optimizers = build_optimizers(wm, heads, cfg, include_ac=False)
    use_amp = (dev.type == "cuda") and cfg.optim.fp16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg.data.split_seed + 1)
    sidecar_path = (ckpt_path + ".trainer.pt") if ckpt_path else None
    if sidecar_path is not None and initial_step > 0:
        restored = _load_sidecar(sidecar_path, initial_step, optimizers, scaler, rng)
        if on_log is not None:
            on_log(initial_step, {
                "train_step": initial_step,
                "trainer_state_restored": 1.0 if restored else 0.0,
            })
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 64x64 inputs: safe + faster
    last: Dict[str, float] = {}
    import time as _time
    last_save_t = _time.monotonic()

    def _maybe_save(step: int, force: bool = False) -> None:
        nonlocal last_save_t
        if ckpt_path is None:
            return
        due_by_step = (step - initial_step) % ckpt_every == 0
        due_by_time = (_time.monotonic() - last_save_t) >= ckpt_every_seconds
        if force or due_by_step or due_by_time:
            _save_pair(wm, heads, step=step, ckpt_path=ckpt_path,
                       heads_ckpt_path=heads_ckpt_path)
            if sidecar_path is not None:
                _save_sidecar(sidecar_path, step, optimizers, scaler, rng)
            last_save_t = _time.monotonic()

    for step in range(initial_step + 1, initial_step + steps + 1):
        batch = replay.sample(
            cfg.data.batch_size, cfg.data.seq_len, rng,
            prioritize_achievement_frac=cfg.data.prioritize_achievement_frac,
        )
        with torch.autocast("cuda", enabled=use_amp):
            total, metrics = wm_joint_losses(wm, heads, batch, cfg, dev)
        scaler.scale(total).backward()
        if use_amp:
            for opt in optimizers.values():
                scaler.unscale_(opt)
        _clip_and_step(optimizers, cfg.optim.grad_clip)
        scaler.update()
        wm.update_ema(step)
        last = metrics
        if (step - initial_step) % log_every == 0:
            metrics["train_step"] = step
            if on_log is not None:
                on_log(step, metrics)
        _maybe_save(step)
    _maybe_save(initial_step + steps, force=True)
    return last


# ---------------------------------------------------------------------------
# Gate evaluation (ZG1 collapse, ZG2 free-run, ZG3 readout, ZG5 termination)
# ---------------------------------------------------------------------------

def _val_batches(replay: ZephyrReplay, cfg: ZephyrConfig, max_batches: int):
    rng = np.random.default_rng(cfg.data.split_seed + 2)
    for _ in range(max_batches):
        yield replay.sample(
            cfg.data.batch_size, cfg.data.seq_len, rng,
            prioritize_achievement_frac=cfg.data.prioritize_achievement_frac,
        )


@torch.no_grad()
def evaluate_gates(
    wm: ZephyrWM,
    heads: ZephyrHeads,
    val_replay: ZephyrReplay,
    cfg: ZephyrConfig,
    device: str = "cpu",
    max_batches: int = 50,
) -> Dict[str, object]:
    """Compute ZG1/ZG2/ZG3/ZG5 evidence on held-out validation episodes."""
    dev = torch.device(device)
    wm.eval(); heads.eval()
    horizons = (1, 5, 10, min(15, cfg.data.seq_len - 1))

    latent_stds: List[float] = []
    jepa_by_h = {h: [] for h in horizons}
    persist_by_h = {h: [] for h in horizons}
    readout_err: List[float] = []
    persist_inv_err: List[float] = []
    cont_scores: List[Tensor] = []
    cont_targets: List[Tensor] = []

    for batch in _val_batches(val_replay, cfg, max_batches):
        batch = _batch_to(batch, dev)
        valid = batch["valid"]
        done = batch["done"]
        b, t = actions_shape(batch)
        z_now, _, z_target = _encode_windows(wm, batch)
        latent_stds.append(float(z_now.std(dim=(0, 1)).mean()))

        # free-running rollout errors at offsets k (masked, no grad)
        zs_prior, _ = _prior_rollout(wm, z_now[:, 0], batch["actions"], grad=False)
        z_prior = torch.stack(zs_prior, dim=1)
        je_mask = (valid & ~done)
        for h in horizons:
            if h >= t:
                continue
            err = ((z_prior[:, h:] - z_target[:, h:]) ** 2).mean(dim=-1)
            m = je_mask[:, h:].float()
            if m.sum() > 0:
                jepa_by_h[h].append(float((err * m).sum() / m.sum()))
            # offset-k persistence in target space
            base, nxt = z_target[:, :-h], z_target[:, h:]
            perr = ((base - nxt) ** 2).mean(dim=-1)
            if m.sum() > 0:
                persist_by_h[h].append(float((perr * m).sum() / m.sum()))

        # readout vs lag-1 persistence (both in transformed label space)
        z_next_real = wm.encode(
            batch["obs_next"].reshape((b * t,) + batch["obs"].shape[2:])
        ).reshape(b, t, -1)
        h_pre = _posterior_states(wm, z_now, z_next_real, batch["actions"])
        s_all = torch.stack(
            [heads.state_input(z_now[:, i], h_pre[i]) for i in range(t)], dim=1
        )
        s_flat = s_all.reshape(b * t, -1)
        readout_pred = heads.readout(s_flat).reshape(b, t, -1)
        readout_err.append(float(masked_mse(readout_pred, batch["inventory"], valid)))
        inv = batch["inventory"]
        persist_inv = torch.cat([inv[:, :1], inv[:, :-1]], dim=1)
        persist_inv_err.append(float(masked_mse(persist_inv, inv, valid)))

        cont_logits = heads.contin(s_flat).reshape(b, t)
        m = valid.bool()
        cont_scores.append(torch.sigmoid(cont_logits)[m].cpu())
        cont_targets.append(done[m].cpu())

    zg1_std = float(np.mean(latent_stds)) if latent_stds else 0.0

    zg2 = {}
    zg2_all_pass = bool(jepa_by_h[horizons[0]])
    for h in horizons:
        if not jepa_by_h[h]:
            zg2[h] = {"available": False}
            zg2_all_pass = False
            continue
        freerun = float(np.mean(jepa_by_h[h]))
        persist = float(np.mean(persist_by_h[h]))
        zg2[h] = {
            "available": True,
            "freerun_mse": freerun,
            "persistence_mse": persist,
            "freerun_beats_persistence": bool(freerun < persist),
            "relative_improvement_pct": float(100.0 * (persist - freerun) / max(persist, 1e-12)),
        }
        zg2_all_pass = zg2_all_pass and freerun < persist

    readout_mse = float(np.mean(readout_err)) if readout_err else float("nan")
    persist_mse = float(np.mean(persist_inv_err)) if persist_inv_err else float("nan")
    zg3_improvement = float(100.0 * (persist_mse - readout_mse) / max(persist_mse, 1e-12))

    if cont_scores:
        scores = torch.cat(cont_scores)
        targets = torch.cat(cont_targets)
        n_pos = int((targets > 0.5).sum())
        n = int(targets.numel())
        preds = (scores > 0.5).float()
        tp = float(((preds > 0.5) & (targets > 0.5)).sum())
        fp = float(((preds > 0.5) & (targets <= 0.5)).sum())
        fn = float(((preds <= 0.5) & (targets > 0.5)).sum())
        recall = tp / max(1.0, tp + fn)
        precision = tp / max(1e-9, tp + fp)
        zg5 = {
            "available": True,
            "positive_fraction": n_pos / max(1, n),
            "majority_baseline_recall": 0.0,
            "majority_baseline_accuracy": max(n_pos, n - n_pos) / max(1, n),
            "recall_at_0p5": recall,
            "precision_at_0p5": precision,
            "auc": roc_auc(scores, targets),
        }
        zg5_pass = recall >= 0.5 and precision >= 0.5 and zg5["auc"] >= 0.9
    else:
        zg5 = {"available": False}
        zg5_pass = False

    return {
        "zg1": {
            "latent_std_online": zg1_std,
            "collapse_threshold": 0.05,
            "collapse_detected": bool(zg1_std < 0.05),
            "passed": bool(zg1_std >= 0.05),
            "note": "latents are LayerNorm'd; healthy per-dim batch std is ~0.1-0.5, "
                    "collapsed ~0 (threshold 0.05)",
        },
        "zg2": {"horizons": {str(k): v for k, v in zg2.items()}, "passed": bool(zg2_all_pass)},
        "zg3": {
            "readout_mse": readout_mse,
            "persistence_mse": persist_mse,
            "relative_improvement_pct": zg3_improvement,
            "passed": bool(readout_mse < persist_mse and zg3_improvement >= 30.0),
        },
        "zg5": {**zg5, "passed": bool(zg5_pass)},
        "all_core_gates_passed": bool(zg1_std >= 0.05 and zg2_all_pass and zg5_pass
                                      and readout_mse < persist_mse
                                      and zg3_improvement >= 30.0),
    }
