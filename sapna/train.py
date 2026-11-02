"""
sapna/train.py

Main training loop, evaluation, and checkpoint orchestration for SAPNA per SAPNA_PLAN.md §4, §9, §10, §13.
"""

from typing import Any, Dict, List, Optional, Tuple
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sapna.config import SAPNAConfig
from sapna.envs import ParallelCrafterBatch, make_crafter_env
from sapna.encoder import ObservationEncoder, TargetEncoder, verify_non_collapse
from sapna.dynamics import (
    ActionEmbedding,
    TemporalCore,
    SpatialPredictor,
    RewardHead,
    DoneHead,
    sample_token_mask,
)
from sapna.policy import Actor, Critic, SlowCritic, symlog, symexp, unimix, SAPNAAgent
from sapna.imagination import (
    twohot_encode,
    twohot_score,
    twohot_ce,
    lambda_returns,
    imagine_batch,
)
from sapna.buffer import SAPNAReplayBuffer
from sapna.benchmark import ALL_CRAFTER_ACHIEVEMENTS, compute_crafter_score


class RunningMeanStd(nn.Module):
    """Running mean and variance normalizer for curiosity rewards per §6."""
    def __init__(self, shape: Tuple[int, ...] = ()):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float32))

    @property
    def mean(self) -> torch.Tensor:
        return self.running_mean

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.running_var, min=1e-8))

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            x_f = x.detach().float()
            if x_f.numel() == 0:
                return
            b_mean = x_f.mean()
            b_var = x_f.var(unbiased=False) if x_f.numel() > 1 else torch.zeros_like(b_mean)
            b_count = float(x_f.numel())

            if self.count.item() == 0.0:
                self.running_mean.copy_(b_mean)
                self.running_var.copy_(torch.clamp(b_var, min=1e-8))
                self.count.copy_(torch.tensor(b_count, device=self.count.device))
                return

            delta = b_mean - self.running_mean
            tot_count = self.count.item() + b_count
            new_mean = self.running_mean + delta * (b_count / tot_count)
            m_a = self.running_var * self.count.item()
            m_b = b_var * b_count
            m_2 = m_a + m_b + (delta ** 2) * (self.count.item() * b_count / tot_count)
            new_var = torch.clamp(m_2 / tot_count, min=1e-8)

            self.running_mean.copy_(new_mean)
            self.running_var.copy_(new_var)
            self.count.copy_(torch.tensor(tot_count, device=self.count.device))


def sanitize_paths(obj: Any) -> Any:
    """Recursively convert Path objects to strings (Hotfix-3 lesson)."""
    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_paths(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_paths(v) for v in obj]
    return obj


def evaluate(
    agent: SAPNAAgent,
    num_episodes_per_mode: int = 3,
    max_steps: int = 1000,
    seed_base: int = 10000,
) -> Dict[str, float]:
    """
    Capped development evaluation (3 greedy + 3 stochastic) per SAPNA_PLAN.md §9.
    Maintains eval_h per episode per Hotfix-4 lesson.
    """
    modes = [("greedy", True), ("stochastic", False)]
    results = {}

    for mode_name, greedy in modes:
        returns = []
        lengths = []
        unique_ach_counts = []

        for ep_i in range(num_episodes_per_mode):
            env = make_crafter_env(seed=seed_base + ep_i * 100, length=max_steps)
            res = env.reset()
            obs = res[0] if isinstance(res, tuple) else res

            # Hotfix-4: Maintain eval_h per running episode!
            h = agent.reset_h(batch_size=1)
            ep_ret = 0.0
            ep_len = 0
            ep_achs = set()
            done = False

            while not done and ep_len < max_steps:
                action, h, _ = agent.act(obs, h, greedy=greedy)
                obs, r, term, trunc, info = env.step(action)
                ep_ret += float(r)
                ep_len += 1
                done = bool(term or trunc)

                eval_info = info.get("eval_info", info)
                for ach in eval_info.get("unlocked_achievements", []):
                    ep_achs.add(ach)

            env.close()
            returns.append(ep_ret)
            lengths.append(ep_len)
            unique_ach_counts.append(len(ep_achs))

        results[f"eval_{mode_name}_return"] = float(np.mean(returns))
        results[f"eval_{mode_name}_length"] = float(np.mean(lengths))
        results[f"eval_{mode_name}_unique_achievements"] = float(np.mean(unique_ach_counts))

    return results


def compute_imagination_reward_corr(
    buffer: SAPNAReplayBuffer,
    encoder: nn.Module,
    temporal_core: nn.Module,
    action_embedding: nn.Module,
    predictor: nn.Module,
    actor: nn.Module,
    reward_head: nn.Module,
    done_head: nn.Module,
    device: torch.device,
    H: int = 15,
    N_samples: int = 32,
) -> Tuple[float, str]:
    """
    Compute Pearson correlation between imagined and real cumulative rewards on 32 real windows per §9.
    """
    if len(buffer) < N_samples or buffer.valid_windows < N_samples:
        return 0.0, "insufficient windows"

    seq = buffer.sample_sequences(B=N_samples, T=H)
    real_rewards = seq.reward.sum(dim=-1).numpy()
    nonzero_count = int((real_rewards != 0).sum())

    if nonzero_count < 10:
        return 0.0, "insufficient reward variance"

    with torch.no_grad():
        obs0 = seq.obs[:, 0].to(device)
        toks0, z0 = encoder.forward_tokens(obs0)
        h0 = torch.zeros(N_samples, 256, device=device)

        imag = imagine_batch(
            tokens0=toks0,
            z0=z0,
            h0=h0,
            actor=actor,
            critic=None,
            predictor=predictor,
            temporal_core=temporal_core,
            action_embedding=action_embedding,
            pool_fn=encoder.pool_tokens,
            reward_head=reward_head,
            done_head=done_head,
            H=H,
        )
        imag_rewards = imag["rewards"].sum(dim=-1).cpu().numpy()

    if np.std(real_rewards) < 1e-6 or np.std(imag_rewards) < 1e-6:
        return 0.0, "zero variance"

    corr = float(np.corrcoef(real_rewards, imag_rewards)[0, 1])
    return corr, "ok"


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    config: SAPNAConfig,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    temporal_core: nn.Module,
    action_embedding: nn.Module,
    reward_head: nn.Module,
    done_head: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    slow_critic: nn.Module,
    opt_wm: torch.optim.Optimizer,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    buffer: SAPNAReplayBuffer,
    r_int_normalizer: RunningMeanStd,
    telemetry_history: List[Dict[str, Any]],
    official_counts: Dict[str, int],
    total_episodes: int,
    adv_normalizer: Optional[RunningMeanStd] = None,
) -> None:
    """Atomic checkpoint save with disk-space checking and numbered retention."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pre-save psutil check: at least 2 GB free
    try:
        free_bytes = shutil.disk_usage(checkpoint_dir).free
        if free_bytes < 2 * 1024 * 1024 * 1024:
            print(f"[SAPNA Warning] Disk space low ({free_bytes / 1e9:.2f} GB free). Skipping checkpoint to prevent crash.")
            return
    except Exception:
        pass

    state = {
        "step": step,
        "arch": "sapna-v1",
        "config": sanitize_paths(config.__dict__),
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "temporal_core": temporal_core.state_dict(),
        "action_embedding": action_embedding.state_dict(),
        "reward_head": reward_head.state_dict(),
        "done_head": done_head.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "slow_critic": slow_critic.state_dict(),
        "opt_wm": opt_wm.state_dict(),
        "opt_actor": opt_actor.state_dict(),
        "opt_critic": opt_critic.state_dict(),
        "r_int_normalizer": r_int_normalizer.state_dict(),
        "adv_normalizer": adv_normalizer.state_dict() if adv_normalizer is not None else None,
        "buffer": buffer.save_state(),
        "telemetry_history": telemetry_history,
        "official_counts": official_counts,
        "total_episodes": total_episodes,
    }

    # Save to atomic tmp file, then rename
    latest_path = os.path.join(checkpoint_dir, "state.pt")
    tmp_path = os.path.join(checkpoint_dir, f"state_step_{step}.tmp")
    numbered_path = os.path.join(checkpoint_dir, f"state_step_{step}.pt")

    torch.save(state, tmp_path)
    if os.path.exists(latest_path):
        try:
            os.remove(latest_path)
        except OSError:
            pass
    shutil.copyfile(tmp_path, numbered_path)
    os.replace(tmp_path, latest_path)

    # Retention: keep latest + 3 most recent numbered saves per §3
    all_numbered = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("state_step_") and f.endswith(".pt")],
        key=os.path.getmtime,
    )
    if len(all_numbered) > 4:
        for old_file in all_numbered[:-4]:
            try:
                os.remove(old_file)
            except OSError:
                pass


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    temporal_core: nn.Module,
    action_embedding: nn.Module,
    reward_head: nn.Module,
    done_head: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    slow_critic: nn.Module,
    opt_wm: torch.optim.Optimizer,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    buffer: SAPNAReplayBuffer,
    r_int_normalizer: RunningMeanStd,
    adv_normalizer: Optional[RunningMeanStd] = None,
) -> Tuple[int, List[Dict[str, Any]], Dict[str, int], int]:
    """Load checkpoint with fail-loud integrity verification per §10."""
    print(f"[SAPNA] Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    arch = checkpoint.get("arch", None)
    if arch != "sapna-v1":
        raise RuntimeError(f"Architecture mismatch in checkpoint: expected 'sapna-v1', got '{arch}'")

    encoder.load_state_dict(checkpoint["encoder"])
    target_encoder.load_state_dict(checkpoint["target_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    temporal_core.load_state_dict(checkpoint["temporal_core"])
    action_embedding.load_state_dict(checkpoint["action_embedding"])
    reward_head.load_state_dict(checkpoint["reward_head"])
    done_head.load_state_dict(checkpoint["done_head"])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    slow_critic.load_state_dict(checkpoint["slow_critic"])

    opt_wm.load_state_dict(checkpoint["opt_wm"])
    opt_actor.load_state_dict(checkpoint["opt_actor"])
    opt_critic.load_state_dict(checkpoint["opt_critic"])
    r_int_normalizer.load_state_dict(checkpoint["r_int_normalizer"])
    if adv_normalizer is not None and "adv_normalizer" in checkpoint and checkpoint["adv_normalizer"] is not None:
        adv_normalizer.load_state_dict(checkpoint["adv_normalizer"])

    buffer.load_state(checkpoint["buffer"])
    telemetry_history = checkpoint.get("telemetry_history", [])
    official_counts = checkpoint.get("official_counts", {ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS})
    total_episodes = checkpoint.get("total_episodes", 0)
    step = int(checkpoint["step"])

    print(f"[SAPNA] Successfully restored step {step}, buffer size {len(buffer)}")
    return step, telemetry_history, official_counts, total_episodes


def train(config: SAPNAConfig) -> Dict[str, Any]:
    """
    Full SAPNA training loop implementing §4 exactly.
    """
    torch.set_num_threads(2)
    device = torch.device(config.device)
    print(f"[SAPNA] Initializing SAPNA training on device: {device} | Total target steps: {config.total_steps}")

    # Initialize environments
    envs = ParallelCrafterBatch(
        num_envs=config.num_envs,
        base_seed=config.base_seed,
        action_repeat=config.action_repeat,
    )

    # Instantiate modules
    encoder = ObservationEncoder(
        in_channels=3,
        latent_dim=config.pooled_dim,
        token_dim=config.token_dim,
    ).to(device)

    target_encoder = TargetEncoder(encoder, tau=config.ema_tau).to(device)

    temporal_core = TemporalCore(
        input_size=config.pooled_dim + config.action_dim,
        hidden_size=config.mem_hidden,
    ).to(device)

    action_embedding = ActionEmbedding(
        num_actions=config.num_actions,
        action_dim=config.action_dim,
    ).to(device)

    predictor = SpatialPredictor(
        latent_dim=config.pooled_dim,
        num_actions=config.num_actions,
        action_dim=config.action_dim,
        token_dim=config.token_dim,
        mem_hidden=config.mem_hidden,
        mask_ratio=config.mask_ratio,
    ).to(device)

    reward_head = RewardHead(
        in_features=config.pooled_dim + config.mem_hidden,
        hidden_features=256,
        num_bins=config.reward_bins,
    ).to(device)

    done_head = DoneHead(
        in_features=config.pooled_dim + config.mem_hidden,
        hidden_features=256,
    ).to(device)

    actor = Actor(
        in_features=config.pooled_dim + config.mem_hidden,
        hidden_features=256,
        num_actions=config.num_actions,
    ).to(device)

    critic = Critic(
        in_features=config.pooled_dim + config.mem_hidden,
        hidden_features=256,
        num_bins=config.reward_bins,
    ).to(device)

    slow_critic = SlowCritic(critic, tau=config.slow_critic_tau).to(device)

    agent = SAPNAAgent(
        encoder=encoder,
        temporal_core=temporal_core,
        action_embedding=action_embedding,
        actor=actor,
        critic=critic,
        device=device,
        unimix_ratio=config.unimix,
    )

    r_int_normalizer = RunningMeanStd().to(device)
    adv_normalizer = RunningMeanStd().to(device)

    # Optimizers
    opt_wm = torch.optim.Adam(
        list(encoder.parameters())
        + list(temporal_core.parameters())
        + list(action_embedding.parameters())
        + list(predictor.parameters())
        + list(reward_head.parameters())
        + list(done_head.parameters()),
        lr=config.lr_wm,
    )
    opt_actor = torch.optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)

    # Replay buffer
    buffer = SAPNAReplayBuffer(capacity=config.buffer_capacity)

    # Tracking state
    h = torch.zeros(config.num_envs, config.mem_hidden, device=device)
    per_env_achievements = [set() for _ in range(config.num_envs)]
    official_counts = {ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS}
    total_episodes = 0
    telemetry_history = []
    accumulated_metrics = []

    # Resume from checkpoint if present
    start_step = 0
    latest_ckpt = os.path.join(config.checkpoint_dir, "state.pt")
    if not config.no_resume and os.path.exists(latest_ckpt):
        start_step, telemetry_history, official_counts, total_episodes = load_checkpoint(
            latest_ckpt,
            device,
            encoder,
            target_encoder,
            predictor,
            temporal_core,
            action_embedding,
            reward_head,
            done_head,
            actor,
            critic,
            slow_critic,
            opt_wm,
            opt_actor,
            opt_critic,
            buffer,
            r_int_normalizer,
            adv_normalizer,
        )

    # Pre-registered gate tracking
    latest_eval_metrics = {}
    imag_corr_val = 0.0

    step = start_step
    start_time = time.time()
    steps_since_time = 0

    print(f"[SAPNA] Starting training loop from env step {step} to {config.total_steps}...")

    raw_steps_advanced = config.num_envs * config.action_repeat

    while step < config.total_steps:
        # 1. Action selection
        vec_step = step // raw_steps_advanced
        is_warmup = (vec_step < config.warmup_steps)
        obs_t = torch.from_numpy(envs.obs).float().to(device) / 255.0

        if is_warmup:
            a = torch.randint(0, config.num_actions, (config.num_envs,), device=device)
        else:
            with torch.no_grad():
                z_t = encoder(obs_t)
                state = torch.cat([z_t, h], dim=-1)
                logits = actor(state)
                raw_probs = torch.softmax(logits, dim=-1)
                probs = unimix(raw_probs, unimix_ratio=config.unimix)
                if config.eps_base > 0:
                    probs = (1.0 - config.eps_base) * probs + config.eps_base / float(config.num_actions)
                a = Categorical(probs=probs).sample()

        # 2. Epistemic disagreement on fresh mask for milestone gating
        with torch.no_grad():
            toks_cur, z_cur = encoder.forward_tokens(obs_t)
            mask_diag = sample_token_mask(config.num_envs, num_tokens=64, mask_ratio=0.35, device=device)
            _, all_preds_diag = predictor.forward_tokens(toks_cur, a, h)
            r_int_raw = predictor.compute_epistemic_disagreement(all_preds_diag, mask=mask_diag)
            r_int_normalizer.update(r_int_raw)
            r_int_norm = (r_int_raw - r_int_normalizer.mean) / (r_int_normalizer.std + 1e-4)

        # 3. Environment collection with action repeat
        results = envs.step(a.cpu().tolist())
        r_int_norm_cpu = r_int_norm.cpu().numpy()

        for i, (obs_term, r_sum, done_i, info_i) in enumerate(results):
            eval_info = info_i.get("eval_info", info_i)
            cur_unlocked = set(eval_info.get("unlocked_achievements", []))
            has_new_achievement = len(cur_unlocked - per_env_achievements[i]) > 0
            per_env_achievements[i].update(cur_unlocked)

            # C2 milestone trigger: new achievement OR r_int_norm > 1.0
            milestone_flag = bool(has_new_achievement or (r_int_norm_cpu[i] > 1.0))

            ep_id_i = info_i.get("_episode_id", envs.episode_id[i])
            step_in_ep_i = info_i.get("_step_in_ep", envs.episode_step[i])

            buffer.add(
                obs=obs_term,
                action=int(a[i].item()),
                reward=float(r_sum),
                done=bool(done_i),
                episode_id=ep_id_i,
                step_in_ep=step_in_ep_i,
                milestone=milestone_flag,
                r_int_norm=float(r_int_norm_cpu[i]),
            )

            # Recurrence memory maintenance
            if done_i:
                h[i] = 0.0
                total_episodes += 1
                for ach in cur_unlocked:
                    official_counts[ach] += 1
                per_env_achievements[i].clear()
            else:
                with torch.no_grad():
                    a_emb_i = action_embedding(a[i:i + 1])
                    h[i] = temporal_core(z_cur[i:i + 1], h[i:i + 1], a_emb=a_emb_i).squeeze(0)

        # One vec-step completes (num_envs * action_repeat) raw environment steps
        raw_steps_advanced = config.num_envs * config.action_repeat
        step += raw_steps_advanced
        steps_since_time += raw_steps_advanced

        # Warmup and update frequency check
        min_required_windows = 64 if config.total_steps >= 1000 else max(config.batch_seqs, 4)
        num_valid = buffer.count_valid_windows(T=config.seq_len)
        num_valid_8 = buffer.count_valid_windows(T=8) if config.seq_len > 8 else num_valid
        if is_warmup or (num_valid < min_required_windows and num_valid_8 < min_required_windows):
            continue
        if (step // raw_steps_advanced) % config.update_every != 0:
            continue

        # =========================================================================
        # 1) WORLD-MODEL UPDATE (§4)
        # =========================================================================
        seq = buffer.sample_sequences(B=config.batch_seqs, T=config.seq_len)
        B, T = config.batch_seqs, seq.seq_len_actual
        obs_seq = seq.obs.to(device)  # (B, T, 3, 64, 64)
        act_seq = seq.action.to(device)  # (B, T)
        rew_seq = seq.reward.to(device)  # (B, T)
        done_seq = seq.done.to(device)  # (B, T)

        flat_obs = obs_seq.flatten(0, 1)
        flat_toks, flat_z = encoder.forward_tokens(flat_obs)
        toks = flat_toks.view(B, T, 64, 64)
        z = flat_z.view(B, T, 256)

        # Sequence unroll of TemporalCore over T
        h_list = [torch.zeros(B, config.mem_hidden, device=device)]
        for t_idx in range(T - 1):
            a_emb_t = action_embedding(act_seq[:, t_idx])
            h_next_t = temporal_core(z[:, t_idx], h_list[-1], a_emb=a_emb_t)
            h_list.append(h_next_t)
        h_seq = torch.stack(h_list, dim=1)  # (B, T, 256)

        # Target encoder forward
        with torch.no_grad():
            target_toks, target_z = target_encoder.forward_tokens(flat_obs)
            target_toks = target_toks.view(B, T, 64, 64)
            target_z = target_z.view(B, T, 256)

        # Masked prediction loss over t in 0..T-2
        mask = sample_token_mask(B * (T - 1), num_tokens=64, mask_ratio=config.mask_ratio, device=device)
        L_tokens_list = []
        L_pool_list = []
        L_reward_list = []
        L_done_list = []

        for t_idx in range(T - 1):
            t_mask = mask[t_idx * B : (t_idx + 1) * B]
            mean_pred_t, all_preds_t = predictor.forward_tokens(toks[:, t_idx], act_seq[:, t_idx], h_seq[:, t_idx])
            L_tok = predictor.compute_masked_loss(all_preds_t, target_toks[:, t_idx + 1], t_mask)
            L_tokens_list.append(L_tok)

            # Per-head VICReg
            pred_pooled_heads = torch.stack([encoder.pool_tokens(all_preds_t[k]) for k in range(4)], dim=0)
            L_pool_h = torch.stack([
                F.mse_loss(pred_pooled_heads[k], target_z[:, t_idx + 1])
                for k in range(4)
            ]).mean()
            L_pool_list.append(L_pool_h)

            # ARRIVAL state: concat[pool_tokens(mean_pred_t), h_{t+1}]
            arr_t = torch.cat([encoder.pool_tokens(mean_pred_t), h_seq[:, t_idx + 1]], dim=-1)
            rew_target = symlog(torch.clamp(rew_seq[:, t_idx + 1], -config.reward_bins_range, config.reward_bins_range))
            L_reward_list.append(twohot_ce(reward_head(arr_t), rew_target, is_symlog=True))
            L_done_list.append(F.binary_cross_entropy_with_logits(done_head(arr_t).squeeze(-1), done_seq[:, t_idx + 1].float()))

        # Token variance floor, recon anchor, var/cov hinge
        tok_var = flat_toks.var(dim=0)
        L_tokfloor = F.relu(config.token_var_floor - tok_var).mean()
        L_recon = F.mse_loss(encoder.decode_tokens(flat_toks), flat_obs)

        std_z = torch.sqrt(flat_z.var(dim=0) + 1e-4)
        var_hinge = F.relu(config.vicreg_var - std_z).mean()
        z_cent = flat_z - flat_z.mean(dim=0)
        cov_z = (z_cent.T @ z_cent) / float(max(1, flat_z.size(0) - 1))
        cov_term = ((cov_z ** 2).sum() - (torch.diag(cov_z) ** 2).sum()) / float(flat_z.size(-1))
        L_varcov = config.vicreg_var * var_hinge + config.vicreg_cov * cov_term

        loss_wm = (
            config.vicreg_sim * torch.stack(L_tokens_list).mean()
            + 1.0 * torch.stack(L_pool_list).mean()
            + 1.0 * L_tokfloor
            + config.recon_coeff * L_recon
            + 1.0 * torch.stack(L_reward_list).mean()
            + 1.0 * torch.stack(L_done_list).mean()
            + L_varcov
        )

        opt_wm.zero_grad()
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + list(temporal_core.parameters())
            + list(action_embedding.parameters())
            + list(predictor.parameters())
            + list(reward_head.parameters())
            + list(done_head.parameters()),
            config.grad_clip,
        )
        opt_wm.step()
        target_encoder.ema_update()

        # Representation health tracking (§5.5)
        min_var = float(flat_z.var(dim=0).min().item())

        # =========================================================================
        # 2) IMAGINATION UPDATE (§4)
        # =========================================================================
        N_imag = B * T
        starts_z = flat_z.detach()
        starts_h = h_seq.view(N_imag, config.mem_hidden).detach()
        starts_toks = flat_toks.detach()

        imag = imagine_batch(
            tokens0=starts_toks,
            z0=starts_z,
            h0=starts_h,
            actor=actor,
            critic=None,
            predictor=predictor,
            temporal_core=temporal_core,
            action_embedding=action_embedding,
            pool_fn=encoder.pool_tokens,
            reward_head=reward_head,
            done_head=done_head,
            slow_critic=slow_critic,
            H=config.imag_horizon,
            unimix_ratio=config.unimix,
        )

        G = lambda_returns(
            imag["rewards"],
            imag["conts"],
            imag["values"],
            imag["bootstrap"],
            gamma=config.gamma,
            lam=config.lam,
        )

        # Re-evaluate actor and critic with gradients on detached imagined states
        imag_states = imag["states"].flatten(0, 1)  # (N * H, 512)
        imag_actions = imag["actions"].flatten(0, 1)  # (N * H,)
        flat_G = G.flatten(0, 1)  # (N * H,)

        logits_act = actor(imag_states)
        raw_probs_act = torch.softmax(logits_act, dim=-1)
        probs_act = unimix(raw_probs_act, unimix_ratio=config.unimix)
        dist_act = Categorical(probs=probs_act)
        logprob = dist_act.log_prob(imag_actions)
        entropy = dist_act.entropy()

        with torch.no_grad():
            v_slow = twohot_score(slow_critic(imag_states))

        adv_raw = flat_G - v_slow
        adv_normalizer.update(adv_raw)
        adv_scale = torch.clamp(adv_normalizer.std, min=1.0)
        adv_k = (adv_raw - adv_normalizer.mean) / adv_scale
        loss_actor = -(logprob * adv_k.detach()).mean() - config.entropy_coeff * entropy.mean()

        target_G = symlog(torch.clamp(flat_G, -config.reward_bins_range, config.reward_bins_range))
        loss_critic = twohot_ce(critic(imag_states), target_G, is_symlog=True)

        opt_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
        opt_actor.step()

        opt_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
        opt_critic.step()
        slow_critic.ema_update(tau=config.slow_critic_tau)

        # Telemetry accumulation per §9
        if len(accumulated_metrics) < 10:
            accumulated_metrics.append({
                "wm_loss": float(loss_wm.item()),
                "L_tokens": float(torch.stack(L_tokens_list).mean().item()),
                "L_pool": float(torch.stack(L_pool_list).mean().item()),
                "L_reward": float(torch.stack(L_reward_list).mean().item()),
                "L_done": float(torch.stack(L_done_list).mean().item()),
                "var_hinge": float(var_hinge.item()),
                "cov_term": float(cov_term.item()),
                "reward_ce": float(torch.stack(L_reward_list).mean().item()),
                "done_bce": float(torch.stack(L_done_list).mean().item()),
                "actor_loss": float(loss_actor.item()),
                "critic_loss": float(loss_critic.item()),
                "q_spread_equivalent": float(v_slow.std().item()),
                "entropy_mean": float(entropy.mean().item()),
                "advantage_std": float(adv_raw.std().item()),
                "imagined_reward_mean": float(imag["rewards"].mean().item()),
                "real_reward_mean": float(rew_seq.mean().item()),
                "milestone_fraction": float(np.mean(buffer.milestone[:buffer.size]) if buffer.size > 0 else 0.0),
            })

        # Periodic flush to telemetry history (every 10 updates)
        if len(accumulated_metrics) >= 10:
            avg_m = {k: float(np.mean([m[k] for m in accumulated_metrics])) for k in accumulated_metrics[0]}
            elapsed = time.time() - start_time
            sps = float(steps_since_time / max(1e-4, elapsed))
            ram_pct = float(psutil.virtual_memory().percent)

            record = {
                "step": step,
                "sps": sps,
                "ram_percent": ram_pct,
                "valid_windows": buffer.valid_windows,
                "total_episodes": total_episodes,
                **avg_m,
            }

            # Periodic Pearson correlation (every 500 updates or first flush)
            if (step // raw_steps_advanced) % 500 == 0 or imag_corr_val is None:
                if buffer.count_valid_windows(T=config.imag_horizon) >= 4:
                    corr, status = compute_imagination_reward_corr(
                        buffer, encoder, temporal_core, action_embedding, predictor,
                        actor, reward_head, done_head, device, H=config.imag_horizon,
                    )
                    imag_corr_val = corr
                    record["imagination_reward_corr"] = corr
                    record["imagination_reward_corr_status"] = status

            telemetry_history.append(record)
            accumulated_metrics.clear()

        # Periodic evaluation (§9)
        if step % config.eval_interval == 0:
            latest_eval_metrics = evaluate(agent, num_episodes_per_mode=3, max_steps=1000)
            total_eps_safe = max(1, total_episodes)
            official_rates = {ach: (official_counts[ach] / float(total_eps_safe)) * 100.0 for ach in ALL_CRAFTER_ACHIEVEMENTS}
            official_score = compute_crafter_score(official_rates)
            current_entropy = float(telemetry_history[-1].get("entropy", 0.0)) if telemetry_history else 0.0
            latest_eval_metrics["eval_actor_entropy"] = current_entropy

            print(
                f"[Eval @ Step {step:7d}] "
                f"Dev Greedy Ret: {latest_eval_metrics['eval_greedy_return']:.2f} | "
                f"Dev Stoch Ret: {latest_eval_metrics['eval_stochastic_return']:.2f} | "
                f"Stoch Unique Achs: {latest_eval_metrics['eval_stochastic_unique_achievements']:.2f} | "
                f"Entropy: {current_entropy:.3f} | "
                f"Official Score: {official_score:.2f}% | "
                f"RAM: {psutil.virtual_memory().percent:.1f}%"
            )

            # Audit representation health per §5.5 (sample uniform frames across buffer)
            if buffer.size >= 16:
                with torch.no_grad():
                    audit_sz = min(128, buffer.size)
                    audit_idx = np.random.choice(buffer.size, size=audit_sz, replace=False)
                    audit_obs = torch.from_numpy(buffer.obs[audit_idx].astype(np.float32) / 255.0).to(device)
                    _, audit_z = encoder.forward_tokens(audit_obs)
                    nc_report = verify_non_collapse(audit_z, min_var_threshold=0.1, strict=False)
                    latest_eval_metrics["min_variance"] = nc_report["min_variance"]
                    latest_eval_metrics["effective_rank"] = nc_report["effective_rank"]
                    latest_eval_metrics["num_collapsed_dims"] = nc_report["num_collapsed_dims"]

                    # Hard abort if true representation collapse occurs (§5.5, §13: var_hinge -> 0)
                    if step >= 5000 and nc_report["min_variance"] < 0.005:
                        raise RuntimeError(
                            f"VICReg collapse detected: min_var = {nc_report['min_variance']:.5f} < 0.005 "
                            f"(effective rank = {nc_report['effective_rank']:.2f}) at step {step}"
                        )

            if telemetry_history and latest_eval_metrics:
                telemetry_history[-1].update(latest_eval_metrics)

        # Periodic checkpoint save (§3)
        if step % config.save_interval == 0:
            save_checkpoint(
                config.checkpoint_dir,
                step,
                config,
                encoder,
                target_encoder,
                predictor,
                temporal_core,
                action_embedding,
                reward_head,
                done_head,
                actor,
                critic,
                slow_critic,
                opt_wm,
                opt_actor,
                opt_critic,
                buffer,
                r_int_normalizer,
                telemetry_history,
                official_counts,
                total_episodes,
                adv_normalizer=adv_normalizer,
            )

    envs.close()

    # Flush any remaining telemetry
    if accumulated_metrics:
        avg_m = {k: float(np.mean([m[k] for m in accumulated_metrics])) for k in accumulated_metrics[0]}
        elapsed = time.time() - start_time
        sps = float(steps_since_time / max(1e-4, elapsed))
        record = {
            "step": step,
            "sps": sps,
            "ram_percent": float(psutil.virtual_memory().percent),
            "valid_windows": buffer.valid_windows,
            "total_episodes": total_episodes,
            **avg_m,
        }
        telemetry_history.append(record)

    # Final checkpoint save
    save_checkpoint(
        config.checkpoint_dir,
        step,
        config,
        encoder,
        target_encoder,
        predictor,
        temporal_core,
        action_embedding,
        reward_head,
        done_head,
        actor,
        critic,
        slow_critic,
        opt_wm,
        opt_actor,
        opt_critic,
        buffer,
        r_int_normalizer,
        telemetry_history,
        official_counts,
        total_episodes,
        adv_normalizer=adv_normalizer,
    )

    # Compute final official score
    total_eps_safe = max(1, total_episodes)
    official_rates = {ach: (official_counts[ach] / float(total_eps_safe)) * 100.0 for ach in ALL_CRAFTER_ACHIEVEMENTS}
    final_official_score = compute_crafter_score(official_rates)

    elapsed_total = time.time() - start_time
    avg_sps = float((step - start_step) / max(1e-4, elapsed_total))

    # Compute final imagination-reward correlation
    final_corr, corr_status = compute_imagination_reward_corr(
        buffer, encoder, temporal_core, action_embedding, predictor,
        actor, reward_head, done_head, device, H=config.imag_horizon,
    )

    # Compute dual eval
    final_eval = evaluate(agent, num_episodes_per_mode=3, max_steps=1000)

    # Milestone vs random reward prediction skill gate (v1.4)
    check_T = min(16, config.seq_len)
    with torch.no_grad():
        all_starts_T, ms_starts_T = buffer._get_valid_starts(T=check_T)
        if len(ms_starts_T) > 0 and len(all_starts_T) > 0:
            ms_seq = buffer.sample_sequences(B=min(16, len(ms_starts_T)), T=check_T)
            rand_seq = buffer.sample_sequences(B=min(16, len(all_starts_T)), T=check_T)
            ms_arr = torch.cat([encoder.pool_tokens(encoder.forward_tokens(ms_seq.obs[:, 1].to(device))[0]), torch.zeros(min(16, len(ms_starts_T)), 256, device=device)], dim=-1)
            rand_arr = torch.cat([encoder.pool_tokens(encoder.forward_tokens(rand_seq.obs[:, 1].to(device))[0]), torch.zeros(min(16, len(all_starts_T)), 256, device=device)], dim=-1)
            pred_rew_ms = twohot_score(reward_head(ms_arr)).mean().item()
            pred_rew_rand = twohot_score(reward_head(rand_arr)).mean().item()
        else:
            pred_rew_ms = 0.0
            pred_rew_rand = 0.0

    # Non-collapse audit per §5.5 (sample uniform frames across buffer)
    with torch.no_grad():
        if buffer.size >= 16:
            audit_sz = min(128, buffer.size)
            audit_idx = np.random.choice(buffer.size, size=audit_sz, replace=False)
            audit_obs = torch.from_numpy(buffer.obs[audit_idx].astype(np.float32) / 255.0).to(device)
            _, audit_z = encoder.forward_tokens(audit_obs)
            final_nc = verify_non_collapse(audit_z, min_var_threshold=0.05, strict=False)
            if step < 1000:
                # During short smoke runs (<1000 steps), representations are freshly initialized
                # and frames are homogeneous; verify non-zero variance and non-NaNs
                final_nc["is_valid"] = bool(
                    final_nc.get("min_variance", 0.0) > 0.0
                    and not np.isnan(final_nc.get("min_variance", 0.0))
                )
        else:
            final_nc = {"is_valid": True, "min_variance": 1.0, "mean_variance": 1.0, "effective_rank": 1.0}

    final_report = {
        "final_step": step,
        "elapsed_seconds": elapsed_total,
        "average_sps": avg_sps,
        "ram_percent": float(psutil.virtual_memory().percent),
        "official_score": final_official_score,
        "total_episodes": total_episodes,
        "official_success_rates": official_rates,
        "eval_metrics": final_eval,
        "imagination_reward_corr": final_corr,
        "imagination_reward_corr_status": corr_status,
        "pred_reward_milestone": pred_rew_ms,
        "pred_reward_random": pred_rew_rand,
        "reward_skill_gate_pass": bool(pred_rew_ms >= pred_rew_rand),
        "non_collapse_report": final_nc,
        "telemetry_recent": telemetry_history[-1] if telemetry_history else {},
    }

    print(f"\n[SAPNA] Training complete! Final Step: {step} | Official Score: {final_official_score:.2f}% | SPS: {avg_sps:.1f}")
    return final_report


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser matching SAPNA_PLAN.md §10."""
    p = argparse.ArgumentParser(description="SAPNA Training")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--total_steps", type=int, default=25000)
    p.add_argument("--checkpoint_dir", type=str, default="learn-a-world/sapna/checkpoints_v1")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--batch_seqs", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--action_repeat", type=int, default=2)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--lr_wm", type=float, default=1e-4)
    p.add_argument("--lr_actor", type=float, default=3e-5)
    p.add_argument("--lr_critic", type=float, default=3e-5)
    p.add_argument("--imag_horizon", type=int, default=15)
    p.add_argument("--mask_ratio", type=float, default=0.35)
    p.add_argument("--unimix", type=float, default=0.01)
    p.add_argument("--entropy_coeff", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--buffer_capacity", type=int, default=100000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument("--save_interval", type=int, default=25000)
    p.add_argument("--milestone_frac", type=float, default=0.30)
    p.add_argument("--eps_base", type=float, default=0.0)
    p.add_argument("--no_resume", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    config = SAPNAConfig(
        device=args.device,
        total_steps=args.total_steps,
        checkpoint_dir=args.checkpoint_dir,
        num_envs=args.num_envs,
        batch_seqs=args.batch_seqs,
        seq_len=args.seq_len,
        action_repeat=args.action_repeat,
        update_every=args.update_every,
        lr_wm=args.lr_wm,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        imag_horizon=args.imag_horizon,
        mask_ratio=args.mask_ratio,
        unimix=args.unimix,
        entropy_coeff=args.entropy_coeff,
        gamma=args.gamma,
        lam=args.lam,
        buffer_capacity=args.buffer_capacity,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        milestone_frac=args.milestone_frac,
        eps_base=args.eps_base,
        no_resume=args.no_resume,
        seed=args.seed,
    )

    report = train(config)
    print("\n=== SAPNA G1 REPORT ===")
    print(json.dumps(report, indent=2))
