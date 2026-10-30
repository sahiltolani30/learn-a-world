"""HeadTrainer: latent caching and head training with gate verification.

Backbone is frozen; latent caching encodes obs and runs GRU once per shard.
All downstream heads train strictly on cached (z, h) latents without backbone forwards.
Logs training metrics and verifies quality gates after each iteration.
"""

from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from atlas.backbone import FrozenBackbone
from atlas.collect import ShardLoader
from atlas.config import LATENT_DIM, NUM_ACTIONS, RECURRENT_DIM, STATE_DIM, TrainConfig
from atlas.critic import CriticNet, critic_loss, symlog, two_hot_encode
from atlas.epistemic import EpistemicNet
from atlas.readout import VitalsReadout
from atlas.safety import ContinueNet, focal_bce
from atlas.skills import OptionNet, ReachabilityNet, build_her_batch, compute_her_loss, build_validation_reach_pairs


def compute_discounted_returns(rewards: np.ndarray, gamma: float = 0.97, horizon: int = 25) -> np.ndarray:
    """Computes Monte-Carlo discounted returns over horizon H (V3 F1)."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    for t in range(T):
        h = min(horizon, T - t)
        discounts = gamma ** np.arange(h)
        returns[t] = float(np.sum(rewards[t:t + h] * discounts))
    return returns


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based AUROC in pure numpy."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    n_pos = int(np.sum(labels))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(scores)) + 1
    sum_pos_ranks = float(np.sum(ranks[labels]))
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class HeadTrainer:
    """Trains OptionNet, ReachabilityNet, EpistemicNet, ContinueNet, and VitalsReadout."""

    def __init__(
        self,
        backbone: FrozenBackbone,
        option_net: OptionNet,
        reach_net: ReachabilityNet,
        epistemic_net: EpistemicNet,
        safety_net: ContinueNet,
        readout_net: Optional[VitalsReadout] = None,
        critic_net: Optional[CriticNet] = None,
        cfg: Optional[TrainConfig] = None,
        log_dir: Optional[Union[str, Path]] = None,
    ):
        self.backbone = backbone
        self.device = self.backbone.device
        self.option_net = option_net
        self.reach_net = reach_net
        self.epistemic_net = epistemic_net
        self.safety_net = safety_net
        self.readout_net = readout_net
        self.critic_net = critic_net or CriticNet(cfg=getattr(self.option_net, "cfg", None)).to(self.device)
        self.cfg = cfg or TrainConfig()

        self.log_dir = Path(log_dir) if log_dir else Path("atlas_runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train_log.jsonl"

    def cache_latents(
        self,
        shard_path: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Encodes observations and computes GRU latents, saving sidecar *.latents.npz."""
        p = Path(shard_path)
        if cache_dir is not None:
            c_dir = Path(cache_dir)
            c_dir.mkdir(parents=True, exist_ok=True)
            out_path = c_dir / f"{p.stem}.latents.npz"
        else:
            out_path = p.with_suffix(".latents.npz")

        for candidate in [out_path, p.with_suffix(".latents.npz")]:
            if candidate.exists():
                try:
                    if candidate.stat().st_mtime >= p.stat().st_mtime:
                        with np.load(candidate) as c_data:
                            if "z" in c_data:
                                shard_meta = ShardLoader.load_shard(p, load_obs=False)
                                if len(c_data["z"]) == len(shard_meta["actions"]):
                                    return candidate
                except Exception:
                    pass

        shard = ShardLoader.load_shard(p, load_obs=True)
        obs = shard["obs"]
        actions = shard["actions"]
        ep_lengths = shard["episode_lengths"]

        T_total = len(actions)
        z_list: List[np.ndarray] = []

        # 1) Batched visual encoding (chunk size 256 for tight memory bounds)
        batch_size = 256
        for i in range(0, T_total, batch_size):
            obs_batch = obs[i: i + batch_size]
            with torch.no_grad():
                z_batch = self.backbone.encode(obs_batch)
            z_list.append(z_batch.cpu().numpy())
            del obs_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        z_arr = np.concatenate(z_list, axis=0)
        del z_list

        # Free raw pixel observation images immediately before recurrence
        del obs
        del shard
        gc.collect()

        # 2) GRU sequence recurrence per episode
        h_arr = np.zeros((T_total, RECURRENT_DIM), dtype=np.float32)
        target_residuals = np.zeros((T_total, LATENT_DIM), dtype=np.float32)

        start = 0
        for ep_len in ep_lengths:
            end = start + ep_len
            z_ep = torch.from_numpy(z_arr[start:end]).float().to(self.device)
            a_ep = torch.from_numpy(actions[start:end]).to(self.device)

            h_t = torch.zeros(1, RECURRENT_DIM, device=self.device)
            for t in range(ep_len):
                if t > 0:
                    with torch.no_grad():
                        h_t = self.backbone.recurrent_step(z_ep[t: t + 1], a_ep[t - 1: t], h_t)
                h_arr[start + t] = h_t.cpu().numpy().squeeze(0)
                if t < ep_len - 1:
                    with torch.no_grad():
                        z_next_pred = self.backbone.model.predictor(z_ep[t: t + 1], h_t, a_ep[t: t + 1])
                        residual = z_ep[t + 1: t + 2] - z_next_pred
                        target_residuals[start + t] = residual.cpu().numpy().squeeze(0)

            start = end
            del z_ep, a_ep

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Write to local temp file first to avoid network FUSE memory buffering
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        np.savez_compressed(
            tmp_path,
            z=z_arr,
            h=h_arr,
            target_residuals=target_residuals,
        )
        shutil.move(str(tmp_path), str(out_path))

        del z_arr, h_arr, target_residuals
        gc.collect()
        return out_path

    def train_epoch(
        self,
        train_eps: List[Dict[str, Any]],
        val_eps: List[Dict[str, Any]],
        iter_idx: int,
        extra_log: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs one training iteration over cached episodes and evaluates quality gates."""
        # Setup optimizers
        params_skills = list(self.option_net.parameters()) + list(self.reach_net.parameters())
        opt_skills = optim.Adam(params_skills, lr=3e-4)
        opt_epistemic = optim.Adam(self.epistemic_net.parameters(), lr=3e-4)
        opt_safety = optim.Adam(self.safety_net.parameters(), lr=3e-4)

        opt_readout = None
        if self.readout_net is not None:
            opt_readout = optim.Adam(self.readout_net.parameters(), lr=3e-4)

        opt_critic = optim.Adam(
            self.critic_net.parameters(),
            lr=getattr(skills_cfg if 'skills_cfg' in locals() else getattr(self.option_net, "cfg", None), "critic_lr", 3e-4),
        )

        skills_cfg = getattr(self.option_net, "cfg", None)
        c_gamma = getattr(skills_cfg, "critic_gamma", 0.97)
        c_horizon = getattr(skills_cfg, "critic_horizon", 25)
        c_epochs = getattr(skills_cfg, "critic_epochs_per_iter", 8)

        # Build flat transition tensors
        z_train = np.concatenate([ep["z"] for ep in train_eps], axis=0)
        h_train = np.concatenate([ep["h"] for ep in train_eps], axis=0)
        s_train = np.concatenate([z_train, h_train], axis=-1)
        a_train = np.concatenate([ep["actions"] for ep in train_eps], axis=0)
        res_train = np.concatenate([ep["target_residuals"] for ep in train_eps], axis=0)
        dones_train = np.concatenate([ep["dones"] for ep in train_eps], axis=0)
        inv_train = np.concatenate([ep["inventory"] for ep in train_eps], axis=0)

        # Compute Monte-Carlo discounted returns for critic (V3 F1)
        g_list = [compute_discounted_returns(ep["rewards"], gamma=c_gamma, horizon=c_horizon) for ep in train_eps]
        g_train = np.concatenate(g_list, axis=0)

        s_t = torch.from_numpy(s_train).float().to(self.device)
        a_t = torch.from_numpy(a_train).long().to(self.device)
        res_t = torch.from_numpy(res_train).float().to(self.device)
        dones_t = torch.from_numpy(dones_train).float().to(self.device)
        inv_t = torch.from_numpy(inv_train).float().to(self.device)

        # Two-hot targets for CriticNet
        v_train = symlog(torch.from_numpy(g_train).float().to(self.device))
        target_twohot_all = two_hot_encode(
            v_train,
            num_bins=self.critic_net.num_bins,
            v_min=self.critic_net.v_min,
            v_max=self.critic_net.v_max,
        )

        N = len(s_t)
        batch_size = min(self.cfg.batch_size, max(1, N))
        max_epoch_samples = min(N, 32768)

        # Train CriticNet (8 epochs, V3 F1)
        critic_ce = 0.0
        critic_val_mean = 0.0
        critic_val_p90 = 0.0
        if N > 0:
            for c_ep in range(c_epochs):
                indices = np.random.permutation(N)[:max_epoch_samples]
                for b_start in range(0, max_epoch_samples, batch_size):
                    b_idx = indices[b_start: b_start + batch_size]
                    b_s = s_t[b_idx]
                    b_target = target_twohot_all[b_idx]
                    opt_critic.zero_grad()
                    logits = self.critic_net(b_s)
                    loss_c = critic_loss(logits, b_target)
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1.0)
                    opt_critic.step()
                    self.critic_net.update_target()

            with torch.no_grad():
                val_sub = s_t[:min(1024, N)]
                val_logits = self.critic_net(val_sub)
                val_twohot = target_twohot_all[:min(1024, N)]
                critic_ce = float(critic_loss(val_logits, val_twohot).item())
                val_vals = self.critic_net.value(val_sub).detach().cpu().numpy()
                critic_val_mean = float(np.mean(val_vals))
                critic_val_p90 = float(np.percentile(val_vals, 90))

        # Build HER batch with trained critic (AWR weighting, V3 F2)
        max_her = getattr(self.cfg, "her_max_pairs", 32_768)
        r_max = getattr(skills_cfg, "her_reward_weight_max", 4.0)
        u_weight = getattr(skills_cfg, "her_unlock_weight", 2.0)
        p_gain = getattr(skills_cfg, "progress_weight_gain", 3.0)
        horizons = getattr(skills_cfg, "her_horizons", (5, 10))
        her_train = build_her_batch(
            train_eps,
            horizons=horizons,
            max_samples=max_her,
            her_reward_weight_max=r_max,
            her_unlock_weight=u_weight,
            progress_weight_gain=p_gain,
            critic=self.critic_net,
            critic_gamma=c_gamma,
            device=self.device,
        )
        her_val = (
            build_her_batch(
                val_eps,
                horizons=horizons,
                max_samples=max_her,
                her_reward_weight_max=r_max,
                her_unlock_weight=u_weight,
                progress_weight_gain=p_gain,
                critic=self.critic_net,
                critic_gamma=c_gamma,
                device=self.device,
            )
            if val_eps
            else None
        )

        initial_ce_loss = 0.0
        final_ce_loss = 0.0
        last_skills_metrics: Dict[str, float] = {}

        n_her = her_train["s_pos"].shape[0] if her_train is not None else 0
        skill_epochs = getattr(self.cfg, "skill_epochs_per_iter", 16)
        other_epochs = self.cfg.epochs_per_iter
        print(f"  [Train] HER batch: {n_her} pairs (skill epochs: {skill_epochs}) | Transitions: {N} (other epochs: {other_epochs})")

        max_epoch_samples = min(N, 32768)

        # 1) Train OptionNet & ReachabilityNet (V2.2 F5 skill-epoch boost: 16 epochs)
        if her_train is not None and her_train["s_pos"].shape[0] > 0:
            for s_ep in range(skill_epochs):
                opt_skills.zero_grad()
                loss_skills, metrics_skills = compute_her_loss(self.option_net, self.reach_net, her_train)
                loss_skills.backward()
                nn.utils.clip_grad_norm_(params_skills, 1.0)
                opt_skills.step()
                last_skills_metrics = metrics_skills
                if s_ep == 0:
                    initial_ce_loss = metrics_skills["ce_loss"]
                final_ce_loss = metrics_skills["ce_loss"]

        # 2) Train EpistemicNet, ContinueNet, ReadoutNet in mini-batches (6 epochs)
        for epoch in range(other_epochs):
            t_epoch_start = time.time()
            indices = np.random.permutation(N)[:max_epoch_samples]
            for b_start in range(0, max_epoch_samples, batch_size):
                b_idx = indices[b_start: b_start + batch_size]
                b_s = s_t[b_idx]
                b_a = a_t[b_idx]
                b_res = res_t[b_idx]
                b_dones = dones_t[b_idx]
                b_inv = inv_t[b_idx]

                # Epistemic
                opt_epistemic.zero_grad()
                loss_epi = self.epistemic_net.compute_loss(b_s, b_a, b_res)
                loss_epi.backward()
                nn.utils.clip_grad_norm_(self.epistemic_net.parameters(), 1.0)
                opt_epistemic.step()

                # Safety
                opt_safety.zero_grad()
                risk_logits = self.safety_net.risk_logits(b_s, b_a)
                loss_safety = focal_bce(risk_logits, b_dones, weight=25.0, gamma=2.0)
                loss_safety.backward()
                nn.utils.clip_grad_norm_(self.safety_net.parameters(), 1.0)
                opt_safety.step()

                # Readout
                if opt_readout is not None:
                    opt_readout.zero_grad()
                    loss_readout = self.readout_net.compute_loss(b_s, b_inv)
                    loss_readout.backward()
                    nn.utils.clip_grad_norm_(self.readout_net.parameters(), 1.0)
                    opt_readout.step()

            dt_ep = time.time() - t_epoch_start
            print(f"    -> Other Heads Epoch {epoch+1}/{other_epochs} completed in {dt_ep:.1f}s (HER CE: {final_ce_loss:.3f})")

        # Gate Checks (§5.11 / PLAN_V2)
        # 1. Reachability AUROC: hard (50/50 cross-ep + far within-ep) and easy (cross-ep)
        reach_auroc_hard = 0.5
        reach_auroc_easy = 0.5
        if val_eps:
            val_easy, val_hard = build_validation_reach_pairs(val_eps, device=self.device)
            with torch.no_grad():
                if val_easy["s_reach"].shape[0] > 0:
                    psi_easy = self.reach_net(val_easy["s_reach"], val_easy["g_reach"]).cpu().numpy()
                    reach_auroc_easy = _compute_auroc(psi_easy, val_easy["reach_labels"].cpu().numpy())
                if val_hard["s_reach"].shape[0] > 0:
                    psi_hard = self.reach_net(val_hard["s_reach"], val_hard["g_reach"]).cpu().numpy()
                    reach_auroc_hard = _compute_auroc(psi_hard, val_hard["reach_labels"].cpu().numpy())
        elif her_val is not None and her_val["s_reach"].shape[0] > 0:
            with torch.no_grad():
                psi_preds = self.reach_net(her_val["s_reach"], her_val["g_reach"]).cpu().numpy()
            reach_auroc_hard = _compute_auroc(psi_preds, her_val["reach_labels"].cpu().numpy())
            reach_auroc_easy = reach_auroc_hard

        # 2. Continue-head death recall (validation episodes only)
        death_recall = None
        safety_gate_passed: Union[bool, str] = "skipped_no_deaths"
        if val_eps:
            z_val = np.concatenate([ep["z"] for ep in val_eps], axis=0)
            h_val = np.concatenate([ep["h"] for ep in val_eps], axis=0)
            s_val = np.concatenate([z_val, h_val], axis=-1)
            a_val = np.concatenate([ep["actions"] for ep in val_eps], axis=0)
            dones_val = np.concatenate([ep["dones"] for ep in val_eps], axis=0)

            death_mask = dones_val.astype(bool)
            if np.sum(death_mask) > 0:
                with torch.no_grad():
                    s_val_t = torch.from_numpy(s_val).float().to(self.device)
                    a_val_t = torch.from_numpy(a_val).long().to(self.device)
                    risks = self.safety_net.risk(s_val_t, a_val_t).cpu().numpy()
                death_recall = float(np.mean(risks[death_mask] > self.safety_net.cfg.risk_threshold))
                safety_gate_passed = bool(death_recall >= 0.40)
            else:
                death_recall = None
                safety_gate_passed = "skipped_no_deaths"

        # 3. Policy CE decreasing
        ce_decreasing = bool(final_ce_loss <= initial_ce_loss + 0.1)

        # 4. Action histogram
        if her_train is not None and her_train["actions_pos"].shape[0] > 0:
            a_pos_cpu = her_train["actions_pos"].cpu()
            counts = torch.bincount(a_pos_cpu, minlength=NUM_ACTIONS).tolist()
            her_action_histogram = [int(c) for c in counts]
        else:
            her_action_histogram = [0] * NUM_ACTIONS

        gates = {
            "reach_auroc_hard": float(reach_auroc_hard),
            "reach_auroc_easy": float(reach_auroc_easy),
            "reach_auroc": float(reach_auroc_hard),
            "reach_gate_passed": bool(reach_auroc_hard >= 0.65 or not val_eps),
            "death_recall": death_recall,
            "safety_gate_passed": safety_gate_passed,
            "initial_ce_loss": float(initial_ce_loss),
            "final_ce_loss": float(final_ce_loss),
            "ce_decreasing": ce_decreasing,
            "her_action_histogram": her_action_histogram,
        }

        # Log training step
        log_entry = {
            "iter": iter_idx,
            "final_ce_loss": final_ce_loss,
            "reach_auroc_hard": float(reach_auroc_hard),
            "reach_auroc_easy": float(reach_auroc_easy),
            "reach_auroc": float(reach_auroc_hard),
            "death_recall": death_recall,
            "her_action_histogram": her_action_histogram,
            "her_weight_mean": float(last_skills_metrics.get("her_weight_mean", 1.0)),
            "her_weight_p95": float(last_skills_metrics.get("her_weight_p95", 1.0)),
            "her_unlock_pair_fraction": float(last_skills_metrics.get("her_unlock_pair_fraction", 0.0)),
            "critic_ce": float(critic_ce),
            "critic_value_mean": float(critic_val_mean),
            "critic_value_p90": float(critic_val_p90),
            "gates": gates,
        }
        if extra_log:
            log_entry.update(extra_log)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_entry
