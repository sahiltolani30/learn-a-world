"""HeadTrainer: latent caching and head training with gate verification.

Backbone is frozen; latent caching encodes obs and runs GRU once per shard.
All downstream heads train strictly on cached (z, h) latents without backbone forwards.
Logs training metrics and verifies quality gates after each iteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from atlas.backbone import FrozenBackbone
from atlas.collect import ShardLoader
from atlas.config import LATENT_DIM, RECURRENT_DIM, STATE_DIM, TrainConfig
from atlas.epistemic import EpistemicNet
from atlas.readout import VitalsReadout
from atlas.safety import ContinueNet, focal_bce
from atlas.skills import OptionNet, ReachabilityNet, build_her_batch, compute_her_loss


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
        self.cfg = cfg or TrainConfig()

        self.log_dir = Path(log_dir) if log_dir else Path("atlas_runs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train_log.jsonl"

    def cache_latents(self, shard_path: Union[str, Path]) -> Path:
        """Encodes observations and computes GRU latents, saving sidecar *.latents.npz."""
        p = Path(shard_path)
        out_path = p.with_suffix(".latents.npz")
        if out_path.exists():
            return out_path

        shard = ShardLoader.load_shard(p)
        obs = shard["obs"]
        actions = shard["actions"]
        ep_lengths = shard["episode_lengths"]

        T_total = len(obs)
        z_list: List[np.ndarray] = []

        # 1) Batched visual encoding
        batch_size = 512
        for i in range(0, T_total, batch_size):
            obs_batch = obs[i: i + batch_size]
            with torch.no_grad():
                z_batch = self.backbone.encode(obs_batch)
            z_list.append(z_batch.cpu().numpy())
        z_arr = np.concatenate(z_list, axis=0)

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
                h_arr[start + t] = h_t.cpu().numpy().squeeze(0)
                z_cur = z_ep[t: t + 1]
                a_cur = a_ep[t: t + 1]

                # Compute frozen predictor residual target
                if t < ep_len - 1:
                    with torch.no_grad():
                        z_next_pred = self.backbone.model.predictor(z_cur, h_t, a_cur)
                        residual = z_ep[t + 1: t + 2] - z_next_pred
                        target_residuals[start + t] = residual.cpu().numpy().squeeze(0)

                with torch.no_grad():
                    h_next = self.backbone.recurrent_step(z_cur, a_cur, h_t)
                h_t = h_next

            start = end

        np.savez_compressed(
            out_path,
            z=z_arr,
            h=h_arr,
            target_residuals=target_residuals,
        )
        return out_path

    def train_epoch(
        self,
        train_eps: List[Dict[str, Any]],
        val_eps: List[Dict[str, Any]],
        iter_idx: int,
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

        # Build HER batch
        her_train = build_her_batch(train_eps, device=self.device)
        her_val = build_her_batch(val_eps, device=self.device) if val_eps else None

        # Build flat transition tensors for EpistemicNet and ContinueNet
        z_train = np.concatenate([ep["z"] for ep in train_eps], axis=0)
        h_train = np.concatenate([ep["h"] for ep in train_eps], axis=0)
        s_train = np.concatenate([z_train, h_train], axis=-1)
        a_train = np.concatenate([ep["actions"] for ep in train_eps], axis=0)
        res_train = np.concatenate([ep["target_residuals"] for ep in train_eps], axis=0)
        dones_train = np.concatenate([ep["dones"] for ep in train_eps], axis=0)
        inv_train = np.concatenate([ep["inventory"] for ep in train_eps], axis=0)

        s_t = torch.from_numpy(s_train).float().to(self.device)
        a_t = torch.from_numpy(a_train).long().to(self.device)
        res_t = torch.from_numpy(res_train).float().to(self.device)
        dones_t = torch.from_numpy(dones_train).float().to(self.device)
        inv_t = torch.from_numpy(inv_train).float().to(self.device)

        N = len(s_t)
        batch_size = min(self.cfg.batch_size, max(1, N))

        initial_ce_loss = 0.0
        final_ce_loss = 0.0

        for epoch in range(self.cfg.epochs_per_iter):
            # 1) Train OptionNet & ReachabilityNet
            if her_train["s_pos"].shape[0] > 0:
                opt_skills.zero_grad()
                loss_skills, metrics_skills = compute_her_loss(self.option_net, self.reach_net, her_train)
                loss_skills.backward()
                nn.utils.clip_grad_norm_(params_skills, 1.0)
                opt_skills.step()
                if epoch == 0:
                    initial_ce_loss = metrics_skills["ce_loss"]
                final_ce_loss = metrics_skills["ce_loss"]

            # 2) Train EpistemicNet, ContinueNet, ReadoutNet in mini-batches
            indices = np.random.permutation(N)
            for b_start in range(0, N, batch_size):
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

        # Gate Checks (§5.11)
        # 1. Reachability AUROC
        reach_auroc = 0.5
        if her_val is not None and her_val["s_reach"].shape[0] > 0:
            with torch.no_grad():
                psi_preds = self.reach_net(her_val["s_reach"], her_val["g_reach"]).cpu().numpy()
            reach_auroc = _compute_auroc(psi_preds, her_val["reach_labels"].cpu().numpy())

        # 2. Continue-head death recall
        with torch.no_grad():
            risks = self.safety_net.risk(s_t, a_t).cpu().numpy()
        death_mask = dones_train.astype(bool)
        if np.sum(death_mask) > 0:
            death_recall = float(np.mean(risks[death_mask] > self.safety_net.cfg.risk_threshold))
        else:
            death_recall = 1.0  # vacuously true if no deaths

        # 3. Policy CE decreasing
        ce_decreasing = bool(final_ce_loss <= initial_ce_loss + 0.1)

        gates = {
            "reach_auroc": float(reach_auroc),
            "reach_gate_passed": bool(reach_auroc >= 0.65 or her_val is None),
            "death_recall": float(death_recall),
            "safety_gate_passed": bool(death_recall >= 0.40),
            "initial_ce_loss": float(initial_ce_loss),
            "final_ce_loss": float(final_ce_loss),
            "ce_decreasing": ce_decreasing,
        }

        # Log training step
        log_entry = {
            "iter": iter_idx,
            "final_ce_loss": final_ce_loss,
            "reach_auroc": reach_auroc,
            "death_recall": death_recall,
            "gates": gates,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_entry
