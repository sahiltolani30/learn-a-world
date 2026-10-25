#!/usr/bin/env python3
"""
scripts/diagnose_planner_autopsy.py
===================================
Diagnostic Autopsy of RecurrentMPCPlanner Scoring on jepa_crafter_v3_t2_final.pt.
Strict read-only diagnostic: modifies NO weights and alters NO mechanism.

Evaluates 120 pickaxe-in-hand states (identical protocol to Step 0 probe) to analyze:
1. Five Score Components per candidate:
   - cumulative reward
   - terminal value
   - uncertainty penalty
   - novelty
   - TTE gain
   plus total score, planned action, and candidate source (prior vs random/CEM).
2. Calibrated vs Raw Event Probabilities:
   - Runs each state twice: with EventHeadCalibrator loaded and without.
   - Logs 22 event probabilities for both modes.
3. Imagined Mask Dynamics:
   - For each candidate rollout: tracks how often mask flips 0 -> 1 per achievement.
   - Records highest probability reached by each of the 22 heads.
4. Comprehensive Aggregate Statistics:
   - Mean and spread (std) of all 5 score components across all candidates.
   - Share of total score driven by terminal value (mean ratio, net ratio, absolute share).
   - Mask flip rates per achievement.
   - Mean and max event probabilities: Raw vs Calibrated.

Output:
- results/diagnostics/planner_autopsy_t2_final.json (+ Drive mirror)
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.calibration import EventHeadCalibrator
from models.jepa_v3 import JEPAv3

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)


def resolve_checkpoint(ckpt_path: str, drive_dir: Optional[str] = None) -> Path:
    p = Path(ckpt_path)
    if p.exists():
        return p
    candidates = [
        REPO_ROOT / ckpt_path,
        REPO_ROOT / "checkpoints" / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / ckpt_path,
            dp / "checkpoints" / p.name,
            Path("/content/drive/MyDrive/learn-a-world") / ckpt_path,
            Path("/content/drive/MyDrive/learn-a-world/checkpoints") / p.name,
            Path("/content") / "checkpoints" / p.name,
            Path("/content") / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint: {ckpt_path}")


def resolve_calibrator(calibrator_path: str, drive_dir: Optional[str] = None) -> Path:
    p = Path(calibrator_path)
    if p.exists():
        return p
    candidates = [
        REPO_ROOT / calibrator_path,
        REPO_ROOT / "results" / "calibration" / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / calibrator_path,
            dp / "results" / "calibration" / p.name,
            Path("/content/drive/MyDrive/learn-a-world") / calibrator_path,
            Path("/content/drive/MyDrive/learn-a-world/results/calibration") / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"[ABORT] Cannot locate calibrator: {calibrator_path}")


def load_probe_states(
    drive_dir: Optional[str] = None,
    max_states: int = 120,
) -> List[Dict[str, Any]]:
    """Loads the 120 pickaxe-in-hand states reusing the Step 0 probe logic."""
    drive_p = Path(drive_dir) if drive_dir else REPO_ROOT
    shard_candidates = [
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
    ]
    cur_dirs = [
        drive_p / "data" / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]
    cur_dir = cur_dirs[0] if (cur_dirs[0].exists() and len(list(cur_dirs[0].glob("*.npz"))) > 0) else cur_dirs[1]
    all_shards = [s for s in shard_candidates if s.exists()]
    if cur_dir.exists():
        t2_shards = sorted(cur_dir.glob("shard_curiosity_v3_t02_*.npz"))
        if not t2_shards:
            t2_shards = sorted(cur_dir.glob("*.npz"))
        all_shards.extend(t2_shards)

    print(f"[Autopsy] Scanning {len(all_shards)} shards for pickaxe and stone trajectories...")
    targeted_episodes = []

    for sp in all_shards:
        try:
            data = np.load(str(sp), allow_pickle=True)
            obs = data["obs"]
            actions = data["actions"]
            events = data["events"]
            dones = data["dones"]
            masks_before = data["mask_before"] if "mask_before" in data else None

            n_trans = len(obs)
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            ep_start = 0
            for ep_end in ep_indices:
                ep_events = events[ep_start : ep_end + 1]
                ep_masks = masks_before[ep_start : ep_end + 1] if masks_before is not None else None

                has_stone = bool(np.any(ep_events[:, IDX_STONE] > 0))
                has_pickaxe = bool(
                    np.any(ep_events[:, IDX_WOOD_PICKAXE] > 0)
                    or (ep_masks is not None and np.any(ep_masks[:, IDX_WOOD_PICKAXE] > 0))
                )

                if has_stone or has_pickaxe:
                    targeted_episodes.append({
                        "shard": sp.name,
                        "has_stone": has_stone,
                        "obs": obs[ep_start : ep_end + 1],
                        "actions": actions[ep_start : ep_end + 1],
                        "events": ep_events,
                        "masks_before": ep_masks,
                    })
                ep_start = ep_end + 1
        except Exception:
            continue

    print(f"[Autopsy] Found {len(targeted_episodes)} relevant episodes ({sum(1 for e in targeted_episodes if e['has_stone'])} contain stone).")

    pickaxe_states = []
    for ep in targeted_episodes:
        ep_obs = ep["obs"]
        ep_ev = ep["events"]
        ep_masks = ep["masks_before"]
        has_stone = ep["has_stone"]
        stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]
        t_stone = int(stone_steps[0]) if len(stone_steps) > 0 else len(ep_obs)

        for t in range(len(ep_obs)):
            has_pickaxe = False
            if ep_masks is not None and ep_masks[t, IDX_WOOD_PICKAXE] > 0:
                has_pickaxe = True
            elif np.any(ep_ev[:t, IDX_WOOD_PICKAXE] > 0):
                has_pickaxe = True

            if has_pickaxe and t <= t_stone:
                pickaxe_states.append({
                    "episode": ep,
                    "t": t,
                    "is_stone_trajectory": has_stone,
                })

    if len(pickaxe_states) > max_states:
        stone_st = [s for s in pickaxe_states if s["is_stone_trajectory"]]
        other_st = [s for s in pickaxe_states if not s["is_stone_trajectory"]]
        target_half = max(1, max_states // 2)
        stride_s = max(1, len(stone_st) // target_half)
        stride_o = max(1, len(other_st) // target_half)
        sampled = stone_st[::stride_s] + other_st[::stride_o]
        return sampled[:max_states]
    return pickaxe_states


def run_state_autopsy(
    model: JEPAv3,
    state_item: Dict[str, Any],
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    elite_ratio: float = 0.15,
    gamma: float = 0.99,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.02,
    prior_mix: float = 0.5,
    prior_temperature: float = 1.0,
    action_persistence: float = 0.0,
    denormalize_q: bool = False,
    tte_weight: float = 0.0,
    standardize_scores: bool = True,
    num_actions: int = 17,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs the planner candidate generation and scoring for a single state TWICE:
    once with calibrator active, once without.
    Returns: (calibrated_results, raw_results)
    """
    ep = state_item["episode"]
    t = state_item["t"]
    obs_seq = ep["obs"][: t + 1]
    act_seq = ep["actions"][: t + 1]

    with torch.no_grad():
        t_obs_seq = torch.from_numpy(obs_seq).permute(0, 3, 1, 2).float() / 255.0
        t_obs_seq = t_obs_seq.to(device)
        t_act_seq = torch.from_numpy(act_seq).long().to(device)

        z_seq = model.encode(t_obs_seq, use_target=False)
        h_seq = model.recurrent_core.unroll_sequence(z_seq.unsqueeze(0), t_act_seq.unsqueeze(0)).squeeze(0)
        h_0 = h_seq[-1].unsqueeze(0)  # [1, 256]
        z_0 = z_seq[-1].unsqueeze(0)  # [1, 128]
        s_0 = torch.cat([z_0, h_0], dim=-1)  # [1, 384]

        mask_np = ep["masks_before"][t] if ep["masks_before"] is not None else np.zeros(22, dtype=np.float32)
        m_0_bin = torch.from_numpy(mask_np).float().to(device).unsqueeze(0)  # [1, 22]
        m_0_soft = m_0_bin.clone()

        C = num_candidates
        N_prior = int(round(C * prior_mix))
        N_cem = C - N_prior

        def evaluate_mode(use_calib: bool) -> Dict[str, Any]:
            # 1. Root-state event probabilities across all 17 discrete actions
            all_actions_t = torch.arange(num_actions, device=device)
            s_0_exp = s_0.repeat(num_actions, 1)
            m_0_exp = m_0_bin.repeat(num_actions, 1)
            root_event_probs = model.event_head.predict_proba(s_0_exp, all_actions_t, m_0_exp, use_calibration=use_calib)
            root_probs_np = root_event_probs.cpu().numpy()  # [17, 22]
            root_probs_mean_action = np.mean(root_probs_np, axis=0).tolist()
            root_probs_max_action = np.max(root_probs_np, axis=0).tolist()

            # Initialize CEM distribution
            action_dist = torch.ones(horizon, num_actions, device=device) / num_actions

            candidates_data: List[Dict[str, Any]] = []
            final_actions_seq = None
            final_scores = None

            for iteration in range(cem_iterations):
                # 2. Sample N_prior candidates via PolicyPrior
                prior_actions = torch.empty((N_prior, horizon), dtype=torch.long, device=device)
                z_roll = z_0.expand(N_prior, -1)
                h_roll = h_0.expand(N_prior, -1)
                s_roll = s_0.expand(N_prior, -1)
                m_bin_roll = m_0_bin.expand(N_prior, -1).clone()

                for h_step in range(horizon):
                    a_h = model.sample_prior_action(s_roll, m_bin_roll, temperature=prior_temperature)
                    prior_actions[:, h_step] = a_h
                    p_h = model.event_head.predict_proba(s_roll, a_h, m_bin_roll, use_calibration=use_calib)
                    m_bin_roll = torch.where(p_h > 0.5, torch.tensor(1.0, device=device), m_bin_roll)
                    z_roll, h_roll, s_roll = model.predict_next_step(z_roll, h_roll, a_h)

                # 3. Sample N_cem candidates from CEM categorical distribution
                cem_actions = torch.empty((N_cem, horizon), dtype=torch.long, device=device)
                for h_step in range(horizon):
                    probs_h = action_dist[h_step]
                    fresh_actions = torch.multinomial(probs_h, num_samples=N_cem, replacement=True)
                    if action_persistence > 0.0 and h_step > 0:
                        repeat_mask = torch.rand(N_cem, device=device) < action_persistence
                        prev_actions = cem_actions[:, h_step - 1]
                        cem_actions[:, h_step] = torch.where(repeat_mask, prev_actions, fresh_actions)
                    else:
                        cem_actions[:, h_step] = fresh_actions

                actions_seq = torch.cat([prior_actions, cem_actions], dim=0)  # [C, horizon]
                sources = ["prior"] * N_prior + ["random"] * N_cem

                # 4. Rollout and score all C candidates through recurrent dynamics
                z_t = z_0.expand(C, -1)
                h_t = h_0.expand(C, -1)
                s_t = s_0.expand(C, -1)
                m_bin_t = m_0_bin.expand(C, -1).clone()
                m_soft_t = m_0_soft.expand(C, -1).clone()
                m_soft_init = m_soft_t.clone()
                cumulative_reward = torch.zeros(C, device=device)

                max_event_probs = torch.zeros((C, 22), dtype=torch.float, device=device)

                for step_idx in range(horizon):
                    a_t = actions_seq[:, step_idx]
                    r_hat, decomp = model.predict_reward(
                        s_t,
                        a_t,
                        m_bin_t,
                        m_soft=m_soft_t,
                        return_decomposition=True,
                        use_calibration=use_calib,
                    )
                    p_t = decomp["event_probs"]  # [C, 22]

                    max_event_probs = torch.maximum(max_event_probs, p_t)
                    cumulative_reward = cumulative_reward + (gamma ** step_idx) * r_hat

                    # Simulated belief progression:
                    # Soft mask updated by running maximum of predicted probability (for reward)
                    m_soft_t = torch.maximum(m_soft_t, p_t)
                    # Binary mask updated with 0.5 threshold (for model input)
                    m_bin_t = torch.where(p_t > 0.5, torch.tensor(1.0, device=device), m_bin_t)
                    z_t, h_t, s_t = model.predict_next_step(z_t, h_t, a_t)

                m_soft_final = m_soft_t
                m_soft_increase = torch.clamp(m_soft_final - m_soft_init, min=0.0)

                # Terminal evaluation
                a_terminal = model.sample_prior_action(s_t, m_bin_t, temperature=prior_temperature)
                q_term = model.q_min(s_t, a_terminal, use_target=True)
                q_std_term = model.q_std(s_t, a_terminal, use_target=True)

                if denormalize_q and hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                    q_term = model.return_normalizer.denormalize(q_term)
                    norm = model.return_normalizer
                    scale = max(norm.high - norm.low, 1.0)
                    q_std_term = q_std_term / (norm.target_max * 0.8) * scale

                terminal_value = (gamma ** horizon) * q_term
                raw_uncertainty = q_std_term
                raw_novelty = torch.norm(z_t - z_0, dim=-1)

                raw_tte = torch.zeros_like(terminal_value)
                if hasattr(model, "time_to_event_head") and model.time_to_event_head is not None:
                    tau_0 = model.time_to_event_head(s_0)
                    tau_H = model.time_to_event_head(s_t)
                    unachieved_mask = (1.0 - m_0_soft).expand(C, -1)
                    prog = ((tau_H - tau_0) * unachieved_mask).clamp(min=0.0).sum(dim=-1)
                    raw_tte = prog

                if standardize_scores:
                    def _standardize(term: torch.Tensor) -> Tuple[torch.Tensor, float]:
                        std = term.std()
                        if std < 1e-8:
                            return torch.zeros_like(term), float(std.item())
                        return (term - term.mean()) / std, float(std.item())

                    z_reward, std_reward = _standardize(cumulative_reward)
                    z_term, std_term = _standardize(terminal_value)
                    z_unc, std_unc = _standardize(raw_uncertainty)
                    z_nov, std_nov = _standardize(raw_novelty)
                    z_tte, std_tte = _standardize(raw_tte)

                    scores = z_reward + z_term - z_unc + z_nov + z_tte
                else:
                    uncertainty_pen = uncertainty_weight * raw_uncertainty
                    novelty_val = novelty_weight * raw_novelty
                    tte_gain_val = tte_weight * raw_tte
                    scores = cumulative_reward + terminal_value - uncertainty_pen + novelty_val + tte_gain_val

                # Refit CEM distribution for next iteration
                num_elites = max(1, int(round(C * elite_ratio)))
                elite_indices = torch.topk(scores, k=num_elites).indices
                elite_seqs = actions_seq[elite_indices]

                for h_step in range(horizon):
                    counts = torch.bincount(elite_seqs[:, h_step], minlength=num_actions).float()
                    action_dist[h_step] = (counts + 0.1) / (num_elites + 0.1 * num_actions)

                # Keep candidates data from the final decision iteration
                if iteration == (cem_iterations - 1):
                    final_actions_seq = actions_seq
                    final_scores = scores

                    cum_np = cumulative_reward.cpu().numpy()
                    term_np = terminal_value.cpu().numpy()
                    raw_unc_np = raw_uncertainty.cpu().numpy()
                    raw_nov_np = raw_novelty.cpu().numpy()
                    raw_tte_np = raw_tte.cpu().numpy()
                    tot_np = scores.cpu().numpy()
                    final_soft_np = m_soft_final.cpu().numpy()
                    inc_np = m_soft_increase.cpu().numpy()
                    max_p_np = max_event_probs.cpu().numpy()

                    if standardize_scores:
                        z_cum_np = z_reward.cpu().numpy()
                        z_term_np = z_term.cpu().numpy()
                        z_unc_np = z_unc.cpu().numpy()
                        z_nov_np = z_nov.cpu().numpy()
                        z_tte_np = z_tte.cpu().numpy()
                        cand_spread_reward = float(z_reward.std().item()) if std_reward >= 1e-8 else 0.0
                        cand_spread_term = float(z_term.std().item()) if std_term >= 1e-8 else 0.0
                        cand_spread_unc = float(z_unc.std().item()) if std_unc >= 1e-8 else 0.0
                        cand_spread_nov = float(z_nov.std().item()) if std_nov >= 1e-8 else 0.0
                        cand_spread_tte = float(z_tte.std().item()) if std_tte >= 1e-8 else 0.0
                    else:
                        unc_np = uncertainty_pen.cpu().numpy()
                        nov_np = novelty_val.cpu().numpy()
                        tte_np = tte_gain_val.cpu().numpy()

                    for c_idx in range(C):
                        cand_entry = {
                            "candidate_idx": c_idx,
                            "source": sources[c_idx],
                            "action": int(actions_seq[c_idx, 0].item()),
                            "raw_cumulative_reward": float(cum_np[c_idx]),
                            "raw_terminal_value": float(term_np[c_idx]),
                            "raw_uncertainty": float(raw_unc_np[c_idx]),
                            "raw_novelty": float(raw_nov_np[c_idx]),
                            "raw_tte_gain": float(raw_tte_np[c_idx]),
                            "cumulative_reward": float(cum_np[c_idx]),
                            "terminal_value": float(term_np[c_idx]),
                            "uncertainty_penalty": float(raw_unc_np[c_idx] if standardize_scores else unc_np[c_idx]),
                            "novelty": float(raw_nov_np[c_idx] if standardize_scores else nov_np[c_idx]),
                            "tte_gain": float(raw_tte_np[c_idx] if standardize_scores else tte_np[c_idx]),
                            "total_score": float(tot_np[c_idx]),
                            "highest_event_probs": [float(v) for v in max_p_np[c_idx]],
                            "final_soft_mask": [float(v) for v in final_soft_np[c_idx]],
                            "soft_mask_increase": [float(v) for v in inc_np[c_idx]],
                        }
                        if standardize_scores:
                            cand_entry.update({
                                "z_cumulative_reward": float(z_cum_np[c_idx]),
                                "z_terminal_value": float(z_term_np[c_idx]),
                                "z_uncertainty": float(z_unc_np[c_idx]),
                                "z_novelty": float(z_nov_np[c_idx]),
                                "z_tte_gain": float(z_tte_np[c_idx]),
                                "spread_cumulative_reward": cand_spread_reward,
                                "spread_terminal_value": cand_spread_term,
                                "spread_uncertainty": cand_spread_unc,
                                "spread_novelty": cand_spread_nov,
                                "spread_tte_gain": cand_spread_tte,
                            })
                        candidates_data.append(cand_entry)

            best_idx = int(torch.argmax(final_scores).item())
            best_action = int(final_actions_seq[best_idx, 0].item())

            return {
                "best_action": best_action,
                "best_score": float(final_scores[best_idx].item()),
                "root_event_probs_mean_action": root_probs_mean_action,
                "root_event_probs_max_action": root_probs_max_action,
                "candidates": candidates_data,
            }

        # 1. Run with calibrator active
        calibrated_res = evaluate_mode(use_calib=True)
        # 2. Run without calibrator
        raw_res = evaluate_mode(use_calib=False)

        return calibrated_res, raw_res


def compute_autopsy_aggregates(
    state_results: List[Dict[str, Any]],
    standardize_scores: bool = True,
) -> Dict[str, Any]:
    """Computes full cross-candidate and cross-state aggregate statistics."""
    all_raw_cum_rewards = []
    all_raw_term_values = []
    all_raw_unc_penalties = []
    all_raw_novelties = []
    all_raw_tte_gains = []

    all_z_cum_rewards = []
    all_z_term_values = []
    all_z_uncs = []
    all_z_novs = []
    all_z_ttes = []

    all_spread_cum_rewards = []
    all_spread_term_values = []
    all_spread_uncs = []
    all_spread_novs = []
    all_spread_ttes = []

    all_total_scores = []
    terminal_shares = []
    abs_terminal_shares = []

    # Soft mask progress per achievement (index 0..21)
    calib_final_masks = [[] for _ in range(22)]
    calib_mask_increases = [[] for _ in range(22)]
    raw_final_masks = [[] for _ in range(22)]
    raw_mask_increases = [[] for _ in range(22)]

    # Rollout max event probs per achievement
    calib_max_probs = [[] for _ in range(22)]
    raw_max_probs = [[] for _ in range(22)]

    for st in state_results:
        cal_cands = st["calibrated"]["candidates"]
        raw_cands = st["raw"]["candidates"]

        for c_idx, c in enumerate(cal_cands):
            all_raw_cum_rewards.append(c.get("raw_cumulative_reward", c["cumulative_reward"]))
            all_raw_term_values.append(c.get("raw_terminal_value", c["terminal_value"]))
            all_raw_unc_penalties.append(c.get("raw_uncertainty", c["uncertainty_penalty"]))
            all_raw_novelties.append(c.get("raw_novelty", c["novelty"]))
            all_raw_tte_gains.append(c.get("raw_tte_gain", c["tte_gain"]))
            all_total_scores.append(c["total_score"])

            if standardize_scores and "z_cumulative_reward" in c:
                all_z_cum_rewards.append(c["z_cumulative_reward"])
                all_z_term_values.append(c["z_terminal_value"])
                all_z_uncs.append(c["z_uncertainty"])
                all_z_novs.append(c["z_novelty"])
                all_z_ttes.append(c["z_tte_gain"])

                all_spread_cum_rewards.append(c["spread_cumulative_reward"])
                all_spread_term_values.append(c["spread_terminal_value"])
                all_spread_uncs.append(c["spread_uncertainty"])
                all_spread_novs.append(c["spread_novelty"])
                all_spread_ttes.append(c["spread_tte_gain"])

                tot = c["total_score"]
                if abs(tot) > 1e-6:
                    terminal_shares.append(c["z_terminal_value"] / tot)

                abs_sum = (
                    abs(c["z_cumulative_reward"])
                    + abs(c["z_terminal_value"])
                    + abs(c["z_uncertainty"])
                    + abs(c["z_novelty"])
                    + abs(c["z_tte_gain"])
                )
                if abs_sum > 1e-6:
                    abs_terminal_shares.append(abs(c["z_terminal_value"]) / abs_sum)
            else:
                tot = c["total_score"]
                if abs(tot) > 1e-6:
                    terminal_shares.append(c["terminal_value"] / tot)

                abs_sum = (
                    abs(c["cumulative_reward"])
                    + abs(c["terminal_value"])
                    + abs(c["uncertainty_penalty"])
                    + abs(c["novelty"])
                    + abs(c["tte_gain"])
                )
                if abs_sum > 1e-6:
                    abs_terminal_shares.append(abs(c["terminal_value"]) / abs_sum)

            for ach_idx in range(22):
                calib_final_masks[ach_idx].append(c["final_soft_mask"][ach_idx])
                calib_mask_increases[ach_idx].append(c["soft_mask_increase"][ach_idx])
                calib_max_probs[ach_idx].append(c["highest_event_probs"][ach_idx])

        for c_idx, c in enumerate(raw_cands):
            for ach_idx in range(22):
                raw_final_masks[ach_idx].append(c["final_soft_mask"][ach_idx])
                raw_mask_increases[ach_idx].append(c["soft_mask_increase"][ach_idx])
                raw_max_probs[ach_idx].append(c["highest_event_probs"][ach_idx])

    def stats_dict(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        if len(a) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
        }

    # Soft mask progress summary (replaces flip-rate metric)
    soft_mask_progress_summary = {}
    for ach_idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
        c_fm = np.array(calib_final_masks[ach_idx])
        c_inc = np.array(calib_mask_increases[ach_idx])
        r_fm = np.array(raw_final_masks[ach_idx])
        r_inc = np.array(raw_mask_increases[ach_idx])
        soft_mask_progress_summary[ach_name] = {
            "achievement_index": ach_idx,
            "calibrated_mean_final_soft_mask": float(np.mean(c_fm)),
            "calibrated_mean_mask_increase": float(np.mean(c_inc)),
            "raw_mean_final_soft_mask": float(np.mean(r_fm)),
            "raw_mean_mask_increase": float(np.mean(r_inc)),
        }

    # Event probabilities Raw vs Calibrated
    event_prob_comparison = {}
    for ach_idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
        c_p = np.array(calib_max_probs[ach_idx])
        r_p = np.array(raw_max_probs[ach_idx])
        event_prob_comparison[ach_name] = {
            "achievement_index": ach_idx,
            "raw_mean_max_prob": float(np.mean(r_p)),
            "raw_max_prob": float(np.max(r_p)),
            "calibrated_mean_max_prob": float(np.mean(c_p)),
            "calibrated_max_prob": float(np.max(c_p)),
            "delta_mean_prob": float(np.mean(c_p) - np.mean(r_p)),
            "delta_max_prob": float(np.max(c_p) - np.max(r_p)),
        }

    mean_tot = float(np.mean(all_total_scores)) if all_total_scores else 0.0
    mean_term = float(np.mean(all_z_term_values if (standardize_scores and all_z_term_values) else all_raw_term_values)) if all_raw_term_values else 0.0
    net_term_share = (mean_term / mean_tot) if abs(mean_tot) > 1e-6 else 0.0

    score_components = {
        "raw_cumulative_reward": stats_dict(all_raw_cum_rewards),
        "raw_terminal_value": stats_dict(all_raw_term_values),
        "raw_uncertainty": stats_dict(all_raw_unc_penalties),
        "raw_novelty": stats_dict(all_raw_novelties),
        "raw_tte_gain": stats_dict(all_raw_tte_gains),
        "cumulative_reward": stats_dict(all_raw_cum_rewards),
        "terminal_value": stats_dict(all_raw_term_values),
        "uncertainty_penalty": stats_dict(all_raw_unc_penalties),
        "novelty": stats_dict(all_raw_novelties),
        "tte_gain": stats_dict(all_raw_tte_gains),
        "total_score": stats_dict(all_total_scores),
    }
    if standardize_scores and all_z_cum_rewards:
        score_components.update({
            "z_cumulative_reward": stats_dict(all_z_cum_rewards),
            "z_terminal_value": stats_dict(all_z_term_values),
            "z_uncertainty": stats_dict(all_z_uncs),
            "z_novelty": stats_dict(all_z_novs),
            "z_tte_gain": stats_dict(all_z_ttes),
        })

    ret = {
        "total_states_analyzed": len(state_results),
        "total_candidates_analyzed": len(all_total_scores),
        "score_components": score_components,
        "terminal_value_share": {
            "mean_ratio": float(np.mean(terminal_shares)) if terminal_shares else 0.0,
            "net_share": net_term_share,
            "absolute_component_share": float(np.mean(abs_terminal_shares)) if abs_terminal_shares else 0.0,
            "description": "Terminal value contribution to overall planner candidate evaluation",
        },
        "soft_mask_progress": soft_mask_progress_summary,
        "event_probabilities_raw_vs_calibrated": event_prob_comparison,
    }
    if standardize_scores and all_spread_cum_rewards:
        ret["standardized_term_spreads"] = {
            "cumulative_reward": stats_dict(all_spread_cum_rewards),
            "terminal_value": stats_dict(all_spread_term_values),
            "uncertainty": stats_dict(all_spread_uncs),
            "novelty": stats_dict(all_spread_novs),
            "tte_gain": stats_dict(all_spread_ttes),
            "description": "Spread (std) of each standardized score component across candidates at each step (target ~1.0 for active terms)",
        }
    return ret


def run_planner_autopsy(
    checkpoint: str = "checkpoints/jepa_crafter_v3_t2_final.pt",
    calibrator_path: str = "results/calibration/event_head_calibrator.json",
    output_json: str = "results/diagnostics/planner_autopsy_t2_final.json",
    max_states: int = 120,
    standardize_scores: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
) -> Dict[str, Any]:
    dev = torch.device(device)
    ckpt_file = resolve_checkpoint(checkpoint, drive_dir)
    cal_file = resolve_calibrator(calibrator_path, drive_dir)

    print("=" * 80)
    print("      RECURRENT MPC PLANNER AUTOPSY: jepa_crafter_v3_t2_final.pt")
    print(f"      Checkpoint  : {ckpt_file}")
    print(f"      Calibrator  : {cal_file}")
    print(f"      Device      : {dev}")
    print(f"      Max States  : {max_states}")
    print(f"      Standardize : {standardize_scores}")
    print(f"      Output Path : {output_json}")
    print("=" * 80, flush=True)

    # 1. Load Model and Calibrator
    model = JEPAv3.from_checkpoint(str(ckpt_file), device=device)
    model.eval()

    calibrator = EventHeadCalibrator.load(str(cal_file))
    model.set_calibrator(calibrator)

    # 2. Load 120 Probe States
    sampled_states = load_probe_states(drive_dir, max_states=max_states)
    print(f"[Autopsy] Successfully prepared {len(sampled_states)} probe states.\n", flush=True)

    state_results = []
    t_start = time.time()

    for idx, item in enumerate(sampled_states):
        ep = item["episode"]
        t = item["t"]
        recorded_act = int(ep["actions"][t])
        m_0_np = ep["masks_before"][t].tolist() if ep["masks_before"] is not None else [0] * 22

        t0_st = time.time()
        cal_res, raw_res = run_state_autopsy(
            model=model,
            state_item=item,
            horizon=10,
            num_candidates=64,
            cem_iterations=2,
            elite_ratio=0.15,
            gamma=0.99,
            uncertainty_weight=0.5,
            novelty_weight=0.02,
            prior_mix=0.5,
            prior_temperature=1.0,
            action_persistence=0.0,
            denormalize_q=False,
            tte_weight=0.0,
            standardize_scores=standardize_scores,
            num_actions=17,
            device=dev,
        )
        elapsed = time.time() - t0_st

        state_entry = {
            "state_idx": idx + 1,
            "step": t,
            "recorded_action": recorded_act,
            "is_stone_trajectory": item["is_stone_trajectory"],
            "initial_mask": m_0_np,
            "calibrated": cal_res,
            "raw": raw_res,
            "eval_time_sec": round(elapsed, 3),
        }
        state_results.append(state_entry)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(sampled_states):
            agreed_cal = bool(cal_res["best_action"] == recorded_act)
            agreed_raw = bool(raw_res["best_action"] == recorded_act)
            print(
                f"  [{idx + 1:3d}/{len(sampled_states)}] step={t:<3d} | "
                f"rec={recorded_act} plan_cal={cal_res['best_action']} ({'Y' if agreed_cal else 'N'}) "
                f"plan_raw={raw_res['best_action']} ({'Y' if agreed_raw else 'N'}) | "
                f"score_cal={cal_res['best_score']:+.4f} score_raw={raw_res['best_score']:+.4f} | "
                f"({elapsed:.2f}s)",
                flush=True,
            )

    total_time = time.time() - t_start
    print(f"\n[Autopsy] Completed 120 states in {total_time:.1f}s ({total_time / len(sampled_states):.2f}s/state).")
    print("[Autopsy] Computing aggregate cross-candidate and cross-state statistics...", flush=True)

    aggregates = compute_autopsy_aggregates(state_results, standardize_scores=standardize_scores)

    final_document = {
        "checkpoint": str(ckpt_file),
        "calibrator_path": str(cal_file),
        "num_states": len(state_results),
        "default_screening_config": {
            "horizon": 10,
            "num_candidates": 64,
            "cem_iterations": 2,
            "elite_ratio": 0.15,
            "gamma": 0.99,
            "uncertainty_weight": 0.5,
            "novelty_weight": 0.02,
            "prior_mix": 0.5,
            "prior_temperature": 1.0,
            "action_persistence": 0.0,
            "denormalize_q": False,
            "tte_weight": 0.0,
            "standardize_scores": standardize_scores,
            "num_actions": 17,
        },
        "total_runtime_seconds": round(total_time, 2),
        "aggregate": aggregates,
        "states": state_results,
    }

    # Save Results
    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(final_document, f, indent=2)
    print(f"\n--> Successfully saved autopsy to: {out_p}")

    # Mirror to Drive if available
    if drive_dir:
        dp = Path(drive_dir)
        if dp.exists():
            drive_out = dp / "results" / "diagnostics" / out_p.name
            drive_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_p), str(drive_out))
            print(f"[Drive] Mirrored autopsy to: {drive_out}")

    # Print Official Terminal Summary Report
    print("\n" + "=" * 80)
    print("                    PLANNER AUTOPSY SUMMARY REPORT")
    print(f"               (Standardize Scores: {standardize_scores})")
    print("=" * 80)
    sc = aggregates["score_components"]
    if standardize_scores and "raw_cumulative_reward" in sc:
        print("Raw Score Parts Across All Candidates (Mean +/- Std):")
        print(f"  1. Cumulative Reward   : {sc['raw_cumulative_reward']['mean']:+9.4f} +/- {sc['raw_cumulative_reward']['std']:.4f} [min: {sc['raw_cumulative_reward']['min']:+.4f}, max: {sc['raw_cumulative_reward']['max']:+.4f}]")
        print(f"  2. Terminal Value      : {sc['raw_terminal_value']['mean']:+9.4f} +/- {sc['raw_terminal_value']['std']:.4f} [min: {sc['raw_terminal_value']['min']:+.4f}, max: {sc['raw_terminal_value']['max']:+.4f}]")
        print(f"  3. Uncertainty (Q std) : {sc['raw_uncertainty']['mean']:+9.4f} +/- {sc['raw_uncertainty']['std']:.4f} [min: {sc['raw_uncertainty']['min']:+.4f}, max: {sc['raw_uncertainty']['max']:+.4f}]")
        print(f"  4. Novelty (latent norm: {sc['raw_novelty']['mean']:+9.4f} +/- {sc['raw_novelty']['std']:.4f} [min: {sc['raw_novelty']['min']:+.4f}, max: {sc['raw_novelty']['max']:+.4f}]")
        print(f"  5. TTE Gain (proximity): {sc['raw_tte_gain']['mean']:+9.4f} +/- {sc['raw_tte_gain']['std']:.4f} [min: {sc['raw_tte_gain']['min']:+.4f}, max: {sc['raw_tte_gain']['max']:+.4f}]")

        print("\nStandardized Score Parts (z-score) Across All Candidates (Mean +/- Std):")
        print(f"  1. z(Cumulative Reward): {sc['z_cumulative_reward']['mean']:+9.4f} +/- {sc['z_cumulative_reward']['std']:.4f} [min: {sc['z_cumulative_reward']['min']:+.4f}, max: {sc['z_cumulative_reward']['max']:+.4f}]")
        print(f"  2. z(Terminal Value)   : {sc['z_terminal_value']['mean']:+9.4f} +/- {sc['z_terminal_value']['std']:.4f} [min: {sc['z_terminal_value']['min']:+.4f}, max: {sc['z_terminal_value']['max']:+.4f}]")
        print(f"  3. z(Uncertainty)      : {sc['z_uncertainty']['mean']:+9.4f} +/- {sc['z_uncertainty']['std']:.4f} [min: {sc['z_uncertainty']['min']:+.4f}, max: {sc['z_uncertainty']['max']:+.4f}]")
        print(f"  4. z(Novelty)          : {sc['z_novelty']['mean']:+9.4f} +/- {sc['z_novelty']['std']:.4f} [min: {sc['z_novelty']['min']:+.4f}, max: {sc['z_novelty']['max']:+.4f}]")
        print(f"  5. z(TTE Gain)         : {sc['z_tte_gain']['mean']:+9.4f} +/- {sc['z_tte_gain']['std']:.4f} [min: {sc['z_tte_gain']['min']:+.4f}, max: {sc['z_tte_gain']['max']:+.4f}]")
        print(f"  -> Total Score         : {sc['total_score']['mean']:+9.4f} +/- {sc['total_score']['std']:.4f}")

        if "standardized_term_spreads" in aggregates:
            sp = aggregates["standardized_term_spreads"]
            print("\nStandardized Term Spreads Across Candidates (Target ~1.0 for active terms):")
            print(f"  1. Cumulative Reward Spread : {sp['cumulative_reward']['mean']:.4f} +/- {sp['cumulative_reward']['std']:.4f}")
            print(f"  2. Terminal Value Spread    : {sp['terminal_value']['mean']:.4f} +/- {sp['terminal_value']['std']:.4f}")
            print(f"  3. Uncertainty Spread       : {sp['uncertainty']['mean']:.4f} +/- {sp['uncertainty']['std']:.4f}")
            print(f"  4. Novelty Spread           : {sp['novelty']['mean']:.4f} +/- {sp['novelty']['std']:.4f}")
            print(f"  5. TTE Gain Spread          : {sp['tte_gain']['mean']:.4f} +/- {sp['tte_gain']['std']:.4f}")
    else:
        print("Five Score Parts Across All Candidates (Mean +/- Std):")
        print(f"  1. Cumulative Reward   : {sc['cumulative_reward']['mean']:+9.4f} +/- {sc['cumulative_reward']['std']:.4f} [min: {sc['cumulative_reward']['min']:+.4f}, max: {sc['cumulative_reward']['max']:+.4f}]")
        print(f"  2. Terminal Value      : {sc['terminal_value']['mean']:+9.4f} +/- {sc['terminal_value']['std']:.4f} [min: {sc['terminal_value']['min']:+.4f}, max: {sc['terminal_value']['max']:+.4f}]")
        print(f"  3. Uncertainty Penalty : {sc['uncertainty_penalty']['mean']:+9.4f} +/- {sc['uncertainty_penalty']['std']:.4f} [min: {sc['uncertainty_penalty']['min']:+.4f}, max: {sc['uncertainty_penalty']['max']:+.4f}]")
        print(f"  4. Novelty             : {sc['novelty']['mean']:+9.4f} +/- {sc['novelty']['std']:.4f} [min: {sc['novelty']['min']:+.4f}, max: {sc['novelty']['max']:+.4f}]")
        print(f"  5. TTE Gain            : {sc['tte_gain']['mean']:+9.4f} +/- {sc['tte_gain']['std']:.4f} [min: {sc['tte_gain']['min']:+.4f}, max: {sc['tte_gain']['max']:+.4f}]")
        print(f"  -> Total Score         : {sc['total_score']['mean']:+9.4f} +/- {sc['total_score']['std']:.4f}")

    tv_share = aggregates["terminal_value_share"]
    print("\nTerminal Value Contribution:")
    print(f"  Absolute Component Share : {tv_share['absolute_component_share'] * 100:6.2f}% (Expected ~20% with standardization, down from ~60%)")
    print(f"  Net Contribution Share   : {tv_share['net_share'] * 100:6.2f}%")
    print(f"  Mean Ratio (Term / Tot)  : {tv_share['mean_ratio'] * 100:6.2f}%")

    print("\nTop 8 Achievements by Calibrated Mean Final Soft Mask (Imagined Rollouts):")
    sorted_progress = sorted(
        aggregates["soft_mask_progress"].items(),
        key=lambda kv: kv[1]["calibrated_mean_final_soft_mask"],
        reverse=True,
    )
    for name, data in sorted_progress[:8]:
        print(f"  {name:<22}: Calibrated Final {data['calibrated_mean_final_soft_mask']:.4f} (Inc: {data['calibrated_mean_mask_increase']:.4f}) | Raw Final {data['raw_mean_final_soft_mask']:.4f} (Inc: {data['raw_mean_mask_increase']:.4f})")

    print("\nKey Achievements Soft-Mask Progress:")
    for k in ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_wood_sword", "collect_drink"]:
        p_data = aggregates["soft_mask_progress"].get(k, {})
        print(
            f"  {k:<20} | Calibrated Final: {p_data.get('calibrated_mean_final_soft_mask', 0.0):.4f}, Inc: {p_data.get('calibrated_mean_mask_increase', 0.0):.4f} | "
            f"Raw Final: {p_data.get('raw_mean_final_soft_mask', 0.0):.4f}, Inc: {p_data.get('raw_mean_mask_increase', 0.0):.4f}"
        )

    print("\nEvent Probabilities: Raw vs Calibrated (Selected Key Achievements):")
    key_achs = ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_wood_sword", "collect_drink"]
    for k in key_achs:
        p_data = aggregates["event_probabilities_raw_vs_calibrated"].get(k, {})
        print(
            f"  {k:<20} | Raw Mean: {p_data.get('raw_mean_max_prob', 0.0):.4f}, Max: {p_data.get('raw_max_prob', 0.0):.4f} | "
            f"Calib Mean: {p_data.get('calibrated_mean_max_prob', 0.0):.4f}, Max: {p_data.get('calibrated_max_prob', 0.0):.4f}"
        )
    print("=" * 80 + "\n")

    return final_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic Autopsy of RecurrentMPCPlanner Scoring.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_t2_final.pt",
        help="Path to jepa_crafter_v3_t2_final.pt",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        default="results/calibration/event_head_calibrator.json",
        help="Path to event_head_calibrator.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/diagnostics/planner_autopsy_t2_final.json",
        help="Output JSON path",
    )
    parser.add_argument("--max_states", type=int, default=120, help="Number of probe states to autopsy (default 120)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument(
        "--standardize-scores",
        "--standardize_scores",
        dest="standardize_scores",
        action="store_true",
        default=True,
        help="Live per-step standardization of planner score terms (default: on).",
    )
    parser.add_argument(
        "--no-standardize-scores",
        "--no_standardize_scores",
        dest="standardize_scores",
        action="store_false",
        help="Disable score standardization and use legacy weighted sums.",
    )
    args = parser.parse_args()

    run_planner_autopsy(
        checkpoint=args.checkpoint,
        calibrator_path=args.calibrator,
        output_json=args.output_json,
        max_states=args.max_states,
        standardize_scores=args.standardize_scores,
        device=args.device,
        drive_dir=args.drive_dir,
    )


if __name__ == "__main__":
    main()
