#!/usr/bin/env python3
"""
scripts/diagnose_step0_battery.py — Phase 3 Step 0 Diagnostics Battery
========================================================================

Executes the 3 Step 0 diagnostic tests on checkpoints/jepa_crafter_v3_t2_final.pt:
1. EYES TEST (D3 Re-run): Per-head PR-AUC and recall@0.5 for all 22 heads on held-out slice.
   Output: results/diagnostics/t2_final_head_audit.json (+ .png)
2. PLANNER TEST: Pickaxe-in-hand states probe (action agreement & value gap stone vs wander).
   Output: results/diagnostics/t2_planner_stone_probe.json
3. JUDGE TEST: D1 Q-calibration restricted to stone-adjacent states (within 16 steps of stone + pickaxe).
   Output: results/diagnostics/t2_final_q_calibration_stone_subset.json

No training is performed. Diagnostic tests only.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)


def ensure_drive_mounted(drive_dir: Path) -> bool:
    """Mounts Google Drive if running in Colab and path is absent."""
    if drive_dir.exists():
        return True
    try:
        from google.colab import drive  # type: ignore
        print("[Drive] Google Colab detected. Mounting Google Drive...")
        drive.mount("/content/drive")
        return drive_dir.exists()
    except Exception as e:
        print(f"[Drive] Note: Could not auto-mount Drive ({e}).")
        return False


def find_checkpoint(name: str, candidate_dirs: List[Path]) -> Path:
    """Locates checkpoint across candidate paths."""
    for d in candidate_dirs:
        if d.exists():
            target = d / name
            if target.exists():
                return target
            matches = list(d.rglob(f"*{name}*"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint '{name}' in candidates: {candidate_dirs}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1: EYES TEST (D3 Re-run on Held-out Slice)
# ──────────────────────────────────────────────────────────────────────────────

def run_eyes_test(
    model: JEPAv3,
    sequences_dir: Path,
    output_json: Path,
    output_png: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("1. EYES TEST: EVENT HEAD AUDIT (D3 PROTOCOL ON HELD-OUT SLICE)")
    print("=" * 80)

    # Phase 0 D3 held-out slice uses the standard evaluation shards (8,000 transitions)
    shard_candidates = [
        sequences_dir / "shard_structured_random_0000.npz",
        sequences_dir / "shard_teacher_hybrid_0000.npz",
    ]
    existing_shards = [p for p in shard_candidates if p.exists()]
    if not existing_shards:
        # Fallback to any shards in directory
        existing_shards = sorted(sequences_dir.glob("*.npz"))[:2]

    if not existing_shards:
        raise FileNotFoundError(f"No sequence shards found in {sequences_dir}")

    print(f"[Eyes-Test] Loading held-out slice from: {[s.name for s in existing_shards]}")
    all_obs, all_act, all_ev, all_mask, all_rew, all_lengths = [], [], [], [], [], []

    for p in existing_shards:
        data = np.load(str(p), allow_pickle=True)
        all_obs.append(data["obs"])
        all_act.append(data["actions"])
        all_ev.append(data["events"])
        all_mask.append(data["mask_before"])
        all_rew.append(data["rewards"])
        lengths = data.get("episode_lengths", None)
        if lengths is not None and len(lengths) > 0:
            all_lengths.extend(lengths.tolist())
        else:
            dones = data["dones"]
            start = 0
            for i, d in enumerate(dones):
                if d or i == len(dones) - 1:
                    all_lengths.append(i + 1 - start)
                    start = i + 1

    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_act, axis=0)
    events = np.concatenate(all_ev, axis=0)
    mask_before = np.concatenate(all_mask, axis=0)
    rewards = np.concatenate(all_rew, axis=0)
    lengths = all_lengths
    N = len(obs)

    print(f"[Eyes-Test] Evaluating {N} transitions across {len(lengths)} episodes on {device}...")
    dev = torch.device(device)
    model.eval()

    all_pred_probs = []
    all_pred_health = []
    idx = 0

    with torch.no_grad():
        for ep_len in lengths:
            ep_slice = slice(idx, idx + ep_len)
            b_obs = obs[ep_slice]
            b_act = actions[ep_slice]
            b_mask = mask_before[ep_slice]

            t_obs = torch.from_numpy(b_obs).permute(0, 3, 1, 2).float() / 255.0
            t_obs = t_obs.to(dev)
            t_act = torch.from_numpy(b_act).long().to(dev)
            t_mask = torch.from_numpy(b_mask).float().to(dev)

            z = model.encode(t_obs, use_target=True)
            h = model.recurrent_core.unroll_sequence(z.unsqueeze(0), t_act.unsqueeze(0)).squeeze(0)
            s = torch.cat([z, h], dim=-1)

            p_event = model.event_head.predict_proba(s, t_act, t_mask)
            pred_h = model.health_head(s, t_act).squeeze(-1)

            all_pred_probs.append(p_event.cpu().numpy())
            all_pred_health.append(pred_h.cpu().numpy())
            idx += ep_len

    pred_probs = np.concatenate(all_pred_probs, axis=0)
    pred_health = np.concatenate(all_pred_health, axis=0)

    event_counts = events.sum(axis=-1)
    target_health = np.clip(rewards - event_counts, -1.0, 1.0)
    health_mae = float(np.mean(np.abs(pred_health - target_health)))

    per_achievement = {}
    active_pr_aucs = []

    print("\n" + "-" * 85)
    print(f"{'Achievement Head':<24} | {'PR-AUC':<10} | {'Recall@0.5':<12} | {'ROC-AUC':<10} | {'Base Rate':<10} | {'Pos N'}")
    print("-" * 85)

    for i, name in enumerate(CRAFTER_ACHIEVEMENTS):
        y_true = events[:, i].astype(np.int32)
        y_pred = pred_probs[:, i]

        pos_count = int(np.sum(y_true))
        base_rate = float(pos_count / N)
        mean_pred = float(np.mean(y_pred))

        pred_pos_05 = y_pred > 0.5
        tp = np.sum((y_pred > 0.5) & (y_true == 1))
        recall_05 = float(tp / pos_count) if pos_count > 0 else 0.0
        prec_05 = float(tp / np.sum(pred_pos_05)) if np.sum(pred_pos_05) > 0 else 0.0

        if pos_count > 0:
            pr_auc = float(average_precision_score(y_true, y_pred))
            try:
                roc_auc = float(roc_auc_score(y_true, y_pred))
            except Exception:
                roc_auc = 0.5
            active_pr_aucs.append(pr_auc)
        else:
            pr_auc = 0.0
            roc_auc = 0.5

        per_achievement[name] = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "positive_samples": pos_count,
            "base_rate": base_rate,
            "mean_pred_prob": mean_pred,
            "recall_at_0.5": recall_05,
            "precision_at_0.5": prec_05,
        }

        tag = "  <-- STONE HEAD" if name == "collect_stone" else ""
        print(f"{name:<24} | {pr_auc:<10.4f} | {recall_05:<12.4f} | {roc_auc:<10.4f} | {base_rate:<10.6f} | {pos_count}{tag}")

    macro_pr = float(np.mean(active_pr_aucs)) if active_pr_aucs else 0.0

    print("-" * 85)
    stone_stats = per_achievement.get("collect_stone", {})
    print(f"\n[Eyes-Test Summary] STONE HEAD ('collect_stone'):")
    print(f"  PR-AUC      : {stone_stats.get('pr_auc', 0.0):.6f}")
    print(f"  Recall@0.5  : {stone_stats.get('recall_at_0.5', 0.0):.4f}")
    print(f"  ROC-AUC     : {stone_stats.get('roc_auc', 0.0):.4f}")
    print(f"  Pos Samples : {stone_stats.get('positive_samples', 0)}")
    print(f"  Base Rate   : {stone_stats.get('base_rate', 0.0):.6f}")
    print(f"  Macro PR-AUC (Active Heads): {macro_pr:.4f} | Health MAE: {health_mae:.4f}")
    print("=" * 80)

    audit_result = {
        "checkpoint": str(output_json),
        "total_transitions": N,
        "macro_pr_auc_active_heads": macro_pr,
        "health_head_mae": health_mae,
        "collect_stone_head": stone_stats,
        "per_achievement": per_achievement,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(audit_result, f, indent=2)
    print(f"[Eyes-Test] Saved results JSON to: {output_json}")

    # Generate Audit Plot
    names = [a for a in CRAFTER_ACHIEVEMENTS if per_achievement[a]["positive_samples"] > 0]
    pr_vals = [per_achievement[a]["pr_auc"] for a in names]
    rec_vals = [per_achievement[a]["recall_at_0.5"] for a in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35
    colors_pr = ["#e74c3c" if n == "collect_stone" else "#3498db" for n in names]

    ax.bar(x - width / 2, pr_vals, width, label="PR-AUC", color=colors_pr)
    ax.bar(x + width / 2, rec_vals, width, label="Recall@0.5", color="#2ecc71", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("T2 Final Event Head Audit (D3 Re-run Held-out Slice)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"[Eyes-Test] Saved audit plot to: {output_png}")

    return audit_result


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2: PLANNER TEST (Pickaxe-in-Hand States Stone Probe)
# ──────────────────────────────────────────────────────────────────────────────

def run_planner_test(
    model: JEPAv3,
    all_shard_paths: List[Path],
    output_json: Path,
    device: str = "cuda",
    max_states_to_probe: int = 120,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("2. PLANNER TEST: PICKAXE-IN-HAND STATES STONE PROBE")
    print("=" * 80)

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=10,
        num_candidates=64,
        cem_iterations=2,
        device=device,
    )
    dev = torch.device(device)
    model.eval()

    # Discover all episodes containing stone and pickaxe
    print(f"[Planner-Test] Scanning {len(all_shard_paths)} shards for pickaxe and stone trajectories...")
    targeted_episodes = []

    for sp in all_shard_paths:
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
                has_pickaxe = bool(np.any(ep_events[:, IDX_WOOD_PICKAXE] > 0) or (ep_masks is not None and np.any(ep_masks[:, IDX_WOOD_PICKAXE] > 0)))

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

    print(f"[Planner-Test] Found {len(targeted_episodes)} relevant episodes ({sum(1 for e in targeted_episodes if e['has_stone'])} contain stone).")

    # Extract pickaxe-in-hand states
    pickaxe_states = []
    for ep in targeted_episodes:
        ep_obs = ep["obs"]
        ep_act = ep["actions"]
        ep_ev = ep["events"]
        ep_masks = ep["masks_before"]
        has_stone = ep["has_stone"]

        # Find stone step if present
        stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]
        t_stone = int(stone_steps[0]) if len(stone_steps) > 0 else len(ep_obs)

        # Pickaxe active range
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
                    "steps_to_stone": (t_stone - t) if has_stone else -1,
                })

    print(f"[Planner-Test] Total pickaxe-in-hand states identified: {len(pickaxe_states)}")

    # Subsample states evenly
    if len(pickaxe_states) > max_states_to_probe:
        # Prioritize stone trajectories
        stone_st = [s for s in pickaxe_states if s["is_stone_trajectory"]]
        other_st = [s for s in pickaxe_states if not s["is_stone_trajectory"]]
        stride_s = max(1, len(stone_st) // (max_states_to_probe // 2))
        stride_o = max(1, len(other_st) // (max_states_to_probe // 2))
        sampled_states = stone_st[::stride_s] + other_st[::stride_o]
        sampled_states = sampled_states[:max_states_to_probe]
    else:
        sampled_states = pickaxe_states

    print(f"[Planner-Test] Probing planner on {len(sampled_states)} pickaxe-in-hand states...")

    agreements = []
    value_gaps = []
    stone_approach_q_vals = []
    wandering_q_vals = []
    probe_records = []

    t0 = time.time()
    for idx, item in enumerate(sampled_states):
        ep = item["episode"]
        t = item["t"]
        recorded_act = int(ep["actions"][t])
        obs_seq = ep["obs"][: t + 1]
        act_seq = ep["actions"][: t + 1]

        # Compute recurrent memory h_t by sequence unroll
        with torch.no_grad():
            t_obs_seq = torch.from_numpy(obs_seq).permute(0, 3, 1, 2).float() / 255.0
            t_obs_seq = t_obs_seq.to(dev)
            t_act_seq = torch.from_numpy(act_seq).long().to(dev)

            z_seq = model.encode(t_obs_seq, use_target=False)
            h_seq = model.recurrent_core.unroll_sequence(z_seq.unsqueeze(0), t_act_seq.unsqueeze(0)).squeeze(0)
            h_t = h_seq[-1]
            z_t = z_seq[-1]
            s_t = torch.cat([z_t, h_t], dim=-1).unsqueeze(0)  # [1, 384]

            # Current state observation and belief
            curr_obs = ep["obs"][t]
            belief = AgentBeliefV3(recurrent_dim=256)
            belief.set_h(h_t.detach().cpu().numpy())
            if ep["masks_before"] is not None:
                belief.set_mask(ep["masks_before"][t])

            # 1. Run Planner
            top_action, debug = planner.plan(curr_obs, belief)
            is_agree = bool(top_action == recorded_act)
            agreements.append(is_agree)

            # 2. Evaluate Q-values across all 17 discrete actions
            all_actions_t = torch.arange(17, device=dev)
            s_t_expanded = s_t.repeat(17, 1)
            q_all = model.q_mean(s_t_expanded, all_actions_t).cpu().numpy().flatten()

            # Q for recorded/stone-progressing action
            q_stone_action = float(q_all[recorded_act])
            # Q for wandering/other actions (mean of all other 16 actions)
            other_indices = [a for a in range(17) if a != recorded_act]
            q_wander_mean = float(np.mean(q_all[other_indices]))
            q_gap = q_stone_action - q_wander_mean

            value_gaps.append(q_gap)
            stone_approach_q_vals.append(q_stone_action)
            wandering_q_vals.append(q_wander_mean)

            probe_records.append({
                "step": t,
                "recorded_action": recorded_act,
                "planned_action": top_action,
                "agreement": is_agree,
                "q_stone_action": q_stone_action,
                "q_wander_mean": q_wander_mean,
                "value_gap": q_gap,
                "is_stone_trajectory": item["is_stone_trajectory"],
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(sampled_states):
            print(f"  Probed {idx + 1:3d}/{len(sampled_states)} states ({time.time() - t0:.1f}s) | Agreement: {np.mean(agreements):.1%} | Mean Value Gap: {np.mean(value_gaps):+.4f}")

    action_agreement_rate = float(np.mean(agreements))
    mean_val_gap = float(np.mean(value_gaps))
    mean_q_stone = float(np.mean(stone_approach_q_vals))
    mean_q_wander = float(np.mean(wandering_q_vals))

    print("\n" + "-" * 60)
    print(">>> PLANNER STONE PROBE RESULTS SUMMARY <<<")
    print(f"Probed Pickaxe States      : {len(sampled_states)}")
    print(f"Action-Agreement Rate      : {action_agreement_rate * 100:.2f}%")
    print(f"Mean Q (Stone Actions)     : {mean_q_stone:.4f}")
    print(f"Mean Q (Wandering Actions) : {mean_q_wander:.4f}")
    print(f"Mean Value Gap (Stone-Wand): {mean_val_gap:+.4f}")
    print("-" * 60)

    planner_results = {
        "num_probed_states": len(sampled_states),
        "action_agreement_rate": action_agreement_rate,
        "mean_value_gap": mean_val_gap,
        "mean_q_stone_approaching": mean_q_stone,
        "mean_q_wandering": mean_q_wander,
        "probe_records": probe_records[:50],  # sample records
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(planner_results, f, indent=2)
    print(f"[Planner-Test] Saved probe JSON to: {output_json}")

    return planner_results


# ──────────────────────────────────────────────────────────────────────────────
# TEST 3: JUDGE TEST (Stone States Only D1 Q-Calibration)
# ──────────────────────────────────────────────────────────────────────────────

def run_judge_test(
    model: JEPAv3,
    all_shard_paths: List[Path],
    output_json: Path,
    device: str = "cuda",
    gamma: float = 0.99,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("3. JUDGE TEST: D1 RESTRICTED TO STONE-ADJACENT STATES")
    print("=" * 80)

    dev = torch.device(device)
    model.eval()

    all_q_pred = []
    all_realized = []
    total_stone_adjacent_transitions = 0

    print(f"[Judge-Test] Extracting stone-adjacent transitions from {len(all_shard_paths)} shards...")

    for sp in all_shard_paths:
        try:
            data = np.load(str(sp), allow_pickle=True)
            obs = data["obs"]
            actions = data["actions"]
            rewards = data["rewards"]
            events = data["events"]
            dones = data["dones"]
            masks_before = data["mask_before"] if "mask_before" in data else None

            n_trans = len(obs)
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            ep_start = 0
            for ep_end in ep_indices:
                ep_len = ep_end - ep_start + 1
                ep_ev = events[ep_start : ep_end + 1]
                ep_rew = rewards[ep_start : ep_end + 1]
                ep_obs = obs[ep_start : ep_end + 1]
                ep_act = actions[ep_start : ep_end + 1]
                ep_mask = masks_before[ep_start : ep_end + 1] if masks_before is not None else None

                # Find stone events
                stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]
                has_pickaxe_any = bool(np.any(ep_ev[:, IDX_WOOD_PICKAXE] > 0) or (ep_mask is not None and np.any(ep_mask[:, IDX_WOOD_PICKAXE] > 0)))

                if len(stone_steps) > 0 or has_pickaxe_any:
                    # Identify stone-adjacent steps in this episode:
                    # 1. Any step within 16 steps before a stone event
                    # 2. Any pickaxe-in-hand step
                    stone_adjacent_mask = np.zeros(ep_len, dtype=bool)

                    for s_step in stone_steps:
                        w_start = max(0, s_step - 16)
                        stone_adjacent_mask[w_start : s_step + 1] = True

                    if ep_mask is not None:
                        stone_adjacent_mask |= (ep_mask[:, IDX_WOOD_PICKAXE] > 0)

                    indices = np.where(stone_adjacent_mask)[0]
                    if len(indices) > 0:
                        total_stone_adjacent_transitions += len(indices)

                        # Compute discounted cumulative return from each step
                        running_ret = 0.0
                        realized_returns = np.zeros(ep_len, dtype=np.float32)
                        for t in reversed(range(ep_len)):
                            running_ret = ep_rew[t] + gamma * running_ret
                            realized_returns[t] = running_ret

                        # Predict Q for the episode
                        with torch.no_grad():
                            t_obs = torch.from_numpy(ep_obs).permute(0, 3, 1, 2).float() / 255.0
                            t_obs = t_obs.to(dev)
                            t_act = torch.from_numpy(ep_act).long().to(dev)

                            z = model.encode(t_obs, use_target=True)
                            h = model.recurrent_core.unroll_sequence(z.unsqueeze(0), t_act.unsqueeze(0)).squeeze(0)
                            s = torch.cat([z, h], dim=-1)

                            q_pred = model.q_mean(s, t_act).cpu().numpy().flatten()
                            if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                                q_pred = model.return_normalizer.denormalize(torch.tensor(q_pred)).numpy()

                            all_q_pred.extend(q_pred[indices].tolist())
                            all_realized.extend(realized_returns[indices].tolist())

                ep_start = ep_end + 1
        except Exception:
            continue

    q_arr = np.array(all_q_pred, dtype=np.float32)
    ret_arr = np.array(all_realized, dtype=np.float32)
    N_stone = len(q_arr)

    print(f"[Judge-Test] Total stone-adjacent transitions evaluated: {N_stone:,}")

    if N_stone > 10:
        r_val, p_val = pearsonr(q_arr, ret_arr)
        rho_val, rho_p = spearmanr(q_arr, ret_arr)
        mae_val = float(np.mean(np.abs(q_arr - ret_arr)))
        rmse_val = float(np.sqrt(np.mean((q_arr - ret_arr) ** 2)))
        q_pred_mean = float(np.mean(q_arr))
        q_pred_std = float(np.std(q_arr))
        ret_mean = float(np.mean(ret_arr))
        ret_std = float(np.std(ret_arr))
    else:
        r_val, p_val, rho_val, rho_p = 0.0, 1.0, 0.0, 1.0
        mae_val, rmse_val, q_pred_mean, q_pred_std, ret_mean, ret_std = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Overall benchmark numbers from T2 Final Honest Value Check
    overall_r = 0.3665
    overall_mae = 0.8661

    print("\n" + "=" * 80)
    print("      D1 Q-CALIBRATION: OVERALL (ALL STATES) vs STONE-ADJACENT SUBSET")
    print("=" * 80)
    print(f"{'Metric':<36} | {'Overall (T2 Final)':<20} | {'Stone-Adjacent Subset':<22}")
    print("-" * 80)
    print(f"{'Evaluated Transitions':<36} | {'8,255':<20} | {f'{N_stone:,}':<22}")
    print(f"{'Pearson r':<36} | {overall_r:<20.4f} | {r_val:<22.4f}")
    print(f"{'MAE':<36} | {overall_mae:<20.4f} | {mae_val:<22.4f}")
    print(f"{'Spearman rho':<36} | {'0.3405':<20} | {rho_val:<22.4f}")
    print(f"{'RMSE':<36} | {'1.1167':<20} | {rmse_val:<22.4f}")
    print(f"{'Predicted Q-Mean':<36} | {'0.3648':<20} | {q_pred_mean:<22.4f}")
    print(f"{'Predicted Q-Std (Uncollapsed)':<36} | {'0.5419':<20} | {q_pred_std:<22.4f}")
    print(f"{'Realized Return Mean':<36} | {'0.7530':<20} | {ret_mean:<22.4f}")
    print(f"{'Realized Return Std':<36} | {'1.1163':<20} | {ret_std:<22.4f}")
    print("=" * 80)

    judge_results = {
        "evaluated_stone_transitions": N_stone,
        "pearson_r_stone_subset": float(r_val),
        "pearson_p_value": float(p_val),
        "spearman_rho_stone_subset": float(rho_val),
        "mae_stone_subset": float(mae_val),
        "rmse_stone_subset": float(rmse_val),
        "q_pred_mean_stone": float(q_pred_mean),
        "q_pred_std_stone": float(q_pred_std),
        "realized_return_mean_stone": float(ret_mean),
        "realized_return_std_stone": float(ret_std),
        "baseline_comparison": {
            "overall_pearson_r": overall_r,
            "overall_mae": overall_mae,
            "delta_pearson_r": float(r_val - overall_r),
            "delta_mae": float(mae_val - overall_mae),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(judge_results, f, indent=2)
    print(f"[Judge-Test] Saved stone subset calibration JSON to: {output_json}")

    return judge_results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Step 0 Diagnostics Battery")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    ensure_drive_mounted(drive_p)

    device = args.device
    print("=" * 80)
    print("          PHASE 3 STEP 0 DIAGNOSTICS BATTERY (EYES, PLANNER, JUDGE)")
    print(f"  Google Drive Path: {drive_p} (exists: {drive_p.exists()})")
    print(f"  Device           : {device} (CUDA: {torch.cuda.is_available()})")
    print("=" * 80)

    # 1. Resolve Checkpoint
    ckpt_candidates = [
        drive_p / "checkpoints",
        Path("/content/drive/My Drive/learn-a-world/checkpoints"),
        REPO_ROOT / "checkpoints",
        Path("/content") / "checkpoints",
        Path("/content"),
    ]
    t2_ckpt = find_checkpoint("jepa_crafter_v3_t2_final.pt", ckpt_candidates)
    print(f"\n[Model] Loading T2 Final Checkpoint: {t2_ckpt} ({t2_ckpt.stat().st_size / 1e6:.1f} MB)")

    model = JEPAv3()
    model.load_checkpoint(str(t2_ckpt), device=device)
    model.to(torch.device(device))
    model.eval()

    # Load calibrator if available
    cal_paths = [
        REPO_ROOT / "results" / "calibration" / "event_head_calibrator.json",
        drive_p / "results" / "calibration" / "event_head_calibrator.json",
    ]
    for cp in cal_paths:
        if cp.exists():
            try:
                model.load_calibrator(cp)
                print(f"[Model] Attached EventHeadCalibrator from {cp}")
                break
            except Exception:
                pass

    # 2. Resolve Shards Directories
    sequences_candidates = [
        REPO_ROOT / "data" / "sequences",
        drive_p / "data" / "sequences",
    ]
    seq_dir = sequences_candidates[0] if (sequences_candidates[0].exists() and len(list(sequences_candidates[0].glob("*.npz"))) > 0) else sequences_candidates[1]

    curiosity_candidates = [
        drive_p / "data" / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]
    cur_dir = curiosity_candidates[0] if (curiosity_candidates[0].exists() and len(list(curiosity_candidates[0].glob("*.npz"))) > 0) else curiosity_candidates[1]

    all_shard_paths = sorted(cur_dir.glob("*.npz")) if cur_dir.exists() else []
    if seq_dir.exists():
        all_shard_paths.extend(sorted(seq_dir.glob("*.npz")))

    print(f"[Data] Found {len(all_shard_paths)} total shards across curiosity & sequence dirs.")

    # 3. Output paths
    out_eyes_json = REPO_ROOT / "results" / "diagnostics" / "t2_final_head_audit.json"
    out_eyes_png = REPO_ROOT / "results" / "diagnostics" / "t2_final_head_audit.png"
    out_planner_json = REPO_ROOT / "results" / "diagnostics" / "t2_planner_stone_probe.json"
    out_judge_json = REPO_ROOT / "results" / "diagnostics" / "t2_final_q_calibration_stone_subset.json"

    # ──────────────────────────────────────────────────────────────────────────
    # RUN TEST 1: EYES TEST
    # ──────────────────────────────────────────────────────────────────────────
    eyes_res = run_eyes_test(
        model=model,
        sequences_dir=seq_dir,
        output_json=out_eyes_json,
        output_png=out_eyes_png,
        device=device,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # RUN TEST 2: PLANNER TEST
    # ──────────────────────────────────────────────────────────────────────────
    planner_res = run_planner_test(
        model=model,
        all_shard_paths=all_shard_paths,
        output_json=out_planner_json,
        device=device,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # RUN TEST 3: JUDGE TEST
    # ──────────────────────────────────────────────────────────────────────────
    judge_res = run_judge_test(
        model=model,
        all_shard_paths=all_shard_paths,
        output_json=out_judge_json,
        device=device,
    )

    # Copy all outputs to Drive
    if drive_p.exists():
        drive_diag = drive_p / "results" / "diagnostics"
        drive_diag.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(out_eyes_json), str(drive_diag / out_eyes_json.name))
        shutil.copy2(str(out_eyes_png), str(drive_diag / out_eyes_png.name))
        shutil.copy2(str(out_planner_json), str(drive_diag / out_planner_json.name))
        shutil.copy2(str(out_judge_json), str(drive_diag / out_judge_json.name))
        print(f"\n[Drive] Mirrored all 4 diagnostic artifacts to Drive: {drive_diag}")

    print("\n" + "=" * 80)
    print("STEP 0 DIAGNOSTICS BATTERY COMPLETE")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMPS FOR STEP 0 (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print("\n--- BEGIN t2_final_head_audit.json ---")
    print(json.dumps(eyes_res, indent=2))
    print("--- END t2_final_head_audit.json ---\n")

    print("\n--- BEGIN t2_planner_stone_probe.json ---")
    print(json.dumps(planner_res, indent=2))
    print("--- END t2_planner_stone_probe.json ---\n")

    print("\n--- BEGIN t2_final_q_calibration_stone_subset.json ---")
    print(json.dumps(judge_res, indent=2))
    print("--- END t2_final_q_calibration_stone_subset.json ---\n")


if __name__ == "__main__":
    main()
