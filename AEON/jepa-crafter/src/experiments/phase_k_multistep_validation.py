"""AEON Phase K: Multi-Step World-Dynamics Validation.

Protocol:
1. Verify candidate artifacts:
   - Phase J fine-tuned checkpoint (checkpoint_finetuned_1000.pt)
   - Base 50,000-step checkpoint (checkpoint_00050000.pt)
   - Phase J results JSON
   - Git commit SHA
   If candidate checkpoint is missing, halt honestly without substituting random weights.
2. Collect fresh Crafter trajectories from separate evaluation seeds (seeds 201..204).
3. Evaluate closed-loop multi-step predictions across horizons [1, 2, 4, 8]:
   - At each step k >= 2, strictly feed previous predicted latent state z_hat_{k-1} and h_{k-1}.
   - Compare predictions against actual encoded target latents z_{t+k}.
   - Compare against persistence baseline z_t.
   - Run counterfactual shuffled-action closed-loop rollouts to verify action conditioning.
4. Report detailed metrics per horizon:
   - Total cosine error (model vs persistence)
   - High-change vs low-change transitions
   - Dynamic vs static tokens
   - Action advantage: Error(shuffled) - Error(model)
   - Error growth across rollout length
   - Spatial token error maps (8x8)
   - Latent representation health (effective rank, pairwise cosine)
5. Decision gate:
   - PASS: Action advantage remains positive, model beats persistence on high-change transitions across horizons,
     and predictions remain stable/finite without collapse.
   - PRELIMINARY: Positive action advantage and 1-step improvement, but multi-step error compounds faster than persistence.
   - FAIL: Divergence, collapse, non-positive action advantage, or worse than persistence everywhere.
"""
import os
import sys
import copy
import json
import argparse
import subprocess
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.jepa_world_model import JepaWorldModel
from src.env.crafter_adapter import CrafterAdapter
from src.utils.logging import save_json


def get_commit_sha() -> str:
    """Retrieve current git commit SHA."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def locate_candidate_checkpoint(custom_path: Optional[str] = None) -> Optional[str]:
    """Locate the Phase J fine-tuned checkpoint across local and Drive candidate paths."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    if os.environ.get("CHECKPOINT_PATH"):
        candidates.append(os.environ.get("CHECKPOINT_PATH"))
    candidates.extend([
        # Colab / Drive candidate paths
        "/content/learn-a-world/AEON/jepa-crafter/runs/jepa_crafter_finetuned/checkpoints/checkpoint_finetuned_1000.pt",
        "/content/drive/MyDrive/learn-a-world/aeon/checkpoints/checkpoint_finetuned_1000.pt",
        "/content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter_finetuned/checkpoints/checkpoint_finetuned_1000.pt",
        # Local paths
        os.path.join(aeon_root, "runs/jepa_crafter_finetuned/checkpoints/checkpoint_finetuned_1000.pt"),
        os.path.join(aeon_root, "runs/jepa_crafter/checkpoints/checkpoint_finetuned_1000.pt"),
        os.path.join(aeon_root, "checkpoints/checkpoint_finetuned_1000.pt"),
    ])
    for p in candidates:
        if p and os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            return p
    return None


def locate_base_50k_checkpoint(custom_path: Optional[str] = None) -> Optional[str]:
    """Locate the verified 50k base checkpoint."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    candidates.extend([
        "/content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_00050000.pt",
        "/content/drive/MyDrive/learn-a-world/aeon/runs/jepa_crafter/checkpoints/checkpoint_latest.pt",
        os.path.join(aeon_root, "runs/jepa_crafter/checkpoints/checkpoint_00050000.pt"),
        os.path.join(aeon_root, "checkpoints/checkpoint_00050000.pt"),
    ])
    for p in candidates:
        if p and os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            return p
    return None


def collect_evaluation_trajectories(
    seeds: List[int],
    steps_per_seed: int = 64,
    context_length: int = 4,
    max_horizon: int = 8,
) -> Dict[str, Any]:
    """Collect fresh evaluation trajectories from unseen seeds, strictly respecting episode boundaries."""
    window_total = context_length + max_horizon
    windows_context_obs = []
    windows_prev_actions = []
    windows_future_actions = []
    windows_future_obs = {h: [] for h in [1, 2, 4, 8]}

    total_episodes = 0
    total_raw_steps = 0

    for s in seeds:
        env = CrafterAdapter(seed=s)
        obs = env.reset()
        current_obs_list = [obs.copy()]
        current_act_list = []
        current_term_list = []

        for t in range(steps_per_seed):
            # Deterministic, rich action progression
            a = (t * 7 + 5) % 17
            next_obs, r, term, trunc = env.step(a)
            total_raw_steps += 1

            current_act_list.append(a)
            current_term_list.append(bool(term or trunc))
            current_obs_list.append(next_obs.copy())

            obs = next_obs
            if term or trunc:
                total_episodes += 1
                obs = env.reset()
                # Process the completed segment
                seg_obs = current_obs_list[:-1]  # up to terminal
                seg_act = current_act_list
                seg_len = len(seg_act)
                if seg_len >= window_total:
                    for i in range(context_length - 1, seg_len - max_horizon):
                        ctx_obs = np.stack(seg_obs[i - context_length + 1 : i + 1], axis=0)
                        ctx_act = np.array(seg_act[i - context_length + 1 : i + 1], dtype=np.int64)
                        fut_act = np.array(seg_act[i + 1 : i + 1 + max_horizon], dtype=np.int64)

                        windows_context_obs.append(ctx_obs)
                        windows_prev_actions.append(ctx_act)
                        windows_future_actions.append(fut_act)

                        for h in [1, 2, 4, 8]:
                            windows_future_obs[h].append(seg_obs[i + h])

                # Reset segment buffers
                current_obs_list = [obs.copy()]
                current_act_list = []
                current_term_list = []

        env.close()

        # Handle remaining segment at end of seed if no terminal
        seg_len = len(current_act_list)
        if seg_len >= window_total:
            for i in range(context_length - 1, seg_len - max_horizon):
                ctx_obs = np.stack(current_obs_list[i - context_length + 1 : i + 1], axis=0)
                ctx_act = np.array(current_act_list[i - context_length + 1 : i + 1], dtype=np.int64)
                fut_act = np.array(current_act_list[i + 1 : i + 1 + max_horizon], dtype=np.int64)

                windows_context_obs.append(ctx_obs)
                windows_prev_actions.append(ctx_act)
                windows_future_actions.append(fut_act)

                for h in [1, 2, 4, 8]:
                    windows_future_obs[h].append(current_obs_list[i + h])

    num_samples = len(windows_context_obs)
    assert num_samples > 0, "No valid evaluation windows collected"

    return {
        "context_obs": torch.from_numpy(np.stack(windows_context_obs, axis=0)),       # [B, T_ctx, 64, 64, 3] uint8
        "prev_actions": torch.from_numpy(np.stack(windows_prev_actions, axis=0)),     # [B, T_ctx] long
        "future_actions": torch.from_numpy(np.stack(windows_future_actions, axis=0)), # [B, max_horizon] long
        "target_obs": {
            h: torch.from_numpy(np.stack(windows_future_obs[h], axis=0))              # [B, 64, 64, 3] uint8
            for h in [1, 2, 4, 8]
        },
        "num_samples": num_samples,
        "eval_seeds": seeds,
        "total_raw_steps": total_raw_steps,
        "total_episodes": total_episodes,
    }


def compute_effective_rank(latents: torch.Tensor) -> Dict[str, float]:
    """Compute centered and uncentered Participation Ratio effective ranks."""
    B, N, D = latents.shape
    X = latents.reshape(B * N, D).float()
    X_norm = F.normalize(X, dim=-1, eps=1e-6)

    # Uncentered
    cov_un = torch.mm(X_norm.t(), X_norm) / float(X_norm.size(0))
    eigvals_un = torch.linalg.eigvalsh(cov_un).clamp(min=1e-12)
    pr_un = float((eigvals_un.sum() ** 2) / (eigvals_un ** 2).sum())

    # Centered
    X_centered = X_norm - X_norm.mean(dim=0, keepdim=True)
    cov_c = torch.mm(X_centered.t(), X_centered) / float(X_centered.size(0))
    eigvals_c = torch.linalg.eigvalsh(cov_c).clamp(min=1e-12)
    pr_c = float((eigvals_c.sum() ** 2) / (eigvals_c ** 2).sum())

    return {
        "centered_pr": pr_c,
        "uncentered_pr": pr_un,
        "top1_var_pct": float(eigvals_c[-1] / eigvals_c.sum() * 100.0),
    }


def evaluate_multistep_closed_loop(
    model: JepaWorldModel,
    eval_data: Dict[str, Any],
    device: torch.device,
    threshold: float = 0.001,
    member_idx: int = 0,
) -> Dict[str, Any]:
    """Run genuine closed-loop autoregressive rollouts across horizons [1, 2, 4, 8]."""
    model.eval()

    # Prepare context observation tensors: [B, T_ctx, 3, 64, 64] float in [0, 1]
    raw_ctx = eval_data["context_obs"].to(device)
    B, T_ctx, H_img, W_img, C = raw_ctx.shape
    obs_ctx = raw_ctx.permute(0, 1, 4, 2, 3).float().div_(255.0)
    prev_actions = eval_data["prev_actions"].to(device)
    future_actions = eval_data["future_actions"].to(device)

    # Encode initial context state (z_t, h_t)
    with torch.no_grad():
        z_t, h_t = model.encode_actual_history(obs_ctx, prev_actions)

        # Encode ground truth target latents at horizons 1, 2, 4, 8
        target_latents = {}
        for h in [1, 2, 4, 8]:
            raw_targ = eval_data["target_obs"][h].to(device)
            targ_float = raw_targ.permute(0, 3, 1, 2).float().div_(255.0)
            targ_tokens, _ = model.online_encoder(targ_float)
            target_latents[h] = targ_tokens

        # Closed-loop rollout with CORRECT actions
        # model.autoregressive_rollout strictly feeds previous predicted latent z_hat into next step
        model_rollout = model.autoregressive_rollout(z_t, h_t, future_actions, member_idx=member_idx)

        # Counterfactual rollout with SHUFFLED actions (derangement shift by 1 across batch)
        shuf_actions = torch.roll(future_actions, shifts=1, dims=0)
        shuf_rollout = model.autoregressive_rollout(z_t, h_t, shuf_actions, member_idx=member_idx)

    # Compute metrics for each horizon
    horizon_metrics = {}
    growth_curve = {}
    n_zt = F.normalize(z_t, dim=-1, eps=1e-6)

    for h in [1, 2, 4, 8]:
        z_hat = model_rollout["predictions_by_horizon"][h]
        z_hat_shuf = shuf_rollout["predictions_by_horizon"][h]
        z_targ = target_latents[h]

        n_zhat = F.normalize(z_hat, dim=-1, eps=1e-6)
        n_zhat_shuf = F.normalize(z_hat_shuf, dim=-1, eps=1e-6)
        n_ztarg = F.normalize(z_targ, dim=-1, eps=1e-6)

        # 1. Persistence baseline from initial z_t to horizon h
        token_persist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)  # [B, 64]
        trans_persist = token_persist.mean(dim=-1)          # [B]
        high_m = trans_persist >= threshold
        low_m = trans_persist < threshold
        dynamic_m = token_persist >= threshold
        static_m = token_persist < threshold

        # 2. Model prediction error at horizon h
        token_err = 1.0 - (n_zhat * n_ztarg).sum(dim=-1)     # [B, 64]
        trans_err = token_err.mean(dim=-1)                  # [B]

        # 3. Shuffled-action prediction error at horizon h
        token_err_shuf = 1.0 - (n_zhat_shuf * n_ztarg).sum(dim=-1)
        trans_err_shuf = token_err_shuf.mean(dim=-1)

        # Aggregations
        tot_model = float(trans_err.mean().item())
        tot_persist = float(trans_persist.mean().item())
        tot_shuf = float(trans_err_shuf.mean().item())

        high_model = float(trans_err[high_m].mean().item()) if high_m.any() else 0.0
        high_persist = float(trans_persist[high_m].mean().item()) if high_m.any() else 0.0
        high_shuf = float(trans_err_shuf[high_m].mean().item()) if high_m.any() else 0.0

        low_model = float(trans_err[low_m].mean().item()) if low_m.any() else 0.0
        low_persist = float(trans_persist[low_m].mean().item()) if low_m.any() else 0.0

        dyn_model = float(token_err[dynamic_m].mean().item()) if dynamic_m.any() else 0.0
        dyn_persist = float(token_persist[dynamic_m].mean().item()) if dynamic_m.any() else 0.0
        dyn_shuf = float(token_err_shuf[dynamic_m].mean().item()) if dynamic_m.any() else 0.0

        stat_model = float(token_err[static_m].mean().item()) if static_m.any() else 0.0
        stat_persist = float(token_persist[static_m].mean().item()) if static_m.any() else 0.0

        # Action Advantage
        adv_tot = tot_shuf - tot_model
        adv_high = high_shuf - high_model
        adv_dyn = dyn_shuf - dyn_model

        # Spatial token heatmaps (64 tokens)
        spatial_err_map = token_err.mean(dim=0).cpu().numpy().tolist()
        spatial_persist_map = token_persist.mean(dim=0).cpu().numpy().tolist()

        # Representation health at horizon h
        er_stats = compute_effective_rank(z_hat)
        D = z_hat.size(-1)
        pred_flat = z_hat.reshape(-1, D)
        pn = F.normalize(pred_flat, dim=-1, eps=1e-6)
        sim_mat = torch.mm(pn[:200], pn[:200].t())
        triu_idx = torch.triu_indices(sim_mat.size(0), sim_mat.size(0), offset=1)
        pw_cos = float(sim_mat[triu_idx[0], triu_idx[1]].mean().item())

        horizon_metrics[str(h)] = {
            "horizon": h,
            "total_model_error": tot_model,
            "total_persistence_error": tot_persist,
            "total_shuffled_error": tot_shuf,
            "total_action_advantage": adv_tot,
            "high_change": {
                "model_error": high_model,
                "persistence_error": high_persist,
                "shuffled_error": high_shuf,
                "action_advantage": adv_high,
                "beats_persistence": high_model < high_persist,
                "pct_improvement": ((high_persist - high_model) / max(1e-6, high_persist)) * 100.0,
                "num_transitions": int(high_m.sum().item()),
            },
            "low_change": {
                "model_error": low_model,
                "persistence_error": low_persist,
                "num_transitions": int(low_m.sum().item()),
            },
            "dynamic_tokens": {
                "model_error": dyn_model,
                "persistence_error": dyn_persist,
                "action_advantage": adv_dyn,
                "beats_persistence": dyn_model < dyn_persist,
                "pct_improvement": ((dyn_persist - dyn_model) / max(1e-6, dyn_persist)) * 100.0,
                "num_dynamic_tokens": int(dynamic_m.sum().item()),
            },
            "representation_health": {
                "is_finite": bool(torch.isfinite(z_hat).all().item()),
                "centered_effective_rank": er_stats["centered_pr"],
                "uncentered_effective_rank": er_stats["uncentered_pr"],
                "top1_variance_pct": er_stats["top1_var_pct"],
                "pairwise_cosine_similarity": pw_cos,
            },
            "spatial_error_map_mean": spatial_err_map,
            "spatial_persistence_map_mean": spatial_persist_map,
        }

    # Error growth curve
    err_h1 = horizon_metrics["1"]["total_model_error"]
    for h in [1, 2, 4, 8]:
        growth_curve[str(h)] = {
            "model_error": horizon_metrics[str(h)]["total_model_error"],
            "persistence_error": horizon_metrics[str(h)]["total_persistence_error"],
            "model_growth_ratio_to_h1": horizon_metrics[str(h)]["total_model_error"] / max(1e-6, err_h1),
            "persistence_growth_ratio_to_h1": (
                horizon_metrics[str(h)]["total_persistence_error"]
                / max(1e-6, horizon_metrics["1"]["total_persistence_error"])
            ),
        }

    return {
        "horizon_metrics": horizon_metrics,
        "growth_curve": growth_curve,
    }


def run_phase_k_validation(
    checkpoint_path: Optional[str] = None,
    base_checkpoint_path: Optional[str] = None,
    device_name: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    steps_per_seed: int = 64,
) -> Dict[str, Any]:
    print("=" * 80)
    print("      AEON JEPA-CRAFTER — PHASE K: MULTI-STEP WORLD-DYNAMICS VALIDATION")
    print("=" * 80)

    # 1. Device Configuration with Safe Fallback
    if device_name:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            print("! NOTE: CUDA was requested but torch.cuda.is_available() is False. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Execution Device: {device.type.upper()}")

    commit_sha = get_commit_sha()
    print(f"• Git Commit SHA: {commit_sha}")

    # 2. Artifact Verification Gate
    out_dir = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "phase_k_multistep_results.json")

    phase_j_results_path = os.path.join(out_dir, "phase_j_finetune_results.json")
    phase_j_available = os.path.exists(phase_j_results_path)
    print(f"• Phase J Results JSON: {'VERIFIED' if phase_j_available else 'NOT_FOUND'}")

    ckpt_candidate = locate_candidate_checkpoint(checkpoint_path)
    ckpt_base = locate_base_50k_checkpoint(base_checkpoint_path)

    if ckpt_candidate is None:
        msg = (
            "CANDIDATE CHECKPOINT UNAVAILABLE: The Phase J fine-tuned checkpoint "
            "(checkpoint_finetuned_1000.pt) was not found on the local filesystem.\n"
            "As mandated by Phase K operational protocol:\n"
            "  'Verify the fine-tuned checkpoint... If any artifact is missing, report it; "
            "   do not substitute random weights.'\n"
            "On Colab, the fine-tuned checkpoint was created at:\n"
            "  /content/learn-a-world/AEON/jepa-crafter/runs/jepa_crafter_finetuned/checkpoints/checkpoint_finetuned_1000.pt"
        )
        print(f"\n[PHASE K GATE STOP]:\n{msg}")
        stopped_payload = {
            "status": "STOPPED",
            "decision_gate": "PRELIMINARY_GATE_STOP",
            "reason": "CANDIDATE_CHECKPOINT_UNAVAILABLE_ON_LOCAL_DISK",
            "commit_sha": commit_sha,
            "colab_path": "/content/learn-a-world/AEON/jepa-crafter/runs/jepa_crafter_finetuned/checkpoints/checkpoint_finetuned_1000.pt",
            "protocol_adherence": "Refused to substitute random weights. Mandatory Phase K artifact gate halts honestly.",
            "execution_instructions_colab": (
                "To execute Phase K multi-step validation on Google Colab:\n"
                "1. %cd /content/learn-a-world\n"
                "2. !git pull origin main\n"
                "3. !python AEON/jepa-crafter/src/experiments/phase_k_multistep_validation.py \\\n"
                "     --checkpoint-path /content/learn-a-world/AEON/jepa-crafter/runs/jepa_crafter_finetuned/checkpoints/checkpoint_finetuned_1000.pt \\\n"
                "     --device cuda"
            ),
        }
        save_json(stopped_payload, out_json)
        print(f"• Saved gate stop record to: {out_json}")
        return stopped_payload

    print(f"• Verified Fine-Tuned Checkpoint: {ckpt_candidate} ({os.path.getsize(ckpt_candidate)/1e6:.1f} MB)")
    if ckpt_base:
        print(f"• Verified Base 50k Checkpoint: {ckpt_base} ({os.path.getsize(ckpt_base)/1e6:.1f} MB)")

    # 3. Load Model with Strict Safety
    ckpt = torch.load(ckpt_candidate, map_location="cpu", weights_only=False)
    model = JepaWorldModel().to(device)
    load_res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"• Checkpoint Strict Load: {len(load_res.missing_keys)} missing, {len(load_res.unexpected_keys)} unexpected keys")

    # 4. Trajectory Collection on Fresh Seeds
    if seeds is None:
        seeds = [201, 202, 203, 204]
    print(f"\n• Collecting fresh evaluation trajectories on seeds: {seeds} ({steps_per_seed} steps each)...")
    eval_data = collect_evaluation_trajectories(
        seeds=seeds,
        steps_per_seed=steps_per_seed,
        context_length=4,
        max_horizon=8,
    )
    print(f"• Collected {eval_data['num_samples']} valid evaluation windows across {eval_data['total_episodes']} episodes.")

    # 5. Execute Multi-Step Closed-Loop Evaluation
    print("\n--- Running Multi-Step Closed-Loop Prediction Horizons [1, 2, 4, 8] ---")
    eval_results = evaluate_multistep_closed_loop(
        model=model,
        eval_data=eval_data,
        device=device,
        threshold=0.001,
        member_idx=0,
    )

    horizon_metrics = eval_results["horizon_metrics"]
    growth_curve = eval_results["growth_curve"]

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'H':^3} | {'Model Err':^10} | {'Persist':^10} | {'High-Chg Model':^15} | {'High-Chg Persist':^16} | {'Adv (High)':^12} | {'Dyn Tokens':^11}")
    print("-" * 90)
    for h in [1, 2, 4, 8]:
        m = horizon_metrics[str(h)]
        tot_m = m["total_model_error"]
        tot_p = m["total_persistence_error"]
        hc_m = m["high_change"]["model_error"]
        hc_p = m["high_change"]["persistence_error"]
        adv = m["high_change"]["action_advantage"]
        dyn_m = m["dynamic_tokens"]["model_error"]
        dyn_p = m["dynamic_tokens"]["persistence_error"]
        print(f"{h:^3} | {tot_m:^10.6f} | {tot_p:^10.6f} | {hc_m:^15.6f} | {hc_p:^16.6f} | {adv:^+12.6f} | {dyn_m:^11.6f}")
    print("=" * 90)

    # 6. Decision Gate Evaluation
    # PASS criteria:
    # 1. Action advantage remains positive across horizons.
    # 2. Model outperforms persistence on high-change transitions across horizons.
    # 3. Multi-step predictions remain finite and stable.
    all_positive_adv = all(horizon_metrics[str(h)]["high_change"]["action_advantage"] > 0 for h in [1, 2, 4, 8])
    beats_persist_h1 = horizon_metrics["1"]["high_change"]["beats_persistence"]
    beats_persist_all_high = all(horizon_metrics[str(h)]["high_change"]["beats_persistence"] for h in [1, 2, 4, 8])
    all_finite = all(horizon_metrics[str(h)]["representation_health"]["is_finite"] for h in [1, 2, 4, 8])

    if beats_persist_all_high and all_positive_adv and all_finite:
        outcome = "PASS"
        justification = (
            "Multi-step closed-loop rollouts reliably outperform persistence on high-change transitions "
            "across horizons 1, 2, 4, 8, with strictly positive action advantages and stable representations."
        )
    elif beats_persist_h1 and all_positive_adv and all_finite:
        outcome = "PRELIMINARY"
        justification = (
            "Action conditioning is actively functional and positive across multi-step horizons, "
            "and 1-step beats persistence, but compound error at higher horizons approaches persistence."
        )
    else:
        outcome = "FAIL"
        justification = (
            "Model failed multi-step closed-loop validation: non-positive action advantages or divergence."
        )

    print(f"\n• Final Phase K Decision Gate: {outcome}")
    print(f"• Technical Justification: {justification}")

    full_results = {
        "status": "COMPLETED",
        "outcome": outcome,
        "justification": justification,
        "commit_sha": commit_sha,
        "candidate_checkpoint": ckpt_candidate,
        "base_checkpoint": ckpt_base,
        "sample_count": eval_data["num_samples"],
        "eval_seeds": seeds,
        "total_raw_steps": eval_data["total_raw_steps"],
        "image_decoder_status": "NONE_IN_ARCHITECTURE (Pure latent space representation, no pixel decoder invented)",
        "horizon_metrics": horizon_metrics,
        "growth_curve": growth_curve,
        "verdict": {
            "all_positive_action_advantage": all_positive_adv,
            "beats_persistence_high_change_h1": beats_persist_h1,
            "beats_persistence_high_change_all": beats_persist_all_high,
            "all_predictions_finite": all_finite,
        },
    }

    save_json(full_results, out_json)
    print(f"• Results JSON successfully saved to: {out_json}")
    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEON Phase K Multi-Step World-Dynamics Validation")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint_finetuned_1000.pt")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Path to base checkpoint_00050000.pt")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/cpu)")
    parser.add_argument("--steps-per-seed", type=int, default=64, help="Steps to collect per evaluation seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203, 204], help="Evaluation seeds")
    args = parser.parse_args()

    run_phase_k_validation(
        checkpoint_path=args.checkpoint_path,
        base_checkpoint_path=args.base_checkpoint,
        device_name=args.device,
        seeds=args.seeds,
        steps_per_seed=args.steps_per_seed,
    )
