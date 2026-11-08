"""Gate 4 Evaluator: Static vs Changed Token Error Decomposition and Multi-Step Probing.

Strictly follows Project ATLAS protocols:
- Operates purely on visual latents, recurrent belief, and discrete actions.
- Zero access to simulator internals, info dicts, coordinates, or recipe tables.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from true_jepa.model import LatentFrame, TrueJEPA


def decompose_static_and_changed_errors(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    change_threshold: float = 0.1,
) -> Dict[str, float]:
    """Decomposes one-step prediction errors into static vs changed spatial tokens.

    Args:
        pred_tokens: [..., K, D] model prediction
        target_tokens: [..., K, D] true target
        current_tokens: [..., K, D] previous observation (persistence baseline)
        change_threshold: L2 displacement threshold for a token to be classified as changed.

    Returns:
        Dictionary of MSEs and relative error ratios for static, changed, and overall tokens.
    """
    with torch.no_grad():
        # Ensure flat [N, D] or [N, K, D] shapes
        p = pred_tokens.reshape(-1, pred_tokens.shape[-2], pred_tokens.shape[-1])
        t = target_tokens.reshape(-1, target_tokens.shape[-2], target_tokens.shape[-1])
        c = current_tokens.reshape(-1, current_tokens.shape[-2], current_tokens.shape[-1])

        # Target displacement per token: [N, K]
        displacement = torch.norm(t - c, p=2, dim=-1)

        changed_mask = displacement >= change_threshold
        static_mask = ~changed_mask

        # Squared errors per token: [N, K]
        err_model_tok = (p - t).square().mean(dim=-1)
        err_persist_tok = (c - t).square().mean(dim=-1)

        # Overall
        mse_total_model = float(err_model_tok.mean().item())
        mse_total_persist = float(err_persist_tok.mean().item())
        rel_err_total = mse_total_model / max(mse_total_persist, 1e-8)

        # Static
        if static_mask.any():
            mse_static_model = float(err_model_tok[static_mask].mean().item())
            mse_static_persist = float(err_persist_tok[static_mask].mean().item())
            rel_err_static = mse_static_model / max(mse_static_persist, 1e-8)
        else:
            mse_static_model = 0.0
            mse_static_persist = 0.0
            rel_err_static = 1.0

        # Changed
        if changed_mask.any():
            mse_changed_model = float(err_model_tok[changed_mask].mean().item())
            mse_changed_persist = float(err_persist_tok[changed_mask].mean().item())
            rel_err_changed = mse_changed_model / max(mse_changed_persist, 1e-8)
        else:
            mse_changed_model = 0.0
            mse_changed_persist = 0.0
            rel_err_changed = 1.0

        frac_changed = float(changed_mask.float().mean().item())

        return {
            "mse_total_model": mse_total_model,
            "mse_total_persist": mse_total_persist,
            "rel_err_total": rel_err_total,
            "mse_static_model": mse_static_model,
            "mse_static_persist": mse_static_persist,
            "rel_err_static": rel_err_static,
            "mse_changed_model": mse_changed_model,
            "mse_changed_persist": mse_changed_persist,
            "rel_err_changed": rel_err_changed,
            "fraction_changed_tokens": frac_changed,
        }


@torch.no_grad()
def evaluate_gate4_formulation(
    model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
    change_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Evaluates Gate 4 one-step prediction, persistence ratio, static/changed decomposition,

    shuffled-action control, and multi-step teacher-forced vs free-running rollouts.
    """
    model.eval()

    total_model_errors = []
    total_persist_errors = []
    shuffled_errors = []

    static_model_errors = []
    static_persist_errors = []
    changed_model_errors = []
    changed_persist_errors = []
    changed_fractions = []

    horizons = [1, 2, 4, 8]
    tf_errors: Dict[int, List[float]] = {h: [] for h in horizons}
    fr_errors: Dict[int, List[float]] = {h: [] for h in horizons}

    for ep in eval_dataset:
        obs = torch.from_numpy(ep["obs"]).to(device)  # [T+1, 64, 64, 3]
        actions = torch.from_numpy(ep["action"]).to(device)  # [T]
        rewards = torch.from_numpy(ep["reward"]).to(device)  # [T]
        T = actions.shape[0]

        if T < 10:
            continue

        # Encode all observations
        online_frames = model.encode(obs)
        target_frames = model.target_encode(obs)

        # 1-Step Predictive Utility & Controls
        hidden = model.initial_belief(batch=1, device=device)
        for t in range(min(T - 1, 50)):
            curr_frame = LatentFrame(
                tokens=online_frames.tokens[t : t + 1],
                pooled=online_frames.pooled[t : t + 1],
            )
            tgt_frame = LatentFrame(
                tokens=target_frames.tokens[t + 1 : t + 2],
                pooled=target_frames.pooled[t + 1 : t + 2],
            )
            act = actions[t : t + 1]
            pred = model.predict(curr_frame, act, hidden)

            # Static vs Changed decomposition on normalized tokens
            p_norm = F.normalize(pred.tokens, p=2, dim=-1)
            t_norm = F.normalize(tgt_frame.tokens, p=2, dim=-1)
            c_norm = F.normalize(curr_frame.tokens, p=2, dim=-1)

            decomp = decompose_static_and_changed_errors(
                pred_tokens=p_norm,
                target_tokens=t_norm,
                current_tokens=c_norm,
                change_threshold=change_threshold,
            )

            total_model_errors.append(decomp["mse_total_model"])
            total_persist_errors.append(decomp["mse_total_persist"])
            if decomp["fraction_changed_tokens"] > 0:
                changed_model_errors.append(decomp["mse_changed_model"])
                changed_persist_errors.append(decomp["mse_changed_persist"])
            static_model_errors.append(decomp["mse_static_model"])
            static_persist_errors.append(decomp["mse_static_persist"])
            changed_fractions.append(decomp["fraction_changed_tokens"])

            # Shuffled Action Control
            shuf_act = (act + 7) % 17
            pred_shuf = model.predict(curr_frame, shuf_act, hidden)
            p_shuf_norm = F.normalize(pred_shuf.tokens, p=2, dim=-1)
            e_shuf = (p_shuf_norm - t_norm).square().mean().item()
            shuffled_errors.append(e_shuf)

            # Step belief forward
            prev_r = rewards[t : t + 1]
            hidden = model.belief_step(curr_frame, act, prev_r, hidden)

        # Multi-Step Rollout Horizons
        for start_idx in range(0, min(T - 8, 40), 8):
            h_tf = model.initial_belief(batch=1, device=device)
            h_fr = h_tf.clone()
            cur_fr = LatentFrame(
                tokens=online_frames.tokens[start_idx : start_idx + 1],
                pooled=online_frames.pooled[start_idx : start_idx + 1],
            )

            for step in range(1, 9):
                step_idx = start_idx + step - 1
                act = actions[step_idx : step_idx + 1]
                rew = rewards[step_idx : step_idx + 1]
                cur_tf = LatentFrame(
                    tokens=online_frames.tokens[step_idx : step_idx + 1],
                    pooled=online_frames.pooled[step_idx : step_idx + 1],
                )
                tgt = LatentFrame(
                    tokens=target_frames.tokens[step_idx + 1 : step_idx + 2],
                    pooled=target_frames.pooled[step_idx + 1 : step_idx + 2],
                )

                # Teacher-forced prediction
                pred_tf = model.predict(cur_tf, act, h_tf)
                h_tf = model.belief_step(cur_tf, act, rew, h_tf)

                # Free-running prediction
                pred_fr = model.predict(cur_fr, act, h_fr)
                h_fr = model.belief_step(pred_fr, act, rew, h_fr)
                cur_fr = pred_fr

                if step in horizons:
                    p_tf_n = F.normalize(pred_tf.tokens, p=2, dim=-1)
                    p_fr_n = F.normalize(pred_fr.tokens, p=2, dim=-1)
                    t_n = F.normalize(tgt.tokens, p=2, dim=-1)

                    tf_errors[step].append((p_tf_n - t_n).square().mean().item())
                    fr_errors[step].append((p_fr_n - t_n).square().mean().item())

    # Aggregate metrics
    mean_model_err = float(np.mean(total_model_errors))
    mean_persist_err = float(np.mean(total_persist_errors))
    mean_shuf_err = float(np.mean(shuffled_errors))
    rel_pred_err = mean_model_err / max(mean_persist_err, 1e-8)

    mean_static_model = float(np.mean(static_model_errors)) if static_model_errors else 0.0
    mean_static_persist = float(np.mean(static_persist_errors)) if static_persist_errors else 0.0
    rel_static = mean_static_model / max(mean_static_persist, 1e-8)

    mean_changed_model = float(np.mean(changed_model_errors)) if changed_model_errors else 0.0
    mean_changed_persist = float(np.mean(changed_persist_errors)) if changed_persist_errors else 0.0
    rel_changed = mean_changed_model / max(mean_changed_persist, 1e-8)

    action_sensitive = mean_model_err < mean_shuf_err
    shuf_gap = mean_shuf_err - mean_model_err

    return {
        "one_step_model_error": mean_model_err,
        "one_step_persistence_error": mean_persist_err,
        "one_step_relative_error": rel_pred_err,
        "one_step_shuffled_error": mean_shuf_err,
        "shuffled_action_sensitive": bool(action_sensitive),
        "shuffled_action_sensitivity_gap": float(shuf_gap),
        "static_tokens": {
            "mse_model": mean_static_model,
            "mse_persist": mean_static_persist,
            "relative_error": rel_static,
        },
        "changed_tokens": {
            "mse_model": mean_changed_model,
            "mse_persist": mean_changed_persist,
            "relative_error": rel_changed,
        },
        "average_fraction_changed_tokens": float(np.mean(changed_fractions)),
        "teacher_forced_errors_by_horizon": {str(h): float(np.mean(tf_errors[h])) for h in horizons},
        "free_running_errors_by_horizon": {str(h): float(np.mean(fr_errors[h])) for h in horizons},
    }
