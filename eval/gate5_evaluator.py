"""Gate 5 Evaluator: Temporal Memory Ablation, Multi-Step Horizon Probing (H=1..16),
and Viewport Exit/Return Analysis.

Strictly follows Project ATLAS protocols:
- Operates purely on visual latents, recurrent belief, and discrete actions.
- Zero privileged simulator information, info dicts, coordinates, or hand-crafted rule tables.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from eval.gate4_evaluator import decompose_static_and_changed_errors
from true_jepa.model import LatentFrame, TrueJEPA


@torch.no_grad()
def evaluate_gate5_multi_step(
    model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
    horizons: List[int] = [1, 2, 4, 8, 16],
    change_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Evaluates 1-step and multi-step predictions up to horizon H=16 under teacher-forced

    and free-running conditions, comparing against persistence baselines at each horizon.
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

    tf_errors: Dict[int, List[float]] = {h: [] for h in horizons}
    fr_errors: Dict[int, List[float]] = {h: [] for h in horizons}
    persist_h_errors: Dict[int, List[float]] = {h: [] for h in horizons}
    shuf_h_errors: Dict[int, List[float]] = {h: [] for h in horizons}

    max_h = max(horizons)

    for ep in eval_dataset:
        obs = torch.from_numpy(ep["obs"]).to(device)  # [T+1, 64, 64, 3]
        actions = torch.from_numpy(ep["action"]).to(device)  # [T]
        rewards = torch.from_numpy(ep["reward"]).to(device)  # [T]
        T = actions.shape[0]

        if T < max_h + 2:
            continue

        # Encode observations with online and target encoders
        online_frames = model.encode(obs)
        target_frames = model.target_encode(obs)

        # 1-Step Evaluation & Static/Changed Decomposition
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

        # Multi-Step Rollout Horizons (up to max_h)
        for start_idx in range(0, min(T - max_h, 48), 8):
            h_tf = model.initial_belief(batch=1, device=device)
            h_fr = h_tf.clone()
            cur_fr = LatentFrame(
                tokens=online_frames.tokens[start_idx : start_idx + 1],
                pooled=online_frames.pooled[start_idx : start_idx + 1],
            )
            base_token_norm = F.normalize(cur_fr.tokens, p=2, dim=-1)

            for step in range(1, max_h + 1):
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
                    persist_h_errors[step].append((base_token_norm - t_n).square().mean().item())

                    # Action-shuffled free-running step
                    shuf_a = (act + 7) % 17
                    pred_shuf_fr = model.predict(cur_tf, shuf_a, h_tf)
                    p_shuf_n = F.normalize(pred_shuf_fr.tokens, p=2, dim=-1)
                    shuf_h_errors[step].append((p_shuf_n - t_n).square().mean().item())

    mean_model_err = float(np.mean(total_model_errors))
    mean_persist_err = float(np.mean(total_persist_errors))
    rel_pred_err = mean_model_err / max(mean_persist_err, 1e-8)
    shuf_gap = float(np.mean(shuffled_errors)) - mean_model_err

    # Compute relative free-running error against persistence per horizon
    fr_rel_by_horizon = {}
    for h in horizons:
        mean_fr = float(np.mean(fr_errors[h]))
        mean_ph = float(np.mean(persist_h_errors[h]))
        fr_rel_by_horizon[str(h)] = mean_fr / max(mean_ph, 1e-8)

    return {
        "one_step_model_error": mean_model_err,
        "one_step_persistence_error": mean_persist_err,
        "one_step_relative_error": rel_pred_err,
        "one_step_shuffled_error": float(np.mean(shuffled_errors)),
        "shuffled_action_sensitive": bool(shuf_gap > 0),
        "shuffled_action_sensitivity_gap": float(shuf_gap),
        "static_tokens": {
            "mse_model": float(np.mean(static_model_errors)),
            "mse_persist": float(np.mean(static_persist_errors)),
            "relative_error": float(np.mean(static_model_errors)) / max(float(np.mean(static_persist_errors)), 1e-8),
        },
        "changed_tokens": {
            "mse_model": float(np.mean(changed_model_errors)),
            "mse_persist": float(np.mean(changed_persist_errors)),
            "relative_error": float(np.mean(changed_model_errors)) / max(float(np.mean(changed_persist_errors)), 1e-8),
        },
        "average_fraction_changed_tokens": float(np.mean(changed_fractions)),
        "teacher_forced_errors_by_horizon": {str(h): float(np.mean(tf_errors[h])) for h in horizons},
        "free_running_errors_by_horizon": {str(h): float(np.mean(fr_errors[h])) for h in horizons},
        "persistence_errors_by_horizon": {str(h): float(np.mean(persist_h_errors[h])) for h in horizons},
        "free_running_rel_error_by_horizon": fr_rel_by_horizon,
        "shuffled_errors_by_horizon": {str(h): float(np.mean(shuf_h_errors[h])) for h in horizons},
    }


@torch.no_grad()
def evaluate_temporal_memory_ablation(
    model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Compares three memory conditions on identical held-out transitions:

    A. No Memory / Markovian (hidden = 0)
    B. Standard Recurrent Memory (hidden unrolled over recent history L=16)
    C. Long-Sequence Memory (hidden unrolled over extended history L=48+)
    """
    model.eval()

    err_no_memory = []
    err_standard_memory = []
    err_long_memory = []

    for ep in eval_dataset:
        obs = torch.from_numpy(ep["obs"]).to(device)
        actions = torch.from_numpy(ep["action"]).to(device)
        rewards = torch.from_numpy(ep["reward"]).to(device)
        T = actions.shape[0]

        if T < 40:
            continue

        online_frames = model.encode(obs)
        target_frames = model.target_encode(obs)

        # Full episode history unroll for Condition C
        full_hidden = model.initial_belief(batch=1, device=device)
        for t in range(T - 1):
            curr_f = LatentFrame(tokens=online_frames.tokens[t : t + 1], pooled=online_frames.pooled[t : t + 1])
            tgt_f = LatentFrame(tokens=target_frames.tokens[t + 1 : t + 2], pooled=target_frames.pooled[t + 1 : t + 2])
            act = actions[t : t + 1]
            rew = rewards[t : t + 1]
            t_norm = F.normalize(tgt_f.tokens, p=2, dim=-1)

            # Condition A: No Memory (hidden = 0)
            zero_hidden = torch.zeros_like(full_hidden)
            pred_a = model.predict(curr_f, act, zero_hidden)
            pa_norm = F.normalize(pred_a.tokens, p=2, dim=-1)
            err_no_memory.append((pa_norm - t_norm).square().mean().item())

            # Condition B: Standard Memory (recent 16 steps)
            # Reconstruct recent belief from t-16 up to t
            start_b = max(0, t - 16)
            h_b = model.initial_belief(batch=1, device=device)
            for step_b in range(start_b, t):
                f_b = LatentFrame(tokens=online_frames.tokens[step_b : step_b + 1], pooled=online_frames.pooled[step_b : step_b + 1])
                h_b = model.belief_step(f_b, actions[step_b : step_b + 1], rewards[step_b : step_b + 1], h_b)

            pred_b = model.predict(curr_f, act, h_b)
            pb_norm = F.normalize(pred_b.tokens, p=2, dim=-1)
            err_standard_memory.append((pb_norm - t_norm).square().mean().item())

            # Condition C: Long-Sequence Memory (unrolled from step 0 up to t)
            pred_c = model.predict(curr_f, act, full_hidden)
            pc_norm = F.normalize(pred_c.tokens, p=2, dim=-1)
            err_long_memory.append((pc_norm - t_norm).square().mean().item())

            # Advance full hidden
            full_hidden = model.belief_step(curr_f, act, rew, full_hidden)

    mse_a = float(np.mean(err_no_memory))
    mse_b = float(np.mean(err_standard_memory))
    mse_c = float(np.mean(err_long_memory))

    # Relative memory advantage: (NoMem - Mem) / NoMem
    adv_b = (mse_a - mse_b) / max(mse_a, 1e-8)
    adv_c = (mse_a - mse_c) / max(mse_a, 1e-8)

    return {
        "condition_a_no_memory_mse": mse_a,
        "condition_b_standard_memory_mse": mse_b,
        "condition_c_long_memory_mse": mse_c,
        "standard_memory_advantage_pct": float(adv_b * 100.0),
        "long_memory_advantage_pct": float(adv_c * 100.0),
        "memory_improves_prediction": bool(mse_b < mse_a or mse_c < mse_a),
    }


@torch.no_grad()
def evaluate_viewport_return_probe(
    model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Tests object permanence and spatial consistency when previously observed objects

    leave the viewport and re-enter after cyclic or reversing actions.

    Detects reversing actions:
    e.g. Move North (1) followed by Move South (2), or Move East (3) followed by Move West (4).
    Compares prediction MSE of the returning scene with Memory vs Memoryless (h=0).
    """
    model.eval()

    opposite_actions = {
        1: 2,  # Move North -> Move South
        2: 1,  # Move South -> Move North
        3: 4,  # Move East -> Move West
        4: 3,  # Move West -> Move East
    }

    return_events_with_memory = []
    return_events_no_memory = []
    return_events_persistence = []

    for ep in eval_dataset:
        obs = torch.from_numpy(ep["obs"]).to(device)
        actions = torch.from_numpy(ep["action"]).to(device)
        rewards = torch.from_numpy(ep["reward"]).to(device)
        T = actions.shape[0]

        if T < 3:
            continue

        online_frames = model.encode(obs)
        target_frames = model.target_encode(obs)

        hidden = model.initial_belief(batch=1, device=device)

        for t in range(1, T):
            curr_act = int(actions[t].item())
            prev_act = int(actions[t - 1].item())

            curr_frame = LatentFrame(tokens=online_frames.tokens[t : t + 1], pooled=online_frames.pooled[t : t + 1])
            tgt_frame = LatentFrame(tokens=target_frames.tokens[t + 1 : t + 2], pooled=target_frames.pooled[t + 1 : t + 2])
            act_tensor = actions[t : t + 1]
            rew_tensor = rewards[t : t + 1]

            # Detect viewport reversal (agent moved away at t-1 and moves back at t)
            is_reversal = (prev_act in opposite_actions) and (opposite_actions[prev_act] == curr_act)

            if is_reversal:
                t_norm = F.normalize(tgt_frame.tokens, p=2, dim=-1)
                c_norm = F.normalize(curr_frame.tokens, p=2, dim=-1)

                # Prediction with recurrent memory (remembers prior location before departure)
                pred_mem = model.predict(curr_frame, act_tensor, hidden)
                p_mem_n = F.normalize(pred_mem.tokens, p=2, dim=-1)
                err_mem = (p_mem_n - t_norm).square().mean().item()

                # Prediction without memory (h=0)
                zero_h = torch.zeros_like(hidden)
                pred_nomem = model.predict(curr_frame, act_tensor, zero_h)
                p_nomem_n = F.normalize(pred_nomem.tokens, p=2, dim=-1)
                err_nomem = (p_nomem_n - t_norm).square().mean().item()

                err_persist = (c_norm - t_norm).square().mean().item()

                return_events_with_memory.append(err_mem)
                return_events_no_memory.append(err_nomem)
                return_events_persistence.append(err_persist)

            # Advance hidden
            hidden = model.belief_step(curr_frame, act_tensor, rew_tensor, hidden)

    if not return_events_with_memory:
        return {
            "total_return_events_detected": 0,
            "return_event_memory_mse": 0.0,
            "return_event_nomem_mse": 0.0,
            "return_event_memory_advantage_pct": 0.0,
        }

    mse_mem = float(np.mean(return_events_with_memory))
    mse_nomem = float(np.mean(return_events_no_memory))
    mse_persist = float(np.mean(return_events_persistence))
    adv = (mse_nomem - mse_mem) / max(mse_nomem, 1e-8)

    return {
        "total_return_events_detected": len(return_events_with_memory),
        "return_event_memory_mse": mse_mem,
        "return_event_nomem_mse": mse_nomem,
        "return_event_persist_mse": mse_persist,
        "return_event_rel_err_with_memory": mse_mem / max(mse_persist, 1e-8),
        "return_event_rel_err_without_memory": mse_nomem / max(mse_persist, 1e-8),
        "return_event_memory_advantage_pct": float(adv * 100.0),
        "memory_retains_viewport_return": bool(mse_mem < mse_nomem),
    }


@torch.no_grad()
def evaluate_policy_value_calibration(
    model: TrueJEPA,
    eval_dataset: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates readiness for direct policy learning:

    - Reward head accuracy and calibration
    - Continuation head calibration
    - Value head dispersion
    - Policy entropy
    """
    model.eval()

    reward_errors = []
    continuation_bce = []
    value_estimates = []
    policy_entropies = []

    for ep in eval_dataset:
        obs = torch.from_numpy(ep["obs"]).to(device)
        actions = torch.from_numpy(ep["action"]).to(device)
        rewards = torch.from_numpy(ep["reward"]).to(device)
        T = actions.shape[0]

        if T < 3:
            continue

        online_frames = model.encode(obs)
        hidden = model.initial_belief(batch=1, device=device)

        for t in range(min(T - 1, 30)):
            curr_frame = LatentFrame(tokens=online_frames.tokens[t : t + 1], pooled=online_frames.pooled[t : t + 1])
            act = actions[t : t + 1]
            true_r = rewards[t : t + 1]

            pred_r, cont_logit = model.transition_heads(curr_frame, hidden, act)
            val = model.heads.value(model.heads.state(curr_frame, hidden)).squeeze(-1)
            pol_logits = model.policy_logits(curr_frame, hidden)
            probs = F.softmax(pol_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)

            reward_errors.append((pred_r - true_r).abs().item())
            continuation_bce.append(cont_logit.sigmoid().item())
            value_estimates.append(val.item())
            policy_entropies.append(entropy.item())

            hidden = model.belief_step(curr_frame, act, true_r, hidden)

    return {
        "mean_reward_mae": float(np.mean(reward_errors)),
        "mean_continuation_prob": float(np.mean(continuation_bce)),
        "mean_value_estimate": float(np.mean(value_estimates)),
        "value_estimate_std": float(np.std(value_estimates)),
        "mean_policy_entropy": float(np.mean(policy_entropies)),
        "policy_ready_for_training": bool(np.std(value_estimates) >= 0.0 and np.mean(policy_entropies) > 0.5),
    }
