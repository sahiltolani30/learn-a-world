#!/usr/bin/env python3
"""
scripts/run_empirical_probe_audit.py

Comprehensive Empirical Counter-Verification & Probe Runner for Prometheus & SAPNA JEPA World Models.
Non-destructively probes checkpoint weights, representation dynamics, multi-step predictor rollouts,
loss telemetry, and action sensitivity.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import crafter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "prometheus") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "prometheus"))

from sapna.encoder import ObservationEncoder as SapnaEncoder, TargetEncoder as SapnaTargetEncoder
from sapna.dynamics import SpatialPredictor as SapnaPredictor, TemporalCore as SapnaTemporalCore
from sapna.policy import Actor as SapnaActor, Critic as SapnaCritic
from src.diagnostics import CRAFTER_ACTION_NAMES, ALL_CRAFTER_ACHIEVEMENTS

from src.agent.agent import JEPAAgent as PromAgent


def get_canonical_crafter_scenes() -> Dict[str, np.ndarray]:
    """Generates 6 canonical Crafter visual scenes using controlled game state."""
    scenes = {}
    
    # 1. Spawn Scene (Daylight grass/tree/water)
    env = crafter.Env(seed=42)
    obs_spawn = env.reset()
    scenes["1_spawn_default"] = obs_spawn
    
    # 2. Facing Tree (Harvest affordance)
    env.reset()
    px, py = env._player.pos
    env._world[px + 1, py] = "tree"
    env._player.facing = (1, 0)
    scenes["2_facing_tree"] = env.render()
    
    # 3. Facing Grass with Wood=2 (Crafting table affordance)
    env.reset()
    px, py = env._player.pos
    env._world[px + 1, py] = "grass"
    env._player.facing = (1, 0)
    env._player.inventory["wood"] = 2
    scenes["3_facing_grass_wood2"] = env.render()
    
    # 4. Facing Table (Tool craft affordance)
    env.reset()
    px, py = env._player.pos
    env._world[px + 1, py] = "table"
    env._player.facing = (1, 0)
    env._player.inventory["wood"] = 1
    scenes["4_facing_table"] = env.render()
    
    # 5. Facing Stone (Mining affordance)
    env.reset()
    px, py = env._player.pos
    env._world[px + 1, py] = "stone"
    env._player.facing = (1, 0)
    env._player.inventory["wood_pickaxe"] = 1
    scenes["5_facing_stone_pickaxe"] = env.render()
    
    # 6. Night Scene (Low illuminance)
    env.reset()
    env._step = 150  # Advance to night
    scenes["6_night_low_light"] = env.render()
    
    return scenes


def audit_weight_integrity(state_dict: Dict[str, torch.Tensor], module_name: str) -> Dict[str, Any]:
    """Checks for NaNs, Infs, zero norms, and computes weight statistics."""
    total_params = 0
    total_l2_sq = 0.0
    nan_count = 0
    inf_count = 0
    zero_param_tensors = 0
    layer_stats = {}

    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.is_floating_point():
            n_elems = v.numel()
            total_params += n_elems
            l2_val = v.norm().item()
            total_l2_sq += l2_val ** 2
            
            n_nan = torch.isnan(v).sum().item()
            n_inf = torch.isinf(v).sum().item()
            nan_count += n_nan
            inf_count += n_inf
            if l2_val == 0.0:
                zero_param_tensors += 1
                
            layer_stats[k] = {
                "shape": list(v.shape),
                "l2_norm": round(l2_val, 4),
                "mean": round(v.mean().item(), 6),
                "std": round(v.std().item(), 6),
            }

    return {
        "module": module_name,
        "total_float_params": total_params,
        "total_l2_norm": round(np.sqrt(total_l2_sq), 4),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_param_tensors": zero_param_tensors,
        "is_healthy": (nan_count == 0 and inf_count == 0 and total_params > 0),
        "layers": layer_stats,
    }


def compute_effective_rank(matrix: torch.Tensor) -> float:
    """Computes participation ratio / effective rank of a 2D matrix via SVD."""
    if matrix.dim() != 2:
        matrix = matrix.flatten(1)
    # SVD
    s = torch.linalg.svdvals(matrix.float())
    s_sq = s ** 2
    sum_s_sq = s_sq.sum().item()
    if sum_s_sq == 0.0:
        return 0.0
    pr = (s.sum().item() ** 2) / sum_s_sq
    return float(pr)


def run_representation_dynamics_audit(
    encoder: SapnaEncoder,
    scenes: Dict[str, np.ndarray],
    device: torch.device
) -> Dict[str, Any]:
    """Audits spatial token variance, dimension variance, effective rank, and scene cosine similarities."""
    encoder.eval()
    scene_names = list(scenes.keys())
    tensors = []
    for name in scene_names:
        img = scenes[name]
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensors.append(t)
    batch_scenes = torch.cat(tensors, dim=0).to(device)  # (N, 3, 64, 64)

    with torch.no_grad():
        tokens, pooled = encoder.forward_tokens(batch_scenes)  # (N, 64, 64), (N, 256)
        recons = encoder.decode_tokens(tokens)                 # (N, 3, 64, 64)

    N = len(scene_names)
    
    # 1. Per-scene spatial token metrics
    per_scene_metrics = {}
    for i, name in enumerate(scene_names):
        tok_i = tokens[i]  # (64, 64) - 64 spatial positions, 64 feature channels
        
        # Spatial token variance across the 64 positions:
        spatial_var = tok_i.var(dim=0).mean().item()  # mean variance of each feature across positions
        spatial_std = tok_i.std(dim=0).mean().item()
        
        # Feature dimension variance across the 64 channels:
        channel_var = tok_i.var(dim=1).mean().item()  # mean variance of each position across features
        channel_std = tok_i.std(dim=1).mean().item()
        
        # Effective rank / participation ratio of 64x64 token matrix
        eff_rank = compute_effective_rank(tok_i)
        
        # Reconstruction MSE
        recon_mse = F.mse_loss(recons[i], batch_scenes[i]).item()
        
        # Norms
        token_norms = tok_i.norm(dim=-1)  # (64,)
        
        per_scene_metrics[name] = {
            "spatial_var_across_positions": round(spatial_var, 6),
            "spatial_std_across_positions": round(spatial_std, 6),
            "channel_var_across_features": round(channel_var, 6),
            "channel_std_across_features": round(channel_std, 6),
            "effective_rank_64": round(eff_rank, 2),
            "reconstruction_mse": round(recon_mse, 6),
            "token_norm_mean": round(token_norms.mean().item(), 4),
            "token_norm_min": round(token_norms.min().item(), 4),
            "token_norm_max": round(token_norms.max().item(), 4),
            "pooled_norm": round(pooled[i].norm().item(), 4),
        }

    # 2. Pairwise Cosine Similarity Matrices
    def cosine_mat(vecs: torch.Tensor) -> np.ndarray:
        normed = F.normalize(vecs, dim=-1)
        sim = torch.mm(normed, normed.t()).cpu().numpy()
        return sim

    # Token grid cosines (flatten 64*64 = 4096)
    tokens_flat = tokens.flatten(1)  # (N, 4096)
    cos_tokens = cosine_mat(tokens_flat)
    
    # Pooled 256 cosines
    cos_pooled = cosine_mat(pooled)

    pairwise_table = []
    for i in range(N):
        for j in range(i + 1, N):
            pairwise_table.append({
                "scene_a": scene_names[i],
                "scene_b": scene_names[j],
                "token_grid_cosine": round(float(cos_tokens[i, j]), 4),
                "pooled_cosine": round(float(cos_pooled[i, j]), 4),
                "token_l2_dist": round(float((tokens_flat[i] - tokens_flat[j]).norm().item()), 4),
                "pooled_l2_dist": round(float((pooled[i] - pooled[j]).norm().item()), 4),
            })

    # Summary statistics
    all_tok_cos = [p["token_grid_cosine"] for p in pairwise_table]
    all_pool_cos = [p["pooled_cosine"] for p in pairwise_table]

    return {
        "per_scene": per_scene_metrics,
        "pairwise": pairwise_table,
        "summary": {
            "mean_token_grid_cosine": round(float(np.mean(all_tok_cos)), 4),
            "max_token_grid_cosine": round(float(np.max(all_tok_cos)), 4),
            "min_token_grid_cosine": round(float(np.min(all_tok_cos)), 4),
            "mean_pooled_cosine": round(float(np.mean(all_pool_cos)), 4),
            "max_pooled_cosine": round(float(np.max(all_pool_cos)), 4),
            "min_pooled_cosine": round(float(np.min(all_pool_cos)), 4),
            "spatial_collapse_detected": bool(np.mean([m["spatial_var_across_positions"] for m in per_scene_metrics.values()]) < 0.01),
            "scene_collapse_detected": bool(np.mean(all_tok_cos) > 0.95),
        }
    }


def run_predictor_rollout_audit(
    encoder: SapnaEncoder,
    target_encoder: SapnaTargetEncoder,
    predictor: SapnaPredictor,
    temporal_core: SapnaTemporalCore,
    device: torch.device,
    num_rollouts: int = 15,
    horizon: int = 5
) -> Dict[str, Any]:
    """
    Executes real Crafter rollouts across horizons t=1..5 and compares autoregressively predicted
    tokens with true target tokens.
    """
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    temporal_core.eval()

    rollout_horizon_stats = {t: {"token_cos": [], "pooled_cos": [], "l2_err": [], "norm_drift": [], "disagreement": []} for t in range(1, horizon + 1)}

    for ep in range(num_rollouts):
        env = crafter.Env(seed=100 + ep)
        obs_curr = env.reset()
        
        # Initial encoding
        x_0 = torch.from_numpy(obs_curr).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            curr_tokens, curr_pooled = encoder.forward_tokens(x_0)
            initial_norm = curr_pooled.norm().item()
            h = torch.zeros(1, 256, device=device)

        # Autoregressive rollout
        pred_tokens = curr_tokens
        pred_h = h

        for t in range(1, horizon + 1):
            # Select action
            action = int(np.random.choice([1, 2, 3, 4, 5, 6]))  # Movement and action
            a_tensor = torch.tensor([action], dtype=torch.long, device=device)
            
            # Step real env to get true ground truth next observation
            obs_next, reward, done, info = env.step(action)
            x_next = torch.from_numpy(obs_next).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            
            with torch.no_grad():
                # True target representation from TargetEncoder
                true_target_tokens, true_target_pooled = target_encoder.forward_tokens(x_next)
                
                # Autoregressive prediction step
                mean_pred_tokens, all_pred_tokens = predictor.forward_tokens(pred_tokens, a_tensor, pred_h)
                pred_pooled = encoder.pool_tokens(mean_pred_tokens)
                a_emb = predictor.action_embedding(a_tensor)
                pred_h = temporal_core(pred_pooled, pred_h, a_emb=a_emb)
                
                # Metrics at horizon t
                tok_cos = F.cosine_similarity(mean_pred_tokens.flatten(1), true_target_tokens.flatten(1)).item()
                pool_cos = F.cosine_similarity(pred_pooled, true_target_pooled).item()
                l2_diff = (pred_pooled - true_target_pooled).norm().item()
                norm_drift = pred_pooled.norm().item() / max(initial_norm, 1e-6)
                
                # Epistemic disagreement across heads
                head_preds_flat = all_pred_tokens.flatten(2)  # (4, 1, 4096)
                disagree = head_preds_flat.var(dim=0).mean().item()
                
            rollout_horizon_stats[t]["token_cos"].append(tok_cos)
            rollout_horizon_stats[t]["pooled_cos"].append(pool_cos)
            rollout_horizon_stats[t]["l2_err"].append(l2_diff)
            rollout_horizon_stats[t]["norm_drift"].append(norm_drift)
            rollout_horizon_stats[t]["disagreement"].append(disagree)
            
            pred_tokens = mean_pred_tokens
            if done:
                break

    horizon_table = []
    for t in range(1, horizon + 1):
        horizon_table.append({
            "horizon_step": t,
            "mean_token_cosine": round(float(np.mean(rollout_horizon_stats[t]["token_cos"])), 4),
            "std_token_cosine": round(float(np.std(rollout_horizon_stats[t]["token_cos"])), 4),
            "mean_pooled_cosine": round(float(np.mean(rollout_horizon_stats[t]["pooled_cos"])), 4),
            "std_pooled_cosine": round(float(np.std(rollout_horizon_stats[t]["pooled_cos"])), 4),
            "mean_l2_error": round(float(np.mean(rollout_horizon_stats[t]["l2_err"])), 4),
            "mean_norm_drift": round(float(np.mean(rollout_horizon_stats[t]["norm_drift"])), 4),
            "mean_ensemble_disagreement": round(float(np.mean(rollout_horizon_stats[t]["disagreement"])), 6),
        })

    return {
        "num_rollouts": num_rollouts,
        "horizons": horizon_table,
    }


def run_action_sensitivity_audit(
    encoder: SapnaEncoder,
    predictor: SapnaPredictor,
    device: torch.device,
    scene_img: np.ndarray
) -> Dict[str, Any]:
    """Audits action sensitivity across all 17 discrete Crafter actions on a fixed scene."""
    encoder.eval()
    predictor.eval()

    x = torch.from_numpy(scene_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        tokens, pooled = encoder.forward_tokens(x)
        tokens_17 = tokens.repeat(17, 1, 1)  # (17, 64, 64)
        actions = torch.arange(17, dtype=torch.long, device=device)
        h_zero = torch.zeros(17, 256, device=device)
        
        mean_preds, all_preds = predictor.forward_tokens(tokens_17, actions, h_zero)  # (17, 64, 64), (4, 17, 64, 64)
        pred_pooled = encoder.pool_tokens(mean_preds)  # (17, 256)
        
        # Compute epistemic disagreement per action
        head_preds_flat = all_preds.permute(1, 0, 2, 3).flatten(2)  # (17, 4, 4096)
        disagreements = head_preds_flat.var(dim=1).mean(dim=-1).cpu().numpy()  # (17,)

    # Pairwise distance between predicted tokens across action pairs
    preds_flat = mean_preds.flatten(1)  # (17, 4096)
    pairwise_dists = torch.cdist(preds_flat, preds_flat, p=2).cpu().numpy()
    
    # Distance from input tokens (persistence baseline)
    in_tokens_flat = tokens.flatten(1)  # (1, 4096)
    delta_from_input = (preds_flat - in_tokens_flat).norm(dim=-1).cpu().numpy()  # (17,)

    per_action_stats = []
    for a in range(17):
        name = CRAFTER_ACTION_NAMES[a] if a < len(CRAFTER_ACTION_NAMES) else str(a)
        per_action_stats.append({
            "action_idx": a,
            "action_name": name,
            "delta_from_curr_tokens": round(float(delta_from_input[a]), 6),
            "epistemic_disagreement": round(float(disagreements[a]), 8),
        })

    # Non-diagonal pairwise distances
    mask_off_diag = ~np.eye(17, dtype=bool)
    off_diag_dists = pairwise_dists[mask_off_diag]

    return {
        "per_action": per_action_stats,
        "summary": {
            "min_action_separation": round(float(off_diag_dists.min()), 6),
            "mean_action_separation": round(float(off_diag_dists.mean()), 6),
            "max_action_separation": round(float(off_diag_dists.max()), 6),
            "min_delta_from_input": round(float(delta_from_input.min()), 6),
            "max_delta_from_input": round(float(delta_from_input.max()), 6),
            "disagreement_min": round(float(disagreements.min()), 8),
            "disagreement_max": round(float(disagreements.max()), 8),
            "disagreement_spread": round(float(disagreements.max() - disagreements.min()), 8),
            "disagreement_ratio_max_min": round(float(disagreements.max() / max(disagreements.min(), 1e-12)), 2),
            "action_neglect_detected": bool(off_diag_dists.mean() < 0.01),
        }
    }


def analyze_training_telemetry(checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes the 137 telemetry history snapshots recorded during the 25k run."""
    telem = checkpoint_data.get("telemetry_history", [])
    if not telem:
        return {"error": "No telemetry history found"}

    steps = [e["step"] for e in telem]
    wm_losses = [e.get("wm_loss", 0.0) for e in telem]
    l_tokens = [e.get("L_tokens", 0.0) for e in telem]
    l_pool = [e.get("L_pool", 0.0) for e in telem]
    var_hinges = [e.get("var_hinge", 0.0) for e in telem]
    cov_terms = [e.get("cov_term", 0.0) for e in telem]
    critic_losses = [e.get("critic_loss", 0.0) for e in telem]
    actor_losses = [e.get("actor_loss", 0.0) for e in telem]
    entropies = [e.get("entropy_mean", 0.0) for e in telem]
    q_spreads = [e.get("q_spread_equivalent", 0.0) for e in telem]
    adv_stds = [e.get("advantage_std", 0.0) for e in telem]
    imag_rewards = [e.get("imagined_reward_mean", 0.0) for e in telem]
    real_rewards = [e.get("real_reward_mean", 0.0) for e in telem]
    milestone_fracs = [e.get("milestone_fraction", 0.0) for e in telem]

    def curve_summary(vals: List[float]) -> Dict[str, Any]:
        return {
            "initial": round(float(vals[0]), 6),
            "final": round(float(vals[-1]), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
            "mean": round(float(np.mean(vals)), 6),
            "delta_final_minus_initial": round(float(vals[-1] - vals[0]), 6),
            "relative_change_pct": round(float((vals[-1] - vals[0]) / max(abs(vals[0]), 1e-6) * 100.0), 2),
        }

    # Selected progression points (start, 25%, 50%, 75%, end)
    indices = [0, len(telem) // 4, len(telem) // 2, (3 * len(telem)) // 4, -1]
    trajectory_table = []
    for idx in indices:
        e = telem[idx]
        trajectory_table.append({
            "step": e["step"],
            "wm_loss": round(float(e.get("wm_loss", 0)), 4),
            "L_tokens": round(float(e.get("L_tokens", 0)), 4),
            "L_pool": round(float(e.get("L_pool", 0)), 4),
            "var_hinge": round(float(e.get("var_hinge", 0)), 4),
            "cov_term": round(float(e.get("cov_term", 0)), 4),
            "critic_loss": round(float(e.get("critic_loss", 0)), 4),
            "actor_loss": round(float(e.get("actor_loss", 0)), 4),
            "entropy": round(float(e.get("entropy_mean", 0)), 4),
            "q_spread": round(float(e.get("q_spread_equivalent", 0)), 4),
            "advantage_std": round(float(e.get("advantage_std", 0)), 4),
            "milestone_fraction": round(float(e.get("milestone_fraction", 0)), 4),
        })

    return {
        "num_telemetry_points": len(telem),
        "step_range": [steps[0], steps[-1]],
        "metrics_summary": {
            "wm_loss": curve_summary(wm_losses),
            "L_tokens": curve_summary(l_tokens),
            "L_pool": curve_summary(l_pool),
            "var_hinge": curve_summary(var_hinges),
            "cov_term": curve_summary(cov_terms),
            "critic_loss": curve_summary(critic_losses),
            "actor_loss": curve_summary(actor_losses),
            "entropy_mean": curve_summary(entropies),
            "q_spread_equivalent": curve_summary(q_spreads),
            "advantage_std": curve_summary(adv_stds),
            "imagined_reward_mean": curve_summary(imag_rewards),
            "real_reward_mean": curve_summary(real_rewards),
            "milestone_fraction": curve_summary(milestone_fracs),
        },
        "trajectory_table": trajectory_table,
    }


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("  EMPIRICAL COUNTER-VERIFICATION & PROBE RUNNER")
    print("=" * 80)

    # 1. Checkpoint Loading
    ckpt_path_sapna = REPO_ROOT / "learn-a-world/sapna/checkpoints_v1/state_step_25008.pt"
    ckpt_path_prom0 = REPO_ROOT / "checkpoints/checkpoint_step_0.pt"
    ckpt_path_prom_state = REPO_ROOT / "checkpoints/state.pt"

    print(f"Loading SAPNA 25k Checkpoint: {ckpt_path_sapna}")
    ck_sapna = torch.load(str(ckpt_path_sapna), map_location=device, weights_only=False)

    # Instantiate SAPNA modules
    sapna_enc = SapnaEncoder().to(device)
    sapna_target_enc = SapnaTargetEncoder(sapna_enc).to(device)
    sapna_pred = SapnaPredictor().to(device)
    sapna_core = SapnaTemporalCore().to(device)
    sapna_actor = SapnaActor().to(device)
    sapna_critic = SapnaCritic().to(device)

    sapna_enc.load_state_dict(ck_sapna["encoder"])
    sapna_target_enc.load_state_dict(ck_sapna["target_encoder"])
    sapna_pred.load_state_dict(ck_sapna["predictor"])
    sapna_core.load_state_dict(ck_sapna["temporal_core"])
    sapna_actor.load_state_dict(ck_sapna["actor"])
    sapna_critic.load_state_dict(ck_sapna["critic"])

    print("All SAPNA modules successfully loaded!")

    # 2. Weight Integrity Audit
    print("\n[+] Executing Weight Integrity Audit...")
    weight_audit = {
        "encoder": audit_weight_integrity(ck_sapna["encoder"], "encoder"),
        "target_encoder": audit_weight_integrity(ck_sapna["target_encoder"], "target_encoder"),
        "predictor": audit_weight_integrity(ck_sapna["predictor"], "predictor"),
        "temporal_core": audit_weight_integrity(ck_sapna["temporal_core"], "temporal_core"),
        "actor": audit_weight_integrity(ck_sapna["actor"], "actor"),
        "critic": audit_weight_integrity(ck_sapna["critic"], "critic"),
    }

    # 3. Canonical Scenes & Representation Dynamics Audit
    print("\n[+] Generating Canonical Crafter Scenes...")
    scenes = get_canonical_crafter_scenes()
    print(f"Generated {len(scenes)} canonical scenes: {list(scenes.keys())}")

    print("\n[+] Executing Representation Dynamics Audit...")
    rep_audit = run_representation_dynamics_audit(sapna_enc, scenes, device)

    # 4. Predictor Multi-Step Autoregressive Rollout Audit (Horizons t=1..5)
    print("\n[+] Executing Multi-Step Predictor Rollouts (t=1..5)...")
    rollout_audit = run_predictor_rollout_audit(
        encoder=sapna_enc,
        target_encoder=sapna_target_enc,
        predictor=sapna_pred,
        temporal_core=sapna_core,
        device=device,
        num_rollouts=15,
        horizon=5,
    )

    # 5. Action Sensitivity Audit
    print("\n[+] Executing Action Sensitivity Audit...")
    action_audit = run_action_sensitivity_audit(
        encoder=sapna_enc,
        predictor=sapna_pred,
        device=device,
        scene_img=scenes["3_facing_grass_wood2"],
    )

    # 6. Training Telemetry Analysis
    print("\n[+] Analyzing 25k Training Telemetry History...")
    telemetry_audit = analyze_training_telemetry(ck_sapna)

    # 7. Official Counts & Buffer Stats
    official_counts = ck_sapna.get("official_counts", {})
    total_episodes = ck_sapna.get("total_episodes", 0)
    buffer_data = ck_sapna.get("buffer", None)
    buffer_count = len(buffer_data.get("obs", [])) if isinstance(buffer_data, dict) else "unknown"

    all_metrics = {
        "checkpoint_info": {
            "path": str(ckpt_path_sapna),
            "step": ck_sapna.get("step"),
            "arch": ck_sapna.get("arch"),
            "config": ck_sapna.get("config"),
            "total_episodes": total_episodes,
            "buffer_transition_count": buffer_count,
            "official_achievement_counts": official_counts,
        },
        "weight_integrity": weight_audit,
        "representation_dynamics": rep_audit,
        "predictor_rollout": rollout_audit,
        "action_sensitivity": action_audit,
        "telemetry_dynamics": telemetry_audit,
    }

    # Save output to results/diagnostics/empirical_probe_metrics.json
    out_dir = REPO_ROOT / "results/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "empirical_probe_metrics.json"
    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[+] Empirical metrics successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
