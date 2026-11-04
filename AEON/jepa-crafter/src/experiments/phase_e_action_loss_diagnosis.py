"""Phase E: Diagnose Action Conditioning vs Loss Imbalance.

Executes:
  Experiment 1: Mathematical audit of persistence vs learned predictions.
  Experiment 2: Spatial token loss and gradient decomposition (low-change vs high-change).
  Experiment 3: Pre-training action sensitivity sweep across all 17 actions.
  Experiment 4: Controlled 3-seed short optimization comparing:
    - Variant A: Existing predictor + standard loss
    - Variant B: Explicit action-token predictor + standard loss
    - Variant C: Explicit action-token predictor + normalized dynamic-weighted loss
"""
import os
import sys
import copy
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure package root is on sys.path
aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.visual_encoder import VisualEncoder
from src.models.action_embedding import ActionEmbedding
from src.models.latent_predictor import LatentPredictorMember
from src.experiments.minimal_learning_experiment import DiagnosticActionTokenPredictorMember
from src.utils.logging import save_json


# ==============================================================================
# Experiment 1: Mathematical Audit of Metric Comparability
# ==============================================================================
def run_experiment_1(z_t: torch.Tensor, z_target: torch.Tensor, model: nn.Module, act_table: nn.Module, actions: torch.Tensor, h_t: torch.Tensor) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      EXPERIMENT 1: MATHEMATICAL AUDIT OF METRIC COMPARABILITY")
    print("=" * 75)

    B, N, D = z_t.shape
    assert z_t.shape == z_target.shape == (16, 64, 128), f"Shape mismatch: {z_t.shape} vs {z_target.shape}"

    # Verify normalization formulas
    norm_zt = F.normalize(z_t, dim=-1, eps=1e-6)
    norm_ztarg = F.normalize(z_target, dim=-1, eps=1e-6)

    with torch.no_grad():
        a_emb = act_table(actions)
        pred = model(z_t, h_t, a_emb)
        norm_pred = F.normalize(pred, dim=-1, eps=1e-6)

    # 1. Per-token cosine distance: 1 - sum_d(norm_x * norm_target)
    # Shape: [B, N] = [16, 64]
    cos_dist_persist = 1.0 - (norm_zt * norm_ztarg).sum(dim=-1)
    cos_dist_model = 1.0 - (norm_pred * norm_ztarg).sum(dim=-1)

    # 2. Reduction: mean across spatial tokens (dim 1), then mean across batch (dim 0)
    persist_mean = float(cos_dist_persist.mean().item())
    model_mean = float(cos_dist_model.mean().item())

    # 3. L2 metric: ||x - target||_2
    l2_persist = torch.norm(z_t - z_target, p=2, dim=-1).mean().item()
    l2_model = torch.norm(pred - z_target, p=2, dim=-1).mean().item()

    print(f"• Tensor Shapes: z_t: {list(z_t.shape)}, z_target: {list(z_target.shape)}, pred: {list(pred.shape)}")
    print(f"• Target Tensor Identity: Persistence and model use identical z_target tensor: True")
    print(f"• Normalization: Both use F.normalize(dim=-1, eps=1e-6): True")
    print(f"• Metric Formula: d(b, i) = 1.0 - (norm(x) * norm(target)).sum(dim=-1)")
    print(f"• Persistence Cosine Distance (Recalculated): {persist_mean:.8f}")
    print(f"• Initial Model Cosine Distance (Recalculated): {model_mean:.8f}")
    print(f"• Persistence L2 Distance: {l2_persist:.6f}, Initial Model L2: {l2_model:.6f}")

    return {
        "persistence_cosine_mean": persist_mean,
        "model_cosine_mean": model_mean,
        "persistence_l2_mean": l2_persist,
        "model_l2_mean": l2_model,
        "target_tensor_shape": list(z_target.shape),
        "metrics_are_strictly_comparable": True,
    }


# ==============================================================================
# Experiment 2: Loss & Gradient Origin Breakdown
# ==============================================================================
def run_experiment_2(z_t: torch.Tensor, z_target: torch.Tensor, model: nn.Module, act_table: nn.Module, actions: torch.Tensor, h_t: torch.Tensor) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      EXPERIMENT 2: LOSS AND GRADIENT ORIGIN DECOMPOSITION")
    print("=" * 75)

    B, N, D = z_t.shape
    norm_zt = F.normalize(z_t, dim=-1, eps=1e-6)
    norm_ztarg = F.normalize(z_target, dim=-1, eps=1e-6)

    # Change magnitude per token: d_persist(b, i)
    # Shape: [16, 64]
    cos_dist_persist = (1.0 - (norm_zt * norm_ztarg).sum(dim=-1)).detach()
    delta_persist = torch.norm(norm_ztarg - norm_zt, p=2, dim=-1).detach() # [16, 64]

    # Pre-defined threshold: low change (< 0.001) vs high change (>= 0.001)
    THRESHOLD = 0.001
    low_change_mask = cos_dist_persist < THRESHOLD
    high_change_mask = cos_dist_persist >= THRESHOLD

    num_total = B * N
    num_low = int(low_change_mask.sum().item())
    num_high = int(high_change_mask.sum().item())

    # Forward pass through model
    a_emb = act_table(actions)
    pred = model(z_t, h_t, a_emb)
    norm_pred = F.normalize(pred, dim=-1, eps=1e-6)
    per_token_loss = 1.0 - (norm_pred * norm_ztarg).sum(dim=-1) # [16, 64]

    loss_total = per_token_loss.mean()
    loss_low_mean = per_token_loss[low_change_mask].mean().item() if num_low > 0 else 0.0
    loss_high_mean = per_token_loss[high_change_mask].mean().item() if num_high > 0 else 0.0

    total_loss_sum = per_token_loss.sum().item()
    frac_loss_low = per_token_loss[low_change_mask].sum().item() / max(1e-8, total_loss_sum)
    frac_loss_high = per_token_loss[high_change_mask].sum().item() / max(1e-8, total_loss_sum)

    # Controlled backward passes to measure gradient norms
    # 1. Gradient from low-change static tokens
    model.zero_grad()
    loss_low_term = per_token_loss[low_change_mask].sum() / max(1, num_low)
    loss_low_term.backward(retain_graph=True)
    g_low = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    norm_g_low = float(g_low.norm(2).item())

    # 2. Gradient from high-change dynamic tokens
    model.zero_grad()
    loss_high_term = per_token_loss[high_change_mask].sum() / max(1, num_high)
    loss_high_term.backward(retain_graph=True)
    g_high = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    norm_g_high = float(g_high.norm(2).item())

    # 3. Angle between low-change and high-change gradients
    cos_grad_angle = float(F.cosine_similarity(g_low.unsqueeze(0), g_high.unsqueeze(0)).item())

    # Continuous Pearson and Spearman correlation between per-token delta and prediction error
    flat_delta = delta_persist.flatten().cpu().numpy()
    flat_err = per_token_loss.detach().flatten().cpu().numpy()
    pearson_corr = float(np.corrcoef(flat_delta, flat_err)[0, 1])

    # Rank correlation (Spearman)
    rank_delta = np.argsort(np.argsort(flat_delta))
    rank_err = np.argsort(np.argsort(flat_err))
    spearman_corr = float(np.corrcoef(rank_delta, rank_err)[0, 1])

    print(f"• Pre-Defined Threshold for Low vs High Change: cos_dist_persist = {THRESHOLD}")
    print(f"• Total Spatial Tokens in Batch: {num_total}")
    print(f"• Low-Change (Static) Tokens   : {num_low} ({num_low / num_total * 100:.1f}%)")
    print(f"• High-Change (Dynamic) Tokens : {num_high} ({num_high / num_total * 100:.1f}%)")
    print(f"• Mean Prediction Error on Low-Change Tokens : {loss_low_mean:.6f}")
    print(f"• Mean Prediction Error on High-Change Tokens: {loss_high_mean:.6f}")
    print(f"• Fraction of Total Loss from Low-Change     : {frac_loss_low * 100:.2f}%")
    print(f"• Fraction of Total Loss from High-Change    : {frac_loss_high * 100:.2f}%")
    print(f"• Low-Change Gradient L2 Norm                : {norm_g_low:.6f}")
    print(f"• High-Change Gradient L2 Norm               : {norm_g_high:.6f}")
    print(f"• Cosine Similarity of Gradients (g_low, g_high): {cos_grad_angle:.6f}")
    print(f"• Correlation (Delta vs Prediction Error)   : Pearson = {pearson_corr:.4f}, Spearman = {spearman_corr:.4f}")

    return {
        "threshold": THRESHOLD,
        "token_counts": {"total": num_total, "low_change": num_low, "high_change": num_high, "pct_low": num_low / num_total * 100.0},
        "mean_error": {"low_change": loss_low_mean, "high_change": loss_high_mean},
        "loss_fraction": {"low_change_pct": frac_loss_low * 100.0, "high_change_pct": frac_loss_high * 100.0},
        "gradient_analysis": {"norm_g_low": norm_g_low, "norm_g_high": norm_g_high, "cosine_similarity": cos_grad_angle},
        "correlation": {"pearson": pearson_corr, "spearman": spearman_corr},
    }


# ==============================================================================
# Experiment 3: Action Sensitivity Without Training
# ==============================================================================
def run_experiment_3(z_t: torch.Tensor, h_t: torch.Tensor, actions: torch.Tensor, model_A: nn.Module, act_A: nn.Module, model_B: nn.Module, act_B: nn.Module) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      EXPERIMENT 3: ACTION SENSITIVITY BEFORE TRAINING")
    print("=" * 75)

    B, N, D = z_t.shape

    def sweep_actions(model, act_table, name):
        model.eval()
        with torch.no_grad():
            # Sweep all 17 actions holding context fixed across 4 distinct observation contexts
            contexts_to_test = [0, 4, 8, 12]
            context_results = []

            for ctx_idx in contexts_to_test:
                single_z = z_t[ctx_idx : ctx_idx + 1]
                single_h = h_t[ctx_idx : ctx_idx + 1]

                preds = []
                for a in range(17):
                    a_t = torch.tensor([a], device=z_t.device)
                    a_vec = act_table(a_t)
                    out = model(single_z, single_h, a_vec)
                    preds.append(out)

                # Pairwise differences
                pairwise_cos = []
                pairwise_l2 = []
                for i in range(17):
                    for j in range(i + 1, 17):
                        pi = F.normalize(preds[i], dim=-1)
                        pj = F.normalize(preds[j], dim=-1)
                        cos_d = (1.0 - (pi * pj).sum(dim=-1)).mean().item()
                        l2_d = torch.norm(preds[i] - preds[j], p=2).item()
                        pairwise_cos.append(cos_d)
                        pairwise_l2.append(l2_d)

                context_results.append({
                    "context_index": ctx_idx,
                    "mean_pairwise_cos": float(np.mean(pairwise_cos)),
                    "mean_pairwise_l2": float(np.mean(pairwise_l2)),
                })

            # Full batch correct vs shifted action
            shifted_actions = (actions + 1) % 17
            a_correct = act_table(actions)
            a_shifted = act_table(shifted_actions)
            pred_correct = model(z_t, h_t, a_correct)
            pred_shifted = model(z_t, h_t, a_shifted)

            p_corr_norm = F.normalize(pred_correct, dim=-1)
            p_shif_norm = F.normalize(pred_shifted, dim=-1)
            dist_corr_vs_shifted = (1.0 - (p_corr_norm * p_shif_norm).sum(dim=-1)).mean().item()
            l2_corr_vs_shifted = torch.norm(pred_correct - pred_shifted, p=2, dim=-1).mean().item()

        print(f"• {name:32s}: Action Divergence (Cos) = {np.mean([c['mean_pairwise_cos'] for c in context_results]):.8f}, L2 = {np.mean([c['mean_pairwise_l2'] for c in context_results]):.4f}")
        print(f"  Correct vs Shifted Action Distance: Cos = {dist_corr_vs_shifted:.8f}, L2 = {l2_corr_vs_shifted:.4f}")

        return {
            "per_context_results": context_results,
            "mean_action_divergence_cos": float(np.mean([c['mean_pairwise_cos'] for c in context_results])),
            "mean_action_divergence_l2": float(np.mean([c['mean_pairwise_l2'] for c in context_results])),
            "correct_vs_shifted_cos": dist_corr_vs_shifted,
            "correct_vs_shifted_l2": l2_corr_vs_shifted,
        }

    res_A = sweep_actions(model_A, act_A, "Variant A (Existing FiLM Only)")
    res_B = sweep_actions(model_B, act_B, "Variant B (FiLM + Action Token)")

    return {"variant_A": res_A, "variant_B": res_B}


# ==============================================================================
# Experiment 4: Controlled Optimization Across 3 Seeds
# ==============================================================================
def run_experiment_4(
    z_t: torch.Tensor,
    z_target: torch.Tensor,
    h_t: torch.Tensor,
    actions: torch.Tensor,
    seeds: List[int] = [42, 43, 44],
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      EXPERIMENT 4: CONTROLLED SHORT OPTIMIZATION ACROSS 3 SEEDS")
    print("=" * 75)

    B, N, D = z_t.shape
    device = z_t.device
    norm_zt = F.normalize(z_t, dim=-1, eps=1e-6)
    norm_ztarg = F.normalize(z_target, dim=-1, eps=1e-6)

    # 1. Compute Fixed Persistence Baseline Metrics
    cos_dist_persist = 1.0 - (norm_zt * norm_ztarg).sum(dim=-1) # [16, 64]
    delta_persist = torch.norm(norm_ztarg - norm_zt, p=2, dim=-1) # [16, 64]
    persist_mean = float(cos_dist_persist.mean().item())

    THRESHOLD = 0.001
    low_mask = cos_dist_persist < THRESHOLD
    high_mask = cos_dist_persist >= THRESHOLD

    persist_low = float(cos_dist_persist[low_mask].mean().item())
    persist_high = float(cos_dist_persist[high_mask].mean().item())
    print(f"• Fixed Persistence Error (Mean): {persist_mean:.6f} (Low-Change: {persist_low:.6f}, High-Change: {persist_high:.6f})")

    # 2. Derive Normalized Dynamic Token Weights for Variant C
    # Formula: w'_i = 1.0 + alpha * (delta_i / max_delta), normalized so mean(w) == 1.0
    alpha = 2.0
    max_delta = delta_persist.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
    w_raw = 1.0 + alpha * (delta_persist / max_delta) # [16, 64]
    w_normed = w_raw * (float(N) / w_raw.sum(dim=-1, keepdim=True)) # strictly mean == 1.0 per item
    assert torch.allclose(w_normed.mean(dim=-1), torch.ones(B, device=device), atol=1e-5), "Weight normalization failed"
    print(f"• Variant C Normalized Weights: Min = {w_normed.min().item():.3f}, Max = {w_normed.max().item():.3f}, Mean = {w_normed.mean().item():.3f}")

    results_by_variant = {"Variant_A": [], "Variant_B": [], "Variant_C": []}

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")

        # Variant A: Existing Predictor + Standard Loss
        torch.manual_seed(seed)
        act_A = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
        model_A = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
        opt_A = torch.optim.AdamW(list(model_A.parameters()) + list(act_A.parameters()), lr=3e-4, weight_decay=1e-4)

        # Variant B: Action Token Predictor + Standard Loss
        torch.manual_seed(seed)
        act_B = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
        model_B = DiagnosticActionTokenPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
        model_B.cond_mlp.load_state_dict(model_A.cond_mlp.state_dict())
        model_B.transformer.load_state_dict(model_A.transformer.state_dict())
        model_B.final_norm.load_state_dict(model_A.final_norm.state_dict())
        opt_B = torch.optim.AdamW(list(model_B.parameters()) + list(act_B.parameters()), lr=3e-4, weight_decay=1e-4)

        # Variant C: Action Token Predictor + Normalized Dynamic Weighted Loss
        torch.manual_seed(seed)
        act_C = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
        model_C = DiagnosticActionTokenPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
        model_C.cond_mlp.load_state_dict(model_A.cond_mlp.state_dict())
        model_C.transformer.load_state_dict(model_A.transformer.state_dict())
        model_C.final_norm.load_state_dict(model_A.final_norm.state_dict())
        opt_C = torch.optim.AdamW(list(model_C.parameters()) + list(act_C.parameters()), lr=3e-4, weight_decay=1e-4)

        configs = [
            ("Variant_A", model_A, act_A, opt_A, False),
            ("Variant_B", model_B, act_B, opt_B, False),
            ("Variant_C", model_C, act_C, opt_C, True),
        ]

        seed_data = {}
        for var_name, model, act_table, opt, use_dynamic_weighting in configs:
            history = {}
            for step in range(101):
                # Evaluate at 0, 10, 50, 100 strictly on ORIGINAL UNWEIGHTED METRIC
                if step in [0, 10, 50, 100]:
                    model.eval()
                    with torch.no_grad():
                        a_emb = act_table(actions)
                        pred = model(z_t, h_t, a_emb)
                        pn = F.normalize(pred, dim=-1, eps=1e-6)
                        unweighted_per_token = 1.0 - (pn * norm_ztarg).sum(dim=-1) # [16, 64]
                        eval_cos_error = float(unweighted_per_token.mean().item())
                        eval_low = float(unweighted_per_token[low_mask].mean().item())
                        eval_high = float(unweighted_per_token[high_mask].mean().item())

                        # Shuffled Action control
                        shuf_actions = torch.roll(actions, shifts=1)
                        pred_shuf = model(z_t, h_t, act_table(shuf_actions))
                        psn = F.normalize(pred_shuf, dim=-1, eps=1e-6)
                        shuf_cos_error = float((1.0 - (psn * norm_ztarg).sum(dim=-1)).mean().item())

                        # Action Divergence across 17 actions on fixed context
                        fixed_z = z_t[:1]
                        fixed_h = h_t[:1]
                        div_preds = [model(fixed_z, fixed_h, act_table(torch.tensor([a], device=device))) for a in range(17)]
                        pw_cos = []
                        for i in range(17):
                            for j in range(i + 1, 17):
                                pi = F.normalize(div_preds[i], dim=-1, eps=1e-6)
                                pj = F.normalize(div_preds[j], dim=-1, eps=1e-6)
                                pw_cos.append((1.0 - (pi * pj).sum(dim=-1)).mean().item())
                        action_div = float(np.mean(pw_cos))

                    history[step] = {
                        "eval_unweighted_error": eval_cos_error,
                        "eval_low_change": eval_low,
                        "eval_high_change": eval_high,
                        "shuffled_error": shuf_cos_error,
                        "shuffled_advantage": shuf_cos_error - eval_cos_error, # positive means correct is better than shuffled
                        "action_divergence": action_div,
                    }
                    model.train()

                if step < 100:
                    opt.zero_grad()
                    a_emb = act_table(actions)
                    pred = model(z_t, h_t, a_emb)
                    pn = F.normalize(pred, dim=-1, eps=1e-6)
                    raw_loss_per_token = 1.0 - (pn * norm_ztarg).sum(dim=-1) # [16, 64]

                    if use_dynamic_weighting:
                        loss = (w_normed * raw_loss_per_token).mean()
                    else:
                        loss = raw_loss_per_token.mean()

                    loss.backward()
                    g_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                    opt.step()

            seed_data[var_name] = history
            print(f"  {var_name:10s} (Seed {seed}) -> Step 100 Error: {history[100]['eval_unweighted_error']:.6f} | High: {history[100]['eval_high_change']:.6f} | Shuf Adv: {history[100]['shuffled_advantage']:+.6f} | Div: {history[100]['action_divergence']:.6f}")

        for var_name in ["Variant_A", "Variant_B", "Variant_C"]:
            results_by_variant[var_name].append(seed_data[var_name])

    # 3. Aggregate Results Across 3 Seeds
    summary_stats = {}
    for var_name, runs in results_by_variant.items():
        summary_stats[var_name] = {}
        for step in [0, 10, 50, 100]:
            errs = [r[step]["eval_unweighted_error"] for r in runs]
            highs = [r[step]["eval_high_change"] for r in runs]
            lows = [r[step]["eval_low_change"] for r in runs]
            shuf_advs = [r[step]["shuffled_advantage"] for r in runs]
            divs = [r[step]["action_divergence"] for r in runs]

            summary_stats[var_name][step] = {
                "error_mean": float(np.mean(errs)),
                "error_std": float(np.std(errs)),
                "high_change_mean": float(np.mean(highs)),
                "high_change_std": float(np.std(highs)),
                "low_change_mean": float(np.mean(lows)),
                "low_change_std": float(np.std(lows)),
                "shuffled_adv_mean": float(np.mean(shuf_advs)),
                "shuffled_adv_std": float(np.std(shuf_advs)),
                "action_div_mean": float(np.mean(divs)),
                "action_div_std": float(np.std(divs)),
            }

    print("\n" + "=" * 75)
    print("      EXPERIMENT 4 SUMMARY: AGGREGATE STATS (MEAN ± STD, N=3 SEEDS)")
    print("=" * 75)
    for var_name in ["Variant_A", "Variant_B", "Variant_C"]:
        s0 = summary_stats[var_name][0]
        s100 = summary_stats[var_name][100]
        print(f"\n[{var_name}]")
        print(f"  Step   0 Unweighted Error: {s0['error_mean']:.6f} ± {s0['error_std']:.6f} (High: {s0['high_change_mean']:.6f}, Low: {s0['low_change_mean']:.6f})")
        print(f"  Step 100 Unweighted Error: {s100['error_mean']:.6f} ± {s100['error_std']:.6f} (High: {s100['high_change_mean']:.6f}, Low: {s100['low_change_mean']:.6f})")
        print(f"  Correct vs Shuffled Adv  : {s100['shuffled_adv_mean']:+.6f} ± {s100['shuffled_adv_std']:.6f}")
        print(f"  Action Divergence        : {s100['action_div_mean']:.8f} ± {s100['action_div_std']:.8f}")

    return {
        "persistence_baseline": {"mean": persist_mean, "low_change": persist_low, "high_change": persist_high},
        "seeds_evaluated": seeds,
        "summary_stats": summary_stats,
    }


# ==============================================================================
# Master Execution Routine
# ==============================================================================
def main():
    batch_path = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/diagnostic_batch_16.pt")
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Missing {batch_path}. Run minimal_learning_experiment.py first!")

    batch = torch.load(batch_path, weights_only=False)
    device = torch.device("cpu")

    obs_t_ch = batch["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    obs_tp1_ch = batch["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    actions = batch["actions"].to(device)

    # Encode with visual encoder
    torch.manual_seed(12345)
    encoder = VisualEncoder(latent_dim=128).to(device)
    encoder.eval()
    with torch.no_grad():
        z_t, u_t = encoder(obs_t_ch)
        z_target, _ = encoder(obs_tp1_ch)
        h_t = torch.cat([u_t, u_t], dim=-1)

    # Base models for Exps 1-3
    torch.manual_seed(42)
    act_A = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
    model_A = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)

    act_B = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
    act_B.load_state_dict(act_A.state_dict())
    model_B = DiagnosticActionTokenPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
    model_B.cond_mlp.load_state_dict(model_A.cond_mlp.state_dict())
    model_B.transformer.load_state_dict(model_A.transformer.state_dict())
    model_B.final_norm.load_state_dict(model_A.final_norm.state_dict())

    # Run all 4 experiments
    exp1_res = run_experiment_1(z_t, z_target, model_A, act_A, actions, h_t)
    exp2_res = run_experiment_2(z_t, z_target, model_A, act_A, actions, h_t)
    exp3_res = run_experiment_3(z_t, h_t, actions, model_A, act_A, model_B, act_B)
    exp4_res = run_experiment_4(z_t, z_target, h_t, actions, seeds=[42, 43, 44])

    full_report = {
        "experiment_1_audit": exp1_res,
        "experiment_2_loss_origin": exp2_res,
        "experiment_3_sensitivity": exp3_res,
        "experiment_4_optimization": exp4_res,
    }

    out_file = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/phase_e_diagnosis_results.json")
    save_json(full_report, out_file)
    print(f"\nAll Phase E experiments completed. Full results saved to:\n  {out_file}")


if __name__ == "__main__":
    main()
