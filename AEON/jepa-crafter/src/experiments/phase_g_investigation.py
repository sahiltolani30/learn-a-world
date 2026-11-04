"""AEON Phase G: Visual Representation Root-Cause Investigation.

Comprehensive empirical audit covering:
- G1: Encoder Geometry Audit (shapes, normalization, centered vs uncentered effective rank, variance spectrum, pairwise cosine similarities, visual distinguishability)
- G2: Latent Representation Prediction Support Test (Persistence vs Simple Residual Predictor vs Existing Predictor)
- G3: Target and Loss Pipeline Audit (normalization asymmetry analysis, gradient flow, detachment, EMA schedule)
- G4: Isolated Encoder-Training Test (Frozen Encoder vs Trainable Encoder vs Trainable + Variance Regularizer)
"""
import os
import sys
import copy
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

aeon_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if aeon_root not in sys.path:
    sys.path.insert(0, aeon_root)

from src.models.visual_encoder import VisualEncoder
from src.models.action_embedding import ActionEmbedding
from src.models.latent_predictor import LatentPredictorMember
from src.models.ema_encoder import EmaTargetEncoder
from src.utils.logging import save_json


# ==============================================================================
# Helper Functions: Effective Rank & Geometry
# ==============================================================================
def compute_rank_and_spectrum(features: torch.Tensor) -> Dict[str, Any]:
    """Compute centered and uncentered participation ratio, entropy ER, and spectrum.
    
    Args:
        features: [M, D] tensor of feature vectors
    Returns:
        dict of metrics
    """
    M, D = features.shape
    features_float = features.float()

    # 1. Centered Covariance
    mean_vec = features_float.mean(dim=0, keepdim=True)
    centered = features_float - mean_vec
    cov_c = torch.mm(centered.t(), centered) / max(1, M - 1)
    ev_c = torch.linalg.eigvalsh(cov_c).clamp(min=0.0)
    
    tr_c = ev_c.sum().item()
    sum_sq_c = (ev_c ** 2).sum().item()
    pr_c = (tr_c ** 2 / max(1e-12, sum_sq_c)) if sum_sq_c > 1e-12 else 0.0
    
    p_c = ev_c / max(1e-12, tr_c)
    # Filter zeros for entropy
    p_c_pos = p_c[p_c > 1e-12]
    ent_c = float(torch.exp(-torch.sum(p_c_pos * torch.log(p_c_pos))).item())

    # 2. Uncentered Covariance (Second Moment Matrix)
    cov_u = torch.mm(features_float.t(), features_float) / max(1, M)
    ev_u = torch.linalg.eigvalsh(cov_u).clamp(min=0.0)

    tr_u = ev_u.sum().item()
    sum_sq_u = (ev_u ** 2).sum().item()
    pr_u = (tr_u ** 2 / max(1e-12, sum_sq_u)) if sum_sq_u > 1e-12 else 0.0

    p_u = ev_u / max(1e-12, tr_u)
    p_u_pos = p_u[p_u > 1e-12]
    ent_u = float(torch.exp(-torch.sum(p_u_pos * torch.log(p_u_pos))).item())

    # Variance Ratios
    sorted_ev_c, _ = torch.sort(ev_c, descending=True)
    sorted_ev_u, _ = torch.sort(ev_u, descending=True)

    return {
        "num_samples": M,
        "feature_dim": D,
        "centered": {
            "participation_ratio": pr_c,
            "entropy_effective_rank": ent_c,
            "top1_var_pct": float(sorted_ev_c[0].item() / max(1e-12, tr_c) * 100.0),
            "top5_var_pct": float(sorted_ev_c[:5].sum().item() / max(1e-12, tr_c) * 100.0),
            "top10_var_pct": float(sorted_ev_c[:10].sum().item() / max(1e-12, tr_c) * 100.0),
            "top20_var_pct": float(sorted_ev_c[:20].sum().item() / max(1e-12, tr_c) * 100.0),
            "trace": tr_c,
        },
        "uncentered": {
            "participation_ratio": pr_u,
            "entropy_effective_rank": ent_u,
            "top1_var_pct": float(sorted_ev_u[0].item() / max(1e-12, tr_u) * 100.0),
            "top5_var_pct": float(sorted_ev_u[:5].sum().item() / max(1e-12, tr_u) * 100.0),
            "top10_var_pct": float(sorted_ev_u[:10].sum().item() / max(1e-12, tr_u) * 100.0),
            "top20_var_pct": float(sorted_ev_u[:20].sum().item() / max(1e-12, tr_u) * 100.0),
            "trace": tr_u,
        },
        "channel_std": {
            "min": float(centered.std(dim=0).min().item()),
            "max": float(centered.std(dim=0).max().item()),
            "mean": float(centered.std(dim=0).mean().item()),
            "num_collapsed_under_0_01": int((centered.std(dim=0) < 0.01).sum().item()),
        },
        "feature_norms": {
            "mean": float(torch.norm(features_float, dim=-1).mean().item()),
            "std": float(torch.norm(features_float, dim=-1).std().item()),
            "min": float(torch.norm(features_float, dim=-1).min().item()),
            "max": float(torch.norm(features_float, dim=-1).max().item()),
        },
        "feature_mean_norm": float(torch.norm(mean_vec).item()),
    }


def compute_pairwise_cosine_stats(features: torch.Tensor, max_samples: int = 500) -> Dict[str, float]:
    """Compute pairwise cosine similarities before and after subtracting mean vector."""
    if features.size(0) > max_samples:
        indices = torch.randperm(features.size(0))[:max_samples]
        sub = features[indices].float()
    else:
        sub = features.float()

    N = sub.size(0)
    mask = ~torch.eye(N, dtype=bool, device=sub.device)

    # 1. Uncentered
    u_norm = F.normalize(sub, dim=-1, eps=1e-6)
    sim_u = torch.mm(u_norm, u_norm.t())[mask]

    # 2. Centered (subtract batch mean)
    mean_vec = sub.mean(dim=0, keepdim=True)
    c_sub = sub - mean_vec
    c_norm = F.normalize(c_sub, dim=-1, eps=1e-6)
    sim_c = torch.mm(c_norm, c_norm.t())[mask]

    return {
        "uncentered_mean": float(sim_u.mean().item()),
        "uncentered_std": float(sim_u.std().item()),
        "uncentered_min": float(sim_u.min().item()),
        "uncentered_max": float(sim_u.max().item()),
        "centered_mean": float(sim_c.mean().item()),
        "centered_std": float(sim_c.std().item()),
        "centered_min": float(sim_c.min().item()),
        "centered_max": float(sim_c.max().item()),
    }


# ==============================================================================
# Model Definition: Simple Residual Predictor
# ==============================================================================
class SimpleResidualPredictor(nn.Module):
    """Diagnostic residual predictor: z_hat = z_t + delta_theta(z_t, a_emb)."""

    def __init__(self, latent_dim: int = 128, action_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        # Zero-initialize the final layer so f_0(z, a) = z (exact persistence at init)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        B, N, D = z_t.shape
        a_expanded = a_emb.unsqueeze(1).expand(B, N, -1)  # [B, N, action_dim]
        cat_in = torch.cat([z_t, a_expanded], dim=-1)     # [B, N, D + action_dim]
        delta = self.mlp(cat_in)                           # [B, N, D]
        return z_t + delta


# ==============================================================================
# Step G1: Encoder Geometry Audit
# ==============================================================================
def run_g1_audit(dataset: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      AEON PHASE G1: ENCODER GEOMETRY AUDIT")
    print("=" * 75)

    device = torch.device("cpu")
    torch.manual_seed(12345)
    encoder = VisualEncoder(latent_dim=128).to(device)
    encoder.eval()

    train_obs = dataset["train"]["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    heldout_obs = dataset["heldout"]["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)

    with torch.no_grad():
        tr_tokens, tr_pool = encoder(train_obs)
        ho_tokens, ho_pool = encoder(heldout_obs)

    # 1. Architecture details
    arch_info = {
        "input_shape": [3, 64, 64],
        "stem_shape": [64, 32, 32],
        "block1_shape": [64, 32, 32],
        "down1_shape": [96, 16, 16],
        "block2_shape": [96, 16, 16],
        "down2_shape": [128, 8, 8],
        "block3_shape": [128, 8, 8],
        "spatial_grid": [8, 8],
        "spatial_tokens_count": 64,
        "feature_dim": 128,
        "normalization_stages": {
            "stem": "GroupNorm(8, 64) -> GELU",
            "block1": "2x (Conv3x3 -> GroupNorm(8, 64) -> GELU) + residual",
            "down1": "GroupNorm(8, 96) -> GELU",
            "block2": "2x (Conv3x3 -> GroupNorm(8, 96) -> GELU) + residual",
            "down2": "GroupNorm(8, 128) -> GELU",
            "block3": "2x (Conv3x3 -> GroupNorm(8, 128) -> GELU) + residual",
            "output_tokens": "NONE (raw conv activations + pos_embed, NO LayerNorm or L2Norm)",
            "attention_pool": "Softmax attention over 64 tokens with learned query (no LayerNorm)",
        },
        "online_vs_target_sharing": "In JepaWorldModel: target is EmaTargetEncoder (deepcopy, momentum 0.996->0.999, stop-grad). In Phase F experiments: frozen encoder with identical weights.",
    }

    # 2. Effective rank calculations
    tr_spatial_flat = tr_tokens.reshape(-1, 128)
    ho_spatial_flat = ho_tokens.reshape(-1, 128)

    rank_train_spatial = compute_rank_and_spectrum(tr_spatial_flat)
    rank_heldout_spatial = compute_rank_and_spectrum(ho_spatial_flat)
    rank_train_pooled = compute_rank_and_spectrum(tr_pool)
    rank_heldout_pooled = compute_rank_and_spectrum(ho_pool)

    # 3. Pairwise cosine similarity stats
    cos_train_spatial = compute_pairwise_cosine_stats(tr_spatial_flat, max_samples=500)
    cos_heldout_spatial = compute_pairwise_cosine_stats(ho_spatial_flat, max_samples=500)
    cos_train_pooled = compute_pairwise_cosine_stats(tr_pool)
    cos_heldout_pooled = compute_pairwise_cosine_stats(ho_pool)

    # Within-image spatial cosine similarity (spatial diversity across 64 patches of single image)
    within_img_uncent = []
    within_img_cent = []
    for b in range(min(16, tr_tokens.size(0))):
        toks = tr_tokens[b]
        st = compute_pairwise_cosine_stats(toks)
        within_img_uncent.append(st["uncentered_mean"])
        within_img_cent.append(st["centered_mean"])

    within_img_stats = {
        "within_image_uncentered_mean": float(np.mean(within_img_uncent)),
        "within_image_centered_mean": float(np.mean(within_img_cent)),
    }

    # 4. Visual Distinguishability Test across visually distinct scenes
    # Find pairwise pixel differences between the first 16 held-out frames
    ho_first16 = heldout_obs[:16]
    ho_first16_pool = ho_pool[:16]
    ho_first16_pool_cent = ho_first16_pool - ho_first16_pool.mean(dim=0, keepdim=True)
    
    pw_pix_diff = []
    pw_pool_uncent_dist = []
    pw_pool_cent_dist = []
    for i in range(16):
        for j in range(i + 1, 16):
            pix_d = float(torch.abs(ho_first16[i] - ho_first16[j]).mean().item())
            cos_u = float((1.0 - (F.normalize(ho_first16_pool[i], dim=-1) * F.normalize(ho_first16_pool[j], dim=-1)).sum()).item())
            cos_c = float((1.0 - (F.normalize(ho_first16_pool_cent[i], dim=-1) * F.normalize(ho_first16_pool_cent[j], dim=-1)).sum()).item())
            pw_pix_diff.append(pix_d)
            pw_pool_uncent_dist.append(cos_u)
            pw_pool_cent_dist.append(cos_c)

    corr_uncent = float(np.corrcoef(pw_pix_diff, pw_pool_uncent_dist)[0, 1])
    corr_cent = float(np.corrcoef(pw_pix_diff, pw_pool_cent_dist)[0, 1])

    distinguishability = {
        "mean_pixel_difference": float(np.mean(pw_pix_diff)),
        "uncentered_cosine_dist_mean": float(np.mean(pw_pool_uncent_dist)),
        "centered_cosine_dist_mean": float(np.mean(pw_pool_cent_dist)),
        "correlation_pix_diff_to_uncentered_dist": corr_uncent,
        "correlation_pix_diff_to_centered_dist": corr_cent,
    }

    print(f"• Spatial Tokens Centered PR (Train): {rank_train_spatial['centered']['participation_ratio']:.2f} (Entropy ER: {rank_train_spatial['centered']['entropy_effective_rank']:.2f})")
    print(f"• Spatial Tokens Uncentered PR (Train): {rank_train_spatial['uncentered']['participation_ratio']:.2f} (Entropy ER: {rank_train_spatial['uncentered']['entropy_effective_rank']:.2f})")
    print(f"• Pooled Vectors Centered PR (Train): {rank_train_pooled['centered']['participation_ratio']:.2f} (Entropy ER: {rank_train_pooled['centered']['entropy_effective_rank']:.2f})")
    print(f"• Spatial Cosine Sim (Train): Uncentered = {cos_train_spatial['uncentered_mean']:.4f} | Centered = {cos_train_spatial['centered_mean']:.4f}")
    print(f"• Pooled Cosine Sim (Train):  Uncentered = {cos_train_pooled['uncentered_mean']:.4f} | Centered = {cos_train_pooled['centered_mean']:.4f}")
    print(f"• Within-Image Spatial Cosine Sim: Uncentered = {within_img_stats['within_image_uncentered_mean']:.4f} | Centered = {within_img_stats['within_image_centered_mean']:.4f}")

    return {
        "architecture": arch_info,
        "train_spatial_tokens": rank_train_spatial,
        "heldout_spatial_tokens": rank_heldout_spatial,
        "train_pooled_vectors": rank_train_pooled,
        "heldout_pooled_vectors": rank_heldout_pooled,
        "cosine_similarity": {
            "train_spatial": cos_train_spatial,
            "heldout_spatial": cos_heldout_spatial,
            "train_pooled": cos_train_pooled,
            "heldout_pooled": cos_heldout_pooled,
            "within_image": within_img_stats,
        },
        "distinguishability": distinguishability,
    }


# ==============================================================================
# Step G2: Latent Representation Prediction Support Test
# ==============================================================================
def run_g2_prediction_test(dataset: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      AEON PHASE G2: LATENT PREDICTION SUPPORT TEST")
    print("=" * 75)

    device = torch.device("cpu")
    torch.manual_seed(12345)
    encoder = VisualEncoder(latent_dim=128).to(device)
    encoder.eval()

    def encode_split(raw):
        obs_t = raw["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        obs_tp1 = raw["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        actions = raw["actions"].to(device)
        with torch.no_grad():
            zt, ut = encoder(obs_t)
            ztarg, utarg = encoder(obs_tp1)
            ht = torch.cat([ut, ut], dim=-1)
        return zt, ztarg, ht, actions

    tr_zt, tr_ztarg, tr_ht, tr_act = encode_split(dataset["train"])
    ho_zt, ho_ztarg, ho_ht, ho_act = encode_split(dataset["heldout"])

    threshold = 0.001

    # Baseline A: Persistence
    def eval_persistence(zt, ztarg):
        n_zt = F.normalize(zt, dim=-1, eps=1e-6)
        n_ztarg = F.normalize(ztarg, dim=-1, eps=1e-6)
        dist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)
        low_m = dist < threshold
        high_m = dist >= threshold
        return {
            "total_error": float(dist.mean().item()),
            "low_change_error": float(dist[low_m].mean().item()) if low_m.any() else 0.0,
            "high_change_error": float(dist[high_m].mean().item()) if high_m.any() else 0.0,
            "high_change_pct": float(high_m.float().mean().item() * 100.0),
        }

    persist_train = eval_persistence(tr_zt, tr_ztarg)
    persist_heldout = eval_persistence(ho_zt, ho_ztarg)

    print(f"• Baseline A (Persistence): Held-Out Total = {persist_heldout['total_error']:.6f} | High-Change = {persist_heldout['high_change_error']:.6f} | Low-Change = {persist_heldout['low_change_error']:.6f}")

    def evaluate_model_on_split(model, act_table, zt, ht, actions, ztarg):
        model.eval()
        with torch.no_grad():
            n_ztarg = F.normalize(ztarg, dim=-1, eps=1e-6)
            n_zt = F.normalize(zt, dim=-1, eps=1e-6)
            persist_dist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)
            low_m = persist_dist < threshold
            high_m = persist_dist >= threshold

            # Correct action
            a_emb = act_table(actions)
            pred = model(zt, ht, a_emb)
            n_pred = F.normalize(pred, dim=-1, eps=1e-6)
            err = 1.0 - (n_pred * n_ztarg).sum(dim=-1)

            # Shuffled action
            shuf_act = torch.roll(actions, shifts=1)
            a_emb_shuf = act_table(shuf_act)
            pred_shuf = model(zt, ht, a_emb_shuf)
            n_pred_shuf = F.normalize(pred_shuf, dim=-1, eps=1e-6)
            err_shuf = 1.0 - (n_pred_shuf * n_ztarg).sum(dim=-1)

            total_err = float(err.mean().item())
            low_err = float(err[low_m].mean().item()) if low_m.any() else 0.0
            high_err = float(err[high_m].mean().item()) if high_m.any() else 0.0

            shuf_total = float(err_shuf.mean().item())
            shuf_high = float(err_shuf[high_m].mean().item()) if high_m.any() else 0.0

            adv_total = shuf_total - total_err
            adv_high = shuf_high - high_err

        return {
            "total_error": total_err,
            "low_change_error": low_err,
            "high_change_error": high_err,
            "shuffled_total_error": shuf_total,
            "shuffled_high_error": shuf_high,
            "action_advantage_total": adv_total,
            "action_advantage_high": adv_high,
        }

    seeds = [42, 43, 44]
    results_residual = []
    results_existing = []

    for seed in seeds:
        # 1. Train Baseline B: Simple Residual Predictor
        torch.manual_seed(seed)
        res_model = SimpleResidualPredictor(latent_dim=128, action_dim=64).to(device)
        act_res = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
        opt_res = torch.optim.Adam(list(res_model.parameters()) + list(act_res.parameters()), lr=3e-4)

        for step in range(300):
            opt_res.zero_grad()
            a_emb = act_res(tr_act)
            pred = res_model(tr_zt, tr_ht, a_emb)
            pn = F.normalize(pred, dim=-1, eps=1e-6)
            tn = F.normalize(tr_ztarg, dim=-1, eps=1e-6)
            loss = (1.0 - (pn * tn).sum(dim=-1)).mean()
            loss.backward()
            opt_res.step()

        res_train = evaluate_model_on_split(res_model, act_res, tr_zt, tr_ht, tr_act, tr_ztarg)
        res_ho = evaluate_model_on_split(res_model, act_res, ho_zt, ho_ht, ho_act, ho_ztarg)
        results_residual.append({"seed": seed, "train": res_train, "heldout": res_ho})

        # 2. Train Baseline C: Existing Predictor
        torch.manual_seed(seed)
        exist_model = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
        act_exist = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
        opt_exist = torch.optim.Adam(list(exist_model.parameters()) + list(act_exist.parameters()), lr=3e-4)

        for step in range(300):
            opt_exist.zero_grad()
            a_emb = act_exist(tr_act)
            pred = exist_model(tr_zt, tr_ht, a_emb)
            pn = F.normalize(pred, dim=-1, eps=1e-6)
            tn = F.normalize(tr_ztarg, dim=-1, eps=1e-6)
            loss = (1.0 - (pn * tn).sum(dim=-1)).mean()
            loss.backward()
            opt_exist.step()

        exist_train = evaluate_model_on_split(exist_model, act_exist, tr_zt, tr_ht, tr_act, tr_ztarg)
        exist_ho = evaluate_model_on_split(exist_model, act_exist, ho_zt, ho_ht, ho_act, ho_ztarg)
        results_existing.append({"seed": seed, "train": exist_train, "heldout": exist_ho})

    def aggregate_runs(runs):
        keys = ["total_error", "low_change_error", "high_change_error", "action_advantage_total", "action_advantage_high"]
        agg = {"train": {}, "heldout": {}}
        for split in ["train", "heldout"]:
            for k in keys:
                vals = [r[split][k] for r in runs]
                agg[split][f"{k}_mean"] = float(np.mean(vals))
                agg[split][f"{k}_std"] = float(np.std(vals))
        return agg

    agg_residual = aggregate_runs(results_residual)
    agg_existing = aggregate_runs(results_existing)

    print(f"• Baseline B (Residual Predictor) Held-Out Total: {agg_residual['heldout']['total_error_mean']:.6f} ± {agg_residual['heldout']['total_error_std']:.6f} (High: {agg_residual['heldout']['high_change_error_mean']:.6f})")
    print(f"• Baseline C (Existing Predictor) Held-Out Total: {agg_existing['heldout']['total_error_mean']:.6f} ± {agg_existing['heldout']['total_error_std']:.6f} (High: {agg_existing['heldout']['high_change_error_mean']:.6f})")
    print(f"• Baseline B Held-Out High Advantage: {agg_residual['heldout']['action_advantage_high_mean']:+.6f} ± {agg_residual['heldout']['action_advantage_high_std']:.6f}")
    print(f"• Baseline C Held-Out High Advantage: {agg_existing['heldout']['action_advantage_high_mean']:+.6f} ± {agg_existing['heldout']['action_advantage_high_std']:.6f}")

    return {
        "persistence": {"train": persist_train, "heldout": persist_heldout},
        "baseline_b_residual": {"runs": results_residual, "aggregate": agg_residual},
        "baseline_c_existing": {"runs": results_existing, "aggregate": agg_existing},
    }


# ==============================================================================
# Step G3: Verify Target and Loss Pipeline
# ==============================================================================
def run_g3_pipeline_audit(dataset: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      AEON PHASE G3: TARGET & LOSS PIPELINE AUDIT")
    print("=" * 75)

    device = torch.device("cpu")
    torch.manual_seed(12345)
    encoder = VisualEncoder(latent_dim=128).to(device)
    encoder.eval()

    obs_t = dataset["train"]["obs_t"][:16].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    obs_tp1 = dataset["train"]["obs_tp1"][:16].permute(0, 3, 1, 2).float().div_(255.0).to(device)

    with torch.no_grad():
        zt, _ = encoder(obs_t)
        ztarg, _ = encoder(obs_tp1)

    # 1. Normalization Asymmetry Test
    ln = nn.LayerNorm(128).to(device)
    zt_ln = ln(zt)
    ztarg_ln = ln(ztarg)

    cos_dist = lambda a, b: (1.0 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)).mean().item()

    d_raw_to_raw = cos_dist(zt, ztarg)                  # 0.0062
    d_ln_to_ln = cos_dist(zt_ln, ztarg_ln)              # 0.0091
    d_asym_zt_to_ztarg = cos_dist(zt_ln, ztarg)          # 0.1781 (Asymmetric LayerNorm)
    d_identity_to_ln = cos_dist(zt, zt_ln)              # 0.1733 (Self against LayerNorm self)

    norm_stats = {
        "raw_persistence_distance": d_raw_to_raw,
        "symmetric_layernorm_persistence_distance": d_ln_to_ln,
        "asymmetric_layernorm_predictor_to_raw_target": d_asym_zt_to_ztarg,
        "identity_self_to_layernorm_distance": d_identity_to_ln,
        "target_tokens_mean_across_channels": float(ztarg.mean(dim=-1).mean().item()),
        "target_tokens_std_across_channels": float(ztarg.std(dim=-1).mean().item()),
        "predictor_final_layernorm_mean": float(zt_ln.mean(dim=-1).mean().item()),
        "predictor_final_layernorm_std": float(zt_ln.std(dim=-1).mean().item()),
    }

    # 2. Gradient Flow & Detachment Audit
    model = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
    act_table = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
    
    zt_grad = zt.clone().requires_grad_(True)
    ht = torch.zeros(16, 256, requires_grad=True, device=device)
    a_tokens = torch.zeros(16, dtype=torch.long, device=device)
    
    a_emb = act_table(a_tokens)
    pred = model(zt_grad, ht, a_emb)
    loss = (1.0 - (F.normalize(pred, dim=-1) * F.normalize(ztarg, dim=-1)).sum(dim=-1)).mean()
    loss.backward()

    grad_stats = {
        "grad_norm_zt_input": float(torch.norm(zt_grad.grad).item()),
        "grad_norm_ht_context": float(torch.norm(ht.grad).item()),
        "grad_norm_action_embedding": float(torch.norm(act_table.embedding.weight.grad).item()),
        "grad_norm_cond_mlp": float(torch.norm(model.cond_mlp[0].weight.grad).item()),
        "grad_norm_transformer_layer0": float(torch.norm(model.transformer.layers[0].linear1.weight.grad).item()),
        "grad_norm_final_norm": float(torch.norm(model.final_norm.weight.grad).item()),
    }

    # 3. EMA Schedule & Target Stability
    ema_target = EmaTargetEncoder(encoder)
    param_before = [p.clone() for p in ema_target.encoder.parameters()]
    ema_target.update(encoder, momentum=0.996)
    param_after = [p.clone() for p in ema_target.encoder.parameters()]
    max_param_shift = max(torch.norm(a - b).item() for a, b in zip(param_before, param_after))

    ema_stats = {
        "ema_update_on_identical_weights_shift": max_param_shift,
        "target_requires_grad": any(p.requires_grad for p in ema_target.encoder.parameters()),
        "target_eval_mode": not ema_target.encoder.training,
    }

    print(f"• Raw Persistence Distance: {norm_stats['raw_persistence_distance']:.6f}")
    print(f"• Symmetric LayerNorm Persistence Distance: {norm_stats['symmetric_layernorm_persistence_distance']:.6f}")
    print(f"• Asymmetric LayerNorm Distance (Self): {norm_stats['identity_self_to_layernorm_distance']:.6f}")
    print(f"• Asymmetric Persistence Distance: {norm_stats['asymmetric_layernorm_predictor_to_raw_target']:.6f}")
    print(f"• Gradient Norms: Input z_t={grad_stats['grad_norm_zt_input']:.4f}, ActionEmb={grad_stats['grad_norm_action_embedding']:.4f}, CondMLP={grad_stats['grad_norm_cond_mlp']:.4f}")

    return {
        "normalization_asymmetry": norm_stats,
        "gradient_flow": grad_stats,
        "ema_target_integrity": ema_stats,
    }


# ==============================================================================
# Step G4: Isolated Encoder-Training Test
# ==============================================================================
def run_g4_encoder_training_test(dataset: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      AEON PHASE G4: ISOLATED ENCODER-TRAINING TEST")
    print("=" * 75)

    device = torch.device("cpu")
    tr_obs_t = dataset["train"]["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    tr_obs_tp1 = dataset["train"]["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    tr_act = dataset["train"]["actions"].to(device)

    ho_obs_t = dataset["heldout"]["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    ho_obs_tp1 = dataset["heldout"]["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    ho_act = dataset["heldout"]["actions"].to(device)

    num_updates = 150
    batch_size = 32
    threshold = 0.001
    seeds = [42, 43, 44]

    def evaluate_pipeline(online_enc, targ_enc, pred, act_table):
        online_enc.eval()
        targ_enc.eval()
        pred.eval()
        with torch.no_grad():
            ho_zt, _ = online_enc(ho_obs_t)
            ho_ztarg, _ = targ_enc(ho_obs_tp1)

            n_zt = F.normalize(ho_zt, dim=-1, eps=1e-6)
            n_ztarg = F.normalize(ho_ztarg, dim=-1, eps=1e-6)

            # Persistence
            persist_dist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)
            low_m = persist_dist < threshold
            high_m = persist_dist >= threshold

            # Correct prediction
            a_emb = act_table(ho_act)
            p = pred(ho_zt, None, a_emb)
            n_p = F.normalize(p, dim=-1, eps=1e-6)
            err = 1.0 - (n_p * n_ztarg).sum(dim=-1)

            # Shuffled prediction
            shuf_act = torch.roll(ho_act, 1)
            a_emb_shuf = act_table(shuf_act)
            p_shuf = pred(ho_zt, None, a_emb_shuf)
            n_pshuf = F.normalize(p_shuf, dim=-1, eps=1e-6)
            err_shuf = 1.0 - (n_pshuf * n_ztarg).sum(dim=-1)

            # Effective rank of online spatial tokens
            flat_ho = ho_zt.reshape(-1, 128)
            cent = flat_ho - flat_ho.mean(dim=0, keepdim=True)
            cov_c = torch.mm(cent.t(), cent) / max(1, flat_ho.size(0) - 1)
            ev_c = torch.linalg.eigvalsh(cov_c).clamp(min=0.0)
            pr_c = float((ev_c.sum()**2 / max(1e-12, (ev_c**2).sum())).item())

            return {
                "persist_total": float(persist_dist.mean().item()),
                "persist_high": float(persist_dist[high_m].mean().item()) if high_m.any() else 0.0,
                "persist_low": float(persist_dist[low_m].mean().item()) if low_m.any() else 0.0,
                "pred_total": float(err.mean().item()),
                "pred_high": float(err[high_m].mean().item()) if high_m.any() else 0.0,
                "pred_low": float(err[low_m].mean().item()) if low_m.any() else 0.0,
                "action_adv_total": float((err_shuf.mean() - err.mean()).item()),
                "action_adv_high": float((err_shuf[high_m].mean() - err[high_m].mean()).item()),
                "online_effective_rank": pr_c,
            }

    results_A, results_B, results_C = [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        N_train = tr_obs_t.size(0)

        # -------------------------------------------------------------
        # Variant A: Frozen Encoder Baseline
        # -------------------------------------------------------------
        enc_A = VisualEncoder(128).to(device)
        enc_A.eval()
        targ_A = copy.deepcopy(enc_A)
        targ_A.eval()
        pred_A = SimpleResidualPredictor(128, 64).to(device)
        act_A = ActionEmbedding(17, 64).to(device)
        opt_A = torch.optim.Adam(list(pred_A.parameters()) + list(act_A.parameters()), lr=3e-4)

        for u in range(num_updates):
            batch_idx = torch.randperm(N_train)[:batch_size]
            b_ot = tr_obs_t[batch_idx]
            b_otp1 = tr_obs_tp1[batch_idx]
            b_a = tr_act[batch_idx]

            opt_A.zero_grad()
            with torch.no_grad():
                zt, _ = enc_A(b_ot)
                ztarg, _ = targ_A(b_otp1)
            p = pred_A(zt, None, act_A(b_a))
            loss = (1.0 - (F.normalize(p, dim=-1) * F.normalize(ztarg, dim=-1)).sum(dim=-1)).mean()
            loss.backward()
            opt_A.step()

        res_A = evaluate_pipeline(enc_A, targ_A, pred_A, act_A)
        results_A.append(res_A)

        # -------------------------------------------------------------
        # Variant B: Trainable Encoder + Standard JEPA Objective
        # -------------------------------------------------------------
        enc_B = VisualEncoder(128).to(device)
        enc_B.train()
        targ_B = copy.deepcopy(enc_B)
        targ_B.eval()
        for p in targ_B.parameters():
            p.requires_grad = False
        pred_B = SimpleResidualPredictor(128, 64).to(device)
        act_B = ActionEmbedding(17, 64).to(device)
        opt_B = torch.optim.Adam([
            {"params": enc_B.parameters(), "lr": 1e-4},
            {"params": list(pred_B.parameters()) + list(act_B.parameters()), "lr": 3e-4},
        ])

        for u in range(num_updates):
            batch_idx = torch.randperm(N_train)[:batch_size]
            b_ot = tr_obs_t[batch_idx]
            b_otp1 = tr_obs_tp1[batch_idx]
            b_a = tr_act[batch_idx]

            opt_B.zero_grad()
            zt, _ = enc_B(b_ot)
            with torch.no_grad():
                ztarg, _ = targ_B(b_otp1)
            p = pred_B(zt, None, act_B(b_a))
            loss = (1.0 - (F.normalize(p, dim=-1) * F.normalize(ztarg, dim=-1)).sum(dim=-1)).mean()
            loss.backward()
            opt_B.step()

            # Target EMA update
            with torch.no_grad():
                for tp, op in zip(targ_B.parameters(), enc_B.parameters()):
                    tp.data.mul_(0.996).add_(op.data, alpha=0.004)

        res_B = evaluate_pipeline(enc_B, targ_B, pred_B, act_B)
        results_B.append(res_B)

        # -------------------------------------------------------------
        # Variant C: Trainable Encoder + JEPA Objective + Variance Regularizer
        # -------------------------------------------------------------
        enc_C = VisualEncoder(128).to(device)
        enc_C.train()
        targ_C = copy.deepcopy(enc_C)
        targ_C.eval()
        for p in targ_C.parameters():
            p.requires_grad = False
        pred_C = SimpleResidualPredictor(128, 64).to(device)
        act_C = ActionEmbedding(17, 64).to(device)
        opt_C = torch.optim.Adam([
            {"params": enc_C.parameters(), "lr": 1e-4},
            {"params": list(pred_C.parameters()) + list(act_C.parameters()), "lr": 3e-4},
        ])

        for u in range(num_updates):
            batch_idx = torch.randperm(N_train)[:batch_size]
            b_ot = tr_obs_t[batch_idx]
            b_otp1 = tr_obs_tp1[batch_idx]
            b_a = tr_act[batch_idx]

            opt_C.zero_grad()
            zt, _ = enc_C(b_ot)
            with torch.no_grad():
                ztarg, _ = targ_C(b_otp1)
            p = pred_C(zt, None, act_C(b_a))
            jepa_loss = (1.0 - (F.normalize(p, dim=-1) * F.normalize(ztarg, dim=-1)).sum(dim=-1)).mean()

            # Spatial token channel-variance hinge regularizer: gamma = 1.0
            flat_tokens = zt.reshape(-1, 128)
            std_c = torch.sqrt(flat_tokens.var(dim=0) + 1e-4)
            var_loss = torch.mean(F.relu(1.0 - std_c))

            total_loss = jepa_loss + 0.10 * var_loss
            total_loss.backward()
            opt_C.step()

            with torch.no_grad():
                for tp, op in zip(targ_C.parameters(), enc_C.parameters()):
                    tp.data.mul_(0.996).add_(op.data, alpha=0.004)

        res_C = evaluate_pipeline(enc_C, targ_C, pred_C, act_C)
        results_C.append(res_C)

    def agg_g4(res_list):
        agg = {}
        for k in res_list[0].keys():
            vals = [r[k] for r in res_list]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        return agg

    agg_A = agg_g4(results_A)
    agg_B = agg_g4(results_B)
    agg_C = agg_g4(results_C)

    print(f"• Variant A (Frozen Encoder):       Pred Total = {agg_A['pred_total_mean']:.6f} | High = {agg_A['pred_high_mean']:.6f} | Adv High = {agg_A['action_adv_high_mean']:+.6f} | EffRank = {agg_A['online_effective_rank_mean']:.2f}")
    print(f"• Variant B (Trainable JEPA):       Pred Total = {agg_B['pred_total_mean']:.6f} | High = {agg_B['pred_high_mean']:.6f} | Adv High = {agg_B['action_adv_high_mean']:+.6f} | EffRank = {agg_B['online_effective_rank_mean']:.2f}")
    print(f"• Variant C (Trainable + VarReg):   Pred Total = {agg_C['pred_total_mean']:.6f} | High = {agg_C['pred_high_mean']:.6f} | Adv High = {agg_C['action_adv_high_mean']:+.6f} | EffRank = {agg_C['online_effective_rank_mean']:.2f}")

    return {
        "variant_a_frozen": {"runs": results_A, "aggregate": agg_A},
        "variant_b_trainable_jepa": {"runs": results_B, "aggregate": agg_B},
        "variant_c_trainable_var_reg": {"runs": results_C, "aggregate": agg_C},
    }


# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    print("=" * 75)
    print("      AEON JEPA-CRAFTER: PHASE G ROOT-CAUSE INVESTIGATION")
    print("=" * 75)

    dataset_path = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/dataset_256_transitions.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Phase F dataset not found at: {dataset_path}")

    dataset = torch.load(dataset_path, weights_only=False)
    print(f"• Loaded 256 Transitions (128 Train / 128 Held-Out) from: {dataset_path}")

    # G1: Encoder Geometry Audit
    g1_results = run_g1_audit(dataset)

    # G2: Latent Representation Prediction Support Test
    g2_results = run_g2_prediction_test(dataset)

    # G3: Target and Loss Pipeline Audit
    g3_results = run_g3_pipeline_audit(dataset)

    # G4: Isolated Encoder-Training Test
    g4_results = run_g4_encoder_training_test(dataset)

    # Save comprehensive results
    output_dir = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase_g_results.json")

    full_results = {
        "g1_encoder_geometry": g1_results,
        "g2_prediction_support": g2_results,
        "g3_target_loss_pipeline": g3_results,
        "g4_isolated_encoder_training": g4_results,
    }
    save_json(full_results, out_file)
    print(f"\n[DONE] All Phase G investigation tasks completed successfully!")
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
