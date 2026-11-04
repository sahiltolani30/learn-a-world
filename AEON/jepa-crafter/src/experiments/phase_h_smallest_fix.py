"""AEON Phase H: Smallest Architectural Change Verification Experiment.

Controlled 2x2 factorial evaluation of:
- Factor 1: Residual connection (No vs Yes)
- Factor 2: Normalization alignment (No vs Yes)

Variants:
- Variant A: Baseline control (LatentPredictorMember with Pre-LN Transformer + LayerNorm vs raw target)
- Variant B: Normalization-aligned predictor (Pre-LN Transformer directly into raw target space, no LayerNorm)
- Variant C: Residual predictor with baseline normalization (LayerNorm(zt + delta))
- Variant D: Residual + Normalization-aligned predictor (zt + delta, direct raw target space)

Evaluated across 3 optimization seeds (42, 43, 44) on the Phase F 256-transition dataset
(128 Train, 128 Held-Out). Held-out worlds (Seeds 103, 104) are strictly held-out.
"""
import os
import sys
import copy
import json
import subprocess
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
from src.utils.logging import save_json


# ==============================================================================
# Predictor Architectures
# ==============================================================================
class VariantAPredictor(nn.Module):
    """Variant A: Existing baseline predictor (control). Pre-LN Transformer + LayerNorm."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.member = LatentPredictorMember(
            latent_dim=latent_dim, context_dim=context_dim, action_dim=action_dim
        )
    def forward(self, zt, ht, a_emb):
        return self.member(zt, ht, a_emb)


class VariantBPredictor(nn.Module):
    """Variant B: Normalization-aligned predictor. No final LayerNorm; outputs in raw token space."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        return self.transformer(modulated)


class VariantCPredictor(nn.Module):
    """Variant C: Residual predictor with baseline normalization. z_hat = LayerNorm(zt + delta)."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        delta = self.transformer(modulated)
        return self.final_norm(zt + delta)


class VariantDPredictor(nn.Module):
    """Variant D: Residual + Normalization-aligned predictor. z_hat = zt + delta (no final LayerNorm)."""
    def __init__(self, latent_dim=128, context_dim=256, action_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, zt, ht, a_emb):
        cond_input = torch.cat([ht, a_emb], dim=-1)
        film_params = self.cond_mlp(cond_input)
        gamma, beta = film_params.chunk(2, dim=-1)
        modulated = (1.0 + gamma.unsqueeze(1)) * zt + beta.unsqueeze(1)
        delta = self.transformer(modulated)
        return zt + delta


# ==============================================================================
# Effective Rank & Geometry Helpers
# ==============================================================================
def compute_rank_and_spectrum(features: torch.Tensor) -> Dict[str, Any]:
    M, D = features.shape
    features_float = features.float()

    # Centered Covariance
    mean_vec = features_float.mean(dim=0, keepdim=True)
    centered = features_float - mean_vec
    cov_c = torch.mm(centered.t(), centered) / max(1, M - 1)
    ev_c = torch.linalg.eigvalsh(cov_c).clamp(min=0.0)
    
    tr_c = ev_c.sum().item()
    sum_sq_c = (ev_c ** 2).sum().item()
    pr_c = (tr_c ** 2 / max(1e-12, sum_sq_c)) if sum_sq_c > 1e-12 else 0.0
    
    p_c = ev_c / max(1e-12, tr_c)
    p_c_pos = p_c[p_c > 1e-12]
    ent_c = float(torch.exp(-torch.sum(p_c_pos * torch.log(p_c_pos))).item())

    # Uncentered Covariance
    cov_u = torch.mm(features_float.t(), features_float) / max(1, M)
    ev_u = torch.linalg.eigvalsh(cov_u).clamp(min=0.0)
    tr_u = ev_u.sum().item()
    sum_sq_u = (ev_u ** 2).sum().item()
    pr_u = (tr_u ** 2 / max(1e-12, sum_sq_u)) if sum_sq_u > 1e-12 else 0.0
    
    p_u = ev_u / max(1e-12, tr_u)
    p_u_pos = p_u[p_u > 1e-12]
    ent_u = float(torch.exp(-torch.sum(p_u_pos * torch.log(p_u_pos))).item())

    sorted_ev_c, _ = torch.sort(ev_c, descending=True)
    channel_stds = centered.std(dim=0)

    return {
        "num_samples": M,
        "feature_dim": D,
        "centered_pr": float(pr_c),
        "centered_entropy_er": float(ent_c),
        "uncentered_pr": float(pr_u),
        "uncentered_entropy_er": float(ent_u),
        "top1_var_pct": float(sorted_ev_c[0].item() / max(1e-12, tr_c) * 100.0),
        "top5_var_pct": float(sorted_ev_c[:5].sum().item() / max(1e-12, tr_c) * 100.0),
        "channel_std_min": float(channel_stds.min().item()),
        "channel_std_max": float(channel_stds.max().item()),
        "channel_std_mean": float(channel_stds.mean().item()),
        "collapsed_channels_under_0_01": int((channel_stds < 0.01).sum().item()),
    }


def compute_pairwise_cosine(features: torch.Tensor, max_samples: int = 500) -> Dict[str, float]:
    if features.size(0) > max_samples:
        indices = torch.randperm(features.size(0))[:max_samples]
        features = features[indices]
    
    normed = F.normalize(features, dim=-1, eps=1e-6)
    sim_mat = torch.mm(normed, normed.t())
    N = sim_mat.size(0)
    triu_indices = torch.triu_indices(N, N, offset=1)
    vals = sim_mat[triu_indices[0], triu_indices[1]]

    return {
        "mean_cosine_similarity": float(vals.mean().item()),
        "std_cosine_similarity": float(vals.std().item()),
    }


# ==============================================================================
# Evaluation Function
# ==============================================================================
def evaluate_variant(
    model: nn.Module,
    act_table: nn.Module,
    zt: torch.Tensor,
    ht: torch.Tensor,
    actions: torch.Tensor,
    ztarg: torch.Tensor,
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics on a dataset split."""
    model.eval()
    act_table.eval()
    with torch.no_grad():
        B, N, D = zt.shape
        n_ztarg = F.normalize(ztarg, dim=-1, eps=1e-6)
        n_zt = F.normalize(zt, dim=-1, eps=1e-6)

        # Token-level and transition-level persistence
        token_persist_dist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)     # [B, 64]
        trans_persist_dist = token_persist_dist.mean(dim=-1)          # [B]

        # Masks
        trans_high_m = trans_persist_dist >= threshold
        trans_low_m = trans_persist_dist < threshold
        token_dynamic_m = token_persist_dist >= threshold
        token_static_m = token_persist_dist < threshold

        # Correct-action prediction
        a_emb = act_table(actions)
        pred = model(zt, ht, a_emb)
        n_pred = F.normalize(pred, dim=-1, eps=1e-6)
        token_err = 1.0 - (n_pred * n_ztarg).sum(dim=-1)            # [B, 64]
        trans_err = token_err.mean(dim=-1)                           # [B]

        # Shuffled-action prediction (deterministic derangement roll=1)
        shuf_actions = torch.roll(actions, shifts=1)
        a_emb_shuf = act_table(shuf_actions)
        pred_shuf = model(zt, ht, a_emb_shuf)
        n_pred_shuf = F.normalize(pred_shuf, dim=-1, eps=1e-6)
        token_err_shuf = 1.0 - (n_pred_shuf * n_ztarg).sum(dim=-1)   # [B, 64]
        trans_err_shuf = token_err_shuf.mean(dim=-1)                 # [B]

        # Transition-level errors
        total_err = float(trans_err.mean().item())
        high_change_err = float(trans_err[trans_high_m].mean().item()) if trans_high_m.any() else 0.0
        low_change_err = float(trans_err[trans_low_m].mean().item()) if trans_low_m.any() else 0.0

        # Region-level errors (dynamic vs static tokens)
        dynamic_token_err = float(token_err[token_dynamic_m].mean().item()) if token_dynamic_m.any() else 0.0
        static_token_err = float(token_err[token_static_m].mean().item()) if token_static_m.any() else 0.0

        # Shuffled errors
        shuf_total_err = float(trans_err_shuf.mean().item())
        shuf_high_err = float(trans_err_shuf[trans_high_m].mean().item()) if trans_high_m.any() else 0.0
        shuf_dynamic_err = float(token_err_shuf[token_dynamic_m].mean().item()) if token_dynamic_m.any() else 0.0

        # Action advantages (positive means correct action is better)
        adv_total = shuf_total_err - total_err
        adv_high = shuf_high_err - high_change_err
        adv_dynamic = shuf_dynamic_err - dynamic_token_err

        # Representation health of predicted tokens
        rank_stats = compute_rank_and_spectrum(pred.reshape(-1, D))
        cos_stats = compute_pairwise_cosine(pred.reshape(-1, D))

    return {
        "transition_level": {
            "total_error": total_err,
            "high_change_error": high_change_err,
            "low_change_error": low_change_err,
            "shuffled_total_error": shuf_total_err,
            "shuffled_high_change_error": shuf_high_err,
            "action_advantage_total": adv_total,
            "action_advantage_high": adv_high,
        },
        "region_level": {
            "dynamic_tokens_error": dynamic_token_err,
            "static_tokens_error": static_token_err,
            "shuffled_dynamic_error": shuf_dynamic_err,
            "action_advantage_dynamic": adv_dynamic,
            "num_dynamic_tokens": int(token_dynamic_m.sum().item()),
            "num_static_tokens": int(token_static_m.sum().item()),
        },
        "representation_health": {
            "effective_rank": rank_stats,
            "pairwise_cosine": cos_stats,
        },
    }


def compute_persistence_baseline(
    zt: torch.Tensor,
    ztarg: torch.Tensor,
    threshold: float = 0.001,
) -> Dict[str, Any]:
    with torch.no_grad():
        n_ztarg = F.normalize(ztarg, dim=-1, eps=1e-6)
        n_zt = F.normalize(zt, dim=-1, eps=1e-6)

        token_persist = 1.0 - (n_zt * n_ztarg).sum(dim=-1)
        trans_persist = token_persist.mean(dim=-1)

        trans_high_m = trans_persist >= threshold
        trans_low_m = trans_persist < threshold
        token_dyn_m = token_persist >= threshold
        token_stat_m = token_persist < threshold

        return {
            "transition_level": {
                "total_error": float(trans_persist.mean().item()),
                "high_change_error": float(trans_persist[trans_high_m].mean().item()) if trans_high_m.any() else 0.0,
                "low_change_error": float(trans_persist[trans_low_m].mean().item()) if trans_low_m.any() else 0.0,
                "num_high_change_transitions": int(trans_high_m.sum().item()),
                "num_low_change_transitions": int(trans_low_m.sum().item()),
            },
            "region_level": {
                "dynamic_tokens_error": float(token_persist[token_dyn_m].mean().item()) if token_dyn_m.any() else 0.0,
                "static_tokens_error": float(token_persist[token_stat_m].mean().item()) if token_stat_m.any() else 0.0,
                "num_dynamic_tokens": int(token_dyn_m.sum().item()),
                "num_static_tokens": int(token_stat_m.sum().item()),
            },
        }


# ==============================================================================
# Audit Gradient Flow
# ==============================================================================
def audit_gradient_flow(model: nn.Module, act_table: nn.Module, zt: torch.Tensor, ht: torch.Tensor, actions: torch.Tensor, ztarg: torch.Tensor) -> Dict[str, float]:
    model.train()
    act_table.train()
    
    zt_g = zt[:16].clone().detach().requires_grad_(True)
    ht_g = ht[:16].clone().detach().requires_grad_(True)
    act_g = actions[:16]
    targ_g = ztarg[:16]

    a_emb = act_table(act_g)
    pred = model(zt_g, ht_g, a_emb)
    pn = F.normalize(pred, dim=-1, eps=1e-6)
    tn = F.normalize(targ_g, dim=-1, eps=1e-6)
    loss = (1.0 - (pn * tn).sum(dim=-1)).mean()

    model.zero_grad()
    act_table.zero_grad()
    loss.backward()

    grads = {
        "grad_norm_zt": float(torch.norm(zt_g.grad).item()) if zt_g.grad is not None else 0.0,
        "grad_norm_ht": float(torch.norm(ht_g.grad).item()) if ht_g.grad is not None else 0.0,
        "grad_norm_action_embedding": float(torch.norm(act_table.embedding.weight.grad).item()) if act_table.embedding.weight.grad is not None else 0.0,
    }
    
    # Check model params
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[f"grad_norm_{name}"] = float(torch.norm(p.grad).item())

    return grads


# ==============================================================================
# Main Phase H Runner
# ==============================================================================
def run_phase_h_experiment():
    print("=" * 80)
    print("      AEON JEPA-CRAFTER — PHASE H: SMALLEST ARCHITECTURAL FIX EXPERIMENT")
    print("=" * 80)

    # 1. System & Metadata Audit
    device = torch.device("cpu")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    checkpoint_path = os.path.join(aeon_root, "runs/jepa_crafter/checkpoints/checkpoint_00050000.pt")
    dataset_path = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/dataset_256_transitions.pt")
    output_dir = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation/phase_h")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase_h_results.json")

    print(f"• Git Commit SHA: {git_sha}")
    print(f"• Baseline Checkpoint: {checkpoint_path}")
    print(f"• Dataset Source: {dataset_path}")
    print(f"• Results Output: {out_file}")

    # 2. Load Real Crafter Transitions Dataset
    dataset = torch.load(dataset_path)
    encoder = VisualEncoder(latent_dim=128).to(device)
    encoder.eval()

    def prepare_split(data):
        obs_t = data["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        obs_tp1 = data["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        acts = data["actions"].to(device)
        with torch.no_grad():
            zt, ut = encoder(obs_t)
            ztarg, _ = encoder(obs_tp1)
            ht = torch.cat([ut, ut], dim=-1)
        return zt, ht, acts, ztarg

    tr_zt, tr_ht, tr_actions, tr_ztarg = prepare_split(dataset["train"])
    ho_zt, ho_ht, ho_actions, ho_ztarg = prepare_split(dataset["heldout"])

    print(f"• Train Transitions: {tr_zt.size(0)} (Seeds: {dataset['train'].get('seeds', [101, 102])})")
    print(f"• Held-out Transitions: {ho_zt.size(0)} (Seeds: {dataset['heldout'].get('seeds', [103, 104])})")

    # 3. Persistence Baselines
    persist_train = compute_persistence_baseline(tr_zt, tr_ztarg)
    persist_heldout = compute_persistence_baseline(ho_zt, ho_ztarg)

    print("\n--- Persistence Baseline (Held-out) ---")
    print(f"  Total Error: {persist_heldout['transition_level']['total_error']:.6f}")
    print(f"  High-Change Transition Error: {persist_heldout['transition_level']['high_change_error']:.6f} (n={persist_heldout['transition_level']['num_high_change_transitions']})")
    print(f"  Low-Change Transition Error:  {persist_heldout['transition_level']['low_change_error']:.6f} (n={persist_heldout['transition_level']['num_low_change_transitions']})")
    print(f"  Dynamic Tokens Error:         {persist_heldout['region_level']['dynamic_tokens_error']:.6f} (n={persist_heldout['region_level']['num_dynamic_tokens']})")
    print(f"  Static Tokens Error:          {persist_heldout['region_level']['static_tokens_error']:.6f} (n={persist_heldout['region_level']['num_static_tokens']})")

    # 4. Run the 4 Authorized Variants across 3 Seeds
    seeds = [42, 43, 44]
    updates = 300
    lr = 3e-4

    variant_specs = [
        ("variant_a_baseline_control", VariantAPredictor, "Baseline Predictor (Control: Pre-LN Transformer + LayerNorm vs raw target)"),
        ("variant_b_norm_aligned", VariantBPredictor, "Normalization-Aligned Predictor (Pre-LN Transformer in raw target space, no LayerNorm)"),
        ("variant_c_residual", VariantCPredictor, "Residual Predictor with Baseline Normalization (LayerNorm(zt + delta))"),
        ("variant_d_residual_norm_aligned", VariantDPredictor, "Residual + Normalization-Aligned Predictor (zt + delta in raw space)"),
    ]

    all_results = {
        "metadata": {
            "git_sha": git_sha,
            "checkpoint_path": checkpoint_path,
            "dataset_path": dataset_path,
            "seeds": seeds,
            "updates_per_variant": updates,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss": "1 - CosineSimilarity",
            "fair_target_space": "Raw VisualEncoder token space [B, 64, 128] (mean ~0.64)",
            "action_advantage_convention": "shuffled_error - correct_error (positive means correct action is better)",
        },
        "persistence_baseline": {
            "train": persist_train,
            "heldout": persist_heldout,
        },
        "variants": {},
    }

    for key, model_cls, desc in variant_specs:
        print("\n" + "-" * 75)
        print(f"Running {key.upper()}: {desc}")
        print("-" * 75)

        seed_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_cls(latent_dim=128, context_dim=256, action_dim=64).to(device)
            act_table = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(act_table.parameters()), lr=lr
            )

            # Training loop
            model.train()
            act_table.train()
            train_losses = []
            for step in range(updates):
                optimizer.zero_grad()
                a_emb = act_table(tr_actions)
                pred = model(tr_zt, tr_ht, a_emb)
                pn = F.normalize(pred, dim=-1, eps=1e-6)
                tn = F.normalize(tr_ztarg, dim=-1, eps=1e-6)
                loss = (1.0 - (pn * tn).sum(dim=-1)).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            # Gradient flow check on seed 42
            grad_flow = None
            if seed == 42:
                grad_flow = audit_gradient_flow(model, act_table, tr_zt, tr_ht, tr_actions, tr_ztarg)

            # Evaluate on Train and Held-out
            eval_train = evaluate_variant(model, act_table, tr_zt, tr_ht, tr_actions, tr_ztarg)
            eval_ho = evaluate_variant(model, act_table, ho_zt, ho_ht, ho_actions, ho_ztarg)

            run_data = {
                "seed": seed,
                "final_train_loss": train_losses[-1],
                "initial_train_loss": train_losses[0],
                "train_metrics": eval_train,
                "heldout_metrics": eval_ho,
                "gradient_flow": grad_flow,
            }
            seed_runs.append(run_data)

            print(
                f"  Seed {seed} -> Heldout Total: {eval_ho['transition_level']['total_error']:.6f} "
                f"| High-Change: {eval_ho['transition_level']['high_change_error']:.6f} "
                f"| Dynamic: {eval_ho['region_level']['dynamic_tokens_error']:.6f} "
                f"| Adv (High): {eval_ho['transition_level']['action_advantage_high']:+.6f}"
            )

        # Aggregate across seeds
        def agg_field(metric_dict_list, path_keys):
            vals = []
            for item in metric_dict_list:
                curr = item
                for k in path_keys:
                    curr = curr[k]
                vals.append(vals if isinstance(curr, list) else curr)
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

        summary = {
            "heldout_total_error": agg_field(seed_runs, ["heldout_metrics", "transition_level", "total_error"]),
            "heldout_high_change_error": agg_field(seed_runs, ["heldout_metrics", "transition_level", "high_change_error"]),
            "heldout_low_change_error": agg_field(seed_runs, ["heldout_metrics", "transition_level", "low_change_error"]),
            "heldout_dynamic_tokens_error": agg_field(seed_runs, ["heldout_metrics", "region_level", "dynamic_tokens_error"]),
            "heldout_static_tokens_error": agg_field(seed_runs, ["heldout_metrics", "region_level", "static_tokens_error"]),
            "heldout_action_advantage_total": agg_field(seed_runs, ["heldout_metrics", "transition_level", "action_advantage_total"]),
            "heldout_action_advantage_high": agg_field(seed_runs, ["heldout_metrics", "transition_level", "action_advantage_high"]),
            "heldout_action_advantage_dynamic": agg_field(seed_runs, ["heldout_metrics", "region_level", "action_advantage_dynamic"]),
            "train_total_error": agg_field(seed_runs, ["train_metrics", "transition_level", "total_error"]),
            "train_loss_final": agg_field(seed_runs, ["final_train_loss"]),
            "effective_rank_centered_pr": agg_field(seed_runs, ["heldout_metrics", "representation_health", "effective_rank", "centered_pr"]),
            "effective_rank_centered_er": agg_field(seed_runs, ["heldout_metrics", "representation_health", "effective_rank", "centered_entropy_er"]),
            "pairwise_cosine_mean": agg_field(seed_runs, ["heldout_metrics", "representation_health", "pairwise_cosine", "mean_cosine_similarity"]),
        }

        # Compare against persistence
        ho_persist_tot = persist_heldout["transition_level"]["total_error"]
        ho_persist_high = persist_heldout["transition_level"]["high_change_error"]
        ho_persist_dyn = persist_heldout["region_level"]["dynamic_tokens_error"]

        beats_persist_total = summary["heldout_total_error"]["mean"] < ho_persist_tot
        beats_persist_high = summary["heldout_high_change_error"]["mean"] < ho_persist_high
        beats_persist_dyn = summary["heldout_dynamic_tokens_error"]["mean"] < ho_persist_dyn
        positive_adv_high = summary["heldout_action_advantage_high"]["mean"] > 0
        positive_adv_dyn = summary["heldout_action_advantage_dynamic"]["mean"] > 0

        summary["verdicts"] = {
            "beats_persistence_total": bool(beats_persist_total),
            "beats_persistence_high_change": bool(beats_persist_high),
            "beats_persistence_dynamic_tokens": bool(beats_persist_dyn),
            "positive_action_advantage_high": bool(positive_adv_high),
            "positive_action_advantage_dynamic": bool(positive_adv_dyn),
        }

        print(
            f"\n  Summary: Mean Total = {summary['heldout_total_error']['mean']:.6f} ± {summary['heldout_total_error']['std']:.6f} "
            f"(Persistence = {ho_persist_tot:.6f})"
        )
        print(
            f"  Mean High-Change = {summary['heldout_high_change_error']['mean']:.6f} ± {summary['heldout_high_change_error']['std']:.6f} "
            f"(Persistence = {ho_persist_high:.6f})"
        )
        print(
            f"  Mean Dynamic Tokens = {summary['heldout_dynamic_tokens_error']['mean']:.6f} ± {summary['heldout_dynamic_tokens_error']['std']:.6f} "
            f"(Persistence = {ho_persist_dyn:.6f})"
        )
        print(
            f"  Mean Action Adv (High) = {summary['heldout_action_advantage_high']['mean']:+.6f} ± {summary['heldout_action_advantage_high']['std']:.6f}"
        )
        print(
            f"  Beats Persistence: Total={beats_persist_total}, High={beats_persist_high}, Dynamic={beats_persist_dyn} | Pos Adv={positive_adv_high}"
        )

        all_results["variants"][key] = {
            "description": desc,
            "seed_runs": seed_runs,
            "summary": summary,
        }

    # 5. Save Results
    save_json(all_results, out_file)
    print("\n" + "=" * 80)
    print(f"AEON Phase H Experiment Complete. Results successfully written to:\n{out_file}")
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    run_phase_h_experiment()
