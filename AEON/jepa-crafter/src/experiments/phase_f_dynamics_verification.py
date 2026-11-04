"""Phase F: Verify Whether Dynamic Weighting Learns Useful Dynamics.

Protocol:
1. Collect 256 real Crafter transitions across 4 seeds:
   - Training split: Seeds 101 & 102 (128 transitions)
   - Held-out evaluation split: Seeds 103 & 104 (128 transitions)
2. Compare Variants A, B, C across 3 seeds (42, 43, 44) for 300 updates:
   - Variant A: Existing predictor + standard unweighted cosine loss
   - Variant B: Explicit action-token predictor + standard unweighted cosine loss
   - Variant C: Explicit action-token predictor + normalized dynamic-weighted loss
3. Log both Train and Held-Out metrics at updates 0, 10, 50, 100, 300:
   - Unweighted prediction error
   - Persistence error under exact same metric
   - High-change (dynamic) and Low-change (static) token errors (threshold = 0.001)
   - Shuffled action error & Action Advantage (Adv = E_shuffled - E_correct)
   - Action divergence across 17 tokens
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

from src.env.crafter_adapter import CrafterAdapter
from src.models.visual_encoder import VisualEncoder
from src.models.action_embedding import ActionEmbedding
from src.models.latent_predictor import LatentPredictorMember
from src.experiments.minimal_learning_experiment import DiagnosticActionTokenPredictorMember
from src.utils.logging import save_json


# ==============================================================================
# Step 1: Collect 256 Transitions (128 Train / 128 Held-Out)
# ==============================================================================
def collect_dataset_256() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collect 256 transitions from 4 independent seeds."""
    seeds_train = [101, 102]
    seeds_heldout = [103, 104]

    def collect_from_seeds(seed_list):
        obs_t_list, actions_list, obs_tp1_list, rewards_list, terms_list, diffs_list = [], [], [], [], [], []
        for s in seed_list:
            env = CrafterAdapter(seed=s)
            obs = env.reset()
            for t in range(64):
                # Pseudo-random deterministic action progression
                a = (t * 5 + 3) % 17
                next_obs, r, term, trunc = env.step(a)
                diff = float(np.abs(next_obs.astype(float) - obs.astype(float)).mean())

                obs_t_list.append(obs.copy())
                actions_list.append(a)
                obs_tp1_list.append(next_obs.copy())
                rewards_list.append(float(r))
                terms_list.append(bool(term))
                diffs_list.append(diff)

                obs = next_obs
                if term:
                    obs = env.reset()
            env.close()

        return {
            "obs_t": torch.from_numpy(np.stack(obs_t_list, axis=0)),       # [128, 64, 64, 3] uint8
            "actions": torch.tensor(actions_list, dtype=torch.long),        # [128]
            "obs_tp1": torch.from_numpy(np.stack(obs_tp1_list, axis=0)),   # [128, 64, 64, 3] uint8
            "rewards": torch.tensor(rewards_list, dtype=torch.float32),     # [128]
            "terminals": torch.tensor(terms_list, dtype=torch.bool),        # [128]
            "pixel_diffs": diffs_list,
        }

    train_data = collect_from_seeds(seeds_train)
    heldout_data = collect_from_seeds(seeds_heldout)
    return train_data, heldout_data


# ==============================================================================
# Step 2: Evaluation Function
# ==============================================================================
def evaluate_split(
    model: nn.Module,
    act_table: nn.Module,
    z_t: torch.Tensor,
    h_t: torch.Tensor,
    actions: torch.Tensor,
    z_target: torch.Tensor,
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """Evaluate model on a split under original unweighted cosine distance."""
    model.eval()
    B, N, D = z_t.shape
    device = z_t.device

    norm_zt = F.normalize(z_t, dim=-1, eps=1e-6)
    norm_ztarg = F.normalize(z_target, dim=-1, eps=1e-6)

    # Persistence metrics
    cos_persist = 1.0 - (norm_zt * norm_ztarg).sum(dim=-1) # [B, N]
    persist_mean = float(cos_persist.mean().item())

    low_mask = cos_persist < threshold
    high_mask = cos_persist >= threshold
    persist_low = float(cos_persist[low_mask].mean().item()) if low_mask.any() else 0.0
    persist_high = float(cos_persist[high_mask].mean().item()) if high_mask.any() else 0.0

    with torch.no_grad():
        # Correct action prediction
        a_emb = act_table(actions)
        pred = model(z_t, h_t, a_emb)
        pn = F.normalize(pred, dim=-1, eps=1e-6)
        cos_error_per_token = 1.0 - (pn * norm_ztarg).sum(dim=-1) # [B, N]
        eval_error = float(cos_error_per_token.mean().item())
        eval_low = float(cos_error_per_token[low_mask].mean().item()) if low_mask.any() else 0.0
        eval_high = float(cos_error_per_token[high_mask].mean().item()) if high_mask.any() else 0.0

        # Shuffled action prediction (derangement shift by 1)
        # Keeps obs, context, target, sample order 100% constant; changes ONLY action
        shuf_actions = torch.roll(actions, shifts=1)
        a_emb_shuf = act_table(shuf_actions)
        pred_shuf = model(z_t, h_t, a_emb_shuf)
        psn = F.normalize(pred_shuf, dim=-1, eps=1e-6)
        cos_shuf_per_token = 1.0 - (psn * norm_ztarg).sum(dim=-1)
        shuf_error = float(cos_shuf_per_token.mean().item())
        shuf_high = float(cos_shuf_per_token[high_mask].mean().item()) if high_mask.any() else 0.0

        # Action Advantage: positive means correct action is better than shuffled
        action_adv = shuf_error - eval_error
        action_adv_high = shuf_high - eval_high

        # Action Divergence across 17 actions on first context
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

    model.train()
    return {
        "unweighted_error": eval_error,
        "persistence_error": persist_mean,
        "low_change_error": eval_low,
        "low_change_persist": persist_low,
        "high_change_error": eval_high,
        "high_change_persist": persist_high,
        "shuffled_error": shuf_error,
        "action_advantage": action_adv,
        "action_advantage_high": action_adv_high,
        "action_divergence": action_div,
        "high_change_token_pct": float(high_mask.sum().item()) / float(B * N) * 100.0,
    }


# ==============================================================================
# Step 3: Master Execution Routine
# ==============================================================================
def main():
    print("=" * 75)
    print("      AEON JEPA-CRAFTER: PHASE F DYNAMICS VERIFICATION")
    print("=" * 75)

    device = torch.device("cpu")
    output_dir = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset Collection
    dataset_path = os.path.join(output_dir, "dataset_256_transitions.pt")
    if os.path.exists(dataset_path):
        print("\n[1/4] Loading existing 256-transition dataset...")
        data = torch.load(dataset_path, weights_only=False)
        train_raw, heldout_raw = data["train"], data["heldout"]
    else:
        print("\n[1/4] Collecting 256 real transitions across 4 Crafter seeds...")
        train_raw, heldout_raw = collect_dataset_256()
        torch.save({"train": train_raw, "heldout": heldout_raw}, dataset_path)
        print(f"• Saved 256 transitions (128 Train / 128 Held-Out) to: {dataset_path}")

    # Visual Encoder Latent Extraction (Frozen Encoder)
    torch.manual_seed(12345)
    encoder = VisualEncoder(latent_dim=128).to(device)
    encoder.eval()

    def encode_data(raw_dict):
        obs_t = raw_dict["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        obs_tp1 = raw_dict["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
        actions = raw_dict["actions"].to(device)
        with torch.no_grad():
            z_t, u_t = encoder(obs_t)
            z_target, _ = encoder(obs_tp1)
            h_t = torch.cat([u_t, u_t], dim=-1)
        return z_t, z_target, h_t, actions

    train_z_t, train_z_target, train_h_t, train_actions = encode_data(train_raw)
    heldout_z_t, heldout_z_target, heldout_h_t, heldout_actions = encode_data(heldout_raw)

    print(f"• Training Set: {train_z_t.size(0)} transitions | Held-Out Set: {heldout_z_t.size(0)} transitions")

    # Fixed threshold for dynamic tokens
    THRESHOLD = 0.001

    # Compute Normalized Dynamic Weights for Variant C on Training Set
    norm_tr_zt = F.normalize(train_z_t, dim=-1, eps=1e-6)
    norm_tr_ztarg = F.normalize(train_z_target, dim=-1, eps=1e-6)
    delta_persist = torch.norm(norm_tr_ztarg - norm_tr_zt, p=2, dim=-1) # [128, 64]
    max_delta = delta_persist.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
    w_raw = 1.0 + 2.0 * (delta_persist / max_delta) # [128, 64]
    w_normed = w_raw * (64.0 / w_raw.sum(dim=-1, keepdim=True)) # strictly mean == 1.0 per item
    assert torch.allclose(w_normed.mean(dim=-1), torch.ones(train_z_t.size(0), device=device), atol=1e-5)

    # 2. Optimization Comparison Across 3 Seeds
    seeds = [42, 43, 44]
    updates_to_log = [0, 10, 50, 100, 300]
    results_by_variant = {"Variant_A": [], "Variant_B": [], "Variant_C": []}

    print("\n[2/4] Training Variants A, B, C for 300 updates across 3 seeds...")

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
            for step in range(301):
                if step in updates_to_log:
                    train_metrics = evaluate_split(model, act_table, train_z_t, train_h_t, train_actions, train_z_target, threshold=THRESHOLD)
                    heldout_metrics = evaluate_split(model, act_table, heldout_z_t, heldout_h_t, heldout_actions, heldout_z_target, threshold=THRESHOLD)
                    history[step] = {"train": train_metrics, "heldout": heldout_metrics}

                if step < 300:
                    opt.zero_grad()
                    a_emb = act_table(train_actions)
                    pred = model(train_z_t, train_h_t, a_emb)
                    pn = F.normalize(pred, dim=-1, eps=1e-6)
                    raw_loss_per_token = 1.0 - (pn * norm_tr_ztarg).sum(dim=-1) # [128, 64]

                    if use_dynamic_weighting:
                        loss = (w_normed * raw_loss_per_token).mean()
                    else:
                        loss = raw_loss_per_token.mean()

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            seed_data[var_name] = history
            s300 = history[300]
            print(f"  {var_name:10s} (Seed {seed}) -> Step 300 Train Err: {s300['train']['unweighted_error']:.6f} | HeldOut Err: {s300['heldout']['unweighted_error']:.6f} | HeldOut High: {s300['heldout']['high_change_error']:.6f} | HeldOut Adv: {s300['heldout']['action_advantage']:+.6f}")

        for var_name in ["Variant_A", "Variant_B", "Variant_C"]:
            results_by_variant[var_name].append(seed_data[var_name])

    # 3. Aggregate Statistics Across Seeds
    summary_stats = {}
    for var_name, runs in results_by_variant.items():
        summary_stats[var_name] = {}
        for step in updates_to_log:
            summary_stats[var_name][step] = {}
            for split in ["train", "heldout"]:
                errs = [r[step][split]["unweighted_error"] for r in runs]
                highs = [r[step][split]["high_change_error"] for r in runs]
                lows = [r[step][split]["low_change_error"] for r in runs]
                advs = [r[step][split]["action_advantage"] for r in runs]
                advs_high = [r[step][split]["action_advantage_high"] for r in runs]
                divs = [r[step][split]["action_divergence"] for r in runs]

                summary_stats[var_name][step][split] = {
                    "error_mean": float(np.mean(errs)),
                    "error_std": float(np.std(errs)),
                    "high_change_mean": float(np.mean(highs)),
                    "high_change_std": float(np.std(highs)),
                    "low_change_mean": float(np.mean(lows)),
                    "low_change_std": float(np.std(lows)),
                    "action_adv_mean": float(np.mean(advs)),
                    "action_adv_std": float(np.std(advs)),
                    "action_adv_high_mean": float(np.mean(advs_high)),
                    "action_adv_high_std": float(np.std(advs_high)),
                    "action_div_mean": float(np.mean(divs)),
                    "action_div_std": float(np.std(divs)),
                }

    # Reference Persistence Baselines
    tr_ref = evaluate_split(model_A, act_A, train_z_t, train_h_t, train_actions, train_z_target, threshold=THRESHOLD)
    ho_ref = evaluate_split(model_A, act_A, heldout_z_t, heldout_h_t, heldout_actions, heldout_z_target, threshold=THRESHOLD)

    print("\n" + "=" * 75)
    print("      EXPERIMENT SUMMARY: HELD-OUT PERFORMANCE ACROSS 3 SEEDS")
    print("=" * 75)
    print(f"• Held-Out Persistence Baseline: Total = {ho_ref['persistence_error']:.6f} | High-Change = {ho_ref['high_change_persist']:.6f} | Low-Change = {ho_ref['low_change_persist']:.6f}")
    for var_name in ["Variant_A", "Variant_B", "Variant_C"]:
        s300_ho = summary_stats[var_name][300]["heldout"]
        s300_tr = summary_stats[var_name][300]["train"]
        print(f"\n[{var_name}]")
        print(f"  Step 300 Held-Out Error     : {s300_ho['error_mean']:.6f} ± {s300_ho['error_std']:.6f} (Train: {s300_tr['error_mean']:.6f})")
        print(f"  Step 300 Held-Out HighChange: {s300_ho['high_change_mean']:.6f} ± {s300_ho['high_change_std']:.6f}")
        print(f"  Step 300 Held-Out Action Adv: {s300_ho['action_adv_mean']:+.6f} ± {s300_ho['action_adv_std']:.6f} (High-Change Adv: {s300_ho['action_adv_high_mean']:+.6f})")
        print(f"  Step 300 Action Divergence  : {s300_ho['action_div_mean']:.8f} ± {s300_ho['action_div_std']:.8f}")

    final_payload = {
        "dataset_metadata": {
            "total_transitions": 256,
            "train_transitions": 128,
            "heldout_transitions": 128,
            "train_seeds": [101, 102],
            "heldout_seeds": [103, 104],
            "threshold": THRESHOLD,
            "train_persistence": tr_ref["persistence_error"],
            "heldout_persistence": ho_ref["persistence_error"],
            "heldout_high_change_persistence": ho_ref["high_change_persist"],
        },
        "seeds": seeds,
        "summary_stats": summary_stats,
    }

    out_file = os.path.join(output_dir, "phase_f_results.json")
    save_json(final_payload, out_file)
    print(f"\nAll Phase F experiments completed. Results saved to:\n  {out_file}")


if __name__ == "__main__":
    main()
