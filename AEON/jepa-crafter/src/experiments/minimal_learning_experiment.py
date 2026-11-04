"""Phase C/D Minimal Controlled Learning Experiment.

Compares:
  Variant A: Existing Baseline Predictor (FiLM modulation only, sequence length 64)
  Variant B: Diagnostic Action-Token Predictor (FiLM + explicit Action Token in Transformer, sequence length 65)

On a fixed, verified batch of 16 real Crafter environment transitions.
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
from src.env.opaque_actions import OpaqueActionMapping
from src.models.visual_encoder import VisualEncoder
from src.models.action_embedding import ActionEmbedding
from src.models.latent_predictor import LatentPredictorMember
from src.utils.logging import save_json


# ==============================================================================
# Variant B: Isolated Diagnostic Action-Token Predictor
# ==============================================================================
class DiagnosticActionTokenPredictorMember(nn.Module):
    """Isolated experimental variant prepending an explicit action token to the Transformer sequence."""

    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 256,
        action_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Identical conditioning MLP
        cond_in = context_dim + action_dim  # 320
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )

        # Explicit action sequence token projection: A -> D
        self.action_token_proj = nn.Linear(action_dim, latent_dim)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        current_tokens: torch.Tensor,
        context_state: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with explicit action token prepended to sequence."""
        # 1. FiLM condition projection
        cond_input = torch.cat([context_state, action_emb], dim=-1)  # [B, 320]
        film_params = self.cond_mlp(cond_input)                       # [B, 256]
        gamma, beta = film_params.chunk(2, dim=-1)                   # [B, 128], [B, 128]
        gamma = gamma.unsqueeze(1)                                    # [B, 1, 128]
        beta = beta.unsqueeze(1)                                      # [B, 1, 128]

        modulated = (1.0 + gamma) * current_tokens + beta            # [B, 64, 128]

        # 2. Prepend explicit Action Token into Transformer sequence
        a_token = self.action_token_proj(action_emb).unsqueeze(1)     # [B, 1, 128]
        seq = torch.cat([a_token, modulated], dim=1)                 # [B, 65, 128]

        # 3. Transformer self-attention across 65 tokens
        out_seq = self.transformer(seq)                              # [B, 65, 128]

        # 4. Strictly slice spatial tokens [1..64] and exclude action token [0]
        out_tokens = out_seq[:, 1:, :]                               # [B, 64, 128]
        out = self.final_norm(out_tokens)
        return out


# ==============================================================================
# Step 1: Real Transition Extraction & Verification
# ==============================================================================
def extract_verified_diagnostic_batch() -> Dict[str, Any]:
    """Collect 16 verified real Crafter transitions from deterministic seed 42."""
    adapter = CrafterAdapter(seed=42)
    obs = adapter.reset()

    transitions = []
    # Step through 20 steps to get a rich sample of motion and interactions
    for t in range(20):
        # Semi-structured action sequence: movement, turning, interactions
        act_token = (t * 3 + 1) % 17
        next_obs, r, term, trunc = adapter.step(act_token)
        pixel_diff = float(np.abs(next_obs.astype(float) - obs.astype(float)).mean())
        transitions.append({
            "step": t,
            "obs_t": obs.copy(),
            "action": act_token,
            "obs_tp1": next_obs.copy(),
            "reward": float(r),
            "terminated": bool(term),
            "pixel_diff": pixel_diff,
        })
        obs = next_obs
        if term:
            break
    adapter.close()

    # Select exactly 16 transitions
    selected = transitions[:16]
    assert len(selected) == 16, f"Expected 16 transitions, got {len(selected)}"

    batch_obs_t = np.stack([t["obs_t"] for t in selected], axis=0)      # [16, 64, 64, 3] uint8
    batch_actions = np.array([t["action"] for t in selected], dtype=np.int64) # [16]
    batch_obs_tp1 = np.stack([t["obs_tp1"] for t in selected], axis=0)  # [16, 64, 64, 3] uint8
    batch_rewards = np.array([t["reward"] for t in selected], dtype=np.float32)
    batch_terminals = np.array([t["terminated"] for t in selected], dtype=bool)
    pixel_diffs = [t["pixel_diff"] for t in selected]

    return {
        "obs_t": torch.from_numpy(batch_obs_t),
        "actions": torch.from_numpy(batch_actions),
        "obs_tp1": torch.from_numpy(batch_obs_tp1),
        "rewards": torch.from_numpy(batch_rewards),
        "terminals": torch.from_numpy(batch_terminals),
        "pixel_diffs": pixel_diffs,
    }


# ==============================================================================
# Step 2: Evaluation Metrics Calculator
# ==============================================================================
def evaluate_variant(
    model: nn.Module,
    act_table: nn.Module,
    z_t: torch.Tensor,
    h_t: torch.Tensor,
    actions: torch.Tensor,
    z_target: torch.Tensor,
) -> Dict[str, Any]:
    """Compute all required metrics deterministically."""
    model.eval()
    B, N, D = z_t.shape

    with torch.no_grad():
        a_emb = act_table(actions)
        pred = model(z_t, h_t, a_emb)

        # 1. Cosine distance per item: 1 - mean_tokens(cos_sim)
        p_norm = F.normalize(pred, dim=-1)
        t_norm = F.normalize(z_target, dim=-1)
        s_norm = F.normalize(z_t, dim=-1)

        cos_dist_per_item = (1.0 - (p_norm * t_norm).sum(dim=-1)).mean(dim=-1) # [B]
        persist_cos_per_item = (1.0 - (s_norm * t_norm).sum(dim=-1)).mean(dim=-1) # [B]

        mean_cos_dist = float(cos_dist_per_item.mean().item())
        mean_persist_cos = float(persist_cos_per_item.mean().item())
        pct_imp_cos = ((mean_persist_cos - mean_cos_dist) / max(1e-8, mean_persist_cos)) * 100.0

        # 2. Raw L2 distance
        l2_dist_per_item = torch.norm(pred - z_target, p=2, dim=-1).mean(dim=-1) # [B]
        persist_l2_per_item = torch.norm(z_t - z_target, p=2, dim=-1).mean(dim=-1) # [B]

        mean_l2_dist = float(l2_dist_per_item.mean().item())
        mean_persist_l2 = float(persist_l2_per_item.mean().item())
        pct_imp_l2 = ((mean_persist_l2 - mean_l2_dist) / max(1e-8, mean_persist_l2)) * 100.0

        # 3. Shuffled Action Control
        perm = torch.roll(torch.arange(B, device=actions.device), shifts=1)
        shuffled_actions = actions[perm]
        a_emb_shuffled = act_table(shuffled_actions)
        pred_shuffled = model(z_t, h_t, a_emb_shuffled)
        ps_norm = F.normalize(pred_shuffled, dim=-1)
        shuffled_cos = float((1.0 - (ps_norm * t_norm).sum(dim=-1)).mean().item())

        # 4. Zero Action Control
        a_emb_zero = torch.zeros_like(a_emb)
        pred_zero = model(z_t, h_t, a_emb_zero)
        pz_norm = F.normalize(pred_zero, dim=-1)
        zero_cos = float((1.0 - (pz_norm * t_norm).sum(dim=-1)).mean().item())

        # 5. Action Sensitivity / Prediction Divergence across all 17 actions
        fixed_z = z_t[:1]
        fixed_h = h_t[:1]
        div_outputs = []
        for a in range(17):
            a_vec = act_table(torch.tensor([a], device=actions.device))
            out_a = model(fixed_z, fixed_h, a_vec)
            div_outputs.append(out_a)

        pairwise_cos = []
        pairwise_l2 = []
        for i in range(17):
            for j in range(i + 1, 17):
                p_i = F.normalize(div_outputs[i], dim=-1)
                p_j = F.normalize(div_outputs[j], dim=-1)
                cd = (1.0 - (p_i * p_j).sum(dim=-1)).mean().item()
                l2 = torch.norm(div_outputs[i] - div_outputs[j], p=2).item()
                pairwise_cos.append(cd)
                pairwise_l2.append(l2)

        action_div_cos = float(np.mean(pairwise_cos))
        action_div_l2 = float(np.mean(pairwise_l2))

        # 6. Latent Norm and Variance
        latent_norm = float(torch.norm(pred, p=2, dim=-1).mean().item())
        flat_p = pred.reshape(B * N, D)
        latent_var = float(flat_p.var(dim=0).mean().item())

    model.train()
    return {
        "mean_cos_dist": mean_cos_dist,
        "mean_persist_cos": mean_persist_cos,
        "pct_imp_cos": pct_imp_cos,
        "mean_l2_dist": mean_l2_dist,
        "mean_persist_l2": mean_persist_l2,
        "pct_imp_l2": pct_imp_l2,
        "shuffled_cos_dist": shuffled_cos,
        "zero_cos_dist": zero_cos,
        "action_divergence_cos": action_div_cos,
        "action_divergence_l2": action_div_l2,
        "latent_norm": latent_norm,
        "latent_var": latent_var,
        "per_item_cos_dist": cos_dist_per_item.cpu().numpy().tolist(),
        "per_item_persist_cos": persist_cos_per_item.cpu().numpy().tolist(),
        "per_item_beats_persist": (cos_dist_per_item < persist_cos_per_item).cpu().numpy().tolist(),
    }


# ==============================================================================
# Step 3: Run Controlled Learning Experiment
# ==============================================================================
def run_experiment() -> Dict[str, Any]:
    print("=" * 75)
    print("      AEON JEPA-CRAFTER: PHASE C/D MINIMAL LEARNING EXPERIMENT")
    print("=" * 75)

    device = torch.device("cpu")

    # 1. Extract Real Diagnostic Batch
    batch = extract_verified_diagnostic_batch()
    print("\n[1/4] Extracted 16 Real Crafter Transitions from Seed 42:")
    for i in range(16):
        print(f"  Item {i:2d}: Action {batch['actions'][i]:2d}, Pixel Delta: {batch['pixel_diffs'][i]:.3f}, Reward: {batch['rewards'][i]:.2f}")

    # Prepare visual inputs: [16, 3, 64, 64] float32 in [0, 1]
    obs_t_ch = batch["obs_t"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    obs_tp1_ch = batch["obs_tp1"].permute(0, 3, 1, 2).float().div_(255.0).to(device)
    actions = batch["actions"].to(device)

    # 2. Compute Common Visual Latents from Fixed Visual Encoder
    torch.manual_seed(12345)
    encoder = VisualEncoder(latent_dim=128)
    encoder.eval()
    with torch.no_grad():
        z_t, u_t = encoder(obs_t_ch)         # [16, 64, 128], [16, 128]
        z_target, _ = encoder(obs_tp1_ch)    # [16, 64, 128]
        # Simulate temporal context vector [16, 256]
        h_t = torch.cat([u_t, u_t], dim=-1)  # [16, 256]

    print(f"\n[2/4] Encoded Latents: z_t shape: {z_t.shape}, z_target shape: {z_target.shape}")

    # Persistence baseline statistics on this batch
    s_norm = F.normalize(z_t, dim=-1)
    t_norm = F.normalize(z_target, dim=-1)
    batch_persist_cos = (1.0 - (s_norm * t_norm).sum(dim=-1)).mean(dim=-1)
    print(f"• Fixed Batch Persistence Cosine Distance (Mean): {batch_persist_cos.mean().item():.6f}")

    # 3. Initialize Both Variants with Controlled Parameters
    torch.manual_seed(42)
    act_table_A = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
    model_A = LatentPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)

    # For Variant B: copy identical initial weights for shared components
    torch.manual_seed(42)
    act_table_B = ActionEmbedding(num_actions=17, embedding_dim=64).to(device)
    act_table_B.load_state_dict(act_table_A.state_dict())
    model_B = DiagnosticActionTokenPredictorMember(latent_dim=128, context_dim=256, action_dim=64).to(device)
    model_B.cond_mlp.load_state_dict(model_A.cond_mlp.state_dict())
    model_B.transformer.load_state_dict(model_A.transformer.state_dict())
    model_B.final_norm.load_state_dict(model_A.final_norm.state_dict())

    # Verify identical initial state
    with torch.no_grad():
        m_A_0 = evaluate_variant(model_A, act_table_A, z_t, h_t, actions, z_target)
        m_B_0 = evaluate_variant(model_B, act_table_B, z_t, h_t, actions, z_target)

    print("\n[3/4] Initial State (Update 0):")
    print(f"  • Variant A (FiLM Only)       : Cos Dist = {m_A_0['mean_cos_dist']:.6f}, L2 = {m_A_0['mean_l2_dist']:.4f}, Action Div = {m_A_0['action_divergence_cos']:.6f}")
    print(f"  • Variant B (FiLM + ActionTok): Cos Dist = {m_B_0['mean_cos_dist']:.6f}, L2 = {m_B_0['mean_l2_dist']:.4f}, Action Div = {m_B_0['action_divergence_cos']:.6f}")

    # 4. Train Both Models on Identical Batch for Exactly 100 Updates
    opt_A = torch.optim.AdamW(list(model_A.parameters()) + list(act_table_A.parameters()), lr=3e-4, weight_decay=1e-4)
    opt_B = torch.optim.AdamW(list(model_B.parameters()) + list(act_table_B.parameters()), lr=3e-4, weight_decay=1e-4)

    history_A = {0: m_A_0}
    history_B = {0: m_B_0}
    updates_to_log = [1, 10, 50, 100]

    for step in range(1, 101):
        # Update Variant A
        opt_A.zero_grad()
        a_emb_A = act_table_A(actions)
        pred_A = model_A(z_t, h_t, a_emb_A)
        pA_norm = F.normalize(pred_A, dim=-1)
        loss_A = (1.0 - (pA_norm * t_norm).sum(dim=-1)).mean()
        loss_A.backward()
        grad_norm_A = float(nn.utils.clip_grad_norm_(model_A.parameters(), 1.0).item())
        opt_A.step()

        # Update Variant B
        opt_B.zero_grad()
        a_emb_B = act_table_B(actions)
        pred_B = model_B(z_t, h_t, a_emb_B)
        pB_norm = F.normalize(pred_B, dim=-1)
        loss_B = (1.0 - (pB_norm * t_norm).sum(dim=-1)).mean()
        loss_B.backward()
        grad_norm_B = float(nn.utils.clip_grad_norm_(model_B.parameters(), 1.0).item())
        opt_B.step()

        if step in updates_to_log:
            metrics_A = evaluate_variant(model_A, act_table_A, z_t, h_t, actions, z_target)
            metrics_A["train_loss"] = float(loss_A.item())
            metrics_A["grad_norm"] = grad_norm_A
            history_A[step] = metrics_A

            metrics_B = evaluate_variant(model_B, act_table_B, z_t, h_t, actions, z_target)
            metrics_B["train_loss"] = float(loss_B.item())
            metrics_B["grad_norm"] = grad_norm_B
            history_B[step] = metrics_B

            print(f"\n--- Update {step:3d} ---")
            print(f"  [Variant A - FiLM Only]        Loss: {loss_A.item():.6f} | Cos Err: {metrics_A['mean_cos_dist']:.6f} | Shuffled Err: {metrics_A['shuffled_cos_dist']:.6f} | vs Persist: {metrics_A['pct_imp_cos']:+.2f}% | Action Div: {metrics_A['action_divergence_cos']:.6f}")
            print(f"  [Variant B - FiLM + ActionTok] Loss: {loss_B.item():.6f} | Cos Err: {metrics_B['mean_cos_dist']:.6f} | Shuffled Err: {metrics_B['shuffled_cos_dist']:.6f} | vs Persist: {metrics_B['pct_imp_cos']:+.2f}% | Action Div: {metrics_B['action_divergence_cos']:.6f}")

    # Compile Final Artifacts
    output_dir = os.path.join(aeon_root, "artifacts/implementation/phase5_investigation")
    os.makedirs(output_dir, exist_ok=True)

    final_results = {
        "dataset": {
            "source": "CrafterReward-v1 (Seed 42)",
            "num_transitions": 16,
            "actions": batch["actions"].tolist(),
            "pixel_deltas": batch["pixel_diffs"],
            "persistence_cosine_distance": float(batch_persist_cos.mean().item()),
        },
        "history_variant_A": history_A,
        "history_variant_B": history_B,
        "step_100_per_transition": [
            {
                "index": i,
                "action": int(batch["actions"][i]),
                "pixel_delta": float(batch["pixel_diffs"][i]),
                "persistence_cos": float(batch_persist_cos[i].item()),
                "variant_A_cos": float(history_A[100]["per_item_cos_dist"][i]),
                "variant_A_beats_persist": bool(history_A[100]["per_item_beats_persist"][i]),
                "variant_B_cos": float(history_B[100]["per_item_cos_dist"][i]),
                "variant_B_beats_persist": bool(history_B[100]["per_item_beats_persist"][i]),
            }
            for i in range(16)
        ],
    }

    save_json(final_results, os.path.join(output_dir, "minimal_learning_experiment_results.json"))
    print(f"\n[4/4] Experiment complete! Artifacts saved to: {os.path.join(output_dir, 'minimal_learning_experiment_results.json')}")
    print("=" * 75)
    return final_results


if __name__ == "__main__":
    run_experiment()
