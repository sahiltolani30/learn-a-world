"""
Spatial Masked Prediction Probe (U1 Verification).

Tests and visualizes the model's ability to predict hidden/masked spatial tiles on the 8x8 map grid.

Usage:
    PYTHONPATH=/content/learn-a-world/prometheus python3 -m src.probe_masked_prediction [--checkpoint_dir PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import crafter

from src.agent.agent import JEPAAgent
from src.memory.storage import CheckpointManager
from src.models.predictor import sample_token_mask
from src.diagnostics import CRAFTER_ACTION_NAMES


def evaluate_masked_prediction(
    agent: JEPAAgent,
    device: torch.device,
    seed: int = 42,
    action: int = 1,
    mask_ratio: float = 0.35,
) -> Dict[str, Any]:
    """
    Steps an environment with an action, masks 35% of the spatial tiles on the next state,
    and measures the model's latent prediction accuracy on the hidden tiles.
    """
    agent.eval()

    env = crafter.Env(seed=seed)
    obs_t = env.reset()
    obs_next, reward, done, info = env.step(action)

    x_t = torch.from_numpy(obs_t).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    x_next = torch.from_numpy(obs_next).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

    with torch.no_grad():
        tokens_t, _ = agent.encoder.forward_tokens(x_t)
        target_tokens, _ = agent.target_encoder.forward_tokens(x_next)

        a_tensor = torch.tensor([action], dtype=torch.long, device=device)
        h_dim = getattr(agent, "hidden_dim", getattr(agent, "mem_hidden", 256))
        h_zero = torch.zeros(1, h_dim, device=device)

        mean_pred_tokens, all_pred_tokens = agent.predictor.forward_tokens(tokens_t, a_tensor, h=h_zero)

    # Sample 35% mask across 64 spatial tokens
    mask = sample_token_mask(1, 64, mask_ratio=mask_ratio, device=device)
    mask_np = mask[0].cpu().numpy()

    # Per-tile cosine similarity and MSE
    cos_sims = F.cosine_similarity(mean_pred_tokens[0], target_tokens[0], dim=-1).cpu().numpy()
    mse_diffs = ((mean_pred_tokens[0] - target_tokens[0]) ** 2).mean(dim=-1).cpu().numpy()

    cos_masked = cos_sims[mask_np]
    cos_unmasked = cos_sims[~mask_np]
    mse_masked = mse_diffs[mask_np]
    mse_unmasked = mse_diffs[~mask_np]

    return {
        "obs_t": obs_t,
        "obs_next": obs_next,
        "action": action,
        "action_name": CRAFTER_ACTION_NAMES[action],
        "mask_np": mask_np,
        "cos_sims": cos_sims,
        "mse_diffs": mse_diffs,
        "mean_cos_masked": float(cos_masked.mean()),
        "std_cos_masked": float(cos_masked.std()),
        "mean_cos_unmasked": float(cos_unmasked.mean()),
        "mean_mse_masked": float(mse_masked.mean()),
        "mean_mse_unmasked": float(mse_unmasked.mean()),
        "overall_cos": float(cos_sims.mean()),
        "masked_count": int(mask_np.sum()),
        "total_tokens": len(mask_np),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe Spatial Masked Token Prediction (U1)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--save_plot", action="store_true", default=True, help="Save visual comparison figure")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  PROJECT PROMETHEUS: SPATIAL MASKED TOKEN PREDICTION PROBE (U1)")
    print("=" * 80)
    print(f"Device: {device}")

    # Resolve checkpoint
    ckpt_path = None
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        candidate_dirs = []
        if args.checkpoint_dir:
            candidate_dirs.append(Path(args.checkpoint_dir))
        candidate_dirs.extend([
            Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v3"),
            Path("./checkpoints_v3"),
            Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v3"),
        ])
        for d in candidate_dirs:
            if d.exists():
                mgr = CheckpointManager(checkpoint_dir=d)
                found = mgr.find_latest_checkpoint()
                if found:
                    ckpt_path = Path(found)
                    break

    agent = JEPAAgent(device=device)

    if ckpt_path and Path(ckpt_path).exists():
        ckpt_path = Path(ckpt_path)
        print(f"[+] Restoring agent weights from checkpoint: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        components = state.get("components", {})
        if "agent" in components:
            agent.load_state_dict(components["agent"].state_dict() if hasattr(components["agent"], "state_dict") else components["agent"])
        elif "model" in components:
            agent.load_state_dict(components["model"].state_dict() if hasattr(components["model"], "state_dict") else components["model"])
        metadata = state.get("metadata", {})
        step_restored = metadata.get("step", 0)
        print(f"[+] Successfully loaded checkpoint from step {step_restored:,}")
    else:
        print("[!] Note: Running probe on newly initialized agent (pass --checkpoint_dir for trained checkpoint).")

    agent.eval()

    # Test actions: 1=move_left, 16=do
    for test_action in [1, 16]:
        act_name = CRAFTER_ACTION_NAMES[test_action]
        print("\n" + "=" * 80)
        print(f"  TESTING ACTION: '{act_name}' (Action #{test_action}) — 35% MASKED PREDICTION")
        print("=" * 80)

        res = evaluate_masked_prediction(agent, device=device, seed=42, action=test_action, mask_ratio=0.35)

        print(f"Total Spatial Grid Tiles:     {res['total_tokens']} (8x8 Grid)")
        print(f"Masked (Hidden) Tiles:        {res['masked_count']} / 64 (35.0%)")
        print(f"Visible (Context) Tiles:      {64 - res['masked_count']} / 64 (65.0%)")
        print("-" * 80)
        print(f"Hidden Tile Cosine Fidelity:  {res['mean_cos_masked']:.4f} +/- {res['std_cos_masked']:.4f}")
        print(f"Visible Tile Cosine Fidelity: {res['mean_cos_unmasked']:.4f}")
        print(f"Hidden Tile MSE Loss:         {res['mean_mse_masked']:.6f}")
        print(f"Overall Map Cosine Fidelity:  {res['overall_cos']:.4f}")
        print("-" * 80)

        # 8x8 Spatial Tile Grid
        grid_cos = res["cos_sims"].reshape(8, 8)
        grid_mask = res["mask_np"].reshape(8, 8)

        print("Visual 8x8 Spatial Tile Map [Tag: Cosine Fidelity]:")
        print("  Legend: [M:x.xx] = Hidden / Masked Tile  |  [V:x.xx] = Visible Context Tile")
        print("  " + "-" * 70)
        for r in range(8):
            row_items = []
            for c in range(8):
                tag = "M" if grid_mask[r, c] else "V"
                row_items.append(f"[{tag}:{grid_cos[r, c]:.2f}]")
            print(f"  Row {r}: " + " ".join(row_items))

        # Check U1 prediction status
        if res["mean_cos_masked"] >= 0.85:
            print(f"\n>> VERDICT: [EXCELLENT] Model predicts hidden map tiles with high fidelity ({res['mean_cos_masked']:.4f} >= 0.85)!")
        elif res["mean_cos_masked"] >= 0.70:
            print(f"\n>> VERDICT: [PASS] Model learns meaningful tile transition dynamics ({res['mean_cos_masked']:.4f} >= 0.70).")
        else:
            print(f"\n>> VERDICT: [LOW] Hidden tile prediction is under-trained ({res['mean_cos_masked']:.4f} < 0.70).")

    # Save visual plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        res_plot = evaluate_masked_prediction(agent, device=device, seed=42, action=16, mask_ratio=0.35)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(res_plot["obs_t"])
        axes[0].set_title("Input Frame (t)\n[Observation]", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(res_plot["obs_next"])
        axes[1].set_title("Target Next Frame (t+1)\n[Action: 'do']", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 35% Mask Overlay
        mask_2d = res_plot["mask_np"].reshape(8, 8).astype(float)
        axes[2].imshow(mask_2d, cmap="gray_r", interpolation="nearest")
        axes[2].set_title(f"35% Spatial Mask Grid\n[{res_plot['masked_count']} Hidden Tiles = Black]", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Cosine Fidelity Heatmap
        cos_2d = res_plot["cos_sims"].reshape(8, 8)
        im = axes[3].imshow(cos_2d, cmap="viridis", vmin=0.5, vmax=1.0)
        axes[3].set_title(f"Predicted Tile Fidelity\n[Masked Cosine: {res_plot['mean_cos_masked']:.4f}]", fontsize=11, fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.suptitle("Project Prometheus: Spatial Masked Token Prediction Probe (U1)", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        plot_path = Path("spatial_masked_prediction_probe.png")
        plt.savefig(str(plot_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"\n[+] Saved high-resolution 4-panel visual figure to: {plot_path.resolve()}")
    except Exception as e:
        print(f"[!] Plot generation note: {e}")

    print("\n" + "=" * 80)
    print("  PROBE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
