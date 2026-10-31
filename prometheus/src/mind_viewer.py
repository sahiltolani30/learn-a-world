"""
Mind Viewer — Visual X-Ray of the W1a World Model Brain.
Executes for N steps and saves exactly 6 diagnostic X-ray PNGs showing:
  [Panel 1] Real Environment Frame (Ground Truth)
  [Panel 2] Perception Mind-View (What the 64 Spatial Tokens See via recon_head)
  [Panel 3] Anticipated Future / Imagination (What the Predictor Dreams Will Happen)
  [Panel 4] Decision Intent (Policy Probabilities & Twin-Critic Q-Values)
Prints two concise summary lines upon completion.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
import crafter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent import JEPAAgent
from src.diagnostics import CRAFTER_ACTION_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Prometheus W1a Mind Viewer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained state.pt or checkpoint file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of steps to execute and visualize (default 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment seed (default 42)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save the 6 X-ray PNGs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda or cpu)",
    )
    return parser.parse_args()


def find_checkpoint(explicit_path: Optional[str] = None) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {explicit_path}")

    candidates = [
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v5/state.pt"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v5/state.pt"),
        ROOT_DIR / "checkpoints_v5/state.pt",
        Path("./checkpoints_v5/state.pt"),
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v4/state.pt"),
        ROOT_DIR / "checkpoints_v4/state.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    for d in [
        Path("/content/drive/MyDrive/learn-a-world/prometheus/checkpoints_v5"),
        Path("/content/drive/.shortcut-targets-by-id/18tW0Bzv4yPHA5kCgtSjS-ytI4IDUYJaa/learn-a-world/prometheus/checkpoints_v5"),
        ROOT_DIR / "checkpoints_v5",
        Path("./checkpoints_v5"),
    ]:
        if d.exists():
            pts = sorted(list(d.glob("*.pt")), key=lambda x: x.stat().st_mtime, reverse=True)
            if pts:
                return pts[0]

    raise FileNotFoundError("Could not locate any valid checkpoint.")


def render_xray_image(
    step: int,
    obs_np: np.ndarray,
    recon_np: np.ndarray,
    pred_np: np.ndarray,
    probs: np.ndarray,
    q_vals: np.ndarray,
    chosen_action: int,
    reward: float,
    total_return: float,
    info: Dict[str, Any],
    mem_norm: float,
    save_path: Path,
) -> None:
    """
    Renders and saves a high-resolution 4-panel mental X-ray PNG.
    """
    fig = plt.figure(figsize=(16, 4.2), dpi=120)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.8], wspace=0.25)

    # 1. Real World Observation
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(obs_np)
    ax0.set_title(f"1. Real Observation\n(Crafter Ground Truth @ Step {step})", fontsize=10, fontweight="bold")
    ax0.axis("off")

    # 2. Perception Mind-View (Decoded 64 Spatial Tokens)
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(np.clip(recon_np, 0.0, 1.0))
    ax1.set_title("2. What the Brain Sees\n(Decoded 64-Token Latent)", fontsize=10, fontweight="bold", color="#1f77b4")
    ax1.axis("off")

    # 3. 1-Step Anticipation / Dream (Predictor Rollout)
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(np.clip(pred_np, 0.0, 1.0))
    act_name = CRAFTER_ACTION_NAMES[chosen_action]
    ax2.set_title(f"3. What the Brain Imagines\n(Predicted Next Frame after '{act_name}')", fontsize=10, fontweight="bold", color="#2ca02c")
    ax2.axis("off")

    # 4. Decision Intent & Critic Valuation
    ax3 = fig.add_subplot(gs[3])
    y_pos = np.arange(17)
    colors = ["#1f77b4" if i != chosen_action else "#d62728" for i in range(17)]
    ax3.barh(y_pos, probs * 100, color=colors, alpha=0.85, height=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(CRAFTER_ACTION_NAMES, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Policy Probability \u03c0(a) (%)", fontsize=8)
    ax3.set_xlim(0, max(100.0, float(np.max(probs * 100)) * 1.25))

    for idx in range(17):
        p_val = probs[idx] * 100
        q_val = q_vals[idx]
        txt = f"{p_val:4.1f}% | Q:{q_val:+5.2f}"
        ax3.text(p_val + 1.5, idx, txt, va="center", fontsize=7, color="#222222")

    vitals = info.get("inventory", {})
    title_str = (
        f"Step {step} | Chosen: {act_name.upper()} | Return: {total_return:+.1f} | "
        f"HP:{vitals.get('health',9)} Food:{vitals.get('food',9)} Thirst:{vitals.get('drink',9)} Wood:{vitals.get('wood',0)}"
    )
    ax3.set_title(title_str, fontsize=9, fontweight="bold", pad=8)
    ax3.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt_path = find_checkpoint(args.checkpoint)
    raw_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    agent = JEPAAgent(latent_dim=256, num_actions=17, device=device)
    comp = raw_ckpt.get("components", {})
    if "agent" in comp:
        agent.load_state_dict(comp["agent"], strict=False)
    elif "encoder" in comp:
        agent.encoder.load_state_dict(comp["encoder"], strict=False)
    agent.eval()

    # Determine output directory for the 6 PNGs
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        drive_dir = Path("/content/drive/MyDrive/learn-a-world/prometheus/xray_images")
        if drive_dir.parent.exists():
            out_dir = drive_dir
        else:
            out_dir = Path("./mind_xray_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6 Target Snapshot Steps evenly distributed across the run
    # For 60 steps: steps [10, 20, 30, 40, 50, 60]
    total_steps = max(6, args.steps)
    target_snapshot_steps = set(np.linspace(10, total_steps, 6, dtype=int))
    if len(target_snapshot_steps) < 6:
        target_snapshot_steps = {10, 20, 30, 40, 50, 60}

    env = crafter.Env(seed=args.seed)
    obs = env.reset()
    done = False
    mem_h = torch.zeros(1, 256, device=device)
    total_return = 0.0

    recon_errors = []
    saved_images = []

    for step in range(1, total_steps + 1):
        if done:
            obs = env.reset()
            mem_h = torch.zeros(1, 256, device=device)
            done = False

        with torch.no_grad():
            x = agent.validate_and_prepare_obs(obs)
            if x.dim() == 3:
                x = x.unsqueeze(0)

            # 1. Perception Tokens & Reconstructed Mind-View
            tokens, pooled = agent.encoder.forward_tokens(x)
            recon = agent.encoder.decode_tokens(tokens)  # (1, 3, 64, 64)
            recon_mse = float(F.mse_loss(recon, x).item())
            recon_errors.append(recon_mse)

            recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # 2. Decision Intent & Critic
            readout = agent.get_readout(pooled, mem_h)
            logits = agent.actor(readout)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            q1, q2 = agent.critic(readout)
            q_vals = torch.min(q1, q2).squeeze(0).cpu().numpy()

            chosen_action = int(torch.argmax(torch.from_numpy(q_vals)).item())
            action_tensor = torch.tensor([chosen_action], device=device)

            # 3. 1-Step Future Anticipation
            pred_tokens, _ = agent.predictor.forward_tokens(tokens, action_tensor, h=mem_h)
            pred_frame = agent.encoder.decode_tokens(pred_tokens)
            pred_np = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # 4. Memory update
            mem_h = agent.gru_step(pooled, mem_h)
            mem_norm = float(mem_h.norm().item())

        obs_np = obs.copy() if hasattr(obs, "copy") else np.array(obs)
        obs, reward, done, info = env.step(chosen_action)
        total_return += float(reward)

        # Save X-Ray Image if this step is one of the 6 targets
        if step in target_snapshot_steps:
            save_path = out_dir / f"mind_xray_step_{step:03d}.png"
            render_xray_image(
                step=step,
                obs_np=obs_np,
                recon_np=recon_np,
                pred_np=pred_np,
                probs=probs,
                q_vals=q_vals,
                chosen_action=chosen_action,
                reward=float(reward),
                total_return=total_return,
                info=info,
                mem_norm=mem_norm,
                save_path=save_path,
            )
            saved_images.append(str(save_path.resolve()))

    # Compute overall visual metrics
    mean_recon_mse = float(np.mean(recon_errors))
    mean_tok_std = float(tokens.std(dim=-1).mean().item())

    # Exactly Two Summary Lines
    print(f"[Mind Viewer] Completed {total_steps} steps | Return: {total_return:+.2f} | Saved 6 X-ray PNGs to: {out_dir.resolve()}")
    print(f"[Mind Viewer] Reconstruction MSE: {mean_recon_mse:.4f} | Per-token feature std: {mean_tok_std:.4f} | 6 Images: {[Path(p).name for p in saved_images]}")


if __name__ == "__main__":
    main()
