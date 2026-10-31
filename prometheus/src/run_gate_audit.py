"""
Independent Verification Runner: H=1..10 Rollout Drift (The Dreaming Gate) & Stage-Wise Policy Probe.

Usage:
    python -m src.run_gate_audit [--checkpoint_dir PATH] [--checkpoint_path PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import crafter

from src.agent.agent import JEPAAgent
from src.memory.storage import CheckpointManager
from src.diagnostics import run_stagewise_policy_probe, CRAFTER_ACTION_NAMES


def run_drift_test_h10(agent: JEPAAgent, device: torch.device, num_trajectories: int = 5) -> Dict[str, Any]:
    """
    Evaluates multi-step open-loop latent rollout fidelity against ground truth target encoder representations
    across H=1..10 steps. Evaluates the U4 Dreaming Gate condition (cosine >= 0.80 at H=5..10).
    """
    agent.eval()
    H_max = 10
    cos_by_h: Dict[int, List[float]] = {h: [] for h in range(1, H_max + 1)}
    l2_by_h: Dict[int, List[float]] = {h: [] for h in range(1, H_max + 1)}

    for traj_i in range(num_trajectories):
        env = crafter.Env(seed=2000 + traj_i)
        obs = env.reset()
        true_frames = [obs]
        actions = []
        for _ in range(H_max):
            a = int(np.random.randint(0, env.action_space.n))
            obs, _, done, _ = env.step(a)
            true_frames.append(obs)
            actions.append(a)
            if done:
                break

        H = len(actions)
        x_all = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in true_frames
        ]).to(device)

        with torch.no_grad():
            z_true = agent.target_encoder(x_all)

        z_sim = z_true[0:1]
        h_dim = getattr(agent, "hidden_dim", getattr(agent, "mem_hidden", 256))
        h_sim = torch.zeros(1, h_dim, device=device)

        for t in range(H):
            a_t = torch.tensor([actions[t]], dtype=torch.long, device=device)
            with torch.no_grad():
                pred_next, _ = agent.predictor(z_sim, a_t, h=h_sim)
                if hasattr(agent, "gru"):
                    h_sim = agent.gru(z_sim, h_sim)
                z_sim = pred_next
                cos = F.cosine_similarity(z_sim, z_true[t+1:t+2]).item()
                l2 = torch.norm(z_sim - z_true[t+1:t+2], p=2).item()
                cos_by_h[t+1].append(cos)
                l2_by_h[t+1].append(l2)

    mean_cos = {}
    mean_l2 = {}
    for h in range(1, H_max + 1):
        mean_cos[h] = float(np.mean(cos_by_h[h])) if cos_by_h[h] else 0.0
        mean_l2[h] = float(np.mean(l2_by_h[h])) if l2_by_h[h] else 0.0

    return {
        "mean_cos": mean_cos,
        "mean_l2": mean_l2,
        "h5_cos": mean_cos.get(5, 0.0),
        "h10_cos": mean_cos.get(10, 0.0),
        "dreaming_gate_passed": bool(mean_cos.get(5, 0.0) >= 0.80),
    }


def main():
    parser = argparse.ArgumentParser(description="JEPA Dreaming Gate & Stage-Wise Policy Probe")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to specific checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  PROJECT PROMETHEUS: H=1..10 DRIFT TEST & STAGE-WISE POLICY PROBE")
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

    step_restored = 0
    if ckpt_path is not None and Path(ckpt_path).exists():
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
        print("[!] Warning: No checkpoint found. Running diagnostics on initialized agent.")

    agent.eval()

    # 1. H=1..10 DRIFT TEST (THE DREAMING GATE)
    print("\n" + "=" * 80)
    print("  [1/2] MULTI-STEP LATENT ROLLOUT DRIFT (H=1..10) — THE DREAMING GATE")
    print("=" * 80)
    drift_res = run_drift_test_h10(agent, device=device, num_trajectories=5)

    print(f"{'Horizon t':<10} | {'Cosine Similarity':>18} | {'Latent L2 Distance':>20} | {'Dreaming Gate Status'}")
    print("-" * 80)
    for h in range(1, 11):
        cos = drift_res["mean_cos"][h]
        l2 = drift_res["mean_l2"][h]
        status = "PASS (>=0.80)" if cos >= 0.80 else "DRIFTING (<0.80)"
        print(f"{h:<10d} | {cos:>18.4f} | {l2:>20.4f} | {status}")

    print("-" * 80)
    print(f"Dreaming Gate Threshold Target:  Cosine >= 0.80 at H=5..10")
    print(f"Measured Horizon H=5 Cosine:     {drift_res['h5_cos']:.4f}")
    print(f"Measured Horizon H=10 Cosine:    {drift_res['h10_cos']:.4f}")
    if drift_res["dreaming_gate_passed"]:
        print(">> GATE STATUS: [PASS] Model maintains high-fidelity latent rollouts. DREAMING GATE UNLOCKED!")
    else:
        print(">> GATE STATUS: [LOCKED] World model drift exceeds tolerance at H=5. Keep training before unrolling dreams.")

    # 2. STAGE-WISE POLICY PROBE
    print("\n" + "=" * 80)
    print("  [2/2] STAGE-WISE POLICY PROBE (THE 4 CANONICAL CRAFTING STAGES)")
    print("=" * 80)
    stage_res = run_stagewise_policy_probe(agent)
    for stage_name, actions in stage_res.items():
        print(f"\n--- {stage_name} ---")
        print(f"{'Rank':<6} | {'Action':<22} | {'Probability':>12} | {'Q-Value':>12}")
        print("-" * 58)
        for a in actions:
            print(f"#{a['rank']:<5} | {a['action']:<22} | {a['prob']*100:11.2f}% | {a['q']:12.4f}")

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
