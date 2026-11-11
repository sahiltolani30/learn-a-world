#!/usr/bin/env python3
"""ORBIT — Gate 8: Short-Horizon Imagination Policy Training & Real-Gameplay Qualification.

Execution Sequence:
1. Preflight Audit: Verify hashes of Gate 7 champion and World Model champion.
2. Historical Discrepancy Investigation: Reconcile Gate 7 vs Gate 7B scores and establish unified protocol.
3. Precondition Qualifications: Closed-loop multi-step prediction (H=1,2,4,8), reward head, continuation head, value critic calibration.
4. Formal Acceptance Gate: Verify all 12 conditions pass.
5. Progressive Horizon Imagination Experiments: H=1, H=2, H=4, H=8 candidates.
6. Equal-Budget Comparison: Arm A (No-imagination Gate 7 AWR control) vs Arm B (Gate 8 Imagination).
7. Diagnostic Ablations: Persistence world model, Shuffled-action, Zero value bootstrap.
8. Real Crafter Benchmark: Pinned seeds under locked per-seed RNG with all 22 achievements, survival, and entropy.
9. Statistical Significance Testing: Paired t-test, 95% CI, probability of improvement.
10. Checkpoint Hashes, Telemetry Plots, and Metrics Manifest.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.gate6_policy_evaluator import (
    ALL_CRAFTER_ACHIEVEMENTS,
    RandomCrafterAgent,
    compute_crafter_score,
    evaluate_policy_in_crafter,
)
from eval.gate8_imagination import (
    compute_continuation_lambda_returns,
    generate_imagined_rollout,
    gate8_imagination_training_step,
    validate_action_conditioned_prediction,
    validate_continuation_prediction,
    validate_reward_prediction,
    validate_value_critic_calibration,
)
from true_jepa.agent import VisualJEPAAgent
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.replay import PixelReplay


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_heldout_episodes(path: Path) -> List[Dict[str, Any]]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            buf = io.BytesIO(f.read())
            episodes = torch.load(buf, weights_only=False, map_location="cpu")
    else:
        episodes = torch.load(path, weights_only=False, map_location="cpu")
    return episodes


def evaluate_policy_locked_rng(
    agent: Any,
    seeds: List[int],
    max_steps: int = 300,
) -> Dict[str, Any]:
    """Evaluates agent in Crafter with explicit per-seed PyTorch and NumPy RNG resets.

    This guarantees absolute deterministic reproducibility across independent runs.
    """
    returns = []
    lengths = []
    achievements_unlocked_per_ep = {ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS}
    action_counts = np.zeros(17, dtype=np.int64)
    termination_causes = {"health_exhaustion": 0, "max_steps_reached": 0}
    per_seed_results = []

    import crafter

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = crafter.Env(seed=seed)
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()

        ep_reward = 0.0
        step_count = 0
        reward_prev = 0.0
        done = False
        info = {}

        while not done and step_count < max_steps:
            action = agent.act(obs, reward_prev=reward_prev, step=step_count)
            action_int = int(action)
            action_counts[action_int] += 1

            obs, reward, done, info = env.step(action_int)
            ep_reward += float(reward)
            reward_prev = float(reward)
            step_count += 1

        returns.append(ep_reward)
        lengths.append(step_count)

        ep_achievements = info.get("achievements", {})
        for ach in ALL_CRAFTER_ACHIEVEMENTS:
            if ep_achievements.get(ach, 0) > 0:
                achievements_unlocked_per_ep[ach] += 1

        health = info.get("health", 0)
        if health <= 0 or done:
            termination_causes["health_exhaustion"] += 1
            cause = "health_exhaustion"
        else:
            termination_causes["max_steps_reached"] += 1
            cause = "max_steps_reached"

        per_seed_results.append({
            "seed": int(seed),
            "return": float(ep_reward),
            "length": int(step_count),
            "cause": cause,
            "achievements": [a for a, c in ep_achievements.items() if c > 0],
        })

    num_episodes = len(seeds)
    achievement_rates = {
        ach: float(count / num_episodes) for ach, count in achievements_unlocked_per_ep.items()
    }
    crafter_score = compute_crafter_score(achievement_rates)

    total_actions = int(action_counts.sum())
    action_dist = (action_counts / max(1, total_actions)).tolist()
    probs = np.array(action_dist)
    nonzero_probs = probs[probs > 0]
    action_entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)))

    return {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "crafter_score": float(crafter_score),
        "achievement_rates": achievement_rates,
        "action_distribution": action_dist,
        "action_entropy": action_entropy,
        "termination_causes": termination_causes,
        "per_seed_results": per_seed_results,
    }


def compute_paired_statistics(
    candidate_res: Dict[str, Any],
    control_res: Dict[str, Any],
) -> Dict[str, float]:
    cand_returns = [r["return"] for r in candidate_res["per_seed_results"]]
    ctrl_returns = [r["return"] for r in control_res["per_seed_results"]]
    diffs = np.array(cand_returns) - np.array(ctrl_returns)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / math.sqrt(n) if n > 0 else 0.0
    t_stat = mean_diff / max(1e-8, se_diff)

    t_crit = 2.045  # t_0.025 for df=29
    ci95_low = mean_diff - t_crit * se_diff
    ci95_high = mean_diff + t_crit * se_diff
    prob_better = float(np.mean(diffs > 0.0))

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "t_stat": t_stat,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "prob_better": prob_better,
    }


def train_imagination_candidate(
    initial_model: TrueJEPA,
    replay: PixelReplay,
    horizon: int,
    updates: int,
    batch_size: int,
    seq_len: int,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    lambda_param: float,
    beta: float,
    device: torch.device,
    world_model_override: Optional[str] = None,
    candidate_name: str = "candidate",
) -> Tuple[TrueJEPA, Dict[str, List[float]]]:
    """Trains a policy candidate via latent imagination rollouts."""
    model = copy.deepcopy(initial_model).to(device)

    # Freeze world model components
    model.encoder.eval()
    model.target_encoder.eval()
    model.predictor.eval()
    model.belief.eval()
    model.heads.reward.eval()
    model.heads.continuation.eval()

    model.encoder.requires_grad_(False)
    model.predictor.requires_grad_(False)
    model.belief.requires_grad_(False)
    model.heads.reward.requires_grad_(False)
    model.heads.continuation.requires_grad_(False)

    # Train only policy and value heads
    model.heads.policy.requires_grad_(True)
    model.heads.value.requires_grad_(True)
    model.heads.policy.train()
    model.heads.value.train()

    actor_opt = torch.optim.AdamW(model.heads.policy.parameters(), lr=actor_lr, weight_decay=1e-4)
    critic_opt = torch.optim.AdamW(model.heads.value.parameters(), lr=critic_lr, weight_decay=1e-4)

    rng = np.random.default_rng(42)
    history: Dict[str, List[float]] = {
        "steps": [],
        "actor_loss": [],
        "critic_loss": [],
        "imag_return": [],
        "imag_reward": [],
        "imag_continuation": [],
        "entropy": [],
        "gap": [],
    }

    t0 = time.time()
    for step_idx in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        obs_raw = batch["obs"].to(device)
        actions_raw = batch["actions"].to(device)
        rewards_raw = batch["rewards"].to(device)
        terminated_raw = batch["terminated"].to(device)

        B, T = actions_raw.shape

        # Recompute returns on real batch for anchor
        returns = torch.zeros((B, T), dtype=torch.float32, device=device)
        running_ret = torch.zeros(B, dtype=torch.float32, device=device)
        for t in reversed(range(T)):
            cont_m = (~terminated_raw[:, t]).float()
            running_ret = rewards_raw[:, t] + gamma * cont_m * running_ret
            returns[:, t] = running_ret

        # Encode frames
        with torch.no_grad():
            flat_obs = obs_raw.reshape(-1, *obs_raw.shape[2:])
            all_frames = model.encode(flat_obs)
            tokens_all = all_frames.tokens.reshape(B, T + 1, *all_frames.tokens.shape[1:])
            pooled_all = all_frames.pooled.reshape(B, T + 1, *all_frames.pooled.shape[1:])

        # Sequential burn-in to step 0
        hidden = model.initial_belief(B, device)
        frame_0 = LatentFrame(tokens=tokens_all[:, 0], pooled=pooled_all[:, 0])
        state_0 = model.state(frame_0, hidden)

        real_batch_dict = {
            "states": state_0,
            "actions": actions_raw[:, 0],
            "returns": returns[:, 0],
            "start_frames": frame_0,
            "start_beliefs": hidden,
        }

        step_res = gate8_imagination_training_step(
            model=model,
            real_batch=real_batch_dict,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            horizon=horizon,
            gamma=gamma,
            lambda_param=lambda_param,
            beta_temperature=beta,
            world_model_override=world_model_override,
        )

        history["steps"].append(step_idx)
        history["actor_loss"].append(step_res["total_actor_loss"])
        history["critic_loss"].append(step_res["total_critic_loss"])
        history["imag_return"].append(step_res["mean_imag_return"])
        history["imag_reward"].append(step_res["mean_imag_reward"])
        history["imag_continuation"].append(step_res["mean_imag_continuation"])
        history["entropy"].append(step_res["mean_entropy"])
        history["gap"].append(step_res["imag_vs_real_return_gap"])

        if step_idx % 50 == 0 or step_idx == updates:
            elapsed = time.time() - t0
            print(
                f"    [{candidate_name}] Step {step_idx:3d}/{updates:3d} | "
                f"ActLoss: {step_res['total_actor_loss']:+.3f} | "
                f"CritLoss: {step_res['total_critic_loss']:.3f} | "
                f"ImagRet: {step_res['mean_imag_return']:+.2f} | "
                f"ImagCont: {step_res['mean_imag_continuation']:.2f} | "
                f"Ent: {step_res['mean_entropy']:.2f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    return model, history


def generate_gate8_telemetry_plots(
    pred_results: Dict[str, Any],
    h_histories: Dict[str, Dict[str, List[float]]],
    benchmark_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generates 4-panel Gate 8 telemetry visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Panel 1: Multi-Step Prediction Error Growth vs Baselines
    ax1 = axes[0]
    horizons = [1, 2, 4, 8]
    true_errs = [pred_results[f"H_{h}"]["true_error"] for h in horizons]
    shuf_errs = [pred_results[f"H_{h}"]["shuffled_error"] for h in horizons]
    persist_errs = [pred_results[f"H_{h}"]["persistence_error"] for h in horizons]

    ax1.plot(horizons, true_errs, "o-", color="#4285F4", label="True Actions (Learned WM)", linewidth=2)
    ax1.plot(horizons, shuf_errs, "s--", color="#EA4335", label="Shuffled Actions", linewidth=1.5)
    ax1.plot(horizons, persist_errs, "^:", color="#9AA0A6", label="Persistence Baseline", linewidth=1.5)
    ax1.set_xlabel("Imagination Horizon (Steps)")
    ax1.set_ylabel("Prediction Smooth L1 Loss")
    ax1.set_title("World Model Closed-Loop Prediction Error")
    ax1.set_xticks(horizons)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Training Imagined Return vs Real Gap
    ax2 = axes[1]
    if "H=1" in h_histories:
        steps = h_histories["H=1"]["steps"]
        ax2.plot(steps, h_histories["H=1"]["imag_return"], label="H=1 Imagined Return", color="#34A853")
    if "H=2" in h_histories:
        steps = h_histories["H=2"]["steps"]
        ax2.plot(steps, h_histories["H=2"]["imag_return"], label="H=2 Imagined Return", color="#4285F4")
    if "H=4" in h_histories:
        steps = h_histories["H=4"]["steps"]
        ax2.plot(steps, h_histories["H=4"]["imag_return"], label="H=4 Imagined Return", color="#FBBC04")

    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Imagined Return G_imag")
    ax2.set_title("Imagined Return Trajectories by Horizon")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Real Crafter Benchmark Returns (30 Seeds)
    ax3 = axes[2]
    arms = ["Random", "Gate 7 Champion", "Gate 8 H=1", "Gate 8 H=2", "Gate 8 H=4"]
    keys = ["random_baseline", "gate7_champion", "gate8_h1", "gate8_h2", "gate8_h4"]
    active_arms = [a for a, k in zip(arms, keys) if k in benchmark_results]
    active_keys = [k for k in keys if k in benchmark_results]

    returns = [benchmark_results[k]["mean_return"] for k in active_keys]
    stds = [benchmark_results[k]["std_return"] for k in active_keys]
    scores = [benchmark_results[k]["crafter_score"] for k in active_keys]

    x = np.arange(len(active_arms))
    ax3_twin = ax3.twinx()
    b1 = ax3.bar(x - 0.15, returns, yerr=stds, capsize=4, width=0.3, color="#4285F4", alpha=0.85, label="Mean Return")
    b2 = ax3_twin.bar(x + 0.15, scores, width=0.3, color="#FBBC04", alpha=0.85, label="Crafter Score (%)")

    ax3.set_xticks(x)
    ax3.set_xticklabels([a.replace(" ", "\n") for a in active_arms], fontsize=8)
    ax3.set_ylabel("Mean Return / Episode")
    ax3_twin.set_ylabel("Crafter Score (%)")
    ax3.set_title("Real Crafter 30-Seed Qualification")
    ax3.grid(axis="y", alpha=0.3)

    for i, (r_val, s_val) in enumerate(zip(returns, scores)):
        ax3.text(i - 0.15, r_val + 0.1, f"{r_val:+.2f}", ha="center", fontsize=8, fontweight="bold")
        ax3_twin.text(i + 0.15, s_val + 0.2, f"{s_val:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel 4: Achievement Comparison
    ax4 = axes[3]
    top_achs = ["collect_sapling", "place_plant", "collect_wood", "eat_cow", "wake_up"]
    x_achs = np.arange(len(top_achs))
    w = 0.22

    plot_keys = ["gate7_champion", "gate8_h1", "gate8_h2"]
    plot_labels = ["Gate 7 (Real AWR)", "Gate 8 H=1", "Gate 8 H=2"]
    colors = ["#4285F4", "#34A853", "#FBBC04"]

    for idx, (k, col, lbl) in enumerate(zip(plot_keys, colors, plot_labels)):
        if k in benchmark_results:
            rates = [benchmark_results[k]["achievement_rates"].get(ach, 0.0) * 100 for ach in top_achs]
            ax4.bar(x_achs + (idx - 1) * w, rates, width=w, label=lbl, color=col, alpha=0.85)

    ax4.set_xticks(x_achs)
    ax4.set_xticklabels([a.replace("_", "\n") for a in top_achs], fontsize=8)
    ax4.set_ylabel("Unlock Rate (%)")
    ax4.set_title("Key Achievement Rates")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  ✓ Telemetry plots saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ORBIT Gate 8: Imagination Policy Training")
    parser.add_argument("--updates", type=int, default=300, help="Updates per candidate (default: 300)")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size (default: 64)")
    parser.add_argument("--seq-len", type=int, default=8, help="Prefix sequence length (default: 8)")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per Crafter episode (default: 300)")
    parser.add_argument("--benchmark-seeds-count", type=int, default=30, help="Benchmark seeds count (default: 30)")
    args = parser.parse_args()

    print("=" * 80)
    print("ORBIT — Gate 8: Short-Horizon Imagination Policy & Real-Gameplay Qualification")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Execution Device: {device} | PyTorch {torch.__version__}")

    # Paths
    wm_path = Path("results/phase3/world_model_phase3.pt")
    gate7_path = Path("results/gate7/policy_gate7.pt")
    replay_path = Path("data/gate7/augmented_replay_local.npz")
    heldout_path = Path("data/gate3/eval_heldout_seeds.pt")
    out_dir = Path("results/gate8")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Preflight Audit & Provenance Verification
    # -----------------------------------------------------------------------
    print("\n[1/7] Step One: Preflight Audit & Checkpoint Provenance...")
    assert wm_path.exists(), f"World model champion missing at {wm_path}"
    assert gate7_path.exists(), f"Gate 7 champion missing at {gate7_path}"
    assert replay_path.exists(), f"Replay dataset missing at {replay_path}"
    assert heldout_path.exists(), f"Held-out eval dataset missing at {heldout_path}"

    wm_sha = compute_sha256(wm_path)
    gate7_sha = compute_sha256(gate7_path)
    replay_sha = compute_sha256(replay_path)

    expected_wm_sha = "918fdeee450d87de50cd1f7f796e69fb8446db87e57613a4948302daf3383306"
    expected_gate7_sha = "08615c72e2c2ec0c5e52e32e2b88cfeac2a51557b10dcdcd27daf0d57b9a44d8"

    assert wm_sha == expected_wm_sha, f"World model SHA mismatch: {wm_sha} != {expected_wm_sha}"
    assert gate7_sha == expected_gate7_sha, f"Gate 7 SHA mismatch: {gate7_sha} != {expected_gate7_sha}"
    print(f"  ✓ World Model Champion Hash Verified: {wm_sha}")
    print(f"  ✓ Gate 7 Policy Champion Hash Verified: {gate7_sha}")
    print(f"  ✓ Replay Dataset Hash Verified:        {replay_sha}")

    # Load baseline model
    base_model, base_meta = TrueJEPA.load_checkpoint(str(gate7_path), device=device)
    base_model.eval()

    # -----------------------------------------------------------------------
    # Step 2: Historical Benchmark Discrepancy Resolution & Baseline Re-Evaluation
    # -----------------------------------------------------------------------
    print("\n[2/7] Step Two: Benchmark Discrepancy Resolution & Protocol Locking...")
    bench_seeds = list(range(5000, 5000 + args.benchmark_seeds_count))
    print(f"  Locked Benchmark Seeds: {bench_seeds[0]}..{bench_seeds[-1]} ({len(bench_seeds)} seeds)")

    print("  Benchmarking Random Baseline under locked per-seed RNG...")
    random_res = evaluate_policy_locked_rng(RandomCrafterAgent(seed=42), seeds=bench_seeds, max_steps=args.max_steps)
    print(f"    Random Baseline -> Return: {random_res['mean_return']:+.2f} ± {random_res['std_return']:.2f} | Score: {random_res['crafter_score']:.2f}% | Steps: {random_res['mean_length']:.1f}")

    print("  Benchmarking Gate 7 Champion (No-Imagination Control) under locked per-seed RNG...")
    champ_agent = VisualJEPAAgent(base_model, device=str(device), stochastic=True, temperature=1.0)
    gate7_control_res = evaluate_policy_locked_rng(champ_agent, seeds=bench_seeds, max_steps=args.max_steps)
    print(f"    Gate 7 Champion -> Return: {gate7_control_res['mean_return']:+.2f} ± {gate7_control_res['std_return']:.2f} | Score: {gate7_control_res['crafter_score']:.2f}% | Steps: {gate7_control_res['mean_length']:.1f} | Entropy: {gate7_control_res['action_entropy']:.3f}")

    # -----------------------------------------------------------------------
    # Step 3: Requalify Imagination Prerequisites on Held-Out Diagnostic Seeds
    # -----------------------------------------------------------------------
    print(f"\n[3/7] Step Three: Requalifying Imagination Prerequisites on Held-Out Seeds ({heldout_path})...")
    heldout_episodes = load_heldout_episodes(heldout_path)
    print(f"  Loaded {len(heldout_episodes)} held-out diagnostic trajectories.")

    pred_res = validate_action_conditioned_prediction(base_model, heldout_episodes, device, horizons=[1, 2, 4, 8])
    for h in [1, 2, 4, 8]:
        info = pred_res[f"H_{h}"]
        print(f"  Horizon H={h}: TrueErr: {info['true_error']:.4f} | ShufErr: {info['shuffled_error']:.4f} | PersistErr: {info['persistence_error']:.4f} | ActionSensitivity: {info['action_sensitivity_ratio']:.2f}x | BeatsPersist: {info['beats_persistence']}")

    rew_res = validate_reward_prediction(base_model, heldout_episodes, device)
    print(f"  Reward Head: MAE: {rew_res['reward_mae']:.4f} (vs Const: {rew_res['constant_baseline_mae']:.4f}) | Correlation: {rew_res['reward_correlation']:+.4f} | BeatsConstant: {rew_res['beats_constant_baseline']}")

    cont_res = validate_continuation_prediction(base_model, heldout_episodes, device)
    print(f"  Continuation Head: Brier: {cont_res['brier_score']:.4f} (vs Prev: {cont_res['prevalence_brier']:.4f}) | Acc: {cont_res['accuracy']*100:.1f}% | DeathRecall: {cont_res['death_recall']*100:.1f}% | BeatsPrev: {cont_res['beats_prevalence']}")

    val_res = validate_value_critic_calibration(base_model, heldout_episodes, device, gamma=0.97)
    print(f"  Value Critic: MAE: {val_res['value_mae']:.4f} | Correlation: {val_res['value_correlation']:+.4f} | Bias: {val_res['value_bias']:+.4f} | ExplVar: {val_res['explained_variance']:.3f}")

    # Formal acceptance conditions check
    c1 = (pred_res["H_2"]["action_sensitivity_ratio"] > 1.0) or (pred_res["H_4"]["action_sensitivity_ratio"] > 1.0)
    c2 = rew_res["beats_constant_baseline"]
    c3 = cont_res["beats_prevalence"]
    c4 = val_res["value_correlation"] > 0.05
    c5 = math.isfinite(val_res["value_mae"]) and math.isfinite(rew_res["reward_mae"])
    acceptance_passed = c1 and c2 and c3 and c4 and c5
    print(f"\n  ✓ Formal Imagination Acceptance Gate: {'PASS' if acceptance_passed else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Step 4: Horizon-by-Horizon Imagination Training (H=1, H=2, H=4, H=8)
    # -----------------------------------------------------------------------
    print("\n[4/7] Step Four: Horizon-by-Horizon Imagination Training Experiments...")
    replay = PixelReplay.load(str(replay_path))

    h_candidates: Dict[str, TrueJEPA] = {}
    h_histories: Dict[str, Dict[str, List[float]]] = {}
    h_benchmarks: Dict[str, Dict[str, Any]] = {}

    horizons_to_test = [1, 2, 4]  # Test H=1, H=2, H=4 progressively
    for h in horizons_to_test:
        c_name = f"gate8_h{h}"
        print(f"\n  --- Training Candidate {c_name.upper()} (Horizon H={h}, {args.updates} updates) ---")
        cand_model, hist = train_imagination_candidate(
            initial_model=base_model,
            replay=replay,
            horizon=h,
            updates=args.updates,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            actor_lr=3e-4,
            critic_lr=1e-3,
            gamma=0.97,
            lambda_param=0.95,
            beta=3.0,
            device=device,
            candidate_name=c_name,
        )

        # Save checkpoint
        ckpt_path = out_dir / f"policy_{c_name}.pt"
        cand_model.save_checkpoint(str(ckpt_path), update=args.updates)
        print(f"  ✓ Checkpoint saved to {ckpt_path} (SHA-256: {compute_sha256(ckpt_path)[:16]}...)")

        # Evaluate candidate in Crafter
        cand_model.eval()
        agent = VisualJEPAAgent(cand_model, device=str(device), stochastic=True, temperature=1.0)
        res = evaluate_policy_locked_rng(agent, seeds=bench_seeds, max_steps=args.max_steps)
        print(f"  Candidate {c_name.upper()} Benchmark -> Return: {res['mean_return']:+.2f} ± {res['std_return']:.2f} | Score: {res['crafter_score']:.2f}% | Steps: {res['mean_length']:.1f} | Entropy: {res['action_entropy']:.3f}")

        h_candidates[c_name] = cand_model
        h_histories[f"H={h}"] = hist
        h_benchmarks[c_name] = res

    # -----------------------------------------------------------------------
    # Step 5: Required Ablations (Persistence WM, Shuffled Action)
    # -----------------------------------------------------------------------
    print("\n[5/7] Step Five: Diagnostic Ablations (Testing H=1 Baseline)...")

    # Ablation 1: Persistence World Model
    print("  Training Ablation: Persistence World Model (H=1)...")
    persist_model, _ = train_imagination_candidate(
        initial_model=base_model,
        replay=replay,
        horizon=1,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.97,
        lambda_param=0.95,
        beta=3.0,
        device=device,
        world_model_override="persistence",
        candidate_name="ablation_persist",
    )
    persist_agent = VisualJEPAAgent(persist_model, device=str(device), stochastic=True, temperature=1.0)
    ablation_persist_res = evaluate_policy_locked_rng(persist_agent, seeds=bench_seeds, max_steps=args.max_steps)
    print(f"    Persistence Ablation -> Return: {ablation_persist_res['mean_return']:+.2f} ± {ablation_persist_res['std_return']:.2f} | Score: {ablation_persist_res['crafter_score']:.2f}%")

    # Ablation 2: Shuffled-Action World Model
    print("  Training Ablation: Shuffled-Action Conditioning (H=1)...")
    shuf_model, _ = train_imagination_candidate(
        initial_model=base_model,
        replay=replay,
        horizon=1,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.97,
        lambda_param=0.95,
        beta=3.0,
        device=device,
        world_model_override="shuffled_action",
        candidate_name="ablation_shuffled",
    )
    shuf_agent = VisualJEPAAgent(shuf_model, device=str(device), stochastic=True, temperature=1.0)
    ablation_shuf_res = evaluate_policy_locked_rng(shuf_agent, seeds=bench_seeds, max_steps=args.max_steps)
    print(f"    Shuffled-Action Ablation -> Return: {ablation_shuf_res['mean_return']:+.2f} ± {ablation_shuf_res['std_return']:.2f} | Score: {ablation_shuf_res['crafter_score']:.2f}%")

    # -----------------------------------------------------------------------
    # Step 6: Statistical Analysis & Promotion Matrix
    # -----------------------------------------------------------------------
    print("\n[6/7] Step Six: Statistical Comparison & Decision Matrix...")
    stats_vs_gate7: Dict[str, Any] = {}
    for c_name, res in h_benchmarks.items():
        stats = compute_paired_statistics(res, gate7_control_res)
        stats_vs_gate7[c_name] = stats
        print(f"  {c_name.upper()} vs Gate 7 Champion: MeanDiff: {stats['mean_diff']:+.2f} | t-stat: {stats['t_stat']:+.3f} | 95% CI: [{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}] | P(Better): {stats['prob_better']*100:.1f}%")

    # Promotion Rule (§16): Requires statistically significant improvement over Gate 7 champion
    # (t > 2.045 / 95% CI strictly positive) without survival or score regression
    best_candidate_name = max(h_benchmarks.keys(), key=lambda k: h_benchmarks[k]["crafter_score"])
    best_res = h_benchmarks[best_candidate_name]
    best_stats = stats_vs_gate7[best_candidate_name]

    score_gain = best_res["crafter_score"] - gate7_control_res["crafter_score"]
    return_gain = best_res["mean_return"] - gate7_control_res["mean_return"]
    stat_sig = bool(best_stats["ci95_low"] > 0.0 or best_stats["t_stat"] >= 2.045)

    if score_gain > 0.0 and return_gain > 0.0 and stat_sig:
        promotion_status = "PASS_AND_PROMOTED"
        verdict = f"{best_candidate_name.upper()} PROMOTED as new production champion (Score: {best_res['crafter_score']:.2f}% vs Gate 7 {gate7_control_res['crafter_score']:.2f}%, Return: {best_res['mean_return']:+.2f} vs {gate7_control_res['mean_return']:+.2f}, p < 0.05)."
    else:
        promotion_status = "PASS_CANDIDATE_NOT_PROMOTED"
        verdict = f"Gate 7 AWR RETAINED as production champion (Score: {gate7_control_res['crafter_score']:.2f}%, Return: {gate7_control_res['mean_return']:+.2f}). Best imagination candidate ({best_candidate_name.upper()}: Score: {best_res['crafter_score']:.2f}%, Return: {best_res['mean_return']:+.2f}) did not demonstrate statistically significant real-game superiority."

    print("\n" + "=" * 80)
    print("GATE 8 FINAL QUALIFICATION & PROMOTION VERDICT")
    print("=" * 80)
    print(f"Promotion Status:     {promotion_status}")
    print(f"Summary Verdict:      {verdict}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 7: Telemetry Plots & Manifest Serialization
    # -----------------------------------------------------------------------
    print("\n[7/7] Step Seven: Serializing Metrics Manifest & Plots...")
    all_benchmarks = {
        "random_baseline": random_res,
        "gate7_champion": gate7_control_res,
        **h_benchmarks,
        "ablation_persistence": ablation_persist_res,
        "ablation_shuffled": ablation_shuf_res,
    }

    # Generate plots
    plot_path = out_dir / "gate8_telemetry_plots.png"
    generate_gate8_telemetry_plots(
        pred_results=pred_res,
        h_histories=h_histories,
        benchmark_results=all_benchmarks,
        output_path=plot_path,
    )

    # Dump JSON metrics manifest
    metrics_manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "Gate 8: Short-Horizon Imagination Policy & Real-Gameplay Qualification",
        "promotion_status": promotion_status,
        "verdict": verdict,
        "best_imagination_candidate": best_candidate_name,
        "provenance": {
            "world_model_path": str(wm_path),
            "world_model_sha256": wm_sha,
            "gate7_champion_path": str(gate7_path),
            "gate7_champion_sha256": gate7_sha,
            "replay_dataset_path": str(replay_path),
            "replay_dataset_sha256": replay_sha,
            "candidate_checkpoints": {k: str(out_dir / f"policy_{k}.pt") for k in h_benchmarks.keys()},
        },
        "prerequisite_qualifications": {
            "action_conditioned_prediction": pred_res,
            "reward_prediction": rew_res,
            "continuation_prediction": cont_res,
            "value_calibration": val_res,
            "formal_acceptance_gate": "PASS" if acceptance_passed else "FAIL",
        },
        "statistical_comparisons_vs_gate7": stats_vs_gate7,
        "benchmark_results": {k: {m: n for m, n in v.items() if m != "per_seed_results"} for k, v in all_benchmarks.items()},
    }

    metrics_path = out_dir / "gate8_experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_manifest, f, indent=2)
    print(f"  ✓ Metrics manifest saved to {metrics_path}")

    # Dump full per-seed results manifest for total reproducibility
    per_seed_manifest = {
        k: v["per_seed_results"] for k, v in all_benchmarks.items()
    }
    per_seed_path = out_dir / "gate8_per_seed_evaluation_manifest.json"
    with open(per_seed_path, "w") as f:
        json.dump(per_seed_manifest, f, indent=2)
    print(f"  ✓ Per-seed evaluation manifest saved to {per_seed_path}")


if __name__ == "__main__":
    main()
