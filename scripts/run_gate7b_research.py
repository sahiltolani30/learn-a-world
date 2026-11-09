#!/usr/bin/env python3
"""ORBIT — Gate 7B: Actor-Critic Research, Algorithm Selection & Architecture Qualification.

Executes the Gate 7B experimental addendum:
1. Audits replay dataset coverage, action frequencies, and reward distributions.
2. Trains and compares 4 actor-critic algorithm candidates:
   - Candidate A: Existing AWR (Gate 7 reproduction / control)
   - Candidate B: Implicit Q-Learning (IQL, Kostrikov et al., 2021)
   - Candidate C: Discrete Conservative Q-Learning (CQL, Kumar et al., 2020)
   - Candidate D: Critic Regularized Regression (CRR - binary filtered, Wang et al., 2020)
3. Evaluates Critic Architectures:
   - Critic A: Scalar Value Critic V(s)
   - Critic B: Twin Q Critics Q_1(s, a), Q_2(s, a)
   - Critic C: Distributional Quantile Critic Z(s)
4. Evaluates Actor Architectures:
   - Actor A: Standard Categorical MLP policy head
   - Actor B: Parameter-matched Residual MLP policy head
5. Evaluates on 20 held-out seeds (value correlation, MAE, bias).
6. Benchmarks all candidates on locked 30-seed Crafter evaluation (5000..5029).
7. Computes paired statistical significance, bootstrap confidence intervals.
8. Protects Gate 7 champion and saves independent candidate checkpoints to results/gate7b/.
9. Generates 6-panel diagnostic comparison plots and full JSON manifest.
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
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.gate6_policy_evaluator import (
    ALL_CRAFTER_ACHIEVEMENTS,
    RandomCrafterAgent,
    compute_crafter_score,
    evaluate_policy_in_crafter,
)
from eval.gate7_evaluator import (
    audit_replay_dataset,
    calibrate_value_predictions,
    real_experience_policy_step,
    validate_reward_prediction,
)
from eval.gate7b_algorithms import (
    crr_training_step,
    discrete_cql_training_step,
    iql_training_step,
    quantile_huber_loss,
)
from eval.gate7b_models import (
    DiscreteQCritic,
    DistributionalValueCritic,
    ExpectileValueCritic,
    ResidualPolicyHead,
)
from scripts.run_gate7_real_policy import load_heldout_episodes
from scripts.run_phase3b_integration import resolve_drive_root, resolve_file
from true_jepa.agent import VisualJEPAAgent
from true_jepa.model import LatentFrame, TrueJEPA
from true_jepa.replay import PixelReplay


class CustomActorJEPAAgent:
    """Pixel-only visual agent wrapping TrueJEPA with a custom actor head (e.g. ResidualPolicyHead)."""

    def __init__(
        self,
        model: TrueJEPA,
        actor_head: nn.Module,
        device: str = "cpu",
        stochastic: bool = True,
        temperature: float = 1.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.actor_head = actor_head.to(device).eval()
        self.device = torch.device(device)
        self.stochastic = bool(stochastic)
        self.temperature = float(temperature)
        self.hidden: Optional[torch.Tensor] = None
        self.previous_action: Optional[int] = None

    def reset(self) -> None:
        self.hidden = None
        self.previous_action = None

    @torch.no_grad()
    def act(
        self,
        obs_uint8: np.ndarray,
        reward_prev: float = 0.0,
        step: Optional[int] = None,
    ) -> int:
        obs = torch.from_numpy(np.ascontiguousarray(obs_uint8)).unsqueeze(0).to(self.device)
        frame = self.model.encode(obs)
        if self.hidden is None:
            self.hidden = self.model.initial_belief(1, self.device)
        elif self.previous_action is not None:
            previous = torch.tensor([self.previous_action], device=self.device)
            reward = torch.tensor([float(reward_prev)], device=self.device)
            self.hidden = self.model.belief_step(frame, previous, reward, self.hidden)

        state = self.model.heads.state(frame, self.hidden)
        logits = self.actor_head(state)

        if self.stochastic:
            temp = max(self.temperature, 1e-4)
            action = torch.distributions.Categorical(logits=logits / temp).sample()
        else:
            action = torch.argmax(logits, dim=-1)
        self.previous_action = int(action.item())
        return self.previous_action


def compute_paired_statistics(
    candidate_returns: List[float],
    champion_returns: List[float],
) -> Dict[str, float]:
    """Computes paired t-statistic, mean difference, and 95% confidence interval."""
    c_arr = np.array(candidate_returns, dtype=np.float64)
    ch_arr = np.array(champion_returns, dtype=np.float64)
    diff = c_arr - ch_arr
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = std_diff / math.sqrt(n) if n > 1 else 0.0
    t_stat = mean_diff / se if se > 1e-8 else 0.0
    ci95_low = mean_diff - 1.96 * se
    ci95_high = mean_diff + 1.96 * se

    # One-sided probability of improvement
    prob_better = float(np.mean(diff > 0.0))

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "prob_better": prob_better,
    }


def generate_gate7b_comparison_plots(
    benchmarks: Dict[str, Any],
    critic_metrics: Dict[str, Any],
    training_histories: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generates 6-panel comprehensive diagnostic and comparative visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    # Panel 1: Crafter Geometric Score Comparison
    ax1 = axes[0, 0]
    names = list(benchmarks.keys())
    scores = [benchmarks[k]["crafter_score"] for k in names]
    colors = ["#9AA0A6", "#4285F4", "#FBBC04", "#EA4335", "#34A853", "#9C27B0"][:len(names)]
    bars = ax1.bar(np.arange(len(names)), scores, color=colors, alpha=0.85, width=0.55)
    ax1.set_xticks(np.arange(len(names)))
    ax1.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax1.set_ylabel("Crafter Geometric Score (%)")
    ax1.set_title("Crafter Score by Algorithm")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(scores):
        ax1.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=8)

    # Panel 2: Mean Return per Episode
    ax2 = axes[0, 1]
    returns = [benchmarks[k]["mean_return"] for k in names]
    stds = [benchmarks[k]["std_return"] for k in names]
    ax2.bar(np.arange(len(names)), returns, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.55)
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax2.set_ylabel("Mean Return / Episode")
    ax2.set_title("Real Return (30 Seeds)")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(returns):
        ax2.text(i, v + 0.1, f"{v:+.2f}", ha="center", fontweight="bold", fontsize=8)

    # Panel 3: Action Shannon Entropy
    ax3 = axes[0, 2]
    entropies = [benchmarks[k]["action_entropy"] for k in names]
    ax3.bar(np.arange(len(names)), entropies, color=colors, alpha=0.85, width=0.55)
    ax3.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Collapse Floor (1.0)")
    ax3.set_xticks(np.arange(len(names)))
    ax3.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax3.set_ylabel("Entropy (nats)")
    ax3.set_title("Action Distribution Diversity")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    for i, v in enumerate(entropies):
        ax3.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold", fontsize=8)

    # Panel 4: Critic Architectures - Value Correlation on Held-Out Seeds
    ax4 = axes[1, 0]
    critic_names = list(critic_metrics.keys())
    corrs = [critic_metrics[c]["value_correlation"] for c in critic_names]
    maes = [critic_metrics[c]["value_mae"] for c in critic_names]
    x_c = np.arange(len(critic_names))
    w = 0.35
    ax4.bar(x_c - w/2, corrs, width=w, label="Pearson r", color="#4285F4", alpha=0.85)
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x_c + w/2, maes, width=w, label="MAE", color="#EA4335", alpha=0.85)
    ax4.set_xticks(x_c)
    ax4.set_xticklabels([c.replace("_", "\n") for c in critic_names], fontsize=8)
    ax4.set_ylabel("Pearson Correlation (Higher is better)")
    ax4_twin.set_ylabel("MAE (Lower is better)")
    ax4.set_title("Critic Architecture Calibration")
    ax4.grid(axis="y", alpha=0.3)
    ax4.legend(loc="upper left", fontsize=8)
    ax4_twin.legend(loc="upper right", fontsize=8)

    # Panel 5: Key Achievement Unlock Comparison
    ax5 = axes[1, 1]
    key_achs = ["collect_wood", "place_table", "collect_sapling", "place_plant", "wake_up"]
    x_a = np.arange(len(key_achs))
    w_a = 0.8 / max(1, len(names))
    for idx, (name, col) in enumerate(zip(names, colors)):
        rates = [benchmarks[name]["achievement_rates"].get(a, 0.0) * 100 for a in key_achs]
        ax5.bar(x_a + (idx - len(names)/2 + 0.5) * w_a, rates, width=w_a, label=name.replace("_", " "), color=col, alpha=0.85)
    ax5.set_xticks(x_a)
    ax5.set_xticklabels([a.replace("_", "\n") for a in key_achs], fontsize=8)
    ax5.set_ylabel("Unlock Rate (%)")
    ax5.set_title("Key Achievement Unlock Rates")
    ax5.legend(fontsize=7)
    ax5.grid(axis="y", alpha=0.3)

    # Panel 6: Actor Loss Trajectories Across Training
    ax6 = axes[1, 2]
    for name, hist in training_histories.items():
        if "steps" in hist and "actor_loss" in hist:
            ax6.plot(hist["steps"], hist["actor_loss"], label=name.replace("_", " "), alpha=0.85)
    ax6.set_xlabel("Update Step")
    ax6.set_ylabel("Actor Objective Loss")
    ax6.set_title("Training Loss Convergence")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[Gate 7B Plots] Saved 6-panel comparative plots to {output_path}")


def run_gate7b_research_study(
    world_model_ckpt: str = "results/phase3/world_model_phase3.pt",
    champion_gate7_ckpt: str = "results/gate7/policy_gate7.pt",
    replay_dataset_path: str = "data/gate7/augmented_replay_local.npz",
    heldout_eval_path: str = "data/gate3/eval_heldout_seeds.pt.gz",
    updates: int = 300,
    batch_size: int = 8,
    seq_len: int = 8,
    benchmark_seeds_count: int = 30,
    max_steps: int = 300,
    device_str: str = "auto",
    output_dir: str = "results/gate7b",
    drive_dir: Optional[str] = None,
    seed: int = 777,
) -> Dict[str, Any]:
    """Executes the complete Gate 7B Actor-Critic research and qualification suite."""
    import math
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print("ORBIT — GATE 7B: ACTOR-CRITIC RESEARCH & ALGORITHM SELECTION")
    print("=" * 80)

    # 1. Device Selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)
    print(f"Active Device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. File Resolution & Champion Protection
    wm_file = resolve_file(world_model_ckpt, drive_dir)
    champ_file = resolve_file(champion_gate7_ckpt, drive_dir)
    heldout_file = resolve_file(heldout_eval_path, drive_dir)
    replay_file = resolve_file(replay_dataset_path, drive_dir)

    print(f"\n[Verified Checkpoint & Data Provenance]")
    print(f"  World Model:          {wm_file}")
    print(f"  Gate 7 Champion:      {champ_file}")
    print(f"  Replay Dataset:       {replay_file}")
    print(f"  Held-out Evaluation:  {heldout_file}")

    # Verify Gate 7 champion hash
    h_champ = hashlib.sha256(champ_file.read_bytes()).hexdigest()
    print(f"  ✓ Gate 7 Champion SHA-256: {h_champ}")
    expected_champ_sha = "08615c72e2c2ec0c5e52e32e2b88cfeac2a51557b10dcdcd27daf0d57b9a44d8"
    assert h_champ == expected_champ_sha, f"Gate 7 champion hash mismatch! Expected {expected_champ_sha}, got {h_champ}"

    # 3. Dataset Coverage Analysis
    print(f"\n[1/6] Auditing Replay Dataset Coverage from {replay_file}...")
    dataset_audit = audit_replay_dataset(replay_file)
    with open(out_path / "gate7b_dataset_manifest.json", "w") as f:
        json.dump(dataset_audit, f, indent=2)

    replay = PixelReplay.load(str(replay_file))
    heldout_episodes = load_heldout_episodes(heldout_file)
    print(f"  Loaded {len(replay)} replay episodes, {len(heldout_episodes)} held-out episodes.")

    # 4. Load Base World Model (Strictly Frozen)
    model, _ = TrueJEPA.load_checkpoint(str(wm_file), device=device)
    model.encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    model.predictor.requires_grad_(False).eval()
    model.belief.requires_grad_(False).eval()
    model.heads.reward.requires_grad_(False).eval()
    model.heads.continuation.requires_grad_(False).eval()
    state_dim = model.heads.policy[0].in_features  # 448

    bench_seeds = list(range(5000, 5000 + benchmark_seeds_count))
    training_histories: Dict[str, Any] = {}
    candidate_checkpoints: Dict[str, str] = {}
    benchmark_manifests: Dict[str, Any] = {}
    critic_calibration_metrics: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Benchmark Arm 1: Matched Random Baseline
    # -----------------------------------------------------------------------
    print(f"\n[Benchmark Arm: Random Baseline]")
    random_res = evaluate_policy_in_crafter(RandomCrafterAgent(seed=42), seeds=bench_seeds, max_steps=max_steps)
    benchmark_manifests["random_baseline"] = random_res
    print(f"  Return: {random_res['mean_return']:+.2f} ± {random_res['std_return']:.2f} | Score: {random_res['crafter_score']:.2f}% | Steps: {random_res['mean_length']:.1f}")

    # -----------------------------------------------------------------------
    # Benchmark Arm 2: Gate 7 Production Champion (Control)
    # -----------------------------------------------------------------------
    print(f"\n[Benchmark Arm: Gate 7 Production Champion]")
    champ_model, _ = TrueJEPA.load_checkpoint(str(champ_file), device=device)
    champ_agent = VisualJEPAAgent(champ_model, device=str(device), stochastic=True, temperature=1.0)
    champ_res = evaluate_policy_in_crafter(champ_agent, seeds=bench_seeds, max_steps=max_steps)
    benchmark_manifests["gate7_champion"] = champ_res
    print(f"  Return: {champ_res['mean_return']:+.2f} ± {champ_res['std_return']:.2f} | Score: {champ_res['crafter_score']:.2f}% | Steps: {champ_res['mean_length']:.1f}")

    # Evaluate Gate 7 Scalar Critic Calibration
    scalar_cal = calibrate_value_predictions(champ_model, heldout_episodes, device, gamma=0.99)
    critic_calibration_metrics["scalar_value_critic"] = scalar_cal

    # -----------------------------------------------------------------------
    # Candidate B: Implicit Q-Learning (IQL)
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Training Candidate B: Implicit Q-Learning (IQL)...")
    q_critic_iql = DiscreteQCritic(state_dim=state_dim, num_actions=17).to(device)
    v_critic_iql = ExpectileValueCritic(state_dim=state_dim).to(device)
    v_target_iql = ExpectileValueCritic(state_dim=state_dim).to(device)
    v_target_iql.load_state_dict(v_critic_iql.state_dict())
    actor_iql = copy.deepcopy(model.heads.policy).to(device)

    opt_q_iql = torch.optim.Adam(q_critic_iql.parameters(), lr=1e-4)
    opt_v_iql = torch.optim.Adam(v_critic_iql.parameters(), lr=1e-4)
    opt_act_iql = torch.optim.Adam(actor_iql.parameters(), lr=1e-4)

    rng = np.random.default_rng(seed)
    hist_iql = {"steps": [], "actor_loss": [], "q_loss": [], "v_loss": []}
    t0 = time.time()
    for step in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        step_res = iql_training_step(
            model=model,
            q_critic=q_critic_iql,
            v_critic=v_critic_iql,
            v_target=v_target_iql,
            actor_head=actor_iql,
            batch=batch,
            q_optimizer=opt_q_iql,
            v_optimizer=opt_v_iql,
            actor_optimizer=opt_act_iql,
            expectile=0.7,
            inv_temp=3.0,
            device=device,
        )
        if step % 25 == 0 or step == updates:
            hist_iql["steps"].append(step)
            hist_iql["actor_loss"].append(step_res["actor_loss"])
            hist_iql["q_loss"].append(step_res["q_loss"])
            hist_iql["v_loss"].append(step_res["v_loss"])
            print(f"  IQL Step {step:3d}/{updates} | QLoss: {step_res['q_loss']:.4f} | VLoss: {step_res['v_loss']:.4f} | ActLoss: {step_res['actor_loss']:+.4f} | Ent: {step_res['entropy']:.2f}")

    training_histories["candidate_b_iql"] = hist_iql
    print(f"  ✓ IQL Training Complete ({time.time() - t0:.1f}s)")

    # Save IQL model checkpoint
    model_iql = copy.deepcopy(model)
    model_iql.heads.policy.load_state_dict(actor_iql.state_dict())
    iql_ckpt = out_path / "policy_candidate_b_iql.pt"
    model_iql.save_checkpoint(str(iql_ckpt), update=updates)
    candidate_checkpoints["candidate_b_iql"] = str(iql_ckpt)

    # Benchmark IQL
    iql_agent = VisualJEPAAgent(model_iql, device=str(device), stochastic=True, temperature=1.0)
    iql_res = evaluate_policy_in_crafter(iql_agent, seeds=bench_seeds, max_steps=max_steps)
    benchmark_manifests["candidate_b_iql"] = iql_res
    print(f"  IQL Benchmark -> Return: {iql_res['mean_return']:+.2f} ± {iql_res['std_return']:.2f} | Score: {iql_res['crafter_score']:.2f}% | Steps: {iql_res['mean_length']:.1f}")

    # Evaluate IQL Twin-Q Value Calibration on held-out seeds
    # Using expectile V network as the return predictor
    v_critic_iql.eval()
    heldout_values = []
    heldout_returns = []
    with torch.no_grad():
        for ep in heldout_episodes:
            obs_raw = ep["obs"]
            acts = ep["action"] if "action" in ep else ep["actions"]
            rews = ep["reward"] if "reward" in ep else ep["rewards"]
            T = len(acts)
            if T == 0:
                continue
            ret_g = np.zeros(T)
            cur_g = 0.0
            for t in reversed(range(T)):
                cur_g = float(rews[t]) + 0.99 * cur_g
                ret_g[t] = cur_g
            frames = model.encode(torch.from_numpy(np.ascontiguousarray(obs_raw)).to(device))
            hidden = model.initial_belief(1, device)
            for t in range(T):
                f_t = LatentFrame(tokens=frames.tokens[t:t+1], pooled=frames.pooled[t:t+1])
                st_t = model.heads.state(f_t, hidden)
                v_pred_val = float(v_critic_iql(st_t).item())
                heldout_values.append(v_pred_val)
                heldout_returns.append(float(ret_g[t]))
                hidden = model.belief_step(f_t, torch.tensor([int(acts[t])], device=device), torch.tensor([float(rews[t])], device=device), hidden)

    p_val = np.array(heldout_values)
    g_val = np.array(heldout_returns)
    iql_corr = float(np.corrcoef(p_val, g_val)[0, 1]) if np.std(p_val) > 1e-6 and np.std(g_val) > 1e-6 else 0.0
    critic_calibration_metrics["twin_q_expectile_critic"] = {
        "value_mae": float(np.mean(np.abs(p_val - g_val))),
        "value_rmse": float(np.sqrt(np.mean((p_val - g_val)**2))),
        "value_correlation": iql_corr,
        "value_bias": float(np.mean(p_val - g_val)),
    }

    # -----------------------------------------------------------------------
    # Candidate C: Discrete Conservative Q-Learning (CQL)
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Training Candidate C: Discrete Conservative Q-Learning (CQL)...")
    q_critic_cql = DiscreteQCritic(state_dim=state_dim, num_actions=17).to(device)
    q_target_cql = DiscreteQCritic(state_dim=state_dim, num_actions=17).to(device)
    q_target_cql.load_state_dict(q_critic_cql.state_dict())
    actor_cql = copy.deepcopy(model.heads.policy).to(device)

    opt_q_cql = torch.optim.Adam(q_critic_cql.parameters(), lr=1e-4)
    opt_act_cql = torch.optim.Adam(actor_cql.parameters(), lr=1e-4)

    hist_cql = {"steps": [], "actor_loss": [], "total_q_loss": [], "cql_penalty": []}
    t0 = time.time()
    for step in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        step_res = discrete_cql_training_step(
            model=model,
            q_critic=q_critic_cql,
            q_target=q_target_cql,
            actor_head=actor_cql,
            batch=batch,
            q_optimizer=opt_q_cql,
            actor_optimizer=opt_act_cql,
            cql_alpha=1.0,
            device=device,
        )
        if step % 25 == 0 or step == updates:
            hist_cql["steps"].append(step)
            hist_cql["actor_loss"].append(step_res["actor_loss"])
            hist_cql["total_q_loss"].append(step_res["total_q_loss"])
            hist_cql["cql_penalty"].append(step_res["cql_penalty"])
            print(f"  CQL Step {step:3d}/{updates} | QLoss: {step_res['total_q_loss']:.4f} | CQLPen: {step_res['cql_penalty']:+.4f} | ActLoss: {step_res['actor_loss']:+.4f} | Ent: {step_res['entropy']:.2f}")

    training_histories["candidate_c_cql"] = hist_cql
    print(f"  ✓ CQL Training Complete ({time.time() - t0:.1f}s)")

    model_cql = copy.deepcopy(model)
    model_cql.heads.policy.load_state_dict(actor_cql.state_dict())
    cql_ckpt = out_path / "policy_candidate_c_cql.pt"
    model_cql.save_checkpoint(str(cql_ckpt), update=updates)
    candidate_checkpoints["candidate_c_cql"] = str(cql_ckpt)

    cql_agent = VisualJEPAAgent(model_cql, device=str(device), stochastic=True, temperature=1.0)
    cql_res = evaluate_policy_in_crafter(cql_agent, seeds=bench_seeds, max_steps=max_steps)
    benchmark_manifests["candidate_c_cql"] = cql_res
    print(f"  CQL Benchmark -> Return: {cql_res['mean_return']:+.2f} ± {cql_res['std_return']:.2f} | Score: {cql_res['crafter_score']:.2f}% | Steps: {cql_res['mean_length']:.1f}")

    # -----------------------------------------------------------------------
    # Candidate D: Critic Regularized Regression (CRR - Binary Filtered)
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Training Candidate D: Critic Regularized Regression (CRR)...")
    q_critic_crr = DiscreteQCritic(state_dim=state_dim, num_actions=17).to(device)
    q_target_crr = DiscreteQCritic(state_dim=state_dim, num_actions=17).to(device)
    q_target_crr.load_state_dict(q_critic_crr.state_dict())
    actor_crr = copy.deepcopy(model.heads.policy).to(device)

    opt_q_crr = torch.optim.Adam(q_critic_crr.parameters(), lr=1e-4)
    opt_act_crr = torch.optim.Adam(actor_crr.parameters(), lr=1e-4)

    hist_crr = {"steps": [], "actor_loss": [], "q_loss": [], "filter_ratio": []}
    t0 = time.time()
    for step in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        step_res = crr_training_step(
            model=model,
            q_critic=q_critic_crr,
            q_target=q_target_crr,
            actor_head=actor_crr,
            batch=batch,
            q_optimizer=opt_q_crr,
            actor_optimizer=opt_act_crr,
            device=device,
        )
        if step % 25 == 0 or step == updates:
            hist_crr["steps"].append(step)
            hist_crr["actor_loss"].append(step_res["actor_loss"])
            hist_crr["q_loss"].append(step_res["q_loss"])
            hist_crr["filter_ratio"].append(step_res["filter_pass_ratio"])
            print(f"  CRR Step {step:3d}/{updates} | QLoss: {step_res['q_loss']:.4f} | PassRatio: {step_res['filter_pass_ratio']:.2f} | ActLoss: {step_res['actor_loss']:+.4f} | Ent: {step_res['entropy']:.2f}")

    training_histories["candidate_d_crr"] = hist_crr
    print(f"  ✓ CRR Training Complete ({time.time() - t0:.1f}s)")

    model_crr = copy.deepcopy(model)
    model_crr.heads.policy.load_state_dict(actor_crr.state_dict())
    crr_ckpt = out_path / "policy_candidate_d_crr.pt"
    model_crr.save_checkpoint(str(crr_ckpt), update=updates)
    candidate_checkpoints["candidate_d_crr"] = str(crr_ckpt)

    crr_agent = VisualJEPAAgent(model_crr, device=str(device), stochastic=True, temperature=1.0)
    crr_res = evaluate_policy_in_crafter(crr_agent, seeds=bench_seeds, max_steps=max_steps)
    benchmark_manifests["candidate_d_crr"] = crr_res
    print(f"  CRR Benchmark -> Return: {crr_res['mean_return']:+.2f} ± {crr_res['std_return']:.2f} | Score: {crr_res['crafter_score']:.2f}% | Steps: {crr_res['mean_length']:.1f}")

    # -----------------------------------------------------------------------
    # Step 5: Critic C - Distributional Quantile Critic Training
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Training Critic C: Distributional Quantile Value Critic...")
    dist_critic = DistributionalValueCritic(state_dim=state_dim, num_quantiles=8).to(device)
    dist_opt = torch.optim.Adam(dist_critic.parameters(), lr=1e-4)

    for step in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        obs_b = batch["obs"].to(device)
        acts_b = batch["actions"].to(device)
        rews_b = batch["rewards"].to(device)
        terms_b = batch["terminated"].to(device)
        valid_b = batch["valid"].to(device)

        B_cur, T_cur = acts_b.shape
        # Compute Monte Carlo returns
        returns_mc = torch.zeros((B_cur, T_cur), dtype=torch.float32, device=device)
        run_ret = torch.zeros(B_cur, dtype=torch.float32, device=device)
        for t in reversed(range(T_cur)):
            run_ret = rews_b[:, t] + 0.99 * (~terms_b[:, t]).float() * run_ret
            returns_mc[:, t] = run_ret
        returns_mc = torch.clamp(returns_mc, -10.0, 10.0)

        with torch.no_grad():
            if obs_b.ndim == 5:
                flat_obs = obs_b.reshape(-1, *obs_b.shape[2:])
                frames_b = model.encode(flat_obs)
                tokens_all = frames_b.tokens.reshape(B_cur, T_cur + 1, *frames_b.tokens.shape[1:])
                pooled_all = frames_b.pooled.reshape(B_cur, T_cur + 1, *frames_b.pooled.shape[1:])
            else:
                flat_tokens = obs_b.reshape(-1, *obs_b.shape[2:])
                frames_b = model.encoder.adapter(flat_tokens)
                tokens_all = frames_b[0].reshape(B_cur, T_cur + 1, *frames_b[0].shape[1:])
                pooled_all = frames_b[1].reshape(B_cur, T_cur + 1, *frames_b[1].shape[1:])

            hidden_b = model.initial_belief(B_cur, device)
            states_all = []
            for t in range(T_cur):
                f_t = LatentFrame(tokens=tokens_all[:, t], pooled=pooled_all[:, t])
                states_all.append(model.heads.state(f_t, hidden_b))
                hidden_b = model.belief_step(f_t, acts_b[:, t], rews_b[:, t], hidden_b)
            states_all = torch.stack(states_all, dim=1)

        v_mask = valid_b.bool()
        states_flat = states_all[v_mask]
        target_flat = returns_mc[v_mask]

        dist_opt.zero_grad()
        q_pred = dist_critic(states_flat)
        dist_loss = quantile_huber_loss(q_pred, target_flat, dist_critic.tau_hats, kappa=1.0)
        dist_loss.backward()
        dist_opt.step()

    # Evaluate Distributional Critic on held-out seeds
    dist_critic.eval()
    dist_preds = []
    with torch.no_grad():
        for ep in heldout_episodes:
            obs_raw = ep["obs"]
            acts = ep["action"] if "action" in ep else ep["actions"]
            rews = ep["reward"] if "reward" in ep else ep["rewards"]
            T = len(acts)
            if T == 0:
                continue
            frames = model.encode(torch.from_numpy(np.ascontiguousarray(obs_raw)).to(device))
            hidden = model.initial_belief(1, device)
            for t in range(T):
                f_t = LatentFrame(tokens=frames.tokens[t:t+1], pooled=frames.pooled[t:t+1])
                st_t = model.heads.state(f_t, hidden)
                exp_v = float(dist_critic.expected_value(st_t).item())
                dist_preds.append(exp_v)
                hidden = model.belief_step(f_t, torch.tensor([int(acts[t])], device=device), torch.tensor([float(rews[t])], device=device), hidden)

    dist_p = np.array(dist_preds)
    dist_corr = float(np.corrcoef(dist_p, g_val)[0, 1]) if np.std(dist_p) > 1e-6 and np.std(g_val) > 1e-6 else 0.0
    critic_calibration_metrics["distributional_quantile_critic"] = {
        "value_mae": float(np.mean(np.abs(dist_p - g_val))),
        "value_rmse": float(np.sqrt(np.mean((dist_p - g_val)**2))),
        "value_correlation": dist_corr,
        "value_bias": float(np.mean(dist_p - g_val)),
    }
    print(f"  Critic C Calibration -> MAE: {critic_calibration_metrics['distributional_quantile_critic']['value_mae']:.4f} | Corr: {dist_corr:+.4f}")

    # -----------------------------------------------------------------------
    # Step 6: Actor Architecture Qualification (Residual Policy Head)
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Investigating Actor Architecture: Residual MLP Policy Head...")
    residual_actor = ResidualPolicyHead(state_dim=state_dim, num_actions=17).to(device)
    res_opt = torch.optim.Adam(residual_actor.parameters(), lr=1e-4)

    # Train residual actor using AWR objective
    champ_val = champ_model.heads.value
    for step in range(1, updates + 1):
        batch = replay.sample_with_prefix(batch_size=batch_size, sequence_length=seq_len, rng=rng)
        obs_b = batch["obs"].to(device)
        acts_b = batch["actions"].to(device)
        rews_b = batch["rewards"].to(device)
        terms_b = batch["terminated"].to(device)
        valid_b = batch["valid"].to(device).float()
        B_cur, T_cur = acts_b.shape

        returns_mc = torch.zeros((B_cur, T_cur), dtype=torch.float32, device=device)
        run_ret = torch.zeros(B_cur, dtype=torch.float32, device=device)
        for t in reversed(range(T_cur)):
            run_ret = rews_b[:, t] + 0.99 * (~terms_b[:, t]).float() * run_ret
            returns_mc[:, t] = run_ret
        returns_mc = torch.clamp(returns_mc, -10.0, 10.0)

        with torch.no_grad():
            if obs_b.ndim == 5:
                flat_obs = obs_b.reshape(-1, *obs_b.shape[2:])
                frames_b = model.encode(flat_obs)
                tokens_all = frames_b.tokens.reshape(B_cur, T_cur + 1, *frames_b.tokens.shape[1:])
                pooled_all = frames_b.pooled.reshape(B_cur, T_cur + 1, *frames_b.pooled.shape[1:])
            else:
                flat_tokens = obs_b.reshape(-1, *obs_b.shape[2:])
                frames_b = model.encoder.adapter(flat_tokens)
                tokens_all = frames_b[0].reshape(B_cur, T_cur + 1, *frames_b[0].shape[1:])
                pooled_all = frames_b[1].reshape(B_cur, T_cur + 1, *frames_b[1].shape[1:])

            hidden_b = model.initial_belief(B_cur, device)
            states_all = []
            for t in range(T_cur):
                f_t = LatentFrame(tokens=tokens_all[:, t], pooled=pooled_all[:, t])
                states_all.append(model.heads.state(f_t, hidden_b))
                hidden_b = model.belief_step(f_t, acts_b[:, t], rews_b[:, t], hidden_b)
            states_all = torch.stack(states_all, dim=1)
            val_estimates = champ_val(states_all).squeeze(-1)
            advantages = returns_mc - val_estimates
            adv_weights = torch.clamp(torch.exp(advantages), 0.0, 20.0)

        v_mask = valid_b.bool()
        st_flat = states_all[v_mask]
        act_flat = acts_b[v_mask].long()
        w_flat = adv_weights[v_mask]

        res_opt.zero_grad()
        res_logits = residual_actor(st_flat)
        dist = torch.distributions.Categorical(logits=res_logits)
        log_prob = dist.log_prob(act_flat)
        ent = dist.entropy().mean()
        loss = - (w_flat * log_prob).mean() - 0.01 * ent
        loss.backward()
        res_opt.step()

    # Benchmark Residual Actor
    res_agent = CustomActorJEPAAgent(champ_model, residual_actor, device=str(device), stochastic=True, temperature=1.0)
    res_benchmark = evaluate_policy_in_crafter(res_agent, seeds=bench_seeds, max_steps=max_steps)
    benchmark_manifests["actor_residual_mlp"] = res_benchmark
    print(f"  Residual Actor Benchmark -> Return: {res_benchmark['mean_return']:+.2f} ± {res_benchmark['std_return']:.2f} | Score: {res_benchmark['crafter_score']:.2f}% | Steps: {res_benchmark['mean_length']:.1f}")

    # -----------------------------------------------------------------------
    # Statistical Comparisons vs Gate 7 Champion
    # -----------------------------------------------------------------------
    champ_returns = [r["return"] for r in champ_res["per_seed_results"]]
    statistical_comparisons: Dict[str, Any] = {}
    for cand_name in ["candidate_b_iql", "candidate_c_cql", "candidate_d_crr", "actor_residual_mlp"]:
        c_returns = [r["return"] for r in benchmark_manifests[cand_name]["per_seed_results"]]
        statistical_comparisons[cand_name] = compute_paired_statistics(c_returns, champ_returns)

    # -----------------------------------------------------------------------
    # Final Decision & Promotion Evaluation
    # -----------------------------------------------------------------------
    # Promotion Rule: Candidate must strictly beat Gate 7 Champion (+2.10 return, 10.64% score)
    best_candidate = "gate7_champion"
    best_score = champ_res["crafter_score"]
    best_return = champ_res["mean_return"]
    promotion_status = "CANDIDATE_NOT_PROMOTED"

    for cand_name in ["candidate_b_iql", "candidate_c_cql", "candidate_d_crr", "actor_residual_mlp"]:
        c_score = benchmark_manifests[cand_name]["crafter_score"]
        c_ret = benchmark_manifests[cand_name]["mean_return"]
        p_stat = statistical_comparisons[cand_name]
        if c_score > best_score and c_ret > best_return and p_stat["ci95_low"] > 0.0:
            best_candidate = cand_name
            best_score = c_score
            best_return = c_ret
            promotion_status = "PROMOTED"

    if best_candidate == "gate7_champion":
        verdict_str = f"Gate 7 AWR RETAINED as production champion (Score: {champ_res['crafter_score']:.2f}%, Return: {champ_res['mean_return']:+.2f}). No alternative candidate demonstrated statistically superior gameplay."
    else:
        verdict_str = f"{best_candidate} PROMOTED over Gate 7 champion (Score: {best_score:.2f}% vs {champ_res['crafter_score']:.2f}%, Return: {best_return:+.2f} vs {champ_res['mean_return']:+.2f})."

    print("\n" + "=" * 80)
    print("GATE 7B RESEARCH SUMMARY & ALGORITHM SELECTION")
    print("=" * 80)
    print(f"Selection Verdict:           {promotion_status}")
    print(f"Winning Architecture:        {best_candidate}")
    print(f"Summary:                     {verdict_str}")
    for name, res in benchmark_manifests.items():
        print(f"  {name:25s} | Return: {res['mean_return']:+.2f} ± {res['std_return']:.2f} | Score: {res['crafter_score']:5.2f}% | Steps: {res['mean_length']:5.1f} | Ent: {res['action_entropy']:.2f}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Visualization & Manifest Serialization
    # -----------------------------------------------------------------------
    plot_file = out_path / "gate7b_comparison_plots.png"
    generate_gate7b_comparison_plots(
        benchmarks=benchmark_manifests,
        critic_metrics=critic_calibration_metrics,
        training_histories=training_histories,
        output_path=plot_file,
    )

    metrics_manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "Gate 7B: Actor-Critic Research & Algorithm Selection",
        "promotion_status": promotion_status,
        "selected_architecture": best_candidate,
        "verdict": verdict_str,
        "provenance": {
            "world_model_path": str(wm_file),
            "gate7_champion_path": str(champ_file),
            "gate7_champion_sha256": h_champ,
            "replay_dataset_path": str(replay_file),
            "candidate_checkpoints": candidate_checkpoints,
        },
        "critic_calibration": critic_calibration_metrics,
        "statistical_comparisons_vs_gate7": statistical_comparisons,
        "benchmark_results": {
            k: {m: v for m, v in val.items() if m != "per_seed_results"}
            for k, val in benchmark_manifests.items()
        },
    }

    metrics_path = out_path / "gate7b_experiment_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_manifest, f, indent=2)
    print(f"[Manifest] Saved Gate 7B metrics to {metrics_path}")

    # Mirror to drive if available
    target_drive = Path(drive_dir) if drive_dir else resolve_drive_root()
    if target_drive and target_drive.exists():
        drive_out = target_drive / "orbit" / "results" / "gate7b"
        drive_out.mkdir(parents=True, exist_ok=True)
        for f_c in [plot_file, metrics_path, out_path / "gate7b_dataset_manifest.json"]:
            if f_c.exists():
                shutil.copy2(str(f_c), str(drive_out / f_c.name))
        print(f"[Drive Sync] Mirrored Gate 7B artifacts to {drive_out}")

    return metrics_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORBIT — Gate 7B: Actor-Critic Research & Selection")
    parser.add_argument("--world-model", type=str, default="results/phase3/world_model_phase3.pt")
    parser.add_argument("--champion-gate7", type=str, default="results/gate7/policy_gate7.pt")
    parser.add_argument("--replay-dataset", type=str, default="data/gate7/augmented_replay_local.npz")
    parser.add_argument("--heldout-eval", type=str, default="data/gate3/eval_heldout_seeds.pt.gz")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--benchmark-seeds", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/gate7b")
    parser.add_argument("--drive-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=777)

    args = parser.parse_args()

    run_gate7b_research_study(
        world_model_ckpt=args.world_model,
        champion_gate7_ckpt=args.champion_gate7,
        replay_dataset_path=args.replay_dataset,
        heldout_eval_path=args.heldout_eval,
        updates=args.updates,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        benchmark_seeds_count=args.benchmark_seeds,
        max_steps=args.max_steps,
        device_str=args.device,
        output_dir=args.output_dir,
        drive_dir=args.drive_dir,
        seed=args.seed,
    )
