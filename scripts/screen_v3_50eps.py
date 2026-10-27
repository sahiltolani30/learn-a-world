#!/usr/bin/env python3
"""
50-Episode Gameplay Screening of JEPA v3 on Untouched Seeds (2050-2099).
Strict zero-information-leakage policy:
- No handcoded heuristics or ground-truth state cheats.
- Agent observes only visual frames (64x64x3 RGB) and scalar rewards.
- Temporal belief memory maintained by RecurrentCore + AgentBeliefV3.
- Action planning executed by RecurrentMPCPlanner (prior-guided CEM + Clipped Double-Q).

Outputs:
- results/screen_v3_50eps_results.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # registers Crafter-v0
from eval.metrics import CRAFTER_ACHIEVEMENTS, compute_crafter_score
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner


def _sha256_file(filepath: Path) -> Optional[str]:
    if not filepath.exists() or not filepath.is_file():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _resolve_calibrator(cal_path: str, drive_dir: Optional[str] = None) -> Optional[Path]:
    p = Path(cal_path)
    if p.exists() and p.is_file():
        return p.resolve()
    candidates = [
        REPO_ROOT / cal_path,
        REPO_ROOT / "results" / "calibration" / p.name,
        Path("/content/drive/MyDrive/learn-a-world") / cal_path,
        Path("/content/drive/MyDrive/learn-a-world/results/calibration") / p.name,
        Path("/content/learn-a-world") / cal_path,
        Path("/content/learn-a-world/results/calibration") / p.name,
        Path("/content") / cal_path,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / cal_path,
            dp / "results" / "calibration" / p.name,
        ])
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return None


def _resolve_checkpoint(ckpt_path: str, drive_dir: Optional[str] = None) -> str:
    p = Path(ckpt_path)
    if p.exists():
        return str(p)
    candidates = [
        REPO_ROOT / ckpt_path,
        REPO_ROOT / "checkpoints" / p.name,
        Path("/content/drive/MyDrive/learn-a-world") / ckpt_path,
        Path("/content/drive/MyDrive/learn-a-world/checkpoints") / p.name,
        Path("/content/drive/MyDrive/learn-a-world") / p.name,
        Path("/content/drive/MyDrive") / ckpt_path,
        Path("/content/drive/MyDrive/checkpoints") / p.name,
        Path("/content/checkpoints") / p.name,
        Path("/content") / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / ckpt_path,
            dp / "checkpoints" / p.name,
            dp / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(ckpt_path)


def run_screening(
    checkpoint_path: str = "checkpoints/jepa_crafter_v3_phase2.pt",
    calibrator_path: str = "results/calibration/event_head_calibrator.json",
    allow_uncalibrated_diagnostic: bool = False,
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    prior_mix: float = 0.5,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.02,
    device: str = "cpu",
    output_json: str = "results/screen_v3_50eps_results.json",
    drive_dir: Optional[str] = None,
    denormalize_q: bool = False,
    tte_weight: float = 0.0,
    standardize_scores: bool = False,
    frontier_threshold: float = 0.25,
    common_threshold: float = 0.50,
    vital_weight: float = 0.5,
    death_risk_factor: float = 0.0,
    harvest_bonus: float = 0.15,
    action_persistence: float = 0.5,
    use_transition_model: bool = False,
    transition_scoring: bool = False,
    transition_weight: float = 0.5,
    transition_done_penalty: float = 2.0,
    transition_resource_weight: float = 0.25,
) -> Dict[str, Any]:
    dev = torch.device(device)

    print("=" * 80)
    print("      LEARN-A-WORLD v3: 50-EPISODE GAMEPLAY SCREENING (SEEDS 2050-2099)")
    print(f"      Checkpoint: {checkpoint_path}")
    print(f"      Episodes: {num_episodes} | Seeds: {seed_start}..{seed_start + num_episodes - 1}")
    print(f"      Planner: Recurrent MPC (H={horizon}, C={num_candidates}, CEM={cem_iterations})")
    print(f"      Frontier Threshold: {frontier_threshold} | Common Threshold: {common_threshold}")
    print(f"      Denormalize Q: {denormalize_q} | TTE Weight: {tte_weight} | Standardize Scores: {standardize_scores}")
    print(f"      Vital Weight: {vital_weight} | Death Risk Factor: {death_risk_factor} | Harvest Bonus: {harvest_bonus}")
    print(f"      Transition Model: {use_transition_model} | Consequence Scoring: {transition_scoring}")
    print(f"      Action Persistence: {action_persistence} | Device: {dev}")
    print("=" * 80)

    # 1. Load Model
    model = JEPAv3(
        latent_dim=128,
        recurrent_dim=256,
        num_actions=17,
        action_emb_dim=32,
        hidden_dim=256,
        q_heads=3,
        q_bins=51,
        q_max=12.0,
    )
    model.load_checkpoint(checkpoint_path, device=dev)
    model.to(dev)
    model.eval()

    # 1.1 Strict Protocol Lock: Calibrator Verification
    cal_resolved = _resolve_calibrator(calibrator_path, drive_dir)
    cal_sha256 = None
    if cal_resolved is not None and not allow_uncalibrated_diagnostic:
        try:
            from models.calibration import EventHeadCalibrator
            cal = EventHeadCalibrator.load(str(cal_resolved))
            model.set_calibrator(cal)
            cal_sha256 = _sha256_file(cal_resolved)
            print(f"[Protocol Lock] Event-head calibrator verified: {cal_resolved}")
            print(f"  -> Calibrator SHA256: {cal_sha256}")
        except Exception as e:
            raise RuntimeError(
                f"[Protocol Lock Violation] Calibrator failed to parse from {cal_resolved}: {e}. "
                "Benchmark evaluation requires a valid calibrator. "
                "Pass --allow_uncalibrated_diagnostic ONLY for explicit non-benchmark diagnostics."
            ) from e
    else:
        if not allow_uncalibrated_diagnostic:
            raise RuntimeError(
                f"[Protocol Lock Violation] Calibrator not found at '{calibrator_path}' "
                f"(checked repo root: {REPO_ROOT}, drive: {drive_dir}). "
                "Benchmark evaluation requires an explicit, verified calibrator. "
                "Pass --allow_uncalibrated_diagnostic ONLY for explicit non-benchmark diagnostics."
            )
        # Explicit diagnostic mode: detach any auto-loaded calibrator so it runs truly uncalibrated
        model.set_calibrator(None)
        cal_resolved = None
        cal_sha256 = None
        print(f"[Protocol Warning] Running in uncalibrated diagnostic mode! Calibrator detached. Result is non-benchmark.")

    ckpt_p = Path(checkpoint_path)
    ckpt_sha256 = _sha256_file(ckpt_p) if ckpt_p.exists() else None
    git_commit = _get_git_commit()

    is_calibrated = (model.event_head.calibrator is not None)
    if not is_calibrated and not allow_uncalibrated_diagnostic:
        raise RuntimeError(
            "[Protocol Lock Violation] Model event head remains uncalibrated! "
            "Benchmark evaluations must be calibrated. Never silently continue uncalibrated."
        )

    protocol_id = "strict_visual_jepa_clean_v2" if (is_calibrated and not allow_uncalibrated_diagnostic) else "uncalibrated_diagnostic_not_a_benchmark"
    print(f"[Protocol Lock] Protocol ID: {protocol_id} (Git: {git_commit[:8]}, Checkpoint SHA: {ckpt_sha256[:12] if ckpt_sha256 else 'none'}...)")

    # 2. Build Planner & Belief with Adaptive Frontier Thresholds
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=cem_iterations,
        elite_ratio=0.15,
        gamma=0.99,
        uncertainty_weight=uncertainty_weight,
        novelty_weight=novelty_weight,
        prior_mix=prior_mix,
        action_persistence=action_persistence,
        num_actions=17,
        denormalize_q=denormalize_q,
        tte_weight=tte_weight,
        standardize_scores=standardize_scores,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
        vital_weight=vital_weight,
        death_risk_factor=death_risk_factor,
        harvest_bonus=harvest_bonus,
        strict_inference=True,
        use_transition_model=use_transition_model,
        transition_scoring=transition_scoring,
        transition_weight=transition_weight,
        transition_done_penalty=transition_done_penalty,
        transition_resource_weight=transition_resource_weight,
        semantic_memory=None,
        device=dev,
    )
    belief = AgentBeliefV3(
        recurrent_dim=256,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
        strict_inference=True,
    )

    # 3. Environment
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    episode_records: List[Dict[str, Any]] = []
    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    t_start = time.time()

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()

        ep_reward = 0.0
        steps = 0
        t_ep_start = time.time()

        for step in range(max_steps):
            steps += 1

            # Strict Zero Info Leak: plan only from raw pixels + belief
            action, debug = planner.plan(obs, belief)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Update belief using RecurrentCore + EventHead
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(dev)
                z_t = model.encode(obs_t, use_target=True)
                act_t = torch.tensor([action], device=dev, dtype=torch.long)
                h_prev = belief.get_h_tensor(device=dev).unsqueeze(0)
                s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
                mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
                event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)

            belief.update(event_probs, reward, next_h=next_h.squeeze(0))
            obs = next_obs

            if done:
                break

        ep_duration = time.time() - t_ep_start
        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        unlocked = [k for k, v in ach.items() if v > 0]

        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

        episode_records.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": steps,
            "reward": round(ep_reward, 2),
            "unlocked": unlocked,
            "duration_sec": round(ep_duration, 2),
        })

        print(
            f"Ep {ep_idx + 1:02d}/{num_episodes} (Seed {seed}) | "
            f"Steps: {steps:<4d} | Reward: {ep_reward:<5.1f} | "
            f"Achievements ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'None'} "
            f"({ep_duration:.1f}s)"
        )

    env.close()
    total_time = time.time() - t_start

    score_stats = compute_crafter_score(achievement_records)

    summary = {
        "protocol_id": protocol_id,
        "git_commit": git_commit,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": ckpt_sha256,
        "calibrator": str(cal_resolved) if cal_resolved else None,
        "calibrator_sha256": cal_sha256,
        "track": "strict_visual_jepa",
        "strict_inference": True,
        "privileged_inventory_used": False,
        "privileged_achievements_used_for_action": False,
        "achievements_used_for_scoring_only": True,
        "semantic_memory_enabled": False,
        "adaptation_mode": "frozen",
        "seeds": [seed_start, seed_start + num_episodes - 1],
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "num_episodes": num_episodes,
        "planner_configuration": {
            "horizon": horizon,
            "num_candidates": num_candidates,
            "cem_iterations": cem_iterations,
            "elite_ratio": 0.15,
            "prior_mix": prior_mix,
            "action_persistence": action_persistence,
            "uncertainty_weight": uncertainty_weight,
            "novelty_weight": novelty_weight,
            "vital_weight": vital_weight,
            "death_risk_factor": death_risk_factor,
            "harvest_bonus": harvest_bonus,
            "denormalize_q": denormalize_q,
            "tte_weight": tte_weight,
            "standardize_scores": standardize_scores,
            "frontier_threshold": frontier_threshold,
            "common_threshold": common_threshold,
            "strict_inference": True,
            "use_transition_model": use_transition_model,
            "transition_scoring": transition_scoring,
            "transition_weight": transition_weight,
            "transition_done_penalty": transition_done_penalty,
            "transition_resource_weight": transition_resource_weight,
        },
        "use_transition_model": use_transition_model,
        "transition_scoring": transition_scoring,
        "ablation_mode": "active_consequence" if (use_transition_model and transition_scoring) else ("shadow_transition" if use_transition_model else "baseline_t2"),
        "horizon": horizon,
        "num_candidates": num_candidates,
        "cem_iterations": cem_iterations,
        "frontier_threshold": frontier_threshold,
        "common_threshold": common_threshold,
        "vital_weight": vital_weight,
        "action_persistence": action_persistence,
        "death_risk_factor": death_risk_factor,
        "harvest_bonus": harvest_bonus,
        "denormalize_q": denormalize_q,
        "tte_weight": tte_weight,
        "standardize_scores": standardize_scores,
        "total_runtime_seconds": round(total_time, 2),
        "mean_survival_length": float(np.mean(survival_lengths)),
        "std_survival_length": float(np.std(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "crafter_score": score_stats["score"],
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "unlocked_achievement_count": score_stats["unlocked_count"],
        "unlock_percentages": score_stats["unlock_percentages"],
        "episodes": episode_records,
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("                    SCREENING SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Time: {total_time:.1f}s | Mean Episode Time: {total_time / num_episodes:.1f}s")
    print(f"Mean Survival Length: {summary['mean_survival_length']:.1f} +/- {summary['std_survival_length']:.1f} steps")
    print(f"Mean Episode Reward: {summary['mean_episode_reward']:.2f} +/- {summary['std_episode_reward']:.2f}")
    print(f"Official Crafter Score: {summary['crafter_score']:.2f}%")
    print(f"Mean Achievements per Episode: {summary['mean_achievements_per_ep']:.2f}")
    print(f"Unique Achievements Unlocked: {summary['unlocked_achievement_count']} / 22")
    print("\nPer-Achievement Unlock Rates:")
    for ach in CRAFTER_ACHIEVEMENTS:
        pct = score_stats["unlock_percentages"].get(ach, 0.0)
        bar = "#" * int(round(pct / 5))
        print(f"  {ach:<22}: {pct:5.1f}% | {bar}")

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--> Full screening results saved to: {out_p}")

    if drive_dir:
        dp = Path(drive_dir)
        if dp.exists():
            drive_out = dp / "results" / out_p.name
            drive_out.parent.mkdir(parents=True, exist_ok=True)
            if out_p.resolve() != drive_out.resolve():
                shutil.copy2(str(out_p), str(drive_out))
                print(f"[Drive] Mirrored screening results to: {drive_out}")

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print(f"--- BEGIN {out_p.name} ---")
    print(json.dumps(summary, indent=2))
    print(f"--- END {out_p.name} ---\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-episode screening of JEPA v3.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_step1_stone.pt")
    parser.add_argument(
        "--calibrator_path",
        type=str,
        default="results/calibration/event_head_calibrator.json",
        help="Path to event head calibrator JSON. Required for strict benchmark runs.",
    )
    parser.add_argument(
        "--allow_uncalibrated_diagnostic",
        action="store_true",
        default=False,
        help="Explicit diagnostic flag allowing uncalibrated evaluation. Output will be marked as non-benchmark.",
    )
    parser.add_argument("--num_episodes", "--episodes", dest="num_episodes", type=int, default=50)
    parser.add_argument("--seed_start", "--start_seed", dest="seed_start", type=int, default=2050)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="results/screen_v3_step1_results.json")
    parser.add_argument("--drive_dir", type=str, default=None)
    parser.add_argument("--denormalize_q", action="store_true", default=False, help="Denormalize terminal Q to raw return scale.")
    parser.add_argument("--no_denormalize_q", dest="denormalize_q", action="store_false", help="Disable Q denormalization.")
    parser.add_argument("--tte_weight", type=float, default=0.0, help="Weight for learned time-to-event proximity gain.")
    parser.add_argument("--frontier_threshold", type=float, default=0.25, help="Activation threshold for rare frontier achievements.")
    parser.add_argument("--common_threshold", type=float, default=0.50, help="Activation threshold for common survival achievements.")
    parser.add_argument("--vital_weight", type=float, default=0.5, help="Scaling factor for vital health damage/healing.")
    parser.add_argument("--death_risk_factor", type=float, default=0.0, help="Continuation decay rate for lethal damage in imagination.")
    parser.add_argument("--harvest_bonus", type=float, default=0.15, help="Utility bonus for repeat harvesting of essential inputs.")
    parser.add_argument("--action_persistence", type=float, default=0.5, help="Directional action persistence probability in MPC.")
    parser.add_argument(
        "--standardize-scores",
        "--standardize_scores",
        dest="standardize_scores",
        action="store_true",
        help="Live per-step standardization of planner score terms.",
    )
    parser.add_argument(
        "--no-standardize-scores",
        "--no_standardize_scores",
        dest="standardize_scores",
        action="store_false",
        help="Disable score standardization and use absolute reward weights.",
    )
    parser.add_argument("--use_transition_model", action="store_true", default=False, help="Enable transition head evaluation during MPC rollouts.")
    parser.add_argument("--transition_scoring", action="store_true", default=False, help="Enable active transition consequence scoring in MPC (Ablation C).")
    parser.add_argument("--transition_weight", type=float, default=0.5, help="Weight for model-predicted transition reward.")
    parser.add_argument("--transition_done_penalty", type=float, default=2.0, help="Penalty factor for predicted termination probability.")
    parser.add_argument("--transition_resource_weight", type=float, default=0.25, help="Weight for predicted positive resource increments.")
    parser.set_defaults(standardize_scores=False)
    args = parser.parse_args()

    resolved_ckpt = _resolve_checkpoint(args.checkpoint, args.drive_dir)

    run_screening(
        checkpoint_path=resolved_ckpt,
        calibrator_path=args.calibrator_path,
        allow_uncalibrated_diagnostic=args.allow_uncalibrated_diagnostic,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        device=args.device,
        output_json=args.output_json,
        drive_dir=args.drive_dir,
        denormalize_q=args.denormalize_q,
        tte_weight=args.tte_weight,
        standardize_scores=args.standardize_scores,
        frontier_threshold=args.frontier_threshold,
        common_threshold=args.common_threshold,
        vital_weight=args.vital_weight,
        death_risk_factor=args.death_risk_factor,
        harvest_bonus=args.harvest_bonus,
        action_persistence=args.action_persistence,
        use_transition_model=args.use_transition_model,
        transition_scoring=args.transition_scoring,
        transition_weight=args.transition_weight,
        transition_done_penalty=args.transition_done_penalty,
        transition_resource_weight=args.transition_resource_weight,
    )
