#!/usr/bin/env python3
"""
Step 2: Diagnostic Gameplay Failure Path Tracer for Learn-a-World v3.

Strict Zero-Information-Leakage Policy:
- The planner acts purely on raw visual frames (64x64x3 RGB) and recurrent belief.
- Privileged environment info (inventory, ground-truth achievements) is recorded
  STRICTLY as an external diagnostic log AFTER each step.
- Privileged information is NEVER passed into the planner or belief.

Traces 10 locked seeds (2050–2059) to identify the first failed link in Crafter's causal chain:
  wood collection (>= 2)
  -> table placement
  -> return to table (crafting affordance)
  -> wood-pickaxe crafting
  -> stone detection
  -> return to stone
  -> stone mining
  -> survival

Outputs:
- results/diagnostics/gameplay_failure_trace_10ep.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

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


def _resolve_checkpoint(ckpt_path: str, drive_dir: Optional[str] = None) -> str:
    p = Path(ckpt_path)
    if p.exists() and p.is_file():
        return str(p.resolve())
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
        if cand.exists() and cand.is_file():
            return str(cand.resolve())
    return str(ckpt_path)


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


def run_failure_path_trace(
    checkpoint_path: str,
    calibrator_path: str = "results/calibration/event_head_calibrator.json",
    num_episodes: int = 10,
    seed_start: int = 2050,
    max_steps: int = 1000,
    horizon: int = 10,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    prior_mix: float = 0.5,
    uncertainty_weight: float = 0.5,
    novelty_weight: float = 0.02,
    device: str = "cpu",
    output_json: str = "results/diagnostics/gameplay_failure_trace_10ep.json",
    drive_dir: Optional[str] = None,
    allow_uncalibrated_diagnostic: bool = False,
) -> Dict[str, Any]:
    dev = torch.device(device)

    print("=" * 80)
    print("      STEP 2: GAMEPLAY FAILURE PATH TRACER (SEEDS 2050..2059)")
    print(f"      Checkpoint: {checkpoint_path}")
    print(f"      Episodes: {num_episodes} | Seeds: {seed_start}..{seed_start + num_episodes - 1}")
    print(f"      Planner: Recurrent MPC (H={horizon}, C={num_candidates}, CEM={cem_iterations})")
    print(f"      Device: {dev}")
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
    resolved_ckpt = _resolve_checkpoint(checkpoint_path, drive_dir)
    model.load_checkpoint(resolved_ckpt, device=dev)
    model.to(dev)
    model.eval()

    # 1.1 Calibrator Verification
    cal_resolved = _resolve_calibrator(calibrator_path, drive_dir)
    cal_sha256 = None
    if cal_resolved is not None and not allow_uncalibrated_diagnostic:
        try:
            from models.calibration import EventHeadCalibrator
            cal = EventHeadCalibrator.load(str(cal_resolved))
            model.set_calibrator(cal)
            cal_sha256 = _sha256_file(cal_resolved)
            print(f"[Protocol Lock] Event-head calibrator verified: {cal_resolved}")
        except Exception as e:
            raise RuntimeError(f"Calibrator failed to parse from {cal_resolved}: {e}") from e
    else:
        if not allow_uncalibrated_diagnostic:
            raise RuntimeError(f"Calibrator required but not found at '{calibrator_path}'")
        model.set_calibrator(None)
        print("[Protocol Warning] Running uncalibrated diagnostic trace.")

    ckpt_p = Path(resolved_ckpt)
    ckpt_sha256 = _sha256_file(ckpt_p) if ckpt_p.exists() else None
    git_commit = _get_git_commit()

    # 2. Build Planner & Belief
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
        action_persistence=0.5,
        num_actions=17,
        frontier_threshold=0.25,
        common_threshold=0.50,
        vital_weight=0.5,
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256, frontier_threshold=0.25, common_threshold=0.50)

    # 3. Environment
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

    episode_traces: List[Dict[str, Any]] = []
    episode_link_autopsies: List[Dict[str, Any]] = []

    t_start = time.time()

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        belief.reset()

        step_records: List[Dict[str, Any]] = []
        ep_reward = 0.0
        steps = 0
        t_ep_start = time.time()

        max_wood = 0
        table_placed = False
        pickaxe_crafted = False
        stone_collected = False
        sword_crafted = False

        for step in range(max_steps):
            steps += 1

            # Strict Zero Info Leak: plan purely from visual observation + belief
            action, debug = planner.plan(obs, belief)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Privileged diagnostic inspection (NEVER passed into planner or belief)
            inv_diag = dict(info.get("inventory", {})) if isinstance(info, dict) else {}
            ach_diag = dict(info.get("achievements", {})) if isinstance(info, dict) else {}

            curr_wood = inv_diag.get("wood", 0)
            if curr_wood > max_wood:
                max_wood = curr_wood
            if ach_diag.get("place_table", 0) > 0:
                table_placed = True
            if ach_diag.get("make_wood_pickaxe", 0) > 0:
                pickaxe_crafted = True
            if ach_diag.get("collect_stone", 0) > 0:
                stone_collected = True
            if ach_diag.get("make_wood_sword", 0) > 0:
                sword_crafted = True

            # Model feature extraction for diagnostic logging
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                obs_t = obs_t.to(dev)
                z_t = model.encode(obs_t, use_target=True)
                act_t = torch.tensor([action], device=dev, dtype=torch.long)
                h_prev = belief.get_h_tensor(device=dev).unsqueeze(0)
                s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
                mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
                event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)

                # Q-values across all 17 actions from current state
                all_actions = torch.arange(model.num_actions, device=dev).view(1, -1)
                s_expand = s_t.expand(model.num_actions, -1)
                act_all = all_actions.squeeze(0)
                q_all = model.q_mean(s_expand, act_all, use_target=True).cpu().numpy().tolist()

                # Policy prior distribution
                prior_logits = model.policy_prior(s_t, mask_t)
                prior_probs = F.softmax(prior_logits, dim=-1).squeeze(0).cpu().numpy().tolist()

                lat_norm = float(torch.norm(z_t).item())

            # Update strict belief
            belief.update(event_probs, reward, next_h=next_h.squeeze(0))
            obs = next_obs

            # Record step trace
            step_record = {
                "seed": seed,
                "step": steps,
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "inventory_diagnostic": inv_diag,
                "achievement_diagnostic": {k: v for k, v in ach_diag.items() if v > 0},
                "event_probabilities": [round(float(p), 4) for p in event_probs.cpu().numpy()],
                "binary_belief_mask": [int(m) for m in belief.get_binary_mask()],
                "soft_belief_mask": [round(float(sm), 4) for sm in belief.get_soft_mask()],
                "planner_debug": debug,
                "q_values": [round(q, 4) for q in q_all],
                "policy_prior": [round(pr, 4) for pr in prior_probs],
                "latent_norm": round(lat_norm, 4),
            }
            step_records.append(step_record)

            if done:
                break

        ep_duration = time.time() - t_ep_start

        # Determine First Failed Link for this episode:
        # Causal Chain:
        # 1. wood_collection (>= 2 wood accumulated)
        # 2. table_placement
        # 3. wood_pickaxe_crafting
        # 4. stone_mining
        furthest_reached = "wake_up"
        first_failed_link = "none"

        if max_wood >= 2:
            furthest_reached = "wood_collected_ge_2"
            if table_placed:
                furthest_reached = "table_placed"
                if pickaxe_crafted:
                    furthest_reached = "wood_pickaxe_crafted"
                    if stone_collected:
                        furthest_reached = "stone_collected"
                    else:
                        first_failed_link = "stone_mining (pickaxe crafted but no stone mined)"
                else:
                    first_failed_link = "wood_pickaxe_crafting (table placed but pickaxe never crafted)"
            else:
                first_failed_link = "table_placement (>= 2 wood gathered but table never placed)"
        else:
            first_failed_link = "wood_collection (failed to gather >= 2 wood)"

        autopsy = {
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": steps,
            "reward": round(ep_reward, 2),
            "max_wood": max_wood,
            "table_placed": table_placed,
            "pickaxe_crafted": pickaxe_crafted,
            "stone_collected": stone_collected,
            "sword_crafted": sword_crafted,
            "furthest_reached": furthest_reached,
            "first_failed_link": first_failed_link,
            "duration_sec": round(ep_duration, 2),
        }
        episode_link_autopsies.append(autopsy)
        episode_traces.append({
            "summary": autopsy,
            "steps": step_records,
        })

        print(
            f"Ep {ep_idx + 1:02d}/10 (Seed {seed}) | Steps: {steps:<4d} | Reward: {ep_reward:<4.1f} | "
            f"Wood: {max_wood} | Table: {table_placed} | Pick: {pickaxe_crafted} | Stone: {stone_collected} | "
            f"Failed: {first_failed_link}"
        )

    env.close()
    total_time = time.time() - t_start

    # Aggregate failure distribution
    failure_counts: Dict[str, int] = {}
    for a in episode_link_autopsies:
        fl = a["first_failed_link"]
        failure_counts[fl] = failure_counts.get(fl, 0) + 1

    summary = {
        "protocol_id": "strict_visual_jepa_failure_trace_v1",
        "git_commit": git_commit,
        "checkpoint": resolved_ckpt,
        "checkpoint_sha256": ckpt_sha256,
        "calibrator": str(cal_resolved) if cal_resolved else None,
        "calibrator_sha256": cal_sha256,
        "track": "strict_visual_jepa",
        "privileged_inventory_used": False,
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "total_runtime_seconds": round(total_time, 2),
        "failure_distribution": failure_counts,
        "episode_autopsies": episode_link_autopsies,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({
            "metadata": summary,
            "episodes": episode_traces,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("                    FAILURE PATH DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Total Traced Episodes: {num_episodes} (Seeds {seed_start}..{seed_start + num_episodes - 1})")
    print("First Failed Link Breakdown:")
    for link, cnt in failure_counts.items():
        pct = (cnt / num_episodes) * 100.0
        print(f"  {link:<60}: {cnt:2d} / {num_episodes} ({pct:5.1f}%)")
    print(f"\n--> Full trace saved to: {out_p}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Trace gameplay failure path on locked seeds.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_step1_stone.pt")
    parser.add_argument("--calibrator_path", type=str, default="results/calibration/event_head_calibrator.json")
    parser.add_argument("--allow_uncalibrated_diagnostic", action="store_true", default=False)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/gameplay_failure_trace_10ep.json")
    parser.add_argument("--drive_dir", type=str, default=None)
    args = parser.parse_args()

    run_failure_path_trace(
        checkpoint_path=args.checkpoint,
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
    )
