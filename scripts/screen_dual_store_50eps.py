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
import os
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
from dual_store.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from dual_store.mpc_v3 import RecurrentMPCPlanner
from dual_store.playtime_recorder import PlaytimeRecorder


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


def collect_subgoal_trace(raw_entries: List[Any], cap: int = 20) -> List[Dict[str, Any]]:
    """Bounded per-episode subgoal provenance (Stage 8 / plan §7 AC8).

    raw_entries: (step, status, target) tuples appended per step when the
    compositional planner reported an active subgoal. Entries with an empty
    status are dropped and the trace is capped so episode records stay bounded.
    """
    trace: List[Dict[str, Any]] = []
    for entry in raw_entries:
        step, status, target = entry
        if not status:
            continue
        trace.append({"step": int(step), "status": str(status), "target": str(target)})
        if len(trace) >= cap:
            break
    return trace


def _atomic_save_bank(bank: Any, path: Path) -> None:
    """Stage 8: bank persistence is atomic — tmp file + os.replace (caller's job per bank contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    bank.save(str(tmp_path))
    os.replace(tmp_path, path)


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


def _resolve_bank(bank_path: Optional[str], drive_dir: Optional[str] = None) -> Optional[str]:
    if not bank_path:
        return None
    p = Path(bank_path)
    
    # Priority list of search filenames: requested name first, then canonical bank names
    search_names = [p.name]
    if "consolidation" in p.name.lower() or "bank" not in p.name.lower():
        search_names.extend([
            "phase2_causal_memory_bank_v2.json",
            "phase2_causal_memory_bank.json",
        ])

    candidate_roots = [
        Path("."),
        REPO_ROOT,
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "results" / "diagnostics",
        Path("/content/drive/MyDrive/learn-a-world/checkpoints"),
        Path("/content/drive/MyDrive/learn-a-world/results/diagnostics"),
        Path("/content/drive/MyDrive/learn-a-world"),
        Path("/content/drive/MyDrive/checkpoints"),
        Path("/content/drive/MyDrive"),
        Path("/content/checkpoints"),
        Path("/content"),
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidate_roots.extend([dp, dp / "checkpoints", dp / "results" / "diagnostics"])

    candidates = [p]
    for sname in search_names:
        for root in candidate_roots:
            candidates.append(root / sname)

    for cand in candidates:
        if cand.exists() and cand.is_file():
            try:
                with open(cand, "r") as f:
                    meta = json.load(f)
                if "prototypes" in meta and len(meta["prototypes"]) > 0:
                    return str(cand)
                if "bank_path" in meta and meta["bank_path"]:
                    sub_cand = _resolve_bank(meta["bank_path"], drive_dir)
                    if sub_cand and Path(sub_cand).exists():
                        return sub_cand
            except Exception:
                pass
            return str(cand)
    return str(bank_path)


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
    death_risk_factor: float = 0.5,
    harvest_bonus: float = 0.15,
    action_persistence: float = 0.5,
    use_transition_model: bool = False,
    transition_scoring: bool = False,
    transition_weight: float = 0.5,
    transition_done_penalty: float = 2.0,
    transition_resource_weight: float = 0.25,
    causal_memory_bank_path: Optional[str] = None,
    online_learning: bool = False,
    online_bank_path: str = "results/phase2_online_bank.json",
    resume_bank: bool = False,
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

    # 2. Build Optional Phase 2 Compositional Causal Planner
    # Stage 8: with --resume_bank, a persisted online bank takes precedence over
    # --causal_memory_bank so game N+1 resumes from game N's lessons.
    comp_planner = None
    bank = None
    bank_path_loaded: Optional[str] = None
    bank_sha256: Optional[str] = None

    if online_learning and resume_bank:
        online_p = Path(online_bank_path)
        if online_p.exists():
            bank_path_loaded = str(online_p)
        else:
            print(
                f"[Phase 2 Online Learning] --resume_bank set but no bank file at {online_p}; "
                "falling back to --causal_memory_bank or a fresh bank."
            )

    if bank_path_loaded is None and causal_memory_bank_path:
        resolved_bank = _resolve_bank(causal_memory_bank_path, drive_dir)
        bank_p = Path(resolved_bank) if resolved_bank else None
        if bank_p and bank_p.exists():
            bank_path_loaded = str(bank_p)
        else:
            print(f"[Phase 2 Causal Memory] Warning: Bank path not found: {causal_memory_bank_path}")

    if bank_path_loaded is not None:
        from dual_store.causal_memory import RelationHypothesisBank
        from dual_store.compositional_planner import CompositionalCausalPlanner
        bank = RelationHypothesisBank(device=dev)
        bank.load(bank_path_loaded)
        bank_sha256 = _sha256_file(Path(bank_path_loaded))  # sha of INITIAL bank bytes (pre-online updates)
        comp_planner = CompositionalCausalPlanner(bank, device=dev)
        print(f"[Phase 2 Causal Memory] Loaded {len(bank)} prototypes from: {bank_path_loaded}")
    elif online_learning:
        # Online arm with no resumable/offline bank: start empty; lessons persist
        # to --online_bank_path after every episode (the cross-game loop).
        from dual_store.causal_memory import RelationHypothesisBank
        from dual_store.compositional_planner import CompositionalCausalPlanner
        bank = RelationHypothesisBank(device=dev)
        comp_planner = CompositionalCausalPlanner(bank, device=dev)
        print(f"[Phase 2 Online Learning] Starting from a fresh empty bank; lessons will persist to: {online_bank_path}")

    # 3. Build Planner & Belief with Adaptive Frontier Thresholds
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
        compositional_planner=comp_planner,
        semantic_memory=None,
        device=dev,
    )
    belief = AgentBeliefV3(
        recurrent_dim=256,
        frontier_threshold=frontier_threshold,
        common_threshold=common_threshold,
        strict_inference=True,
    )

    # 2.5 Stage 8: playtime recorder for the online cross-game learning loop.
    recorder: Optional[PlaytimeRecorder] = None
    if online_learning:
        if bank is None:
            raise RuntimeError("[Phase 2 Online Learning] online_learning requires a relation bank (loaded or fresh).")
        recorder = PlaytimeRecorder()

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
        subgoal_trace_raw: List[Any] = []  # (step, status, target) for the bounded subgoal provenance

        for step in range(max_steps):
            steps += 1

            # Strict Zero Info Leak: plan only from raw pixels + belief
            action, debug = planner.plan(obs, belief)
            if online_learning and isinstance(debug, dict) and debug.get("active_subgoal_status", ""):
                subgoal_trace_raw.append(
                    (
                        step,
                        str(debug.get("active_subgoal_status", "")),
                        str(debug.get("active_subgoal_target", "")),
                    )
                )

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
                if online_learning:
                    # Stage 8: playtime grounded state comes from the model's state
                    # readout, never from engine inventory (strict protocol).
                    x_hat, _ = model.predict_state(s_t)

            if online_learning:
                mask_before = belief.get_mask()  # copied BEFORE belief.update

            belief.update(event_probs, reward, next_h=next_h.squeeze(0))
            if hasattr(belief, "update_working_memory"):
                belief.update_working_memory(
                    z_t=z_t,
                    action=action,
                    reward=reward,
                    next_h=next_h.squeeze(0),
                )
            obs = next_obs

            if online_learning and recorder is not None:
                # Strict protocol: unlock labels come from the belief only —
                # info['achievements'] is scoring-only and never enters the recorder.
                hist = belief.history[-1]
                recorder.record_step(
                    z_t=z_t,
                    x_hat=x_hat,
                    action=action,
                    reward=reward,
                    done=done,
                    mask_before=mask_before,
                    newly_unlocked=hist["newly_unlocked"],
                    fallback_unlocked=hist.get("fallback_unlocked", []),
                    event_probs=event_probs.detach().cpu().tolist(),
                    thresholds=np.asarray(belief.thresholds).tolist(),
                )

            if done:
                break

        ep_duration = time.time() - t_ep_start
        # achievements are read here for SCORING ONLY (protocol); they never enter the online recorder.
        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        unlocked = [k for k, v in ach.items() if v > 0]

        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

        ep_record: Dict[str, Any] = {
            "episode": ep_idx + 1,
            "seed": seed,
            "steps": steps,
            "reward": round(ep_reward, 2),
            "unlocked": unlocked,
            "duration_sec": round(ep_duration, 2),
        }

        online_stats: Dict[str, int] = {}
        if online_learning and recorder is not None and bank is not None:
            # Stage 8: learn → refresh edges → atomic persist, every episode, so
            # game N's lessons shape game N+1 (bank survives across episodes).
            online_stats = recorder.end_episode(bank)
            _atomic_save_bank(bank, Path(online_bank_path))
            ep_record["online_stats"] = dict(online_stats)
            ep_record["subgoal_trace"] = collect_subgoal_trace(subgoal_trace_raw, cap=20)

        episode_records.append(ep_record)

        print(
            f"Ep {ep_idx + 1:02d}/{num_episodes} (Seed {seed}) | "
            f"Steps: {steps:<4d} | Reward: {ep_reward:<5.1f} | "
            f"Achievements ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'None'} "
            f"({ep_duration:.1f}s)"
        )

        if online_learning:
            stats_view = online_stats if isinstance(online_stats, dict) else {}
            print(
                f"[Phase 2 Online Learning] episode {ep_idx + 1}: "
                f"created={stats_view.get('created', 0)} "
                f"confirmed={stats_view.get('confirmed', 0)} "
                f"denied={stats_view.get('denied', 0)}"
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
        "causal_memory_enabled": bool(comp_planner is not None),
        "causal_memory_prototypes": len(comp_planner.hypothesis_bank) if comp_planner is not None else 0,
        "online_learning": bool(online_learning),
        "online_bank_path": str(online_bank_path) if online_learning else None,
        "bank_path_loaded": bank_path_loaded,
        "bank_sha256": bank_sha256,
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

    if online_learning:
        # Stage 8 provenance: sha of the bank AFTER the last episode's online updates.
        summary["final_bank_sha256"] = _sha256_file(Path(online_bank_path))

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
            if online_learning and Path(online_bank_path).exists():
                drive_bank = dp / "results" / Path(online_bank_path).name
                drive_bank.parent.mkdir(parents=True, exist_ok=True)
                if Path(online_bank_path).resolve() != drive_bank.resolve():
                    shutil.copy2(str(online_bank_path), str(drive_bank))
                    print(f"[Drive] Mirrored online bank to: {drive_bank}")

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print(f"--- BEGIN {out_p.name} ---")
    print(json.dumps(summary, indent=2))
    print(f"--- END {out_p.name} ---\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-episode screening of JEPA v3.")
    parser.add_argument(
        "--checkpoint",
        "--checkpoint_path",
        dest="checkpoint",
        type=str,
        default="checkpoints/jepa_crafter_v3_state_readout_v2.pt",
    )
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
    parser.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        default=None,
        help="Start and end seeds inclusive, e.g. 2050 2099 (overrides --seed_start and --num_episodes).",
    )
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
    parser.add_argument("--death_risk_factor", type=float, default=0.5, help="Continuation decay rate for lethal damage in imagination.")
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
    parser.add_argument(
        "--causal_memory_bank",
        "--causal_memory_bank_path",
        dest="causal_memory_bank",
        type=str,
        default=None,
        help="Path to Phase 2 learned RelationHypothesisBank JSON.",
    )
    parser.add_argument(
        "--online_learning",
        action="store_true",
        default=False,
        help="Stage 8: record strict playtime transitions and update the relation bank after every episode (bank persists across episodes).",
    )
    parser.add_argument(
        "--online_bank_path",
        type=str,
        default="results/phase2_online_bank.json",
        help="Path of the persistent online relation bank (atomically rewritten each episode under --online_learning; loaded when --resume_bank is set).",
    )
    parser.add_argument(
        "--resume_bank",
        action="store_true",
        default=False,
        help="Resume from --online_bank_path if that file exists, instead of --causal_memory_bank.",
    )
    parser.set_defaults(standardize_scores=False)
    args = parser.parse_args()

    if args.seeds is not None:
        args.seed_start = int(args.seeds[0])
        args.num_episodes = int(args.seeds[1]) - int(args.seeds[0]) + 1

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
        causal_memory_bank_path=args.causal_memory_bank,
        online_learning=args.online_learning,
        online_bank_path=args.online_bank_path,
        resume_bank=args.resume_bank,
    )
