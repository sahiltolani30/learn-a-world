#!/usr/bin/env python3
"""
scripts/sweep_inference_configurations.py
=========================================
Systematic Inference-Time Ablation & Hyperparameter Grid Engine.
Evaluates all permutations and combinations of planner mechanics on the exact same
trained checkpoint without retraining a single weight:

  - Tweak A: Value Scale Denormalization (denormalize_q: False vs True)
  - Tweak B: Learned Time-to-Event Guidance (tte_weight: 0.0, 0.2, 0.5)
  - Tweak C: Exploration Noise Damping (uncertainty / novelty weights)
  - Tweak D: Policy Prior Injection (prior_mix: 0.5 vs 0.75)

Evaluates:
  1. Fast Combinatorial Sweep on 120 Pickaxe-in-Hand States (~3-4 min).
  2. Leaderboard ranking by Action Agreement and Value Gap.
  3. Automatic gameplay screening of the Top Champion configuration.

Outputs:
  - results/diagnostics/planner_ablation_sweep.json
  - results/screen_champion_results.json
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)


def resolve_checkpoint(ckpt_path: str, drive_dir: Optional[str] = None) -> Path:
    p = Path(ckpt_path)
    if p.exists():
        return p
    candidates = [
        REPO_ROOT / ckpt_path,
        REPO_ROOT / "checkpoints" / p.name,
    ]
    if drive_dir:
        dp = Path(drive_dir)
        candidates.extend([
            dp / ckpt_path,
            dp / "checkpoints" / p.name,
            Path("/content") / "checkpoints" / p.name,
            Path("/content") / p.name,
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint: {ckpt_path}")


def load_probe_states(
    drive_dir: Optional[str] = None,
    max_states: int = 120,
) -> List[Dict[str, Any]]:
    """Loads and caches the 120 pickaxe-in-hand states from teacher and curiosity shards."""
    drive_p = Path(drive_dir) if drive_dir else REPO_ROOT
    shard_candidates = [
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_final.npz",
        drive_p / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
        REPO_ROOT / "data" / "sequences" / "shard_teacher_hybrid_0000.npz",
    ]
    cur_dirs = [
        drive_p / "data" / "curiosity_shards",
        REPO_ROOT / "data" / "curiosity_shards",
    ]
    cur_dir = cur_dirs[0] if (cur_dirs[0].exists() and len(list(cur_dirs[0].glob("*.npz"))) > 0) else cur_dirs[1]
    all_shards = [s for s in shard_candidates if s.exists()]
    if cur_dir.exists():
        t2_shards = sorted(cur_dir.glob("shard_curiosity_v3_t02_*.npz"))
        if not t2_shards:
            t2_shards = sorted(cur_dir.glob("*.npz"))
        all_shards.extend(t2_shards)

    targeted_episodes = []
    for sp in all_shards:
        try:
            data = np.load(str(sp), allow_pickle=True)
            obs = data["obs"]
            actions = data["actions"]
            events = data["events"]
            dones = data["dones"]
            masks_before = data["mask_before"] if "mask_before" in data else None

            n_trans = len(obs)
            ep_indices = np.where(dones)[0]
            if len(ep_indices) == 0 or ep_indices[-1] != (n_trans - 1):
                ep_indices = np.append(ep_indices, n_trans - 1)

            ep_start = 0
            for ep_end in ep_indices:
                ep_events = events[ep_start : ep_end + 1]
                ep_masks = masks_before[ep_start : ep_end + 1] if masks_before is not None else None

                has_stone = bool(np.any(ep_events[:, IDX_STONE] > 0))
                has_pickaxe = bool(
                    np.any(ep_events[:, IDX_WOOD_PICKAXE] > 0)
                    or (ep_masks is not None and np.any(ep_masks[:, IDX_WOOD_PICKAXE] > 0))
                )

                if has_stone or has_pickaxe:
                    targeted_episodes.append({
                        "shard": sp.name,
                        "has_stone": has_stone,
                        "obs": obs[ep_start : ep_end + 1],
                        "actions": actions[ep_start : ep_end + 1],
                        "events": ep_events,
                        "masks_before": ep_masks,
                    })
                ep_start = ep_end + 1
        except Exception:
            continue

    pickaxe_states = []
    for ep in targeted_episodes:
        ep_obs = ep["obs"]
        ep_ev = ep["events"]
        ep_masks = ep["masks_before"]
        has_stone = ep["has_stone"]
        stone_steps = np.where(ep_ev[:, IDX_STONE] > 0)[0]
        t_stone = int(stone_steps[0]) if len(stone_steps) > 0 else len(ep_obs)

        for t in range(len(ep_obs)):
            has_pickaxe = False
            if ep_masks is not None and ep_masks[t, IDX_WOOD_PICKAXE] > 0:
                has_pickaxe = True
            elif np.any(ep_ev[:t, IDX_WOOD_PICKAXE] > 0):
                has_pickaxe = True

            if has_pickaxe and t <= t_stone:
                pickaxe_states.append({
                    "episode": ep,
                    "t": t,
                    "is_stone_trajectory": has_stone,
                })

    if len(pickaxe_states) > max_states:
        stone_st = [s for s in pickaxe_states if s["is_stone_trajectory"]]
        other_st = [s for s in pickaxe_states if not s["is_stone_trajectory"]]
        stride_s = max(1, len(stone_st) // (max_states // 2))
        stride_o = max(1, len(other_st) // (max_states // 2))
        sampled = stone_st[::stride_s] + other_st[::stride_o]
        return sampled[:max_states]
    return pickaxe_states


def evaluate_config_on_probe(
    model: JEPAv3,
    sampled_states: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    dev = torch.device(device)
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=cfg.get("horizon", 10),
        num_candidates=cfg.get("num_candidates", 64),
        cem_iterations=cfg.get("cem_iterations", 2),
        uncertainty_weight=cfg.get("uncertainty_weight", 0.5),
        novelty_weight=cfg.get("novelty_weight", 0.02),
        prior_mix=cfg.get("prior_mix", 0.5),
        prior_temperature=cfg.get("prior_temperature", 1.0),
        action_persistence=cfg.get("action_persistence", 0.0),
        vital_reward_weight=cfg.get("vital_reward_weight", 0.0),
        denormalize_q=cfg.get("denormalize_q", False),
        tte_weight=cfg.get("tte_weight", 0.0),
        recurring_vitals=cfg.get("recurring_vitals", False),
        hierarchical_tte=cfg.get("hierarchical_tte", False),
        device=dev,
    )

    agreements = []
    value_gaps = []
    t0 = time.time()

    with torch.no_grad():
        for item in sampled_states:
            ep = item["episode"]
            t = item["t"]
            recorded_act = int(ep["actions"][t])
            obs_seq = ep["obs"][: t + 1]
            act_seq = ep["actions"][: t + 1]

            t_obs_seq = torch.from_numpy(obs_seq).permute(0, 3, 1, 2).float() / 255.0
            t_obs_seq = t_obs_seq.to(dev)
            t_act_seq = torch.from_numpy(act_seq).long().to(dev)

            z_seq = model.encode(t_obs_seq, use_target=False)
            h_seq = model.recurrent_core.unroll_sequence(z_seq.unsqueeze(0), t_act_seq.unsqueeze(0)).squeeze(0)
            h_t = h_seq[-1]
            z_t = z_seq[-1]
            s_t = torch.cat([z_t, h_t], dim=-1).unsqueeze(0)

            curr_obs = ep["obs"][t]
            belief = AgentBeliefV3(recurrent_dim=256)
            belief.set_h(h_t.detach().cpu().numpy())
            if ep["masks_before"] is not None:
                belief.set_mask(ep["masks_before"][t])

            # Plan action
            top_action, _ = planner.plan(curr_obs, belief)
            agreements.append(bool(top_action == recorded_act))

            # Q-gap
            all_actions_t = torch.arange(17, device=dev)
            s_t_expanded = s_t.repeat(17, 1)
            q_all = model.q_mean(s_t_expanded, all_actions_t)
            if cfg.get("denormalize_q", False) and hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                q_all = model.return_normalizer.denormalize(q_all)
            q_all_np = q_all.cpu().numpy().flatten()
            q_stone = float(q_all_np[recorded_act])
            other_indices = [a for a in range(17) if a != recorded_act]
            q_wander = float(np.mean(q_all_np[other_indices]))
            value_gaps.append(q_stone - q_wander)

    elapsed = time.time() - t0
    return {
        "config_name": cfg["name"],
        "action_agreement_rate": float(np.mean(agreements)),
        "mean_value_gap": float(np.mean(value_gaps)),
        "eval_seconds": round(elapsed, 2),
        "params": cfg,
    }


def run_screening_for_config(
    model: JEPAv3,
    cfg: Dict[str, Any],
    num_episodes: int = 20,
    seed_start: int = 2050,
    max_steps: int = 1000,
    device: str = "cuda",
) -> Dict[str, Any]:
    dev = torch.device(device)
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=cfg.get("horizon", 10),
        num_candidates=cfg.get("num_candidates", 64),
        cem_iterations=cfg.get("cem_iterations", 2),
        uncertainty_weight=cfg.get("uncertainty_weight", 0.5),
        novelty_weight=cfg.get("novelty_weight", 0.02),
        prior_mix=cfg.get("prior_mix", 0.5),
        prior_temperature=cfg.get("prior_temperature", 1.0),
        action_persistence=cfg.get("action_persistence", 0.0),
        vital_reward_weight=cfg.get("vital_reward_weight", 0.0),
        denormalize_q=cfg.get("denormalize_q", False),
        tte_weight=cfg.get("tte_weight", 0.0),
        recurring_vitals=cfg.get("recurring_vitals", False),
        hierarchical_tte=cfg.get("hierarchical_tte", False),
        device=dev,
    )
    belief = AgentBeliefV3(recurrent_dim=256)
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)

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

        for step in range(max_steps):
            steps += 1
            action, _ = planner.plan(obs, belief)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

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

        ach = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        achievement_records.append(ach)
        survival_lengths.append(steps)
        episode_rewards.append(ep_reward)

    env.close()
    score_stats = compute_crafter_score(achievement_records)
    total_time = time.time() - t_start

    return {
        "config_name": cfg["name"],
        "num_episodes": num_episodes,
        "seed_range": [seed_start, seed_start + num_episodes - 1],
        "crafter_score": score_stats["score"],
        "mean_survival_length": float(np.mean(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "stone_unlock_rate": float(score_stats["unlock_percentages"].get("collect_stone", 0.0)),
        "drink_unlock_rate": float(score_stats["unlock_percentages"].get("collect_drink", 0.0)),
        "wood_pickaxe_unlock_rate": float(score_stats["unlock_percentages"].get("make_wood_pickaxe", 0.0)),
        "total_seconds": round(total_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic Inference-Time Ablation & Grid Engine.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_step1_stone.pt")
    parser.add_argument("--max_states", type=int, default=120)
    parser.add_argument("--screen_top_k", type=int, default=1)
    parser.add_argument("--screen_episodes", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/planner_ablation_sweep.json")
    args = parser.parse_args()

    ckpt_file = resolve_checkpoint(args.checkpoint, args.drive_dir)

    print("=" * 80)
    print("      INFERENCE-TIME ABLATION & COMBINATORIAL GRID ENGINE")
    print(f"      Checkpoint : {ckpt_file}")
    print(f"      Device     : {args.device}")
    print(f"      Probe Size : {args.max_states} states")
    print("=" * 80)

    model = JEPAv3.from_checkpoint(str(ckpt_file), device=args.device)
    model.eval()

    print("[Sweep] Loading 120 pickaxe-in-hand probe states...")
    sampled_states = load_probe_states(args.drive_dir, max_states=args.max_states)
    print(f"[Sweep] Successfully loaded {len(sampled_states)} states.")

    # Complete Senior ML Combinatorial Ablation Grid:
    # Lever A: Value Denormalization (denormalize_q: True)
    # Lever B: Time-to-Event Compass (tte_weight: 0.5)
    # Lever C: Continuous Vital Survival (recurring_vitals: True, vital_reward_weight: 0.3)
    # Lever D: Action Persistence / Mining Momentum (action_persistence: 0.3)
    # Lever E: Hierarchical Tech-Tree Focus (hierarchical_tte: True)
    # Lever F: Deep Test-Time Compute (horizon: 15, num_candidates: 128, cem_iterations: 3)
    CONFIGS = [
        # Baseline
        {"name": "0_Baseline"},

        # Singles (A, B, C, D, E, F)
        {"name": "A_Denorm", "denormalize_q": True},
        {"name": "B_TTE", "tte_weight": 0.5},
        {"name": "C_Vitals", "recurring_vitals": True, "vital_reward_weight": 0.3},
        {"name": "D_Persistence", "action_persistence": 0.3},
        {"name": "E_Hierarchical", "hierarchical_tte": True, "tte_weight": 0.5},
        {"name": "F_DeepCompute", "horizon": 15, "num_candidates": 128, "cem_iterations": 3},

        # Pairs
        {"name": "AB_Denorm_TTE", "denormalize_q": True, "tte_weight": 0.5},
        {"name": "AC_Denorm_Vitals", "denormalize_q": True, "recurring_vitals": True, "vital_reward_weight": 0.3},
        {"name": "AD_Denorm_Persistence", "denormalize_q": True, "action_persistence": 0.3},
        {"name": "AE_Denorm_Hierarchical", "denormalize_q": True, "hierarchical_tte": True, "tte_weight": 0.5},
        {"name": "AF_Denorm_DeepCompute", "denormalize_q": True, "horizon": 15, "num_candidates": 128, "cem_iterations": 3},
        {"name": "BC_TTE_Vitals", "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3},
        {"name": "BD_TTE_Persistence", "tte_weight": 0.5, "action_persistence": 0.3},
        {"name": "BE_TTE_Hierarchical", "tte_weight": 0.5, "hierarchical_tte": True},
        {"name": "CD_Vitals_Persistence", "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3},
        {"name": "CE_Vitals_Hierarchical", "recurring_vitals": True, "vital_reward_weight": 0.3, "hierarchical_tte": True, "tte_weight": 0.5},
        {"name": "DE_Persistence_Hierarchical", "action_persistence": 0.3, "hierarchical_tte": True, "tte_weight": 0.5},

        # Triples
        {"name": "ABC_Denorm_TTE_Vitals", "denormalize_q": True, "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3},
        {"name": "ABD_Denorm_TTE_Persist", "denormalize_q": True, "tte_weight": 0.5, "action_persistence": 0.3},
        {"name": "ABE_Denorm_TTE_Hierarch", "denormalize_q": True, "tte_weight": 0.5, "hierarchical_tte": True},
        {"name": "ACD_Denorm_Vitals_Persist", "denormalize_q": True, "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3},
        {"name": "ACE_Denorm_Vitals_Hierarch", "denormalize_q": True, "recurring_vitals": True, "vital_reward_weight": 0.3, "hierarchical_tte": True, "tte_weight": 0.5},
        {"name": "BCD_TTE_Vitals_Persist", "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3},
        {"name": "BCE_TTE_Vitals_Hierarch", "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3, "hierarchical_tte": True},
        {"name": "CDE_Vitals_Persist_Hier", "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3, "hierarchical_tte": True, "tte_weight": 0.5},

        # Grand Combinations (OpenAI O1 / Senior ML Architectures)
        {"name": "ABCD_Quad_Survival", "denormalize_q": True, "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3},
        {"name": "ABCDE_FullSynergy_H10", "denormalize_q": True, "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3, "hierarchical_tte": True, "uncertainty_weight": 0.1, "novelty_weight": 0.005, "prior_mix": 0.65},
        {"name": "Grand_Master_O1_H15", "denormalize_q": True, "tte_weight": 0.5, "recurring_vitals": True, "vital_reward_weight": 0.3, "action_persistence": 0.3, "hierarchical_tte": True, "uncertainty_weight": 0.1, "novelty_weight": 0.005, "prior_mix": 0.65, "horizon": 15, "num_candidates": 128, "cem_iterations": 3},
    ]

    print(f"\n[Sweep] Evaluating {len(CONFIGS)} combinatorial configurations on 120 states...")
    results = []
    t_start = time.time()

    for idx, cfg in enumerate(CONFIGS):
        res = evaluate_config_on_probe(model, sampled_states, cfg, device=args.device)
        results.append(res)
        print(
            f"  [{idx + 1:02d}/{len(CONFIGS)}] {cfg['name']:<24} | "
            f"Agreement: {res['action_agreement_rate'] * 100:5.2f}% | "
            f"Value Gap: {res['mean_value_gap']:+8.4f} | "
            f"({res['eval_seconds']:.1f}s)"
        )

    # Sort leaderboard by Agreement Rate (descending) and Value Gap
    results.sort(key=lambda r: (r["action_agreement_rate"], r["mean_value_gap"]), reverse=True)

    print("\n" + "=" * 80)
    print("                    ABLATION SWEEP LEADERBOARD")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Configuration':<24} | {'Agreement':<12} | {'Value Gap':<14} | {'Status'}")
    print("-" * 80)
    baseline_agr = 0.2833
    for rank, r in enumerate(results, 1):
        status = "🥇 WINNER" if rank == 1 else ("🥈 TOP 3" if rank <= 3 else ("✅ BEATS BASELINE" if r["action_agreement_rate"] > baseline_agr else "—"))
        print(f"#{rank:<4} | {r['config_name']:<24} | {r['action_agreement_rate'] * 100:5.2f}%       | {r['mean_value_gap']:+8.4f}       | {status}")
    print("=" * 80)
    print(f"Total Sweep Time: {time.time() - t_start:.1f}s\n")

    # Screen Top Champions if requested
    champions_screening = []
    if args.screen_top_k > 0:
        top_cfgs = results[: args.screen_top_k]
        for c_rank, c_res in enumerate(top_cfgs, 1):
            c_cfg = c_res["params"]
            print("\n" + "=" * 80)
            print(f">>> SCREENING CHAMPION #{c_rank}: {c_cfg['name']} ({args.screen_episodes} EPISODES) <<<")
            print("=" * 80)
            screen_res = run_screening_for_config(
                model=model,
                cfg=c_cfg,
                num_episodes=args.screen_episodes,
                seed_start=args.seed_start,
                device=args.device,
            )
            champions_screening.append(screen_res)
            print(f"\n[Champion #{c_rank} Results] {c_cfg['name']}:")
            print(f"  Crafter Score: {screen_res['crafter_score']:.2f}% (Baseline: 4.55%)")
            print(f"  Mean Survival Length: {screen_res['mean_survival_length']:.1f} steps")
            print(f"  Mean Reward: {screen_res['mean_episode_reward']:.2f}")
            print(f"  Drink Rate: {screen_res['drink_unlock_rate']:.1f}%")
            print(f"  Wood Pickaxe Rate: {screen_res['wood_pickaxe_unlock_rate']:.1f}%")
            print(f"  Collect Stone Rate: {screen_res['stone_unlock_rate']:.1f}%\n")

    summary = {
        "checkpoint": str(ckpt_file),
        "total_configs_evaluated": len(CONFIGS),
        "leaderboard": results,
        "champions_screening": champions_screening,
    }

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"--> Saved sweep results to: {out_p}")

    if args.drive_dir:
        dp = Path(args.drive_dir)
        if dp.exists():
            drive_out = dp / "results" / "diagnostics" / out_p.name
            drive_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_p), str(drive_out))
            print(f"[Drive] Mirrored sweep results to: {drive_out}")

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print(f"--- BEGIN {out_p.name} ---")
    print(json.dumps(summary, indent=2))
    print(f"--- END {out_p.name} ---\n")


if __name__ == "__main__":
    main()
