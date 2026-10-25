#!/usr/bin/env python3
"""
scripts/diagnose_planner_probe.py — Step 1 Planner Stone Probe Re-run
=====================================================================

Probes the RecurrentMPCPlanner on 120 pickaxe-in-hand states extracted from
ground-truth teacher and curiosity trajectories (identical protocol to Step 0).

Evaluates:
1. Action-Agreement Rate (vs recorded human/teacher demonstration actions).
   Baseline to beat: 28.33%.
2. Mean Value Gap (Q_stone_action - Q_wander_mean).
   Baseline to beat: +0.0218.

Output:
- results/diagnostics/step1_planner_stone_probe.json
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.mpc_v3 import RecurrentMPCPlanner

ACH_TO_IDX = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}
IDX_WOOD_PICKAXE = ACH_TO_IDX.get("make_wood_pickaxe", 15)
IDX_STONE = ACH_TO_IDX.get("collect_stone", 5)


def ensure_drive_mounted(drive_dir: Path) -> bool:
    if drive_dir.exists():
        return True
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        return drive_dir.exists()
    except Exception:
        return False


def resolve_checkpoint(ckpt_path: str, drive_p: Path) -> Path:
    p = Path(ckpt_path)
    if p.exists():
        return p
    candidates = [
        REPO_ROOT / ckpt_path,
        REPO_ROOT / "checkpoints" / p.name,
        drive_p / ckpt_path,
        drive_p / "checkpoints" / p.name,
        Path("/content") / "checkpoints" / p.name,
        Path("/content") / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"[ABORT] Cannot locate checkpoint: {ckpt_path}")


def run_planner_probe(
    checkpoint: str,
    output_json: str = "results/diagnostics/step1_planner_stone_probe.json",
    max_states_to_probe: int = 120,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
) -> Dict[str, Any]:
    drive_p = Path(drive_dir)
    ensure_drive_mounted(drive_p)
    ckpt_file = resolve_checkpoint(checkpoint, drive_p)

    print("=" * 80)
    print("STEP 1: PLANNER STONE PROBE RE-RUN (120 PICKAXE STATES)")
    print(f"  Checkpoint : {ckpt_file}")
    print(f"  Device     : {device}")
    print(f"  Max States : {max_states_to_probe}")
    print("=" * 80)

    model = JEPAv3.from_checkpoint(str(ckpt_file), device=device)
    model.to(torch.device(device))
    model.eval()

    planner = RecurrentMPCPlanner(
        model=model,
        horizon=10,
        num_candidates=64,
        cem_iterations=2,
        device=device,
    )
    dev = torch.device(device)

    # Shard candidates
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

    print(f"[Planner-Probe] Scanning {len(all_shards)} shards for pickaxe and stone trajectories...")
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
                has_pickaxe = bool(np.any(ep_events[:, IDX_WOOD_PICKAXE] > 0) or (ep_masks is not None and np.any(ep_masks[:, IDX_WOOD_PICKAXE] > 0)))

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

    print(f"[Planner-Probe] Found {len(targeted_episodes)} relevant episodes ({sum(1 for e in targeted_episodes if e['has_stone'])} contain stone).")

    # Extract pickaxe-in-hand states
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
                    "steps_to_stone": (t_stone - t) if has_stone else -1,
                })

    print(f"[Planner-Probe] Total pickaxe-in-hand states identified: {len(pickaxe_states)}")

    if len(pickaxe_states) > max_states_to_probe:
        stone_st = [s for s in pickaxe_states if s["is_stone_trajectory"]]
        other_st = [s for s in pickaxe_states if not s["is_stone_trajectory"]]
        stride_s = max(1, len(stone_st) // (max_states_to_probe // 2))
        stride_o = max(1, len(other_st) // (max_states_to_probe // 2))
        sampled_states = stone_st[::stride_s] + other_st[::stride_o]
        sampled_states = sampled_states[:max_states_to_probe]
    else:
        sampled_states = pickaxe_states

    print(f"[Planner-Probe] Probing planner on {len(sampled_states)} pickaxe-in-hand states...")

    agreements = []
    value_gaps = []
    stone_approach_q_vals = []
    wandering_q_vals = []
    probe_records = []

    t0 = time.time()
    for idx, item in enumerate(sampled_states):
        ep = item["episode"]
        t = item["t"]
        recorded_act = int(ep["actions"][t])
        obs_seq = ep["obs"][: t + 1]
        act_seq = ep["actions"][: t + 1]

        with torch.no_grad():
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

            # 1. Plan top action
            top_action, debug = planner.plan(curr_obs, belief)
            is_agree = bool(top_action == recorded_act)
            agreements.append(is_agree)

            # 2. Evaluate Q-values across all 17 discrete actions
            all_actions_t = torch.arange(17, device=dev)
            s_t_expanded = s_t.repeat(17, 1)
            q_all = model.q_mean(s_t_expanded, all_actions_t)
            if hasattr(model, "return_normalizer") and model.return_normalizer.initialized:
                q_all = model.return_normalizer.denormalize(q_all)
            q_all_np = q_all.cpu().numpy().flatten()

            q_stone_action = float(q_all_np[recorded_act])
            other_indices = [a for a in range(17) if a != recorded_act]
            q_wander_mean = float(np.mean(q_all_np[other_indices]))
            q_gap = q_stone_action - q_wander_mean

            value_gaps.append(q_gap)
            stone_approach_q_vals.append(q_stone_action)
            wandering_q_vals.append(q_wander_mean)

            probe_records.append({
                "step": t,
                "recorded_action": recorded_act,
                "planned_action": top_action,
                "agreement": is_agree,
                "q_stone_action": q_stone_action,
                "q_wander_mean": q_wander_mean,
                "value_gap": q_gap,
                "is_stone_trajectory": item["is_stone_trajectory"],
            })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(sampled_states):
            print(f"  Probed {idx + 1:3d}/{len(sampled_states)} states ({time.time() - t0:.1f}s) | Agreement: {np.mean(agreements):.1%} | Mean Value Gap: {np.mean(value_gaps):+.4f}")

    action_agreement_rate = float(np.mean(agreements))
    mean_val_gap = float(np.mean(value_gaps))
    mean_q_stone = float(np.mean(stone_approach_q_vals))
    mean_q_wander = float(np.mean(wandering_q_vals))

    print("\n" + "=" * 80)
    print(">>> STEP 1 PLANNER STONE PROBE RESULTS <<<")
    print("=" * 80)
    print(f"{'Metric':<36} | {'Baseline (Step 0)':<18} | {'Step 1 Final':<16} | {'Status'}")
    print("-" * 80)
    agr_status = "✅ IMPROVED" if action_agreement_rate > 0.2833 else "⚠️ FLAT"
    print(f"{'Action Agreement Rate':<36} | {'28.33%':<18} | {f'{action_agreement_rate * 100:.2f}%':<16} | {agr_status}")
    gap_status = "✅ STRONGER PULL" if mean_val_gap > 0.0218 else "⚠️ WEAK"
    print(f"{'Mean Value Gap (Stone - Wander)':<36} | {'+0.0218':<18} | {f'{mean_val_gap:+.4f}':<16} | {gap_status}")
    print(f"{'Mean Q (Stone Approaching)':<36} | {'1.0017':<18} | {f'{mean_q_stone:.4f}':<16} | {'—'}")
    print(f"{'Mean Q (Wandering)':<36} | {'0.9799':<18} | {f'{mean_q_wander:.4f}':<16} | {'—'}")
    print("=" * 80)

    results = {
        "num_probed_states": len(sampled_states),
        "action_agreement_rate": action_agreement_rate,
        "mean_value_gap": mean_val_gap,
        "mean_q_stone_approaching": mean_q_stone,
        "mean_q_wandering": mean_q_wander,
        "baseline_comparison": {
            "step0_agreement_rate": 0.2833333333333333,
            "step0_mean_value_gap": 0.021818772532666723,
            "delta_agreement_rate": float(action_agreement_rate - 0.2833333333333333),
            "delta_mean_value_gap": float(mean_val_gap - 0.021818772532666723),
        },
        "probe_records": probe_records[:50],
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Planner-Probe] Saved results JSON to: {out_p}")

    if drive_p.exists():
        drive_out = drive_p / "results" / "diagnostics" / out_p.name
        drive_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(out_p), str(drive_out))
        print(f"[Drive] Mirrored probe JSON to: {drive_out}")

    print("\n" + "=" * 80)
    print("=== RAW JSON DUMP (COPY & PASTE TO CHAT) ===")
    print("=" * 80)
    print(f"--- BEGIN {out_p.name} ---")
    print(json.dumps(results, indent=2))
    print(f"--- END {out_p.name} ---\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 Planner Stone Probe Re-run")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_step1_stone.pt")
    parser.add_argument("--output_json", type=str, default="results/diagnostics/step1_planner_stone_probe.json")
    parser.add_argument("--max_states", type=int, default=120)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    args = parser.parse_args()

    run_planner_probe(
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        max_states_to_probe=args.max_states,
        device=args.device,
        drive_dir=args.drive_dir,
    )


if __name__ == "__main__":
    main()
