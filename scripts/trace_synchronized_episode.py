#!/usr/bin/env python3
"""
scripts/trace_synchronized_episode.py — Synchronized 6-Stage Episode Tracer for Learn-a-World v3
================================================================================================

Generates an end-to-end synchronized per-step trace across all 6 cognitive stages:
  Stage 1: Perception & State Estimation (latent z_t, StateReadoutHead x_hat, true inventory)
  Stage 2: Achievement Belief & Events (EventHead event_probs, m_bin mask, unlocked events)
  Stage 3: Causal Card & Subgoal Selection (CompositionalCausalPlanner active_subgoal, cue sim, chain, deficits)
  Stage 4: Subgoal Persistence & Navigation (continuity across steps, visual similarity delta)
  Stage 5: MPC Action Scoring & Selection (RecurrentMPCPlanner best_action, score breakdown, prior)
  Stage 6: Physical Execution Outcome (environment reward, true inventory delta, done, terminal cause)

Produces:
  1. results/diagnostic_traces/trace_seed_{seed}.json (complete structured data)
  2. results/diagnostic_traces/trace_seed_{seed}.md   (human/LLM-readable autopsy report)
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
from eval.metrics import CRAFTER_ACHIEVEMENTS
from models.jepa_v3 import JEPAv3
from planning.agent_v3 import AgentBeliefV3
from planning.causal_memory import RelationHypothesisBank
from planning.compositional_planner import CompositionalCausalPlanner
from planning.mpc_v3 import RecurrentMPCPlanner
from models.calibration import EventHeadCalibrator

CRAFTER_ACTIONS = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

INVENTORY_NAMES = [
    "wood", "stone", "table", "furnace", "iron", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
    "health", "food", "drink", "energy"
]


def sha256_file(filepath: Path) -> Optional[str]:
    if not filepath.exists() or not filepath.is_file():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True)
        return out.strip()
    except Exception:
        return "unknown"


def resolve_file(path_str: Optional[str], candidate_dirs: List[Path]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists() and p.is_file():
        return p.resolve()
    for d in candidate_dirs:
        cand = d / p.name
        if cand.exists() and cand.is_file():
            return cand.resolve()
        cand2 = d / path_str
        if cand2.exists() and cand2.is_file():
            return cand2.resolve()
    return None


def run_synchronized_trace(
    checkpoint_path: str,
    causal_bank_path: str,
    seed: int = 2050,
    calibrator_path: Optional[str] = None,
    device: str = "cpu",
    horizon: int = 15,
    num_candidates: int = 64,
    cem_iterations: int = 2,
    death_risk_factor: float = 0.5,
    max_steps: int = 400,
    output_json: Optional[str] = None,
    output_md: Optional[str] = None,
) -> Dict[str, Any]:
    dev = torch.device(device)
    candidate_dirs = [
        REPO_ROOT,
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "results",
        REPO_ROOT / "results" / "diagnostics",
        REPO_ROOT / "results" / "calibration",
        Path("/content"),
        Path("/content/checkpoints"),
        Path("/content/drive/MyDrive/learn-a-world"),
        Path("/content/drive/MyDrive/learn-a-world/checkpoints"),
    ]

    resolved_ckpt = resolve_file(checkpoint_path, candidate_dirs)
    if not resolved_ckpt:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_bank = resolve_file(causal_bank_path, candidate_dirs)
    if not resolved_bank:
        raise FileNotFoundError(f"Causal memory bank not found: {causal_bank_path}")

    resolved_cal = resolve_file(calibrator_path, candidate_dirs) if calibrator_path else None

    print("=" * 80)
    print(f"      SYNCHRONIZED EPISODE TRACER (Seed {seed})")
    print(f"      Checkpoint: {resolved_ckpt}")
    print(f"      Bank:       {resolved_bank}")
    print(f"      Calibrator: {resolved_cal}")
    print(f"      Device:     {dev} | Death Risk: {death_risk_factor}")
    print("=" * 80)

    # 1. Load JEPA Model
    model = JEPAv3(latent_dim=128, recurrent_dim=256, num_actions=17).to(dev)
    ckpt = torch.load(resolved_ckpt, map_location=dev)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    if resolved_cal:
        cal = EventHeadCalibrator.load(str(resolved_cal))
        model.set_calibrator(cal)

    # 2. Load Causal Memory Bank & Compositional Planner
    bank = RelationHypothesisBank(device=dev)
    bank.load(resolved_bank)
    comp_planner = CompositionalCausalPlanner(
        hypothesis_bank=bank,
        min_prototype_confidence=0.05,
        sim_threshold=0.35,
        device=dev,
    )

    # 3. Build Recurrent MPC Planner
    planner = RecurrentMPCPlanner(
        model=model,
        horizon=horizon,
        num_candidates=num_candidates,
        cem_iterations=cem_iterations,
        elite_ratio=0.15,
        prior_mix=0.5,
        action_persistence=0.5,
        vital_weight=0.5,
        death_risk_factor=death_risk_factor,
        harvest_bonus=0.15,
        denormalize_q=False,
        tte_weight=0.0,
        standardize_scores=False,
        use_transition_model=True,
        transition_scoring=True,
        transition_weight=0.5,
        transition_done_penalty=2.0,
        transition_resource_weight=0.25,
        compositional_planner=comp_planner,
        strict_inference=True,
        device=dev,
    )

    belief = AgentBeliefV3(recurrent_dim=256, strict_inference=True)

    # 4. Initialize Environment
    env = gym.make("Crafter-v0", render_size=64, max_steps=max_steps)
    obs, info = env.reset(seed=seed)
    belief.reset()

    steps = 0
    ep_reward = 0.0
    records: List[Dict[str, Any]] = []
    prev_subgoal: Optional[str] = None
    prev_cue_sim: Optional[float] = None
    milestones: List[Dict[str, Any]] = []

    prev_inv = dict(info.get("inventory", {}))

    for step in range(max_steps):
        steps += 1

        # STAGE 1: Current Observation & State Readout
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            obs_t = obs_t.to(dev)
            z_t = model.encode(obs_t, use_target=True)
            h_prev = belief.get_h_tensor(device=dev).unsqueeze(0)
            # Dummy act_t for recurrent representation of current frame
            s_curr, _ = model.recurrent_step(z_t, torch.zeros(1, dtype=torch.long, device=dev), h_prev)
            x_hat, _ = model.predict_state(s_curr)
            x_hat_np = x_hat.squeeze(0).cpu().numpy().tolist()

        # STAGE 3 (Pre-plan): Query Causal Subgoal
        subgoal_info = comp_planner.plan_subgoal(
            obs_latent=z_t.squeeze(0),
            grounded_state=x_hat_np,
            m_bin=belief.get_binary_mask(),
        )

        # STAGE 5: MPC Action Scoring & Selection
        action, debug = planner.plan(obs, belief)
        action_name = CRAFTER_ACTIONS[action] if 0 <= action < len(CRAFTER_ACTIONS) else f"act_{action}"

        # STAGE 6 (Environment Step)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)
        ep_reward += float(reward)

        # STAGE 2: Belief Update & Event Head Prediction
        with torch.no_grad():
            act_t = torch.tensor([action], device=dev, dtype=torch.long)
            s_t, next_h = model.recurrent_step(z_t, act_t, h_prev)
            mask_t = belief.get_mask_tensor(device=dev).unsqueeze(0)
            event_probs = model.event_head.predict_proba(s_t, act_t, mask_t).squeeze(0)
            event_probs_np = [round(float(p), 4) for p in event_probs.cpu().numpy()]

        mask_before = belief.get_binary_mask().copy()
        belief.update(event_probs, reward, next_h=next_h.squeeze(0))
        mask_after = belief.get_binary_mask().copy()

        # Physical changes
        curr_inv = dict(next_info.get("inventory", {}))
        inv_delta = {k: curr_inv.get(k, 0) - prev_inv.get(k, 0) for k in set(curr_inv) | set(prev_inv) if curr_inv.get(k, 0) != prev_inv.get(k, 0)}
        ach_diag = dict(next_info.get("achievements", {}))
        new_achs = [k for k, v in ach_diag.items() if v > 0 and k not in [m["achievement"] for m in milestones]]

        for na in new_achs:
            milestones.append({
                "step": steps,
                "achievement": na,
                "reward": ep_reward,
                "action": action_name,
                "inventory": curr_inv.copy(),
            })

        # STAGE 4: Subgoal continuity tracking
        curr_subgoal_target = subgoal_info.get("prototype_id") if subgoal_info else None
        curr_cue_sim = float(subgoal_info.get("visual_sim", 0.0)) if subgoal_info else None
        subgoal_maintained = (curr_subgoal_target is not None and curr_subgoal_target == prev_subgoal)
        sim_delta = (curr_cue_sim - prev_cue_sim) if (curr_cue_sim is not None and prev_cue_sim is not None) else 0.0

        step_record: Dict[str, Any] = {
            "step": steps,
            # Stage 1: Perception & Readout
            "perception": {
                "latent_norm": round(float(z_t.norm().item()), 4),
                "predicted_state": {INVENTORY_NAMES[i]: round(float(x_hat_np[i]), 3) for i in range(len(INVENTORY_NAMES))},
                "true_inventory": curr_inv,
            },
            # Stage 2: Belief & Events
            "belief": {
                "event_probs_top": {
                    CRAFTER_ACHIEVEMENTS[i]: event_probs_np[i]
                    for i in np.argsort(event_probs_np)[-5:][::-1]
                    if event_probs_np[i] > 0.05
                },
                "newly_unlocked_belief": [int(x) for x in belief.history[-1]["newly_unlocked"]],
                "binary_mask_sum": int(mask_after.sum()),
            },
            # Stage 3: Subgoal Selection
            "subgoal": {
                "active": subgoal_info is not None,
                "target_prototype": subgoal_info.get("prototype_id") if subgoal_info else None,
                "status": subgoal_info.get("status") if subgoal_info else "NO_SUBGOAL",
                "recommended_action": subgoal_info.get("recommended_action") if subgoal_info else None,
                "cue_similarity": round(curr_cue_sim, 4) if curr_cue_sim is not None else 0.0,
                "chain_length": len(subgoal_info.get("chain", [])) if subgoal_info else 0,
                "missing_dim": subgoal_info.get("missing_resource_dim") if subgoal_info else None,
                "deficit": round(float(subgoal_info.get("deficit_amount", 0.0)), 2) if subgoal_info else 0.0,
            },
            # Stage 4: Navigation Trend
            "navigation": {
                "subgoal_maintained": subgoal_maintained,
                "cue_sim_delta": round(float(sim_delta), 4),
            },
            # Stage 5: MPC Action
            "mpc": {
                "chosen_action": action,
                "chosen_action_name": action_name,
                "best_score": round(float(debug.get("best_score", 0.0)), 4),
                "reward_sum": round(float(debug.get("reward_sum", 0.0)), 4),
                "subgoal_reward": round(float(debug.get("active_subgoal_status", "") != ""), 2),
            },
            # Stage 6: Outcome
            "outcome": {
                "env_reward": float(reward),
                "done": done,
                "inventory_delta": inv_delta,
                "new_achievements": new_achs,
            },
        }
        records.append(step_record)

        prev_subgoal = curr_subgoal_target
        prev_cue_sim = curr_cue_sim
        prev_inv = curr_inv
        obs = next_obs

        if done:
            break

    # Determine Termination Cause
    final_health = curr_inv.get("health", 0)
    final_food = curr_inv.get("food", 0)
    final_drink = curr_inv.get("drink", 0)
    if final_health <= 0:
        if final_food <= 0 or final_drink <= 0:
            term_cause = "Starvation / Dehydration"
        else:
            term_cause = "Combat Lethal Damage (Zombie / Skeleton)"
    elif steps >= max_steps:
        term_cause = "Max Steps Reached"
    else:
        term_cause = "Unknown / Game Over"

    # Compile Structured Autopsy
    autopsy = {
        "provenance": {
            "git_commit": get_git_commit(),
            "checkpoint": str(resolved_ckpt),
            "checkpoint_sha256": sha256_file(resolved_ckpt),
            "causal_bank": str(resolved_bank),
            "causal_bank_sha256": sha256_file(resolved_bank),
            "calibrator": str(resolved_cal) if resolved_cal else None,
            "seed": seed,
            "total_steps": steps,
            "total_reward": round(ep_reward, 2),
            "termination_cause": term_cause,
        },
        "milestones_achieved": milestones,
        "steps": records,
    }

    # Save JSON
    out_json_path = Path(output_json or f"results/diagnostic_traces/trace_seed_{seed}.json")
    def json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(out_json_path, "w") as f:
        json.dump(autopsy, f, indent=2, default=json_default)
    print(f"\n[Saved JSON Trace]: {out_json_path} ({len(records)} steps)")

    # Save Markdown Report
    out_md_path = Path(output_md or f"results/diagnostic_traces/trace_seed_{seed}.md")
    md_content = generate_markdown_report(autopsy)
    with open(out_md_path, "w") as f:
        f.write(md_content)
    print(f"[Saved Markdown Report]: {out_md_path}")

    return autopsy


def generate_markdown_report(autopsy: Dict[str, Any]) -> str:
    prov = autopsy["provenance"]
    milestones = autopsy["milestones_achieved"]
    steps = autopsy["steps"]

    lines = [
        f"# Synchronized 6-Stage Episode Failure Autopsy (Seed {prov['seed']})",
        "",
        "## 1. Provenance & Run Configuration",
        f"- **Git Commit**: `{prov['git_commit']}`",
        f"- **Checkpoint**: `{prov['checkpoint']}` (SHA256: `{prov['checkpoint_sha256'][:16]}...`)",
        f"- **Causal Memory Bank**: `{prov['causal_bank']}`",
        f"- **Seed**: `{prov['seed']}` | **Steps Survived**: `{prov['total_steps']}` | **Reward**: `{prov['total_reward']}`",
        f"- **Terminal Cause**: **{prov['termination_cause']}**",
        "",
        "## 2. Milestone Achievement Timeline",
        "| Step | Achievement | Action Executed | Physical Inventory State |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for m in milestones:
        inv_str = ", ".join(f"{k}:{v}" for k, v in m["inventory"].items() if v > 0)
        lines.append(f"| Step {m['step']} | **{m['achievement']}** | `{m['action']}` | {inv_str} |")

    lines.extend([
        "",
        "## 3. Critical Failure Analysis: The 6 Cognitive Stages",
        "",
    ])

    # Find critical phases
    table_step = next((m["step"] for m in milestones if m["achievement"] == "place_table"), None)
    pickaxe_step = next((m["step"] for m in milestones if m["achievement"] == "make_wood_pickaxe"), None)

    if pickaxe_step is not None:
        lines.append(f"### Phase: Post-Pickaxe Search for Stone (Steps {pickaxe_step} to {min(pickaxe_step + 25, len(steps))})")
        lines.append("Analyzing why stone was never mined after pickaxe was in inventory:")
        sample_range = range(pickaxe_step - 1, min(pickaxe_step + 25, len(steps)))
    elif table_step is not None:
        lines.append(f"### Phase: Table Placed to Pickaxe Craft Attempt (Steps {table_step} to {min(table_step + 25, len(steps))})")
        lines.append("Analyzing why pickaxe was never crafted after table placement:")
        sample_range = range(table_step - 1, min(table_step + 25, len(steps)))
    else:
        lines.append("### Phase: Initial Wood Harvesting Breakdown")
        sample_range = range(0, min(30, len(steps)))

    lines.extend([
        "",
        "| Step | Action Chosen | Subgoal Target | Status | Cue Sim | Predicted Wood/Pickaxe | True Wood/Pickaxe | Outcome / Reward |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for i in sample_range:
        s = steps[i]
        p = s["perception"]
        sg = s["subgoal"]
        mpc = s["mpc"]
        out = s["outcome"]

        pred_w = p["predicted_state"].get("wood", 0.0)
        pred_p = p["predicted_state"].get("wood_pickaxe", 0.0)
        true_w = p["true_inventory"].get("wood", 0)
        true_p = p["true_inventory"].get("wood_pickaxe", 0)

        pred_str = f"W:{pred_w:.1f} P:{pred_p:.1f}"
        true_str = f"W:{true_w} P:{true_p}"
        sg_str = sg["target_prototype"] or "None"
        rew_str = f"+{out['env_reward']}" if out['env_reward'] > 0 else f"{out['env_reward']}"

        lines.append(
            f"| {s['step']} | `{mpc['chosen_action_name']}` | `{sg_str}` | {sg['status']} | {sg['cue_similarity']:.2f} | {pred_str} | {true_str} | {rew_str} |"
        )

    lines.extend([
        "",
        "## 4. Final 5 Steps Leading to Death",
        "| Step | Action | Health / Food / Drink | Subgoal | MPC Score | Event Probabilities |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in steps[-5:]:
        p = s["perception"]
        sg = s["subgoal"]
        mpc = s["mpc"]
        b = s["belief"]
        vitals = f"H:{p['true_inventory'].get('health',0)} F:{p['true_inventory'].get('food',0)} D:{p['true_inventory'].get('drink',0)}"
        top_probs = ", ".join(f"{k}:{v}" for k, v in list(b["event_probs_top"].items())[:2]) or "None"

        lines.append(
            f"| {s['step']} | `{mpc['chosen_action_name']}` | {vitals} | `{sg['target_prototype'] or 'None'}` ({sg['status']}) | {mpc['best_score']} | {top_probs} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronized 6-Stage Episode Tracer for Learn-a-World v3.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_state_readout_v2.pt")
    parser.add_argument("--causal_memory_bank", "--causal_memory_bank_path", dest="causal_memory_bank", type=str, default="phase2_memory_consolidation_4.json")
    parser.add_argument("--calibrator_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--cem_iterations", type=int, default=2)
    parser.add_argument("--death_risk_factor", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)

    args = parser.parse_args()

    run_synchronized_trace(
        checkpoint_path=args.checkpoint,
        causal_bank_path=args.causal_memory_bank,
        seed=args.seed,
        calibrator_path=args.calibrator_path,
        device=args.device,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        cem_iterations=args.cem_iterations,
        death_risk_factor=args.death_risk_factor,
        max_steps=args.max_steps,
        output_json=args.output_json,
        output_md=args.output_md,
    )
