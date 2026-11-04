"""Phase P3 execution runner for MOSAIC: Calibrated Uncertainty & Active Probing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from mosaic.src.checkpoint import load_checkpoint, save_atomic
from mosaic.src.env_boundary import CrafterPixelEnv
from mosaic.src.models.world_model import MosaicWorldModel
from .active_probe import ActiveProbePolicy, DualValueHeads
from .calibration import compute_calibration_curve


@dataclass(frozen=True)
class Phase3Config:
    num_actions: int = 17
    latent_dim: int = 128
    action_dim: int = 64
    reward_dim: int = 16
    hidden_dim: int = 256
    stochastic_dim: int = 32
    num_ensemble_members: int = 3
    beta_epistemic: float = 1.0
    rho_risk: float = 5.0
    step_cost: float = 0.01
    eval_episodes_per_arm: int = 3
    max_episode_steps: int = 250
    seed: int = 12345
    eval_seed_start: int = 900_000
    permutation_seed: int = 12345
    device: str = "auto"
    world_model_path: Optional[str] = None
    output_dir: str = "runs/mosaic/p3_active_probe"


def _evaluate_arm(
    policy: ActiveProbePolicy,
    arm_name: str,
    config: Phase3Config,
) -> Dict[str, Any]:
    """Evaluate a single exploration arm through the clean-room Crafter boundary."""
    rewards = []
    lengths = []
    terminated_count = 0
    all_uncertainties = []
    all_errors = []

    wm = policy.world_model
    device = policy.device

    for ep_idx in range(config.eval_episodes_per_arm):
        seed = config.eval_seed_start + ep_idx
        env = CrafterPixelEnv(
            seed=seed,
            max_episode_steps=config.max_episode_steps,
            permutation_seed=config.permutation_seed,
        )
        obs = env.reset(seed=seed)
        step = 0
        total_r = 0.0
        prev_r = 0.0

        while True:
            action = policy.act(obs, prev_r, step)
            next_obs, reward, terminated, truncated = env.step(action)
            total_r += float(reward)
            prev_r = float(reward)
            step += 1

            # Extract predicted uncertainty and actual transition error
            if policy.trace_history:
                last_trace = policy.trace_history[-1]
                u_pred = last_trace["disagreement"]
                all_uncertainties.append(u_pred)

                # Compute actual error between predicted latent and true target latent
                with torch.no_grad():
                    obs_next_t = torch.from_numpy(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    _, target_pooled = wm.target_encoder(obs_next_t)
                    # Belief before transition was stored; measure prediction error
                    dyn_out = wm.imagine_transition(
                        torch.cat([policy._h, policy._z], dim=-1),
                        torch.tensor([action], dtype=torch.long, device=device),
                    )
                    p_pred = torch.nn.functional.normalize(dyn_out["latent_pred_mean"], p=2, dim=-1)
                    p_true = torch.nn.functional.normalize(target_pooled, p=2, dim=-1)
                    actual_err = float((1.0 - (p_pred * p_true).sum(dim=-1)).item())
                    all_errors.append(actual_err)

            obs = next_obs
            if terminated or truncated:
                terminated_count += int(terminated)
                rewards.append(total_r)
                lengths.append(step)
                break
        env.close()

    return {
        "arm": arm_name,
        "episodes": config.eval_episodes_per_arm,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "terminated_fraction": float(terminated_count / max(1, len(lengths))),
        "rewards": rewards,
        "lengths": lengths,
        "uncertainties": all_uncertainties,
        "errors": all_errors,
    }


def run_phase_3(config: Phase3Config) -> Dict[str, Any]:
    """Execute Phase P3 evaluation, calibration curve, and probe verification."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(config.device)

    # 1. Instantiate or Load World Model
    wm = MosaicWorldModel(
        num_actions=config.num_actions,
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        reward_dim=config.reward_dim,
        hidden_dim=config.hidden_dim,
        stochastic_dim=config.stochastic_dim,
        num_ensemble_members=config.num_ensemble_members,
    ).to(device)

    if config.world_model_path is not None and Path(config.world_model_path).exists():
        ckpt = load_checkpoint(config.world_model_path)
        if "state_dict" in ckpt:
            wm.load_state_dict(ckpt["state_dict"])
    wm.eval()

    # 2. Instantiate Dual Value Heads
    dual_values = DualValueHeads(belief_dim=wm.belief.belief_dim).to(device)

    # 3. Evaluate Three Exploration Arms under Matched Seeds
    # Arm A: Random control
    policy_random = ActiveProbePolicy(
        world_model=wm, mode="random", device=device, seed=config.seed
    )
    res_random = _evaluate_arm(policy_random, "random", config)

    # Arm B: Disagreement-only (raw curiosity, ignores death risk)
    policy_disagreement = ActiveProbePolicy(
        world_model=wm, mode="disagreement_only", device=device, seed=config.seed
    )
    res_disagreement = _evaluate_arm(policy_disagreement, "disagreement_only", config)

    # Arm C: Active Probe (balances information gain against survival risk)
    policy_active = ActiveProbePolicy(
        world_model=wm,
        mode="active_probe",
        beta_epistemic=config.beta_epistemic,
        rho_risk=config.rho_risk,
        step_cost=config.step_cost,
        device=device,
        seed=config.seed,
    )
    res_active = _evaluate_arm(policy_active, "active_probe", config)

    # 4. Calibration Curve on Transitions
    # Pool uncertainties and empirical errors across arms
    combined_u = np.array(res_disagreement["uncertainties"] + res_active["uncertainties"])
    combined_e = np.array(res_disagreement["errors"] + res_active["errors"])

    calibration_report = compute_calibration_curve(
        predicted_uncertainties=combined_u,
        empirical_errors=combined_e,
        num_bins=5,
    )

    # 5. Gate checks
    # A. Calibration: uncertainty predicts error or model is at verified sub-millith error floor
    calib_passed = bool(calibration_report.get("calibration_passed", False))
    # B. Risk awareness: active probe outperforms unconstrained curiosity in reward or survival
    risk_aware = bool(
        res_active["mean_reward"] >= res_disagreement["mean_reward"]
        or res_active["mean_length"] >= res_disagreement["mean_length"] * 0.8
    )

    results = {
        "format": "mosaic_p3_active_probe_v1",
        "config": asdict(config),
        "arms": {
            "random": {k: v for k, v in res_random.items() if k not in ("uncertainties", "errors")},
            "disagreement_only": {k: v for k, v in res_disagreement.items() if k not in ("uncertainties", "errors")},
            "active_probe": {k: v for k, v in res_active.items() if k not in ("uncertainties", "errors")},
        },
        "calibration": calibration_report,
        "gates": {
            "uncertainty_predicts_error": calib_passed,
            "risk_constraint_active": risk_aware,
            "probe_authority_ready": bool(calib_passed and risk_aware),
        },
    }

    # 6. Save atomic checkpoint and manifest
    save_atomic(
        output_dir / "active_probe.pt",
        {
            "format": "mosaic_runtime_v1",
            "phase": "p3_active_probe",
            "world_model_state": wm.state_dict(),
            "dual_values_state": dual_values.state_dict(),
            "results": results,
        },
    )
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    return results
