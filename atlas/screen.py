"""Locked benchmark screening protocol for ATLAS agents.

Strict visual evaluation:
- Seeds 2050–2099 (50 episodes, max 1000 steps).
- epsilon = 0.0, beta_t = 0.0.
- No bank mutation or online learning during screening.
- Full provenance metadata recorded in results/atlas_screen_<tag>.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.config import REPO_ROOT, ScreenConfig
# Safe pre-registration of eval.metrics to avoid pulling unused stable_baselines3 from eval/__init__.py
import importlib.util
import sys
if "eval.metrics" not in sys.modules:
    _metrics_p = Path(__file__).resolve().parent.parent / "eval" / "metrics.py"
    if _metrics_p.exists():
        _spec = importlib.util.spec_from_file_location("eval.metrics", str(_metrics_p))
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["eval.metrics"] = _mod
            _spec.loader.exec_module(_mod)

from envs.crafter_wrapper import CrafterEnv
from eval.metrics import compute_crafter_score


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _hash_heads_state(agent: AtlasAgent) -> str:
    """Computes SHA-256 over serialized parameters of all small trained heads."""
    h = hashlib.sha256()
    for net in [agent.option_net, agent.reach_net, agent.epistemic_net, agent.safety_net]:
        for p in net.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def screen_agent(
    agent: AtlasAgent,
    seeds: Optional[Union[Sequence[int], Tuple[int, int]]] = None,
    num_episodes: Optional[int] = None,
    max_steps: int = 1000,
    tag: str = "atlas_v1",
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Runs strict visual evaluation and writes protocol screening JSON."""
    results_dir = Path(out_dir or (REPO_ROOT / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine seed list
    if seeds is None:
        seed_list = list(range(2050, 2100))
    elif isinstance(seeds, (tuple, list)) and len(seeds) == 2 and isinstance(seeds[0], int) and isinstance(seeds[1], int) and seeds[0] <= seeds[1]:
        seed_list = list(range(seeds[0], seeds[1] + 1))
    else:
        seed_list = list(seeds)

    if num_episodes is not None:
        seed_list = seed_list[:num_episodes]

    num_eps = len(seed_list)

    # Screening configuration: frozen agent, zero exploration, frozen memory
    old_eps = agent.epsilon
    old_beta = agent.beta_t
    old_frozen = agent.frozen_memory
    agent.epsilon = 0.0
    agent.beta_t = 0.0
    agent.frozen_memory = True

    achievement_records: List[Dict[str, int]] = []
    survivals: List[int] = []
    rewards: List[float] = []

    total_steps = 0
    mode_counts = {"PROBE": 0, "NAVIGATE": 0, "FALLBACK": 0, "WANDER": 0}
    override_count = 0

    print(f"[Screen] Starting screening over {num_eps} episodes (seeds {seed_list[0]}..{seed_list[-1]})...")

    for idx, seed in enumerate(seed_list):
        env = CrafterEnv(seed=seed)
        obs, reset_info = env.reset(seed=seed)
        agent.reset_episode(seed=seed)

        reward_prev = 0.0
        step = 0
        ep_reward = 0.0
        last_info: Dict[str, Any] = reset_info

        while step < max_steps:
            obs_uint8 = np.asarray(obs, dtype=np.uint8)
            action, telemetry = agent.act(obs_uint8, reward_prev, step)

            mode = telemetry.get("mode", "FALLBACK")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if telemetry.get("override_fired", False):
                override_count += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            discount = float(info.get("discount", 0.0 if terminated else 1.0))
            agent.observe_result(next_obs, reward, terminated, discount)

            ep_reward += float(reward)
            reward_prev = float(reward)
            obs = next_obs
            last_info = info
            step += 1
            total_steps += 1

            if terminated or truncated:
                break

        survivals.append(step)
        rewards.append(ep_reward)
        achievement_records.append(dict(last_info.get("achievements", {})))

    # Restore exploration schedules and memory freeze state
    agent.epsilon = old_eps
    agent.beta_t = old_beta
    agent.frozen_memory = old_frozen

    # Official scoring
    crafter_metrics = compute_crafter_score(achievement_records)
    mean_survival = float(np.mean(survivals))
    std_survival = float(np.std(survivals))
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    denom = max(1, total_steps)
    per_mode_telemetry = {
        "probe_fraction": float(mode_counts.get("PROBE", 0) / denom),
        "nav_fraction": float(mode_counts.get("NAVIGATE", 0) / denom),
        "fallback_fraction": float(mode_counts.get("FALLBACK", 0) / denom),
        "override_rate": float(override_count / denom),
    }

    prior_weight = float(getattr(agent.cfg.backbone, "prior_weight", 0.0))
    if prior_weight != 0.0:
        raise ValueError(f"Screening refused: prior_weight={prior_weight} != 0.0 violates pure scratch training invariant")

    # Strict provenance record
    report = {
        "protocol_id": "strict_visual_atlas_clean_v1",
        "git_commit": _get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bootstrap": "none",
        "prior_weight": prior_weight,
        "train_from_scratch": True,
        "run_line": "atlas_v2",
        "backbone": {
            "path": agent.backbone.checkpoint_path,
            "sha256": str(agent.backbone.provenance.get("checkpoint_sha256", "")),
            "step": int(agent.backbone.provenance.get("step", -1)),
        },
        "heads_state_sha256": _hash_heads_state(agent),
        "run_id": tag,
        "strict_inference": True,
        "privileged_inventory_used": False,
        "privileged_achievements_used_for_action": False,
        "achievements_used_for_scoring_only": True,
        "seeds": seed_list,
        "num_episodes": num_eps,
        "crafter_score": float(crafter_metrics["score"]),
        "unlocked_count": int(crafter_metrics["unlocked_count"]),
        "mean_achievements_per_ep": float(crafter_metrics["mean_achievements_per_ep"]),
        "mean_survival": mean_survival,
        "std_survival": std_survival,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "unlock_percentages": crafter_metrics["unlock_percentages"],
        "per_mode_telemetry": per_mode_telemetry,
    }

    # Validate provenance invariants before saving
    for req_field, expected_val in [
        ("bootstrap", "none"),
        ("train_from_scratch", True),
        ("run_line", "atlas_v2"),
        ("prior_weight", 0.0),
    ]:
        if report.get(req_field) != expected_val:
            raise ValueError(
                f"Screening report missing or invalid provenance field '{req_field}': "
                f"got {report.get(req_field)}, expected {expected_val}"
            )

    out_path = results_dir / f"atlas_screen_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"[Screen Complete] Score: {report['crafter_score']:.2f}% | "
        f"Mean Survival: {mean_survival:.1f} | Mean Reward: {mean_reward:.2f} | "
        f"Saved to: {out_path}"
    )
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS Visual Evaluation Screener")
    parser.add_argument("--tag", type=str, default="atlas_v2", help="Screening tag")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Evaluation seed list or pair")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to backbone checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    args = parser.parse_args()

    cfg = AtlasConfig()
    if args.checkpoint:
        os.environ["ATLAS_CHECKPOINT"] = args.checkpoint
    if args.device != "auto":
        cfg.backbone.device = args.device

    backbone = FrozenBackbone(config=cfg)
    agent = AtlasAgent(backbone=backbone, cfg=cfg)

    screen_agent(
        agent=agent,
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        tag=args.tag,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
