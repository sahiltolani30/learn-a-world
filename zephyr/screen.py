"""Zephyr locked benchmark screen (protocol: strict_visual_zephyr_clean_v1).

Strict-inference benchmark: the agent receives ONLY raw pixels on its action
path. Env achievements are read after each episode for SCORING ONLY. The output
JSON records protocol metadata, provenance (git commit, checkpoint SHA-256),
per-episode logs and per-achievement unlock percentages, scored by the official
`eval.metrics.compute_crafter_score` (read-only whitelist import, identical to
the previous line's scorer).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from zephyr.config import ACHIEVEMENTS, IMG_SIZE, ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.planner import ZephyrAgent, ZephyrCEMAgent
from zephyr.wm import ZephyrWM

_PROTOCOL_ID = "strict_visual_zephyr_clean_v1"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _assert_achievements_match_scorer() -> None:
    from eval.metrics import CRAFTER_ACHIEVEMENTS

    if tuple(CRAFTER_ACHIEVEMENTS) != ACHIEVEMENTS:
        raise RuntimeError(
            "config.ACHIEVEMENTS does not match eval.metrics.CRAFTER_ACHIEVEMENTS; "
            "scoring order must be identical"
        )


def _build_agent(
    wm: ZephyrWM, heads: ZephyrHeads, agent_mode: str, device: str
):
    if agent_mode == "actor":
        return ZephyrAgent(wm, heads, device=device)
    if agent_mode == "cem":
        return ZephyrCEMAgent(wm, heads, device=device)
    raise ValueError(f"unknown agent_mode {agent_mode!r} (use 'actor' or 'cem')")


def run_screen(
    checkpoint_wm: str,
    checkpoint_heads: str,
    out_path: str,
    num_episodes: int = 50,
    seed_start: int = 2050,
    max_steps: int = 1000,
    agent_mode: str = "actor",
    device: str = "cpu",
    drive_mirror: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the locked benchmark and write (and return) the report dict."""
    _assert_achievements_match_scorer()
    from eval.metrics import compute_crafter_score  # read-only whitelist import

    wm, wm_meta = ZephyrWM.load_checkpoint(checkpoint_wm, device=device)
    heads, heads_meta = ZephyrHeads.load_checkpoint(checkpoint_heads, device=device)
    if wm.cfg.to_dict() != heads.cfg.to_dict():
        raise ValueError("world-model and heads checkpoint configs differ")
    cfg: ZephyrConfig = wm.cfg

    agent = _build_agent(wm, heads, agent_mode, device)

    episode_logs: List[Dict[str, Any]] = []
    achievement_records: List[Dict[str, int]] = []
    survival_lengths: List[int] = []
    episode_rewards: List[float] = []

    from envs.crafter_wrapper import CrafterEnv  # read-only whitelist import

    for ep in range(num_episodes):
        seed = seed_start + ep
        env = CrafterEnv(seed=seed, render_size=IMG_SIZE, max_steps=max_steps)
        obs, _info = env.reset(seed=seed)
        agent.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        while steps < max_steps:
            action = agent.act(obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if term or trunc:
                terminated = True
                break
        achievements = {k: int(v) for k, v in (info.get("achievements") or {}).items() if v}
        achievement_records.append(achievements)
        survival_lengths.append(steps)
        episode_rewards.append(total_reward)
        episode_logs.append(
            {
                "episode": ep,
                "seed": seed,
                "steps": steps,
                "reward": round(total_reward, 4),
                "ended": terminated,
                "achievements": sorted(achievements.keys()),
            }
        )
        env.close()

    score_stats = compute_crafter_score(achievement_records)
    report: Dict[str, Any] = {
        "protocol_id": _PROTOCOL_ID,
        "protocol_family": "zephyr_locked_50ep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint_wm": os.path.abspath(checkpoint_wm),
        "checkpoint_heads": os.path.abspath(checkpoint_heads),
        "checkpoint_wm_sha256": _sha256(checkpoint_wm),
        "checkpoint_heads_sha256": _sha256(checkpoint_heads),
        "wm_meta_step": wm_meta.get("step"),
        "agent_mode": agent_mode,
        "device": str(device),
        "num_episodes": num_episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + num_episodes - 1,
        "max_steps": max_steps,
        # strict-inference declaration
        "strict_inference": True,
        "inference_access": "strict_visual_zephyr",
        "privileged_inventory_used": False,
        "privileged_achievements_used_for_action": False,
        "achievements_used_for_scoring_only": True,
        "semantic_memory_enabled": False,
        "adaptation_mode": "frozen",
        "training_provenance": "self_play_pixels_only",
        "planner_configuration": {
            "agent_mode": agent_mode,
            "horizon": getattr(agent, "horizon", None),
            "candidates": getattr(agent, "candidates", None),
            "imagination_horizon_train": cfg.ac.imagination_horizon,
        },
        "config": cfg.to_dict(),
        "crafter_score": score_stats["score"],
        "unlocked_achievement_count": score_stats["unlocked_count"],
        "mean_achievements_per_ep": score_stats["mean_achievements_per_ep"],
        "unlock_percentages": score_stats["unlock_percentages"],
        "mean_survival_length": float(np.mean(survival_lengths)),
        "std_survival_length": float(np.std(survival_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "episode_logs": episode_logs,
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    os.replace(tmp_path, out_path)
    if drive_mirror:
        os.makedirs(os.path.dirname(os.path.abspath(drive_mirror)), exist_ok=True)
        shutil.copyfile(out_path, drive_mirror)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-wm", "--checkpoint_wm", required=True)
    parser.add_argument("--checkpoint-heads", "--checkpoint_heads", required=True)
    parser.add_argument("--out-path", "--out_path", required=True)
    parser.add_argument("--num-episodes", "--num_episodes", type=int, default=50)
    parser.add_argument("--seed-start", "--seed_start", type=int, default=2050)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=1000)
    parser.add_argument("--agent-mode", "--agent_mode", choices=["actor", "cem"], default="actor")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--drive-mirror", "--drive_mirror", default=None)
    args = parser.parse_args()

    report = run_screen(
        checkpoint_wm=args.checkpoint_wm,
        checkpoint_heads=args.checkpoint_heads,
        out_path=args.out_path,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        agent_mode=args.agent_mode,
        device=args.device,
        drive_mirror=args.drive_mirror,
    )
    print(f"[zephyr-screen] complete: score={report['crafter_score']:.2f}% "
          f"mean_surv={report['mean_survival_length']:.1f} steps")


if __name__ == "__main__":
    main()

