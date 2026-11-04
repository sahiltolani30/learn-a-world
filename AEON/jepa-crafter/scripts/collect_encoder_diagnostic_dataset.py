#!/usr/bin/env python3
"""Collect real Crafter RGB observations for AEON encoder diagnostics.

Run from AEON/jepa-crafter after installing the project's requirements:
    python scripts/collect_encoder_diagnostic_dataset.py --output-dir artifacts/encoder_diagnostic --seeds 101 102 103 104 --transitions-per-seed 64

This script saves only public RGB observations and environment interaction metadata.
It does NOT read Crafter's privileged internal state. Actions are sampled uniformly
from the environment's discrete action space. If AEON training uses an opaque action
permutation wrapper, use the same wrapper/mapping before using these transitions for
action-conditioned predictor comparisons. The images remain valid for encoder tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore


def unpack_reset(result: Any):
    # Gymnasium returns (obs, info); older Gym returns obs.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


def unpack_step(result: Any):
    # Gymnasium: obs, reward, terminated, truncated, info
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), info
    # Older Gym: obs, reward, done, info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), False, info


def to_rgb_array(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC RGB observation, got shape={arr.shape}")
    if arr.shape[-1] != 3:
        # Do not silently transpose or guess observation semantics.
        raise ValueError(f"Expected RGB last dimension of 3, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def make_contact_sheet(image_paths: list[Path], output_path: Path, max_images: int = 100):
    chosen = image_paths[:max_images]
    if not chosen:
        return
    thumb_w, thumb_h = 192, 192
    label_h, cols = 24, 5
    rows = (len(chosen) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(chosen):
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb_w, thumb_h), Image.Resampling.NEAREST)
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + thumb_h + 3), path.stem[:28], fill="black")
    sheet.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CrafterReward-v1")
    parser.add_argument("--output-dir", default="artifacts/encoder_diagnostic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104])
    parser.add_argument("--transitions-per-seed", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=20261104)
    args = parser.parse_args()

    if args.transitions_per_seed < 1 or not args.seeds:
        parser.error("Provide at least one seed and a positive transition count.")

    out = Path(args.output_dir).resolve()
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.base_seed)
    random.seed(args.base_seed)

    rows = []
    image_paths: list[Path] = []
    total = 0

    for env_seed in args.seeds:
        env = gym.make(args.env_id)
        try:
            obs, reset_info = unpack_reset(env.reset(seed=env_seed))
            obs = to_rgb_array(obs)
            episode_id = 0
            episode_step = 0
            episode_seed = env_seed

            while sum(1 for r in rows if r["seed"] == env_seed) < args.transitions_per_seed:
                if episode_step >= args.max_episode_steps:
                    episode_id += 1
                    episode_step = 0
                    episode_seed = env_seed + episode_id * 100003
                    obs, reset_info = unpack_reset(env.reset(seed=episode_seed))
                    obs = to_rgb_array(obs)

                action_count = getattr(env.action_space, "n", None)
                if action_count is None:
                    raise TypeError(f"Expected discrete action space, got {env.action_space!r}")
                action = int(rng.integers(0, action_count))
                next_obs, reward, terminated, truncated, step_info = unpack_step(env.step(action))
                next_obs = to_rgb_array(next_obs)

                stem = f"seed{env_seed}_ep{episode_id:02d}_step{episode_step:04d}"
                obs_name = f"{stem}_obs.png"
                next_name = f"{stem}_next.png"
                Image.fromarray(obs).save(images_dir / obs_name)
                Image.fromarray(next_obs).save(images_dir / next_name)
                image_paths.extend([images_dir / obs_name, images_dir / next_name])

                rows.append({
                    "seed": env_seed,
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "step": episode_step,
                    "observation_file": f"images/{obs_name}",
                    "next_observation_file": f"images/{next_name}",
                    "action_index_sent_to_env": action,
                    "action_count": int(action_count),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "episode_boundary": bool(terminated or truncated),
                    "observation_shape": list(obs.shape),
                    "observation_dtype": "uint8",
                })
                total += 1
                obs = next_obs
                episode_step += 1

                if terminated or truncated:
                    episode_id += 1
                    episode_step = 0
                    episode_seed = env_seed + episode_id * 100003
                    obs, reset_info = unpack_reset(env.reset(seed=episode_seed))
                    obs = to_rgb_array(obs)
        finally:
            env.close()

    csv_path = out / "transitions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset": "AEON Crafter encoder diagnostic",
        "env_id": args.env_id,
        "seeds": args.seeds,
        "transitions_per_seed": args.transitions_per_seed,
        "transition_count": len(rows),
        "image_count": len(image_paths),
        "base_seed_for_action_sampling": args.base_seed,
        "action_sampling": "uniform random over env.action_space.n",
        "observation_source": "public RGB observation returned by environment",
        "privileged_state_used": False,
        "action_mapping_warning": (
            "Action indices are sent directly to the environment. If AEON training uses "
            "an opaque action permutation wrapper, apply the same wrapper/mapping before "
            "using this dataset for action-conditioned predictor evaluation."
        ),
        "files": {
            "transitions": "transitions.csv",
            "images_dir": "images/",
            "contact_sheet": "contact_sheet.png",
        },
        "python": sys.version,
        "numpy": np.__version__,
    }
    (out / "collection_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_contact_sheet(image_paths, out / "contact_sheet.png", max_images=100)

    print(f"Saved {len(rows)} transitions and {len(image_paths)} PNGs to: {out}")
    print(f"Transition table: {csv_path}")
    print(f"Contact sheet: {out / 'contact_sheet.png'}")
    print("No training was run. Review metadata/action mapping before predictor tests.")


if __name__ == "__main__":
    main()
