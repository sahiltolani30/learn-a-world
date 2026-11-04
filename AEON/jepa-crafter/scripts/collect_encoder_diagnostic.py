#!/usr/bin/env python3
"""Collect real RGB observations and action transitions from Crafter for AEON diagnostics.

Run from AEON/jepa-crafter (or pass --repo-root). This script does not train or
modify AEON. It uses only rendered RGB observations for the saved dataset;
privileged info is intentionally not serialized.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def make_env(env_id: str):
    """Prefer Gymnasium, fall back to Gym for older Crafter installs."""
    errors = []
    for module_name in ("gymnasium", "gym"):
        try:
            module = __import__(module_name)
            return module.make(env_id), module.__name__
        except Exception as exc:  # report both possible import/registration errors
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not create Crafter environment. " + " | ".join(errors))


def unpack_reset(result):
    # Gymnasium: (obs, info); older Gym: obs
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0]
    return result


def unpack_step(result):
    # Gymnasium: obs, reward, terminated, truncated, info
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), info
    # Older Gym: obs, reward, done, info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), False, info


def as_rgb(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC RGB observation, got shape {arr.shape}")
    # Accommodate channel-first observations if encountered.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB-like observation, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CrafterReward-v1")
    p.add_argument("--out", default="artifacts/encoder_diagnostic")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104])
    p.add_argument("--transitions-per-seed", type=int, default=64)
    p.add_argument("--max-steps-per-episode", type=int, default=1000)
    p.add_argument("--contact-sheet-count", type=int, default=100)
    p.add_argument("--action-mode", choices=["random"], default="random")
    args = p.parse_args()

    if args.transitions_per_seed < 1:
        p.error("--transitions-per-seed must be >= 1")
    out = Path(args.out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    env, gym_name = make_env(args.env_id)

    transitions = []
    image_records = []
    saved_images = set()
    episode_counter = 0
    try:
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            try:
                env.action_space.seed(seed)
            except Exception:
                pass
            obs = as_rgb(unpack_reset(env.reset(seed=seed)))
            episode_id = 0
            step_in_episode = 0
            for transition_idx in range(args.transitions_per_seed):
                # Save the current observation once. The action is opaque: only its
                # integer index is recorded, never decoded using environment internals.
                image_name = f"seed{seed}_ep{episode_id:03d}_step{step_in_episode:05d}.png"
                if image_name not in saved_images:
                    Image.fromarray(obs).save(images_dir / image_name)
                    saved_images.add(image_name)
                    image_records.append({"image": image_name, "seed": seed,
                                          "episode": episode_id, "step": step_in_episode})
                action = int(env.action_space.sample())
                result = env.step(action)
                next_obs_raw, reward, terminated, truncated, _info = unpack_step(result)
                next_obs = as_rgb(next_obs_raw)
                next_name = f"seed{seed}_ep{episode_id:03d}_step{step_in_episode + 1:05d}.png"
                if next_name not in saved_images:
                    Image.fromarray(next_obs).save(images_dir / next_name)
                    saved_images.add(next_name)
                    image_records.append({"image": next_name, "seed": seed,
                                          "episode": episode_id, "step": step_in_episode + 1})
                transitions.append({
                    "transition_id": len(transitions),
                    "seed": seed,
                    "episode": episode_id,
                    "step": step_in_episode,
                    "observation": f"images/{image_name}",
                    "action_index": action,
                    "next_observation": f"images/{next_name}",
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "observation_shape": list(obs.shape),
                    "observation_dtype": "uint8",
                })
                step_in_episode += 1
                obs = next_obs
                if terminated or truncated or step_in_episode >= args.max_steps_per_episode:
                    # Never serialize a transition from one episode's final frame
                    # to the next episode's initial frame.
                    episode_counter += 1
                    episode_id += 1
                    step_in_episode = 0
                    obs = as_rgb(unpack_reset(env.reset()))
            # Distinct seeds are explicit; no hidden state or semantic labels saved.
    finally:
        env.close()

    jsonl_path = out / "transitions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in transitions:
            f.write(json.dumps(row) + "\n")

    csv_path = out / "transitions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(transitions[0].keys()))
        writer.writeheader()
        writer.writerows(transitions)

    with (out / "images_index.json").open("w", encoding="utf-8") as f:
        json.dump(image_records, f, indent=2)

    # Contact sheet: evenly sample image records, preserving labels for inspection.
    sample_records = image_records
    if len(sample_records) > args.contact_sheet_count:
        indices = np.linspace(0, len(sample_records) - 1, args.contact_sheet_count, dtype=int)
        sample_records = [sample_records[i] for i in indices]
    thumb_w, thumb_h, label_h, cols = 160, 160, 30, 5
    rows = (len(sample_records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, rec in enumerate(sample_records):
        im = Image.open(images_dir / rec["image"]).convert("RGB")
        im.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        sheet.paste(im, (x, y))
        draw.text((x + 3, y + thumb_h + 2),
                  f"s{rec['seed']} e{rec['episode']} t{rec['step']}", fill="black")
    sheet.save(out / "contact_sheet.png")

    metadata = {
        "dataset": "AEON Crafter encoder diagnostic",
        "env_id": args.env_id,
        "gym_backend": gym_name,
        "seeds": args.seeds,
        "transitions_per_seed_requested": args.transitions_per_seed,
        "transition_count": len(transitions),
        "unique_saved_image_count": len(saved_images),
        "observation_shape": transitions[0]["observation_shape"] if transitions else None,
        "observation_dtype": "uint8",
        "action_count": getattr(env.action_space, "n", None),
        "action_mode": args.action_mode,
        "notes": [
            "Only RGB observations, opaque action indices, rewards, and episode flags are saved.",
            "Environment info dictionaries and privileged semantic state are deliberately not saved.",
            "Check transitions.jsonl to verify no transition crosses an episode boundary.",
            "This script collects data only; it does not load a checkpoint or train the model."
        ]
    }
    with (out / "collection_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps({
        "output_dir": str(out.resolve()),
        "transition_count": len(transitions),
        "image_count": len(saved_images),
        "contact_sheet": str((out / "contact_sheet.png").resolve()),
        "metadata": str((out / "collection_metadata.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
