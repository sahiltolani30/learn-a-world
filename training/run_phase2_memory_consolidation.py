#!/usr/bin/env python3
"""
training/run_phase2_memory_consolidation.py — Phase 2 Memory Consolidation Pipeline
===================================================================================

Extracts empirical trajectory evidence from gameplay sequences, calculates Monte Carlo
discounted returns, and consolidates them into learned TransitionPrototypes in the
RelationHypothesisBank.

Zero hardcoded crafting rules:
- Preconditions discovered from empirical StateReadoutHead estimates.
- Effect deltas discovered from measured state deltas.
- Prototypes ranked by success probability, expected return, and Bayesian confidence.

Outputs:
- checkpoints/phase2_causal_memory_bank.json
- results/diagnostics/phase2_memory_consolidation.json
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.jepa_v3 import JEPAv3
from planning.causal_memory import EpisodicEvidenceStore, RelationHypothesisBank
from training.sequence_buffer import EpisodeBuffer


def consolidate_phase2_memory(
    checkpoint_path: str = "checkpoints/jepa_crafter_v3_state_readout.pt",
    manifest_path: str = "results/phase1_5_manifest.json",
    output_bank_path: str = "checkpoints/phase2_causal_memory_bank.json",
    diagnostic_path: str = "results/diagnostics/phase2_memory_consolidation.json",
    max_episodes: int = 500,
    min_cluster_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    dev = torch.device(device)
    print("=" * 80)
    print("       PHASE 2: GROUNDED PERSISTENT MEMORY CONSOLIDATION")
    print(f"       Checkpoint: {checkpoint_path}")
    print(f"       Manifest: {manifest_path}")
    print(f"       Max Episodes to Ingest: {max_episodes}")
    print(f"       Device: {dev}")
    print("=" * 80)

    # 1. Load JEPA Model
    model = JEPAv3()
    ckpt_p = Path(checkpoint_path)
    if ckpt_p.exists():
        model.load_checkpoint(str(ckpt_p), device=dev)
        print(f"-> Loaded base model from {ckpt_p}")
    model.to(dev)
    model.eval()

    # 2. Build Evidence Store
    store = EpisodicEvidenceStore(max_capacity=100000, gamma=0.99)

    # 3. Ingest Episodes
    shard_files = []
    man_p = Path(manifest_path) if manifest_path else None
    if man_p and man_p.exists():
        with open(man_p, "r") as f:
            manifest = json.load(f)
        raw_shards = manifest.get("train_shards", manifest.get("selected_shards", []))
        for item in raw_shards:
            p_str = item["path"] if isinstance(item, dict) else str(item)
            p = Path(p_str)
            if p.exists():
                shard_files.append(p)
            elif (REPO_ROOT / p_str).exists():
                shard_files.append(REPO_ROOT / p_str)
            elif (REPO_ROOT / "results" / p.name).exists():
                shard_files.append(REPO_ROOT / "results" / p.name)
            elif (REPO_ROOT / "results" / "sequences" / p.name).exists():
                shard_files.append(REPO_ROOT / "results" / "sequences" / p.name)

    # Fallback to direct directory scans if none found from manifest
    if not shard_files:
        search_dirs = [
            REPO_ROOT / "results" / "sequences",
            REPO_ROOT / "results",
            Path("/content/drive/MyDrive/learn-a-world/results/sequences"),
            Path("/content/drive/MyDrive/learn-a-world/results"),
        ]
        for sdir in search_dirs:
            if sdir.exists():
                found = sorted(list(sdir.glob("*.npz")))
                if found:
                    shard_files.extend(found)
                    break

    print(f"-> Ingesting trajectory evidence from {len(shard_files)} shard files...")
    episodes_ingested = 0
    transitions_ingested = 0

    with torch.no_grad():
        for s_idx, sf in enumerate(shard_files):
            if episodes_ingested >= max_episodes:
                break
            try:
                data = np.load(str(sf), allow_pickle=True)
                cat_obs = data.get("obs")
                cat_actions = data.get("actions")
                cat_rewards = data.get("rewards")
                cat_dones = data.get("dones")
                if cat_dones is None:
                    cat_dones = data.get("done")

                if cat_obs is None or cat_actions is None or cat_rewards is None:
                    continue

                ep_lens = data.get("episode_lengths")
                if ep_lens is None:
                    if cat_dones is not None and np.any(cat_dones):
                        done_idx = np.where(cat_dones)[0]
                        ep_lens = np.diff(np.concatenate([[-1], done_idx]))
                    else:
                        ep_lens = np.array([len(cat_actions)], dtype=np.int64)

                offset = 0
                for ep_idx, ep_len in enumerate(ep_lens):
                    if episodes_ingested >= max_episodes:
                        break
                    ep_len = int(ep_len)
                    if ep_len <= 5:
                        offset += ep_len
                        continue

                    ep_o = cat_obs[offset : offset + ep_len]
                    ep_a = cat_actions[offset : offset + ep_len]
                    ep_r = cat_rewards[offset : offset + ep_len]
                    ep_d = cat_dones[offset : offset + ep_len] if cat_dones is not None else np.zeros(ep_len, dtype=bool)
                    offset += ep_len

                    # Batch encode visual observations in 64-step chunks to fit GPU VRAM
                    chunk_size = 64
                    z_chunks = []
                    for b_start in range(0, ep_len, chunk_size):
                        b_end = min(b_start + chunk_size, ep_len)
                        obs_b = torch.from_numpy(ep_o[b_start:b_end]).permute(0, 3, 1, 2).float() / 255.0
                        obs_b = obs_b.to(dev)
                        z_b = model.encode(obs_b, use_target=True)
                        z_chunks.append(z_b)
                    z_all = torch.cat(z_chunks, dim=0)

                    # Recurrent unroll for state readout across the complete episode
                    h_t = torch.zeros(1, 256, device=dev)
                    states_all = []
                    for t in range(ep_len):
                        s_t = torch.cat([z_all[t:t+1], h_t], dim=-1)
                        x_hat_t, _ = model.predict_state(s_t)
                        states_all.append(x_hat_t.squeeze(0))
                        a_t = torch.tensor([ep_a[t]], dtype=torch.long, device=dev)
                        _, h_t = model.recurrent_step(z_all[t:t+1], a_t, h_t)

                    for t in range(ep_len - 1):
                        act = int(ep_a[t])
                        rew = float(ep_r[t])
                        d = bool(ep_d[t])

                        store.record_step(
                            obs_latent=z_all[t],
                            grounded_state=states_all[t],
                            action=act,
                            reward=rew,
                            done=d,
                            next_obs_latent=z_all[t + 1],
                            next_grounded_state=states_all[t + 1],
                            episode_id=episodes_ingested,
                            step_id=t,
                        )
                        transitions_ingested += 1

                    # Commit full episode to evidence store
                    store.commit_episode()
                    episodes_ingested += 1

            except Exception as e:
                continue

    print(f"-> Ingested {transitions_ingested} transitions across {episodes_ingested} episodes into Evidence Store.")

    # 4. Consolidate into Relation Hypothesis Bank
    bank = RelationHypothesisBank(device=dev)
    proto_count = bank.consolidate_from_evidence(store, min_cluster_size=min_cluster_size)
    print(f"-> Consolidating complete: Discovered {proto_count} TransitionPrototypes.")

    # 5. Save Artifacts
    out_bank_p = Path(output_bank_path)
    out_bank_p.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out_bank_p)
    print(f"-> Saved Relation Hypothesis Bank to: {out_bank_p}")

    diag_summary = {
        "phase": "phase2_grounded_memory_consolidation",
        "checkpoint": str(checkpoint_path),
        "episodes_ingested": episodes_ingested,
        "transitions_ingested": transitions_ingested,
        "prototypes_discovered": proto_count,
        "bank_path": str(out_bank_p),
        "prototype_summary": [
            {
                "id": p.prototype_id,
                "action": p.action,
                "success_count": p.success_count,
                "failure_count": p.failure_count,
                "success_prob": p.success_probability,
                "confidence": p.confidence,
                "mean_return": p.mean_return,
                "preconditions": [round(float(v), 2) for v in p.precondition_state],
                "effect_delta": [round(float(v), 2) for v in p.effect_delta],
            }
            for p in bank.prototypes.values()
        ],
    }

    diag_p = Path(diagnostic_path)
    diag_p.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_p, "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"-> Saved consolidation diagnostic to: {diag_p}")

    return diag_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Memory Consolidation.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_state_readout.pt")
    parser.add_argument("--manifest_path", type=str, default="results/phase1_5_manifest.json")
    parser.add_argument("--output_bank", type=str, default="checkpoints/phase2_causal_memory_bank.json")
    parser.add_argument("--diagnostic_path", type=str, default="results/diagnostics/phase2_memory_consolidation.json")
    parser.add_argument("--max_episodes", type=int, default=300)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    args = parser.parse_args()

    consolidate_phase2_memory(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest_path,
        output_bank_path=args.output_bank,
        diagnostic_path=args.diagnostic_path,
        max_episodes=args.max_episodes,
        min_cluster_size=args.min_cluster_size,
    )
