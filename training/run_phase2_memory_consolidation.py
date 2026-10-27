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
        model.load_checkpoint(str(ckpt_p), map_location=dev)
        print(f"-> Loaded base model from {ckpt_p}")
    model.to(dev)
    model.eval()

    # 2. Build Evidence Store
    store = EpisodicEvidenceStore(max_capacity=100000, gamma=0.99)

    # 3. Ingest Episodes
    man_p = Path(manifest_path)
    if not man_p.exists():
        # Fallback to scanning shard files
        shard_dir = REPO_ROOT / "results" / "sequences"
        shard_files = sorted(list(shard_dir.glob("*.npz")))
    else:
        with open(man_p, "r") as f:
            manifest = json.load(f)
        shard_files = [Path(p) for p in manifest.get("train_shards", []) if Path(p).exists()]

    print(f"-> Ingesting trajectory evidence from {len(shard_files)} shard files...")
    episodes_ingested = 0
    transitions_ingested = 0

    with torch.no_grad():
        for s_idx, sf in enumerate(shard_files):
            if episodes_ingested >= max_episodes:
                break
            try:
                data = np.load(str(sf), allow_pickle=True)
                obs = data.get("obs")
                actions = data.get("actions")
                rewards = data.get("rewards")
                dones = data.get("dones")
                if dones is None:
                    dones = data.get("done")

                if obs is None or actions is None or rewards is None:
                    continue

                N = len(actions)
                if N == 0:
                    continue

                # Batch encode observations through visual encoder
                # Sample window
                window_len = min(N, 64)
                obs_t = torch.from_numpy(obs[:window_len]).permute(0, 3, 1, 2).float() / 255.0
                obs_t = obs_t.to(dev)
                z_all = model.encode(obs_t, use_target=True)  # [W, 128]

                # Recurrent unroll for state readout
                h_t = torch.zeros(1, 256, device=dev)
                states_all = []
                for t in range(window_len):
                    s_t = torch.cat([z_all[t:t+1], h_t], dim=-1)
                    x_hat_t, _ = model.predict_state(s_t)
                    states_all.append(x_hat_t.squeeze(0))
                    a_t = torch.tensor([actions[t]], dtype=torch.long, device=dev)
                    _, h_t = model.recurrent_step(z_all[t:t+1], a_t, h_t)

                for t in range(window_len - 1):
                    act = int(actions[t])
                    rew = float(rewards[t])
                    d = bool(dones[t]) if dones is not None else False

                    store.record_step(
                        obs_latent=z_all[t],
                        grounded_state=states_all[t],
                        action=act,
                        reward=rew,
                        done=d,
                        next_obs_latent=z_all[t + 1],
                        next_grounded_state=states_all[t + 1],
                        episode_id=s_idx,
                        step_id=t,
                    )
                    transitions_ingested += 1

                # Commit final window step
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
