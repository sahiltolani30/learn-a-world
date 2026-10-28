#!/usr/bin/env python3
"""
dual_store/consolidate_memory.py — Dual-Store Offline Memory Consolidation Pipeline
==================================================================================

Ingests offline trajectory evidence from gameplay sequences, calculates Monte Carlo
discounted returns, and consolidates them into learned TransitionPrototypes in the
Dual-Store RelationHypothesisBank using the calibrated JEPA v3 State Readout model
(jepa_crafter_v3_state_readout_v2.pt).

Key Guarantees (The Iron Rule):
- Zero hardcoded crafting rules: all preconditions and effects are visually grounded.
- Preserves rare achievement milestones (min_cluster_size=1 for grounded event cards).
- Strictly item-based causal DAG edges (0 <= d < 12), preventing spurious vitals edges.
- Dual serialization: exports both .json and PyTorch .pt bank formats.
- Automatic Drive mirroring when running on Google Colab.

Outputs:
- checkpoints/dual_store_causal_memory_bank.json
- checkpoints/dual_store_memory_bank.pt
- results/diagnostics/dual_store_memory_consolidation.json
"""

import argparse
import hashlib
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

from dual_store.jepa_v3 import JEPAv3
from dual_store.causal_memory import EpisodicEvidenceStore, RelationHypothesisBank


def consolidate_dual_store_memory(
    checkpoint_path: str = "checkpoints/jepa_crafter_v3_state_readout_v2.pt",
    manifest_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_json_path: str = "checkpoints/dual_store_causal_memory_bank.json",
    output_pt_path: str = "checkpoints/dual_store_memory_bank.pt",
    diagnostic_path: str = "results/diagnostics/dual_store_memory_consolidation.json",
    max_episodes: int = 500,
    min_cluster_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    drive_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        print("-> [Notice] CUDA requested but not available; automatically falling back to CPU.")
        device = "cpu"
    dev = torch.device(device)
    print("=" * 80)
    print("       DUAL-STORE: OFFLINE GROUNDED MEMORY CONSOLIDATION")
    print(f"       Model Checkpoint: {checkpoint_path}")
    print(f"       Device: {dev}")
    print(f"       Max Episodes to Ingest: {max_episodes}")
    print(f"       Output JSON: {output_json_path}")
    print(f"       Output PT:   {output_pt_path}")
    print("=" * 80)

    # 1. Resolve Checkpoint
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        candidates = [
            REPO_ROOT / checkpoint_path,
            REPO_ROOT / "checkpoints" / ckpt_p.name,
            Path("/content/drive/MyDrive/learn-a-world") / checkpoint_path,
            Path("/content/drive/MyDrive/learn-a-world/checkpoints") / ckpt_p.name,
            Path("/content/learn-a-world") / checkpoint_path,
        ]
        if drive_dir:
            candidates.extend([Path(drive_dir) / checkpoint_path, Path(drive_dir) / "checkpoints" / ckpt_p.name])
        for c in candidates:
            if c.exists() and c.is_file():
                ckpt_p = c
                break

    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 2. Load JEPA Model
    model = JEPAv3()
    model.load_checkpoint(str(ckpt_p), device=dev)
    print(f"-> Successfully loaded base model from: {ckpt_p}")
    model.to(dev)
    model.eval()

    # 3. Build Evidence Store
    store = EpisodicEvidenceStore(max_capacity=100000, gamma=0.99)

    # 4. Ingest Episodes from Shards
    shard_files: List[Path] = []
    
    # Priority A: Check manifest if provided or search default manifests
    manifest_candidates = [
        manifest_path,
        "results/drive_data_audit_manifest.json",
        "results/phase1_5_manifest.json",
        str(REPO_ROOT / "results/drive_data_audit_manifest.json"),
        str(REPO_ROOT / "results/phase1_5_manifest.json"),
    ]
    for mc in manifest_candidates:
        if mc and Path(mc).exists():
            try:
                with open(mc, "r") as f:
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
                if shard_files:
                    print(f"-> Located {len(shard_files)} shards via manifest: {mc}")
                    break
            except Exception:
                pass

    # Priority B: Direct directory scans
    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    search_dirs.extend([
        REPO_ROOT / "results" / "sequences",
        REPO_ROOT / "results",
        Path("/content/drive/MyDrive/learn-a-world/results/sequences"),
        Path("/content/drive/MyDrive/learn-a-world/results"),
        Path("/content/learn-a-world/results/sequences"),
    ])
    if drive_dir:
        search_dirs.extend([Path(drive_dir) / "results" / "sequences", Path(drive_dir) / "results"])

    for sdir in search_dirs:
        if sdir.exists():
            for npz_p in sorted(sdir.glob("*.npz")):
                if npz_p not in shard_files:
                    shard_files.append(npz_p)

    if not shard_files:
        raise FileNotFoundError("No sequence shards (.npz) found for offline consolidation!")

    print(f"-> Ingesting trajectory evidence from {len(shard_files)} shard file(s)...")
    episodes_ingested = 0
    transitions_ingested = 0
    degraded_steps = 0

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
                cat_events = data.get("events")
                cat_mask_before = data.get("mask_before")
                shards_with_events = cat_events is not None and cat_mask_before is not None

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
                    if shards_with_events:
                        ep_ev = cat_events[offset : offset + ep_len].astype(np.int64)
                        ep_mb = cat_mask_before[offset : offset + ep_len].astype(np.int64)
                    else:
                        ep_ev = ep_mb = None
                    offset += ep_len

                    # Batch encode visual observations in 64-step chunks
                    chunk_size = 64
                    z_chunks = []
                    for b_start in range(0, ep_len, chunk_size):
                        b_end = min(b_start + chunk_size, ep_len)
                        obs_b = torch.from_numpy(ep_o[b_start:b_end]).permute(0, 3, 1, 2).float() / 255.0
                        obs_b = obs_b.to(dev)
                        z_b = model.encode(obs_b, use_target=True)
                        z_chunks.append(z_b)
                    z_all = torch.cat(z_chunks, dim=0)

                    # Recurrent unroll for calibrated state readout
                    h_t = torch.zeros(1, 256, device=dev)
                    states_all = []
                    for t in range(ep_len):
                        s_t = torch.cat([z_all[t:t+1], h_t], dim=-1)
                        x_hat_t, _ = model.predict_state(s_t)
                        states_all.append(x_hat_t.squeeze(0))
                        a_t = torch.tensor([ep_a[t]], dtype=torch.long, device=dev)
                        _, h_t = model.recurrent_step(z_all[t:t+1], a_t, h_t)

                    # Record steps into evidence store
                    for t in range(ep_len):
                        act = int(ep_a[t])
                        rew = float(ep_r[t])
                        d = bool(ep_d[t]) or (t == ep_len - 1)
                        is_final = (t == ep_len - 1)

                        if ep_ev is not None:
                            u_t = (ep_ev[t] * (1 - ep_mb[t])).astype(np.int64)
                            m_before = ep_mb[t].astype(np.int64)
                        else:
                            u_t = np.zeros(22, dtype=np.int64)
                            m_before = np.zeros(22, dtype=np.int64)
                            degraded_steps += 1

                        store.record_step(
                            obs_latent=z_all[t],
                            grounded_state=states_all[t],
                            action=act,
                            reward=rew,
                            done=d,
                            next_obs_latent=z_all[t] if is_final else z_all[t + 1],
                            next_grounded_state=states_all[t] if is_final else states_all[t + 1],
                            episode_id=episodes_ingested,
                            step_id=t,
                            event_vec=u_t.tolist(),
                            mask_before_vec=m_before.tolist(),
                        )
                        transitions_ingested += 1

                    store.commit_episode()
                    episodes_ingested += 1
                    if episodes_ingested % 25 == 0 or episodes_ingested == max_episodes:
                        print(f"   [Consolidating] Episode {episodes_ingested}/{max_episodes} ({transitions_ingested} transitions)...", flush=True)

            except Exception as e:
                print(f"   [Warning] Error reading {sf}: {e}")
                continue

    print(f"-> Ingested {transitions_ingested} transitions across {episodes_ingested} episodes into Evidence Store.")

    # 5. Consolidate into Relation Hypothesis Bank
    bank = RelationHypothesisBank(device=dev)
    proto_count = bank.consolidate_from_evidence(store, min_cluster_size=min_cluster_size)
    print(f"-> Consolidation complete: Discovered {proto_count} TransitionPrototypes.")

    # 6. Save Artifacts: Both JSON and PT
    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out_json)
    print(f"-> [Saved] Dual-Store Causal Memory Bank (JSON): {out_json}")

    out_pt = Path(output_pt_path)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    bank.save(out_pt)
    print(f"-> [Saved] Dual-Store Memory Bank (PT): {out_pt}")

    # 7. Auto-Mirror to Google Drive if available
    resolved_drive = Path(drive_dir) if drive_dir else Path("/content/drive/MyDrive/learn-a-world")
    if resolved_drive.exists():
        try:
            drive_ckpt_dir = resolved_drive / "checkpoints"
            drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_json), str(drive_ckpt_dir / out_json.name))
            shutil.copy2(str(out_pt), str(drive_ckpt_dir / out_pt.name))
            print(f"-> [Drive Sync] Mirrored memory banks to: {drive_ckpt_dir}")
        except Exception as e:
            print(f"-> [Warning] Failed to mirror memory bank to Drive: {e}")

    # 8. Diagnostic Summary
    diag_summary = {
        "pipeline": "dual_store_offline_memory_consolidation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": (
            hashlib.sha256(ckpt_p.read_bytes()).hexdigest()
            if ckpt_p.exists() else None
        ),
        "episodes_ingested": episodes_ingested,
        "transitions_ingested": transitions_ingested,
        "degraded_label_steps": degraded_steps,
        "label_coverage": 1.0 - (degraded_steps / max(transitions_ingested, 1)),
        "prototypes_discovered": proto_count,
        "bank_json_path": str(out_json),
        "bank_pt_path": str(out_pt),
        "edges_count": sum(len(el) for el in bank.edges.values()),
        "prototype_summary": [
            {
                "id": p.prototype_id,
                "action": p.action,
                "event_dim": p.event_dim,
                "status": p.status,
                "success_count": p.success_count,
                "failure_count": p.failure_count,
                "success_prob": p.success_probability,
                "confidence": p.confidence,
                "wilson_lb": p.wilson_lb,
                "mean_return": p.mean_return,
                "needs": [[int(d), round(float(a), 3)] for d, a in p.needs],
                "produces": [[int(d), round(float(a), 3)] for d, a in p.produces],
                "consumes": [[int(d), round(float(a), 3)] for d, a in p.consumes],
                "preconditions": [round(float(v), 2) for v in p.precondition_state],
                "effect_delta": [round(float(v), 2) for v in p.effect_delta],
                "visual_dispersion": round(float(p.visual_dispersion), 4),
            }
            for p in bank.prototypes.values()
        ],
        "causal_edges": {
            dst: [
                {"src": e.src_id, "kind": e.kind, "dim": e.dim, "lift": round(float(e.lift), 3),
                 "support": int(e.support), "mean_dt": round(float(e.mean_dt), 2)}
                for e in edgelist
            ]
            for dst, edgelist in getattr(bank, "edges", {}).items()
        },
    }

    diag_p = Path(diagnostic_path)
    diag_p.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_p, "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"-> [Saved] Consolidation Diagnostic: {diag_p}")

    return diag_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Store Offline Memory Consolidation Pipeline.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_crafter_v3_state_readout_v2.pt")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default="checkpoints/dual_store_causal_memory_bank.json")
    parser.add_argument("--output_pt", type=str, default="checkpoints/dual_store_memory_bank.pt")
    parser.add_argument("--diagnostic_path", type=str, default="results/diagnostics/dual_store_memory_consolidation.json")
    parser.add_argument("--max_episodes", type=int, default=300)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drive_dir", type=str, default=None)
    args = parser.parse_args()

    consolidate_dual_store_memory(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest_path,
        data_dir=args.data_dir,
        output_json_path=args.output_json,
        output_pt_path=args.output_pt,
        diagnostic_path=args.diagnostic_path,
        max_episodes=args.max_episodes,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        drive_dir=args.drive_dir,
    )
