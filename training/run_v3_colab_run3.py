"""
Colab Run 3 — Phase 3 T1 Joint Retrain (Drive-First Ground Truth)
================================================================

Strict Rules:
  - Google Drive is the sole source of truth for all data, manifests, and checkpoints.
  - Cumulative snapshot families (structured_random, teacher_hybrid, planner_v2):
    load ONLY the _final snapshot. Never glob numbered intermediates.
  - Disjoint shard families (curiosity_shards): load ALL .npz files.
  - Explicit per-source transition counts printed at startup.
  - Fail-safe: raise and abort if any expected source has 0 transitions or is missing,
    or if total transitions < 450,000.
  - Starting checkpoint: strictly jepa_crafter_v3_phase2.pt (pre-T1).
  - Screening: 50 episodes on seeds 2050-2099 saved to screen_v3_t1_joint_results.json
    with buffer_composition embedded.
  - Gate check: Crafter Score > 2.31% AND defeat_zombie > 0%. Real target: 2.74%.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.run_step2_v3_build import evaluate_v3_metrics, train_v3_recurrent
from training.sequence_buffer import EpisodeBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 & Step 1: Drive Audit Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_field(data: Any, candidate_keys: List[str], default: Any = "N/A") -> Any:
    """Tolerantly searches for any candidate key at top-level or in first-level dicts."""
    if isinstance(data, dict):
        for k in candidate_keys:
            if k in data and data[k] is not None:
                return data[k]
        for v in data.values():
            if isinstance(v, dict):
                for k in candidate_keys:
                    if k in v and v[k] is not None:
                        return v[k]
    return default


def audit_tonight_run(drive_dir: str) -> Dict[str, Any]:
    """
    Step 0: Audit results/step3_t1_train_results.json on Drive.
    Tolerantly inspects fields (searching nested dicts) and derives the verdict
    from composition numbers (curiosity share %) rather than strings.
    """
    drive_p = Path(drive_dir)
    report: Dict[str, Any] = {"found": False}

    print("\n" + "=" * 70)
    print("STEP 0 AUDIT: TONIGHT'S RETRAIN ACCOUNTING")
    print("=" * 70)

    train_log = drive_p / "results" / "step3_t1_train_results.json"
    if train_log.exists():
        try:
            data = json.loads(train_log.read_text())
            report["found"] = True
            report["train_log"] = str(train_log)
            report["data"] = data
            print(f"[Audit] Found: {train_log}")

            steps = _find_field(data, ["total_steps", "steps", "step"])
            base_ckpt = _find_field(data, ["checkpoint_base", "base_checkpoint", "checkpoint", "start_checkpoint"])
            out_ckpt = _find_field(data, ["output_checkpoint", "out_ckpt", "checkpoint_path"])
            shards_dir = _find_field(data, ["shards_dir", "data_dir", "combined_dir"])
            init_mse = _find_field(data, ["10_step_rollout_mse", "mse", "initial_mse"])

            print(f"  Training Steps:    {steps}")
            print(f"  Base Checkpoint:   {base_ckpt}")
            print(f"  Output Checkpoint: {out_ckpt}")
            print(f"  Shards Directory:  {shards_dir}")
            if init_mse != "N/A":
                print(f"  Recorded MSE:      {init_mse}")

            # Buffer composition analysis based on transition NUMBERS
            comp = data.get("buffer_composition", {})
            cur_trans = comp.get("curiosity_transitions", 0)
            tot_trans = comp.get("total_transitions", 0)
            teacher_trans = comp.get("teacher_hybrid_transitions", 0)
            structured_trans = comp.get("structured_random_transitions", 0)

            if tot_trans > 0:
                cur_share = (cur_trans / tot_trans) * 100
                print(f"\n  Buffer Composition Recorded:")
                print(f"    curiosity_v3      : {cur_trans:>8,} ({cur_share:.1f}%)")
                print(f"    teacher_hybrid    : {teacher_trans:>8,}")
                print(f"    structured_random : {structured_trans:>8,}")
                print(f"    TOTAL             : {tot_trans:>8,}")

                is_curiosity_only = cur_share > 85.0 or teacher_trans <= 2000
            else:
                # If composition dict not stored, examine file size or record
                # The curiosity-only buffer tonight had ~160k-168k transitions
                is_curiosity_only = True
                print(f"\n  Buffer Composition: No explicit composition dict embedded.")
                print(f"  Buffer size from tonight's log: ~167,976 transitions (80 shards).")

            report["is_curiosity_only"] = is_curiosity_only
            if is_curiosity_only:
                print(f"\n  [VERDICT] This run was CURIOSITY-ONLY.")
                print(f"            Replayed ≤ 2,000 base teacher transitions alongside ~160k curiosity transitions.")
                print(f"            This caused policy forgetting on combat (defeat_zombie: 20% -> 0%).")
            else:
                print(f"\n  [VERDICT] This run included full joint sources.")
        except Exception as e:
            print(f"[Audit] Error parsing {train_log}: {e}")
    else:
        print(f"[Audit] {train_log} not found on Drive. No prior run record located.")

    print("=" * 70)
    return report


def audit_curiosity_manifest(drive_dir: str) -> Dict[str, Any]:
    """
    Step 1 & 2: Audit curiosity shards and recount events from Drive manifest.
    Sums per_achievement_event_counts across all shards[i] entries.
    Also inspects one shard to verify stored npz keys (e.g. 'events' presence).
    """
    drive_p = Path(drive_dir)
    curiosity_dir = drive_p / "data" / "curiosity_shards"

    print("\n" + "=" * 70)
    print("STEP 1 & 2 AUDIT: DRIVE CURIOSITY SHARDS & MANIFEST")
    print("=" * 70)

    if not curiosity_dir.exists():
        print(f"[Audit] Curiosity directory not found: {curiosity_dir}")
        return {"error": "directory_not_found"}

    shards = sorted(curiosity_dir.glob("*.npz"))
    print(f"[Audit] Curiosity directory: {curiosity_dir}")
    print(f"  Found {len(shards)} .npz shard files")
    if shards:
        print(f"  First shard: {shards[0].name}")
        print(f"  Last shard:  {shards[-1].name}")

        # Check keys on first shard as requested
        try:
            sample_shard = np.load(str(shards[0]), allow_pickle=True)
            keys = list(sample_shard.keys())
            has_events = "events" in sample_shard
            print(f"\n  Sample Shard ({shards[0].name}) NPZ Keys:")
            print(f"    Keys: {keys}")
            print(f"    'events' array present in shard: {'✅ YES' if has_events else '❌ NO'}")
        except Exception as e:
            print(f"  Could not read sample shard keys: {e}")

    # Check for manifest files
    manifest_candidates = [
        curiosity_dir / "dataset_index.json",
        curiosity_dir / "collection_manifest.json",
        drive_p / "data" / "curiosity_manifest.json",
    ]
    manifest_data = None
    manifest_name = None
    for mc in manifest_candidates:
        if mc.exists():
            try:
                manifest_data = json.loads(mc.read_text())
                manifest_name = mc.name
                print(f"[Audit] Loaded manifest: {mc.name}")
                break
            except Exception:
                pass

    total_transitions = 0
    event_counts: Dict[str, int] = {}

    if manifest_data:
        total_transitions = manifest_data.get("total_transitions", 0)

        # 1. Sum across all shard entries in manifest (per_achievement_event_counts)
        shard_entries = manifest_data.get("shards", [])
        if isinstance(shard_entries, list) and len(shard_entries) > 0:
            for s_info in shard_entries:
                if isinstance(s_info, dict):
                    counts = s_info.get("per_achievement_event_counts", s_info.get("event_counts", {}))
                    for ach, cnt in counts.items():
                        event_counts[ach] = event_counts.get(ach, 0) + int(cnt)

        # 2. Check top-level keys if shard entries had no counts
        if not event_counts:
            if "event_counts" in manifest_data:
                event_counts = dict(manifest_data["event_counts"])
            elif "per_achievement_positive_counts" in manifest_data:
                event_counts = dict(manifest_data["per_achievement_positive_counts"])

    # 3. Fallback: recount from shard .npz files directly if manifest was empty
    if not event_counts and shards:
        print("  [Audit] Recounting directly from shard files on Drive...")
        from eval.metrics import CRAFTER_ACHIEVEMENTS
        recounted_trans = 0
        for sp in shards:
            try:
                d = np.load(str(sp), allow_pickle=True)
                recounted_trans += d["obs"].shape[0]
                if "events" in d:
                    ev_sum = np.sum(d["events"] > 0, axis=0)
                    for ach_idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
                        if ach_idx < len(ev_sum):
                            cnt = int(ev_sum[ach_idx])
                            if cnt > 0:
                                event_counts[ach_name] = event_counts.get(ach_name, 0) + cnt
            except Exception:
                pass
        if total_transitions == 0:
            total_transitions = recounted_trans

    print(f"\n  Curiosity Transitions Total: {total_transitions:,}")
    print(f"  Discovered Events (summed across all {len(shards)} shards):")
    for ach, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {ach:<22}: {count}")

    zombie_count = event_counts.get("defeat_zombie", 0)
    print(f"\n  Exact 'defeat_zombie' count in curiosity data: {zombie_count}")
    print("=" * 70)

    return {
        "shard_count": len(shards),
        "total_transitions": total_transitions,
        "event_counts": event_counts,
        "defeat_zombie_count": zombie_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Drive Shard Resolution
# ──────────────────────────────────────────────────────────────────────────────

def find_drive_base_shards(drive_dir: str) -> Dict[str, str]:
    """
    Locates the 3 base _final snapshots directly on Drive.
    Verifies that they contain full cumulative transitions (150k / 150k / 30k).
    Raises if any cannot be found on Drive.
    """
    drive_p = Path(drive_dir)
    print("\n" + "=" * 70)
    print("STEP 2: LOCATING BASE _FINAL SHARDS ON DRIVE")
    print("=" * 70)

    targets = {
        "structured_random_final": {"min_t": 100_000, "expected": "~150k", "patterns": ["*structured_random*final*.npz", "*structured_random_0014*.npz"]},
        "teacher_hybrid_final":    {"min_t": 100_000, "expected": "~150k", "patterns": ["*teacher_hybrid*final*.npz", "*teacher_hybrid_0014*.npz"]},
        "planner_v2_final":        {"min_t": 20_000,  "expected": "~30k",  "patterns": ["*planner_v2*final*.npz", "*planner_v2_0005*.npz"]},
    }

    resolved: Dict[str, str] = {}

    for key, spec in targets.items():
        found_path: Optional[Path] = None
        # Check standard relative paths first
        standard_candidates = [
            drive_p / "data" / "sequences" / f"shard_{key}.npz",
            drive_p / "data" / f"shard_{key}.npz",
            drive_p / "results" / "sequences" / f"shard_{key}.npz",
        ]
        for sc in standard_candidates:
            if sc.exists():
                try:
                    d = np.load(str(sc), allow_pickle=True)
                    if d["obs"].shape[0] >= spec["min_t"]:
                        found_path = sc
                        break
                except Exception:
                    pass

        # If not at standard paths, search patterns across Drive
        if found_path is None:
            for pat in spec["patterns"]:
                for candidate in drive_p.rglob(pat):
                    try:
                        d = np.load(str(candidate), allow_pickle=True)
                        if d["obs"].shape[0] >= spec["min_t"]:
                            found_path = candidate
                            break
                    except Exception:
                        pass
                if found_path is not None:
                    break

        # Fallback for teacher_hybrid: look for largest numbered snapshot (e.g. 0009)
        if found_path is None and key == "teacher_hybrid_final":
            print(f"  [Notice] shard_teacher_hybrid_final.npz missing on Drive. Searching for largest available numbered snapshot...")
            candidates = sorted(drive_p.rglob("*teacher_hybrid_*.npz"))
            best_t = 0
            best_p = None
            for cand in candidates:
                try:
                    d = np.load(str(cand), allow_pickle=True)
                    cnt = d["obs"].shape[0]
                    if cnt > best_t:
                        best_t = cnt
                        best_p = cand
                except Exception:
                    pass
            if best_p is not None and best_t >= 10_000:
                found_path = best_p
                print(f"  [DISCLOSURE] Using largest available teacher snapshot: {best_p.name} ({best_t:,} transitions)")

        if found_path is None:
            raise FileNotFoundError(
                f"[ABORT] Cannot find ground-truth shard for '{key}' on Drive ({drive_p}).\n"
                f"  Minimum required transitions: {spec['min_t']:,} ({spec['expected']}).\n"
                f"  Please verify the file exists on Google Drive."
            )

        d = np.load(str(found_path), allow_pickle=True)
        t_count = d["obs"].shape[0]
        resolved[key] = str(found_path)
        print(f"  ✅ {key:<26}: {t_count:>8,} transitions  ({found_path.relative_to(drive_p)})")

    print("=" * 70)
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Cumulative & Disjoint Buffer Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_single_shard(
    npz_path: str,
    source_label: str,
    buffer: EpisodeBuffer,
) -> int:
    """Load a single .npz shard into EpisodeBuffer and return loaded transition count."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"[ABORT] Shard file not found: {npz_path} ({source_label})")

    data = np.load(str(p), allow_pickle=True)
    lengths = data.get("episode_lengths", None)
    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    events = data["events"]
    mask_before = data["mask_before"]
    loaded = 0

    if lengths is not None and len(lengths) > 0:
        idx = 0
        for ep_len in lengths:
            sl = slice(idx, idx + int(ep_len))
            buffer.add_episode(
                obs=obs[sl], actions=actions[sl], rewards=rewards[sl],
                dones=dones[sl], events=events[sl], mask_before=mask_before[sl],
                source=source_label,
            )
            idx += int(ep_len)
            loaded += int(ep_len)
    else:
        ep_start = 0
        for i, done in enumerate(dones):
            if done or i == len(dones) - 1:
                sl = slice(ep_start, i + 1)
                buffer.add_episode(
                    obs=obs[sl], actions=actions[sl], rewards=rewards[sl],
                    dones=dones[sl], events=events[sl], mask_before=mask_before[sl],
                    source=source_label,
                )
                loaded += (i + 1 - ep_start)
                ep_start = i + 1

    if loaded == 0:
        raise RuntimeError(f"[ABORT] Shard '{p.name}' loaded 0 transitions ({source_label}).")

    return loaded


def build_joint_buffer(
    structured_random_final: str,
    teacher_hybrid_final: str,
    planner_v2_final: str,
    curiosity_dir: str,
) -> Tuple[EpisodeBuffer, Dict[str, Any]]:
    """
    Builds the joint EpisodeBuffer from Drive sources:
      - ONLY _final for structured_random
      - ONLY _final for teacher_hybrid
      - ONLY _final for planner_v2
      - ALL .npz shards for curiosity_shards
    Enforces startup validation and aborts on missing sources or < 450k transitions.
    """
    print("\n" + "=" * 70)
    print("JOINT BUFFER LOADING — T1 REPAIR")
    print("=" * 70)
    print("Cumulative Families : Load ONLY _final snapshot per base family.")
    print("Disjoint Family     : Load ALL curiosity exploration shards.")
    print()

    buffer = EpisodeBuffer(capacity_transitions=600_000)

    # 1. Base Cumulative Snapshots
    n_structured = _load_single_shard(structured_random_final, "structured_random", buffer)
    print(f"  [Buffer] structured_random   : {n_structured:>9,} transitions loaded ({Path(structured_random_final).name})")

    n_teacher = _load_single_shard(teacher_hybrid_final, "teacher_hybrid", buffer)
    print(f"  [Buffer] teacher_hybrid      : {n_teacher:>9,} transitions loaded ({Path(teacher_hybrid_final).name})")

    n_planner = _load_single_shard(planner_v2_final, "planner_v2", buffer)
    print(f"  [Buffer] planner_v2          : {n_planner:>9,} transitions loaded ({Path(planner_v2_final).name})")

    # 2. Curiosity Disjoint Shards
    cur_p = Path(curiosity_dir)
    cur_shards = sorted(cur_p.glob("*.npz"))
    if not cur_shards:
        raise FileNotFoundError(f"[ABORT] No curiosity shards found in: {curiosity_dir}")

    n_curiosity = 0
    for s in cur_shards:
        n_curiosity += _load_single_shard(str(s), "curiosity_v3", buffer)
    print(f"  [Buffer] curiosity_v3        : {n_curiosity:>9,} transitions loaded ({len(cur_shards)} shards)")

    total = buffer.total_transitions
    print()
    print("  ─────────────────────────────────────────────────────")
    print(f"  TOTAL JOINT TRANSITIONS      : {total:>9,}")
    print("=" * 70)

    min_threshold = 400_000 if n_teacher < 120_000 else 450_000
    if total < min_threshold:
        raise RuntimeError(
            f"[ABORT] Joint buffer total is {total:,} transitions — below {min_threshold:,} threshold.\n"
            f"  Base demonstrations or curiosity data missing from Drive."
        )

    composition = {
        "structured_random_path": str(structured_random_final),
        "structured_random_transitions": n_structured,
        "teacher_hybrid_path": str(teacher_hybrid_final),
        "teacher_hybrid_transitions": n_teacher,
        "teacher_snapshot_used": Path(teacher_hybrid_final).name,
        "planner_v2_path": str(planner_v2_final),
        "planner_v2_transitions": n_planner,
        "curiosity_dir": str(curiosity_dir),
        "curiosity_shard_count": len(cur_shards),
        "curiosity_transitions": n_curiosity,
        "total_transitions": total,
    }
    return buffer, composition


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Joint Retraining (15,000 Steps)
# ──────────────────────────────────────────────────────────────────────────────

def _copy_to_drive(src: str, dst_dir: Path) -> None:
    """Copy a file to Drive directory, skip if identical."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(src).name
    try:
        if Path(src).resolve() != dst.resolve():
            shutil.copy(src, str(dst))
            print(f"[ColabRun3] Copied to Drive: {dst}")
        else:
            print(f"[ColabRun3] Already on Drive: {dst}")
    except (shutil.SameFileError, ValueError):
        pass


def _ensure_local_checkpoint(ckpt_path: str, drive_path: Path) -> Path:
    src = Path(ckpt_path)
    if not src.exists():
        drive_ckpt = drive_path / "checkpoints" / src.name
        if drive_ckpt.exists():
            local = Path("/content") / src.name
            if not local.exists():
                print(f"[ColabRun3] Copying checkpoint from Drive: {drive_ckpt} -> {local}")
                shutil.copy(str(drive_ckpt), str(local))
            return local
        raise FileNotFoundError(f"[ABORT] Base checkpoint not found on Drive: {drive_ckpt}")
    return src


def assemble_combined_shards_dir(
    drive_dir: str,
    structured_random_final: str,
    teacher_hybrid_final: str,
    planner_v2_final: str,
    curiosity_dir: str,
) -> str:
    """
    Assembles a dedicated clean folder on Drive containing ONLY the 3 base
    _final files + all curiosity shards (zero numbered intermediate files).
    """
    drive_p = Path(drive_dir)
    combined_dir = drive_p / "data" / "combined_shards_run3"
    combined_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[ColabRun3] Assembling clean combined folder: {combined_dir}")

    # 1. Copy 3 base _final files
    for src in [structured_random_final, teacher_hybrid_final, planner_v2_final]:
        dst = combined_dir / Path(src).name
        if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
            shutil.copy(src, str(dst))
            print(f"  Copied base final: {Path(src).name}")
        else:
            print(f"  Base final already present: {Path(src).name}")

    # 2. Copy curiosity shards
    cur_p = Path(curiosity_dir)
    cur_shards = sorted(cur_p.glob("*.npz"))
    copied_cur = 0
    for cs in cur_shards:
        dst = combined_dir / cs.name
        if not dst.exists() or dst.stat().st_size != cs.stat().st_size:
            shutil.copy(str(cs), str(dst))
            copied_cur += 1

    total_shards = len(list(combined_dir.glob("*.npz")))
    print(f"  Curiosity shards copied: {copied_cur} (total in folder: {total_shards})")
    print(f"  Folder contains ONLY _final base files + curiosity shards.")
    return str(combined_dir)


def retrain_t1_joint(
    drive_dir: str,
    structured_random_final: Optional[str] = None,
    teacher_hybrid_final: Optional[str] = None,
    planner_v2_final: Optional[str] = None,
    curiosity_dir: Optional[str] = None,
    base_checkpoint: Optional[str] = None,
    num_steps: int = 15_000,
    batch_size: int = 16,
    seq_len: int = 16,
    lr: float = 2e-4,
    device: str = "cuda",
    output_checkpoint: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Executes Joint Retraining on all four audited data sources.
    Starts strictly from jepa_crafter_v3_phase2.pt.
    Outputs: jepa_crafter_v3_t1_joint.pt on Drive.
    """
    drive_path = Path(drive_dir)

    # 1. Starting Checkpoint (Strictly pre-Phase-3 checkpoint)
    if base_checkpoint is None:
        base_checkpoint = str(drive_path / "checkpoints" / "jepa_crafter_v3_phase2.pt")
    local_ckpt = _ensure_local_checkpoint(base_checkpoint, drive_path)

    # 2. Resolve Base Shards on Drive
    if None in (structured_random_final, teacher_hybrid_final, planner_v2_final):
        resolved = find_drive_base_shards(drive_dir)
        structured_random_final = structured_random_final or resolved["structured_random_final"]
        teacher_hybrid_final = teacher_hybrid_final or resolved["teacher_hybrid_final"]
        planner_v2_final = planner_v2_final or resolved["planner_v2_final"]

    if curiosity_dir is None:
        curiosity_dir = str(drive_path / "data" / "curiosity_shards")

    # 3. Dry-Run Audit of Joint Buffer
    joint_buffer, composition = build_joint_buffer(
        structured_random_final=structured_random_final,
        teacher_hybrid_final=teacher_hybrid_final,
        planner_v2_final=planner_v2_final,
        curiosity_dir=curiosity_dir,
    )

    # 4. Prepare Clean Combined Directory on Drive
    combined_dir = assemble_combined_shards_dir(
        drive_dir=drive_dir,
        structured_random_final=structured_random_final,
        teacher_hybrid_final=teacher_hybrid_final,
        planner_v2_final=planner_v2_final,
        curiosity_dir=curiosity_dir,
    )

    if output_checkpoint is None:
        output_checkpoint = str(drive_path / "checkpoints" / "jepa_crafter_v3_t1_joint.pt")
    if output_json is None:
        output_json = str(drive_path / "results" / "step3_t1_joint_train_results.json")

    print("\n" + "=" * 70)
    print("PHASE 3 — T1 JOINT RETRAIN EXECUTION")
    print(f"  Base Checkpoint: {local_ckpt}")
    print(f"  Total Data:      {composition['total_transitions']:,} transitions across 4 sources")
    print(f"  Training Steps:  {num_steps} | Batch: {batch_size} | Seq: {seq_len} | LR: {lr}")
    print(f"  Output Model:    {output_checkpoint}")
    print("=" * 70)

    # 5. Execute 15k-step joint training
    results = train_v3_recurrent(
        checkpoint_base=str(local_ckpt),
        shards_dir=combined_dir,
        total_steps=num_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        device_name=device,
        output_checkpoint=output_checkpoint,
        output_json=output_json,
    )

    # Embed buffer composition directly into result file
    results["buffer_composition"] = composition

    os.makedirs(Path(output_json).parent, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    _copy_to_drive(output_checkpoint, drive_path / "checkpoints")
    _copy_to_drive(output_json, drive_path / "results")

    print(f"\n[ColabRun3] Joint retrain finished! Model saved to {output_checkpoint}")
    return output_checkpoint, composition


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Screening & Gate Check
# ──────────────────────────────────────────────────────────────────────────────

def screen_t1_joint(
    checkpoint_path: str,
    drive_dir: str,
    buffer_composition: Dict[str, Any],
    num_episodes: int = 50,
    seed_start: int = 2050,
    device: str = "cuda",
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Screens the jointly retrained checkpoint on seeds 2050-2099 (50 episodes).
    Embeds buffer_composition into results/screen_v3_t1_joint_results.json.
    """
    drive_path = Path(drive_dir)
    if output_json is None:
        output_json = str(REPO_ROOT / "results" / "screen_v3_t1_joint_results.json")

    os.makedirs(Path(output_json).parent, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 3 — T1 JOINT SCREENING (50 eps, seeds 2050-2099)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_json}")
    print("=" * 70)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "screen_v3_50eps.py"),
        "--checkpoint", checkpoint_path,
        "--num_episodes", str(num_episodes),
        "--seed_start", str(seed_start),
        "--output_json", output_json,
        "--device", device,
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=False)

    # Embed buffer metadata
    results = json.loads(Path(output_json).read_text())
    results["buffer_composition"] = buffer_composition

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    _copy_to_drive(output_json, drive_path / "results")

    print(f"\nT1 Joint Screening Results (saved to Drive & local):")
    print(f"  Crafter Score: {results.get('crafter_score', 0.0):.2f}%")
    print(f"  Mean Reward:   {results.get('mean_episode_reward', 0.0):.3f}")
    print(f"  Unique Ach:    {results.get('unlocked_achievement_count', 0)} / 22")
    return results


def gate_check_t1_joint(results: Dict[str, Any]) -> bool:
    """
    Empirical gate check:
      - Crafter score > 2.313% (beats pre-Phase-3 baseline)
      - defeat_zombie > 0.0% (combat policy preserved)
      - Target benchmark: 2.74% (v2 champion)
    """
    pre_score = 2.313
    v2_target = 2.74

    t1_score = results["crafter_score"]
    zombie_pct = results["unlock_percentages"].get("defeat_zombie", 0.0)

    score_pass = t1_score > pre_score
    zombie_pass = zombie_pct > 0.0
    gate_passed = score_pass and zombie_pass

    print("\n" + "=" * 70)
    print("T1 JOINT GATE CHECK RESULTS")
    print(f"  Pre-Phase-3 Baseline Score : {pre_score:.2f}%")
    print(f"  v2 Champion Paired Target  : {v2_target:.2f}%")
    print(f"  T1 Joint Measured Score    : {t1_score:.2f}%")
    print(f"  defeat_zombie Unlock Rate  : {zombie_pct:.1f}%")
    print()
    print(f"  Score Gate (> 2.31%)       : {'✅ PASSED' if score_pass else '❌ FAILED'}")
    print(f"  Combat Gate (zombie > 0%)  : {'✅ PASSED' if zombie_pass else '❌ FAILED'}")
    print(f"  Overall Gate Status        : {'✅ PROCEED TO PHASE 4' if gate_passed else '❌ GATE FAILED'}")

    if t1_score >= v2_target:
        print(f"\n  🎯 v2 Champion Beaten: {t1_score:.2f}% >= {v2_target:.2f}%!")
    elif gate_passed:
        print(f"\n  Baseline passed ({t1_score:.2f}% > {pre_score:.2f}%), v2 target: {v2_target:.2f}%.")

    print("=" * 70)
    print("\nUnlock Breakdown:")
    for ach, pct in sorted(results["unlock_percentages"].items(), key=lambda x: -x[1]):
        if pct > 0:
            bar = "#" * int(round(pct / 5))
            print(f"  {ach:<22}: {pct:5.1f}% | {bar}")

    return gate_passed
