#!/usr/bin/env python3
"""
scripts/run_t2_verification_battery.py — Automated Verification for Item 3 & Item 4
====================================================================================

Executes both verification protocols sequentially:
1. Item 3: Champion Re-Run on jepa_crafter_v3_t1_joint.pt (seeds 2050–2099, 50 episodes)
2. Item 4: Honest Value Check on jepa_crafter_v3_t2_final.pt (seeds 4000–4047, 48 episodes)
"""

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_drive_mounted(drive_dir: Path) -> bool:
    """Attempts to mount Google Drive if running in Google Colab and drive path missing."""
    if drive_dir.exists():
        return True
    try:
        from google.colab import drive  # type: ignore
        print("[Drive] Google Colab detected. Attempting to mount Google Drive...")
        drive.mount("/content/drive")
        return drive_dir.exists()
    except Exception as e:
        print(f"[Drive] Note: Could not auto-mount Drive ({e}).")
        return False


def find_checkpoint(name: str, candidate_dirs: list) -> Path:
    """Searches candidate directories for checkpoint by name."""
    for d in candidate_dirs:
        p = Path(d)
        if p.exists():
            target = p / name
            if target.exists():
                return target
            matches = list(p.rglob(f"*{name}*"))
            if matches:
                return matches[0]

    # Search broader locations
    broad_roots = [Path("/content/drive"), Path("/content")]
    for root in broad_roots:
        if root.exists():
            try:
                for match in root.rglob(f"*{name}*"):
                    if match.is_file():
                        return match
            except Exception:
                pass

    # Diagnostic dump of any .pt files found
    found_pts = []
    for root in broad_roots:
        if root.exists():
            try:
                found_pts.extend([str(p) for p in root.rglob("*.pt")][:15])
            except Exception:
                pass

    err_msg = (
        f"[ABORT] Cannot locate checkpoint '{name}' in candidates: {candidate_dirs}\n"
        f"Available .pt files detected in /content or Drive:\n"
        + ("\n".join(f"  - {f}" for f in found_pts) if found_pts else "  (None found! Is Google Drive mounted?)")
    )
    raise FileNotFoundError(err_msg)


def main():
    parser = argparse.ArgumentParser(description="Run T2 Item 3 & 4 verification battery")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-item3", action="store_true", help="Skip Item 3 champion re-run")
    parser.add_argument("--skip-item4", action="store_true", help="Skip Item 4 value check")
    args = parser.parse_args()

    drive_p = Path(args.drive_dir)
    ensure_drive_mounted(drive_p)

    device = args.device
    cuda_avail = torch.cuda.is_available()

    print("=" * 80)
    print("           TRANCHE T2 FORENSIC VERIFICATION BATTERY (ITEMS 3 & 4)")
    print(f"  Google Drive Path: {drive_p} (exists: {drive_p.exists()})")
    print(f"  Requested Device : {device} (CUDA available: {cuda_avail})")
    if not cuda_avail and device == "cuda":
        print("  ⚠️ WARNING: CUDA not detected! Colab runtime is set to CPU.")
        print("  ⚠️ To run fast on GPU: In Colab menu -> Runtime -> Change runtime type -> T4 GPU.")
        device = "cpu"
    print("=" * 80)

    ckpt_search_dirs = [
        drive_p / "checkpoints",
        Path("/content/drive/My Drive/learn-a-world/checkpoints"),
        REPO_ROOT / "checkpoints",
        Path("/content") / "checkpoints",
        Path("/content"),
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # ITEM 3: CHAMPION RE-RUN (Seeds 2050–2099)
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_item3:
        print("\n" + "=" * 80)
        print("ITEM 3: CHAMPION RE-RUN (jepa_crafter_v3_t1_joint.pt | SEEDS 2050–2099)")
        print("=" * 80)

        t1_ckpt = find_checkpoint("jepa_crafter_v3_t1_joint.pt", ckpt_search_dirs)
        print(f"[Item 3] Using T1 Champion checkpoint: {t1_ckpt} ({t1_ckpt.stat().st_size / 1e6:.1f} MB)")

        out_json_item3 = REPO_ROOT / "results" / "screen_v3_t1_joint_recheck_results.json"
        cmd3 = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "screen_v3_50eps.py"),
            "--checkpoint", str(t1_ckpt),
            "--num_episodes", "50",
            "--seed_start", "2050",
            "--output_json", str(out_json_item3),
            "--device", device,
        ]
        subprocess.run(cmd3, check=True)

        if drive_p.exists():
            drive_out3 = drive_p / "results" / "screen_v3_t1_joint_recheck_results.json"
            drive_out3.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_json_item3), str(drive_out3))
            print(f"[Item 3] Mirrored results to Drive: {drive_out3}")

        # Print Item 3 results summary
        if out_json_item3.exists():
            with open(out_json_item3) as f:
                d3 = json.load(f)
            print("\n" + "-" * 40)
            print(">>> ITEM 3 RECHECK SCREENING SUMMARY <<<")
            print(f"Crafter Score: {d3.get('crafter_score', 'N/A')}%")
            print(f"Mean Reward:   {d3.get('mean_episode_reward', 'N/A')}")
            print(f"Achievements:  {d3.get('unlocked_achievement_count', 'N/A')} / 22")
            print("-" * 40)

    # ──────────────────────────────────────────────────────────────────────────
    # ITEM 4: HONEST VALUE CHECK (Seeds 4000–4047)
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_item4:
        print("\n" + "=" * 80)
        print("ITEM 4: HONEST VALUE CHECK (jepa_crafter_v3_t2_final.pt | SEEDS 4000–4047)")
        print("=" * 80)

        t2_ckpt = find_checkpoint("jepa_crafter_v3_t2_final.pt", ckpt_search_dirs)
        print(f"[Item 4] Using T2 Final checkpoint: {t2_ckpt} ({t2_ckpt.stat().st_size / 1e6:.1f} MB)")

        calibrator_p = REPO_ROOT / "results" / "calibration" / "event_head_calibrator.json"
        teacher_diag_p = REPO_ROOT / "results" / "diagnostics" / "t1_joint_q_calibration_self.json"
        out_json_item4 = REPO_ROOT / "results" / "diagnostics" / "t2_final_q_calibration_self.json"
        out_png_item4 = REPO_ROOT / "results" / "diagnostics" / "t2_final_q_calibration_self.png"

        cmd4 = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "diagnose_q_calibration_self.py"),
            "--checkpoint", str(t2_ckpt),
            "--calibrator", str(calibrator_p),
            "--teacher_json", str(teacher_diag_p),
            "--num_episodes", "48",
            "--seed_start", "4000",
            "--output_json", str(out_json_item4),
            "--output_png", str(out_png_item4),
            "--drive_dir", str(drive_p) if drive_p.exists() else "",
            "--device", device,
        ]
        subprocess.run(cmd4, check=True)

        # Print Item 4 results summary
        if out_json_item4.exists():
            with open(out_json_item4) as f:
                d4 = json.load(f)
            print("\n" + "-" * 40)
            print(">>> ITEM 4 HONEST VALUE CHECK SUMMARY <<<")
            print(f"Self-Slice Pearson r: {d4.get('pearson_r', 'N/A')}")
            print(f"Baseline r (T1):      0.3598")
            print(f"Target Gate (>0.30):  {'PASS' if d4.get('pearson_r', 0) > 0.30 else 'FAIL'}")
            print(f"Beat Baseline (>T1):  {'YES' if d4.get('pearson_r', 0) > 0.3598 else 'NO'}")
            print("-" * 40)

    print("\n" + "=" * 80)
    print("VERIFICATION BATTERY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
