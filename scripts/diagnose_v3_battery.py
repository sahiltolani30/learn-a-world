#!/usr/bin/env python3
"""
Diagnostic Battery for JEPA v3 Joint Checkpoints.
Runs D1 (Q-Calibration), D3 (Head Audit), and D4 (Imagination) in sequence
and prints a side-by-side comparison against the Phase 2 baseline.
"""

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import subprocess


def run_battery(
    checkpoint: str,
    shards_dir: str = "data/sequences",
    device: str = "cuda",
    output_dir: str = "results/diagnostics",
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    d1_json = str(out_path / "t1_joint_q_calibration.json")
    d1_png = str(out_path / "t1_joint_q_calibration.png")
    d3_json = str(out_path / "t1_joint_head_audit.json")
    d3_png = str(out_path / "t1_joint_head_audit.png")
    d4_json = str(out_path / "t1_joint_imagination.json")
    d4_png = str(out_path / "t1_joint_imagination.png")

    print("\n" + "=" * 70)
    print("RUNNING DIAGNOSTIC BATTERY ON JOINT CHECKPOINT")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Held-out:   {shards_dir}")
    print(f"  Device:     {device}")
    print("=" * 70 + "\n")

    # 1. D1: Q-Calibration
    print(">>> [1/3] Running D1: Q-Value Calibration...")
    cmd_d1 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "diagnose_q_calibration.py"),
        "--checkpoint", checkpoint,
        "--shards_dir", shards_dir,
        "--device", device,
        "--output_json", d1_json,
        "--output_png", d1_png,
    ]
    subprocess.run(cmd_d1, check=True)

    # 2. D3: Head Audit
    print("\n>>> [2/3] Running D3: Event & Health Head Prediction Audit...")
    cmd_d3 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "diagnose_head_audit.py"),
        "--checkpoint", checkpoint,
        "--shards_dir", shards_dir,
        "--device", device,
        "--output_json", d3_json,
        "--output_png", d3_png,
    ]
    subprocess.run(cmd_d3, check=True)

    # 3. D4: Multi-Step Imagination
    print("\n>>> [3/3] Running D4: 10-Step Imagination Rollout...")
    cmd_d4 = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "diagnose_imagination.py"),
        "--checkpoint", checkpoint,
        "--shards_dir", shards_dir,
        "--device", device,
        "--output_json", d4_json,
        "--output_png", d4_png,
    ]
    subprocess.run(cmd_d4, check=True)

    # Load results
    d1_res = json.loads(Path(d1_json).read_text())
    d3_res = json.loads(Path(d3_json).read_text())
    d4_res = json.loads(Path(d4_json).read_text())

    # Load Phase 2 baseline for comparative table if available
    base_d1_file = REPO_ROOT / "results" / "diagnostics" / "v3_q_calibration.json"
    base_d3_file = REPO_ROOT / "results" / "diagnostics" / "v3_head_audit.json"
    base_d4_file = REPO_ROOT / "results" / "diagnostics" / "v3_imagination_calibration.json"

    base_d1 = json.loads(base_d1_file.read_text()) if base_d1_file.exists() else {}
    base_d3 = json.loads(base_d3_file.read_text()) if base_d3_file.exists() else {}
    base_d4 = json.loads(base_d4_file.read_text()) if base_d4_file.exists() else {}

    # Copy to Drive if available
    if drive_dir and Path(drive_dir).exists():
        drive_diag = Path(drive_dir) / "results" / "diagnostics"
        drive_diag.mkdir(parents=True, exist_ok=True)
        import shutil
        for src in [d1_json, d1_png, d3_json, d3_png, d4_json, d4_png]:
            if Path(src).exists():
                shutil.copy(src, str(drive_diag / Path(src).name))
        print(f"\n[Drive] Copied all diagnostic JSONs and plots to: {drive_diag}")

    # Comparative Summary
    r_val = d1_res.get("pearson_r", 0.0)
    r_base = base_d1.get("pearson_r", 0.2967)
    rho_val = d1_res.get("spearman_rho", 0.0)
    rho_base = base_d1.get("spearman_rho", 0.3810)
    q_std = d1_res.get("q_pred_std", 0.0)
    q_std_base = base_d1.get("q_pred_std", 0.4011)

    macro_pr = d3_res.get("macro_pr_auc_active_heads", 0.0)
    macro_base = base_d3.get("macro_pr_auc_active_heads", 0.4873)
    wood_pr = d3_res.get("per_achievement", {}).get("collect_wood", {}).get("pr_auc", 0.0)
    wood_base = base_d3.get("per_achievement", {}).get("collect_wood", {}).get("pr_auc", 0.3867)
    health_mae = d3_res.get("health_head_mae", 0.0)
    health_base = base_d3.get("health_head_mae", 0.0729)

    mse_10 = d4_res.get("step_10_mse", 0.0)
    mse_10_base = base_d4.get("step_10_mse", 0.7778)
    cos_10 = d4_res.get("step_10_cosine", 0.0)
    cos_10_base = base_d4.get("step_10_cosine", 0.8953)

    print("\n" + "=" * 78)
    print("DIAGNOSTIC BATTERY COMPARISON: Phase 2 Baseline vs T1 Joint Checkpoint")
    print("=" * 78)
    print(f"{'Metric':<36} | {'Phase 2 Baseline':<16} | {'T1 Joint':<16} | {'Status'}")
    print("-" * 78)

    # D1 Metrics
    r_pass = r_val > 0.30
    print(f"{'D1: Pearson r (Target > 0.30)':<36} | {r_base:<16.4f} | {r_val:<16.4f} | {'✅ HEALTHY' if r_pass else '⚠️ BELOW 0.30'}")
    print(f"{'D1: Spearman rho':<36} | {rho_base:<16.4f} | {rho_val:<16.4f} | {'✅' if rho_val >= rho_base else '—'}")
    print(f"{'D1: Predicted Q-Std (Uncollapsed)':<36} | {q_std_base:<16.4f} | {q_std:<16.4f} | {'✅' if q_std > 0.10 else '❌ COLLAPSED'}")

    # D3 Metrics
    print("-" * 78)
    print(f"{'D3: Macro PR-AUC (Active Heads)':<36} | {macro_base:<16.4f} | {macro_pr:<16.4f} | {'✅' if macro_pr >= macro_base else '—'}")
    print(f"{'D3: collect_wood PR-AUC':<36} | {wood_base:<16.4f} | {wood_pr:<16.4f} | {'✅' if wood_pr >= wood_base else '—'}")
    print(f"{'D3: HealthHead MAE (Lower is better)':<36} | {health_base:<16.4f} | {health_mae:<16.4f} | {'✅' if health_mae <= health_base else '—'}")

    # D4 Metrics
    print("-" * 78)
    print(f"{'D4: 10-Step Rollout MSE (< 0.85)':<36} | {mse_10_base:<16.4f} | {mse_10:<16.4f} | {'✅ STABLE' if mse_10 < 0.85 else '❌ DRIFT'}")
    print(f"{'D4: 10-Step Cosine Similarity':<36} | {cos_10_base:<16.4f} | {cos_10:<16.4f} | {'✅' if cos_10 >= cos_10_base else '—'}")
    print("=" * 78)

    # Synthesis Verdict
    print("\nDIAGNOSTIC VERDICT FOR T2 PLANNING:")
    if r_pass and macro_pr >= 0.40 and mse_10 < 0.85:
        print("  ✅ MODEL REPRESENTATION & CRITIC ARE HEALTHY.")
        print("     Value correlation > 0.30 and dynamics/event heads are calibrated.")
        print("     Conclusion: The model architecture & training objective are sound.")
        print("     For T2: Focus strictly on targeted exploration data collection!")
    else:
        print("  ⚠️ MODEL DIAGNOSTIC IDENTIFIED AREAS FOR TUNING:")
        if not r_pass:
            print(f"     - Value function correlation ({r_val:.4f}) is below 0.30 threshold.")
        if macro_pr < 0.40:
            print(f"     - Event calibration ({macro_pr:.4f}) shows low average precision.")
        if mse_10 >= 0.85:
            print(f"     - 10-step dynamics error ({mse_10:.4f}) exceeds 0.85 bound.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete diagnostic battery on a JEPA checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--shards_dir", type=str, default="data/sequences", help="Held-out shards directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output_dir", type=str, default="results/diagnostics", help="Output directory")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/learn-a-world", help="Drive root dir")
    args = parser.parse_args()

    run_battery(
        checkpoint=args.checkpoint,
        shards_dir=args.shards_dir,
        device=args.device,
        output_dir=args.output_dir,
        drive_dir=args.drive_dir,
    )
