#!/usr/bin/env python3
"""
scripts/run_planner_sweep.py
===========================
Systematic Permutation Sweep Engine for RecurrentMPCPlanner configurations.
Runs 20 episodes on seeds 2050..2069 per configuration on jepa_crafter_v3_t2_final.pt.

Phase 1 (One Factor at a Time from Baseline):
  1. base: Standardize ON, H=10, C=64, CEM=2, denorm OFF, TTE=0.0
  2. std_off: --no-standardize-scores
  3. h15: --horizon 15
  4. h20: --horizon 20
  5. denorm: --denormalize_q
  6. denorm_std_off: --no-standardize-scores --denormalize_q
  7. tte05: --tte_weight 0.5
  8. c128: --num_candidates 128
  9. cem3: --cem_iterations 3

Phase 2 (Combinations):
  10. h15_tte05: --horizon 15 --tte_weight 0.5
  11. h15_tte05_denorm: --horizon 15 --tte_weight 0.5 --denormalize_q
  12. h20_tte05_c128: --horizon 20 --tte_weight 0.5 --num_candidates 128
  13. std_off_denorm_h15_tte05: --no-standardize-scores --denormalize_q --horizon 15 --tte_weight 0.5
  14. denorm_tte05: --denormalize_q --tte_weight 0.5

Outputs:
  - Individual JSON files in results/sweep/<name>_20ep.json
  - Consolidated comparison table in results/sweep/sweep_summary_table.md
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import zipfile

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEFAULT_DEVICE = "cuda"

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "results" / "sweep"

PHASE_1_CONFIGS = [
    {
        "name": "base",
        "description": "Baseline: Standardize ON, H=10, C=64, CEM=2, Denorm OFF, TTE=0.0",
        "cli_args": [],
        "output_json": "results/sweep/base_20ep.json",
    },
    {
        "name": "std_off",
        "description": "Standardize OFF (legacy weighted sum)",
        "cli_args": ["--no-standardize-scores"],
        "output_json": "results/sweep/std_off_20ep.json",
    },
    {
        "name": "h15",
        "description": "Horizon 15",
        "cli_args": ["--horizon", "15"],
        "output_json": "results/sweep/h15_20ep.json",
    },
    {
        "name": "h20",
        "description": "Horizon 20",
        "cli_args": ["--horizon", "20"],
        "output_json": "results/sweep/h20_20ep.json",
    },
    {
        "name": "denorm",
        "description": "Denormalize Q ON",
        "cli_args": ["--denormalize_q"],
        "output_json": "results/sweep/denorm_20ep.json",
    },
    {
        "name": "denorm_std_off",
        "description": "Standardize OFF + Denormalize Q ON",
        "cli_args": ["--no-standardize-scores", "--denormalize_q"],
        "output_json": "results/sweep/denorm_std_off_20ep.json",
    },
    {
        "name": "tte05",
        "description": "TTE weight 0.5",
        "cli_args": ["--tte_weight", "0.5"],
        "output_json": "results/sweep/tte05_20ep.json",
    },
    {
        "name": "c128",
        "description": "Candidates 128",
        "cli_args": ["--num_candidates", "128"],
        "output_json": "results/sweep/c128_20ep.json",
    },
    {
        "name": "cem3",
        "description": "CEM iterations 3",
        "cli_args": ["--cem_iterations", "3"],
        "output_json": "results/sweep/cem3_20ep.json",
    },
]

PHASE_2_CONFIGS = [
    {
        "name": "h15_tte05",
        "description": "Horizon 15 + TTE 0.5",
        "cli_args": ["--horizon", "15", "--tte_weight", "0.5"],
        "output_json": "results/sweep/h15_tte05_20ep.json",
    },
    {
        "name": "h15_tte05_denorm",
        "description": "Horizon 15 + TTE 0.5 + Denormalize Q",
        "cli_args": ["--horizon", "15", "--tte_weight", "0.5", "--denormalize_q"],
        "output_json": "results/sweep/h15_tte05_denorm_20ep.json",
    },
    {
        "name": "h20_tte05_c128",
        "description": "Horizon 20 + TTE 0.5 + Candidates 128",
        "cli_args": ["--horizon", "20", "--tte_weight", "0.5", "--num_candidates", "128"],
        "output_json": "results/sweep/h20_tte05_c128_20ep.json",
    },
    {
        "name": "std_off_denorm_h15_tte05",
        "description": "Standardize OFF + Denormalize Q + Horizon 15 + TTE 0.5",
        "cli_args": ["--no-standardize-scores", "--denormalize_q", "--horizon", "15", "--tte_weight", "0.5"],
        "output_json": "results/sweep/std_off_denorm_h15_tte05_20ep.json",
    },
    {
        "name": "denorm_tte05",
        "description": "Denormalize Q ON + TTE 0.5 (H=10, Standardize ON)",
        "cli_args": ["--denormalize_q", "--tte_weight", "0.5"],
        "output_json": "results/sweep/denorm_tte05_20ep.json",
    },
]

ALL_CONFIGS = PHASE_1_CONFIGS + PHASE_2_CONFIGS

KEY_ACHIEVEMENTS = [
    ("wood", "collect_wood"),
    ("sapling", "collect_sapling"),
    ("plant", "place_plant"),
    ("drink", "collect_drink"),
    ("table", "place_table"),
    ("pickaxe", "make_wood_pickaxe"),
    ("sword", "make_wood_sword"),
    ("stone", "collect_stone"),
    ("zombie", "defeat_zombie"),
    ("cow", "eat_cow"),
    ("wake_up", "wake_up"),
]


def is_valid_result(json_path: Path) -> bool:
    if not json_path.exists():
        return False
    try:
        with open(json_path) as f:
            d = json.load(f)
        return "crafter_score" in d and len(d.get("episodes", [])) >= 20
    except Exception:
        return False


def build_markdown_table(results: List[Dict[str, Any]]) -> str:
    headers = [
        "Config",
        "Score",
        "Reward",
        "Survival",
        "wood",
        "sapling",
        "plant",
        "drink",
        "table",
        "pickaxe",
        "sword",
        "stone",
        "zombie",
        "cow",
        "wake_up",
    ]
    align = [":---", ":---:", ":---:", ":---:"] + [":---:"] * 11
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for item in results:
        name = item["name"]
        score = f"{item['crafter_score']:.2f}%" if "crafter_score" in item else "N/A"
        rew = f"{item['mean_episode_reward']:.2f}" if "mean_episode_reward" in item else "N/A"
        surv = f"{item['mean_survival_length']:.1f}" if "mean_survival_length" in item else "N/A"

        row = [f"`{name}`", f"**{score}**", rew, surv]
        pcts = item.get("unlock_percentages", {})
        for short_name, full_name in KEY_ACHIEVEMENTS:
            pct = pcts.get(full_name, 0.0)
            if pct > 0:
                row.append(f"{pct:.0f}%")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def run_sweep(
    configs: List[Dict[str, Any]],
    checkpoint: str,
    device: str = "cuda",
    num_episodes: int = 20,
    seed_start: int = 2050,
    parallel: int = 1,
    force: bool = False,
) -> List[Dict[str, Any]]:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 80)
    print(f"PLANNER PERMUTATION SWEEP: {len(configs)} Configurations ({num_episodes} eps, Seeds {seed_start}..{seed_start + num_episodes - 1})")
    print(f"Checkpoint : {checkpoint}")
    print(f"Device     : {device}")
    print(f"Parallel   : {parallel} worker(s)")
    print("=" * 80 + "\n", flush=True)

    t_sweep_start = time.time()

    def _execute_single(cfg: Dict[str, Any], idx: int, total: int, pipe_to_log: bool) -> Optional[Dict[str, Any]]:
        name = cfg["name"]
        out_rel = cfg["output_json"]
        out_path = REPO_ROOT / out_rel
        log_path = SWEEP_DIR / f"{name}.log"

        if not force and is_valid_result(out_path):
            print(f"  [{idx + 1}/{total} SKIP] Result already exists and is valid: {name.upper()} ({out_path.name})", flush=True)
            with open(out_path) as f:
                data = json.load(f)
            data["name"] = name
            return data

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "screen_v3_50eps.py"),
            "--checkpoint", checkpoint,
            "--num_episodes", str(num_episodes),
            "--seed_start", str(seed_start),
            "--output_json", str(out_path),
            "--device", device,
        ] + cfg["cli_args"]

        t_cfg_start = time.time()
        if pipe_to_log:
            print(f"  [{idx + 1}/{total} LAUNCHED] {name.upper()} ({cfg['description']}) -> log: results/sweep/{name}.log", flush=True)
            with open(log_path, "w") as lf:
                res = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT)
        else:
            print(f"\n[{idx + 1}/{total}] Starting configuration: {name.upper()}")
            print(f"  Description: {cfg['description']}")
            print(f"  Running: {' '.join(cmd)}", flush=True)
            res = subprocess.run(cmd, cwd=str(REPO_ROOT))

        cfg_dur = time.time() - t_cfg_start

        if res.returncode != 0:
            print(f"  [ERROR] Configuration {name.upper()} failed with exit code {res.returncode}! (Check log: {log_path})", file=sys.stderr, flush=True)
            return None

        if is_valid_result(out_path):
            with open(out_path) as f:
                data = json.load(f)
            data["name"] = name
            print(f"  --> [COMPLETED {idx + 1}/{total}] {name.upper()} in {cfg_dur:.1f}s | Score: {data.get('crafter_score', 0):.2f}% | Reward: {data.get('mean_episode_reward', 0):.2f}", flush=True)
            return data
        else:
            print(f"  [WARNING] Output file {out_path} for {name.upper()} is missing or incomplete!", file=sys.stderr, flush=True)
            return None

    if parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"--> Spawning pool with {parallel} parallel workers...\n", flush=True)
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_cfg = {
                executor.submit(_execute_single, cfg, idx, len(configs), True): cfg
                for idx, cfg in enumerate(configs)
            }
            for future in as_completed(future_to_cfg):
                res_data = future.result()
                if res_data is not None:
                    results.append(res_data)
    else:
        for idx, cfg in enumerate(configs):
            res_data = _execute_single(cfg, idx, len(configs), False)
            if res_data is not None:
                results.append(res_data)

    total_sweep_dur = time.time() - t_sweep_start
    print(f"\n--> Sweep completed in {total_sweep_dur / 60:.1f} minutes.")
    return results


def export_and_display_summary(results: List[Dict[str, Any]]) -> str:
    # Sort results by Crafter score descending
    sorted_results = sorted(results, key=lambda x: x.get("crafter_score", 0.0), reverse=True)
    table_md = build_markdown_table(sorted_results)

    summary_file = SWEEP_DIR / "sweep_summary_table.md"
    with open(summary_file, "w") as f:
        f.write("# Planner Permutation Sweep Summary (Seeds 2050-2069)\n\n")
        f.write(table_md)
        f.write("\n")

    print("\n" + "=" * 80)
    print("                    PERMUTATION SWEEP LEADERBOARD")
    print("=" * 80)
    print(table_md)
    print("=" * 80)
    print(f"\n--> Summary table saved to: {summary_file}")

    # Create zip bundle for easy download from Colab
    zip_path = SWEEP_DIR / "sweep_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for json_file in SWEEP_DIR.glob("*_20ep.json"):
            zipf.write(json_file, arcname=json_file.name)
        if summary_file.exists():
            zipf.write(summary_file, arcname=summary_file.name)
    print(f"--> All sweep JSONs packaged to: {zip_path}\n")

    return table_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run systematic planner permutation sweep.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/learn-a-world/checkpoints/jepa_crafter_v3_t2_final.pt",
        help="Path to jepa_crafter_v3_t2_final.pt",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1",
        choices=["1", "2", "all"],
        help="Sweep phase to run: 1 (factors 1-9), 2 (combos 10-13), or all (1-13)",
    )
    parser.add_argument("--config", type=str, default=None, help="Run only a single named config")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=2050)
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel worker processes (e.g. 2 or 3)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if result JSON exists")
    parser.add_argument("--report-only", action="store_true", help="Only build and display summary from existing files")
    args = parser.parse_args()

    if args.report_only:
        existing_results = []
        for cfg in ALL_CONFIGS:
            p = REPO_ROOT / cfg["output_json"]
            if is_valid_result(p):
                with open(p) as f:
                    d = json.load(f)
                d["name"] = cfg["name"]
                existing_results.append(d)
        if existing_results:
            export_and_display_summary(existing_results)
        else:
            print("No valid sweep result JSONs found under results/sweep/.")
        return

    if args.config:
        matched = [c for c in ALL_CONFIGS if c["name"] == args.config]
        if not matched:
            print(f"Config '{args.config}' not found. Available: {[c['name'] for c in ALL_CONFIGS]}")
            sys.exit(1)
        target_configs = matched
    elif args.phase == "1":
        target_configs = PHASE_1_CONFIGS
    elif args.phase == "2":
        target_configs = PHASE_2_CONFIGS
    else:
        target_configs = ALL_CONFIGS

    results = run_sweep(
        configs=target_configs,
        checkpoint=args.checkpoint,
        device=args.device,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        parallel=args.parallel,
        force=args.force,
    )

    if results:
        export_and_display_summary(results)


if __name__ == "__main__":
    main()
