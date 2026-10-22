"""
Google Colab Campaign Runner for Learn-a-World v2 (Phases 1-4).
Designed for T4/A100 GPU execution on Google Colab with automatic Drive persistence:
1. Stage 1: Collect ~300k starting sequence dataset:
   - 150,000 transitions: structured_random (broad exploration)
   - 150,000 transitions: teacher_hybrid (expert crafting play)
2. Stage 2: Train task heads (EventHead, HealthHead, QEnsemble, PolicyPrior) for 5,000 steps.
3. Stage 3: Phase 3 Joint Fine-tuning & 3-Round Collect-Retrain Loop (planner_v2 collection + joint unroll training).
4. Stage 4: Phase 4 Locked 100-episode Final Evaluation (seeds 1000..1099) & Promotion Decision.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.eval_v2 import evaluate_v2
from eval.run_final_comparison import run_evaluation_suite
from experiments.utils import get_device
from training.collect_sequences import collect_episodes
from training.sequence_buffer import EpisodeBuffer
from training.trainer_v2 import run_step3_rounds, train_heads

def run_campaign(
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
    n_transitions_per_policy: int = 150_000,
    train_head_steps: int = 5_000,
    rounds: int = 3,
    transitions_per_round: int = 30_000,
    joint_steps_per_round: int = 1_000,
    eval_episodes: int = 100,
    stage: str = "all",
    device_override: Optional[str] = None,
):
    print("=" * 80)
    print("        LEARN-A-WORLD v2: FULL COLAB T4/A100 GPU CAMPAIGN")
    print(f"        Starting Dataset: ~{n_transitions_per_policy * 2:,} steps ({n_transitions_per_policy:,} teacher + {n_transitions_per_policy:,} exploration)")
    print(f"        Stages: {stage.upper()} | Eval Episodes: {eval_episodes}")
    print("=" * 80)

    device = get_device(device_override)
    print(f"--> Using compute device: {device}")

    drive_path = Path(drive_dir)
    use_drive = drive_path.exists()
    if use_drive:
        print(f"--> Google Drive detected at: {drive_path} (Artifacts will be persisted directly to Drive)")
        ckpt_dir = drive_path / "checkpoints"
        data_dir = drive_path / "data" / "sequences"
        results_dir = drive_path / "results"
    else:
        print("--> Note: Google Drive path not found; using local workspace repository paths.")
        ckpt_dir = repo_root / "checkpoints"
        data_dir = repo_root / "data" / "sequences"
        results_dir = repo_root / "results"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_v1_ckpt = repo_root / "checkpoints" / "jepa_crafter.pt"
    if not base_v1_ckpt.exists() and (ckpt_dir / "jepa_crafter.pt").exists():
        base_v1_ckpt = ckpt_dir / "jepa_crafter.pt"

    v2_stage2_ckpt = ckpt_dir / "jepa_crafter_v2.pt"
    v2_best_ckpt = ckpt_dir / "jepa_crafter_v2_best.pt"

    print(f"--> Base v1 Checkpoint:  {base_v1_ckpt}")
    print(f"--> Data Directory:       {data_dir}")
    print(f"--> Checkpoint Directory: {ckpt_dir}")
    print(f"--> Results Directory:    {results_dir}")

    # =========================================================================
    # STAGE 1: Dataset Collection (~300k transitions total)
    # =========================================================================
    combined_buf = EpisodeBuffer(capacity_transitions=n_transitions_per_policy * 2 + 50_000)

    if stage in ("all", "collect", "heads"):
        print("\n" + "=" * 70)
        print(f">>> [Stage 1/4] Collecting Starting Dataset ({n_transitions_per_policy * 2:,} transitions)...")
        print("=" * 70)

        # 1A. Structured Random Play (Exploration: 150k)
        print(f"\n--> [1A] Collecting {n_transitions_per_policy:,} structured_random transitions...")
        buf_rand = collect_episodes(
            policy_name="structured_random",
            n_transitions=n_transitions_per_policy,
            out_dir=str(data_dir),
            chunk_size=10_000,
            seed=100,
        )

        # 1B. Teacher Hybrid Play (Expert Play: 150k)
        print(f"\n--> [1B] Collecting {n_transitions_per_policy:,} teacher_hybrid reflex transitions...")
        buf_teach = collect_episodes(
            policy_name="teacher_hybrid",
            n_transitions=n_transitions_per_policy,
            out_dir=str(data_dir),
            chunk_size=10_000,
            seed=200,
        )

        for ep in buf_rand.episodes:
            combined_buf.add_episode(**ep)
        for ep in buf_teach.episodes:
            combined_buf.add_episode(**ep)

        stats = combined_buf.get_dataset_statistics()
        print(f"\n--> Starting dataset assembled: {stats['total_transitions']:,} transitions across {stats['total_episodes']:,} episodes.")
        print(f"    Unlocked event frequencies: {stats['per_achievement_positive_counts']}")

    if stage == "collect":
        print("\n[Stage 1 Complete] Data collection finished successfully.")
        return

    # =========================================================================
    # STAGE 2: Train Task Heads (EventHead + HealthHead + QEnsemble + PolicyPrior)
    # =========================================================================
    if stage in ("all", "heads"):
        print("\n" + "=" * 70)
        print(f">>> [Stage 2/4] Training Task Heads for {train_head_steps:,} steps on GPU...")
        print("=" * 70)
        val_json = results_dir / "v2_head_validation.json"

        model, val_report = train_heads(
            config_path="experiments/configs/crafter_v2.yaml",
            checkpoint_path=str(base_v1_ckpt),
            buffer=combined_buf,
            total_steps=train_head_steps,
            batch_size=32,
            seq_len=16,
            device_override=str(device),
            out_json_path=str(val_json),
            out_checkpoint_path=str(v2_stage2_ckpt),
        )

        # Also copy checkpoint to best as baseline for phase 3
        model.save_checkpoint(str(v2_best_ckpt), step=train_head_steps)
        print(f"--> Stage 2 Checkpoint saved to: {v2_stage2_ckpt}")
        print(f"--> Baseline v2 Best Checkpoint initialized at: {v2_best_ckpt}")

        # Quick 20-episode screening to verify planner sanity
        print("\n--> Running 20-episode screening of initial v2 planner...")
        evaluate_v2(
            checkpoint_path=str(v2_stage2_ckpt),
            config_path="experiments/configs/crafter_v2.yaml",
            num_episodes=20,
            seed_start=1000,
            max_steps=1000,
            device_override=str(device),
            out_path=str(results_dir / "v2_step2.json"),
        )

    if stage == "heads":
        print("\n[Stage 2 Complete] Task heads trained successfully.")
        return

    # =========================================================================
    # STAGE 3: Phase 3 Joint Fine-Tuning & 3-Round Collect-Retrain Loop
    # =========================================================================
    if stage in ("all", "joint", "rounds"):
        print("\n" + "=" * 70)
        print(f">>> [Stage 3/4] Phase 3 Joint Fine-Tuning & Collect-Retrain Loop ({rounds} Rounds)...")
        print("=" * 70)

        step3_results = run_step3_rounds(
            base_checkpoint_path=str(v2_stage2_ckpt if v2_stage2_ckpt.exists() else base_v1_ckpt),
            config_path="experiments/configs/crafter_v2.yaml",
            rounds=rounds,
            transitions_per_round=transitions_per_round,
            joint_steps_per_round=joint_steps_per_round,
            eval_episodes=50,
            out_dir=str(results_dir),
            device_override=str(device),
        )
        print(f"\n--> Phase 3 finished! Best checkpoint: {step3_results['best_checkpoint']} with Crafter Score: {step3_results['best_score']:.2f}%")

    if stage in ("joint", "rounds"):
        print("\n[Stage 3 Complete] Joint fine-tuning and rounds finished.")
        return

    # =========================================================================
    # STAGE 4: Phase 4 Locked Final Comparison & Promotion Decision
    # =========================================================================
    if stage in ("all", "final", "eval"):
        print("\n" + "=" * 70)
        print(f">>> [Stage 4/4] Phase 4 Locked 100-Episode Final Comparison (Seeds 1000..{1000 + eval_episodes - 1})...")
        print("=" * 70)

        best_eval_ckpt = str(v2_best_ckpt if v2_best_ckpt.exists() else v2_stage2_ckpt)

        final_comparison = run_evaluation_suite(
            v1_checkpoint=str(base_v1_ckpt),
            v2_checkpoint=best_eval_ckpt,
            v1_config="experiments/configs/crafter.yaml",
            v2_config="experiments/configs/crafter_v2.yaml",
            episodes=eval_episodes,
            max_steps=1000,
            seed_start=1000,
            device_override=str(device),
            out_json=str(results_dir / "final_comparison.json"),
            out_md=str(results_dir / "promotion_decision.md"),
        )

        dec = final_comparison["promotion_decision"]
        print("\n" + "=" * 80)
        print(f"       CAMPAIGN FINISHED: {'PROMOTED TO V2 ✅' if dec['promoted'] else 'PROMOTION NOT MET ❌'}")
        print(f"       Results JSON:      {results_dir / 'final_comparison.json'}")
        print(f"       Decision Markdown: {results_dir / 'promotion_decision.md'}")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Full Learn-a-World v2 Colab Campaign")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world",
                        help="Google Drive mount directory")
    parser.add_argument("--n-transitions", type=int, default=150_000,
                        help="Transitions per collection policy (default: 150k -> 300k starting dataset)")
    parser.add_argument("--head-steps", type=int, default=5_000,
                        help="Stage 2 task head training steps")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Phase 3 collect-retrain rounds")
    parser.add_argument("--round-transitions", type=int, default=30_000,
                        help="Transitions collected per Phase 3 round")
    parser.add_argument("--joint-steps", type=int, default=1_000,
                        help="Joint training steps per Phase 3 round")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Phase 4 evaluation episodes (default: 100)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "collect", "heads", "joint", "rounds", "final", "eval"],
                        help="Campaign execution stage")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda/mps/cpu)")
    args = parser.parse_args()

    run_campaign(
        drive_dir=args.drive_dir,
        n_transitions_per_policy=args.n_transitions,
        train_head_steps=args.head_steps,
        rounds=args.rounds,
        transitions_per_round=args.round_transitions,
        joint_steps_per_round=args.joint_steps,
        eval_episodes=args.eval_episodes,
        stage=args.stage,
        device_override=args.device,
    )

if __name__ == "__main__":
    main()
