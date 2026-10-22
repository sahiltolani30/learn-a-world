"""
Google Colab Campaign Runner for Learn-a-World v2 (Phases 1-3).
Executes on GPU (T4):
1. Collects 100k sequence transitions (50k structured_random + 50k teacher_hybrid).
2. Trains all task heads (EventHead, HealthHead, QEnsemble, PolicyPrior) for 5000 steps on GPU.
3. Runs 50-episode screening evaluation of pure v2 agent on seeds 1000-1049.
4. Automatically persists checkpoints and scorecards to Google Drive.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from eval.eval_v2 import evaluate_v2
from training.collect_sequences import collect_episodes
from training.sequence_buffer import EpisodeBuffer
from training.trainer_v2 import train_heads

def run_campaign(
    drive_dir: str = "/content/drive/MyDrive/learn-a-world",
    n_transitions_per_policy: int = 50_000,
    train_steps: int = 5_000,
    eval_episodes: int = 50,
):
    print("=" * 75)
    print("        LEARN-A-WORLD v2: FULL COLAB T4 TRAINING CAMPAIGN")
    print("=" * 75)

    drive_path = Path(drive_dir)
    ckpt_dir = drive_path / "checkpoints" if drive_path.exists() else repo_root / "checkpoints"
    data_dir = drive_path / "data" / "sequences" if drive_path.exists() else repo_root / "data" / "sequences"
    results_dir = drive_path / "results" if drive_path.exists() else repo_root / "results"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_v1_ckpt = ckpt_dir / "jepa_crafter.pt"
    if not base_v1_ckpt.exists():
        # Fallback to repo checkpoints
        base_v1_ckpt = repo_root / "checkpoints" / "jepa_crafter.pt"

    print(f"--> Base v1 Checkpoint: {base_v1_ckpt}")
    print(f"--> Data Directory:      {data_dir}")
    print(f"--> Checkpoint Dir:      {ckpt_dir}")

    # 1. Dataset Collection
    print(f"\n>>> [Stage 1/3] Collecting {n_transitions_per_policy * 2} Sequence Transitions...")
    print(f"    Collecting {n_transitions_per_policy} structured_random transitions...")
    buf_rand = collect_episodes(
        policy_name="structured_random",
        n_transitions=n_transitions_per_policy,
        out_dir=str(data_dir),
        chunk_size=5000,
        seed=100,
    )

    print(f"    Collecting {n_transitions_per_policy} teacher_hybrid reflex transitions...")
    buf_teach = collect_episodes(
        policy_name="teacher_hybrid",
        n_transitions=n_transitions_per_policy,
        out_dir=str(data_dir),
        chunk_size=5000,
        seed=200,
    )

    combined_buf = EpisodeBuffer(capacity_transitions=n_transitions_per_policy * 2 + 10000)
    for ep in buf_rand.episodes:
        combined_buf.add_episode(**ep)
    for ep in buf_teach.episodes:
        combined_buf.add_episode(**ep)

    stats = combined_buf.get_dataset_statistics()
    print(f"\nCombined dataset ready: {stats['total_transitions']} transitions across {stats['total_episodes']} episodes.")

    # 2. Train Task Heads
    v2_ckpt = ckpt_dir / "jepa_crafter_v2.pt"
    val_json = results_dir / "v2_head_validation.json"

    print(f"\n>>> [Stage 2/3] Training Task Heads for {train_steps} Steps (EventHead + HealthHead + QEnsemble + PolicyPrior)...")
    model, val_report = train_heads(
        config_path="experiments/configs/crafter_v2.yaml",
        checkpoint_path=str(base_v1_ckpt),
        buffer=combined_buf,
        total_steps=train_steps,
        batch_size=32,
        seq_len=16,
        device_override=None,  # auto detects CUDA / MPS / CPU
        out_json_path=str(val_json),
        out_checkpoint_path=str(v2_ckpt),
    )

    # 3. Screening Evaluation
    step2_json = results_dir / "v2_step2.json"
    print(f"\n>>> [Stage 3/3] Running {eval_episodes}-Episode Pure v2 Screening Evaluation (Seeds 1000..{1000 + eval_episodes - 1})...")
    eval_res = evaluate_v2(
        checkpoint_path=str(v2_ckpt),
        config_path="experiments/configs/crafter_v2.yaml",
        num_episodes=eval_episodes,
        seed_start=1000,
        max_steps=1000,
        device_override=None,
        out_path=str(step2_json),
    )

    print("\n" + "=" * 75)
    print("                     CAMPAIGN COMPLETION SUMMARY")
    print("=" * 75)
    print(f"  v2 Pure Crafter Score:    {eval_res['crafter_score']:.2f}% (v1 pure was 0.00%)")
    print(f"  Achievements Unlocked:    {eval_res['unlocked_count']}/22")
    print(f"  Mean Survival Steps:      {eval_res['mean_survival_steps']:.1f}")
    print(f"  Mean Reward / Episode:    {eval_res['mean_reward']:.2f}")
    print(f"  Checkpoint Saved:         {v2_ckpt}")
    print(f"  Validation Scorecard:     {val_json}")
    print(f"  Evaluation Results:       {step2_json}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full v2 Colab Campaign")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/learn-a-world")
    parser.add_argument("--n-transitions", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    run_campaign(
        drive_dir=args.drive_dir,
        n_transitions_per_policy=args.n_transitions,
        train_steps=args.steps,
        eval_episodes=args.eval_episodes,
    )
