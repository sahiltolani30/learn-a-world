"""
Phase 1 Diagnostic Runner:
1. Collects a local smoke dataset (2000 structured_random + 2000 teacher_hybrid transitions).
2. Runs 500-step head training using checkpoints/jepa_crafter.pt as base.
3. Evaluates PR-AUC and emits results/v2_head_validation.json.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.collect_sequences import collect_episodes
from training.sequence_buffer import EpisodeBuffer
from training.trainer_v2 import train_heads

def run():
    print("=" * 70)
    print("      PHASE 1: SEQUENCE DATASET & HEAD DIAGNOSTIC SMOKE RUN")
    print("=" * 70)

    data_dir = repo_root / "data" / "sequences"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect structured_random transitions
    print("\n>>> [1/3] Collecting structured_random transitions (2,000 steps)...")
    buf_random = collect_episodes(
        policy_name="structured_random",
        n_transitions=2000,
        out_dir=str(data_dir),
        chunk_size=2000,
        seed=101,
    )

    # 2. Collect teacher_hybrid transitions (captures wood, crafting, survival events)
    print("\n>>> [2/3] Collecting teacher_hybrid transitions (2,000 steps)...")
    buf_teacher = collect_episodes(
        policy_name="teacher_hybrid",
        n_transitions=2000,
        out_dir=str(data_dir),
        chunk_size=2000,
        seed=202,
    )

    # Combine episodes into a unified buffer
    combined_buf = EpisodeBuffer(capacity_transitions=10000)
    for ep in buf_random.episodes:
        combined_buf.add_episode(**ep)
    for ep in buf_teacher.episodes:
        combined_buf.add_episode(**ep)

    stats = combined_buf.get_dataset_statistics()
    print(f"\nCombined dataset: {stats['total_transitions']} transitions across {stats['total_episodes']} episodes.")
    print("Achievement event counts in smoke dataset:")
    for ach, cnt in stats["per_achievement_positive_counts"].items():
        if cnt > 0:
            print(f"   {ach:22s}: {cnt:4d} positive transitions")

    # 3. Train heads for 500 steps
    print("\n>>> [3/3] Training EventHead + HealthHead (500 steps, frozen JEPA base)...")
    model, val_report = train_heads(
        config_path="experiments/configs/crafter_v2.yaml",
        checkpoint_path="checkpoints/jepa_crafter.pt",
        buffer=combined_buf,
        total_steps=500,
        batch_size=32,
        seq_len=16,
        device_override=None,  # auto-detect (mps / cpu)
        out_json_path="results/v2_head_validation.json",
        out_checkpoint_path="checkpoints/jepa_crafter_v2.pt",
    )

    print("\n" + "=" * 70)
    print("                  DIAGNOSTIC VALIDATION SCORECARD")
    print("=" * 70)
    print(f"  Macro Tier-1 PR-AUC:       {val_report['macro_tier1_pr_auc']:.4f}")
    print(f"  Health MAE:                {val_report['health_mae']:.4f}")
    print(f"  Representation Problem:    {val_report['representation_problem']}")
    print("=" * 70)
    print("\nPer-Achievement PR-AUC Breakdown:")
    for ach, m in val_report["per_achievement"].items():
        if m["positive_samples"] > 0 or "wood" in ach or "drink" in ach or "sapling" in ach:
            print(f"   {ach:22s} | PR-AUC: {m['pr_auc']:.4f} | Recall@0.5: {m['recall_at_0.5']:.4f} | Positives: {m['positive_samples']}")
    print("=" * 70)

if __name__ == "__main__":
    run()
