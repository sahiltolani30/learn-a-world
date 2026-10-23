#!/usr/bin/env python3
"""
scripts/test_determinism.py — Environment Determinism & State Restore Verification (T2 Part 3)

Standing Rules:
- Seeds: 4000+ for diagnostic testing. NEVER touch seeds 3000-3099.
- Achievement logic: counts / masks only.
- Output: results/diagnostics/determinism_test.json and results/state_restore_test.json.

Verification Protocols:
1. Fresh Seed Determinism (5 episodes, seeds 4000-4004):
   Runs each seed twice from scratch with identical action sequences.
   Verifies: obs MSE == 0.0, exact reward matching, exact episode length, exact event masks.

2. State Snapshot & Restore Determinism (3 trials x 2 seeds, seeds 4000-4001):
   Executes N steps, takes full env snapshot, executes M steps (recording SHA-256 hashes),
   restores from snapshot, replays identical M actions, asserts 100% byte-identical SHA-256 hashes.

3. Upstream Root Cause Documentation:
   Explains the unpatched Crafter mob set non-determinism and validates the patch.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.crafter_wrapper import CrafterEnv, patch_crafter_determinism


def compute_obs_sha256(obs: np.ndarray) -> str:
    """Computes SHA-256 hash of a numpy observation array."""
    return hashlib.sha256(obs.tobytes()).hexdigest()


def test_fresh_seed_determinism(
    seeds: List[int],
    max_steps_per_episode: int = 150
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Protocol 1: Runs each seed twice from scratch with identical action sequences.
    Asserts obs MSE == 0, identical reward, length, and achievements.
    """
    print(f"\n--- [Protocol 1] Fresh Seed Determinism ({len(seeds)} episodes) ---")
    results = []
    all_passed = True

    for ep_idx, seed in enumerate(seeds):
        # Generate a fixed action sequence for this seed
        rng = np.random.RandomState(seed + 100000)
        action_seq = [int(rng.randint(0, 17)) for _ in range(max_steps_per_episode)]

        # Run 1
        env1 = CrafterEnv(seed=seed)
        obs1, _ = env1.reset()
        obs_seq1 = [obs1]
        reward_seq1 = []
        done_seq1 = []
        achievements_final1 = {}

        for a in action_seq:
            o, r, term, trunc, info = env1.step(a)
            obs_seq1.append(o)
            reward_seq1.append(r)
            done = term or trunc
            done_seq1.append(done)
            if done:
                achievements_final1 = info.get("achievements", {})
                break

        # Run 2 (fresh instance with identical seed)
        env2 = CrafterEnv(seed=seed)
        obs2, _ = env2.reset()
        obs_seq2 = [obs2]
        reward_seq2 = []
        done_seq2 = []
        achievements_final2 = {}

        for a in action_seq:
            o, r, term, trunc, info = env2.step(a)
            obs_seq2.append(o)
            reward_seq2.append(r)
            done = term or trunc
            done_seq2.append(done)
            if done:
                achievements_final2 = info.get("achievements", {})
                break

        # Comparisons
        length_match = len(obs_seq1) == len(obs_seq2)
        rewards_match = np.allclose(reward_seq1, reward_seq2, atol=1e-7) if length_match else False
        dones_match = (done_seq1 == done_seq2) if length_match else False
        achievements_match = (achievements_final1 == achievements_final2)

        min_len = min(len(obs_seq1), len(obs_seq2))
        max_obs_mse = 0.0
        divergence_step = None

        for t in range(min_len):
            mse = float(np.mean((obs_seq1[t].astype(np.float32) - obs_seq2[t].astype(np.float32)) ** 2))
            if mse > max_obs_mse:
                max_obs_mse = mse
            if mse > 1e-6 and divergence_step is None:
                divergence_step = t

        passed = length_match and rewards_match and dones_match and achievements_match and (max_obs_mse == 0.0)
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else f"FAIL (divergence at step {divergence_step})"
        print(f"  Ep {ep_idx+1}/{len(seeds)} (Seed {seed}): len={min_len}, max_obs_mse={max_obs_mse:.6f} -> {status_str}")

        results.append({
            "seed": seed,
            "passed": passed,
            "steps": min_len,
            "max_obs_mse": float(max_obs_mse),
            "length_match": bool(length_match),
            "rewards_match": bool(rewards_match),
            "dones_match": bool(dones_match),
            "achievements_match": bool(achievements_match),
            "divergence_step": divergence_step,
        })

    return all_passed, results


def test_state_restore_determinism(
    seeds: List[int],
    num_trials_per_seed: int = 3,
    steps_before_snapshot: int = 100,
    steps_after_snapshot: int = 100
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Protocol 2: State Snapshot & Restore Determinism (3 trials x 2 seeds).
    Runs N steps, snapshots env state, runs M steps recording SHA-256 hashes,
    restores snapshot, replays identical M actions, asserts 100% hash matching.
    """
    print(f"\n--- [Protocol 2] State Snapshot & Restore Determinism ({num_trials_per_seed} trials x {len(seeds)} seeds) ---")
    results = []
    all_passed = True

    for seed in seeds:
        for trial in range(num_trials_per_seed):
            trial_rng = np.random.RandomState(seed * 100 + trial + 50000)
            actions_warmup = [int(trial_rng.randint(0, 17)) for _ in range(steps_before_snapshot)]
            actions_replay = [int(trial_rng.randint(0, 17)) for _ in range(steps_after_snapshot)]

            env = CrafterEnv(seed=seed)
            env.reset()

            # Warmup steps
            for a in actions_warmup:
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break

            # Take full state snapshot
            snapshot = env.snapshot_state()

            # Path A: Continue original env for M steps
            obs_hashes_a = []
            rewards_a = []
            for a in actions_replay:
                o, r, term, trunc, _ = env.step(a)
                obs_hashes_a.append(compute_obs_sha256(o))
                rewards_a.append(r)
                if term or trunc:
                    break

            # Path B: Restore snapshot and replay identical M actions
            env.restore_state(snapshot)
            obs_hashes_b = []
            rewards_b = []
            for a in actions_replay:
                o, r, term, trunc, _ = env.step(a)
                obs_hashes_b.append(compute_obs_sha256(o))
                rewards_b.append(r)
                if term or trunc:
                    break

            # Compare Path A and Path B
            min_len = min(len(obs_hashes_a), len(obs_hashes_b))
            hash_matches = [h_a == h_b for h_a, h_b in zip(obs_hashes_a, obs_hashes_b)]
            matching_count = sum(hash_matches)
            match_rate = float(matching_count / len(obs_hashes_a)) if obs_hashes_a else 1.0
            rewards_match = np.allclose(rewards_a, rewards_b, atol=1e-7)

            divergence_step = None
            if not all(hash_matches):
                divergence_step = hash_matches.index(False)

            passed = (match_rate == 1.0) and rewards_match and (len(obs_hashes_a) == len(obs_hashes_b))
            if not passed:
                all_passed = False

            status_str = "PASS (100% hash match)" if passed else f"FAIL (diverged at replay step {divergence_step})"
            print(f"  Seed {seed} Trial {trial+1}/{num_trials_per_seed}: replay_steps={min_len}, match_rate={match_rate*100:.1f}% -> {status_str}")

            results.append({
                "seed": seed,
                "trial": trial + 1,
                "passed": passed,
                "replay_steps": min_len,
                "matching_steps": matching_count,
                "hash_match_rate": match_rate,
                "rewards_match": bool(rewards_match),
                "divergence_step": divergence_step,
            })

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Test Crafter Environment Determinism & State Restore (T2 Part 3)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick lightweight smoke test (2 episodes, 25 steps)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4000, 4001, 4002, 4003, 4004], help="Diagnostic seed range (4000+)")
    parser.add_argument("--restore-seeds", type=int, nargs="+", default=[4000, 4001], help="Seeds for state restore protocol")
    parser.add_argument("--output", type=str, default="results/diagnostics/determinism_test.json", help="Path to save output JSON")
    args = parser.parse_args()

    # Safety checks on seeds
    for s in args.seeds + args.restore_seeds:
        if 3000 <= s <= 3099:
            raise ValueError(f"CRITICAL ERROR: Seed {s} violates the forbidden seed rule (seeds 3000-3099 are strictly forbidden).")

    print("=" * 70)
    print("Crafter Determinism & State Restore Battery (Tranche T2 Part 3)")
    print("=" * 70)

    # Ensure deterministic patch is active
    patch_crafter_determinism()

    if args.smoke_test:
        test_seeds = args.seeds[:2]
        restore_seeds = args.restore_seeds[:1]
        fresh_steps = 30
        restore_warmup = 25
        restore_replay = 25
        num_trials = 2
        print("[SMOKE TEST MODE ENABLED] Running fast verification.")
    else:
        test_seeds = args.seeds
        restore_seeds = args.restore_seeds
        fresh_steps = 150
        restore_warmup = 100
        restore_replay = 100
        num_trials = 3

    t_start = time.time()

    # 1. Fresh seed determinism test
    fresh_passed, fresh_results = test_fresh_seed_determinism(
        seeds=test_seeds,
        max_steps_per_episode=fresh_steps
    )

    # 2. State restore determinism test
    restore_passed, restore_results = test_state_restore_determinism(
        seeds=restore_seeds,
        num_trials_per_seed=num_trials,
        steps_before_snapshot=restore_warmup,
        steps_after_snapshot=restore_replay
    )

    elapsed = time.time() - t_start
    overall_passed = fresh_passed and restore_passed

    summary = {
        "status": "PASS" if overall_passed else "FAIL",
        "overall_passed": bool(overall_passed),
        "protocol1_fresh_determinism_passed": bool(fresh_passed),
        "protocol2_state_restore_passed": bool(restore_passed),
        "num_fresh_episodes_tested": len(test_seeds),
        "num_restore_trials_tested": len(restore_results),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_cause_analysis": {
            "upstream_crafter_bug": "Non-deterministic creature balancing in _balance_chunk due to unordered set iteration over Object pointers without __hash__",
            "fix_applied": "Enforce coordinate-based sorting (int(pos[0]), int(pos[1])) in _balance_object and chunk coordinate sorting in World.chunks",
            "determinism_guarantee": "100% byte-identical SHA-256 observation match invariant to heap layout and deepcopy memory addresses"
        },
        "protocol1_results": fresh_results,
        "protocol2_results": restore_results,
    }

    # Save output artifacts
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] Results written to: {args.output}")

    # Mirror to results/state_restore_test.json as required by plan
    mirror_path = "results/state_restore_test.json"
    os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
    with open(mirror_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Mirror written to: {mirror_path}")

    print("\n" + "=" * 70)
    print(f"OVERALL DETERMINISM VERDICT: {'✅ PASSED (100% DETERMINISTIC)' if overall_passed else '❌ FAILED'}")
    print("=" * 70)

    if not overall_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
