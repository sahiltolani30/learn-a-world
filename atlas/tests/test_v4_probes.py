"""Verification tests for T2 Ground-Truth Probes (atlas/probes.py)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from atlas.probes import (
    fit_ridge_regression,
    compute_r2,
    compute_auroc,
    evaluate_probes_suite,
    export_probes,
)


def test_t2_synthetic_ridge_r2():
    """1. Synthetic: y = X @ w_true + 0.01·noise with n=500, D=32 -> R² > 0.95."""
    rng = np.random.RandomState(42)
    n = 500
    d = 32
    X = rng.randn(n, d)
    w_true = rng.randn(d)
    y = X @ w_true + 0.01 * rng.randn(n)

    # Split 80/20 train/val
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    mean_X = X_train.mean(axis=0)
    std_X = np.maximum(X_train.std(axis=0), 1e-6)
    X_tr_std = (X_train - mean_X) / std_X
    X_va_std = (X_val - mean_X) / std_X

    mean_y = y_train.mean()
    y_tr_c = y_train - mean_y
    W = fit_ridge_regression(X_tr_std, y_tr_c, lambda_reg=1e-3)
    y_pred = X_va_std @ W + mean_y

    r2 = compute_r2(y_val, y_pred)
    assert r2 is not None
    assert r2 > 0.95, f"Expected R² > 0.95, got {r2:.4f}"


def test_t2_synthetic_independent_r2():
    """2. Synthetic: y independent of X -> R² ≈ 0 (|R²| < 0.05)."""
    rng = np.random.RandomState(42)
    n = 2000
    d = 16
    X = rng.randn(n, d)
    y = rng.randn(n)  # completely independent

    X_train, X_val = X[:1600], X[1600:]
    y_train, y_val = y[:1600], y[1600:]

    mean_X = X_train.mean(axis=0)
    std_X = np.maximum(X_train.std(axis=0), 1e-6)
    X_tr_std = (X_train - mean_X) / std_X
    X_va_std = (X_val - mean_X) / std_X

    mean_y = y_train.mean()
    y_tr_c = y_train - mean_y
    W = fit_ridge_regression(X_tr_std, y_tr_c, lambda_reg=1e-3)
    y_pred = X_va_std @ W + mean_y

    r2 = compute_r2(y_val, y_pred)
    assert r2 is not None
    assert abs(r2) < 0.05, f"Expected |R²| < 0.05, got {r2:.4f}"


def test_t2_auroc_separable_and_shuffled():
    """3. AUROC: perfectly separable synthetic -> 1.0; shuffled labels -> ≈ 0.5 (±0.05)."""
    rng = np.random.RandomState(42)
    n = 200
    # Separable
    y_true = np.array([0] * 100 + [1] * 100)
    scores_perfect = np.array([rng.uniform(0.0, 0.4) for _ in range(100)] + [rng.uniform(0.6, 1.0) for _ in range(100)])
    auc_perf = compute_auroc(y_true, scores_perfect)
    assert auc_perf is not None
    assert auc_perf == 1.0, f"Expected AUROC 1.0, got {auc_perf}"

    # Shuffled labels (random scores)
    scores_rand = rng.randn(2000)
    y_rand = rng.choice([0, 1], size=2000, p=[0.5, 0.5])
    auc_rand = compute_auroc(y_rand, scores_rand)
    assert auc_rand is not None
    assert abs(auc_rand - 0.5) < 0.05, f"Expected AUROC ~0.5, got {auc_rand}"


def test_t2_fixture_probes_suite_run(tmp_path: Path):
    """4. Fixture shard + fake latents -> full suite runs, JSON has all three feature keys,
    val split used every episode exactly once (no leakage: assert fit n_val == 20% of eps).
    """
    rng = np.random.RandomState(1337)
    num_episodes = 25  # 20% is 5 episodes
    episodes = []
    for ep_i in range(num_episodes):
        T = 40
        z = rng.randn(T, 128).astype(np.float32)
        h = rng.randn(T, 256).astype(np.float32)
        actions = rng.randint(0, 17, size=T, dtype=np.int32)
        rewards = rng.choice([0.0, 1.0], size=T, p=[0.9, 0.1]).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True
        events = np.zeros((T, 22), dtype=bool)
        # Sprinkle some events
        events[rng.choice(T, 5), rng.choice(5, 5)] = True
        mask_before = np.zeros((T, 22), dtype=bool)
        inventory = np.zeros((T, 16), dtype=np.float32)
        inventory[:, 1] = np.log1p(np.cumsum(rewards))  # wood
        inventory[:, 12] = 9.0  # health
        positions = np.cumsum(rng.randn(T, 2), axis=0).astype(np.float32)

        episodes.append({
            "z": z,
            "h": h,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "events": events,
            "mask_before": mask_before,
            "inventory": inventory,
            "positions": positions,
        })

    out_file = tmp_path / "probes_it121.json"
    res = evaluate_probes_suite(
        cached_episodes=episodes,
        iter_idx=121,
        git_commit="test_commit",
        out_path=out_file,
    )

    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)

    assert data["iter"] == 121
    assert data["git_commit"] == "test_commit"
    assert "notes" in data
    assert set(data["probes"].keys()) == {"z", "h", "s"}

    # Check that val split has exactly 20% of episodes (5 episodes * 40 steps = 200 steps)
    pos_x_val = data["probes"]["s"]["pos_x"]["n_val"]
    assert pos_x_val == 5 * 40, f"Expected 200 val steps, got {pos_x_val}"


def test_t2_isolation_no_arbiter_import():
    """5. Grep/AST test: atlas/probes.py imports nothing from atlas/arbiter.py."""
    import ast
    probes_file = Path(__file__).resolve().parent.parent / "probes.py"
    with open(probes_file, "r") as f:
        tree = ast.parse(f.read(), filename=str(probes_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "arbiter" not in alias.name, f"Illegal import in probes.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "arbiter" not in node.module, f"Illegal from-import in probes.py: {node.module}"
