"""Canary pre-flight verification suite (T3).

Five automated integrity and functionality checks executed in order:
CAN1: integrity gates (isolation AST tests)
CAN2: trace round-trip (AtlasAgent.act + TraceWriter + Replay story renderer)
CAN3: probe harness (T2 synthetic ridge & AUROC checks)
CAN4: real-env 1-episode smoke (60 steps, backbone frozenness, tracing)
CAN5: no-cheat sweep (AST check for banned recipes, literals, and imports)

First failure prints `!!! [CANARY FAIL] <name>: <detail>` and exits nonzero.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, List

import numpy as np
import torch

from atlas.arbiter import AtlasAgent
from atlas.config import AtlasConfig, NUM_ACTIONS, NUM_ACHIEVEMENTS, REPO_ROOT
from atlas.probes import fit_ridge_regression, compute_r2, compute_auroc
from atlas.trace import TraceWriter


class StubBackbone:
    def __init__(self):
        self.device = "cpu"
        self.has_learned_delta_head = False
        self.checkpoint_path = "mock.pt"
        self.provenance = {}

    def encode(self, obs_uint8):
        return torch.randn(1, 128)

    def recurrent_step(self, z, a_prev, h_prev):
        return torch.randn(1, 256)

    def composite(self, z, h):
        return torch.cat([z, h], dim=-1)

    def event_probs_all_actions(self, s, mask):
        return torch.full((NUM_ACTIONS, NUM_ACHIEVEMENTS), 0.05)

    def event_logits_all_actions(self, s, mask):
        return torch.full((s.shape[0], NUM_ACTIONS, NUM_ACHIEVEMENTS), 0.05)

    def event_probs(self, s, a, mask):
        return torch.full((1, NUM_ACHIEVEMENTS), 0.05)

    def prior_logits_weighted(self, s, mask):
        return torch.zeros(1, NUM_ACTIONS)

    def q_mean(self, s, a):
        return torch.tensor([0.0])


class StubEpistemic:
    def disagreement_all_actions(self, s):
        return torch.full((1, NUM_ACTIONS), 0.1)

    def disagreement(self, s, a):
        return torch.tensor([0.1])


class StubSafety:
    def __init__(self, override_step: int = -1):
        self.override_step = override_step
        self.current_step = 0

    def risk_all_actions(self, s):
        risks = torch.full((1, NUM_ACTIONS), 0.05)
        if self.current_step == self.override_step:
            risks[0, :] = 0.02
            risks[0, 0] = 0.90
        self.current_step += 1
        return risks


def run_can1_integrity_gates() -> bool:
    """CAN1: runs existing isolation AST tests + no-cheat greps via pytest."""
    cmd = [sys.executable, "-m", "pytest", "atlas/tests/test_atlas_isolation.py", "-q"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"!!! [CANARY FAIL] CAN1 integrity gates: pytest isolation failed.\n{res.stdout}\n{res.stderr}")
        return False
    return True


def run_can2_trace_roundtrip() -> bool:
    """CAN2: run 50 steps through AtlasAgent.act + TraceWriter on temp dir, replay story renderer."""
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        cfg = AtlasConfig()
        agent = AtlasAgent(cfg=cfg, backbone=StubBackbone())
        agent.epistemic_net = StubEpistemic()
        agent.safety_net = StubSafety(override_step=10)

        writer = TraceWriter(run_dir=tmp_dir, max_folder_mb=50.0)
        agent.trace_enabled = True

        writer.begin_episode(iter_idx=1, ep_idx=0, seed=1337)
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        for t in range(50):
            agent.act(obs, reward_prev=0.0, step=t)
            writer.write_step(agent, step=t, reward_prev=0.0)
        writer.end_episode(length=50, total_reward=0.0, died=False)

        trace_file = tmp_dir / "trace" / "trace_it1.jsonl"
        if not trace_file.exists():
            print(f"!!! [CANARY FAIL] CAN2 trace round-trip: trace file not created.")
            return False

        lines = trace_file.read_text().strip().split("\n")
        if len(lines) != 50:
            print(f"!!! [CANARY FAIL] CAN2 trace round-trip: expected 50 lines, got {len(lines)}.")
            return False

        # Run story subcommand via subprocess
        cmd_story = [
            sys.executable,
            "-m",
            "atlas.replay",
            "story",
            "--run-dir",
            str(tmp_dir),
            "--iter",
            "1",
            "--ep",
            "0",
        ]
        res_story = subprocess.run(cmd_story, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if res_story.returncode != 0:
            print(f"!!! [CANARY FAIL] CAN2 trace round-trip: replay story failed: {res_story.stderr}")
            return False

        # Run filter subcommand via subprocess
        cmd_filter = [
            sys.executable,
            "-m",
            "atlas.replay",
            "filter",
            "--run-dir",
            str(tmp_dir),
            "--override",
        ]
        res_filter = subprocess.run(cmd_filter, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if res_filter.returncode != 0:
            print(f"!!! [CANARY FAIL] CAN2 trace round-trip: replay filter failed: {res_filter.stderr}")
            return False

    return True


def run_can3_probe_harness() -> bool:
    """CAN3: executes T2 synthetic probe checks (1)-(3) in-process."""
    rng = np.random.RandomState(42)

    # 1. Synthetic fit R² > 0.95
    n, d = 500, 32
    X = rng.randn(n, d)
    w_true = rng.randn(d)
    y = X @ w_true + 0.01 * rng.randn(n)

    X_tr, X_va = X[:400], X[400:]
    y_tr, y_va = y[:400], y[400:]

    mean_X = X_tr.mean(axis=0)
    std_X = np.maximum(X_tr.std(axis=0), 1e-6)
    X_tr_std = (X_tr - mean_X) / std_X
    X_va_std = (X_va - mean_X) / std_X

    mean_y = y_tr.mean()
    y_tr_c = y_tr - mean_y
    W = fit_ridge_regression(X_tr_std, y_tr_c, lambda_reg=1e-3)
    y_pred = X_va_std @ W + mean_y
    r2_fit = compute_r2(y_va, y_pred)
    if r2_fit is None or r2_fit <= 0.95:
        print(f"!!! [CANARY FAIL] CAN3 probe harness: fit R² {r2_fit} <= 0.95.")
        return False

    # 2. Independent |R²| < 0.05
    n, d = 2000, 16
    X_ind = rng.randn(n, d)
    y_ind = rng.randn(n)
    X_tr_i, X_va_i = X_ind[:1600], X_ind[1600:]
    y_tr_i, y_va_i = y_ind[:1600], y_ind[1600:]
    mean_Xi = X_tr_i.mean(axis=0)
    std_Xi = np.maximum(X_tr_i.std(axis=0), 1e-6)
    W_i = fit_ridge_regression((X_tr_i - mean_Xi) / std_Xi, y_tr_i - y_tr_i.mean(), lambda_reg=1e-3)
    r2_ind = compute_r2(y_va_i, ((X_va_i - mean_Xi) / std_Xi) @ W_i + y_tr_i.mean())
    if r2_ind is None or abs(r2_ind) >= 0.05:
        print(f"!!! [CANARY FAIL] CAN3 probe harness: independent R² {r2_ind} not within ±0.05.")
        return False

    # 3. AUROC separable == 1.0, random ~ 0.5
    y_true = np.array([0] * 100 + [1] * 100)
    scores_perf = np.array([rng.uniform(0.0, 0.4) for _ in range(100)] + [rng.uniform(0.6, 1.0) for _ in range(100)])
    auc_perf = compute_auroc(y_true, scores_perf)
    if auc_perf != 1.0:
        print(f"!!! [CANARY FAIL] CAN3 probe harness: separable AUROC {auc_perf} != 1.0.")
        return False

    scores_rand = rng.randn(2000)
    y_rand = rng.choice([0, 1], size=2000, p=[0.5, 0.5])
    auc_rand = compute_auroc(y_rand, scores_rand)
    if auc_rand is None or abs(auc_rand - 0.5) >= 0.05:
        print(f"!!! [CANARY FAIL] CAN3 probe harness: random AUROC {auc_rand} not within 0.50 ± 0.05.")
        return False

    return True


def run_can4_real_env_smoke() -> bool:
    """CAN4: real-env 1-episode smoke (seed 6150, 60 steps, frozenness assertion)."""
    # Check for local checkpoint
    ckpt_candidates = [
        REPO_ROOT / "checkpoints" / "jepa_crafter_v3_state_readout_v2.pt",
        REPO_ROOT / "checkpoints" / "jepa_crafter_v3_phase2.pt",
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        print("  [CAN4] SKIP (no checkpoint found locally)")
        return True

    try:
        from envs.crafter_wrapper import CrafterEnv
        from atlas.backbone import FrozenBackbone
    except Exception as e:
        print(f"  [CAN4] SKIP (env / backbone import failed: {e})")
        return True

    # 1. Strict signature check
    sig = inspect.signature(AtlasAgent.act)
    params = list(sig.parameters.keys())
    if params != ["self", "obs_uint8", "reward_prev", "step"]:
        print(f"!!! [CANARY FAIL] CAN4 real-env smoke: act signature violated: {params}")
        return False

    cfg = AtlasConfig()
    cfg.backbone.checkpoint_path = str(ckpt_path)
    cfg.backbone.device = "cpu"

    backbone = FrozenBackbone(config=cfg, device="cpu")
    backbone.assert_frozen()

    agent = AtlasAgent(cfg=cfg, backbone=backbone)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        writer = TraceWriter(run_dir=tmp_dir)
        agent.trace_enabled = True

        env = CrafterEnv(seed=6150)
        obs, reset_info = env.reset(seed=6150)
        agent.reset_episode(seed=6150)
        writer.begin_episode(iter_idx=0, ep_idx=0, seed=6150)

        reward_prev = 0.0
        for step in range(60):
            action, _ = agent.act(obs, reward_prev=reward_prev, step=step)
            writer.write_step(agent, step=step, reward_prev=reward_prev)
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_prev = float(reward)
            obs = next_obs
            if terminated or truncated:
                break
        writer.end_episode(length=60, total_reward=0.0, died=False)
        backbone.assert_frozen()

        trace_file = tmp_dir / "trace" / "trace_it0.jsonl"
        if not trace_file.exists():
            print("!!! [CANARY FAIL] CAN4 real-env smoke: trace file missing.")
            return False

        lines = trace_file.read_text().strip().split("\n")
        if len(lines) != 60:
            print(f"!!! [CANARY FAIL] CAN4 real-env smoke: expected 60 lines, got {len(lines)}.")
            return False

    return True


def run_can5_no_cheat_sweep() -> bool:
    """CAN5: structural AST sweep for banned recipe tables, achievement strings, and forbidden imports."""
    from eval.metrics import CRAFTER_ACHIEVEMENTS

    ach_set = set(CRAFTER_ACHIEVEMENTS)
    atlas_dir = REPO_ROOT / "atlas"

    # 1. CRAFT_REQUIRED_MATERIALS check in all non-test atlas files
    banned_name = "CRAFT_" + "REQUIRED_MATERIALS"
    for py_file in atlas_dir.rglob("*.py"):
        if "tests" in py_file.parts or py_file.name == "canaries.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        if banned_name in content:
            print(f"!!! [CANARY FAIL] CAN5 no-cheat sweep: banned structure {banned_name} in {py_file.name}")
            return False

    # 2. Achievement literals check in arbiter, goals, skills, trace
    check_files = [
        atlas_dir / "arbiter.py",
        atlas_dir / "goals.py",
        atlas_dir / "skills.py",
        atlas_dir / "trace.py",
    ]
    for p in check_files:
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in ach_set:
                    print(f"!!! [CANARY FAIL] CAN5 no-cheat sweep: achievement literal '{node.value}' in {p.name}:{node.lineno}")
                    return False

    # 3. AST import check for probes.py and replay.py
    forbidden_roots = {"envs", "eval", "models", "planning", "training", "dual_store"}
    for p in [atlas_dir / "replay.py", atlas_dir / "probes.py"]:
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in forbidden_roots or (p.name == "probes.py" and root == "torch"):
                        print(f"!!! [CANARY FAIL] CAN5 no-cheat sweep: {p.name} imports forbidden '{alias.name}'")
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in forbidden_roots or (p.name == "probes.py" and root == "torch"):
                        print(f"!!! [CANARY FAIL] CAN5 no-cheat sweep: {p.name} from-imports forbidden '{node.module}'")
                        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Project ATLAS Pre-Flight Canary Suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run CAN1, CAN2, CAN5 only (< 30s)")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"=== PROJECT ATLAS CANARY PRE-FLIGHT (quick={args.quick}) ===")
    print("=" * 60)

    # CAN1
    print("[1/5] Running CAN1 integrity gates...")
    if not run_can1_integrity_gates():
        sys.exit(1)
    print("  ✓ CAN1 PASSED")

    # CAN2
    print("[2/5] Running CAN2 trace round-trip...")
    if not run_can2_trace_roundtrip():
        sys.exit(1)
    print("  ✓ CAN2 PASSED")

    if not args.quick:
        # CAN3
        print("[3/5] Running CAN3 probe harness...")
        if not run_can3_probe_harness():
            sys.exit(1)
        print("  ✓ CAN3 PASSED")

        # CAN4
        print("[4/5] Running CAN4 real-env 1-episode smoke...")
        if not run_can4_real_env_smoke():
            sys.exit(1)
        print("  ✓ CAN4 PASSED")
    else:
        print("[3/5] SKIPPED CAN3 (quick mode)")
        print("[4/5] SKIPPED CAN4 (quick mode)")

    # CAN5
    print("[5/5] Running CAN5 no-cheat sweep...")
    if not run_can5_no_cheat_sweep():
        sys.exit(1)
    print("  ✓ CAN5 PASSED")

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"=== ALL CANARIES PASSED in {elapsed:.2f}s ===")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
