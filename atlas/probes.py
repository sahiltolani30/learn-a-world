"""Ground-truth linear probes for ATLAS latent representations (T2).

Decodes state, inventory, vitals, events, and rewards from z, h, and composite s.
Pure NumPy, deterministic, closed-form ridge regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def rankdata_average(a: np.ndarray) -> np.ndarray:
    """Assigns ranks to 1D array data, averaging ties (1-based ranks)."""
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))

    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]

    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Computes AUROC via rank-count formula. Returns None if < 20 positive samples."""
    pos_mask = y_true > 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(y_true) - n_pos)

    if n_pos < 20 or n_neg == 0:
        return None

    ranks = rankdata_average(y_score)
    sum_ranks_pos = float(np.sum(ranks[pos_mask]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Computes R² coefficient of determination."""
    if len(y_true) < 2:
        return None
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-8:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def fit_ridge_regression(
    X_train_std: np.ndarray,
    y_train_c: np.ndarray,
    lambda_reg: float = 1e-3,
) -> np.ndarray:
    """Solves W = (XᵀX + λI)⁻¹ Xᵀ y for standardized X and centered y."""
    D = X_train_std.shape[1]
    I_D = np.eye(D, dtype=np.float64)
    XtX = X_train_std.T @ X_train_std
    Xty = X_train_std.T @ y_train_c
    W = np.linalg.solve(XtX + lambda_reg * I_D, Xty)
    return W


def _extract_episode_targets(ep: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extracts target vectors for a single episode."""
    T = len(ep["z"])
    targets: Dict[str, np.ndarray] = {}

    positions = ep.get("positions")
    if positions is not None and len(positions) == T:
        targets["pos_x"] = np.asarray(positions[:, 0], dtype=np.float32)
        targets["pos_y"] = np.asarray(positions[:, 1], dtype=np.float32)

    inventory = ep.get("inventory")
    if inventory is not None and len(inventory) == T:
        targets["inv_wood"] = np.asarray(inventory[:, 1], dtype=np.float32)
        targets["inv_stone"] = np.asarray(inventory[:, 2], dtype=np.float32)
        targets["vital_health"] = np.asarray(inventory[:, 12], dtype=np.float32)
        targets["vital_food"] = np.asarray(inventory[:, 13], dtype=np.float32)
        targets["vital_drink"] = np.asarray(inventory[:, 14], dtype=np.float32)
        targets["vital_energy"] = np.asarray(inventory[:, 15], dtype=np.float32)

    rewards = ep.get("rewards")
    if rewards is not None and len(rewards) == T:
        rew_next5 = np.zeros(T, dtype=np.float32)
        for t in range(T):
            rew_next5[t] = np.sum(rewards[t : min(t + 6, T)])
        targets["reward_next5"] = rew_next5

    events = ep.get("events")
    if events is not None and len(events) == T:
        evt_any = np.zeros(T, dtype=np.int32)
        for t in range(T):
            evt_any[t] = int(np.any(events[t : min(t + 6, T)]))
        targets["evt_next5_any"] = evt_any

        # Store raw events for later top-5 dim extraction
        targets["_raw_events"] = events

    return targets


def evaluate_probes_suite(
    cached_episodes: List[Dict[str, Any]],
    iter_idx: int,
    git_commit: str = "unknown",
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluates the full linear probe suite on cached episodes.

    Uses 80/20 episode-level split (index % 5 == 0 -> val).
    Reports validation metrics only.
    """
    if not cached_episodes:
        return {}

    # 1) Determine train and val episodes
    train_eps: List[Dict[str, Any]] = []
    val_eps: List[Dict[str, Any]] = []
    for i, ep in enumerate(cached_episodes):
        if i % 5 == 0:
            val_eps.append(ep)
        else:
            train_eps.append(ep)

    if not val_eps or not train_eps:
        # Fallback if too few episodes
        val_eps = cached_episodes
        train_eps = cached_episodes

    # 2) Find top 5 most frequent event dimensions across train episodes
    event_counts = np.zeros(22, dtype=np.int64)
    for ep in train_eps:
        if "events" in ep and len(ep["events"]) > 0:
            event_counts += np.sum(ep["events"], axis=0).astype(np.int64)
    top_5_events = list(np.argsort(-event_counts)[:5])

    # 3) Build feature and target arrays
    def assemble_data(eps: List[Dict[str, Any]]):
        features = {"z": [], "h": [], "s": []}
        targets_list: Dict[str, List[np.ndarray]] = {}

        for ep in eps:
            z_ep = np.asarray(ep["z"], dtype=np.float32)
            h_ep = np.asarray(ep["h"], dtype=np.float32)
            s_ep = np.concatenate([z_ep, h_ep], axis=-1)

            features["z"].append(z_ep)
            features["h"].append(h_ep)
            features["s"].append(s_ep)

            ep_targets = _extract_episode_targets(ep)
            raw_ev = ep_targets.pop("_raw_events", None)

            # Build top 5 event targets
            T = len(z_ep)
            if raw_ev is not None:
                for rank_idx, e_dim in enumerate(top_5_events):
                    evt_e = np.zeros(T, dtype=np.int32)
                    for t in range(T):
                        evt_e[t] = int(np.any(raw_ev[t : min(t + 6, T), e_dim]))
                    ep_targets[f"evt_dim{e_dim}_next5"] = evt_e

            for k, v in ep_targets.items():
                if k not in targets_list:
                    targets_list[k] = []
                targets_list[k].append(v)

        concat_features = {k: np.concatenate(v, axis=0) for k, v in features.items()}
        concat_targets = {k: np.concatenate(v, axis=0) for k, v in targets_list.items()}
        return concat_features, concat_targets

    X_train_dict, y_train_dict = assemble_data(train_eps)
    X_val_dict, y_val_dict = assemble_data(val_eps)

    # 4) Fit probes per feature set and target
    results: Dict[str, Dict[str, Any]] = {"z": {}, "h": {}, "s": {}}

    for feat_name in ["z", "h", "s"]:
        X_tr = X_train_dict[feat_name]
        X_va = X_val_dict[feat_name]

        mean_X = np.mean(X_tr, axis=0)
        std_X = np.maximum(np.std(X_tr, axis=0), 1e-6)

        X_tr_std = (X_tr - mean_X) / std_X
        X_va_std = (X_va - mean_X) / std_X

        for tgt_name, y_tr in y_train_dict.items():
            if tgt_name not in y_val_dict:
                continue
            y_va = y_val_dict[tgt_name]
            n_val = int(len(y_va))

            is_binary = tgt_name.startswith("evt_")
            mean_y = float(np.mean(y_tr))
            y_tr_c = y_tr - mean_y

            W = fit_ridge_regression(X_tr_std, y_tr_c, lambda_reg=1e-3)
            y_pred = X_va_std @ W + mean_y

            if is_binary:
                auc = compute_auroc(y_va, y_pred)
                results[feat_name][tgt_name] = {
                    "auroc": round(auc, 4) if auc is not None else None,
                    "n_val": n_val,
                }
            else:
                r2 = compute_r2(y_va, y_pred)
                results[feat_name][tgt_name] = {
                    "r2": round(r2, 4) if r2 is not None else None,
                    "n_val": n_val,
                }

    notes = (
        "diagnostics only; no code consumes these numbers. Interpretation rubric: "
        "R²/AUROC high on s but low on z -> info exists via recurrent memory (policy problem is learnable); "
        "low on all three -> perception ceiling (frozen backbone lacks info)."
    )

    report = {
        "iter": iter_idx,
        "git_commit": git_commit,
        "probes": results,
        "notes": notes,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


def export_probes(
    report: Dict[str, Any],
    train_log_path: Path,
) -> None:
    """Appends probe_best summary to train_log.jsonl."""
    if not report or "probes" not in report or "s" not in report["probes"]:
        return

    s_probes = report["probes"]["s"]
    probe_best: Dict[str, Any] = {}
    for tgt_name, val_dict in s_probes.items():
        metric = val_dict.get("r2") if "r2" in val_dict else val_dict.get("auroc")
        probe_best[f"{tgt_name}_s"] = metric

    summary_line = {"iter": report.get("iter"), "probe_best": probe_best}
    train_log_path = Path(train_log_path)
    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_log_path, "a") as f:
        f.write(json.dumps(summary_line) + "\n")
