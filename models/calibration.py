"""
Event Head Calibration Module for Learn-a-World v3.

Post-hoc calibration of EventHead raw logits using temperature scaling / Platt scaling.
Calibrates probabilities p_i in [0, 1] for all 22 achievement heads without modifying model weights.

Standing Rules:
- References achievement heads strictly by integer INDEX (0..21) or active bitmasks.
- Never hardcodes achievement names in code logic.
- Supports pure temperature scaling (fit_bias=False) and Platt scaling (fit_bias=True).
- Works transparently with both NumPy arrays and PyTorch tensors.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


def compute_binary_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary predictions.
    probs: [N] predicted probabilities in [0, 1]
    labels: [N] binary ground truth {0, 1}
    """
    if len(probs) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total = len(probs)

    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / total) * abs(bin_acc - bin_conf)

    return float(ece)


class EventHeadCalibrator:
    """
    Per-head post-hoc calibrator for multi-label binary event heads.
    For each head k in 0..num_heads-1:
        calibrated_logit_k = (logit_k / T_k) + b_k
        calibrated_prob_k = sigmoid(calibrated_logit_k)
    """

    def __init__(
        self,
        num_heads: int = 22,
        fit_bias: bool = True,
        min_positives: int = 2,
    ):
        self.num_heads = num_heads
        self.fit_bias = fit_bias
        self.min_positives = min_positives

        # Per-head calibration parameters (strictly indexed by int)
        self.temperatures = np.ones(num_heads, dtype=np.float32)
        self.biases = np.zeros(num_heads, dtype=np.float32)
        self.active_mask = np.zeros(num_heads, dtype=bool)

    def fit_single_head(
        self,
        head_idx: int,
        logits_k: np.ndarray,
        labels_k: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fits temperature T_k (and optional bias b_k) for a single head by minimizing NLL (binary cross entropy).
        Uses log(T) parameterization to guarantee T > 0.
        """
        pos_count = int(np.sum(labels_k > 0))
        if pos_count < self.min_positives:
            # Frontier / zero-data head: keep identity calibration
            self.temperatures[head_idx] = 1.0
            self.biases[head_idx] = 0.0
            self.active_mask[head_idx] = False
            return {
                "temperature": 1.0,
                "bias": 0.0,
                "pos_count": pos_count,
                "status": "frontier_uncalibrated",
            }

        y = labels_k.astype(np.float64)
        z = logits_k.astype(np.float64)

        if self.fit_bias:
            # Optimize params = [log_T, b]
            def loss_fn(params: np.ndarray) -> float:
                log_T, b = params
                T = np.exp(log_T)
                scaled_z = z / T + b
                # Stable log-sigmoid computation
                p = 1.0 / (1.0 + np.exp(-np.clip(scaled_z, -30.0, 30.0)))
                eps = 1e-12
                p = np.clip(p, eps, 1.0 - eps)
                bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
                # Small L2 regularization on bias and deviation from T=1 to avoid runaway
                reg = 1e-4 * (b**2 + log_T**2)
                return bce + reg

            res = minimize(
                loss_fn,
                x0=np.array([0.0, 0.0]),
                method="L-BFGS-B",
                bounds=[(-4.0, 4.0), (-15.0, 15.0)],
            )
            opt_log_T, opt_b = res.x
            opt_T = float(np.exp(opt_log_T))
            opt_b = float(opt_b)
        else:
            # Optimize params = [log_T], bias fixed at 0
            def loss_fn(params: np.ndarray) -> float:
                log_T = params[0]
                T = np.exp(log_T)
                scaled_z = z / T
                p = 1.0 / (1.0 + np.exp(-np.clip(scaled_z, -30.0, 30.0)))
                eps = 1e-12
                p = np.clip(p, eps, 1.0 - eps)
                bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
                reg = 1e-4 * (log_T**2)
                return bce + reg

            res = minimize(
                loss_fn,
                x0=np.array([0.0]),
                method="L-BFGS-B",
                bounds=[(-4.0, 4.0)],
            )
            opt_T = float(np.exp(res.x[0]))
            opt_b = 0.0

        self.temperatures[head_idx] = opt_T
        self.biases[head_idx] = opt_b
        self.active_mask[head_idx] = True

        return {
            "temperature": opt_T,
            "bias": opt_b,
            "pos_count": pos_count,
            "status": "calibrated",
        }

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Fits calibration parameters across all heads.
        logits: [N, num_heads] raw unscaled logits
        labels: [N, num_heads] binary ground truth events
        """
        assert logits.shape[1] == self.num_heads, f"Expected {self.num_heads} heads, got {logits.shape[1]}"
        assert labels.shape == logits.shape, f"Shape mismatch: {logits.shape} vs {labels.shape}"

        results: Dict[int, Dict[str, Any]] = {}
        for k in range(self.num_heads):
            results[k] = self.fit_single_head(k, logits[:, k], labels[:, k])

        return results

    def apply(
        self,
        logits: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Applies calibration to input logits, returning calibrated probabilities in [0, 1].
        Supports both NumPy ndarrays and PyTorch Tensors.
        """
        if isinstance(logits, torch.Tensor):
            dev = logits.device
            dtype = logits.dtype
            t_temps = torch.from_numpy(self.temperatures).to(device=dev, dtype=dtype)
            t_biases = torch.from_numpy(self.biases).to(device=dev, dtype=dtype)
            scaled = (logits / t_temps) + t_biases
            return torch.sigmoid(scaled)
        elif isinstance(logits, np.ndarray):
            scaled = (logits / self.temperatures) + self.biases
            clipped = np.clip(scaled, -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        else:
            raise TypeError(f"Unsupported logits type: {type(logits)}")

    def evaluate_head_calibration(
        self,
        head_idx: int,
        logits_k: np.ndarray,
        labels_k: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluates calibration metrics for a single head before and after calibration.
        """
        y_true = labels_k.astype(np.int32)
        pos_count = int(np.sum(y_true))

        # Uncalibrated baseline
        raw_p = 1.0 / (1.0 + np.exp(-np.clip(logits_k, -30.0, 30.0)))
        raw_ece = compute_binary_ece(raw_p, y_true)
        raw_recall_05 = float(np.sum((raw_p > 0.5) & (y_true == 1)) / pos_count) if pos_count > 0 else 0.0

        # Calibrated
        T_k = self.temperatures[head_idx]
        b_k = self.biases[head_idx]
        cal_z = (logits_k / T_k) + b_k
        cal_p = 1.0 / (1.0 + np.exp(-np.clip(cal_z, -30.0, 30.0)))
        cal_ece = compute_binary_ece(cal_p, y_true)
        cal_recall_05 = float(np.sum((cal_p > 0.5) & (y_true == 1)) / pos_count) if pos_count > 0 else 0.0

        return {
            "head_idx": head_idx,
            "pos_count": pos_count,
            "temperature": float(T_k),
            "bias": float(b_k),
            "before_ece": float(raw_ece),
            "after_ece": float(cal_ece),
            "before_recall_05": float(raw_recall_05),
            "after_recall_05": float(cal_recall_05),
        }

    def evaluate_all(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Evaluates calibration metrics across all heads."""
        metrics = []
        for k in range(self.num_heads):
            metrics.append(self.evaluate_head_calibration(k, logits[:, k], labels[:, k]))
        return metrics

    def save(self, path: Union[str, Path]) -> None:
        """Saves calibration parameters to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_heads": self.num_heads,
            "fit_bias": self.fit_bias,
            "temperatures": [float(t) for t in self.temperatures],
            "biases": [float(b) for b in self.biases],
            "active_mask": [bool(m) for m in self.active_mask],
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EventHeadCalibrator":
        """Loads calibrator from a JSON file."""
        p = Path(path)
        with open(p, "r") as f:
            data = json.load(f)

        calibrator = cls(
            num_heads=data.get("num_heads", 22),
            fit_bias=data.get("fit_bias", True),
        )
        calibrator.temperatures = np.array(data["temperatures"], dtype=np.float32)
        calibrator.biases = np.array(data["biases"], dtype=np.float32)
        calibrator.active_mask = np.array(data["active_mask"], dtype=bool)
        return calibrator
