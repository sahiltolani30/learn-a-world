"""
Official Crafter Benchmark Metric & Scoring Engine for Godfather JEPA.

Computes the canonical geometric mean Crafter score across 22 achievements:
    score = exp( (1 / 22) * sum_{i=1}^{22} ln(1 + p_i) ) - 1
where p_i is the percentage of episodes unlocking achievement i (in [0.0, 100.0]).
Supports polymorphic inputs: sequence of episode dicts, numpy/torch matrices, or precomputed percentages.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np

# Canonical 22 Crafter achievements in alphabetical order
CRAFTER_ACHIEVEMENTS: Tuple[str, ...] = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)

ACHIEVEMENT_TO_IDX: Dict[str, int] = {ach: i for i, ach in enumerate(CRAFTER_ACHIEVEMENTS)}

ACHIEVEMENT_TIERS: Dict[str, List[str]] = {
    "Tier 1 (Basic Gathering & Survival)": [
        "collect_wood",
        "collect_drink",
        "collect_sapling",
        "eat_plant",
        "wake_up",
    ],
    "Tier 2 (Early Crafting & Shelter)": [
        "place_table",
        "make_wood_pickaxe",
        "make_wood_sword",
        "place_plant",
    ],
    "Tier 3 (Mining & Stone Age)": [
        "collect_stone",
        "collect_coal",
        "place_stone",
        "place_furnace",
        "make_stone_pickaxe",
        "make_stone_sword",
    ],
    "Tier 4 (Advanced Metallurgy & Combat)": [
        "collect_iron",
        "collect_diamond",
        "make_iron_pickaxe",
        "make_iron_sword",
        "defeat_zombie",
        "defeat_skeleton",
        "eat_cow",
    ],
}


class CrafterScoreResult(float):
    """
    Polymorphic result object returned by compute_crafter_score.
    Behaves simultaneously as a float (the Crafter score value) and as a
    dictionary mapping containing breakdown statistics for backwards-compatibility.
    """

    def __new__(cls, score: float, data: Dict[str, Any]) -> CrafterScoreResult:
        obj = super().__new__(cls, score)
        obj._data = data
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def values(self) -> Any:
        return self._data.values()

    def items(self) -> Any:
        return self._data.items()


def _empty_score_dict() -> Dict[str, Any]:
    return {
        "score": 0.0,
        "unlocked_count": 0,
        "mean_achievements_per_ep": 0.0,
        "unlock_percentages": {k: 0.0 for k in CRAFTER_ACHIEVEMENTS},
        "canonical_achievements": list(CRAFTER_ACHIEVEMENTS),
    }


def records_to_matrix(records: List[Dict[str, Any]]) -> np.ndarray:
    """Converts a list of episode achievement dictionaries into an [N, 22] uint8 binary matrix."""
    n = len(records)
    mat = np.zeros((n, len(CRAFTER_ACHIEVEMENTS)), dtype=np.uint8)
    for i, rec in enumerate(records):
        for ach, idx in ACHIEVEMENT_TO_IDX.items():
            if rec.get(ach, 0) > 0:
                mat[i, idx] = 1
    return mat


def matrix_to_records(matrix: np.ndarray) -> List[Dict[str, int]]:
    """Converts an [N, 22] binary matrix into a list of episode achievement dictionaries."""
    records: List[Dict[str, int]] = []
    for row in matrix:
        rec = {ach: int(row[idx] > 0) for ach, idx in ACHIEVEMENT_TO_IDX.items()}
        records.append(rec)
    return records


def compute_crafter_score(
    achievement_data: Union[
        Sequence[Dict[str, Any]],
        np.ndarray,
        Dict[str, float],
    ]
) -> CrafterScoreResult:
    """
    Computes official Crafter geometric score:
        S = exp( (1 / 22) * sum_{i=1}^{22} ln(1 + p_i) ) - 1
    where p_i is the percentage of episodes where achievement i was unlocked (0.0 .. 100.0).

    Accepts:
    1. Sequence[Dict[str, Any]]: list of per-episode achievement dicts
    2. np.ndarray (or torch.Tensor): shape [N, 22] binary/count matrix
    3. Dict[str, float]: dictionary of pre-computed percentages {ach_name: pct}
    """
    # Case 1: Pre-computed percentage dictionary
    if isinstance(achievement_data, dict):
        has_canonical = any(k in CRAFTER_ACHIEVEMENTS for k in achievement_data.keys())
        if has_canonical:
            percentages = {ach: float(achievement_data.get(ach, 0.0)) for ach in CRAFTER_ACHIEVEMENTS}
            num_achievements = len(CRAFTER_ACHIEVEMENTS)
        else:
            # Generic dictionary mapping
            percentages = {str(k): float(v) for k, v in achievement_data.items()}
            num_achievements = len(percentages) if percentages else len(CRAFTER_ACHIEVEMENTS)

        if num_achievements == 0:
            return CrafterScoreResult(0.0, _empty_score_dict())

        log_sum = sum(math.log(1.0 + max(0.0, pct)) for pct in percentages.values())
        score = float(math.exp(log_sum / num_achievements) - 1.0)
        unlocked = sum(1 for pct in percentages.values() if pct > 0.0)
        info = {
            "score": score,
            "unlocked_count": unlocked,
            "mean_achievements_per_ep": float(sum(percentages.values()) / 100.0),
            "unlock_percentages": percentages,
            "canonical_achievements": list(CRAFTER_ACHIEVEMENTS),
        }
        return CrafterScoreResult(score, info)

    # Case 2: Numpy array / Tensor
    if hasattr(achievement_data, "shape"):
        mat = np.asarray(achievement_data)
        if mat.size == 0 or mat.ndim == 0:
            return CrafterScoreResult(0.0, _empty_score_dict())
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.shape[-1] != len(CRAFTER_ACHIEVEMENTS):
            raise ValueError(
                f"Expected matrix with {len(CRAFTER_ACHIEVEMENTS)} achievement columns, got shape {mat.shape}"
            )
        num_episodes = mat.shape[0]
        if num_episodes == 0:
            return CrafterScoreResult(0.0, _empty_score_dict())

        binary_mat = (mat > 0).astype(np.float64)
        counts = binary_mat.sum(axis=0)
        percentages = {
            CRAFTER_ACHIEVEMENTS[i]: float((counts[i] / num_episodes) * 100.0)
            for i in range(len(CRAFTER_ACHIEVEMENTS))
        }
        per_ep_totals = binary_mat.sum(axis=1)

    # Case 3: Sequence of dictionaries
    else:
        records = list(achievement_data)
        num_episodes = len(records)
        if num_episodes == 0:
            return CrafterScoreResult(0.0, _empty_score_dict())

        counts = {ach: 0 for ach in CRAFTER_ACHIEVEMENTS}
        per_ep_totals = []
        for rec in records:
            ep_cnt = 0
            for ach in CRAFTER_ACHIEVEMENTS:
                if rec.get(ach, 0) > 0:
                    counts[ach] += 1
                    ep_cnt += 1
            per_ep_totals.append(ep_cnt)

        percentages = {
            ach: float((counts[ach] / num_episodes) * 100.0)
            for ach in CRAFTER_ACHIEVEMENTS
        }

    # Official geometric mean formula
    log_sum = sum(math.log(1.0 + pct) for pct in percentages.values())
    crafter_score = float(math.exp(log_sum / len(CRAFTER_ACHIEVEMENTS)) - 1.0)
    unlocked_count = sum(1 for pct in percentages.values() if pct > 0.0)
    mean_per_ep = float(np.mean(per_ep_totals)) if len(per_ep_totals) > 0 else 0.0

    info = {
        "score": crafter_score,
        "unlocked_count": unlocked_count,
        "mean_achievements_per_ep": mean_per_ep,
        "unlock_percentages": percentages,
        "canonical_achievements": list(CRAFTER_ACHIEVEMENTS),
    }
    return CrafterScoreResult(crafter_score, info)
