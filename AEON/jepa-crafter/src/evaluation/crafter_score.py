"""Official Crafter achievement list and geometric-mean score calculation."""
import math
from typing import Dict, List, Any


CRAFTER_ACHIEVEMENTS: List[str] = [
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
]


def compute_crafter_score(achievement_rates_pct: Dict[str, float]) -> float:
    """Compute the official Crafter benchmark score.
    
    Formula:
        S = exp( (1 / 22) * sum_{i=1}^{22} ln(1 + s_i) ) - 1
    where s_i is achievement success rate in percentage points (0 to 100).
    """
    total_log = 0.0
    for ach in CRAFTER_ACHIEVEMENTS:
        s_i = float(achievement_rates_pct.get(ach, 0.0))
        # Ensure s_i is clamped to [0, 100]
        s_i = max(0.0, min(100.0, s_i))
        total_log += math.log(1.0 + s_i)

    mean_log = total_log / len(CRAFTER_ACHIEVEMENTS)
    score = math.exp(mean_log) - 1.0
    return float(score)
