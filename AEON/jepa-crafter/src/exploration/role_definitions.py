"""Logical exploration roles and objective weight profiles."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class RoleWeights:
    novelty: float = 1.0
    disagreement: float = 0.25
    progress: float = 0.25
    reward: float = 1.0
    termination_penalty: float = 2.0
    uncertainty_penalty: float = 0.5


ROLE_WEIGHTS: Dict[str, RoleWeights] = {
    "explorer": RoleWeights(
        novelty=1.00,
        disagreement=0.25,
        progress=0.25,
        reward=0.20,
        termination_penalty=1.0,
        uncertainty_penalty=0.5,
    ),
    "scientist": RoleWeights(
        novelty=0.25,
        disagreement=1.00,
        progress=0.50,
        reward=0.20,
        termination_penalty=1.0,
        uncertainty_penalty=0.25,
    ),
    "experimenter": RoleWeights(
        novelty=0.50,
        disagreement=0.50,
        progress=1.00,
        reward=0.20,
        termination_penalty=1.0,
        uncertainty_penalty=0.25,
    ),
    "balanced_planner": RoleWeights(
        novelty=0.50,
        disagreement=0.25,
        progress=0.25,
        reward=1.00,
        termination_penalty=2.0,
        uncertainty_penalty=0.50,
    ),
}
