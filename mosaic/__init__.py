"""MOSAIC visual-learning project package.

The package starts with the clean-room control boundary and baseline ladder.
Later world-model, atlas, skill, and planner modules are added only after the
corresponding phase gates pass.
"""

from .src.api import VisualPolicy, validate_action, validate_observation, validate_reward, validate_step
from .src.agent import RandomVisualPolicy, SB3VisualPolicy
from .src.env_boundary import CrafterPixelEnv, GymPixelEnv, OpaqueActionPermutation

__all__ = [
    "CrafterPixelEnv",
    "GymPixelEnv",
    "OpaqueActionPermutation",
    "RandomVisualPolicy",
    "SB3VisualPolicy",
    "VisualPolicy",
    "validate_action",
    "validate_observation",
    "validate_reward",
    "validate_step",
]
