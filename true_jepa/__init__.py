"""A clean, pixel-only action-conditioned JEPA research implementation.

This package intentionally has no imports from the repository's previous agent
lines.  The only environment-facing object is :class:`PixelCrafter`, whose
public interface exposes observations, rewards, and termination flags.
"""

from true_jepa.agent import VisualJEPAAgent
from true_jepa.config import JEPAConfig, TrainConfig
from true_jepa.model import TrueJEPA
from true_jepa.replay import PixelEpisode, PixelReplay

__all__ = [
    "JEPAConfig",
    "TrainConfig",
    "TrueJEPA",
    "VisualJEPAAgent",
    "PixelEpisode",
    "PixelReplay",
]
