"""
src/curiosity package

Intrinsic motivation modules for JEPA World Models.
"""

from src.curiosity.ensemble import (
    RunningMeanStd,
    EpistemicCuriosityEngine,
)
from src.curiosity.rnd import LatentRND
from src.curiosity.episodic import EpisodicNovelty
from src.curiosity.boredom import BoredomMonitor

__all__ = [
    "RunningMeanStd",
    "EpistemicCuriosityEngine",
    "LatentRND",
    "EpisodicNovelty",
    "BoredomMonitor",
]
