"""
src/models

Core neural network representations and dynamics for the JEPA World Model.
"""

from src.models.encoder import ObservationEncoder, ResNetTinyEncoder, ResBlock
from src.models.target import TargetEncoder
from src.models.predictor import (
    ActionEmbedding,
    ResidualMLPPredictorHead,
    ResidualMLPPredictor,
    LatentPredictor,
)
from src.models.losses import VICRegLoss, verify_non_collapse, assert_non_collapse

__all__ = [
    "ObservationEncoder",
    "ResNetTinyEncoder",
    "ResBlock",
    "TargetEncoder",
    "ActionEmbedding",
    "ResidualMLPPredictorHead",
    "ResidualMLPPredictor",
    "LatentPredictor",
    "VICRegLoss",
    "verify_non_collapse",
    "assert_non_collapse",
]
