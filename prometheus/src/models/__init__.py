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
    SpatialTransformerPredictor,
    LatentPredictor,
    sample_token_mask,
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
    "SpatialTransformerPredictor",
    "LatentPredictor",
    "sample_token_mask",
    "VICRegLoss",
    "verify_non_collapse",
    "assert_non_collapse",
]
