"""
Crafter Tabula Rasa JEPA World Model package.
"""

from src.models import (
    ObservationEncoder,
    ResNetTinyEncoder,
    TargetEncoder,
    ActionEmbedding,
    ResidualMLPPredictor,
    LatentPredictor,
    VICRegLoss,
    verify_non_collapse,
    assert_non_collapse,
)

__version__ = "0.1.0"

__all__ = [
    "ObservationEncoder",
    "ResNetTinyEncoder",
    "TargetEncoder",
    "ActionEmbedding",
    "ResidualMLPPredictor",
    "LatentPredictor",
    "VICRegLoss",
    "verify_non_collapse",
    "assert_non_collapse",
]
