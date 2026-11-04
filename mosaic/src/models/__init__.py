"""MOSAIC models package for Phase P2 (Predictive Visual Belief & Dynamics)."""

from .encoder import VisualEncoder, ConvBlock, AttentionPool, AnchorDecoder
from .belief import RecurrentStochasticBelief, StochasticDistribution
from .dynamics import ActionEmbedding, DynamicsPredictorMember, DynamicsEnsemble
from .world_model import MosaicWorldModel
from .diagnostics import (
    evaluate_action_conditioning,
    evaluate_heldout_prediction,
    evaluate_representation_collapse,
)

__all__ = [
    "VisualEncoder",
    "ConvBlock",
    "AttentionPool",
    "AnchorDecoder",
    "RecurrentStochasticBelief",
    "StochasticDistribution",
    "ActionEmbedding",
    "DynamicsPredictorMember",
    "DynamicsEnsemble",
    "MosaicWorldModel",
    "evaluate_action_conditioning",
    "evaluate_heldout_prediction",
    "evaluate_representation_collapse",
]
