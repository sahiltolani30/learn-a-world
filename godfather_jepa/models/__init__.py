"""
Models package for Godfather JEPA.
"""
from godfather_jepa.models.encoder import VisualConv4Encoder, SpatialFeatureProjector
from godfather_jepa.models.projector import ExpanderProjector
from godfather_jepa.models.target_ema import (
    TargetEMA,
    update_target_ema,
    compute_polyak_tau,
    PolyakMomentumScheduler,
)
from godfather_jepa.models.belief import SpatialConvGRU, VectorBeliefCore
from godfather_jepa.models.predictor import BoundedResidualPredictor
from godfather_jepa.models.ensemble import EpistemicPredictorEnsemble
from godfather_jepa.models.symlog_critic import (
    SymlogTwoHotCoder,
    SymlogCategoricalCritic,
    symlog,
    symexp,
)
from godfather_jepa.models.actor import AmortizedPolicyActor
from godfather_jepa.models.survival import SurvivalContinuationHead
from godfather_jepa.models.reward_head import DenseRewardHead
from godfather_jepa.models.odometry import VisualOdometryHead, TopologicalAnchorGraph

__all__ = [
    "VisualConv4Encoder",
    "SpatialFeatureProjector",
    "ExpanderProjector",
    "TargetEMA",
    "update_target_ema",
    "compute_polyak_tau",
    "PolyakMomentumScheduler",
    "SpatialConvGRU",
    "VectorBeliefCore",
    "BoundedResidualPredictor",
    "EpistemicPredictorEnsemble",
    "SymlogTwoHotCoder",
    "SymlogCategoricalCritic",
    "symlog",
    "symexp",
    "AmortizedPolicyActor",
    "SurvivalContinuationHead",
    "DenseRewardHead",
    "VisualOdometryHead",
    "TopologicalAnchorGraph",
]

