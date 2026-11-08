"""
Genesis JEPA: Cleanroom True Joint Embedding Predictive Architecture
====================================================================
Governing Protocol: AGENTS.md & GEMINI.md (Strict Zero-Cheating)
"""
from __future__ import annotations

from genesis_jepa.config import GenesisConfig
from genesis_jepa.world_model import GenesisJEPA
from genesis_jepa.modules.simnorm import SimNorm
from genesis_jepa.modules.slot_attention import InvertedCompetitiveSlotAttention
from genesis_jepa.modules.dynamic_jump import SpatiallyGatedJumpPredictor
from genesis_jepa.modules.losses import WorldModelGate1Loss, normalized_l1_loss

__all__ = [
    "GenesisConfig",
    "GenesisJEPA",
    "SimNorm",
    "InvertedCompetitiveSlotAttention",
    "SpatiallyGatedJumpPredictor",
    "WorldModelGate1Loss",
    "normalized_l1_loss",
]
