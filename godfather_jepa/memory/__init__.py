"""
Memory package for Godfather JEPA.

Two-Tier Complementary Learning System:
- OnlineEpisodicStore: Fast online episodic working memory (Hippocampus).
- SemanticConsolidatedBank: Slow offline consolidated semantic memory (Neocortex).
- OfflineConsolidationEngine: Consolidates online discoveries into permanent causal bank.
"""
from godfather_jepa.memory.online_store import OnlineEpisodicStore, TransitionStep
from godfather_jepa.memory.semantic_bank import SemanticConsolidatedBank
from godfather_jepa.memory.consolidation import OfflineConsolidationEngine

__all__ = [
    "OnlineEpisodicStore",
    "TransitionStep",
    "SemanticConsolidatedBank",
    "OfflineConsolidationEngine",
]
