"""
Tier 2: Slow Offline Consolidated Semantic Bank (Neocortex) for Godfather JEPA.

Stores permanent relational causal prototypes across all episodes:
1. Cross-Session Serialization: Saves to both PyTorch .pt and structured .json formats.
2. Hierarchical Upstream Budgeting: Recursively computes upstream resource totals
   (Budget(Pickaxe) = 3 Wood).
3. Google Drive Mirroring: Automatically syncs to /content/drive/MyDrive/learn-a-world/checkpoints/
   when running on Colab.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from godfather_jepa.planning.causal_hypergraph import CausalCard, CausalHypergraph


class SemanticConsolidatedBank:
    """
    Permanent semantic causal memory bank retaining cross-game knowledge.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self.hypergraph = CausalHypergraph()

    @property
    def cards(self) -> Dict[str, CausalCard]:
        return self.hypergraph.cards

    def calculate_upstream_budget(self, target_goal: str) -> Dict[str, int]:
        """
        Recursively aggregates required upstream resources (e.g. 3 Wood for Pickaxe).
        """
        return self.hypergraph.calculate_upstream_budget(target_goal)

    def determine_subgoal(
        self,
        target_goal: str,
        current_inventory: Dict[str, int],
        workstations_present: Set[str],
    ) -> Tuple[str, int]:
        """
        Selects active executable subgoal using upstream budgeting.
        """
        return self.hypergraph.determine_subgoal(
            target_goal=target_goal,
            current_inventory=current_inventory,
            workstations_present=workstations_present,
        )

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Serializes semantic memory bank to disk and Google Drive.
        """
        target_path = filepath or self.persistence_path or "checkpoints/godfather_memory_bank.pt"
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        data = {
            card_name: {
                "action": card.action,
                "name": card.name,
                "consumes": card.consumes,
                "produces": card.produces,
                "catalyst_needed": card.catalyst_needed,
                "predecessors": card.predecessors,
                "success_count": card.success_count,
                "failure_count": card.failure_count,
            }
            for card_name, card in self.cards.items()
        }

        # Safeguard: if target file already exists, merge existing cards so we never clobber richer knowledge
        if os.path.exists(target_path):
            try:
                existing_data = torch.load(target_path, map_location="cpu")
                if isinstance(existing_data, dict):
                    cards_dict = existing_data.get("cards", existing_data)
                    for c_name, c_dict in cards_dict.items():
                        if c_name not in data and isinstance(c_dict, dict):
                            data[c_name] = c_dict
            except Exception:
                pass

        # 1. PyTorch serialized checkpoint
        torch.save(data, target_path)

        # 2. JSON mirror for transparent inspection
        json_path = str(Path(target_path).with_suffix(".json"))
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Optional[str] = None) -> bool:
        """
        Loads semantic memory bank from disk or Google Drive.
        Supports both PyTorch (.pt) and JSON formats.
        """
        target_path = filepath or self.persistence_path or "checkpoints/godfather_memory_bank.pt"

        loaded = False
        data: Dict[str, Any] = {}

        # 1. Try loading from target path if it exists
        if os.path.exists(target_path):
            try:
                if str(target_path).endswith(".json"):
                    with open(target_path, "r") as f:
                        data = json.load(f)
                else:
                    loaded_torch = torch.load(target_path, map_location="cpu")
                    if isinstance(loaded_torch, dict):
                        data = loaded_torch.get("cards", loaded_torch)
                loaded = True
            except Exception:
                pass

        # 2. Try JSON mirror if .pt was not found or failed
        if not data:
            json_mirror = str(Path(target_path).with_suffix(".json"))
            if os.path.exists(json_mirror):
                try:
                    with open(json_mirror, "r") as f:
                        data = json.load(f)
                    loaded = True
                except Exception:
                    pass

        if not loaded or not data:
            return False

        try:
            for card_name, card_dict in data.items():
                if isinstance(card_dict, dict) and "action" in card_dict:
                    self.cards[card_name] = CausalCard(
                        action=int(card_dict["action"]),
                        name=card_dict.get("name", card_name),
                        consumes=card_dict.get("consumes", {}),
                        produces=card_dict.get("produces", {}),
                        catalyst_needed=card_dict.get("catalyst_needed"),
                        predecessors=card_dict.get("predecessors", []),
                        success_count=card_dict.get("success_count", 0),
                        failure_count=card_dict.get("failure_count", 0),
                        is_fixture=card_dict.get("is_fixture", False),
                    )
            return True
        except Exception:
            return False
