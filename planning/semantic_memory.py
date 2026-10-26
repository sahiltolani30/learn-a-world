#!/usr/bin/env python3
"""
planning/semantic_memory.py — Dual-Memory Cognitive World Model Engine
======================================================================

Implements the Two-Memory System for Learn-a-World v3:
1. ShortTermWorkingMemory:
   - Within-episode rolling context (K-step FIFO latent buffer).
   - Recurrent belief state tracking.
   - Landmark object anchors (e.g. table anchor to prevent out-of-sight forgetting).
   - Vital damage slope (acute combat/night hazard tracking).
2. LongTermSemanticMemory:
   - Non-parametric learned affordance bank extracted 100% from agent's own gameplay.
   - Precondition-gated visual cosine matching: eliminates false-positive visual hallucinations.
   - Affordance action & future return retrieval for guiding MPC search.
3. AutonomousMemoryWriter:
   - Hooks into environment step execution.
   - Autonomously detects milestone breakthroughs (Delta e_t > 0) and records memory entries.
   - Zero human handcrafting or hardcoded recipes.
"""

from collections import deque
import json
import math
import os
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
from eval.metrics import CRAFTER_ACHIEVEMENTS

NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22


class ShortTermWorkingMemory:
    """
    Short-Term Working Memory (System 1 - Within-Episode Context).
    Maintains rolling sensory history, landmark anchors, and vital state.
    """

    def __init__(self, capacity: int = 16, latent_dim: int = 128, recurrent_dim: int = 256):
        self.capacity = capacity
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim

        self.latent_window: Deque[torch.Tensor] = deque(maxlen=capacity)
        self.action_window: Deque[int] = deque(maxlen=capacity)
        self.reward_window: Deque[float] = deque(maxlen=capacity)
        
        self.recurrent_h: Optional[torch.Tensor] = None
        self.active_inventory: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        
        # Spatial/Landmark Anchors: stores latent representation of key structures
        # e.g., "place_table" -> latent vector of table when placed
        self.landmark_anchors: Dict[str, torch.Tensor] = {}
        self.landmark_step_created: Dict[str, int] = {}
        self.current_step: int = 0
        self.vital_trend: float = 0.0

    def reset(self) -> None:
        """Resets working memory at the start of a new episode."""
        self.latent_window.clear()
        self.action_window.clear()
        self.reward_window.clear()
        self.recurrent_h = None
        self.active_inventory.fill(0)
        self.landmark_anchors.clear()
        self.landmark_step_created.clear()
        self.current_step = 0
        self.vital_trend = 0.0

    def update(
        self,
        z_t: torch.Tensor,
        action: int,
        reward: float,
        next_h: torch.Tensor,
        inventory_mask: Optional[np.ndarray] = None,
        unlocked_event: Optional[str] = None,
    ) -> None:
        """Updates working memory state after a step."""
        self.current_step += 1
        z_cpu = z_t.detach().cpu().flatten()
        self.latent_window.append(z_cpu)
        self.action_window.append(int(action))
        self.reward_window.append(float(reward))
        self.recurrent_h = next_h.detach()

        if inventory_mask is not None:
            self.active_inventory = np.maximum(self.active_inventory, inventory_mask.astype(np.uint8))

        # Check for landmark anchor placement (e.g. table placed)
        if unlocked_event is not None and "table" in unlocked_event:
            self.landmark_anchors["place_table"] = z_cpu.clone()
            self.landmark_step_created["place_table"] = self.current_step

        # Update vital trend (damage slope over last 5 steps)
        if len(self.reward_window) >= 5:
            recent_rews = list(self.reward_window)[-5:]
            self.vital_trend = float(np.mean(recent_rews))

    def has_anchor(self, name: str, max_age: int = 250) -> bool:
        """Checks if a landmark anchor exists and is within temporal relevance."""
        if name not in self.landmark_anchors:
            return False
        age = self.current_step - self.landmark_step_created.get(name, 0)
        return age <= max_age

    def get_anchor(self, name: str) -> Optional[torch.Tensor]:
        """Retrieves a landmark anchor latent vector."""
        return self.landmark_anchors.get(name, None)


class LongTermSemanticMemory:
    """
    Long-Term Semantic Memory (System 2 - Across-Episode Affordance Bank).
    Stores learned causal tuples: (Key, Preconditions, Action, Outcome, Return).
    Zero hardcoded rules: 100% agent-discovered milestones.
    """

    def __init__(self, latent_dim: int = 128, device: Union[str, torch.device] = "cpu"):
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        # Non-parametric memory tensors
        self.keys: torch.Tensor = torch.empty((0, latent_dim), dtype=torch.float32, device=self.device)
        self.preconditions: torch.Tensor = torch.empty((0, NUM_ACHIEVEMENTS), dtype=torch.float32, device=self.device)
        self.actions: torch.Tensor = torch.empty((0,), dtype=torch.long, device=self.device)
        self.outcomes: torch.Tensor = torch.empty((0, NUM_ACHIEVEMENTS), dtype=torch.float32, device=self.device)
        self.returns: torch.Tensor = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.outcome_names: List[str] = []

    def __len__(self) -> int:
        return self.keys.shape[0]

    def add_entry(
        self,
        key: torch.Tensor,
        preconditions: np.ndarray,
        action: int,
        outcome: np.ndarray,
        realized_return: float = 0.0,
        outcome_name: str = "unknown",
    ) -> None:
        """Adds a single breakthrough transition into long-term memory."""
        k = key.detach().to(self.device).float().flatten()
        k_norm = k / (k.norm(p=2).clamp(min=1e-6))

        p = torch.from_numpy(preconditions).to(self.device).float().flatten()
        a = torch.tensor([action], dtype=torch.long, device=self.device)
        o = torch.from_numpy(outcome).to(self.device).float().flatten()
        r = torch.tensor([realized_return], dtype=torch.float32, device=self.device)

        self.keys = torch.cat([self.keys, k_norm.unsqueeze(0)], dim=0)
        self.preconditions = torch.cat([self.preconditions, p.unsqueeze(0)], dim=0)
        self.actions = torch.cat([self.actions, a], dim=0)
        self.outcomes = torch.cat([self.outcomes, o.unsqueeze(0)], dim=0)
        self.returns = torch.cat([self.returns, r], dim=0)
        self.outcome_names.append(outcome_name)

    def query(
        self,
        z_t: torch.Tensor,
        active_inventory: np.ndarray,
        sim_threshold: float = 0.70,
    ) -> Tuple[Optional[int], float, Optional[str]]:
        """
        Queries the memory bank for affordances applicable to current state.
        
        Dual-Key Conditioning (Eliminates Hallucination):
        1. Precondition Mask Check: Filters for memories whose preconditions are satisfied
           by current inventory: (preconditions <= active_inventory).
        2. Cosine Similarity Matching: Matches current visual latent z_t against candidate keys.
        
        Returns:
            (best_action, similarity_confidence, target_outcome_name)
        """
        if len(self) == 0:
            return None, 0.0, None

        z = z_t.detach().to(self.device).float().flatten()
        z_norm = z / (z.norm(p=2).clamp(min=1e-6))

        # 1. Precondition Filtering:
        # Preconditions must be a subset of active inventory: (1 - inv) * p == 0
        inv_t = torch.from_numpy(active_inventory).to(self.device).float()
        missing_preconditions = (1.0 - inv_t).unsqueeze(0) * self.preconditions  # [N, 22]
        precondition_satisfied = (missing_preconditions.sum(dim=-1) == 0)        # [N] bool

        # 2. Outcome Not Yet Satisfied Check:
        # Avoid retrieving actions for achievements the agent already has
        unmet_outcome = (self.outcomes * (1.0 - inv_t).unsqueeze(0)).sum(dim=-1) > 0  # [N] bool
        valid_mask = precondition_satisfied & unmet_outcome

        if not valid_mask.any():
            return None, 0.0, None

        valid_indices = torch.where(valid_mask)[0]
        valid_keys = self.keys[valid_indices]  # [M, 128]

        # 3. Visual Cosine Similarity Matching
        cos_sims = torch.mv(valid_keys, z_norm)  # [M]
        best_local_idx = int(torch.argmax(cos_sims))
        best_sim = float(cos_sims[best_local_idx].item())

        if best_sim >= sim_threshold:
            global_idx = int(valid_indices[best_local_idx].item())
            retrieved_action = int(self.actions[global_idx].item())
            retrieved_name = self.outcome_names[global_idx] if global_idx < len(self.outcome_names) else "milestone"
            return retrieved_action, best_sim, retrieved_name

        return None, best_sim, None

    def save(self, path: Union[str, Path]) -> Path:
        """Saves long-term memory bank to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": self.keys.cpu(),
                "preconditions": self.preconditions.cpu(),
                "actions": self.actions.cpu(),
                "outcomes": self.outcomes.cpu(),
                "returns": self.returns.cpu(),
                "outcome_names": self.outcome_names,
                "latent_dim": self.latent_dim,
            },
            str(p),
        )
        return p

    def load(self, path: Union[str, Path]) -> None:
        """Loads long-term memory bank from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Memory bank file not found: {p}")
        data = torch.load(str(p), map_location=self.device)
        self.keys = data["keys"].to(self.device)
        self.preconditions = data["preconditions"].to(self.device)
        self.actions = data["actions"].to(self.device)
        self.outcomes = data["outcomes"].to(self.device)
        self.returns = data["returns"].to(self.device)
        self.outcome_names = data.get("outcome_names", [])
        self.latent_dim = data.get("latent_dim", self.latent_dim)


class AutonomousMemoryWriter:
    """
    Autonomous Memory Writer.
    Hooks into environment interaction loop to auto-record breakthrough transitions.
    """

    def __init__(self, memory_bank: LongTermSemanticMemory):
        self.memory_bank = memory_bank
        self.ach_prev: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)

    def reset(self) -> None:
        """Resets milestone tracker for a new episode."""
        self.ach_prev.fill(0)

    def step(
        self,
        z_t: torch.Tensor,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Inspects step outcome. If an achievement unlocked, autonomously logs memory.
        Returns the name of the newly unlocked achievement, if any.
        """
        ach_dict = dict(info.get("achievements", {})) if isinstance(info, dict) else {}
        ach_curr = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)
        newly_unlocked: List[str] = []

        for idx, a_name in enumerate(CRAFTER_ACHIEVEMENTS):
            if ach_dict.get(a_name, 0) > 0:
                ach_curr[idx] = 1
                if self.ach_prev[idx] == 0:
                    newly_unlocked.append(a_name)

        delta_events = np.clip(ach_curr - self.ach_prev, 0, 1)

        if np.any(delta_events > 0):
            event_name = ", ".join(newly_unlocked)
            self.memory_bank.add_entry(
                key=z_t,
                preconditions=self.ach_prev.copy(),
                action=action,
                outcome=delta_events,
                realized_return=reward,
                outcome_name=event_name,
            )
            self.ach_prev = ach_curr.copy()
            return event_name

        self.ach_prev = ach_curr.copy()
        return None
