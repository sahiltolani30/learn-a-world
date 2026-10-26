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
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

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

        # Generic Object Permanence: track placed structures until subsequent breakthrough
        if unlocked_event is not None:
            if "place_" in unlocked_event:
                self.landmark_anchors["structure"] = z_cpu.clone()
                self.landmark_step_created["structure"] = self.current_step
            else:
                # Any milestone breakthrough fulfills the structure's potential -> release anchor
                self.landmark_anchors.clear()
                self.landmark_step_created.clear()

        # Update vital trend (damage slope over last 5 steps)
        if len(self.reward_window) >= 5:
            recent_rews = list(self.reward_window)[-5:]
            self.vital_trend = float(np.mean(recent_rews))

    def has_anchor(self, name: str = "structure", max_age: int = 250) -> bool:
        """Checks if a landmark anchor exists and is within temporal relevance."""
        # Support both generic 'structure' and legacy names
        key = name if name in self.landmark_anchors else ("structure" if "structure" in self.landmark_anchors else None)
        if key is None:
            return False
        age = self.current_step - self.landmark_step_created.get(key, 0)
        return age <= max_age

    def get_anchor(self, name: str = "structure") -> Optional[torch.Tensor]:
        """Retrieves a landmark anchor latent vector."""
        if name in self.landmark_anchors:
            return self.landmark_anchors[name]
        return self.landmark_anchors.get("structure", None)


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
        sim_threshold: float = 0.58,
    ) -> Tuple[Optional[int], float, Optional[str]]:
        """
        Queries the memory bank for affordances applicable to current state.
        
        Dual-Key Conditioning with Continuous Context Overlap:
        1. Context Overlap: Measures what proportion of memory preconditions are satisfied
           Overlap = (p_i . inv_t) / (||p_i||_1) >= 0.50 (forgiving coincidental noise like plants/cows).
        2. Outcome Not Yet Satisfied Check: Avoids repeating completed achievements.
        3. Visual Cosine Matching & Dopamine Return Weighting:
           Candidate score = cos_sim * (0.7 + 0.3 * overlap) * (1.0 + 0.2 * tanh(R_i))
        
        Returns:
            (best_action, similarity_confidence, target_outcome_name)
        """
        if len(self) == 0:
            return None, 0.0, None

        z = z_t.detach().to(self.device).float().flatten()
        z_norm = z / (z.norm(p=2).clamp(min=1e-6))

        inv_t = torch.from_numpy(active_inventory).to(self.device).float()

        # 1. Continuous Context Overlap
        p_counts = self.preconditions.sum(dim=-1)  # [N]
        matched_preconditions = (self.preconditions * inv_t.unsqueeze(0)).sum(dim=-1)  # [N]
        overlap = torch.where(
            p_counts > 0,
            matched_preconditions / p_counts,
            torch.tensor(1.0, device=self.device),
        )  # [N] in [0.0, 1.0]

        # Require >= 50% prerequisite overlap (filters hallucinations, ignores coincidental noise)
        sufficient_overlap = (overlap >= 0.50)

        # 2. Outcome Not Yet Satisfied Check
        unmet_outcome = (self.outcomes * (1.0 - inv_t).unsqueeze(0)).sum(dim=-1) > 0  # [N] bool
        valid_mask = sufficient_overlap & unmet_outcome

        if not valid_mask.any():
            return None, 0.0, None

        valid_indices = torch.where(valid_mask)[0]
        valid_keys = self.keys[valid_indices]  # [M, 128]

        # 3. Visual Cosine Similarity Matching
        cos_sims = torch.mv(valid_keys, z_norm)  # [M]

        # 4. Dopamine-weighted scoring: visual similarity * context overlap * return confidence
        valid_overlaps = overlap[valid_indices]
        valid_returns = self.returns[valid_indices]
        return_weights = 1.0 + 0.20 * torch.tanh(valid_returns)
        composite_scores = cos_sims * (0.70 + 0.30 * valid_overlaps) * return_weights

        best_local_idx = int(torch.argmax(composite_scores))
        best_sim = float(cos_sims[best_local_idx].item())

        if best_sim >= sim_threshold:
            global_idx = int(valid_indices[best_local_idx].item())
            retrieved_action = int(self.actions[global_idx].item())
            retrieved_name = (
                self.outcome_names[global_idx]
                if global_idx < len(self.outcome_names)
                else "milestone"
            )
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


class DopaminePersonalBestTracker:
    """
    Dopamine Personal-Best Circuit (Intrinsic Record-Breaking Reinforcement).
    Tracks historical lifetime and within-session bests:
    - Survival steps
    - Cumulative reward
    - Milestones unlocked in a single game
    - Unique milestones discovered across lifetime
    When an episode sets a new personal record, it triggers a 'dopamine surge'
    that boosts the transition's return and prioritizes its memory consolidation.
    """

    def __init__(self, dopamine_scale: float = 0.50):
        self.dopamine_scale = dopamine_scale
        self.lifetime_best_reward: float = 0.0
        self.lifetime_best_milestones: int = 0
        self.lifetime_best_steps: int = 0
        self.lifetime_unlocked_set: Set[str] = set()

        self.current_ep_reward: float = 0.0
        self.current_ep_milestones: int = 0
        self.current_ep_steps: int = 0
        self.record_broken_this_ep: bool = False

    def reset_episode(self) -> None:
        """Resets episode counters at the start of each episode."""
        self.current_ep_reward = 0.0
        self.current_ep_milestones = 0
        self.current_ep_steps = 0
        self.record_broken_this_ep = False

    def update_step(
        self,
        reward: float,
        newly_unlocked: List[str],
    ) -> float:
        """
        Updates step metrics and computes intrinsic dopamine bonus.
        Returns dopamine surge value (>= 0.0).
        """
        self.current_ep_steps += 1
        self.current_ep_reward += float(reward)
        self.current_ep_milestones += len(newly_unlocked)

        dopamine_surge = 0.0

        # 1. Unlocked an achievement NEVER seen before across all games
        for name in newly_unlocked:
            if name not in self.lifetime_unlocked_set:
                self.lifetime_unlocked_set.add(name)
                dopamine_surge += self.dopamine_scale * 1.5  # Large surge for unprecedented discovery!
                self.record_broken_this_ep = True

        # 2. Beat previous highest achievement count in a single game
        if self.current_ep_milestones > self.lifetime_best_milestones:
            self.lifetime_best_milestones = self.current_ep_milestones
            dopamine_surge += self.dopamine_scale * 1.0
            self.record_broken_this_ep = True

        # 3. Beat previous highest episode reward
        if self.current_ep_reward > self.lifetime_best_reward and self.lifetime_best_reward > 0.0:
            margin = self.current_ep_reward - self.lifetime_best_reward
            self.lifetime_best_reward = self.current_ep_reward
            dopamine_surge += float(np.clip(margin * 0.25, 0.0, 1.0))
            self.record_broken_this_ep = True
        elif self.lifetime_best_reward == 0.0 and self.current_ep_reward > 0.0:
            self.lifetime_best_reward = self.current_ep_reward

        # 4. Beat longest survival length
        if self.current_ep_steps > self.lifetime_best_steps:
            self.lifetime_best_steps = self.current_ep_steps

        return float(dopamine_surge)


class AutonomousMemoryWriter:
    """
    Autonomous Memory Writer with Dopamine Record Reinforcement.
    Hooks into environment interaction loop to auto-record breakthrough transitions.
    """

    def __init__(
        self,
        memory_bank: LongTermSemanticMemory,
        dopamine_tracker: Optional[DopaminePersonalBestTracker] = None,
    ):
        self.memory_bank = memory_bank
        self.dopamine_tracker = dopamine_tracker
        self.ach_prev: np.ndarray = np.zeros(NUM_ACHIEVEMENTS, dtype=np.uint8)

    def reset(self) -> None:
        """Resets milestone tracker for a new episode."""
        self.ach_prev.fill(0)
        if self.dopamine_tracker:
            self.dopamine_tracker.reset_episode()

    def step(
        self,
        z_t: torch.Tensor,
        action: int,
        reward: float,
        info: Dict[str, Any],
    ) -> Tuple[Optional[str], float]:
        """
        Inspects step outcome. If an achievement unlocked, autonomously logs memory.
        Returns (unlocked_name, dopamine_surge).
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
        dopamine_surge = 0.0

        if self.dopamine_tracker:
            dopamine_surge = self.dopamine_tracker.update_step(reward, newly_unlocked)

        if np.any(delta_events > 0):
            event_name = ", ".join(newly_unlocked)
            effective_return = float(reward) + dopamine_surge
            self.memory_bank.add_entry(
                key=z_t,
                preconditions=self.ach_prev.copy(),
                action=action,
                outcome=delta_events,
                realized_return=effective_return,
                outcome_name=event_name,
            )
            self.ach_prev = ach_curr.copy()
            return event_name, dopamine_surge

        self.ach_prev = ach_curr.copy()
        return None, dopamine_surge

