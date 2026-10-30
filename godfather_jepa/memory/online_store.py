"""
Tier 1: Fast Online Episodic Working Memory (Hippocampus) for Godfather JEPA.

Operates live during gameplay:
1. Records raw single-trial transition tuples (z_t, a_t, r_t, Delta x_t, z_{t+1}).
2. Tracks real-time spatial odometry displacement to placed workstations:
       d_table = sum_{k=t_0}^t Delta p_k
3. Maintains active multi-phase subgoal state machine:
       GATHERING_INPUTS -> PLACING_WORKSTATION -> IMMEDIATE_ASSEMBLY
4. Read-Only Protection: Never mutates or deletes permanent semantic cards during play,
   preventing the fast-falsification trap that destroyed cards in earlier attempts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from godfather_jepa.planning.causal_hypergraph import (
    get_fixture_alias,
)


@dataclass
class TransitionStep:
    """
    Single step recorded in working memory.
    """
    step_idx: int
    z_t: np.ndarray
    action: int
    reward: float
    done: bool
    z_next: np.ndarray
    item_deltas: Dict[str, int] = field(default_factory=dict)
    unlocked_achievement: Optional[str] = None
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


class OnlineEpisodicStore:
    """
    Fast online working memory instantiated per-episode.
    """

    def __init__(self, target_goal: Optional[str] = None):
        self.target_goal = target_goal
        self.steps: List[TransitionStep] = []
        self.workstation_placed_step: Optional[int] = None
        self.workstation_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.placed_workstations: Dict[str, Tuple[int, np.ndarray]] = {}
        self.placed_workstations_history: List[Tuple[str, int, np.ndarray]] = []
        self.current_displacement: np.ndarray = np.zeros(2, dtype=np.float32)
        self.current_inventory: Dict[str, int] = {}
        self.milestones_unlocked: List[str] = []

        # Subgoal State Machine
        self.subgoal_state: str = "GATHERING_INPUTS"

    def record_step(
        self,
        step_idx: int,
        z_t: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        z_next: np.ndarray,
        item_deltas: Optional[Dict[str, int]] = None,
        unlocked_achievement: Optional[str] = None,
        delta_p: Optional[np.ndarray] = None,
    ) -> None:
        """
        Records an interaction step into episodic working memory.
        """
        if delta_p is not None:
            self.current_displacement += delta_p

        # Update inventory tracking
        if item_deltas:
            for item, delta in item_deltas.items():
                self.current_inventory[item] = max(0, self.current_inventory.get(item, 0) + delta)

        phi_minus = sum(-d for d in item_deltas.values() if d < 0) if item_deltas else 0
        phi_plus = sum(d for d in item_deltas.values() if d > 0) if item_deltas else 0
        is_mass_deposition = (phi_minus > 0 and phi_plus == 0)

        if unlocked_achievement:
            self.milestones_unlocked.append(unlocked_achievement)
            is_fixture_ach = is_mass_deposition or unlocked_achievement.startswith("place_")
            if is_fixture_ach:
                station_key = unlocked_achievement
                self.placed_workstations[station_key] = (step_idx, self.current_displacement.copy())
                self.placed_workstations_history.append((station_key, step_idx, self.current_displacement.copy()))
                alias = get_fixture_alias(unlocked_achievement)
                if alias != station_key:
                    self.placed_workstations[alias] = (step_idx, self.current_displacement.copy())
                    self.placed_workstations_history.append((alias, step_idx, self.current_displacement.copy()))
                self.workstation_placed_step = step_idx
                self.workstation_pos = self.current_displacement.copy()
                self.subgoal_state = "IMMEDIATE_ASSEMBLY"

        trans = TransitionStep(
            step_idx=step_idx,
            z_t=z_t,
            action=action,
            reward=reward,
            done=done,
            z_next=z_next,
            item_deltas=item_deltas or {},
            unlocked_achievement=unlocked_achievement,
            displacement=self.current_displacement.copy(),
        )
        self.steps.append(trans)

    def is_adjacent_to_workstation(
        self, workstation_type: Optional[str] = None, threshold: float = 2.5
    ) -> bool:
        """
        Verifies physical spatial proximity to placed workstation.
        Checks all placements in history so earlier workstations are not lost when multiple are placed.
        """
        if workstation_type is not None:
            alias = get_fixture_alias(workstation_type)
            if hasattr(self, "placed_workstations_history") and self.placed_workstations_history:
                for s_type, _, s_pos in self.placed_workstations_history:
                    if s_type == workstation_type or s_type == alias or get_fixture_alias(s_type) == alias:
                        dist = float(np.linalg.norm(self.current_displacement - s_pos))
                        if dist <= threshold:
                            return True
            elif workstation_type in self.placed_workstations:
                pos = self.placed_workstations[workstation_type][1]
                return float(np.linalg.norm(self.current_displacement - pos)) <= threshold
            elif alias in self.placed_workstations:
                pos = self.placed_workstations[alias][1]
                return float(np.linalg.norm(self.current_displacement - pos)) <= threshold

        if self.workstation_placed_step is None:
            return False
        dist = float(np.linalg.norm(self.current_displacement - self.workstation_pos))
        return dist <= threshold

    def get_return_vector(self, workstation_type: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Returns displacement vector pointing back to placed workstation.
        """
        if workstation_type is not None:
            alias = get_fixture_alias(workstation_type)
            target_key = None
            if workstation_type in self.placed_workstations:
                target_key = workstation_type
            elif alias in self.placed_workstations:
                target_key = alias
            if target_key is not None:
                pos = self.placed_workstations[target_key][1]
                return pos - self.current_displacement

        if self.workstation_placed_step is None:
            return None
        return self.workstation_pos - self.current_displacement

    def clear(self) -> None:
        self.steps.clear()
        self.workstation_placed_step = None
        self.workstation_pos = np.zeros(2, dtype=np.float32)
        self.placed_workstations.clear()
        self.current_displacement = np.zeros(2, dtype=np.float32)
        self.current_inventory.clear()
        self.milestones_unlocked.clear()
        self.subgoal_state = "GATHERING_INPUTS"
