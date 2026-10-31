"""
src/curiosity/boredom.py

BoredomMonitor module for JEPA World Model (V3_2_FIX_PLAN.md §3.5 & UPGRADE_1_2_3_SPEC.md).
Exploration-channel only — critic and environment reward are untouched (fossil-proof).

Detection:
- Tracks first-time-in-run achievements per eval window.
- State BORED when zero first-time achievements occur for 3 consecutive eval windows.
- Resets to NORMAL immediately when any new achievement appears.

Escalation Ladder:
- L1: eps_base += 0.05 (capped at 0.25)
- L2: lifelong RND global novelty weight x2
- L3: actor entropy weight 0.05 -> 0.10
- L4: partial predictor-head reset (re-initializes 1 of 4 ensemble heads)

Double Duty (Disease Canary):
- Auto-checks representation health when BORED fires.
"""

from typing import Dict, Any, Set, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BoredomMonitor:
    """
    Adaptive exploration scheduler and plateau detector.
    Tracks achievement discovery and escalates exploration when stuck.
    """

    def __init__(
        self,
        window_threshold: int = 3,
        eps_step: float = 0.05,
        rnd_mult_step: float = 2.0,
    ):
        self.window_threshold = int(window_threshold)
        self.eps_step = float(eps_step)
        self.rnd_mult_step = float(rnd_mult_step)

        # State machine
        self.is_bored: bool = False
        self.consecutive_empty_windows: int = 0
        self.escalation_level: int = 0  # 0: Normal, 1: L1, 2: L2, 3: L3, 4: L4
        self.seen_achievements: Set[str] = set()
        self.last_head_reset: Optional[int] = None

    @property
    def eps_boost(self) -> float:
        """Additive boost to eps_base: L1+ adds eps_step (capped at 0.15 boost)."""
        if self.escalation_level >= 1:
            return min(0.15, self.eps_step * self.escalation_level)
        return 0.0

    @property
    def rnd_mult(self) -> float:
        """Multiplicative factor for lifelong RND exploration: L2+ gives 2.0x."""
        if self.escalation_level >= 2:
            return self.rnd_mult_step
        return 1.0

    @property
    def entropy_weight(self) -> float:
        """Actor entropy regularization weight: L3+ temporarily boosts from 0.05 to 0.10."""
        if self.escalation_level >= 3:
            return 0.10
        return 0.05

    def effective_eps_base(self, default_eps_base: float) -> float:
        """Return base epsilon with L1 boost applied (capped at 0.25)."""
        return min(0.25, default_eps_base + self.eps_boost)

    def step(
        self,
        current_unlocked: Set[str],
        agent: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate eval cycle discovery and update boredom state machine.
        
        Args:
            current_unlocked: Set of all unique achievements unlocked so far.
            agent: Optional JEPAAgent reference for L4 predictor-head reset.
            
        Returns:
            Status report dictionary.
        """
        new_achievements = set(current_unlocked) - self.seen_achievements

        if len(new_achievements) > 0:
            # Discovery! De-escalate completely to NORMAL
            self.seen_achievements.update(new_achievements)
            was_bored = self.is_bored
            self.is_bored = False
            self.consecutive_empty_windows = 0
            self.escalation_level = 0
            return {
                "is_bored": False,
                "escalation_level": 0,
                "new_achievements": sorted(list(new_achievements)),
                "de_escalated": was_bored,
            }

        # Zero new achievements this window
        self.consecutive_empty_windows += 1

        if self.consecutive_empty_windows >= self.window_threshold:
            self.is_bored = True
            # Escalate: 1 level per 2 consecutive BORED windows
            bored_duration = self.consecutive_empty_windows - self.window_threshold
            new_level = min(4, 1 + (bored_duration // 2))

            if new_level > self.escalation_level:
                self.escalation_level = new_level
                print(f"[!] BoredomMonitor: Escalated to Level {self.escalation_level} "
                      f"(empty windows: {self.consecutive_empty_windows})")

                # L4: Partial predictor-head reset (last resort)
                if self.escalation_level == 4 and agent is not None:
                    self._reset_one_predictor_head(agent)

        return {
            "is_bored": self.is_bored,
            "escalation_level": self.escalation_level,
            "consecutive_empty_windows": self.consecutive_empty_windows,
            "eps_boost": self.eps_boost,
            "rnd_mult": self.rnd_mult,
            "entropy_weight": self.entropy_weight,
        }

    def _reset_one_predictor_head(self, agent: Any) -> None:
        """
        L4 action: Re-initialize ONE predictor ensemble head.
        Re-curioses the world model without resetting encoder, policy, or critic.
        """
        if hasattr(agent, "predictor") and hasattr(agent.predictor, "heads"):
            heads = agent.predictor.heads
            # Select head 3 by default to preserve consensus heads 0-2
            target_idx = 3 % len(heads)
            target_head = heads[target_idx]
            nn.init.orthogonal_(target_head.weight, gain=0.1)
            nn.init.zeros_(target_head.bias)
            self.last_head_reset = target_idx
            print(f"[!] BoredomMonitor L4: Partial reset applied to predictor head {target_idx}.")

    def check_canary(self, agent: Any, device: torch.device) -> Dict[str, Any]:
        """
        Double Duty Canary: Runs representation health probe when BORED fires.
        Verifies whether boredom is genuine behavioral plateau or representation collapse.
        """
        agent.eval()
        with torch.no_grad():
            # Create synthetic frames: grass (spawn), stone, table, tree
            b = 4
            frames = torch.zeros(b, 3, 64, 64, device=device)
            # Frame 0: grass (greenish)
            frames[0, 1] = 0.6
            # Frame 1: stone (greyish)
            frames[1] = 0.5
            # Frame 2: table (brownish)
            frames[2, 0] = 0.6
            frames[2, 1] = 0.3
            # Frame 3: tree (dark green)
            frames[3, 1] = 0.3

            tokens, _ = agent.encoder.forward_tokens(frames)
            flat = tokens.flatten(start_dim=1)  # (4, 4096)
            norm = F.normalize(flat, p=2, dim=-1)
            cos_matrix = torch.mm(norm, norm.t())

            # Off-diagonal cosines
            triu_indices = torch.triu_indices(b, b, offset=1)
            pair_cosines = cos_matrix[triu_indices[0], triu_indices[1]]
            mean_cosine = float(pair_cosines.mean().item())

        is_collapsed = mean_cosine > 0.95
        if is_collapsed:
            print(f"[CANARY WARNING] Representation collapse detected! Mean token cosine: {mean_cosine:.4f} > 0.95. "
                  f"Boredom is a symptom of representation disease!")
        else:
            print(f"[CANARY OK] Representation is healthy. Mean token cosine: {mean_cosine:.4f} <= 0.95.")

        return {
            "mean_cosine": mean_cosine,
            "is_collapsed": is_collapsed,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "is_bored": self.is_bored,
            "consecutive_empty_windows": self.consecutive_empty_windows,
            "escalation_level": self.escalation_level,
            "seen_achievements": sorted(list(self.seen_achievements)),
            "last_head_reset": self.last_head_reset,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.is_bored = bool(state.get("is_bored", False))
        self.consecutive_empty_windows = int(state.get("consecutive_empty_windows", 0))
        self.escalation_level = int(state.get("escalation_level", 0))
        self.seen_achievements = set(state.get("seen_achievements", []))
        self.last_head_reset = state.get("last_head_reset", None)
