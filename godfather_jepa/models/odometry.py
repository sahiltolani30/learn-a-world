"""
Topological Visual-Inertial Odometry and Landmark Anchor Graph for Godfather JEPA.

Solves the Off-Screen Workstation Return Problem (where the 9x7 camera viewport
causes placed crafting tables to slip off-screen when the agent searches for trees or stone).

Replaces privileged engine coordinate cheats with:
1. Learned visual ego-motion head predicting relative displacement:
       Delta p = g(z_t, a_t, z_{t+1}) in R^2
2. Topological Landmark Slots:
       Stores visual embedding of workstation z_table and relative displacement vector d_table.
3. Closed-Loop Path Reversal Option:
       Outputs motor bias vector pointing back toward the workstation when holding stone/resources.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class VisualOdometryHead(nn.Module):
    """
    Predicts 2D relative spatial displacement (dx, dy) from transition latents and action.
    """

    def __init__(self, latent_dim: int = 512, num_actions: int = 17, act_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.act_emb = nn.Embedding(num_actions, act_dim)
        in_dim = latent_dim * 2 + act_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        Predicts relative displacement (dx, dy).
        """
        a_vec = self.act_emb(action)
        inp = torch.cat([z_t, z_next, a_vec], dim=-1)
        return self.net(inp)


class TopologicalAnchorGraph:
    """
    Topological anchor manager tracking displacement to placed landmarks (crafting table, furnace).
    """

    def __init__(self):
        self.landmarks: Dict[str, Dict[str, Any]] = {}
        self.current_displacement: np.ndarray = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.landmarks.clear()
        self.current_displacement = np.zeros(2, dtype=np.float32)

    def register_landmark(self, name: str, latent_key: Optional[np.ndarray] = None) -> None:
        """
        Registers a newly placed workstation at the current position.
        """
        self.landmarks[name] = {
            "pos": self.current_displacement.copy(),
            "latent_key": latent_key,
        }

    def update_odometry(self, delta_p: np.ndarray | torch.Tensor) -> None:
        """
        Integrates incremental ego-motion step. Supports numpy arrays and PyTorch tensors.
        """
        if hasattr(delta_p, "detach"):
            delta_p = delta_p.detach().cpu().numpy()
        delta_p = np.asarray(delta_p, dtype=np.float32)
        if delta_p.ndim > 1:
            delta_p = delta_p.squeeze()
        self.current_displacement += delta_p

    def get_return_vector(self, name: str) -> Optional[np.ndarray]:
        """
        Computes 2D vector pointing back from current location to target landmark.
        """
        if name not in self.landmarks:
            return None
        target_pos = self.landmarks[name]["pos"]
        rel_vec = target_pos - self.current_displacement
        return rel_vec

    def get_return_action(self, name: str) -> Optional[int]:
        """
        Maps relative return vector to cardinal primitive movement action:
            Action 1: Left  (-dx)
            Action 2: Right (+dx)
            Action 3: Up    (-dy)
            Action 4: Down  (+dy)
        """
        vec = self.get_return_vector(name)
        if vec is None:
            return None

        dx, dy = vec[0], vec[1]
        dist = np.hypot(dx, dy)
        if dist < 0.5:
            # Already at workstation
            return 0  # noop

        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 1
        else:
            return 4 if dy > 0 else 3
