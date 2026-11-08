"""
genesis_jepa.modules.phase_slam — Action-Free 2D Fourier Phase Correlation SLAM & VTM-Graph
===========================================================================================
Mathematical Foundation:
Action-conditioned odometry hallucinates motion into walls during collisions (drift -0.85 tiles/step)
because background water/mob animations make Z_{t+1} != Z_t.

We eliminate action conditioning:
  1. Action-Free 2D Spatial Phase Cross-Correlation on 8x8 patch grid:
       R(u, v) = ( F{Z_t} * F*{Z_{t+1}} ) / ( |F{Z_t} * F*{Z_{t+1}}| + eps )
       Delta p = argmax F^{-1}{ R(u, v) }
     Static background tiles maintain zero phase shift; localized water ripples act as
     uncorrelated phase noise that cancels out under inverse FFT, holding argmax at (0, 0).
  2. 2D Visual Topological Metric Graph (VTM-Graph) with loop closure.
  3. Online Successor Representation (SR) shortcut homing policy:
       a*(u) = argmax_a psi(u, a)^T phi(v_table)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ActionFreePhaseCorrelationVO(nn.Module):
    """
    Action-free ego-motion estimation via 2D Fourier phase cross-correlation.
    """

    def __init__(self, grid_size: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.eps = eps

        # Precompute 2D Hann window for spectral edge smoothing: (1, 1, 8, 8)
        hann_1d = torch.hann_window(grid_size, periodic=False)
        hann_2d = torch.outer(hann_1d, hann_1d).unsqueeze(0).unsqueeze(0)
        self.register_buffer("hann_window", hann_2d)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Current tokens of shape (B, 64, D)
            z_next: Next tokens of shape (B, 64, D)
        Returns:
            delta_p: Estimated displacement (B, 2) in [dx, dy] coordinates.
        """
        B, K, D = z_t.shape
        G = self.grid_size

        # Reshape tokens to 2D spatial grid: (B, D, 8, 8)
        # Note: z is (B, 64, D) -> transpose to (B, D, 64) -> view as (B, D, 8, 8)
        f_t = z_t.transpose(1, 2).view(B, D, G, G)
        f_next = z_next.transpose(1, 2).view(B, D, G, G)

        # Apply 2D Hann window to reduce boundary spectral leakage
        f_t_win = f_t * self.hann_window
        f_next_win = f_next * self.hann_window

        # 2D Real-to-Complex FFT across spatial dimensions (dim=-2, -1)
        F_t = torch.fft.rfft2(f_t_win, dim=(-2, -1))
        F_next = torch.fft.rfft2(f_next_win, dim=(-2, -1))

        # Normalized Cross-Power Spectrum: (F_t * conj(F_next)) / |F_t * conj(F_next)|
        cross_power = F_next * torch.conj(F_t)
        norm_cross_power = cross_power / (torch.abs(cross_power) + self.eps)

        # 2D Inverse FFT to compute phase correlation surface: (B, D, 8, 8)
        corr_surface = torch.fft.irfft2(norm_cross_power, s=(G, G), dim=(-2, -1))

        # Average correlation surface across feature channels: (B, 8, 8)
        avg_corr = torch.mean(corr_surface, dim=1)

        # Find peak correlation coordinate per batch item
        flat_indices = torch.argmax(avg_corr.view(B, -1), dim=-1)  # (B,)
        dy = (flat_indices // G).float()
        dx = (flat_indices % G).float()

        # Center cyclic wrap-around: displacements > G // 2 are negative shifts
        dx = torch.where(dx > G // 2, dx - G, dx)
        dy = torch.where(dy > G // 2, dy - G, dy)

        # Stack into displacement tensor: (B, 2)
        delta_p = torch.stack([dx, dy], dim=-1)
        return delta_p


@dataclass
class KeyframeNode:
    """Node in the 2D Visual Topological Metric Graph."""
    node_id: int
    pos: np.ndarray             # Estimated metric coordinate [x, y]
    latent_vector: np.ndarray   # Visual representation vector
    is_workstation: bool = False
    step_observed: int = 0


class VisualTopologicalMetricGraph:
    """
    2D Metric-Topological Graph (VTM-Graph) for spatial persistence and shortcut homing.
    """

    def __init__(self, keyframe_distance: float = 3.0, similarity_threshold: float = 0.85) -> None:
        self.keyframe_distance = keyframe_distance
        self.similarity_threshold = similarity_threshold
        self.nodes: List[KeyframeNode] = []
        self.edges: Dict[int, List[int]] = {}
        self.current_pos = np.zeros(2, dtype=np.float32)

    def update_position(self, delta_p: np.ndarray) -> np.ndarray:
        self.current_pos = self.current_pos + np.asarray(delta_p, dtype=np.float32)
        return self.current_pos.copy()

    def maybe_add_keyframe(
        self,
        latent_features: np.ndarray,
        step_idx: int,
        is_workstation: bool = False,
    ) -> Optional[KeyframeNode]:
        """
        Inserts a keyframe if the agent has moved >= keyframe_distance from all nodes,
        or if a catalytic workstation is discovered.
        If the agent has returned to an existing node with high visual similarity (cos >= 0.85),
        detects loop closure, relaxes position, and returns the recognized node.
        """
        latent = np.asarray(latent_features, dtype=np.float32).flatten()
        if not self.nodes:
            node = KeyframeNode(0, self.current_pos.copy(), latent, is_workstation, step_idx)
            self.nodes.append(node)
            self.edges[0] = []
            return node

        # Check distance to all existing nodes
        dists = [float(np.linalg.norm(self.current_pos - n.pos)) for n in self.nodes]
        min_dist = min(dists)

        # 1. Loop Closure on Re-encountering an Earlier Node
        # Only check past nodes (excluding the immediate last node)
        if len(self.nodes) >= 2 and min_dist < 2.0:
            for idx in range(len(self.nodes) - 1):
                n = self.nodes[idx]
                if dists[idx] < 2.0:
                    cos_sim = float(np.dot(latent, n.latent_vector) / (
                        np.linalg.norm(latent) * np.linalg.norm(n.latent_vector) + 1e-6
                    ))
                    if cos_sim >= self.similarity_threshold:
                        # Loop closure confirmed: add shortcut edge between current last node and earlier node
                        last_id = self.nodes[-1].node_id
                        if idx not in self.edges[last_id]:
                            self.edges[last_id].append(idx)
                            self.edges[idx].append(last_id)
                        # Relax current pos toward ground truth node pos
                        self.current_pos = 0.5 * (self.current_pos + n.pos)
                        return n

        # 2. Insert New Keyframe Node if moved far enough
        if min_dist >= self.keyframe_distance or is_workstation:
            new_id = len(self.nodes)
            node = KeyframeNode(new_id, self.current_pos.copy(), latent, is_workstation, step_idx)
            self.nodes.append(node)
            self.edges[new_id] = []

            # Add edge to closest node
            closest_idx = int(np.argmin(dists))
            self.edges[new_id].append(closest_idx)
            self.edges[closest_idx].append(new_id)
            return node

        return None

    def get_workstation_nodes(self) -> List[KeyframeNode]:
        return [n for n in self.nodes if n.is_workstation]


class SuccessorRepresentationHoming:
    """
    Successor Representation (SR) shortcut policy:
        a*(u) = argmax_a psi(u, a)^T phi(v_table)
    Computes zero-shot geometric shortcuts back to workstation from perturbation states.
    """

    def __init__(self, num_nodes: int = 100, num_actions: int = 4, gamma: float = 0.95, lr: float = 0.1) -> None:
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        # Successor feature matrix: psi[node, action, target_node]
        self.psi = np.zeros((num_nodes, num_actions, num_nodes), dtype=np.float32)

    def update(self, u_t: int, a_t: int, u_next: int, a_next: Optional[int] = None) -> None:
        """TD(0) update on successor representation matrix."""
        if u_t >= self.num_nodes or u_next >= self.num_nodes:
            return

        one_hot_s = np.zeros(self.num_nodes, dtype=np.float32)
        one_hot_s[u_t] = 1.0

        if a_next is not None and u_next < self.num_nodes:
            next_psi = self.psi[u_next, a_next]
        else:
            next_psi = np.zeros(self.num_nodes, dtype=np.float32)

        td_target = one_hot_s + self.gamma * next_psi
        self.psi[u_t, a_t] += self.lr * (td_target - self.psi[u_t, a_t])

    def select_homing_action(self, current_node: int, target_node: int) -> int:
        """Selects action maximizing successor representation occupancy at target node."""
        if current_node >= self.num_nodes or target_node >= self.num_nodes:
            return 0

        # Occupancy values across all actions for the target workstation node
        action_occupancies = self.psi[current_node, :, target_node]
        return int(np.argmax(action_occupancies))
