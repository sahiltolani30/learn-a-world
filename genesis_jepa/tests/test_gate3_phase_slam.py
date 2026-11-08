"""
genesis_jepa.tests.test_gate3_phase_slam — Automated Verification Suite for Gate 3
==================================================================================
Verifies:
  1. Collision Immunity: Collision drift <= 0.05 tiles over 20 consecutive collisions
     (static terrain + localized animated water ripples/mobs).
  2. Metric Translation Fidelity: Exact detection of [dx, dy] translation without action priors.
  3. 2D VTM-Graph: Topological keyframe node insertion & loop closure Gauss-Newton relaxation.
  4. Successor Representation Homing: Navigation returns agent to workstation with >= 90% success.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from genesis_jepa.modules.phase_slam import (
    ActionFreePhaseCorrelationVO,
    SuccessorRepresentationHoming,
    VisualTopologicalMetricGraph,
)


def test_collision_drift_immunity_under_water_animations() -> None:
    """
    Verifies that Action-Free Phase Correlation VO exhibits <= 0.05 tiles drift
    when stationary against an obstacle despite animated water ripples in 4 patches.
    """
    vo = ActionFreePhaseCorrelationVO(grid_size=8)
    B, K, D = 1, 64, 192

    # Baseline static scene (e.g. facing boulder in forest)
    z_static = torch.randn(B, K, D)

    total_drift = torch.zeros(2)

    # Simulate 20 consecutive collision steps with water/mob animations
    for step in range(20):
        z_next = z_static.clone()
        # Patches 60..63 represent dynamic water waves/mobs changing every step
        z_next[:, 60:, :] = torch.randn(B, 4, D)

        delta_p = vo(z_static, z_next).squeeze(0)  # (2,)
        total_drift += torch.abs(delta_p)

    max_drift = float(total_drift.max().item())
    assert max_drift <= 0.05, (
        f"Collision drift violation! Total drift={max_drift:.4f} > 0.05 tiles over 20 collisions"
    )


def test_metric_translation_fidelity() -> None:
    """
    Verifies that real camera shifts [dx, dy] are accurately extracted by Phase Correlation VO.
    """
    vo = ActionFreePhaseCorrelationVO(grid_size=8)
    B, D, G = 1, 192, 8

    # Create spatial feature grid: (1, 192, 8, 8)
    grid_t = torch.randn(B, D, G, G)

    # 1. Test Move Right: shift in X by +1
    grid_next_right = torch.roll(grid_t, shifts=1, dims=-1)
    z_t = grid_t.view(B, D, 64).transpose(1, 2)
    z_next_right = grid_next_right.view(B, D, 64).transpose(1, 2)

    delta_right = vo(z_t, z_next_right).squeeze(0)
    assert delta_right[0].item() == 1.0, f"Expected dx=1.0 for right shift, got {delta_right[0].item()}"
    assert delta_right[1].item() == 0.0, f"Expected dy=0.0 for right shift, got {delta_right[1].item()}"

    # 2. Test Move Down: shift in Y by +1
    grid_next_down = torch.roll(grid_t, shifts=1, dims=-2)
    z_next_down = grid_next_down.view(B, D, 64).transpose(1, 2)

    delta_down = vo(z_t, z_next_down).squeeze(0)
    assert delta_down[0].item() == 0.0, f"Expected dx=0.0 for down shift, got {delta_down[0].item()}"
    assert delta_down[1].item() == 1.0, f"Expected dy=1.0 for down shift, got {delta_down[1].item()}"


def test_vtm_graph_node_insertion_and_loop_closure() -> None:
    """
    Verifies 2D VTM-Graph creates nodes on >= 3 tile distance and closes loops.
    """
    graph = VisualTopologicalMetricGraph(keyframe_distance=3.0, similarity_threshold=0.85)

    feat_origin = np.random.randn(192).astype(np.float32)
    feat_origin /= np.linalg.norm(feat_origin)

    # 1. Add origin node
    node0 = graph.maybe_add_keyframe(feat_origin, step_idx=0, is_workstation=True)
    assert node0 is not None
    assert node0.is_workstation
    assert len(graph.nodes) == 1

    # 2. Small step (1 tile) -> No new keyframe
    graph.update_position(np.array([1.0, 0.0]))
    node_small = graph.maybe_add_keyframe(feat_origin, step_idx=1)
    assert node_small is None
    assert len(graph.nodes) == 1

    # 3. Big step (3 tiles total) -> New keyframe inserted
    graph.update_position(np.array([2.0, 0.0]))  # total pos is [3.0, 0.0]
    feat_node1 = np.random.randn(192).astype(np.float32)
    node1 = graph.maybe_add_keyframe(feat_node1, step_idx=3)
    assert node1 is not None
    assert len(graph.nodes) == 2
    assert 1 in graph.edges[0] and 0 in graph.edges[1]

    # 4. Loop closure: Walk back to near origin [0.2, 0.1] with identical origin feature
    graph.update_position(np.array([-2.8, 0.1]))  # pos becomes [0.2, 0.1]
    re_node = graph.maybe_add_keyframe(feat_origin, step_idx=10)
    # Loop closure recognized: re-identified as node 0!
    assert re_node is not None
    assert re_node.node_id == 0, f"Expected loop closure to node 0, got {re_node.node_id}"
    # Current pos relaxed toward ground truth node 0 pos [0.0, 0.0]
    assert np.all(graph.current_pos < np.array([0.2, 0.1]))


def test_successor_representation_homing_policy() -> None:
    """
    Verifies that the Successor Representation shortcut policy returns agent to workstation.
    """
    # 5-node linear chain: Node 0 (Table) <-> Node 1 <-> Node 2 <-> Node 3 <-> Node 4 (Perturbation)
    # Action 0: Move Left (towards table), Action 1: Move Right (away from table)
    sr = SuccessorRepresentationHoming(num_nodes=5, num_actions=2, gamma=0.90, lr=0.2)

    # Train SR dynamics along the chain for 200 iterations
    for _ in range(200):
        # Forward and backward transitions along the chain
        for u in range(1, 5):
            # Action 0 moves Left: u -> u - 1
            sr.update(u_t=u, a_t=0, u_next=u - 1)
        for u in range(0, 4):
            # Action 1 moves Right: u -> u + 1
            sr.update(u_t=u, a_t=1, u_next=u + 1)

    # Evaluate homing policy from perturbed state (Node 4) aiming for workstation (Node 0)
    target_workstation = 0
    correct_steps = 0
    total_evals = 100

    for _ in range(total_evals):
        curr_node = 4
        # Follow homing policy for at most 6 steps
        for _ in range(6):
            if curr_node == target_workstation:
                break
            action = sr.select_homing_action(curr_node, target_workstation)
            # Action 0 moves left towards 0
            if action == 0:
                curr_node -= 1
            else:
                curr_node = min(4, curr_node + 1)

        if curr_node == target_workstation:
            correct_steps += 1

    success_rate = correct_steps / total_evals
    assert success_rate >= 0.90, f"SR Homing success rate {success_rate:.2f} < 0.90 threshold!"
