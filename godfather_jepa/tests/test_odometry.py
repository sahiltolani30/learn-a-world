"""
Adversarial Empirical Challenge Suite for Requirement R4:
Visual Odometry & Topological Anchor Graph (godfather_jepa/models/odometry.py).

Verifies:
1. VisualOdometryHead forward pass, gradient flow, action sensitivity, and numerical stability.
2. TopologicalAnchorGraph accumulation of Delta p across long random trajectories.
3. Path reversal return vector geometry, all cardinal directions, diagonal quadrants, and arrival deadband.
4. Station placed at origin, moving far away, return vector orientation, and multiple placed stations.
5. Closed-loop simulated navigation back to placed workstations.
6. Adversarial edge cases: PyTorch tensor type mismatch in update_odometry, shape mismatches, and boundary transitions.
"""
import pytest
import numpy as np
import torch

from godfather_jepa.models.odometry import VisualOdometryHead, TopologicalAnchorGraph


class TestVisualOdometryHeadAdversarial:
    """Adversarial stress testing of VisualOdometryHead neural module."""

    @pytest.mark.parametrize("batch_size", [1, 2, 7, 16, 64, 128])
    def test_forward_output_shape(self, batch_size: int):
        head = VisualOdometryHead(latent_dim=512, num_actions=17)
        z_t = torch.randn(batch_size, 512)
        z_next = torch.randn(batch_size, 512)
        actions = torch.randint(0, 17, (batch_size,))

        disp = head(z_t, actions, z_next)
        assert disp.shape == (batch_size, 2)
        assert not torch.isnan(disp).any()
        assert not torch.isinf(disp).any()

    def test_unbatched_1d_input(self):
        """Single 1D transition input handling."""
        head = VisualOdometryHead(latent_dim=512, num_actions=17)
        z_t = torch.randn(512)
        z_next = torch.randn(512)
        action = torch.tensor(3)

        disp = head(z_t, action, z_next)
        assert disp.shape == (2,)
        assert not torch.isnan(disp).any()

    def test_all_17_crafter_actions_supported(self):
        """Ensures embedding layer supports all actions [0..16] without index errors."""
        head = VisualOdometryHead(latent_dim=256, num_actions=17)
        z_t = torch.randn(17, 256)
        z_next = torch.randn(17, 256)
        actions = torch.arange(17)

        disp = head(z_t, actions, z_next)
        assert disp.shape == (17, 2)

    def test_out_of_bounds_action_raises(self):
        """Action 17 or higher is invalid for num_actions=17."""
        head = VisualOdometryHead(latent_dim=256, num_actions=17)
        z_t = torch.randn(1, 256)
        z_next = torch.randn(1, 256)
        with pytest.raises(IndexError):
            head(z_t, torch.tensor([17]), z_next)

    def test_action_rank2_dimension_mismatch_raises(self):
        """
        Adversarial Finding: Passing actions shaped (B, 1) rather than (B,)
        causes torch.cat dimension mismatch (concatenating 2D latents with 3D embeddings).
        """
        head = VisualOdometryHead(latent_dim=256, num_actions=17)
        z_t = torch.randn(4, 256)
        z_next = torch.randn(4, 256)
        action_rank2 = torch.tensor([[1], [2], [3], [4]])

        with pytest.raises(RuntimeError, match="Tensors must have same number of dimensions"):
            head(z_t, action_rank2, z_next)

    def test_gradient_flow_and_differentiability(self):
        """Ensures non-zero, finite gradients propagate back to input latents and weights."""
        head = VisualOdometryHead(latent_dim=128, num_actions=17)
        z_t = torch.randn(4, 128, requires_grad=True)
        z_next = torch.randn(4, 128, requires_grad=True)
        actions = torch.tensor([0, 1, 2, 5])

        disp = head(z_t, actions, z_next)
        loss = (disp ** 2).sum()
        loss.backward()

        assert z_t.grad is not None
        assert z_next.grad is not None
        assert not torch.isnan(z_t.grad).any()
        assert not torch.isnan(z_next.grad).any()
        for param in head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_numerical_stability_extreme_latents(self):
        """Tests that head does not produce NaNs under large and small latent norms."""
        head = VisualOdometryHead(latent_dim=64, num_actions=17)
        actions = torch.tensor([1, 2, 3])

        # Large scale
        z_large = torch.randn(3, 64) * 1000.0
        disp_large = head(z_large, actions, z_large)
        assert not torch.isnan(disp_large).any()

        # Small scale
        z_small = torch.randn(3, 64) * 1e-5
        disp_small = head(z_small, actions, z_small)
        assert not torch.isnan(disp_small).any()

        # All zeros
        z_zero = torch.zeros(3, 64)
        disp_zero = head(z_zero, actions, z_zero)
        assert not torch.isnan(disp_zero).any()

    def test_action_sensitivity_invariant(self):
        """For identical latents, different actions should produce distinct displacements."""
        head = VisualOdometryHead(latent_dim=64, num_actions=17)
        z = torch.randn(1, 64).repeat(4, 1)
        actions = torch.tensor([1, 2, 3, 4])  # Left, Right, Up, Down

        disp = head(z, actions, z)
        # Check that rows are not all identical
        diff_1_2 = torch.norm(disp[0] - disp[1]).item()
        diff_1_3 = torch.norm(disp[0] - disp[2]).item()
        assert diff_1_2 > 1e-6 or diff_1_3 > 1e-6


class TestTopologicalAnchorGraphAccumulation:
    """Verifies incremental integration of relative ego-motion Delta p."""

    def test_linear_step_accumulation(self):
        graph = TopologicalAnchorGraph()
        step = np.array([1.5, -0.5], dtype=np.float32)
        for _ in range(10):
            graph.update_odometry(step)
        np.testing.assert_allclose(graph.current_displacement, [15.0, -5.0], rtol=1e-5)

    def test_exact_path_reversal(self):
        """Stepping forward and backward must return displacement to [0.0, 0.0]."""
        graph = TopologicalAnchorGraph()
        path = [
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([-1.0, 2.0], dtype=np.float32),
            np.array([-2.0, -6.0], dtype=np.float32),
        ]
        for delta in path:
            graph.update_odometry(delta)
        np.testing.assert_allclose(graph.current_displacement, [0.0, 0.0], atol=1e-6)

    def test_high_frequency_random_walk_accumulation(self):
        """5,000 step random walk must match numpy vector sum exactly."""
        rng = np.random.RandomState(42)
        steps = rng.randn(5000, 2).astype(np.float32)
        expected_total = np.sum(steps, axis=0)

        graph = TopologicalAnchorGraph()
        for step in steps:
            graph.update_odometry(step)

        np.testing.assert_allclose(graph.current_displacement, expected_total, atol=1e-4)

    def test_torch_tensor_input_auto_conversion(self):
        """
        Verifies that passing torch.Tensor directly to update_odometry is automatically
        converted and properly accumulates into displacement without raising TypeError.
        """
        graph = TopologicalAnchorGraph()
        tensor_disp = torch.tensor([1.0, 2.0], requires_grad=True)
        graph.update_odometry(tensor_disp)
        np.testing.assert_allclose(graph.current_displacement, [1.0, 2.0])

        # Test 2D batched tensor of shape (1, 2)
        tensor_disp_batched = torch.tensor([[3.0, -1.0]])
        graph.update_odometry(tensor_disp_batched)
        np.testing.assert_allclose(graph.current_displacement, [4.0, 1.0])


class TestTopologicalAnchorGraphNavigation:
    """Verifies landmark registration, return vectors, and cardinal return actions."""

    @pytest.mark.parametrize("dx,dy,expected_return_action,desc", [
        (10.0, 0.0, 1, "Agent moved Right (+x) -> return action is Left (1)"),
        (-10.0, 0.0, 2, "Agent moved Left (-x) -> return action is Right (2)"),
        (0.0, -10.0, 4, "Agent moved Up (-y) -> return action is Down (4)"),
        (0.0, 10.0, 3, "Agent moved Down (+y) -> return action is Up (3)"),
    ])
    def test_station_at_origin_cardinal_directions(
        self, dx: float, dy: float, expected_return_action: int, desc: str
    ):
        graph = TopologicalAnchorGraph()
        graph.register_landmark("table")
        graph.update_odometry(np.array([dx, dy], dtype=np.float32))

        ret_vec = graph.get_return_vector("table")
        np.testing.assert_allclose(ret_vec, [-dx, -dy], atol=1e-5)

        action = graph.get_return_action("table")
        assert action == expected_return_action, desc

    @pytest.mark.parametrize("dx,dy,expected_return_action,desc", [
        (10.0, 4.0, 1, "Q1 dominant X (+10, +4) -> return Left (1)"),
        (4.0, 10.0, 3, "Q1 dominant Y (+4, +10) -> return Up (3)"),
        (-10.0, 4.0, 2, "Q2 dominant X (-10, +4) -> return Right (2)"),
        (-4.0, 10.0, 3, "Q2 dominant Y (-4, +10) -> return Up (3)"),
        (-10.0, -4.0, 2, "Q3 dominant X (-10, -4) -> return Right (2)"),
        (-4.0, -10.0, 4, "Q3 dominant Y (-4, -10) -> return Down (4)"),
        (10.0, -4.0, 1, "Q4 dominant X (+10, -4) -> return Left (1)"),
        (4.0, -10.0, 4, "Q4 dominant Y (+4, -10) -> return Down (4)"),
    ])
    def test_station_at_origin_diagonal_quadrants(
        self, dx: float, dy: float, expected_return_action: int, desc: str
    ):
        graph = TopologicalAnchorGraph()
        graph.register_landmark("table")
        graph.update_odometry(np.array([dx, dy], dtype=np.float32))

        action = graph.get_return_action("table")
        assert action == expected_return_action, desc

    def test_arrival_threshold_deadband(self):
        """When distance < 0.5, returns Action 0 (noop, arrived)."""
        graph = TopologicalAnchorGraph()
        graph.register_landmark("table")

        # Exactly at landmark
        assert graph.get_return_action("table") == 0

        # Within 0.5 radius (dist = 0.49)
        graph.update_odometry(np.array([0.3, 0.3], dtype=np.float32))  # hypot ~ 0.424
        assert graph.get_return_action("table") == 0

        # Outside 0.5 radius (dist = 0.51)
        graph.reset()
        graph.register_landmark("table")
        graph.update_odometry(np.array([0.51, 0.0], dtype=np.float32))
        assert graph.get_return_action("table") == 1  # must move Left

    def test_extreme_distance_corner_case(self):
        """Station at origin, moving 100,000 units away."""
        graph = TopologicalAnchorGraph()
        graph.register_landmark("table")
        graph.update_odometry(np.array([100000.0, -50000.0], dtype=np.float32))

        ret_vec = graph.get_return_vector("table")
        np.testing.assert_allclose(ret_vec, [-100000.0, 50000.0])
        # |dx| (100k) > |dy| (50k) -> dx < 0 -> Action 1 (Left)
        action = graph.get_return_action("table")
        assert action == 1

    def test_multiple_placed_stations_and_closed_loop_navigation(self):
        """
        Registers multiple stations ('table' at (0, 0), 'furnace' at (20, 10)).
        Moves agent far away to (50, -30).
        Simulates closed-loop step-by-step navigation back to furnace, then table.
        """
        graph = TopologicalAnchorGraph()

        # Step 1: Register table at (0, 0)
        graph.register_landmark("table")

        # Step 2: Agent moves to (20, 10) and places furnace
        graph.update_odometry(np.array([20.0, 10.0], dtype=np.float32))
        graph.register_landmark("furnace")

        # Step 3: Agent moves away to (50, -30)
        graph.update_odometry(np.array([30.0, -40.0], dtype=np.float32))
        np.testing.assert_allclose(graph.current_displacement, [50.0, -30.0])

        # Step 4: Closed-loop navigation back to furnace
        step_size = 1.0
        action_deltas = {
            1: np.array([-step_size, 0.0], dtype=np.float32),  # Left
            2: np.array([step_size, 0.0], dtype=np.float32),   # Right
            3: np.array([0.0, -step_size], dtype=np.float32),  # Up
            4: np.array([0.0, step_size], dtype=np.float32),   # Down
        }

        # Navigate to furnace
        steps_taken = 0
        while steps_taken < 200:
            act = graph.get_return_action("furnace")
            if act == 0:
                break
            graph.update_odometry(action_deltas[act])
            steps_taken += 1

        assert graph.get_return_action("furnace") == 0
        dist_to_furnace = np.hypot(*(graph.current_displacement - np.array([20.0, 10.0])))
        assert dist_to_furnace < 0.5, f"Did not reach furnace: dist={dist_to_furnace}"

        # Navigate to table
        steps_taken = 0
        while steps_taken < 200:
            act = graph.get_return_action("table")
            if act == 0:
                break
            graph.update_odometry(action_deltas[act])
            steps_taken += 1

        assert graph.get_return_action("table") == 0
        dist_to_table = np.hypot(*graph.current_displacement)
        assert dist_to_table < 0.5, f"Did not reach table: dist={dist_to_table}"

    def test_unregistered_landmark_returns_none(self):
        graph = TopologicalAnchorGraph()
        assert graph.get_return_vector("phantom") is None
        assert graph.get_return_action("phantom") is None

    def test_reset_clears_landmarks_and_displacement(self):
        graph = TopologicalAnchorGraph()
        graph.register_landmark("table")
        graph.update_odometry(np.array([10.0, 20.0], dtype=np.float32))

        graph.reset()
        assert len(graph.landmarks) == 0
        np.testing.assert_allclose(graph.current_displacement, [0.0, 0.0])
        assert graph.get_return_vector("table") is None

    def test_landmark_re_registration_updates_location(self):
        """If landmark is placed again, its recorded position should be updated."""
        graph = TopologicalAnchorGraph()
        graph.register_landmark("table")
        np.testing.assert_allclose(graph.landmarks["table"]["pos"], [0.0, 0.0])

        graph.update_odometry(np.array([15.0, 25.0], dtype=np.float32))
        graph.register_landmark("table")
        np.testing.assert_allclose(graph.landmarks["table"]["pos"], [15.0, 25.0])
