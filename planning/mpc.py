r"""
Model Predictive Control (MPC) Latent-Space Planner for Learn-a-World.
Plans action sequences purely by imagining futures in latent representation space.
Features:
- Batched latent rollout through the learned JEPA predictor.
- Cross-Entropy Method (CEM) and Random Shooting.
- Anti-trapping state novelty penalty to navigate around walls and dead ends.
- Minimum trajectory distance scoring to reach goal states efficiently.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from models.jepa import JEPAWorldModel

class LatentMPCPlanner:
    """
    Sampling-based planner operating directly in abstract latent space.
    Does not interact with the real environment during planning.
    Includes history novelty penalty to prevent getting stuck in local wall minima.
    """

    def __init__(
        self,
        model: JEPAWorldModel,
        horizon: int = 5,
        num_candidates: int = 64,
        cem_iterations: int = 3,
        elite_ratio: float = 0.1,
        novelty_weight: float = 0.35,
        history_window: int = 15,
        num_actions: int = 4,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.novelty_weight = novelty_weight
        self.history_window = history_window
        self.num_actions = num_actions
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.model.eval()
        self.recent_latents: List[torch.Tensor] = []

    def reset_history(self) -> None:
        """Clears recent state history between episodes."""
        self.recent_latents = []

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Converts [H, W, C] uint8 image to [1, C, H, W] float32 tensor."""
        if obs.ndim == 3:
            tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif obs.ndim == 4:
            tensor = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255.0
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
        return tensor.to(self.device)

    @torch.no_grad()
    def plan(
        self,
        obs: np.ndarray,
        goal_obs: np.ndarray,
        goal_z: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Plans the optimal immediate action to reach the goal.
        Args:
            obs: Current observation image [64, 64, 3]
            goal_obs: Goal observation image [64, 64, 3]
            goal_z: Optional pre-encoded goal latent representation
        Returns:
            best_action: Integer action in [0, num_actions - 1]
        """
        # Encode current state and goal
        x_t = self._preprocess_obs(obs)
        z_0 = self.model.encode(x_t, use_target=True)  # [1, D]

        # Update recent history
        if len(self.recent_latents) >= self.history_window:
            self.recent_latents.pop(0)
        self.recent_latents.append(z_0.squeeze(0))

        if goal_z is None:
            x_g = self._preprocess_obs(goal_obs)
            z_goal = self.model.encode(x_g, use_target=True)  # [1, D]
        else:
            z_goal = goal_z

        num_elites = max(2, int(self.num_candidates * self.elite_ratio))

        # Categorical distribution per horizon step: shape [H, num_actions]
        action_probs = (
            torch.ones((self.horizon, self.num_actions), device=self.device)
            / self.num_actions
        )

        best_seq = None
        best_score = -float("inf")

        for iteration in range(self.cem_iterations):
            # Sample candidate action sequences: [N, H]
            candidates = torch.stack(
                [
                    torch.multinomial(action_probs[h], self.num_candidates, replacement=True)
                    for h in range(self.horizon)
                ],
                dim=1,
            )  # [N, H]

            # Roll out all candidate sequences in parallel
            z_current = z_0.repeat(self.num_candidates, 1)  # [N, D]
            min_distances = torch.full((self.num_candidates,), float("inf"), device=self.device)

            for h in range(self.horizon):
                step_actions = candidates[:, h]  # [N]
                z_current = self.model.predictor(z_current, step_actions)  # [N, D]

                # Distance to goal at this horizon step
                dists = torch.norm(z_current - z_goal, p=2, dim=-1)  # [N]
                min_distances = torch.minimum(min_distances, dists)

            # Base score: negative goal distance
            scores = -min_distances

            # Novelty bonus to prevent staying stuck in recent wall loops
            if len(self.recent_latents) > 1 and self.novelty_weight > 0:
                recent_t = torch.stack(self.recent_latents, dim=0)  # [M, D]
                novelties = torch.cdist(z_current, recent_t, p=2).min(dim=1).values  # [N]
                scores = scores + self.novelty_weight * novelties

            # Find top elites
            elite_indices = torch.topk(scores, num_elites).indices
            elite_candidates = candidates[elite_indices]  # [num_elites, H]

            current_best_score = float(scores[elite_indices[0]].item())
            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = elite_candidates[0]

            # Update distribution towards elites with smoothing
            new_probs = torch.zeros_like(action_probs)
            for h in range(self.horizon):
                counts = torch.bincount(elite_candidates[:, h], minlength=self.num_actions).float()
                new_probs[h] = counts / counts.sum()

            alpha = 0.5  # smoothing factor
            action_probs = alpha * new_probs + (1.0 - alpha) * action_probs

        return int(best_seq[0].item())

    @torch.no_grad()
    def rollout_trajectory(
        self,
        z_start: torch.Tensor,
        action_sequence: List[int],
    ) -> List[torch.Tensor]:
        r"""
        Imagines a trajectory from latent state z_start given a sequence of actions.
        Returns:
            latents: List of imagined latent vectors [z_0, \hat{z}_1, \dots, \hat{z}_H]
        """
        latents = [z_start]
        curr_z = z_start
        for action in action_sequence:
            act_tensor = torch.tensor([action], device=self.device)
            curr_z = self.model.predictor(curr_z, act_tensor)
            latents.append(curr_z)
        return latents
