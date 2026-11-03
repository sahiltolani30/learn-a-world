"""Cross-Entropy Method (CEM) and Random Shooting latent space planner."""
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from src.models.jepa_world_model import JepaWorldModel
from src.planning.latent_rollout import LatentRollout
from src.planning.candidate_scorer import CandidateScorer
from src.exploration.role_definitions import RoleWeights, ROLE_WEIGHTS


class CemPlanner:
    """Discrete-action Cross-Entropy Method planner in latent space."""

    def __init__(
        self,
        model: JepaWorldModel,
        scorer: Optional[CandidateScorer] = None,
        num_actions: int = 17,
        horizon: int = 8,
        population_size: int = 64,
        num_iterations: int = 2,
        num_elites: int = 8,
        smoothing: float = 0.10,
        algorithm: str = "cem",
    ):
        self.model = model
        self.rollout = LatentRollout(model)
        self.scorer = scorer if scorer is not None else CandidateScorer()
        
        self.num_actions = num_actions
        self.horizon = horizon
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.num_elites = num_elites
        self.smoothing = smoothing
        self.algorithm = algorithm

        # Diagnostic state of last planning decision
        self.last_plan_info: Dict[str, Any] = {}

    @torch.no_grad()
    def plan_first_action(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        role_weights: Optional[RoleWeights] = None,
    ) -> int:
        """Find the optimal sequence and return strictly its first action token.
        
        Args:
            latent: [1, 64, 128] or [64, 128]
            context: [1, 256] or [256]
            role_weights: RoleWeights
        Returns:
            first_action_token: int in [0, 16]
        """
        if role_weights is None:
            role_weights = ROLE_WEIGHTS["balanced_planner"]

        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        device = latent.device

        # If random shooting fallback requested
        if self.algorithm == "random_shooting":
            candidates = torch.randint(0, self.num_actions, (self.population_size, self.horizon), device=device)
            rollouts = self.rollout.rollout_candidates(latent, context, candidates)
            scores = self.scorer.score_rollouts(rollouts, role_weights)
            best_idx = torch.argmax(scores).item()
            best_seq = candidates[best_idx].cpu().numpy()

            self.last_plan_info = {
                "best_score": float(scores[best_idx].item()),
                "predicted_first_reward": float(rollouts["rewards"][best_idx, 0].item()),
                "predicted_first_term": float(rollouts["termination_probs"][best_idx, 0].item()),
                "predicted_first_disag": float(rollouts["disagreements"][best_idx, 0].item()),
                "best_sequence": best_seq.tolist(),
            }
            return int(best_seq[0])

        # CEM distribution over 17 tokens for each horizon step: [H, 17]
        action_dist = np.full((self.horizon, self.num_actions), 1.0 / self.num_actions, dtype=np.float32)

        best_score = -float("inf")
        best_sequence = None
        best_rollout_metrics = {}

        for iteration in range(self.num_iterations):
            # 1. Sample candidate sequences from categorical distributions
            candidate_list = []
            for _ in range(self.population_size):
                seq = [np.random.choice(self.num_actions, p=action_dist[step]) for step in range(self.horizon)]
                candidate_list.append(seq)
            candidates = torch.tensor(candidate_list, dtype=torch.long, device=device)

            # 2. Parallel latent rollout
            rollouts = self.rollout.rollout_candidates(latent, context, candidates)

            # 3. Score candidates
            scores = self.scorer.score_rollouts(rollouts, role_weights)

            # 4. Select top elites
            elite_scores, elite_indices = torch.topk(scores, k=self.num_elites)
            elite_seqs = candidates[elite_indices].cpu().numpy()  # [num_elites, H]

            if elite_scores[0].item() > best_score:
                best_score = float(elite_scores[0].item())
                best_idx = elite_indices[0].item()
                best_sequence = candidates[best_idx].cpu().numpy()
                best_rollout_metrics = {
                    "best_score": best_score,
                    "predicted_first_reward": float(rollouts["rewards"][best_idx, 0].item()),
                    "predicted_first_term": float(rollouts["termination_probs"][best_idx, 0].item()),
                    "predicted_first_disag": float(rollouts["disagreements"][best_idx, 0].item()),
                    "best_sequence": best_sequence.tolist(),
                }

            # 5. Update categorical distributions with Laplace / mixture smoothing
            new_dist = np.zeros_like(action_dist)
            for step in range(self.horizon):
                counts = np.bincount(elite_seqs[:, step], minlength=self.num_actions)
                elite_probs = counts / float(self.num_elites)
                # Smoothing
                new_dist[step] = (1.0 - self.smoothing) * elite_probs + self.smoothing * (1.0 / self.num_actions)
                new_dist[step] /= new_dist[step].sum()

            action_dist = new_dist

        self.last_plan_info = best_rollout_metrics
        return int(best_sequence[0])
