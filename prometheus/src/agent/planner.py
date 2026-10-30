"""
src/agent/planner.py

Categorical Cross-Entropy Method (CEM) Latent Planner for JEPA World Model.
Fulfills requirements:
- R3: Zero-shortcut model-based planning in JEPA latent representation space.
- R3: Actor-seeded candidate trajectory sampling and terminal Critic value grounding.
- R5: Vectorized unroll via predictor.rollout in latent space under torch.no_grad().
"""

from typing import Tuple, Optional, Dict, Any, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.predictor import ResidualMLPPredictor
from src.agent.actor_critic import ActorPrior, TwinCritic


class CategoricalCEMPlanner:
    """
    Categorical Cross-Entropy Method (CEM) Planner in JEPA Latent Space.

    Algorithm:
    1. Initialize categorical action distribution P in [0, 1]^{H x |A|} across horizon H=5,
       seeded by the Actor policy prior pi_psi(a | z_0).
    2. For each iteration i in {0, ..., I-1}:
       a. Vectorized sample of N=50 candidate action sequences from P.
       b. Vectorized rollout in latent space:
          traj_mean, traj_all = predictor.rollout(z_0, actions, batch_first=True)
       c. Evaluate trajectory returns:
          J = sum_{t=0}^{H-1} gamma^t * r_int(t) + gamma^H * min_{j=1,2} Q_j(z_H, a_H)
       d. Select top M = alpha * N elite trajectories.
       e. Vectorized update of categorical distribution P using elite frequencies with
          Laplace smoothing and momentum.
    3. Return optimal first action a_0^* from the highest-return trajectory.
    """

    def __init__(
        self,
        predictor: ResidualMLPPredictor,
        actor_prior: Optional[ActorPrior] = None,
        critic: Optional[TwinCritic] = None,
        curiosity_engine: Optional[nn.Module] = None,
        horizon: int = 5,
        num_candidates: int = 50,
        elite_ratio: float = 0.10,
        num_iterations: int = 2,
        gamma: float = 0.99,
        smoothing: float = 0.5,
        laplace_smoothing: float = 0.05,
        temperature: float = 1.0,
        num_actions: Optional[int] = None,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if num_candidates < 2:
            raise ValueError(f"num_candidates must be >= 2, got {num_candidates}")
        if not (0.0 < elite_ratio <= 1.0):
            raise ValueError(f"elite_ratio must be in (0.0, 1.0], got {elite_ratio}")
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}")

        self.predictor = predictor
        self.actor_prior = actor_prior
        self.critic = critic
        self.curiosity_engine = curiosity_engine
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.elite_ratio = elite_ratio
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.smoothing = smoothing
        self.laplace_smoothing = laplace_smoothing
        self.temperature = temperature

        if num_actions is not None:
            self.num_actions = num_actions
        elif hasattr(predictor, "num_actions"):
            self.num_actions = predictor.num_actions
        elif actor_prior is not None and hasattr(actor_prior, "num_actions"):
            self.num_actions = actor_prior.num_actions
        else:
            self.num_actions = 17

        self.num_elites = max(1, int(self.num_candidates * self.elite_ratio))

    def _init_distribution(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Initialize categorical search distribution P in [0, 1]^{H x |A|}.
        Seeded from ActorPrior if available, else uniform.

        Args:
            z_0: Initial latent state tensor of shape (1, latent_dim).

        Returns:
            P: Categorical distribution tensor of shape (H, num_actions).
        """
        device = z_0.device
        if self.actor_prior is not None:
            with torch.no_grad():
                probs = self.actor_prior.get_probs(z_0, temperature=self.temperature)  # (1, |A|)
                p_0 = probs.squeeze(0)  # (|A|,)
                P = p_0.unsqueeze(0).repeat(self.horizon, 1)  # (H, |A|)
        else:
            P = torch.full(
                (self.horizon, self.num_actions),
                1.0 / self.num_actions,
                dtype=torch.float32,
                device=device,
            )

        return P

    @torch.no_grad()
    def plan(
        self,
        z_0: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Execute Categorical CEM planning from latent state z_0.

        Args:
            z_0: Current latent representation of shape (1, latent_dim) or (latent_dim,).

        Returns:
            best_action: Optimal first action a_0^* as an integer.
            plan_info: Diagnostic dictionary containing expected return, trajectory, and latency.
        """
        start_time = time.perf_counter()

        # Format input latent state: (1, latent_dim)
        if z_0.dim() == 1:
            z_0 = z_0.unsqueeze(0)
        elif z_0.dim() == 2 and z_0.size(0) != 1:
            z_0 = z_0[0:1]

        device = z_0.device
        P = self._init_distribution(z_0)  # shape (H, num_actions)

        # Pre-expand initial latent state to population size N
        z_0_batch = z_0.expand(self.num_candidates, -1)  # (N, latent_dim)
        discounts = (self.gamma ** torch.arange(self.horizon, device=device, dtype=torch.float32)).unsqueeze(1)  # (H, 1)

        best_return: Optional[float] = None
        best_trajectory: Optional[torch.Tensor] = None

        for iteration in range(self.num_iterations):
            # 1. Vectorized sample of N candidate action sequences: shape (N, H)
            # torch.multinomial over 2D (H, |A|) samples (H, N), transpose to (N, H)
            actions = torch.multinomial(
                P,
                num_samples=self.num_candidates,
                replacement=True,
            ).transpose(0, 1)

            # Prior seeding: inject greedy prior or previous best trajectory
            if iteration == 0 and self.actor_prior is not None:
                actions[0, 0] = torch.argmax(P[0])
            elif iteration > 0 and best_trajectory is not None:
                actions[0] = best_trajectory

            # 2. Vectorized latent rollout under JEPA dynamics: (H, N, D) and (K, H, N, D)
            traj_mean, traj_all = self.predictor.rollout(
                z_0_batch,
                actions,
                batch_first=True,
            )

            # 3. Vectorized trajectory return computation
            if self.curiosity_engine is not None and hasattr(self.curiosity_engine, "compute_intrinsic_reward"):
                # Accumulate step-by-step if curiosity engine has specialized normalization
                returns = torch.zeros(self.num_candidates, dtype=torch.float32, device=device)
                for t in range(self.horizon):
                    r_int_t = self.curiosity_engine.compute_intrinsic_reward(traj_all[:, t], update_stats=False)
                    returns += (self.gamma ** t) * r_int_t
            else:
                # Vectorized epistemic disagreement across heads: (H, N)
                mean_k = traj_all.mean(dim=0, keepdim=True)  # (1, H, N, D)
                r_all = ((traj_all - mean_k) ** 2).mean(dim=-1).mean(dim=0)  # (H, N)
                returns = (discounts * r_all).sum(dim=0)  # (N,)

            # Terminal Critic value grounding
            if self.critic is not None:
                z_H = traj_mean[self.horizon - 1]  # shape (N, latent_dim)
                if self.actor_prior is not None:
                    a_H = self.actor_prior.sample(z_H, deterministic=True)
                else:
                    a_H = actions[:, self.horizon - 1]

                v_H = self.critic.q_min(z_H, a_H)  # shape (N,)
                returns += (self.gamma ** self.horizon) * v_H

            # 4. Elite selection: top M candidates
            elite_indices = torch.topk(returns, k=self.num_elites).indices
            elite_actions = actions[elite_indices]  # (num_elites, H)

            # Track global best trajectory
            top_idx = elite_indices[0]
            top_return = returns[top_idx].item()
            if best_return is None or top_return > best_return:
                best_return = top_return
                best_trajectory = actions[top_idx].clone()

            # 5. Vectorized update of categorical distribution P over elite trajectories
            # Convert elite actions to one-hot and sum over candidates: shape (H, num_actions)
            counts = F.one_hot(elite_actions, num_classes=self.num_actions).float().sum(dim=0)
            p_elite = (counts + self.laplace_smoothing) / (
                counts.sum(dim=-1, keepdim=True) + self.laplace_smoothing * self.num_actions
            )
            P = (1.0 - self.smoothing) * P + self.smoothing * p_elite

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if best_trajectory is not None:
            best_action = int(best_trajectory[0].item())
            best_seq = best_trajectory.cpu().tolist()
        else:
            best_action = int(torch.argmax(P[0]).item())
            best_seq = [best_action] * self.horizon
            best_return = 0.0

        plan_info = {
            "best_action": best_action,
            "best_return": float(best_return),
            "best_trajectory": best_seq,
            "planning_latency_ms": elapsed_ms,
            "final_distribution": P.detach().cpu(),
            "num_candidates": self.num_candidates,
            "horizon": self.horizon,
            "num_iterations": self.num_iterations,
        }

        return best_action, plan_info

    def select_action(self, z_0: torch.Tensor) -> int:
        """
        Convenience wrapper executing planning and returning only the optimal action integer.

        Args:
            z_0: Latent state tensor.

        Returns:
            a_0^*: Optimal discrete action integer.
        """
        action, _ = self.plan(z_0)
        return action
