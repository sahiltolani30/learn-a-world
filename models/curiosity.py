r"""
Curiosity-Driven Exploration Module for Learn-a-World.
Computes self-supervised intrinsic exploration bonuses based on world model prediction errors:
r_intrinsic = ||\hat{z}_{t+1} - \bar{z}_{t+1}||_2
And provides curiosity-guided action selection to drive exploration to novel states.
"""

from typing import List, Optional, Union
import numpy as np
import torch

from models.jepa import JEPAWorldModel

class CuriosityModule:
    """
    Computes intrinsic curiosity bonuses from JEPA prediction errors
    and provides active novelty-directed exploration actions.
    """

    REVERSE_ACTIONS = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(
        self,
        model: JEPAWorldModel,
        eta: float = 0.2,
        max_bonus: float = 1.0,
        memory_size: int = 100,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.eta = eta
        self.max_bonus = max_bonus
        self.memory_size = memory_size
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.visited_memory: List[torch.Tensor] = []
        self.last_action: Optional[int] = None

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Converts observation numpy array to PyTorch tensor."""
        if obs.ndim == 3:
            t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif obs.ndim == 4:
            t = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255.0
        else:
            raise ValueError(f"Invalid obs shape: {obs.shape}")
        return t.to(self.device)

    def record_visited(self, z: torch.Tensor) -> None:
        """Appends latent state to short-term exploration memory."""
        if len(self.visited_memory) >= self.memory_size:
            self.visited_memory.pop(0)
        self.visited_memory.append(z.detach().squeeze(0))

    def reset_memory(self) -> None:
        """Clears exploration memory between episodes."""
        self.visited_memory = []
        self.last_action = None

    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        obs: np.ndarray,
        action: Union[int, np.integer],
        next_obs: np.ndarray,
    ) -> float:
        r"""
        Calculates intrinsic exploration bonus:
        r_int = min(max_bonus, eta * ||\hat{z}_{t+1} - \bar{z}_{t+1}||_2)
        """
        self.model.eval()
        x_t = self._preprocess_obs(obs)
        x_next = self._preprocess_obs(next_obs)
        act_tensor = torch.tensor([int(action)], device=self.device)

        z_t = self.model.encode(x_t, use_target=False)
        pred_z_next = self.model.predictor(z_t, act_tensor)
        target_z_next = self.model.encode(x_next, use_target=True)

        error = torch.norm(pred_z_next - target_z_next, p=2).item()
        bonus = float(min(self.max_bonus, self.eta * error))
        return bonus

    @torch.no_grad()
    def select_curious_action(
        self,
        obs: np.ndarray,
        num_actions: int = 4,
    ) -> int:
        """
        Selects the candidate action whose predicted outcome is most novel
        relative to recent memory, with an anti-self-loop movement penalty
        and anti-backtracking momentum.
        """
        self.model.eval()
        x_t = self._preprocess_obs(obs)
        z_t = self.model.encode(x_t, use_target=False)
        self.record_visited(z_t)

        actions = torch.arange(num_actions, device=self.device)
        z_batch = z_t.repeat(num_actions, 1)
        pred_z = self.model.predictor(z_batch, actions)

        movement = torch.norm(pred_z - z_batch, p=2, dim=-1)

        if len(self.visited_memory) > 1:
            mem = torch.stack(self.visited_memory, dim=0)
            dists = torch.cdist(pred_z, mem, p=2).min(dim=1).values
        else:
            dists = torch.zeros(num_actions, device=self.device)

        scores = dists + 0.5 * movement

        # Anti-backtracking penalty to prevent 2-cycle oscillation
        if self.last_action is not None and self.last_action in self.REVERSE_ACTIONS:
            scores[self.REVERSE_ACTIONS[self.last_action]] -= 1.0

        best_action = int(torch.argmax(scores).item())
        self.last_action = best_action
        return best_action
