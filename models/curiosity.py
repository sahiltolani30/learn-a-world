r"""
Curiosity-Driven Exploration Module for Learn-a-World.
Computes self-supervised intrinsic exploration bonuses based on world model prediction errors:
r_intrinsic = ||\hat{z}_{t+1} - \bar{z}_{t+1}||_2
And provides curiosity-guided action selection to drive exploration to novel states.

Phase 3 addition: Plan2ExploreCuriosity — 3-predictor ensemble disagreement over imagined
next latents (Plan2Explore). NOT Q-ensemble disagreement.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        num_actions: Optional[int] = None,
    ) -> int:
        """
        Selects the candidate action whose predicted outcome is most novel
        relative to recent memory, with an anti-self-loop movement penalty
        and anti-backtracking momentum.
        """
        n_act = (
            num_actions
            if num_actions is not None
            else getattr(self.model, "num_actions", 4)
        )
        self.model.eval()
        x_t = self._preprocess_obs(obs)
        z_t = self.model.encode(x_t, use_target=False)
        self.record_visited(z_t)

        actions = torch.arange(n_act, device=self.device)
        z_batch = z_t.repeat(n_act, 1)
        pred_z = self.model.predictor(z_batch, actions)

        movement = torch.norm(pred_z - z_batch, p=2, dim=-1)

        if len(self.visited_memory) > 1:
            mem = torch.stack(self.visited_memory, dim=0)
            dists = torch.cdist(pred_z, mem, p=2).min(dim=1).values
        else:
            dists = torch.zeros(n_act, device=self.device)

        scores = dists + 0.5 * movement

        # Anti-backtracking penalty to prevent 2-cycle oscillation (for 4-dir movement)
        if (
            self.last_action is not None
            and self.last_action in self.REVERSE_ACTIONS
            and self.REVERSE_ACTIONS[self.last_action] < n_act
        ):
            scores[self.REVERSE_ACTIONS[self.last_action]] -= 1.0

        best_action = int(torch.argmax(scores).item())
        self.last_action = best_action
        return best_action


# ---------------------------------------------------------------------------
# Plan2Explore Curiosity — Phase 3
# ---------------------------------------------------------------------------

class _PredictorHead(nn.Module):
    """Single latent dynamics predictor: (z, h, a) → next_z."""

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_actions: int = 17,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_emb_dim = action_emb_dim
        self.action_emb = nn.Embedding(num_actions, action_emb_dim)
        in_dim = latent_dim + recurrent_dim + action_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim]
        h: [B, recurrent_dim]
        action: [B] int64
        Returns: next_z_pred: [B, latent_dim]  (residual: z + delta)
        """
        if action.dtype in (torch.int64, torch.int32):
            if action.ndim > 1:
                action = action.squeeze(-1)
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        feat = torch.cat([z, h, a_emb], dim=-1)
        delta_z = self.net(feat)
        return z + delta_z


class Plan2ExploreCuriosity(nn.Module):
    """
    Plan2Explore intrinsic curiosity module (Phase 3).

    Curiosity bonus = mean per-dimension std across K independent latent dynamics predictors.
    Each predictor independently learns to predict z_{t+1} from (z_t, h_t, a_t).
    High disagreement → the world model's imagination is uncertain → informative transition.

    This is NOT Q-ensemble disagreement. Q measures value uncertainty; predictor ensemble
    measures dynamics model novelty — exploration novel to the model's own imagination.

    Usage during collection:
        bonus = curiosity.curiosity_bonus(z, h, action)      # [B] float, add to task reward
        loss  = curiosity.predictor_loss(z, h, action, z_tgt) # train online every N steps
    """

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_actions: int = 17,
        num_predictors: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.num_predictors = num_predictors

        self.predictors = nn.ModuleList([
            _PredictorHead(
                latent_dim=latent_dim,
                recurrent_dim=recurrent_dim,
                action_emb_dim=action_emb_dim,
                hidden_dim=hidden_dim,
                num_actions=num_actions,
            )
            for _ in range(num_predictors)
        ])

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-predictor next-z predictions.
        z: [B, latent_dim]
        h: [B, recurrent_dim]
        action: [B] int64
        Returns: predictions: [K, B, latent_dim]
        """
        preds = [p(z, h, action) for p in self.predictors]
        return torch.stack(preds, dim=0)  # [K, B, latent_dim]

    @torch.no_grad()
    def curiosity_bonus(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes scalar curiosity bonus per sample.
        bonus = mean over latent dims of std over K predictors.
        z: [B, latent_dim], h: [B, recurrent_dim], action: [B] int64
        Returns: [B] float
        """
        self.eval()
        preds = self.forward(z, h, action)  # [K, B, latent_dim]
        # std along K dimension, mean along latent dimension → [B]
        bonus = preds.std(dim=0).mean(dim=-1)  # [B]
        return bonus

    def predictor_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean MSE loss for each predictor vs stop-gradient target.
        z_target should be stop-grad (e.g. target encoder output).
        Returns scalar loss.
        """
        self.train()
        z_tgt = z_target.detach()
        losses = [
            F.mse_loss(pred(z, h, action), z_tgt)
            for pred in self.predictors
        ]
        return torch.stack(losses).mean()

    def make_optimizers(self, lr: float = 3e-4) -> List["torch.optim.Adam"]:
        """Returns K independent Adam optimizers (one per predictor)."""
        return [
            torch.optim.Adam(p.parameters(), lr=lr)
            for p in self.predictors
        ]

    def save_checkpoint(self, path: Union[str, "Path"]) -> None:
        """Saves the curiosity module state dict."""
        torch.save({"state_dict": self.state_dict()}, str(path))

    def load_checkpoint(self, path: Union[str, "Path"], device: Union[str, torch.device] = "cpu") -> None:
        """Loads curiosity module state dict from checkpoint."""
        ckpt = torch.load(str(path), map_location=device)
        self.load_state_dict(ckpt["state_dict"])
