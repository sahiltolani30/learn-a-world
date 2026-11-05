"""Zephyr agents (strict inference): Mode A actor-greedy, Mode B CEM ablation.

Both agents take ONLY the raw observation on the action path — no env reward,
inventory, achievements, positions, or maps. Internal belief state is updated
from the observation alone via the world model. This is even stricter than the
previous line's screen (which fed env reward into its belief update).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from zephyr.ac import imagine_trajectory, unimix_probs
from zephyr.config import ZephyrConfig
from zephyr.heads import ZephyrHeads
from zephyr.wm import ZephyrWM


class _BaseAgent:
    def __init__(self, wm: ZephyrWM, heads: ZephyrHeads, device: str = "cpu") -> None:
        self.wm = wm.to(device).eval()
        self.heads = heads.to(device).eval()
        self.device = torch.device(device)
        self.cfg: ZephyrConfig = wm.cfg
        self._h: Optional[Tensor] = None
        self._z: Optional[Tensor] = None
        self._last_action: Optional[int] = None

    def reset(self) -> None:
        self._h = None
        self._z = None
        self._last_action = None

    def _observe(self, obs_uint8: np.ndarray) -> None:
        """Update internal state from one raw observation (no_grad, strict)."""
        obs = torch.from_numpy(np.ascontiguousarray(obs_uint8)).unsqueeze(0)
        obs = obs.to(self.device)
        with torch.no_grad():
            z = self.wm.encode(obs)
            if self._h is None:
                self._h = self.wm.init_state(1, self.device)
            if self._last_action is not None:
                prev = torch.tensor([self._last_action], device=self.device)
                self._h = self.wm.belief_step(z, prev, self._h)
            self._z = z

    @property
    def state(self) -> Tuple[Tensor, Tensor]:
        if self._z is None or self._h is None:
            raise RuntimeError("agent has not observed any frame yet")
        return self._z, self._h


class ZephyrAgent(_BaseAgent):
    """MODE A — the primary controller: greedy actor on the latent state."""

    def act(self, obs_uint8: np.ndarray) -> int:
        self._observe(obs_uint8)
        with torch.no_grad():
            z, h = self.state
            s = self.heads.state_input(z, h)
            logits = self.heads.actor(s)
            action = int(torch.argmax(logits, dim=-1).item())
        self._last_action = action
        return action


class ZephyrCEMAgent(_BaseAgent):
    """MODE B — ablation planner: CEM over the latent imagination.

    Scores candidate action sequences with predicted reward (optionally plus
    ensemble-disagreement intrinsic) discounted by predicted continues and a
    terminal critic value. Exists to test whether explicit consequence scoring
    adds anything over the trained actor (plan ablations A4/A5).
    """

    def __init__(
        self,
        wm: ZephyrWM,
        heads: ZephyrHeads,
        device: str = "cpu",
        horizon: int = 15,
        candidates: int = 64,
        iterations: int = 2,
        elite_frac: float = 0.15,
        use_intrinsic: bool = False,
        intrinsic_weight: float = 0.0,
    ) -> None:
        super().__init__(wm, heads, device)
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.iterations = int(iterations)
        self.elite_frac = float(elite_frac)
        self.use_intrinsic = bool(use_intrinsic)
        self.intrinsic_weight = float(intrinsic_weight)

    def _rollout_scores(self, z0: Tensor, h0: Tensor, seq: Tensor) -> Tensor:
        """Score [C, H] action sequences by closed-loop imagination. Returns [C]."""
        cfg = self.cfg
        gamma = cfg.ac.gamma
        coder = self.heads.reward.coder
        critic_coder = self.heads.critic.coder

        z = z0.expand(self.candidates, -1)
        h = h0.expand(self.candidates, -1)
        total = torch.zeros(self.candidates, device=self.device)
        discount = torch.ones(self.candidates, device=self.device)

        for t in range(self.horizon):
            a = seq[:, t]
            z_next = self.wm.predict(z, a, h)
            h_next = self.wm.belief_step(z_next, a, h)
            s_next = self.heads.state_input(z_next, h_next)
            r = coder.decode(self.heads.reward(s_next))
            c = 1.0 - torch.sigmoid(self.heads.contin(s_next))
            if self.use_intrinsic and self.intrinsic_weight > 0.0:
                from zephyr.ac import ensemble_disagreement

                preds = self.heads.ensemble(z, a, h)
                r = r + self.intrinsic_weight * ensemble_disagreement(preds)
            total = total + discount * r
            discount = discount * gamma * c
            z, h = z_next, h_next
        v_term = critic_coder.decode(self.heads.critic(self.heads.state_input(z, h)))
        return total + discount * v_term

    def act(self, obs_uint8: np.ndarray) -> int:
        self._observe(obs_uint8)
        with torch.no_grad():
            z0, h0 = self.state
            z0 = z0.expand(self.candidates, -1)
            h0 = h0.expand(self.candidates, -1)

            # Initialize per-step categorical parameters from the actor prior.
            s0 = self.heads.state_input(z0, h0)
            logits0 = self.heads.actor(s0)
            params = unimix_probs(logits0, self.cfg.ac.actor_unimix)  # [C, 17] identical rows
            params = params[0:1].repeat(self.candidates, 1)  # [C, 17] init distribution
            n_elite = max(2, int(self.elite_frac * self.candidates))
            best_action = int(torch.argmax(params.mean(dim=0)).item())

            for _ in range(self.iterations):
                probs = params / params.sum(dim=1, keepdim=True).clamp(min=1e-8)
                seq = torch.stack(
                    [
                        torch.multinomial(probs_row, self.horizon, replacement=True)
                        for probs_row in probs
                    ],
                    dim=0,
                )  # [C, H]
                scores = self._rollout_scores(z0, h0, seq)
                elite_idx = torch.topk(scores, n_elite).indices
                elite_seq = seq[elite_idx]  # [E, H]
                one_hot = torch.nn.functional.one_hot(elite_seq, self.cfg.heads.num_actions)
                elite_mean = one_hot.float().mean(dim=0)  # [H, 17]
                # Refit: first-step distribution sharpens around elite first actions.
                params = elite_mean[0:1].repeat(self.candidates, 1) + 1e-2
                best_action = int(elite_seq[0, 0].item())
        self._last_action = best_action
        return best_action
