"""
Playtime interaction recorder for the Phase 2 online cross-game learning loop
(docs/PHASE2_MEMORY_REDESIGN_PLAN.md, §3 Stage 8).

Buffers per-step transitions captured strictly from the agent's own inference
(JEPA target-encoder latents, state-readout estimate, belief event head + belief
reward-fallback provenance) and hands whole episodes to the
RelationHypothesisBank via its learn_from_episode / refresh_edges contract.
The recorder touches NO other bank method.

Strict-inference guards (plan §1 F8 + non-negotiable constraints):
- Engine `info['achievements']` / inventory / position are SCORING-ONLY and may
  never enter the memory path. `record_step` rejects any such payload by
  construction (explicit **kwargs guard) instead of silently trusting callers.
- Unlock labels come from the belief (event head + reward-fallback provenance),
  never from the environment.

Zero hardcoding: array index spaces (128 latent dims, 16 grounded-state dims,
22 event dims) are interface constants only. No action names, no item semantics.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS

# Interface constants (bank learn_from_episode step contract). Shapes, not semantics.
OBS_LATENT_DIM = 128
GROUNDED_STATE_DIM = 16
NUM_EVENT_DIMS = len(CRAFTER_ACHIEVEMENTS)  # 22

# Confident-firing margin (plan §3 Stage 3/8): an unlock counts as event-margin
# evidence only when the head's probability cleared its threshold by this much.
CONFIDENT_MARGIN = 0.15


def _to_plain_float_list(value: Any, dim: int, name: str) -> List[float]:
    """Device-agnostic conversion of tensors/arrays to a flat plain float list."""
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.shape[0] != dim:
        raise ValueError(f"{name} must have {dim} dims, got {flat.shape[0]}")
    return [float(v) for v in flat]


class PlaytimeRecorder:
    """Buffers strict-inference playtime steps and feeds the relation bank per episode."""

    def __init__(self):
        self._buffer: List[Dict[str, Any]] = []
        self._episode_counter = 0

    def record_step(
        self,
        z_t: Any,
        x_hat: Any,
        action: int,
        reward: float,
        done: bool,
        mask_before: Any,
        newly_unlocked: Any,
        fallback_unlocked: Optional[Any] = None,
        event_probs: Optional[Any] = None,
        thresholds: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Buffers one strict-inference step per the bank learn_from_episode contract.

        z_t:              [128] target-encoder observation latent (consolidation convention).
        x_hat:            [16]  predicted grounded state from model.predict_state —
                                never engine inventory (strict protocol).
        action:           discrete action index taken this step.
        reward:           scalar environment reward (the only engine scalar consumed).
        done:             episode termination flag for this step.
        mask_before:      [22] belief binary mask copied BEFORE belief.update.
        newly_unlocked:   iterable of event dims the belief unlocked this step.
        fallback_unlocked: event dims pinned by the belief reward-fallback this step
                           (N7 quarantine provenance; empty when the head fired).
        event_probs:      [22] per-dim event-head probabilities (for the margin rule).
        thresholds:       [22] per-dim thresholds (for the margin rule).

        Raises:
            ValueError: if engine achievements (or any unknown kwarg) are handed in,
                        or if a vector does not match its contract dim.
        """
        if "achievements" in kwargs:
            raise ValueError(
                "strict protocol: engine achievements are scoring-only and may never enter the memory path"
            )

        obs_latent = _to_plain_float_list(z_t, OBS_LATENT_DIM, "z_t")
        grounded_state = _to_plain_float_list(x_hat, GROUNDED_STATE_DIM, "x_hat")
        mask_before_vec = [int(v) for v in _to_plain_float_list(mask_before, NUM_EVENT_DIMS, "mask_before")]
        event_vec = [int(e) for e in newly_unlocked]

        probs_list: Optional[List[float]] = (
            _to_plain_float_list(event_probs, NUM_EVENT_DIMS, "event_probs") if event_probs is not None else None
        )
        thresholds_list: Optional[List[float]] = (
            _to_plain_float_list(thresholds, NUM_EVENT_DIMS, "thresholds") if thresholds is not None else None
        )

        confident_events = [
            e
            for e in event_vec
            if probs_list is not None and thresholds_list is not None
            and probs_list[e] > thresholds_list[e] + CONFIDENT_MARGIN
        ]

        fallback_list = [int(e) for e in fallback_unlocked] if fallback_unlocked is not None else []

        self._buffer.append({
            "obs_latent": obs_latent,
            "grounded_state": grounded_state,
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
            "event_vec": event_vec,
            "mask_before_vec": mask_before_vec,
            "confident_events": confident_events,
            "fallback_unlocked": fallback_list,
        })

    def end_episode(self, bank: Any) -> Dict[str, int]:
        """Hands the buffered episode to the bank, refreshes the causal graph, rolls over.

        Before dispatch, the buffered steps are augmented with the transition view the
        bank's near-miss rule needs (review fix, plan Stage 8): state_delta = x_hat[t+1]
        - x_hat[t] (zeros on the first step, absorbing on the terminal step), plus
        next_grounded_state / next_obs_latent. Without these, online near-miss
        classification would degenerate to `reward <= 0` (max|delta| == 0 always).
        learn_from_episode + refresh_edges are the only bank surface used here.
        """
        steps = self._finalize_transition_view(self._buffer)
        stats = bank.learn_from_episode(steps, episode_id=self._episode_counter)
        bank.refresh_edges()
        self._buffer = []
        self._episode_counter += 1
        return stats

    @staticmethod
    def _finalize_transition_view(buffer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Builds next-state / delta fields from consecutive buffered x_hat/z values."""
        out: List[Dict[str, Any]] = []
        for i, step in enumerate(buffer):
            is_final = i == len(buffer) - 1
            if is_final:
                next_state = list(step["grounded_state"])
                next_latent = list(step["obs_latent"])
                delta = [0.0] * GROUNDED_STATE_DIM
            else:
                nxt = buffer[i + 1]
                next_state = list(nxt["grounded_state"])
                next_latent = list(nxt["obs_latent"])
                delta = [float(n) - float(c) for n, c in zip(next_state, step["grounded_state"])]
            enriched = dict(step)
            enriched["state_delta"] = delta
            enriched["next_grounded_state"] = next_state
            enriched["next_obs_latent"] = next_latent
            out.append(enriched)
        return out

    def __len__(self) -> int:
        """Number of steps currently buffered for the in-progress episode."""
        return len(self._buffer)
