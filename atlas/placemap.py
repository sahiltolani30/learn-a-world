"""TopoMap: visual keyframe topological memory using cosine latent matching.

Pure latent cosine matching in z-space. No coordinates, no odometry, no dead-reckoning.
Episode-local ring records the current trajectory; persistent store carries across episodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import numpy as np

from atlas.config import LATENT_DIM, RECURRENT_DIM, TopoMapConfig


@dataclass
class Keyframe:
    z: np.ndarray          # shape (128,)
    context: np.ndarray    # shape (256,)
    step: int
    visits: int = 1

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=np.float32).reshape(-1)
        self.context = np.asarray(self.context, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(self.z)
        if norm > 1e-8:
            self._normed_z = self.z / norm
        else:
            self._normed_z = self.z

    @property
    def normed_z(self) -> np.ndarray:
        return self._normed_z


def _cosine_similarity(normed_a: np.ndarray, normed_b: np.ndarray) -> float:
    return float(np.dot(normed_a, normed_b))


class TopoMap:
    """Maintains episode-local and cross-episode persistent visual keyframes."""

    def __init__(self, cfg: Optional[TopoMapConfig] = None):
        self.cfg = cfg or TopoMapConfig()
        self.episode_keyframes: Deque[Keyframe] = deque(maxlen=self.cfg.max_keyframes)
        self.persistent_keyframes: Deque[Keyframe] = deque(maxlen=self.cfg.persistent_capacity)
        self.last_keyframe_step: int = -99999

    def reset_episode(self) -> None:
        """Clears episode-local ring while preserving persistent cross-episode store."""
        self.episode_keyframes.clear()
        self.last_keyframe_step = -99999

    def observe(
        self,
        z: np.ndarray,
        context: np.ndarray,
        step: int,
    ) -> Optional[Keyframe]:
        """Observes current latent view. Adds keyframe on scene change."""
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        context = np.asarray(context, dtype=np.float32).reshape(-1)

        norm_z = np.linalg.norm(z)
        unit_z = z / norm_z if norm_z > 1e-8 else z

        if len(self.episode_keyframes) == 0:
            kf = Keyframe(z=z, context=context, step=step, visits=1)
            self.episode_keyframes.append(kf)
            self.persistent_keyframes.append(kf)
            self.last_keyframe_step = step
            return kf

        # Find nearest in episode-local memory
        sims = [
            _cosine_similarity(unit_z, kf.normed_z)
            for kf in self.episode_keyframes
        ]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        best_dist = 1.0 - best_sim

        if best_sim >= self.cfg.match_threshold:
            self.episode_keyframes[best_idx].visits += 1

        if (best_dist > self.cfg.scene_change_threshold and
                (step - self.last_keyframe_step) >= self.cfg.min_keyframe_interval):
            kf = Keyframe(z=z, context=context, step=step, visits=1)
            self.episode_keyframes.append(kf)
            self.persistent_keyframes.append(kf)
            self.last_keyframe_step = step
            return kf

        return None

    def _all_keyframes(self) -> List[Keyframe]:
        """Union of episode and persistent keyframes (references preserved)."""
        seen_ids = set()
        out: List[Keyframe] = []
        for kf in list(self.episode_keyframes) + list(self.persistent_keyframes):
            if id(kf) not in seen_ids:
                seen_ids.add(id(kf))
                out.append(kf)
        return out

    def nearest(self, z: np.ndarray, k: int = 1) -> List[Tuple[Keyframe, float]]:
        """Returns top k nearest keyframes by cosine similarity."""
        all_kf = self._all_keyframes()
        if not all_kf:
            return []

        z = np.asarray(z, dtype=np.float32).reshape(-1)
        norm_z = np.linalg.norm(z)
        unit_z = z / norm_z if norm_z > 1e-8 else z

        sims = [(_cosine_similarity(unit_z, kf.normed_z), kf) for kf in all_kf]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [(kf, sim) for sim, kf in sims[:k]]

    def match(self, z: np.ndarray) -> Optional[Tuple[Keyframe, float]]:
        """Returns the nearest keyframe if cosine similarity >= match_threshold, else None."""
        nearest = self.nearest(z, k=1)
        if not nearest:
            return None
        kf, sim = nearest[0]
        if sim >= self.cfg.match_threshold:
            return kf, sim
        return None

    def save(self, path: Union[str, Path]) -> None:
        """Saves persistent keyframes to npz archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not self.persistent_keyframes:
            z_arr = np.zeros((0, LATENT_DIM), dtype=np.float32)
            ctx_arr = np.zeros((0, RECURRENT_DIM), dtype=np.float32)
            step_arr = np.zeros((0,), dtype=np.int64)
            visits_arr = np.zeros((0,), dtype=np.int64)
        else:
            z_arr = np.stack([kf.z for kf in self.persistent_keyframes], axis=0)
            ctx_arr = np.stack([kf.context for kf in self.persistent_keyframes], axis=0)
            step_arr = np.array([kf.step for kf in self.persistent_keyframes], dtype=np.int64)
            visits_arr = np.array([kf.visits for kf in self.persistent_keyframes], dtype=np.int64)
        np.savez_compressed(
            p,
            z=z_arr,
            context=ctx_arr,
            step=step_arr,
            visits=visits_arr,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Loads persistent keyframes from npz archive."""
        p = Path(path)
        if not p.exists():
            return
        data = np.load(p)
        z_arr = data["z"]
        ctx_arr = data["context"]
        step_arr = data["step"]
        visits_arr = data["visits"]

        self.persistent_keyframes.clear()
        for i in range(len(z_arr)):
            kf = Keyframe(
                z=z_arr[i],
                context=ctx_arr[i],
                step=int(step_arr[i]),
                visits=int(visits_arr[i]),
            )
            self.persistent_keyframes.append(kf)
