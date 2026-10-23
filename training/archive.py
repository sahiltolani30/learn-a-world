"""
training/archive.py — Archive Restarter for Exploration Jump-Starts (Tranche T2 Part 4)

Maintains a bounded ring buffer of 'interesting' environment states (states where rare
achievements were unlocked). If curiosity collection stagnates (no novel events for K episodes),
an archived state is sampled and restored so collection resumes from an advanced frontier.

Standing Rules:
- All achievement indexing is purely integer-based (0 to 21).
- ZERO achievement names in code logic.
- Purely count-based thresholds.
"""

import copy
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ArchiveRestarter:
    """
    Ring-buffer archive of promising environment checkpoints for targeted restarts.
    """

    RARE_THRESHOLD: int = 10     # achievements with global count < 10 are considered rare
    ARCHIVE_SIZE: int = 50       # maximum number of archived states in the ring buffer
    STAGNATION_THRESHOLD: int = 20  # trigger restart after 20 consecutive stagnant episodes

    def __init__(
        self,
        rare_threshold: int = RARE_THRESHOLD,
        archive_size: int = ARCHIVE_SIZE,
        stagnation_threshold: int = STAGNATION_THRESHOLD,
    ):
        self.rare_threshold = rare_threshold
        self.archive_size = archive_size
        self.stagnation_threshold = stagnation_threshold

        self._archive: List[Dict[str, Any]] = []
        self._write_idx: int = 0
        self.total_restarts: int = 0
        self.episodes_since_novel: int = 0

    def is_interesting(self, event_mask: np.ndarray, global_counts: np.ndarray) -> bool:
        """
        Determines if an event contains any achievement that is currently rare
        (global count < rare_threshold).
        event_mask: [num_achievements] binary array of newly unlocked events.
        global_counts: [num_achievements] array of cumulative global counts.
        """
        unlocked = np.where(event_mask > 0)[0]
        if len(unlocked) == 0:
            return False
        # Check if ANY unlocked achievement has count < rare_threshold
        rare_unlocked = [idx for idx in unlocked if global_counts[idx] < self.rare_threshold]
        return len(rare_unlocked) > 0

    def get_rare_indices(self, event_mask: np.ndarray, global_counts: np.ndarray) -> List[int]:
        """Returns list of rare achievement indices unlocked in event_mask."""
        unlocked = np.where(event_mask > 0)[0]
        return [int(idx) for idx in unlocked if global_counts[idx] < self.rare_threshold]

    def add(
        self,
        seed: int,
        step: int,
        snapshot: Dict[str, Any],
        obs: np.ndarray,
        unlocked_indices: List[int],
    ) -> None:
        """
        Adds a promising state snapshot to the archive.
        """
        entry = {
            "seed": int(seed),
            "step": int(step),
            "snapshot": copy.deepcopy(snapshot),
            "obs": np.copy(obs),
            "unlocked_indices": list(unlocked_indices),
            "timestamp": time.time(),
            "times_sampled": 0,
        }

        if len(self._archive) < self.archive_size:
            self._archive.append(entry)
        else:
            self._archive[self._write_idx] = entry
            self._write_idx = (self._write_idx + 1) % self.archive_size

    def should_restart(self, episodes_since_novel: Optional[int] = None) -> bool:
        """
        Checks if the stagnation threshold has been reached and the archive has candidate states.
        """
        count = self.episodes_since_novel if episodes_since_novel is None else episodes_since_novel
        return (count >= self.stagnation_threshold) and (len(self._archive) > 0)

    def sample(self) -> Optional[Dict[str, Any]]:
        """
        Samples an archived state for restart. Prefers states with fewer past samples (UCB-style).
        """
        if not self._archive:
            return None

        # Inverse-sample weighting: prioritize states that haven't been reused frequently
        sample_counts = np.array([e["times_sampled"] for e in self._archive], dtype=np.float32)
        weights = 1.0 / (sample_counts + 1.0)
        probs = weights / np.sum(weights)

        idx = np.random.choice(len(self._archive), p=probs)
        chosen = self._archive[idx]
        chosen["times_sampled"] += 1
        self.total_restarts += 1
        self.episodes_since_novel = 0

        return chosen

    def record_episode_novelty(self, has_novel_event: bool) -> None:
        """Updates internal stagnation tracker after an episode."""
        if has_novel_event:
            self.episodes_since_novel = 0
        else:
            self.episodes_since_novel += 1

    def __len__(self) -> int:
        return len(self._archive)

    def summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of archive contents."""
        return {
            "size": len(self._archive),
            "max_size": self.archive_size,
            "total_restarts_triggered": self.total_restarts,
            "episodes_since_novel": self.episodes_since_novel,
            "archived_achievements": [e["unlocked_indices"] for e in self._archive],
        }
