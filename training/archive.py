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
    Archive of promising environment checkpoints for targeted restarts.
    Supports rarity-aware eviction, rarity-weighted sampling, and state restoration
    including belief mask and recurrent hidden state.
    """

    RARE_THRESHOLD: int = 10        # achievements with global count < 10 are considered rare
    ARCHIVE_SIZE: int = 50          # maximum number of archived states in the ring buffer
    STAGNATION_THRESHOLD: int = 20  # trigger restart after 20 consecutive stagnant episodes
    RESTART_PROB: float = 0.40      # 40% of episodes proactively restart from archive

    def __init__(
        self,
        rare_threshold: int = RARE_THRESHOLD,
        archive_size: int = ARCHIVE_SIZE,
        stagnation_threshold: int = STAGNATION_THRESHOLD,
        restart_prob: float = RESTART_PROB,
    ):
        self.rare_threshold = rare_threshold
        self.archive_size = archive_size
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = float(restart_prob)

        self._archive: List[Dict[str, Any]] = []
        self._write_idx: int = 0
        self.total_restarts: int = 0
        self.total_random_restarts: int = 0
        self.total_stagnation_restarts: int = 0
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
        mask: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        global_counts: Optional[np.ndarray] = None,
    ) -> None:
        """
        Adds a promising state snapshot to the archive.
        Stores snapshot, next_obs, full true mask at snapshot time, and belief GRU h.
        When full, replaces the entry with the highest global count (least rare),
        protecting the sole carrier of the rarest mask.
        """
        entry = {
            "seed": int(seed),
            "step": int(step),
            "snapshot": copy.deepcopy(snapshot),
            "obs": np.copy(obs),
            "mask": np.copy(mask) if mask is not None else np.zeros(22, dtype=np.uint8),
            "h": np.copy(h) if h is not None else np.zeros(256, dtype=np.float32),
            "unlocked_indices": list(unlocked_indices),
            "timestamp": time.time(),
            "times_sampled": 0,
        }

        if len(self._archive) < self.archive_size:
            self._archive.append(entry)
            return

        # Rarity-aware eviction when archive is at capacity
        if global_counts is None:
            self._archive[self._write_idx] = entry
            self._write_idx = (self._write_idx + 1) % self.archive_size
            return

        # 1. Determine rarest unlocked count for each entry in the archive
        entry_rarest_counts = []
        for e in self._archive:
            idxs = e.get("unlocked_indices", [])
            if len(idxs) > 0:
                c = min(int(global_counts[k]) for k in idxs)
            else:
                c = int(1e9)
            entry_rarest_counts.append(c)

        min_archive_count = min(entry_rarest_counts)
        sole_rarest_indices = [
            i for i, c in enumerate(entry_rarest_counts) if c == min_archive_count
        ]
        # Never evict the sole entry carrying the globally rarest mask
        protected_idx = sole_rarest_indices[0] if len(sole_rarest_indices) == 1 else -1

        candidates = [i for i in range(len(self._archive)) if i != protected_idx]
        if not candidates:
            candidates = list(range(len(self._archive)))

        # Evict the entry whose rarest unlocked index has the HIGHEST global count
        # Tiebreak: highest times_sampled, then oldest timestamp
        evict_idx = max(
            candidates,
            key=lambda i: (
                entry_rarest_counts[i],
                self._archive[i]["times_sampled"],
                -self._archive[i]["timestamp"],
            ),
        )

        self._archive[evict_idx] = entry

    def should_restart(
        self,
        stagnant_count: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[bool, str]:
        """
        Determines whether to trigger an archive restart:
        1. Stagnation trigger: stagnant_count >= stagnation_threshold.
        2. Proactive random restart: restart_prob (40% default) decided by RNG.
        Returns: (should_restart: bool, reason: 'none' | 'stagnation' | 'random')
        """
        if len(self._archive) == 0:
            return False, "none"

        count = self.episodes_since_novel if stagnant_count is None else stagnant_count
        if count >= self.stagnation_threshold:
            return True, "stagnation"

        r_val = rng.random() if rng is not None else random.random()
        if r_val < self.restart_prob:
            return True, "random"

        return False, "none"

    def sample(self, global_counts: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Samples an archived state for restart.
        Weighted by 1 / (global_count of rarest index + 1), with least-sampled as tiebreak.
        """
        if not self._archive:
            return None

        if global_counts is None:
            sample_counts = np.array([e["times_sampled"] for e in self._archive], dtype=np.float32)
            weights = 1.0 / (sample_counts + 1.0)
        else:
            weights = []
            for e in self._archive:
                idxs = e.get("unlocked_indices", [])
                if len(idxs) > 0:
                    c = min(int(global_counts[k]) for k in idxs)
                else:
                    c = int(1e6)
                # Weight by 1 / (count + 1), modulated by least-sampled
                w = (1.0 / (float(c) + 1.0)) / ((float(e["times_sampled"]) + 1.0) ** 0.2)
                weights.append(w)
            weights = np.array(weights, dtype=np.float32)

        probs = weights / np.sum(weights)
        idx = int(np.random.choice(len(self._archive), p=probs))
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

    def get_mask_histogram(self, global_counts: Optional[np.ndarray] = None) -> Dict[str, int]:
        """Computes histogram of entries by their rarest unlocked index."""
        hist: Dict[str, int] = {}
        for e in self._archive:
            idxs = e.get("unlocked_indices", [])
            if len(idxs) > 0:
                if global_counts is not None:
                    rarest_idx = min(idxs, key=lambda k: global_counts[k])
                else:
                    rarest_idx = min(idxs)
                key = f"ach_{rarest_idx:02d}"
            else:
                key = "ach_none"
            hist[key] = hist.get(key, 0) + 1
        return hist

    def summary(self, global_counts: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Returns diagnostic summary of archive contents."""
        return {
            "size": len(self._archive),
            "max_size": self.archive_size,
            "total_restarts": self.total_restarts,
            "total_random_restarts": self.total_random_restarts,
            "total_stagnation_restarts": self.total_stagnation_restarts,
            "episodes_since_novel": self.episodes_since_novel,
            "mask_histogram": self.get_mask_histogram(global_counts),
            "archived_achievements": [e["unlocked_indices"] for e in self._archive],
        }

