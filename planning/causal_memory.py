#!/usr/bin/env python3
"""
planning/causal_memory.py — Grounded Persistent Memory Architecture (Phase 2)
==============================================================================

Implements Section 6 (Phase 2) of the JEPA Causal Chain Roadmap:
- 2A: Grounded belief tracking in latent & state estimation space.
- 2B: Two-layer long-term storage:
      1. EpisodicEvidenceStore: Raw interaction trajectories (z_t, x_hat_t, a_t, r_t, d_t, G_t).
      2. RelationHypothesisBank: Learned transition prototypes with empirical preconditions,
         effect deltas, success rates, risk, and Bayesian counterexample revision.
- Strict Inference Invariant: 100% learned from empirical observations & model predictions.
  Zero access to engine inventory, achievements, maps, coordinates, or human crafting recipes.
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_ACTIONS = 17
STATE_DIM = 16  # Canonical 16-D resource/vital state space


@dataclass
class EpisodicEvidence:
    """Raw trajectory transition evidence from actual agent gameplay."""
    obs_latent: List[float]       # [128] Visual latent z_t
    grounded_state: List[float]   # [16] Learned state estimate x_hat_t
    action: int                   # Action taken a_t in [0..16]
    reward: float                 # Observed scalar reward r_t
    done: bool                    # Termination boundary d_t
    next_obs_latent: List[float]  # [128] Next visual latent z_{t+1}
    next_grounded_state: List[float] # [16] Next learned state estimate x_hat_{t+1}
    state_delta: List[float]      # [16] Measured state delta x_{t+1} - x_t
    return_g: float               # Discounted cumulative return G_t
    episode_id: int               # Unique episode identifier
    step_id: int                  # Step index within episode
    confidence: float = 1.0       # Model confidence / inverse uncertainty


class EpisodicEvidenceStore:
    """
    Layer 1: Experience Evidence Store.
    Preserves verified gameplay trajectories without human filtering or recipe bias.
    """

    def __init__(self, max_capacity: int = 50000, gamma: float = 0.99):
        self.max_capacity = max_capacity
        self.gamma = gamma
        self.entries: List[EpisodicEvidence] = []
        self._current_episode_buffer: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def record_step(
        self,
        obs_latent: Union[torch.Tensor, np.ndarray, List[float]],
        grounded_state: Union[torch.Tensor, np.ndarray, List[float]],
        action: int,
        reward: float,
        done: bool,
        next_obs_latent: Union[torch.Tensor, np.ndarray, List[float]],
        next_grounded_state: Union[torch.Tensor, np.ndarray, List[float]],
        episode_id: int,
        step_id: int,
        confidence: float = 1.0,
    ) -> None:
        """Buffers step transition for retroactive return calculation at episode boundary."""
        def _to_list(val: Any) -> List[float]:
            if isinstance(val, torch.Tensor):
                return val.detach().cpu().flatten().numpy().tolist()
            if isinstance(val, np.ndarray):
                return val.flatten().tolist()
            return list(val)

        z = _to_list(obs_latent)
        x = _to_list(grounded_state)
        next_z = _to_list(next_obs_latent)
        next_x = _to_list(next_grounded_state)
        delta = [nx - cur for nx, cur in zip(next_x, x)]

        self._current_episode_buffer.append({
            "obs_latent": z,
            "grounded_state": x,
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
            "next_obs_latent": next_z,
            "next_grounded_state": next_x,
            "state_delta": delta,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "confidence": float(confidence),
        })

        if done:
            self.commit_episode()

    def commit_episode(self) -> int:
        """
        Backpropagates true Monte Carlo returns across the finished episode trajectory:
        G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
        and appends verified evidence to long-term storage.
        """
        if not self._current_episode_buffer:
            return 0

        T = len(self._current_episode_buffer)
        rewards = [step["reward"] for step in self._current_episode_buffer]
        returns = [0.0] * T
        g = 0.0
        for t in reversed(range(T)):
            g = rewards[t] + self.gamma * g
            returns[t] = g

        new_count = 0
        for t, step in enumerate(self._current_episode_buffer):
            evidence = EpisodicEvidence(
                obs_latent=step["obs_latent"],
                grounded_state=step["grounded_state"],
                action=step["action"],
                reward=step["reward"],
                done=step["done"],
                next_obs_latent=step["next_obs_latent"],
                next_grounded_state=step["next_grounded_state"],
                state_delta=step["state_delta"],
                return_g=round(returns[t], 4),
                episode_id=step["episode_id"],
                step_id=step["step_id"],
                confidence=step["confidence"],
            )
            self._append_evidence(evidence)
            new_count += 1

        self._current_episode_buffer.clear()
        return new_count

    def _append_evidence(self, evidence: EpisodicEvidence) -> None:
        if len(self.entries) >= self.max_capacity:
            # FIFO eviction for oldest low-return evidence
            self.entries.pop(0)
        self.entries.append(evidence)

    def query_by_action(self, action: int) -> List[EpisodicEvidence]:
        """Returns all stored evidence where specified action was executed."""
        return [e for e in self.entries if e.action == action]

    def query_resource_transitions(self, resource_dim: int, min_delta: float = 0.5) -> List[EpisodicEvidence]:
        """Returns evidence where a specific resource delta was observed (e.g. wood gathered)."""
        return [e for e in self.entries if e.state_delta[resource_dim] >= min_delta]

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes evidence store to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "phase2_evidence_v1",
            "max_capacity": self.max_capacity,
            "gamma": self.gamma,
            "entries": [asdict(e) for e in self.entries],
        }
        with open(p, "w") as f:
            json.dump(data, f)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads evidence store from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Evidence store not found: {p}")
        with open(p, "r") as f:
            data = json.load(f)
        self.max_capacity = data.get("max_capacity", 50000)
        self.gamma = data.get("gamma", 0.99)
        self.entries = [EpisodicEvidence(**item) for item in data.get("entries", [])]


@dataclass
class TransitionPrototype:
    """
    Learned Causal Relation Hypothesis.
    Represents an empirical transition prototype discovered from trajectory evidence.
    """
    prototype_id: str
    action: int                                # Action button [0..16]
    precondition_key: List[float]              # [128] Visual cue centroid (what is seen before)
    precondition_state: List[float]            # [16] Empirical minimum required resource state
    effect_delta: List[float]                  # [16] Discovered resource delta (production/consumption)
    postcondition_key: List[float]             # [128] Post-action visual latent centroid
    success_count: int                         # Number of times action succeeded
    failure_count: int                         # Number of counterexamples (attempts with 0 progress)
    mean_return: float                         # Realized future return G*
    mean_duration: float = 1.0                 # Expected steps to complete
    confidence: float = 0.5                    # Bayesian confidence score in [0, 1]

    @property
    def success_probability(self) -> float:
        """Laplace-smoothed empirical success probability."""
        return (self.success_count + 1.0) / (self.success_count + self.failure_count + 2.0)

    def is_applicable(self, current_state: Union[torch.Tensor, np.ndarray, List[float]], tolerance: float = 0.25) -> bool:
        """
        Tests if current state satisfies empirical preconditions.
        current_state: 16-D resource state.
        Preconditions require that available items meet the discovered requirement (x >= x_pre - tolerance).
        """
        if isinstance(current_state, torch.Tensor):
            curr = current_state.detach().cpu().flatten().numpy()
        else:
            curr = np.array(current_state, dtype=np.float32).flatten()
        pre = np.array(self.precondition_state, dtype=np.float32)
        # Check that player possesses required non-zero items (e.g. wood, pickaxe)
        req_indices = np.where(pre > 0.5)[0]
        if len(req_indices) == 0:
            return True
        return bool(np.all(curr[req_indices] >= (pre[req_indices] - tolerance)))


class RelationHypothesisBank:
    """
    Layer 2: Relation Hypothesis Bank (Learned Transition Prototypes).
    Consolidates raw experience evidence into learned causal prototypes with
    Bayesian confidence tracking and counterexample revision.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.prototypes: Dict[str, TransitionPrototype] = {}

    def __len__(self) -> int:
        return len(self.prototypes)

    def get_prototypes_for_action(self, action: int) -> List[TransitionPrototype]:
        """Returns all candidate prototypes for a discrete action."""
        return [p for p in self.prototypes.values() if p.action == action]

    def consolidate_from_evidence(
        self,
        evidence_store: EpisodicEvidenceStore,
        min_cluster_size: int = 2,
        cos_sim_threshold: float = 0.85,
    ) -> int:
        """
        Consolidates raw episodic evidence into learned transition prototypes.
        Groups similar visual/action/outcome experiences and derives empirical
        preconditions and effect deltas without human recipes.
        """
        if len(evidence_store) == 0:
            return 0

        created_or_updated = 0
        for act in range(NUM_ACTIONS):
            ev_list = evidence_store.query_by_action(act)
            if not ev_list:
                continue

            # Separate productive breakthroughs (reward > 0 or resource delta > 0.25) from idle transitions
            breakthroughs = [
                e for e in ev_list
                if e.reward > 0.0 or any(d > 0.25 for d in e.state_delta)
            ]
            failures = [
                e for e in ev_list
                if e.reward <= 0.0 and all(abs(d) < 0.25 for d in e.state_delta)
            ]

            if not breakthroughs:
                continue

            # Cluster breakthroughs by visual cue cosine similarity
            clusters: List[List[EpisodicEvidence]] = []
            for b in breakthroughs:
                placed = False
                z_b = np.array(b.obs_latent, dtype=np.float32)
                z_b_norm = z_b / max(float(np.linalg.norm(z_b)), 1e-6)

                for cluster in clusters:
                    z_c = np.array(cluster[0].obs_latent, dtype=np.float32)
                    z_c_norm = z_c / max(float(np.linalg.norm(z_c)), 1e-6)
                    sim = float(np.dot(z_b_norm, z_c_norm))
                    if sim >= cos_sim_threshold:
                        cluster.append(b)
                        placed = True
                        break
                if not placed:
                    clusters.append([b])

            # Build or update prototype for each cluster exceeding min_cluster_size
            for c_idx, cluster in enumerate(clusters):
                if len(cluster) < min_cluster_size:
                    continue

                proto_id = f"proto_act{act}_cluster{c_idx}"

                # Calculate cluster centroids
                z_pre_mean = np.mean([e.obs_latent for e in cluster], axis=0)
                z_pre_norm = (z_pre_mean / max(float(np.linalg.norm(z_pre_mean)), 1e-6)).tolist()

                z_post_mean = np.mean([e.next_obs_latent for e in cluster], axis=0)
                z_post_norm = (z_post_mean / max(float(np.linalg.norm(z_post_mean)), 1e-6)).tolist()

                # Empirical minimum required preconditions (e.g. minimum wood present in cluster)
                pre_min = np.min([e.grounded_state for e in cluster], axis=0)
                # Keep significant preconditions (item >= 0.5)
                precond_state = [float(v) if v >= 0.5 else 0.0 for v in pre_min]

                # Mean observed effect delta
                delta_mean = np.mean([e.state_delta for e in cluster], axis=0).tolist()
                mean_ret = float(np.mean([e.return_g for e in cluster]))

                # Count counterexamples matching the visual cue where action failed
                neg_count = 0
                for f in failures:
                    z_f = np.array(f.obs_latent, dtype=np.float32)
                    z_f_norm = z_f / max(float(np.linalg.norm(z_f)), 1e-6)
                    sim = float(np.dot(np.array(z_pre_norm), z_f_norm))
                    if sim >= cos_sim_threshold:
                        neg_count += 1

                pos_count = len(cluster)
                # Bayesian confidence based on sample size and signal-to-noise
                conf = float(pos_count / (pos_count + neg_count + 1.0))

                self.prototypes[proto_id] = TransitionPrototype(
                    prototype_id=proto_id,
                    action=act,
                    precondition_key=z_pre_norm,
                    precondition_state=precond_state,
                    effect_delta=delta_mean,
                    postcondition_key=z_post_norm,
                    success_count=pos_count,
                    failure_count=neg_count,
                    mean_return=mean_ret,
                    confidence=conf,
                )
                created_or_updated += 1

        return created_or_updated

    def register_interaction_outcome(
        self,
        prototype_id: str,
        succeeded: bool,
    ) -> Optional[TransitionPrototype]:
        """
        Acceptance Criterion 3 (Revision):
        Contradictory observations lower confidence or revise a relation
        instead of reinforcing a false belief.
        """
        if prototype_id not in self.prototypes:
            return None
        proto = self.prototypes[prototype_id]
        if succeeded:
            proto.success_count += 1
        else:
            proto.failure_count += 1
        # Recalculate confidence
        proto.confidence = float(proto.success_count / (proto.success_count + proto.failure_count + 1.0))
        return proto

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes relation hypothesis bank to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "phase2_relation_bank_v1",
            "prototypes": {k: asdict(v) for k, v in self.prototypes.items()},
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads relation hypothesis bank from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Relation hypothesis bank not found: {p}")
        with open(p, "r") as f:
            data = json.load(f)
        self.prototypes = {
            k: TransitionPrototype(**v)
            for k, v in data.get("prototypes", {}).items()
        }
