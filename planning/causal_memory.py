#!/usr/bin/env python3
"""
planning/causal_memory.py — Grounded Persistent Memory Architecture (Phase 2, v2)
==================================================================================

Implements the Phase 2 Causal Memory Redesign (docs/PHASE2_MEMORY_REDESIGN_PLAN.md,
Stages 3+4): the Causal Card schema v2, event-grounded consolidation v2, incremental
online learning (match-and-update, never re-cluster), and the causal graph
(produces_need + enabling_lift edges).

Layers:
  1. EpisodicEvidenceStore: raw interaction trajectories
     (z_t, x_hat_t, a_t, r_t, d_t, G_t, event_vec_t, mask_before_t, source).
  2. RelationHypothesisBank: learned transition prototypes ("cards") — one card per
     (action, unlocked event) or (action) for event-free productive transitions —
     plus a causal graph over cards.

Invariants (non-negotiable, see plan §"Non-negotiable constraints"):
  - ZERO hardcoding: no action names, no item semantics, no recipe amounts in logic.
    Array index spaces (17 actions, 16 state dims, 22 achievements) are interface
    constants only.
  - Strict inference: everything is learned from empirical observations and model
    predictions. Zero access to engine inventory, achievements, maps, coordinates,
    or human crafting recipes. Unlock labels are belief-derived (event head +
    reward fallback); fallback-pinned dims are guesses, not observations, and are
    quarantined (may create hypothesis cards, may never confirm).
  - mean_return is a DIAGNOSTIC field only; no consumer may rank by it.
"""

from collections import deque
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS

REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_ACTIONS = 17
STATE_DIM = 16  # Canonical 16-D resource/vital state space
NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22

# 80% two-sided Wilson score interval (z of the standard normal at 90% quantile).
WILSON_Z_80 = 1.2816

# Visual cosine similarity threshold for clustering / member matching.
COS_SIM_THRESHOLD = 0.85

# Consolidation classification thresholds (plan §3 Stage 3).
PRODUCTIVE_DELTA = 0.35     # max|state_delta| >= this => productive event-free success
NEAR_MISS_DELTA = 0.35      # max|state_delta| < this (and no unlock, reward <= 0) => near-miss
NEED_QUANTILE = 0.15        # success-state quantile used as the need threshold q_d
NEED_MIN_LEVEL = 0.45       # q_d below this is noise -> dim skipped
NEED_AMOUNT_RANGE = (0.5, 4.0)  # clip range for stored need amounts
NEED_MIN_SUPPORT = 3        # min successes at/above q_d for the dim to be testable
NEED_MIN_GAP = 0.20         # s_d - f_d must exceed this
NEED_MIN_FAILURES = 2       # |F| < 2 -> dim untestable
BH_FDR_Q = 0.10             # Benjamini-Hochberg FDR level across the 16 dims
PRODUCES_THRESHOLD = 0.25   # mean delta >= +0.25 -> produces; <= -0.25 -> consumes

# Enabling-lift edge gates (plan §3 Stage 4).
ENABLING_MIN_LIFT = 1.5
ENABLING_MIN_SUPPORT = 5

# Hypothesis/confirmation gate: weighted successes (offline 1.0, online 0.5) below
# this keep a card in "hypothesis" status with needs withheld.
CONFIRM_WEIGHTED_SUCCESSES = 3.0

# Online evidence ring-buffer cap.
ONLINE_EVIDENCE_CAP = 200_000

BANK_SCHEMA_VERSION = "phase2_relation_bank_v2"
EVIDENCE_SCHEMA_VERSION = "phase2_evidence_v2"
CARD_SCHEMA_VERSION = "phase2_card_v2"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_interval(k: float, n: float, z: float = WILSON_Z_80) -> Tuple[float, float]:
    """80% two-sided Wilson score interval for a binomial proportion.

    Returns (lower_bound, upper_bound). n <= 0 -> (0.0, 1.0) (no information).
    Degenerate proportions stay inside [0, 1]: p = 0 with n > 0 gives
    (0.0, ub) with ub > 0; p = 1 gives (lb, 1.0) with lb < 1.
    """
    k = float(k)
    n = float(n)
    if n <= 0:
        return (0.0, 1.0)
    k = min(max(k, 0.0), n)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lb = max(0.0, center - half)
    ub = min(1.0, center + half)
    if p <= 0.0:
        lb = 0.0  # exact boundary (floating point dust otherwise)
    if p >= 1.0:
        ub = 1.0
    return (lb, ub)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_proportion_pvalue_one_sided(k1: float, n1: float, k2: float, n2: float) -> float:
    """One-sided (p1 > p2) pooled two-proportion z-test upper-tail p-value."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se <= 0.0:
        return 0.0 if p1 > p2 else 1.0
    z = (p1 - p2) / se
    return float(min(1.0, max(0.0, 1.0 - _norm_cdf(z))))


def _bh_fdr_reject(pvals: Sequence[Tuple[int, float]], q: float) -> List[int]:
    """Benjamini-Hochberg step-up. pvals: [(dim, p_value)]. Returns rejected dims."""
    if not pvals:
        return []
    m = len(pvals)
    ordered = sorted(pvals, key=lambda t: (t[1], t[0]))
    keep_upto = 0
    for idx, (_dim, p) in enumerate(ordered, start=1):
        if p <= q * idx / m:
            keep_upto = idx
    return [dim for dim, _p in ordered[:keep_upto]]


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float64).flatten()
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return [0.0] * arr.size
    return (arr / norm).tolist()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float64).flatten()
    vb = np.asarray(b, dtype=np.float64).flatten()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _mean_pairwise_cosine(vecs: Sequence[Sequence[float]], max_sample: int = 128) -> float:
    """Mean pairwise intra-cluster cosine; 1.0 for single-member clusters."""
    if len(vecs) <= 1:
        return 1.0
    V = torch.tensor(vecs, dtype=torch.float32)
    if len(V) > max_sample:
        # Subsample for fast dispersion estimation on large clusters
        idx = torch.linspace(0, len(V) - 1, steps=max_sample).long()
        V = V[idx]
    V = torch.nn.functional.normalize(V, p=2, dim=-1)
    gram = torch.mm(V, V.t())
    triu_indices = torch.triu_indices(len(V), len(V), offset=1)
    sims = gram[triu_indices[0], triu_indices[1]]
    return float(sims.mean().item()) if len(sims) > 0 else 1.0


def _greedy_cosine_subclusters(latents: Sequence[Sequence[float]], threshold: float) -> List[List[int]]:
    """Greedy clustering by cosine >= threshold against each cluster's seed (first member).

    Returns lists of indices into `latents`, in first-appearance order.
    """
    if not latents:
        return []
    Z = torch.tensor(latents, dtype=torch.float32)
    Z = torch.nn.functional.normalize(Z, p=2, dim=-1)
    N = Z.shape[0]
    clusters: List[List[int]] = []
    seeds: List[torch.Tensor] = []
    for i in range(N):
        zi = Z[i]
        if not seeds:
            clusters.append([i])
            seeds.append(zi)
            continue
        S = torch.stack(seeds, dim=0)  # [K, 128]
        sims = torch.mv(S, zi)
        max_val, max_idx = sims.max(dim=0)
        if max_val.item() >= threshold:
            clusters[max_idx.item()].append(i)
        else:
            clusters.append([i])
            seeds.append(zi)
    return clusters


def _to_flat_list(val: Any) -> List[float]:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().flatten().numpy().tolist()
    if isinstance(val, np.ndarray):
        return val.flatten().tolist()
    return [float(v) for v in val]


# ---------------------------------------------------------------------------
# Layer 1: Episodic evidence
# ---------------------------------------------------------------------------

@dataclass
class EpisodicEvidence:
    """Raw trajectory transition evidence from actual agent gameplay."""
    obs_latent: List[float]       # [128] Visual latent z_t
    grounded_state: List[float]   # [16] Learned state estimate x_hat_t
    action: int                   # Action taken a_t in [0..16]
    reward: float                 # Observed scalar reward r_t
    done: bool                    # Termination boundary d_t
    next_obs_latent: List[float]  # [128] Next visual latent z_{t+1}
    next_grounded_state: List[float]  # [16] Next learned state estimate x_hat_{t+1}
    state_delta: List[float]      # [16] Measured state delta x_{t+1} - x_t
    return_g: float               # Discounted cumulative return G_t
    episode_id: int               # Unique episode identifier
    step_id: int                  # Step index within episode
    confidence: float = 1.0       # Model confidence / inverse uncertainty
    event_vec: Optional[List[int]] = None        # [22] first-time unlock u_t = events*(1-mask_before)
    mask_before_vec: Optional[List[int]] = None  # [22] dims already unlocked before this step
    source: str = "offline"                      # "offline" | "online"


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
        event_vec: Optional[Union[torch.Tensor, np.ndarray, Sequence[int]]] = None,
        mask_before_vec: Optional[Union[torch.Tensor, np.ndarray, Sequence[int]]] = None,
        source: str = "offline",
    ) -> None:
        """Buffers step transition for retroactive return calculation at episode boundary.

        event_vec / mask_before_vec are belief-derived [22] binary vectors:
        u_t = events * (1 - mask_before) is the first-time-unlock decomposition.
        source records evidence provenance ("offline" shards | "online" playtime).
        """
        def _to_int_list(val: Any) -> Optional[List[int]]:
            if val is None:
                return None
            return [int(v) for v in _to_flat_list(val)]

        z = _to_flat_list(obs_latent)
        x = _to_flat_list(grounded_state)
        next_z = _to_flat_list(next_obs_latent)
        next_x = _to_flat_list(next_grounded_state)
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
            "event_vec": _to_int_list(event_vec),
            "mask_before_vec": _to_int_list(mask_before_vec),
            "source": str(source),
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
                event_vec=step.get("event_vec"),
                mask_before_vec=step.get("mask_before_vec"),
                source=step.get("source", "offline"),
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
        """Serializes evidence store to disk (schema phase2_evidence_v2)."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": EVIDENCE_SCHEMA_VERSION,
            "max_capacity": self.max_capacity,
            "gamma": self.gamma,
            "entries": [asdict(e) for e in self.entries],
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, p)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads evidence store from disk. Tolerant of v1 files (missing per-entry
        event/mask/source keys fall back to None / "offline")."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Evidence store not found: {p}")
        with open(p, "r") as f:
            data = json.load(f)
        self.max_capacity = data.get("max_capacity", 50000)
        self.gamma = data.get("gamma", 0.99)
        valid_fields = {f.name for f in dataclass_fields(EpisodicEvidence)}
        entries = []
        for item in data.get("entries", []):
            filtered = {k: v for k, v in item.items() if k in valid_fields}
            entries.append(EpisodicEvidence(**filtered))
        self.entries = entries


# ---------------------------------------------------------------------------
# Layer 2: Causal cards (transition prototypes, schema v2)
# ---------------------------------------------------------------------------

@dataclass
class TransitionPrototype:
    """
    Learned Causal Card — one card per (action, unlocked event); event-free
    productive transitions group at the action level with event_dim=None.

    NEEDS dims survive an enrichment test (successes vs matched near-misses,
    Wilson separation + effect gap + BH-FDR); amounts are clipped success
    quantiles. PRODUCES/CONSUMES come from the mean state delta. unlock_prob is
    the Laplace-smoothed P(event fires | applied). mean_return is DIAGNOSTIC ONLY.
    """
    prototype_id: str
    action: int                                # Action button [0..16]
    precondition_key: List[float]              # [128] Visual cue centroid (what is seen before)
    precondition_state: List[float]            # [16] Sparse needs vector (legacy interface)
    effect_delta: List[float]                  # [16] Discovered resource delta (production/consumption)
    postcondition_key: List[float]             # [128] Post-action visual latent centroid
    success_count: float                       # Weighted-float success tally (see note)
    failure_count: float                       # Weighted-float failure tally (see note)
    mean_return: float                         # DIAGNOSTIC ONLY — never used for ranking
    mean_duration: float = 1.0                 # Expected steps to complete
    confidence: float = 0.5                    # max(0.05, wilson_lb of effective stats)
    achievement_delta: List[float] = field(default_factory=lambda: [0.0] * NUM_ACHIEVEMENTS)  # [22] soft
    event_dim: Optional[int] = None            # 0..21 grounded unlock; None = event-free
    needs: List[List[float]] = field(default_factory=list)      # [[dim, amount], ...]
    produces: List[List[float]] = field(default_factory=list)   # [[dim, amount], ...] (delta >= +0.25)
    consumes: List[List[float]] = field(default_factory=list)   # [[dim, |delta|], ...] (delta <= -0.25)
    unlock_prob: float = 0.0
    wilson_lb: float = 0.0
    visual_dispersion: float = 1.0             # mean pairwise intra-cluster cosine SIMILARITY:
                                               # HIGH = homogeneous cue (informative WHEN);
                                               # LOW = fired across varied scenes (cue uninformative).
    status: str = "confirmed"                  # "hypothesis" if weighted successes < 3
    success_offline: float = 0.0
    success_online: float = 0.0
    failure_offline: float = 0.0
    failure_online: float = 0.0
    created_from: str = "offline"
    schema_version: str = CARD_SCHEMA_VERSION

    # Counting convention: success_count/failure_count are raw float tallies that
    # always include every observation at weight 1.0; the offline/online WEIGHTING
    # (offline 1.0, online 0.5) is applied where it is consumed — in
    # RelationHypothesisBank.effective_success_stats and the weighted-successes
    # status gate. success_offline/online (+ failure) track provenance.

    @property
    def success_probability(self) -> float:
        """Laplace-smoothed empirical success probability over the float tallies."""
        return (self.success_count + 1.0) / (self.success_count + self.failure_count + 2.0)

    @property
    def attempt_count(self) -> int:
        """Total recorded attempts for this card (truncated to int)."""
        return int(self.success_count + self.failure_count)

    @property
    def weighted_successes(self) -> float:
        """Evidence-weighted successes: offline weight 1.0, online weight 0.5."""
        return float(self.success_offline) + 0.5 * float(self.success_online)

    def marginal_reward(self, achievement_mask: Union[torch.Tensor, np.ndarray, Sequence[float]]) -> float:
        """
        One-time-aware marginal value of applying this card, given the current
        achievement-unlock mask (m_bin, [22], per-dim in [0, 1]).

        Event-grounded cards: unlock_prob * (1 - mask[event_dim]) — the expected
        value decays to zero once the event is already unlocked (one-time aware).
        Event-free cards: 0.0 — their value arises only from downstream demand in
        the causal graph (they are chain steps, never top-level targets).
        """
        if self.event_dim is None:
            return 0.0
        if isinstance(achievement_mask, torch.Tensor):
            mask = achievement_mask.detach().cpu().flatten().numpy()
        else:
            mask = np.asarray(achievement_mask, dtype=np.float64).flatten()
        e = int(self.event_dim)
        if e < 0 or e >= len(mask):
            return 0.0
        already = float(np.clip(mask[e], 0.0, 1.0))
        return float(self.unlock_prob * (1.0 - already))

    def is_applicable(self, current_state: Union[torch.Tensor, np.ndarray, List[float]], tolerance: float = 0.25) -> bool:
        """
        Tests if current state satisfies empirical preconditions (the sparse needs
        vector). current_state: 16-D resource state. A need (d, amt) is satisfied
        when current_state[d] >= amt - tolerance.
        """
        if isinstance(current_state, torch.Tensor):
            curr = current_state.detach().cpu().flatten().numpy()
        else:
            curr = np.array(current_state, dtype=np.float32).flatten()
        pre = np.array(self.precondition_state, dtype=np.float32)
        req_indices = np.where(pre > 1e-8)[0]
        if len(req_indices) == 0:
            return True
        return bool(np.all(curr[req_indices] >= (pre[req_indices] - tolerance)))


@dataclass
class RelationEdge:
    """Directed edge in the causal graph over cards."""
    src_id: str
    dst_id: str
    kind: str                                  # "produces_need" | "enabling_lift"
    dim: Optional[int] = None                  # produces_need: which 16-d dim
    lift: float = 1.0
    support: int = 0
    mean_dt: float = 0.0


# ---------------------------------------------------------------------------
# Layer 2: Relation hypothesis bank (cards + causal graph + online learning)
# ---------------------------------------------------------------------------

class RelationHypothesisBank:
    """
    Layer 2: Relation Hypothesis Bank (Learned Causal Cards + Causal Graph).

    Offline consolidation (consolidate_from_evidence) clusters event-grounded
    successes into cards and runs the needs-enrichment test. Online learning
    (learn_from_episode) matches incoming playtime steps against existing cards
    and confirms/denies/creates — it NEVER re-clusters (stable card ids).
    refresh_edges() rebuilds the causal graph from produces/needs and the unlock
    ledger (temporal lift, acyclic insertion).
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.prototypes: Dict[str, TransitionPrototype] = {}
        # Causal graph edges, keyed by dst_id.
        self.edges: Dict[str, List[RelationEdge]] = {}
        # Total evidence observations consumed (offline corpus size set by
        # consolidation; grows by steps_recorded on each learn_from_episode).
        self.evidence_count: int = 0
        # Online evidence ring buffer (source="online" EpisodicEvidence, capped).
        self.online_evidence: Deque[EpisodicEvidence] = deque(maxlen=ONLINE_EVIDENCE_CAP)
        # Per-session interaction attempts per prototype (UCB exploration term).
        self.session_attempts: Dict[str, int] = {}
        # Provenance hash of the last saved payload.
        self.bank_sha256: Optional[str] = None
        # Unlock ledger: episode_id -> ordered [(step_idx, mask_before dims, unlock dims)].
        self.unlock_ledger: Dict[int, List[Tuple[int, FrozenSet[int], FrozenSet[int]]]] = {}
        # episode_id -> {event_dim: first unlock step} for mean_dt computation.
        self.episode_first_unlock: Dict[int, Dict[int, int]] = {}
        # Diagnostic: enabling edges rejected by the acyclicity check.
        self.rejected_edges: List[RelationEdge] = []
        # Non-schema match support: stored member latents per card (excluded from
        # asdict; serialized separately in save()).
        self._member_latents: Dict[str, List[List[float]]] = {}
        # Online-born card id counter per (action, event_dim).
        self._online_card_counter: Dict[Tuple[int, int], int] = {}

    def __len__(self) -> int:
        return len(self.prototypes)

    def get_prototypes_for_action(self, action: int) -> List[TransitionPrototype]:
        """Returns all candidate prototypes for a discrete action."""
        return [p for p in self.prototypes.values() if p.action == action]

    # ------------------------------------------------------------------
    # Effective statistics
    # ------------------------------------------------------------------

    def effective_success_stats(self, p: TransitionPrototype) -> Tuple[float, float]:
        """(k, n) for Wilson confidence: offline observations weigh 1.0, online 0.5.

        Raw tallies already include online observations at weight 1.0, so the
        online share is discounted by half a count per online success here.
        register_interaction_outcome bumps stay at full weight (deliberate,
        planner-driven interactions).
        """
        k = float(p.success_count) - 0.5 * float(p.success_online)
        k = max(k, 0.0)
        n = float(p.success_count) + float(p.failure_count)
        return (k, n)

    def _recompute_confidence(self, p: TransitionPrototype) -> None:
        k, n = self.effective_success_stats(p)
        p.wilson_lb = float(wilson_interval(k, n)[0])
        p.confidence = max(0.05, p.wilson_lb)

    def _recompute_status(self, p: TransitionPrototype) -> None:
        p.status = "confirmed" if p.weighted_successes >= CONFIRM_WEIGHTED_SUCCESSES else "hypothesis"
        if p.status == "confirmed":
            # Pure causal consumption grounding: an action requires what it consumes from inventory (dims 0..11).
            p.needs = [[float(d), amt] for d, amt in p.consumes if 0 <= int(d) < 12]
            p.precondition_state = [0.0] * STATE_DIM
            for nd in p.needs:
                d = int(nd[0])
                if 0 <= d < STATE_DIM:
                    p.precondition_state[d] = float(nd[1])
        else:
            p.needs = []
            p.precondition_state = [0.0] * STATE_DIM

    # ------------------------------------------------------------------
    # Unlock decomposition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fired_event_dims(event_vec: Sequence[int], mask_before_vec: Sequence[int]) -> List[int]:
        """u_t = events * (1 - mask_before): first-time unlock dims for one step."""
        n = min(len(event_vec), len(mask_before_vec), NUM_ACHIEVEMENTS)
        return [d for d in range(n) if int(event_vec[d]) and not int(mask_before_vec[d])]

    @staticmethod
    def _fallback_unlock_dims(reward: float) -> Tuple[List[int], bool]:
        """Degraded unlock label for evidence without event vectors.

        Legacy shards carry no events/masks. A reward >= 0.9 step is treated as an
        (unattributed) unlock success — grouped event-free, never event-grounded,
        because a fallback pin is a guess, not an observation (plan constraint N7).
        """
        if reward >= 0.9:
            return [d for d in range(NUM_ACHIEVEMENTS)], True
        return [], False

    # ------------------------------------------------------------------
    # Offline consolidation (v2)
    # ------------------------------------------------------------------

    def consolidate_from_evidence(
        self,
        evidence_store: EpisodicEvidenceStore,
        min_cluster_size: int = 2,
        cos_sim_threshold: float = 0.85,
    ) -> int:
        """
        Consolidates raw episodic evidence into learned causal cards (v2).

        1. Per action: successes = steps with any(u_t) OR (no unlock AND
           max|state_delta| >= 0.35); near-misses = no unlock AND max|delta| < 0.35
           AND reward <= 0 (includes the old uncounted mid-band).
        2. Unlocked successes group by EACH fired event dim (multi-unlock joins both
           groups); event-free productive steps group at action level; sub-cluster
           greedily by visual cosine >= threshold (seed = first member).
        3. Per card: centroids, effect delta, dispersion, Laplace unlock_prob,
           needs via enrichment test (successes vs matched near-misses, Wilson
           separation + gap + BH-FDR q=0.10 across the 16 state dims), weighted
           float stats and Wilson-lower-bound confidence.
        4. Builds the unlock ledger from event-labelled evidence, then rebuilds
           the causal graph (refresh_edges).

        Returns the number of cards written.
        """
        if len(evidence_store) == 0:
            return 0

        entries = list(evidence_store.entries)
        self.evidence_count = len(entries)
        self._merge_unlock_ledger_from_evidence(entries)

        written = 0
        print(f"-> Consolidating evidence into causal cards across {NUM_ACTIONS} actions...", flush=True)
        for act in range(NUM_ACTIONS):
            ev_list = [e for e in entries if e.action == act]
            if not ev_list:
                continue

            successes: List[Tuple[EpisodicEvidence, List[int], bool]] = []
            near_misses: List[EpisodicEvidence] = []
            for e in ev_list:
                if e.event_vec is not None:
                    fired = self._fired_event_dims(e.event_vec, e.mask_before_vec or [0] * NUM_ACHIEVEMENTS)
                    degraded = False
                else:
                    fired, degraded = self._fallback_unlock_dims(e.reward)
                # Event-free breakthrough criterion requires material change on inventory dims 0..11.
                # Natural vital decay (hunger/thirst) does not make an action an event-free productive card.
                max_delta = max((abs(d) for d in e.state_delta[:12]), default=0.0)
                if fired:
                    successes.append((e, fired, degraded))
                elif degraded or max_delta >= PRODUCTIVE_DELTA:
                    # Degraded fallback successes carry no grounded event dim ->
                    # grouped event-free (N7: a guess is not an observation).
                    successes.append((e, [], True))
                elif max_delta < NEAR_MISS_DELTA and e.reward <= 0.0:
                    near_misses.append(e)
                # else: reward > 0 with no unlock and no productive delta ->
                # unclassified; counted nowhere (same policy as v1 for idle steps).

            if not successes:
                continue

            # Group successes: event-grounded by EACH fired dim, event-free at
            # action level.
            groups: Dict[Tuple[int, Optional[int]], List[EpisodicEvidence]] = {}
            for e, fired, degraded in successes:
                if degraded or not fired:
                    groups.setdefault((act, None), []).append(e)
                else:
                    for ed in fired:
                        groups.setdefault((act, ed), []).append(e)

            near_miss_latents = [nm.obs_latent for nm in near_misses]
            nm_states = [nm.grounded_state for nm in near_misses]
            if near_miss_latents:
                nm_tensor = torch.as_tensor(near_miss_latents, dtype=torch.float32)
                nm_tensor = torch.nn.functional.normalize(nm_tensor, p=2, dim=-1)
            else:
                nm_tensor = torch.empty((0, 128), dtype=torch.float32)

            action_written = 0
            for (a, edim), members in groups.items():
                latents = [m.obs_latent for m in members]
                subclusters = _greedy_cosine_subclusters(latents, cos_sim_threshold)
                for k, cluster_idx in enumerate(subclusters):
                    cluster = [members[i] for i in cluster_idx]
                    if len(cluster) < min_cluster_size:
                        continue
                    proto_id = f"card_a{a}_e{edim if edim is not None else 'none'}_c{k}"
                    proto, member_latents = self._build_card_from_cluster(
                        proto_id=proto_id,
                        action=a,
                        event_dim=edim,
                        cluster=cluster,
                        near_misses=near_misses,
                        near_miss_latents=near_miss_latents,
                        nm_states=nm_states,
                        cos_sim_threshold=cos_sim_threshold,
                        nm_tensor=nm_tensor,
                    )
                    self.prototypes[proto_id] = proto
                    self._member_latents[proto_id] = member_latents
                    written += 1
                    action_written += 1

            print(f"   [Action {act:2d}/17] {len(successes)} successes, {len(near_misses)} near-misses -> {action_written} card(s) created", flush=True)

        print(f"-> Building causal graph edges...", flush=True)
        self.refresh_edges()
        return written

    def _build_card_from_cluster(
        self,
        proto_id: str,
        action: int,
        event_dim: Optional[int],
        cluster: List[EpisodicEvidence],
        near_misses: List[EpisodicEvidence],
        near_miss_latents: List[List[float]],
        nm_states: List[List[float]],
        cos_sim_threshold: float,
        nm_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[TransitionPrototype, List[List[float]]]:
        """Finalizes one card from a sub-cluster of successes (+ matched near-misses)."""
        member_latents = [list(m.obs_latent) for m in cluster]

        z_pre = _l2_normalize(np.mean([m.obs_latent for m in cluster], axis=0))
        z_post = _l2_normalize(np.mean([m.next_obs_latent for m in cluster], axis=0))
        delta_mean = np.mean([m.state_delta for m in cluster], axis=0).tolist()
        mean_ret = float(np.mean([m.return_g for m in cluster]))
        dispersion = _mean_pairwise_cosine(member_latents)

        # Matched near-misses: cos(z_nm, ANY MEMBER) >= threshold (member rule,
        # not the bare centroid). Vectorized with PyTorch matrix multiplication.
        matched_nm: List[int] = []
        if nm_tensor is not None and len(nm_tensor) > 0 and len(member_latents) > 0:
            mem_tensor = torch.as_tensor(member_latents, dtype=torch.float32)
            mem_tensor = torch.nn.functional.normalize(mem_tensor, p=2, dim=-1)
            M = nm_tensor.shape[0]
            chunk_size = 4096
            matched_indices: List[int] = []
            for c_start in range(0, M, chunk_size):
                c_end = min(c_start + chunk_size, M)
                sims = torch.mm(nm_tensor[c_start:c_end], mem_tensor.t())
                max_sims = sims.max(dim=1).values
                hits = (max_sims >= cos_sim_threshold).nonzero().flatten() + c_start
                matched_indices.extend(hits.tolist())
            matched_nm = matched_indices
        elif len(near_miss_latents) > 0 and len(member_latents) > 0:
            for j, z_nm in enumerate(near_miss_latents):
                sim = max((_cosine(z_nm, m) for m in member_latents), default=0.0)
                if sim >= cos_sim_threshold:
                    matched_nm.append(j)

        n_succ = float(len(cluster))
        n_fail = float(len(matched_nm))

        if event_dim is not None:
            n_applied = n_succ + n_fail
            unlock_prob = (n_succ + 1.0) / (n_applied + 2.0)
            achievement_delta = [0.0] * NUM_ACHIEVEMENTS
            achievement_delta[int(event_dim)] = 1.0
        else:
            unlock_prob = 0.0
            achievement_delta = [0.0] * NUM_ACHIEVEMENTS

        produces = [[float(d), float(delta_mean[d])]
                    for d in range(STATE_DIM) if delta_mean[d] >= PRODUCES_THRESHOLD]
        consumes = [[float(d), float(abs(delta_mean[d]))]
                    for d in range(STATE_DIM) if delta_mean[d] <= -PRODUCES_THRESHOLD]

        status = "confirmed" if n_succ >= CONFIRM_WEIGHTED_SUCCESSES else "hypothesis"
        needs: List[List[float]] = []
        if status == "confirmed":
            # Pure causal consumption grounding: what was deducted from inventory (dims 0..11)
            # is what this action needs to be executed (Conservation of Inventory).
            needs = [[float(d), amt] for d, amt in consumes if 0 <= int(d) < 12]

        # Legacy precondition interface: sparse needs vector.
        precondition_state = [0.0] * STATE_DIM
        for nd in needs:
            d = int(nd[0])
            if 0 <= d < STATE_DIM:
                precondition_state[d] = float(nd[1])

        wilson_lb = float(wilson_interval(n_succ, n_succ + n_fail)[0])

        proto = TransitionPrototype(
            prototype_id=proto_id,
            action=action,
            precondition_key=z_pre,
            precondition_state=precondition_state,
            effect_delta=[float(v) for v in delta_mean],
            postcondition_key=z_post,
            success_count=n_succ,
            failure_count=n_fail,
            mean_return=mean_ret,
            mean_duration=1.0,
            confidence=max(0.05, wilson_lb),
            achievement_delta=achievement_delta,
            event_dim=event_dim,
            needs=needs,
            produces=produces,
            consumes=consumes,
            unlock_prob=float(unlock_prob),
            wilson_lb=wilson_lb,
            visual_dispersion=float(dispersion),
            status=status,
            success_offline=n_succ,
            success_online=0.0,
            failure_offline=n_fail,
            failure_online=0.0,
            created_from="offline",
            schema_version=CARD_SCHEMA_VERSION,
        )
        return proto, member_latents

    @staticmethod
    def _enrich_needs(
        success_states: List[Sequence[float]],
        near_miss_states: List[Sequence[float]],
    ) -> List[List[float]]:
        """
        Needs-enrichment test per 16-d state dim (plan §3 Stage 3).

        For dim d: q_d = NEEDS_QUANTILE of successes' x[d]; skip if q_d < 0.45.
        s_d = frac(successes with x[d] >= q_d); f_d = frac(matched near-misses with
        x[d] >= q_d). |F| < 2 -> dim untestable. Keep iff support >= 3 AND
        Wilson-80% interval of s_d does not overlap f_d's AND (s_d - f_d) >= 0.20
        AND BH-FDR q=0.10 across the 16 per-card dims (one-sided two-proportion
        z-test p-values; untested dims enter the family with p = 1.0).
        """
        needs: List[List[float]] = []
        if not success_states:
            return needs
        S = np.asarray(success_states, dtype=np.float64)
        F = np.asarray(near_miss_states, dtype=np.float64) if near_miss_states else None
        n_s = S.shape[0]
        n_f = F.shape[0] if F is not None else 0

        tested: List[Tuple[int, float]] = []
        passed_gates: Dict[int, Tuple[float, float]] = {}  # dim -> (q_d, s_d - f_d)
        for d in range(STATE_DIM):
            q_d = float(np.quantile(S[:, d], NEED_QUANTILE))
            if q_d < NEED_MIN_LEVEL:
                continue
            k_s = int(np.sum(S[:, d] >= q_d))
            if k_s < NEED_MIN_SUPPORT:
                continue
            if F is None or n_f < NEED_MIN_FAILURES:
                # Untestable without a failure comparison — stays out of needs.
                continue
            k_f = int(np.sum(F[:, d] >= q_d))
            s_d = k_s / n_s
            f_d = k_f / n_f
            lb_s = wilson_interval(k_s, n_s)[0]
            ub_f = wilson_interval(k_f, n_f)[1]
            if not (lb_s > ub_f):
                continue
            if (s_d - f_d) < NEED_MIN_GAP:
                continue
            p_val = _two_proportion_pvalue_one_sided(k_s, n_s, k_f, n_f)
            tested.append((d, p_val))
            passed_gates[d] = (q_d, s_d - f_d)

        # BH-FDR across the 16 per-card dims: dims never entering the test count
        # as p = 1.0 in the family.
        family: List[Tuple[int, float]] = list(tested)
        tested_dims = {d for d, _ in tested}
        for d in range(STATE_DIM):
            if d not in tested_dims:
                family.append((d, 1.0))
        kept = set(_bh_fdr_reject(family, BH_FDR_Q))

        for d in sorted(kept & set(passed_gates.keys())):
            q_d = passed_gates[d][0]
            amount = float(min(max(q_d, NEED_AMOUNT_RANGE[0]), NEED_AMOUNT_RANGE[1]))
            needs.append([float(d), amount])
        return needs

    # ------------------------------------------------------------------
    # Unlock ledger
    # ------------------------------------------------------------------

    def _merge_unlock_ledger_from_evidence(self, entries: List[EpisodicEvidence]) -> None:
        """Builds/merges the per-episode unlock ledger from event-labelled evidence."""
        by_episode: Dict[int, List[EpisodicEvidence]] = {}
        for e in entries:
            if e.event_vec is None:
                continue
            by_episode.setdefault(int(e.episode_id), []).append(e)

        for ep, steps in by_episode.items():
            steps = sorted(steps, key=lambda s: s.step_id)
            new_steps = [
                (
                    int(s.step_id),
                    frozenset(d for d in range(min(len(s.mask_before_vec or []), NUM_ACHIEVEMENTS))
                              if (s.mask_before_vec or [])[d]),
                    frozenset(self._fired_event_dims(s.event_vec, s.mask_before_vec or [0] * NUM_ACHIEVEMENTS)),
                )
                for s in steps
            ]
            merged = {t: (m, u) for t, m, u in self.unlock_ledger.get(ep, [])}
            for t, m, u in new_steps:
                merged[t] = (m, u)
            ordered = [(t, m, u) for t, (m, u) in sorted(merged.items())]
            self.unlock_ledger[ep] = ordered
            self._rebuild_first_unlock(ep)

    def _rebuild_first_unlock(self, episode_id: int) -> None:
        first: Dict[int, int] = {}
        for t, _mask, unlock in self.unlock_ledger.get(episode_id, []):
            for d in unlock:
                if d not in first or t < first[d]:
                    first[d] = t
        self.episode_first_unlock[episode_id] = first

    # ------------------------------------------------------------------
    # Causal graph
    # ------------------------------------------------------------------

    def _produced_dims(self, p: TransitionPrototype) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for entry in p.produces:
            if len(entry) >= 2:
                out[int(entry[0])] = float(entry[1])
        return out

    def _needed_dims(self, p: TransitionPrototype) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for entry in p.needs:
            if len(entry) >= 2:
                out[int(entry[0])] = float(entry[1])
        return out

    def _strongest_event_card(self, event_dim: int) -> Optional[TransitionPrototype]:
        """Card with event_dim == e with the highest wilson_lb (tie: lexicographic id)."""
        candidates = [p for p in self.prototypes.values() if p.event_dim == event_dim]
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: (-p.wilson_lb, p.prototype_id))[0]

    def _enabling_candidates(self) -> List[Dict[str, Any]]:
        """Temporal-lift candidates over event pairs from the unlock ledger."""
        total_steps = sum(len(steps) for steps in self.unlock_ledger.values())
        if total_steps == 0:
            return []
        flat_steps = []
        for ep, steps in self.unlock_ledger.items():
            for t, mask, unlock in steps:
                flat_steps.append((ep, t, mask, unlock))

        events = set()
        for _ep, _t, _mask, unlock in flat_steps:
            events.update(unlock)

        # Pre-index unlock occurrences per event and condition step counts
        unlock_steps_by_event: Dict[int, List[Tuple[int, int, FrozenSet[int]]]] = {}
        for ep, t, mask, unlock in flat_steps:
            for u in unlock:
                unlock_steps_by_event.setdefault(u, []).append((ep, t, mask))

        cond_count_by_event: Dict[int, int] = {}
        for _ep, _t, mask, _unlock in flat_steps:
            for d in mask:
                cond_count_by_event[d] = cond_count_by_event.get(d, 0) + 1

        candidates: List[Dict[str, Any]] = []
        for i in sorted(events):
            cond_count = cond_count_by_event.get(i, 0)
            if cond_count == 0:
                continue
            for j in sorted(events):
                if i == j:
                    continue
                unlock_steps = unlock_steps_by_event.get(j, [])
                if not unlock_steps:
                    continue
                p_base = len(unlock_steps) / total_steps
                if p_base <= 0.0:
                    continue
                co_steps = [(ep, t) for (ep, t, mask) in unlock_steps if i in mask]
                co_count = len(co_steps)
                if co_count == 0:
                    continue
                p_cond = co_count / cond_count
                lift = p_cond / p_base
                dts = []
                first_maps = self.episode_first_unlock
                for (ep, t) in co_steps:
                    fi = first_maps.get(ep, {}).get(i)
                    if fi is not None:
                        dts.append(t - fi)
                mean_dt = float(np.mean(dts)) if dts else 0.0
                if lift >= ENABLING_MIN_LIFT and co_count >= ENABLING_MIN_SUPPORT and mean_dt > 0.0:
                    candidates.append({
                        "src_event": i,
                        "dst_event": j,
                        "lift": float(lift),
                        "support": int(co_count),
                        "mean_dt": mean_dt,
                    })
        return candidates

    @staticmethod
    def _creates_cycle(adjacency: Dict[str, set], src: str, dst: str) -> bool:
        """Would adding src -> dst create a directed cycle? (dst can already reach src.)"""
        if src == dst:
            return True
        seen = set()
        stack = [dst]
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        return False

    def refresh_edges(self) -> int:
        """
        Rebuilds ALL causal edges and returns the total count.

        (a) produces_need: consumer card with needs [(d, amt)] <- every OTHER card
            producing d (mean delta >= +0.25).
        (b) enabling_lift: ordered event pair (i, j) with lift = P(j unlocks | i in
            mask_before) / P(j) >= 1.5, co-occurrence >= 5, mean_dt > 0; mapped to
            each event's strongest card; inserted in lift-descending order with a
            DFS acyclicity check on the enabling-only subgraph (rejected edges are
            kept in self.rejected_edges).
        """
        self.edges = {}
        self.rejected_edges = []

        # (a) produces_need
        for consumer in self.prototypes.values():
            needed = self._needed_dims(consumer)
            for d in sorted(needed):
                for producer in self.prototypes.values():
                    if producer.prototype_id == consumer.prototype_id:
                        continue
                    produced = self._produced_dims(producer)
                    if d in produced and produced[d] >= PRODUCES_THRESHOLD:
                        edge = RelationEdge(
                            src_id=producer.prototype_id,
                            dst_id=consumer.prototype_id,
                            kind="produces_need",
                            dim=int(d),
                            lift=1.0,
                            support=int(producer.attempt_count),
                            mean_dt=0.0,
                        )
                        self.edges.setdefault(consumer.prototype_id, []).append(edge)

        # (b) enabling_lift (enabling-only subgraph, acyclic insertion)
        enabling_adjacency: Dict[str, set] = {}
        candidates = self._enabling_candidates()
        candidates.sort(key=lambda c: (-c["lift"], c["src_event"], c["dst_event"]))
        for cand in candidates:
            src_card = self._strongest_event_card(cand["src_event"])
            dst_card = self._strongest_event_card(cand["dst_event"])
            if src_card is None or dst_card is None or src_card.prototype_id == dst_card.prototype_id:
                continue
            edge = RelationEdge(
                src_id=src_card.prototype_id,
                dst_id=dst_card.prototype_id,
                kind="enabling_lift",
                dim=None,
                lift=cand["lift"],
                support=cand["support"],
                mean_dt=cand["mean_dt"],
            )
            if self._creates_cycle(enabling_adjacency, edge.src_id, edge.dst_id):
                self.rejected_edges.append(edge)
                continue
            enabling_adjacency.setdefault(edge.src_id, set()).add(edge.dst_id)
            self.edges.setdefault(edge.dst_id, []).append(edge)

        return sum(len(v) for v in self.edges.values())

    def producers_for_dim(self, dim: int) -> List[TransitionPrototype]:
        """Cards that (learn to) produce the given 16-d dim.

        Uses the stored produces list when present; legacy cards without produces
        fall back to their effect_delta (mean delta >= +0.25).
        """
        producers = []
        for p in self.prototypes.values():
            produced = self._produced_dims(p)
            if not produced:
                produced = {d: float(p.effect_delta[d])
                            for d in range(min(STATE_DIM, len(p.effect_delta)))
                            if p.effect_delta[d] >= PRODUCES_THRESHOLD}
            if dim in produced and produced[dim] >= PRODUCES_THRESHOLD:
                producers.append(p)
        return producers

    # ------------------------------------------------------------------
    # Online learning (match-and-update; NEVER re-cluster)
    # ------------------------------------------------------------------

    def _card_match_similarity(self, z: Sequence[float], proto: TransitionPrototype,
                               include_precondition_key: bool = True) -> float:
        """Max cosine between z and the card's stored members (plus its centroid)."""
        zn = np.asarray(z, dtype=np.float64).flatten()
        sims: List[float] = []
        if include_precondition_key and proto.precondition_key:
            sims.append(_cosine(zn, proto.precondition_key))
        for m in self._member_latents.get(proto.prototype_id, []):
            sims.append(_cosine(zn, m))
        return max(sims) if sims else 0.0

    def _match_card(self, z: Sequence[float], action: int,
                    event_dim: Optional[int]) -> Optional[TransitionPrototype]:
        """Best card with matching action whose stored members the step visually
        matches (cos >= COS_SIM_THRESHOLD).

        event_dim int -> only cards grounded to that event (success matching).
        event_dim None -> any card of the action (near-miss denial matching).
        """
        best: Optional[TransitionPrototype] = None
        best_sim = COS_SIM_THRESHOLD
        for p in self.prototypes.values():
            if p.action != action:
                continue
            if event_dim is not None and p.event_dim != event_dim:
                continue
            sim = self._card_match_similarity(z, p)
            if sim >= COS_SIM_THRESHOLD and sim > best_sim:
                best, best_sim = p, sim
        return best

    def _update_keys_online(self, proto: TransitionPrototype, z: Sequence[float],
                            next_z: Optional[Sequence[float]]) -> None:
        """n-weighted running mean of the visual centroids; needs are NEVER touched."""
        members = self._member_latents.setdefault(proto.prototype_id, [])
        members.append([float(v) for v in z])
        proto.precondition_key = _l2_normalize(np.mean(np.asarray(members, dtype=np.float64), axis=0))
        nz = next_z if next_z is not None else z
        try:
            n_prev = max(float(proto.success_count - 1.0), 0.0)
            old = np.asarray(proto.postcondition_key, dtype=np.float64)
            blended = (old * n_prev + np.asarray(nz, dtype=np.float64)) / (n_prev + 1.0)
            proto.postcondition_key = _l2_normalize(blended)
        except Exception:
            proto.postcondition_key = _l2_normalize(nz)

    def _create_online_card(
        self,
        action: int,
        event_dim: int,
        z: Sequence[float],
        next_z: Optional[Sequence[float]],
        state_delta: Optional[Sequence[float]],
        count_success: bool,
    ) -> TransitionPrototype:
        """Creates an online-born hypothesis card.

        Event-grounded no-match creations record the observation
        (success_online = 1.0). Fallback-pinned creations record NOTHING
        (success_online = 0.0, reduced Laplace prior) — N7 quarantine.
        """
        n = self._online_card_counter.get((action, event_dim), 0)
        self._online_card_counter[(action, event_dim)] = n + 1
        proto_id = f"card_a{action}_e{event_dim}_o{n}"
        s_online = 1.0 if count_success else 0.0
        delta = [float(v) for v in state_delta] if state_delta is not None else [0.0] * STATE_DIM
        achievement_delta = [0.0] * NUM_ACHIEVEMENTS
        achievement_delta[int(event_dim)] = 1.0
        proto = TransitionPrototype(
            prototype_id=proto_id,
            action=int(action),
            precondition_key=_l2_normalize(z),
            precondition_state=[0.0] * STATE_DIM,
            effect_delta=delta,
            postcondition_key=_l2_normalize(next_z if next_z is not None else z),
            success_count=s_online,
            failure_count=0.0,
            mean_return=0.0,
            mean_duration=1.0,
            confidence=0.05,
            achievement_delta=achievement_delta,
            event_dim=int(event_dim),
            needs=[],           # needs amounts are never derived online
            produces=[[float(d), delta[d]] for d in range(STATE_DIM) if delta[d] >= PRODUCES_THRESHOLD],
            consumes=[[float(d), abs(delta[d])] for d in range(STATE_DIM) if delta[d] <= -PRODUCES_THRESHOLD],
            unlock_prob=(s_online + 1.0) / (s_online + 2.0),
            wilson_lb=0.0,
            visual_dispersion=1.0,
            status="hypothesis",
            success_offline=0.0,
            success_online=s_online,
            failure_offline=0.0,
            failure_online=0.0,
            created_from="online",
            schema_version=CARD_SCHEMA_VERSION,
        )
        self._recompute_confidence(proto)
        self._recompute_status(proto)
        self.prototypes[proto_id] = proto
        self._member_latents[proto_id] = [[float(v) for v in z]]
        return proto

    def _confirm_card(
        self,
        proto: TransitionPrototype,
        z: Sequence[float],
        next_z: Optional[Sequence[float]],
        state_delta: Optional[Sequence[float]] = None,
    ) -> None:
        proto.success_online += 1.0
        proto.success_count += 1.0
        self._update_keys_online(proto, z, next_z)
        if state_delta is not None and any(abs(d) >= PRODUCES_THRESHOLD for d in state_delta[:12]):
            n_prev = max(float(proto.success_count - 1.0), 0.0)
            old_delta = np.asarray(proto.effect_delta, dtype=np.float64)
            new_delta = (old_delta * n_prev + np.asarray(state_delta, dtype=np.float64)) / (n_prev + 1.0)
            proto.effect_delta = [float(v) for v in new_delta]
            proto.produces = [[float(d), float(new_delta[d])]
                             for d in range(STATE_DIM) if new_delta[d] >= PRODUCES_THRESHOLD]
            proto.consumes = [[float(d), float(abs(new_delta[d]))]
                             for d in range(STATE_DIM) if new_delta[d] <= -PRODUCES_THRESHOLD]
        self._recompute_confidence(proto)
        self._recompute_status(proto)

    def _append_online_evidence(self, evidence: EpisodicEvidence) -> None:
        self.online_evidence.append(evidence)
        self.evidence_count += 1

    def learn_from_episode(self, steps: List[Dict[str, Any]], episode_id: Optional[int] = None) -> Dict[str, int]:
        """
        Incremental online learning from one episode of playtime steps.
        Match-and-update only — never re-clusters (stable card ids).

        Each step dict requires: obs_latent, grounded_state, action, reward, done,
        event_vec [22], mask_before_vec [22], confident_events (caller-computed).
        Optional: fallback_unlocked (reward-fallback pinned dims, default []),
        state_delta / next_grounded_state / next_obs_latent.

        Per step: (1) ledger entry; (2) fired event dims not in fallback_unlocked
        match (action, event_dim) cards by member cosine -> CONFIRM
        (success_online += 1.0, key running means, Wilson refresh), or CREATE an
        event-grounded hypothesis card when unmatched and caller-confident;
        (3) near-miss steps (no unlock, max|delta| < 0.35, reward <= 0) matched to
        a card -> DENY (failure_online += 1.0); (4) fallback-pinned fired dims are
        quarantined: they may create reduced-prior hypothesis cards on no-match
        but NEVER increment success_online of any card; (5) needs are never
        updated online; (6) every step is appended to online_evidence
        (source="online", return_g=0.0). The caller then calls refresh_edges().

        Returns {"created", "confirmed", "denied", "steps_recorded"}.
        """
        if episode_id is None:
            int_keys = [k for k in self.unlock_ledger.keys() if isinstance(k, int)]
            episode_id = (max(int_keys) + 1) if int_keys else 0
        episode_id = int(episode_id)

        created = 0
        confirmed = 0
        denied = 0

        for idx, step in enumerate(steps):
            z = [float(v) for v in step["obs_latent"]]
            x = [float(v) for v in step["grounded_state"]]
            action = int(step["action"])
            reward = float(step.get("reward", 0.0))
            done = bool(step.get("done", False))
            event_vec = [int(v) for v in (step.get("event_vec") or [0] * NUM_ACHIEVEMENTS)]
            mask_vec = [int(v) for v in (step.get("mask_before_vec") or [0] * NUM_ACHIEVEMENTS)]
            confident_events = set(int(v) for v in (step.get("confident_events") or []))
            fallback_unlocked = set(int(v) for v in (step.get("fallback_unlocked") or []))

            if step.get("state_delta") is not None:
                state_delta = [float(v) for v in step["state_delta"]]
            elif step.get("next_grounded_state") is not None:
                nx = [float(v) for v in step["next_grounded_state"]]
                state_delta = [b - a for a, b in zip(x, nx)]
            else:
                state_delta = [0.0] * STATE_DIM
            next_z = step.get("next_obs_latent")
            next_z = [float(v) for v in next_z] if next_z is not None else None
            next_x = step.get("next_grounded_state")
            next_x = [float(v) for v in next_x] if next_x is not None else [a + b for a, b in zip(x, state_delta)]

            fired = self._fired_event_dims(event_vec, mask_vec)
            max_delta = max((abs(d) for d in state_delta), default=0.0)

            # (1) Unlock ledger entry (episode-relative step index).
            mask_dims = frozenset(d for d in range(min(len(mask_vec), NUM_ACHIEVEMENTS)) if mask_vec[d])
            self.unlock_ledger.setdefault(episode_id, []).append((idx, mask_dims, frozenset(fired)))
            first = self.episode_first_unlock.setdefault(episode_id, {})
            for d in fired:
                if d not in first or idx < first[d]:
                    first[d] = idx

            # (6) Online evidence ring buffer (diagnostic/audit; never re-clustered).
            self._append_online_evidence(EpisodicEvidence(
                obs_latent=z,
                grounded_state=x,
                action=action,
                reward=reward,
                done=done,
                next_obs_latent=next_z if next_z is not None else list(z),
                next_grounded_state=next_x,
                state_delta=state_delta,
                return_g=0.0,
                episode_id=episode_id,
                step_id=idx,
                confidence=1.0,
                event_vec=event_vec,
                mask_before_vec=mask_vec,
                source="online",
            ))

            if fired:
                for ed in fired:
                    if ed in fallback_unlocked:
                        # (4) N7 quarantine: a fallback pin is a guess, not an
                        # observation. Matching cards are never confirmed; only
                        # no-match creation of a reduced-prior hypothesis card.
                        match = self._match_card(z, action, ed)
                        if match is None:
                            self._create_online_card(action, ed, z, next_z, state_delta,
                                                     count_success=False)
                            created += 1
                        continue
                    match = self._match_card(z, action, ed)
                    if match is not None:
                        # (2) CONFIRM
                        self._confirm_card(match, z, next_z, state_delta)
                        confirmed += 1
                    elif ed in confident_events:
                        self._create_online_card(action, ed, z, next_z, state_delta,
                                                 count_success=True)
                        created += 1
                    # else: unconfirmed noisy unlock — counted nowhere.
            elif max_delta < NEAR_MISS_DELTA and reward <= 0.0:
                # (3) Near-miss -> DENY the best visually matching card of this action.
                match = self._match_card(z, action, event_dim=None)
                if match is not None:
                    match.failure_online += 1.0
                    match.failure_count += 1.0
                    self._recompute_confidence(match)
                    denied += 1

        return {
            "created": created,
            "confirmed": confirmed,
            "denied": denied,
            "steps_recorded": len(steps),
        }

    # ------------------------------------------------------------------
    # Interaction outcomes (kept from v1)
    # ------------------------------------------------------------------

    def register_interaction_outcome(
        self,
        prototype_id: str,
        succeeded: bool,
    ) -> Optional[TransitionPrototype]:
        """
        Acceptance Criterion 3 (Revision):
        Contradictory observations lower confidence or revise a relation
        instead of reinforcing a false belief. Bumps the aggregate tallies at
        full weight (deliberate, planner-driven interactions) and recomputes
        confidence = max(0.05, wilson_lb(effective stats)).
        """
        if prototype_id not in self.prototypes:
            return None
        proto = self.prototypes[prototype_id]
        if succeeded:
            proto.success_count += 1.0
        else:
            proto.failure_count += 1.0
        self.session_attempts[prototype_id] = self.session_attempts.get(prototype_id, 0) + 1
        self._recompute_confidence(proto)
        return proto

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the bank (schema phase2_relation_bank_v2).

        Payload: prototypes (asdict), edges (dst -> [edge dicts]), evidence_count,
        online_evidence, session_attempts, unlock ledger, member latents (kept OUT
        of the prototype asdicts), bank_sha256.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": BANK_SCHEMA_VERSION,
            "prototypes": {k: asdict(v) for k, v in self.prototypes.items()},
            "edges": {dst: [asdict(e) for e in edge_list] for dst, edge_list in self.edges.items()},
            "evidence_count": int(self.evidence_count),
            "online_evidence": [asdict(e) for e in self.online_evidence],
            "session_attempts": dict(self.session_attempts),
            "unlock_ledger": [
                [int(ep), [[int(t), sorted(int(d) for d in mask), sorted(int(d) for d in unlock)]
                           for (t, mask, unlock) in steps]]
                for ep, steps in self.unlock_ledger.items()
            ],
            "member_latents": {k: v for k, v in self._member_latents.items()},
        }
        payload["bank_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.bank_sha256 = payload["bank_sha256"]
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, p)

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads a bank from disk. Tolerant of v1 files (missing card fields ->
        dataclass defaults; missing edges -> refresh_edges())."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Relation hypothesis bank not found: {p}")
        with open(p, "r") as f:
            data = json.load(f)

        proto_fields = {f.name for f in dataclass_fields(TransitionPrototype)}
        prototypes: Dict[str, TransitionPrototype] = {}
        for key, item in data.get("prototypes", {}).items():
            filtered = {k: v for k, v in item.items() if k in proto_fields}
            proto = TransitionPrototype(**filtered)
            if "schema_version" not in item:
                # v1 card migration: derive Wilson stats and status from raw counts.
                k = float(proto.success_count)
                n = k + float(proto.failure_count)
                proto.wilson_lb = float(wilson_interval(k, n)[0])
                proto.confidence = max(0.05, proto.wilson_lb)
                proto.status = "confirmed" if k >= CONFIRM_WEIGHTED_SUCCESSES else "hypothesis"
                proto.created_from = "offline"
                proto.schema_version = "phase2_card_v1"
            if not proto.achievement_delta:
                proto.achievement_delta = [0.0] * NUM_ACHIEVEMENTS
            prototypes[key] = proto
        self.prototypes = prototypes

        edge_fields = {f.name for f in dataclass_fields(RelationEdge)}
        edges: Dict[str, List[RelationEdge]] = {}
        for dst, edge_dicts in data.get("edges", {}).items():
            edges[dst] = [RelationEdge(**{k: v for k, v in ed.items() if k in edge_fields})
                          for ed in edge_dicts]
        self.edges = edges

        self.evidence_count = int(data.get("evidence_count", 0))
        evidence_fields = {f.name for f in dataclass_fields(EpisodicEvidence)}
        self.online_evidence = deque(
            (EpisodicEvidence(**{k: v for k, v in item.items() if k in evidence_fields})
             for item in data.get("online_evidence", [])),
            maxlen=ONLINE_EVIDENCE_CAP,
        )
        self.session_attempts = dict(data.get("session_attempts", {}))
        self.bank_sha256 = data.get("bank_sha256")

        self.unlock_ledger = {}
        for ep, steps in data.get("unlock_ledger", []):
            self.unlock_ledger[int(ep)] = [
                (int(t), frozenset(int(d) for d in mask), frozenset(int(d) for d in unlock))
                for t, mask, unlock in steps
            ]
        for ep in self.unlock_ledger:
            self._rebuild_first_unlock(ep)

        self._member_latents = {
            k: [[float(v) for v in member] for member in members]
            for k, members in data.get("member_latents", {}).items()
        }
        self._online_card_counter = {}

        if "edges" not in data:
            self.refresh_edges()
