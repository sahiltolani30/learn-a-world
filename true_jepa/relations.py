"""Anonymous relation induction from learned latent transitions.

This module has no vocabulary for objects, actions, recipes, achievements, or
materials.  It stores only vectors produced by the visual model and empirical
outcomes from the interaction stream.  A relation becomes usable only after
repeated successes and failures support it; one transition can never create a
gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


@dataclass
class TransitionEvidence:
    """A single observation/action/success record in representation space."""

    context: np.ndarray
    action: int
    effect: np.ndarray
    outcome: float
    episode: int
    step: int

    def __post_init__(self) -> None:
        self.context = np.asarray(self.context, dtype=np.float32).reshape(-1)
        self.effect = np.asarray(self.effect, dtype=np.float32).reshape(-1)
        if self.context.shape != self.effect.shape:
            raise ValueError("context and effect must have the same latent shape")


@dataclass
class RelationHypothesis:
    """A statistically supported anonymous transition prototype."""

    action: int
    context_success: np.ndarray
    context_failure: np.ndarray
    effect_mean: np.ndarray
    effect_scale: float
    successes: int = 0
    failures: int = 0
    episodes: set[int] = field(default_factory=set)

    @property
    def support(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return (self.successes + 1.0) / (self.support + 2.0)

    @property
    def evidence_confidence(self) -> float:
        """Conservative lower confidence bound for a Bernoulli success rate."""

        if self.support == 0:
            return 0.0
        rate = self.success_rate
        uncertainty = np.sqrt(rate * (1.0 - rate) / max(1, self.support))
        return float(max(0.0, rate - 1.96 * uncertainty))

    @property
    def precondition_direction(self) -> np.ndarray:
        return _unit(self.context_success - self.context_failure)

    def score_context(self, context: np.ndarray) -> float:
        """Return a learned success-vs-failure compatibility score."""

        value = _unit(context)
        return float(
            np.dot(value, _unit(self.context_success))
            - np.dot(value, _unit(self.context_failure))
        )

    def update(self, evidence: TransitionEvidence, success: bool) -> None:
        total_before = self.support
        if success:
            self.successes += 1
            self.context_success += (
                evidence.context - self.context_success
            ) / float(self.successes)
        else:
            self.failures += 1
            self.context_failure += (
                evidence.context - self.context_failure
            ) / float(self.failures)
        self.effect_mean += (evidence.effect - self.effect_mean) / float(total_before + 1)
        magnitude = float(np.linalg.norm(evidence.effect))
        self.effect_scale += (magnitude - self.effect_scale) / float(total_before + 1)
        self.episodes.add(int(evidence.episode))


@dataclass(frozen=True)
class DependencyHypothesis:
    """An empirical enabling edge between two anonymous relations."""

    producer: int
    consumer: int
    alignment: float
    consumer_confidence: float
    effect_scale: float


class AnonymousRelationMemory:
    """Online relation bank over model latents, with no semantic lookup table."""

    def __init__(
        self,
        effect_similarity: float = 0.80,
        minimum_support: int = 8,
        minimum_episodes: int = 3,
        success_margin: float = 0.05,
    ) -> None:
        self.effect_similarity = float(effect_similarity)
        self.minimum_support = int(minimum_support)
        self.minimum_episodes = int(minimum_episodes)
        self.success_margin = float(success_margin)
        self._relations: List[RelationHypothesis] = []

    @property
    def relations(self) -> Tuple[RelationHypothesis, ...]:
        return tuple(self._relations)

    def _match(self, evidence: TransitionEvidence) -> RelationHypothesis | None:
        candidates = [relation for relation in self._relations if relation.action == evidence.action]
        if not candidates:
            return None
        similarities = [
            float(np.dot(_unit(relation.effect_mean), _unit(evidence.effect)))
            for relation in candidates
        ]
        best = int(np.argmax(similarities))
        return candidates[best] if similarities[best] >= self.effect_similarity else None

    def observe(self, evidence: TransitionEvidence) -> int:
        """Add evidence and return the anonymous relation id that absorbed it."""

        success = float(evidence.outcome) > self.success_margin
        relation = self._match(evidence)
        if relation is None:
            relation = RelationHypothesis(
                action=int(evidence.action),
                context_success=evidence.context.copy() if success else np.zeros_like(evidence.context),
                context_failure=np.zeros_like(evidence.context) if success else evidence.context.copy(),
                effect_mean=evidence.effect.copy(),
                effect_scale=float(np.linalg.norm(evidence.effect)),
                successes=int(success),
                failures=int(not success),
                episodes={int(evidence.episode)},
            )
            self._relations.append(relation)
        else:
            relation.update(evidence, success)
        for index, candidate in enumerate(self._relations):
            if candidate is relation:
                return index
        raise RuntimeError("relation disappeared during update")

    def usable(self, relation_id: int) -> bool:
        relation = self._relations[int(relation_id)]
        return (
            relation.support >= self.minimum_support
            and len(relation.episodes) >= self.minimum_episodes
            and relation.evidence_confidence > 0.0
        )

    def rank_actions(self, context: np.ndarray, actions: Sequence[int]) -> List[Tuple[int, float]]:
        """Rank only actions with repeated, statistically supported evidence."""

        ranked: List[Tuple[int, float]] = []
        for action in actions:
            candidates = [
                (relation, index)
                for index, relation in enumerate(self._relations)
                if relation.action == int(action) and self.usable(index)
            ]
            score = max(
                (relation.score_context(context) * relation.evidence_confidence for relation, _ in candidates),
                default=float("-inf"),
            )
            ranked.append((int(action), float(score)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def dependencies(self) -> Tuple[DependencyHypothesis, ...]:
        """Infer candidate enabling edges from learned effects and preconditions."""

        result: List[DependencyHypothesis] = []
        for producer_id, producer in enumerate(self._relations):
            if not self.usable(producer_id):
                continue
            for consumer_id, consumer in enumerate(self._relations):
                if producer_id == consumer_id or not self.usable(consumer_id):
                    continue
                alignment = float(np.dot(_unit(producer.effect_mean), consumer.precondition_direction))
                if alignment > self.effect_similarity:
                    result.append(
                        DependencyHypothesis(
                            producer=producer_id,
                            consumer=consumer_id,
                            alignment=alignment,
                            consumer_confidence=consumer.evidence_confidence,
                            effect_scale=producer.effect_scale,
                        )
                    )
        return tuple(result)

    def state_dict(self) -> Dict[str, object]:
        """Serialize learned vectors and evidence, not environment semantics."""

        return {
            "effect_similarity": self.effect_similarity,
            "minimum_support": self.minimum_support,
            "minimum_episodes": self.minimum_episodes,
            "success_margin": self.success_margin,
            "relations": [
                {
                    "action": relation.action,
                    "context_success": relation.context_success,
                    "context_failure": relation.context_failure,
                    "effect_mean": relation.effect_mean,
                    "effect_scale": relation.effect_scale,
                    "successes": relation.successes,
                    "failures": relation.failures,
                    "episodes": sorted(relation.episodes),
                }
                for relation in self._relations
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "AnonymousRelationMemory":
        bank = cls(
            effect_similarity=float(state["effect_similarity"]),
            minimum_support=int(state["minimum_support"]),
            minimum_episodes=int(state["minimum_episodes"]),
            success_margin=float(state["success_margin"]),
        )
        for raw in state["relations"]:
            bank._relations.append(
                RelationHypothesis(
                    action=int(raw["action"]),
                    context_success=np.asarray(raw["context_success"], dtype=np.float32),
                    context_failure=np.asarray(raw["context_failure"], dtype=np.float32),
                    effect_mean=np.asarray(raw["effect_mean"], dtype=np.float32),
                    effect_scale=float(raw["effect_scale"]),
                    successes=int(raw["successes"]),
                    failures=int(raw["failures"]),
                    episodes=set(int(value) for value in raw["episodes"]),
                )
            )
        return bank
