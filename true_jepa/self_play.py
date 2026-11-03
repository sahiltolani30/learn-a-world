"""Pixel-only self-play collection and T4-sized training coordinator."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol

import numpy as np
import torch

from true_jepa.agent import BatchedVisualJEPAAgent, VisualJEPAAgent
from true_jepa.config import TrainConfig
from true_jepa.model import TrueJEPA
from true_jepa.objectives import imagination_policy_step, train_world_model_step
from true_jepa.relations import AnonymousRelationMemory, TransitionEvidence
from true_jepa.replay import PixelEpisode, PixelReplay
from true_jepa.vector_env import PixelBatch


class PixelEnvironment(Protocol):
    """The only environment contract visible to the collector."""

    num_actions: int

    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]: ...

    def close(self) -> None: ...


class RandomPolicy:
    """Uniform exploration baseline; action count comes from the environment."""

    def __init__(self, num_actions: int, rng: np.random.Generator) -> None:
        self.num_actions = int(num_actions)
        self.rng = rng

    def act(self, _obs: np.ndarray, _reward_prev: float, _step: int) -> int:
        return int(self.rng.integers(0, self.num_actions))


@dataclass
class CollectionStats:
    episodes: int = 0
    frames: int = 0
    extrinsic_reward: float = 0.0


class SelfPlayCoordinator:
    """Alternate interaction, JEPA updates, imagination, and relation learning.

    A population snapshot is a model state produced by this same run.  It is
    used as a data-diversity mechanism, not as an oracle or a semantic policy.
    Every policy receives exactly ``(obs_uint8, reward_prev, step)``.
    """

    def __init__(
        self,
        env_factory: Callable[[], PixelEnvironment],
        model: TrueJEPA,
        train_cfg: TrainConfig | None = None,
        device: str = "cpu",
        seed: int = 0,
        num_envs: int = 1,
        collection_policy: str = "mixed",
    ) -> None:
        self.env_factory = env_factory
        self.model = model.to(device)
        self.train_cfg = train_cfg or TrainConfig()
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if collection_policy not in {"mixed", "random"}:
            raise ValueError("collection_policy must be 'mixed' or 'random'")
        self.collection_policy = collection_policy
        self.rng = np.random.default_rng(seed)
        self.replay = PixelReplay(self.train_cfg.replay_capacity_frames)
        self.relations = AnonymousRelationMemory()
        self.update = 0
        self.env_frames = 0
        self.snapshots: List[Dict[str, torch.Tensor]] = []
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.world_optimizer = torch.optim.AdamW(
            trainable,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.model.heads.policy.parameters(), lr=self.train_cfg.policy_learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            self.model.heads.value.parameters(), lr=self.train_cfg.policy_learning_rate
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.train_cfg.amp and self.device.type == "cuda")

    def _policy(self, source: str):
        if source == "random":
            return RandomPolicy(self.model.num_actions, self.rng)
        if source == "snapshot" and self.snapshots:
            clone = copy.deepcopy(self.model).to(self.device)
            clone.load_state_dict(self.snapshots[int(self.rng.integers(0, len(self.snapshots)))])
            return VisualJEPAAgent(clone, device=str(self.device), stochastic=True)
        return VisualJEPAAgent(self.model, device=str(self.device), stochastic=True)

    def _source(self) -> str:
        draw = float(self.rng.random())
        if draw < 0.15:
            return "random"
        if draw < 0.35 and self.snapshots:
            return "snapshot"
        return "current"

    @torch.no_grad()
    def _learn_relations(self, episode: PixelEpisode, episode_id: int) -> None:
        frames = torch.from_numpy(episode.obs).to(self.device)
        latent = self.model.target_encode(frames).pooled.float().cpu().numpy()
        for step, action in enumerate(episode.actions.tolist()):
            self.relations.observe(
                TransitionEvidence(
                    context=latent[step],
                    action=int(action),
                    effect=latent[step + 1] - latent[step],
                    outcome=float(episode.rewards[step]),
                    episode=episode_id,
                    step=step,
                )
            )

    def collect(self, episodes: int) -> CollectionStats:
        """Collect fresh episodes with current, historical, and random policies."""

        if self.num_envs > 1:
            return self._collect_parallel(episodes)

        stats = CollectionStats()
        for episode_id in range(int(episodes)):
            env = self.env_factory()
            source = "random" if self.collection_policy == "random" else self._source()
            policy = RandomPolicy(self.model.num_actions, self.rng) if source == "random" else self._policy(source)
            obs = env.reset()
            observations = [np.asarray(obs, dtype=np.uint8)]
            actions: List[int] = []
            rewards: List[float] = []
            terminated: List[bool] = []
            truncated: List[bool] = []
            previous_reward = 0.0
            step = 0
            try:
                while True:
                    action = policy.act(observations[-1], previous_reward, step)
                    next_obs, reward, is_terminated, is_truncated = env.step(action)
                    actions.append(int(action))
                    rewards.append(float(reward))
                    terminated.append(bool(is_terminated))
                    truncated.append(bool(is_truncated))
                    observations.append(np.asarray(next_obs, dtype=np.uint8))
                    stats.frames += 1
                    stats.extrinsic_reward += float(reward)
                    previous_reward = float(reward)
                    step += 1
                    if is_terminated or is_truncated:
                        break
            finally:
                env.close()
            episode = PixelEpisode(
                obs=np.stack(observations, axis=0),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray(terminated, dtype=bool),
                truncated=np.asarray(truncated, dtype=bool),
                seed=episode_id,
                source=source,
            )
            self.replay.add(episode)
            stats.episodes += 1
        self.env_frames += stats.frames
        return stats

    def _collect_parallel(self, episodes: int) -> CollectionStats:
        """Collect completed episodes from N lockstep environments."""

        envs = PixelBatch(self.env_factory, self.num_envs)
        policy = (
            None
            if self.collection_policy == "random"
            else BatchedVisualJEPAAgent(self.model, self.num_envs, device=str(self.device))
        )
        observations = [[envs.obs[index].copy()] for index in range(self.num_envs)]
        actions: List[List[int]] = [[] for _ in range(self.num_envs)]
        rewards: List[List[float]] = [[] for _ in range(self.num_envs)]
        terminated: List[List[bool]] = [[] for _ in range(self.num_envs)]
        truncated: List[List[bool]] = [[] for _ in range(self.num_envs)]
        previous_rewards = np.zeros(self.num_envs, dtype=np.float32)
        steps = np.zeros(self.num_envs, dtype=np.int64)
        completed = 0
        stats = CollectionStats()
        try:
            while completed < int(episodes):
                if policy is None:
                    proposed = self.rng.integers(0, envs.num_actions, size=self.num_envs)
                else:
                    proposed = policy.act(envs.obs, previous_rewards, steps)
                    random_mask = self.rng.random(self.num_envs) < 0.10
                    if bool(random_mask.any()):
                        proposed[random_mask] = self.rng.integers(
                            0, envs.num_actions, size=int(random_mask.sum())
                        )
                    policy.record_actions(proposed)
                transitions = envs.step(proposed)
                for index, (terminal_obs, reward, is_terminated, is_truncated) in enumerate(transitions):
                    observations[index].append(terminal_obs)
                    actions[index].append(int(proposed[index]))
                    rewards[index].append(float(reward))
                    terminated[index].append(bool(is_terminated))
                    truncated[index].append(bool(is_truncated))
                    previous_rewards[index] = float(reward)
                    steps[index] += 1
                    stats.frames += 1
                    stats.extrinsic_reward += float(reward)
                    if is_terminated or is_truncated:
                        episode = PixelEpisode(
                            obs=np.stack(observations[index]),
                            actions=np.asarray(actions[index], dtype=np.int64),
                            rewards=np.asarray(rewards[index], dtype=np.float32),
                            terminated=np.asarray(terminated[index], dtype=bool),
                            truncated=np.asarray(truncated[index], dtype=bool),
                            seed=int(envs.episode_ids[index]),
                            source=(
                                "random"
                                if self.collection_policy == "random"
                                else "batched_self_play"
                            ),
                        )
                        self.replay.add(episode)
                        completed += 1
                        stats.episodes += 1
                        observations[index] = [envs.obs[index].copy()]
                        actions[index] = []
                        rewards[index] = []
                        terminated[index] = []
                        truncated[index] = []
                        previous_rewards[index] = 0.0
                        steps[index] = 0
                        if policy is not None:
                            policy.reset(np.asarray([index]))
                        if completed >= int(episodes):
                            break
        finally:
            envs.close()
        self.env_frames += stats.frames
        return stats

    def _refresh_relations(self) -> None:
        """Re-embed replay with the current EMA encoder before inducing edges."""

        self.relations = AnonymousRelationMemory()
        for episode_id, episode in enumerate(self.replay.records):
            self._learn_relations(episode, episode_id)

    def train(
        self, updates: int, train_policy: bool = True
    ) -> List[Dict[str, float]]:
        """Train JEPA and optionally improve policy in imagined rollouts."""

        if self.replay.episodes == 0:
            raise RuntimeError("collect episodes before training")
        metrics: List[Dict[str, float]] = []
        for _ in range(int(updates)):
            batch = self.replay.sample(
                self.train_cfg.batch_size,
                self.train_cfg.sequence_length,
                self.rng,
            )
            result = train_world_model_step(
                self.model,
                batch,
                self.world_optimizer,
                self.device,
                self.train_cfg,
                self.update,
                self.scaler,
            )
            if train_policy:
                start_obs = batch["obs"][:, 0].to(self.device)
                with torch.no_grad():
                    start = self.model.encode(start_obs)
                    hidden = self.model.initial_belief(start_obs.shape[0], self.device)
                policy_result = imagination_policy_step(
                    self.model,
                    start,
                    hidden,
                    self.actor_optimizer,
                    self.value_optimizer,
                    self.model.cfg.rollout_horizon,
                    self.train_cfg.gamma,
                    self.train_cfg.actor_entropy_weight,
                )
                result.update(policy_result)
            metrics.append(result)
            self.update += 1
        self.snapshots.append(copy.deepcopy(self.model.state_dict()))
        if len(self.snapshots) > 4:
            self.snapshots.pop(0)
        self._refresh_relations()
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "format": "true_jepa_run_v1",
                "model": self.model.state_dict(),
                "target_encoder": self.model.target_encoder.state_dict(),
                "world_optimizer": self.world_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "update": self.update,
                "env_frames": self.env_frames,
                "relations": self.relations.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        env_factory: Callable[[], PixelEnvironment],
        model: TrueJEPA,
        train_cfg: TrainConfig | None = None,
        device: str = "cpu",
        seed: int = 0,
        num_envs: int = 1,
        collection_policy: str = "mixed",
        replay_path: str | None = None,
    ) -> "SelfPlayCoordinator":
        """Restore model, optimizers, relation hypotheses, and counters."""

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "true_jepa_run_v1":
            raise ValueError("unexpected true-JEPA run format")
        runner = cls(
            env_factory,
            model,
            train_cfg,
            device=device,
            seed=seed,
            num_envs=num_envs,
            collection_policy=collection_policy,
        )
        runner.model.load_state_dict(payload["model"])
        runner.model.target_encoder.load_state_dict(payload["target_encoder"])
        runner.world_optimizer.load_state_dict(payload["world_optimizer"])
        runner.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        runner.value_optimizer.load_state_dict(payload["value_optimizer"])
        runner.update = int(payload["update"])
        runner.env_frames = int(payload["env_frames"])
        runner.relations = AnonymousRelationMemory.from_state_dict(payload["relations"])
        if replay_path is not None:
            runner.replay = PixelReplay.load(replay_path)
        return runner
