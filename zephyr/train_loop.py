"""Zephyr data collection and Stage-1 joint actor-critic training loop.

FrameCollector steps a ZephyrVecEnv under a random or actor policy, assembles
full-episode records with per-step privileged labels (training targets only),
computes intrinsic rewards (ensemble disagreement + episodic novelty) for the
actor-policy path, and persists shards.

ZephyrTrainer trains world model + actor + critic jointly from replay
(DreamerV3-style): the policy is never stale relative to the encoder, which
structurally avoids the previous line's freeze-then-drift promotion failure.
"""
from __future__ import annotations

import copy
import glob
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from zephyr.ac import (
    actor_lambda_return_loss,
    critic_twohot_loss,
    imagine_trajectory,
    trajectory_lambda_returns,
)
from zephyr.buffer import EpisodeRecord, ZephyrReplay
from zephyr.config import NUM_ACTIONS, ZephyrConfig
from zephyr.explore import EpisodicNovelty
from zephyr.heads import ZephyrHeads
from zephyr.train_wm import _batch_to, build_optimizers, wm_joint_losses
from zephyr.wm import ZephyrWM

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Uniform-random policy; zero intrinsic reward."""

    def __init__(self, num_envs: int, seed: int = 1234) -> None:
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.last_actions = np.full(num_envs, -1, dtype=np.int64)

    def act(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actions = self.rng.integers(0, NUM_ACTIONS, size=self.num_envs).astype(np.int64)
        intrinsic = np.zeros(self.num_envs, dtype=np.float32)
        return actions, intrinsic

    def on_episode_end(self, env_index: int) -> None:
        return None


class ActorPolicy:
    """Batched actor sampling with intrinsic reward (disagreement + novelty).

    Maintains per-env beliefs across steps:
        h_{t+1} = belief_step(z_{t+1}, a_t, h_t),
    with fresh episodes skipping the update (h stays zero until the first action).
    """

    def __init__(
        self,
        wm: ZephyrWM,
        heads: ZephyrHeads,
        num_envs: int,
        device: str,
        disagreement_weight: float = 1.0,
        knn_weight: float = 1.0,
        deterministic: bool = False,
    ) -> None:
        self.wm = wm
        self.heads = heads
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.disagreement_weight = float(disagreement_weight)
        self.knn_weight = float(knn_weight)
        self.deterministic = bool(deterministic)
        self.novelty = [EpisodicNovelty(wm.cfg.explore) for _ in range(num_envs)]
        self.h = torch.zeros(num_envs, wm.cfg.world_model.belief_dim, device=self.device)
        self.fresh = np.ones(num_envs, dtype=bool)
        self.last_actions = np.full(num_envs, -1, dtype=np.int64)
        self._unimix = wm.cfg.ac.actor_unimix

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(self.device)
        z = self.wm.encode(obs)
        cont = torch.from_numpy((~self.fresh).astype(np.float32)).to(self.device)
        prev = torch.from_numpy(np.clip(self.last_actions, 0, None)).to(self.device)
        h_updated = self.wm.belief_step(z, prev, self.h)
        self.h = cont.unsqueeze(1) * h_updated + (1.0 - cont).unsqueeze(1) * self.h

        s = self.heads.state_input(z, self.h)
        logits = self.heads.actor(s)
        probs = (1.0 - self._unimix) * torch.softmax(logits, dim=-1) \
            + self._unimix / logits.shape[-1]
        if self.deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = torch.distributions.Categorical(probs=probs).sample()

        from zephyr.ac import ensemble_disagreement

        preds = self.heads.ensemble(z, prev, self.h)
        disagreement = ensemble_disagreement(preds)

        intrinsic = np.zeros(self.num_envs, dtype=np.float32)
        h_np = self.h.detach().float().cpu().numpy()
        for i in range(self.num_envs):
            nov = self.novelty[i].score(h_np[i])
            self.novelty[i].add(h_np[i])
            intrinsic[i] = (
                self.disagreement_weight * float(disagreement[i]) + self.knn_weight * nov
            )

        actions_np = actions.cpu().numpy().astype(np.int64)
        self.last_actions = actions_np.copy()
        self.fresh[:] = False
        return actions_np, intrinsic

    def on_episode_end(self, env_index: int) -> None:
        self.h[env_index].zero_()
        self.fresh[env_index] = True
        self.last_actions[env_index] = -1
        self.novelty[env_index].reset()


class FrameCollector:
    """Vec-env collection loop assembling full-episode records into replay."""

    def __init__(
        self,
        env,
        policy,
        replay: ZephyrReplay,
        source: str,
        shard_dir: Optional[str] = None,
        shard_every_episodes: int = 10,
    ) -> None:
        self.env = env
        self.policy = policy
        self.replay = replay
        self.source = source
        self.shard_dir = shard_dir
        self.shard_every_episodes = int(shard_every_episodes)
        self._started = False
        self._prev_obs: List[np.ndarray] = []
        self._episodes: List[Dict] = []
        self._pending: List[EpisodeRecord] = []
        self.shard_counter = 0
        self.episodes_collected = 0
        self.achievement_events = 0

    def _new_episode_empty(self, inv: np.ndarray, ach: np.ndarray) -> Dict:
        return {
            "obs": [], "actions": [], "rewards": [], "rewards_int": [], "done": [],
            "inventory": [], "achievements_cum": [], "achievements_new": [],
            "cur_inv": inv, "cur_ach": ach,
        }

    def _ensure_started(self) -> None:
        if self._started:
            return
        obs, infos = self.env.reset()
        n = self.env.num_envs
        self._prev_obs = [obs[i] for i in range(n)]
        self._episodes = [
            self._new_episode_empty(
                np.asarray(infos[i]["zephyr_inventory"], dtype=np.float32),
                np.asarray(infos[i]["zephyr_achievements_cum"], dtype=bool),
            )
            for i in range(n)
        ]
        self._started = True

    def collect(self, target_frames: int) -> Dict[str, int]:
        """Run until ``target_frames`` more frames are collected."""
        self._ensure_started()
        frames_done = 0
        n = self.env.num_envs
        while frames_done < target_frames:
            obs_batch = np.stack(self._prev_obs, axis=0)
            actions, intrinsic = self.policy.act(obs_batch)
            obs2, rewards, dones, infos = self.env.step(actions)
            for i in range(n):
                ep = self._episodes[i]
                ach_cum_now = np.asarray(infos[i]["zephyr_achievements_cum"], dtype=bool)
                ach_new = ach_cum_now & ~ep["cur_ach"]
                ep["obs"].append(self._prev_obs[i])
                ep["actions"].append(int(actions[i]))
                ep["rewards"].append(float(rewards[i]))
                ep["rewards_int"].append(float(intrinsic[i]))
                ep["done"].append(bool(dones[i]))
                ep["inventory"].append(ep["cur_inv"])
                ep["achievements_cum"].append(ep["cur_ach"])
                ep["achievements_new"].append(ach_new)
                self.achievement_events += int(ach_new.sum())
                ep["cur_ach"] = ach_cum_now
                ep["cur_inv"] = np.asarray(infos[i]["zephyr_inventory"], dtype=np.float32)
                if dones[i]:
                    self.replay.add_episode(self._finalize(ep, int(infos[i]["episode_seed"])))
                    self.policy.on_episode_end(i)
                    self._episodes[i] = self._new_episode_empty(
                        np.asarray(infos[i]["next_inventory"], dtype=np.float32),
                        np.asarray(infos[i]["next_achievements_cum"], dtype=bool),
                    )
                self._prev_obs[i] = obs2[i]
            frames_done += n
            if (self.shard_dir is not None and self.shard_every_episodes > 0
                    and len(self._pending) >= self.shard_every_episodes):
                self._flush_shard()
        if self.shard_dir is not None and self._pending:
            self._flush_shard()
        return {
            "frames": frames_done,
            "episodes": self.episodes_collected,
            "achievement_events": self.achievement_events,
        }

    def _finalize(self, ep: Dict, seed: int) -> EpisodeRecord:
        record = EpisodeRecord(
            obs=np.stack(ep["obs"], axis=0).astype(np.uint8),
            actions=np.asarray(ep["actions"], dtype=np.int16),
            rewards=np.asarray(ep["rewards"], dtype=np.float32),
            rewards_int=np.asarray(ep["rewards_int"], dtype=np.float32),
            done=np.asarray(ep["done"], dtype=bool),
            inventory=np.stack(ep["inventory"], axis=0).astype(np.float32),
            achievements_cum=np.stack(ep["achievements_cum"], axis=0).astype(bool),
            achievements_new=np.stack(ep["achievements_new"], axis=0).astype(bool),
            seed=seed,
            source=self.source,
        )
        self.episodes_collected += 1
        self._pending.append(record)
        return record

    def _flush_shard(self) -> None:
        if not self._pending or self.shard_dir is None:
            return
        os.makedirs(self.shard_dir, exist_ok=True)
        tmp_replay = ZephyrReplay(capacity_frames=None)
        for rec in self._pending:
            tmp_replay.add_episode(rec)
        path = os.path.join(self.shard_dir, f"zephyr_shard_{self.shard_counter:06d}.npz")
        tmp_path = path + ".tmp.npz"
        tmp_replay.save_shard(tmp_path)
        os.replace(tmp_path, path)
        self.shard_counter += 1
        self._pending = []

    def flush_open_episodes(self) -> int:
        """Finalize any in-flight episodes so their collected frames are not discarded."""
        added = 0
        n = self.env.num_envs
        for i in range(n):
            ep = self._episodes[i]
            if len(ep["obs"]) > 0:
                seed = int(getattr(self.env, "base_seed", 0) + i)
                rec = self._finalize(ep, seed=seed)
                self.replay.add_episode(rec)
                added += 1
                self._episodes[i] = self._new_episode_empty(ep["cur_inv"], ep["cur_ach"])
        if self.shard_dir is not None and self._pending:
            self._flush_shard()
        return added



# ---------------------------------------------------------------------------
# Stage-1 joint trainer
# ---------------------------------------------------------------------------


class ZephyrTrainer:
    """Resumable S1 trainer: WM + actor-critic + exploration, joint updates."""

    def __init__(
        self,
        cfg: ZephyrConfig,
        device: str = "cuda",
        run_dir: str = "zephyr_runs/z1",
        load_checkpoints: Optional[Tuple[str, str]] = None,
        env_in_process: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.run_dir = run_dir
        self.env_in_process = bool(env_in_process)
        self.data_dir = os.path.join(run_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

        if load_checkpoints is not None:
            self.wm, _ = ZephyrWM.load_checkpoint(load_checkpoints[0], device=device)
            self.heads, _ = ZephyrHeads.load_checkpoint(load_checkpoints[1], device=device)
        else:
            self.wm = ZephyrWM(cfg).to(self.device)
            self.heads = ZephyrHeads(cfg).to(self.device)
        self.slow_critic = copy.deepcopy(self.heads.critic).to(self.device)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.slow_critic.eval()

        self.optimizers = build_optimizers(self.wm, self.heads, cfg, include_ac=True)
        self.replay = ZephyrReplay(capacity_frames=cfg.data.replay_capacity_frames)
        self.collector: Optional[FrameCollector] = None
        self.collector_generation = 0
        self.env_frames = 0
        self.train_steps = 0
        self.total_episodes = 0
        self._amp_scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp())
        self._rng = np.random.default_rng(cfg.data.split_seed + 3)

    def _use_amp(self) -> bool:
        return self.device.type == "cuda" and self.cfg.optim.fp16

    # -- collection ---------------------------------------------------------------

    def _ensure_collector(self, source: str) -> FrameCollector:
        if self.collector is not None:
            return self.collector
        from zephyr.vec_env import ZephyrVecEnv

        base_seed = int(self.cfg.data.train_seed_base + 1_000_000_000 * self.collector_generation)
        env = ZephyrVecEnv(
            num_envs=self.cfg.data.num_envs,
            base_seed=base_seed,
            max_steps=self.cfg.data.episode_max_steps,
            in_process=self.env_in_process,
        )
        policy = ActorPolicy(
            self.wm, self.heads, self.cfg.data.num_envs, device=str(self.device),
            disagreement_weight=self.cfg.explore.disagreement_weight,
            knn_weight=self.cfg.explore.knn_weight,
        )
        self.collector = FrameCollector(
            env, policy, self.replay, source=source, shard_dir=self.data_dir,
            shard_every_episodes=10,
        )
        self.collector.episodes_collected = 0
        return self.collector

    def collect(self, frames: int, source: str = "actor") -> Dict[str, int]:
        collector = self._ensure_collector(source)
        before = collector.episodes_collected
        stats = collector.collect(frames)
        self.env_frames += stats["frames"]
        self.total_episodes += collector.episodes_collected - before
        return stats

    # -- training -------------------------------------------------------------------

    def train(self, steps: int, log_every: int = 200, on_log=None) -> Dict[str, float]:
        cfg = self.cfg
        dev = self.device
        last: Dict[str, float] = {}
        for _ in range(steps):
            batch = self.replay.sample(
                cfg.data.batch_size, cfg.data.seq_len, self._rng,
                prioritize_achievement_frac=cfg.data.prioritize_achievement_frac,
            )
            batch = _batch_to(batch, dev)
            with torch.autocast("cuda", enabled=self._use_amp()):
                total_wm, wm_metrics, states = wm_joint_losses(
                    self.wm, self.heads, batch, cfg, dev, return_states=True
                )
                z_all, h_pre = states["z_now"], states["h_pre"]
                stride = max(1, cfg.ac.imag_start_stride)
                idx = list(range(0, z_all.shape[1] - 1, stride))
                z0 = torch.cat([z_all[:, i] for i in idx], dim=0)
                h0 = torch.cat([h_pre[i] for i in idx], dim=0)

                imagination = imagine_trajectory(
                    self.wm, self.heads, z0, h0,
                    horizon=cfg.ac.imagination_horizon, cfg=cfg,
                )
                beta = cfg.explore.beta(self.env_frames)
                intrinsic_weight = beta * cfg.explore.disagreement_weight
                with torch.no_grad():
                    slow_logits = self.slow_critic(imagination["s"][-1])
                lam_ret = trajectory_lambda_returns(
                    imagination, self.heads.critic.coder, cfg,
                    intrinsic_weight=intrinsic_weight, slow_value_logits=slow_logits,
                )
                actor_loss = actor_lambda_return_loss(
                    lam_ret, imagination["log_prob"], imagination["entropy"], cfg
                )
                v_logits = imagination["value_logits"][:-1]
                critic_loss = critic_twohot_loss(
                    v_logits.reshape(-1, v_logits.shape[-1]),
                    lam_ret.reshape(-1),
                    self.heads.critic.coder,
                )
                total = (
                    total_wm
                    + cfg.ac.critic_loss_weight * critic_loss
                    + actor_loss
                )

            self._amp_scaler.scale(total).backward()
            if self._use_amp():
                for opt in self.optimizers.values():
                    self._amp_scaler.unscale_(opt)
            for name, opt in self.optimizers.items():
                torch.nn.utils.clip_grad_norm_(
                    [p for group in opt.param_groups for p in group["params"]],
                    cfg.optim.grad_clip,
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
            self._amp_scaler.update()

            self.train_steps += 1
            self.wm.update_ema(self.train_steps)
            with torch.no_grad():
                decay = cfg.ac.slow_target_decay
                for p_slow, p in zip(self.slow_critic.parameters(),
                                     self.heads.critic.parameters()):
                    p_slow.mul_(decay).add_(p, alpha=1.0 - decay)

            last = {**wm_metrics,
                    "actor_loss": float(actor_loss.detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "lambda_return_mean": float(lam_ret.mean().detach()),
                    "intrinsic_weight": intrinsic_weight}
            if on_log is not None and self.train_steps % log_every == 0:
                on_log(self.train_steps, last)
        return last

    # -- checkpointing ----------------------------------------------------------------

    def save_run(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.run_dir, "state.pt")
        payload = {
            "format": "zephyr_run_v1",
            "config": self.cfg.to_dict(),
            "env_frames": self.env_frames,
            "train_steps": self.train_steps,
            "total_episodes": self.total_episodes,
            "collector_generation": self.collector_generation,
            "wm_state": self.wm.state_dict(),
            "heads_state": self.heads.state_dict(),
            "slow_critic_state": self.slow_critic.state_dict(),
            "opt_states": {k: v.state_dict() for k, v in self.optimizers.items()},
            "rng": {
                "torch": torch.get_rng_state(),
                "np": self._rng.bit_generator.state,
                "python": random.getstate(),
            },
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def save_checkpoints(self, wm_path: str, heads_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(wm_path)), exist_ok=True)
        self.wm.save_checkpoint(wm_path + ".tmp", step=self.train_steps)
        self.heads.save_checkpoint(heads_path + ".tmp", step=self.train_steps)
        os.replace(wm_path + ".tmp", wm_path)
        os.replace(heads_path + ".tmp", heads_path)

    @classmethod
    def load_run(cls, path: str, device: str = "cuda") -> "ZephyrTrainer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "zephyr_run_v1":
            raise ValueError(f"unexpected run format {payload.get('format')!r}")
        cfg = ZephyrConfig.from_dict(payload["config"])
        trainer = cls(cfg, device=device, run_dir=os.path.dirname(os.path.abspath(path)))
        trainer.wm.load_state_dict(payload["wm_state"])
        trainer.heads.load_state_dict(payload["heads_state"])
        trainer.slow_critic.load_state_dict(payload["slow_critic_state"])
        for name, state in payload["opt_states"].items():
            trainer.optimizers[name].load_state_dict(state)
        trainer.env_frames = int(payload["env_frames"])
        trainer.train_steps = int(payload["train_steps"])
        trainer.total_episodes = int(payload["total_episodes"])
        trainer.collector_generation = int(payload.get("collector_generation", 0))
        torch.set_rng_state(payload["rng"]["torch"])
        trainer._rng.bit_generator.state = payload["rng"]["np"]
        random.setstate(payload["rng"]["python"])
        trainer.load_shards(trainer.data_dir)
        return trainer

    def load_shards(self, shard_dir: str) -> int:
        """Load persisted shards (most recent last → FIFO keeps the newest)."""
        paths = sorted(glob.glob(os.path.join(shard_dir, "zephyr_shard_*.npz")))
        for path in paths:
            shard = ZephyrReplay.load_shard(path)
            for rec in shard.records:
                self.replay.add_episode(rec)
        if paths:
            self.collector_generation = self.collector_generation + 1
        return len(paths)

    @staticmethod
    def _generation_from_paths(paths: List[str]) -> int:
        del paths  # generation is derived from run state, not shard names
        return 0

    # -- monitoring -----------------------------------------------------------------

    def run_dev_screen(self, out_path: str, agent_mode: str = "actor") -> Dict:
        """Quick screen on held-out dev seeds (never the locked benchmark seeds)."""
        from zephyr.screen import run_screen

        wm_path = os.path.join(self.run_dir, "ckpt", "dev_wm.pt")
        heads_path = os.path.join(self.run_dir, "ckpt", "dev_heads.pt")
        self.save_checkpoints(wm_path, heads_path)
        return run_screen(
            wm_path, heads_path, out_path=out_path,
            num_episodes=self.cfg.data.dev_screen_episodes,
            seed_start=self.cfg.data.dev_screen_seed_start,
            max_steps=self.cfg.data.episode_max_steps,
            agent_mode=agent_mode, device=str(self.device),
        )
