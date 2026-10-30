"""
src/train.py

Production-Grade, Modular Training Pipeline for Crafter Tabula Rasa JEPA World Model
with Consolidation Redesign (ROOT_CAUSE_FIX_PLAN.md):
- C1: Critic learns pure external reward (critic_intrinsic_weight=0.0) preventing Q fossilization.
- C2: Milestone-prioritized sampling with retroactive 30-step trailing credit window.
- C3: n-step (5-step) Bellman critic targets propagating external rewards along trajectories.
- C4: Per-action normalized predictor disagreement in actor loss + episodic k-NN novelty-modulated epsilon.
- C5: High-throughput gradient update interval (update_every=2, batch_size=128, optional action_repeat).
- C6: Dual stochastic and greedy evaluation reporting.
- C7: Telemetry tracking Q spread, external Q mean, intrinsic curiosity, and non-collapse invariants.

CLI Arguments:
  --device: Computing device ('cpu', 'cuda', 'mps', or auto-detect).
  --total_steps: Total environment interaction steps (default: 10,000).
  --batch_size: Replay buffer training batch size (default: 128).
  --checkpoint_dir: Path to directory for persistent checkpoints (default: './checkpoints').
  --eval_interval: Frequency of evaluation episodes in steps (default: 1,000).
  --seed: Random seed for environment, buffers, and network initialization (default: 42).
  --eval_episodes: Number of episodes per evaluation cycle (default: 3).
  --critic_intrinsic_weight: Weight for intrinsic reward in critic TD backup (default: 0.0).
  --nstep: Lookahead horizon for n-step TD critic targets (default: 5).
  --actor_intrinsic_weight: Weight for per-action normalized curiosity in actor loss (default: 0.1).
  --eps_base, --eps_novelty_coef, --eps_max: Novelty-modulated epsilon exploration parameters.
  --milestone_t1_priority, --milestone_t2_priority: Priority sampling rates for milestones in T1 and T2.
  --update_every: Environment steps between gradient update steps (default: 2).
  --action_repeat: Environment frames per agent action (default: 1).
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Set
from collections import deque
from pathlib import Path
import os
import sys
import time
import argparse
import shutil
import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.wrapper import make_crafter_env, FORBIDDEN_CHEAT_KEYS
from src.env.parallel import ParallelCrafterBatch
from src.agent.agent import JEPAAgent
from src.curiosity.ensemble import EpistemicCuriosityEngine, RunningMeanStd
from src.curiosity.episodic import EpisodicNovelty
from src.memory.buffer import TwoTierReplayBuffer
from src.memory.storage import CheckpointManager
from src.models.losses import VICRegLoss, verify_non_collapse

# Safe RAM telemetry utility (Rule R3)
try:
    import psutil
    def get_ram_percent() -> float:
        return float(psutil.virtual_memory().percent)
except ImportError:
    def get_ram_percent() -> float:
        return 0.0


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the JEPA training pipeline."""
    parser = argparse.ArgumentParser(
        description="Tabula Rasa JEPA World Model Training on Crafter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required CLI arguments from Milestone 5 specification
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=10_000,
        help="Total environment interaction steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for world model and policy updates",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save/load persistent model and memory checkpoints",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1_000,
        help="Interval in steps between episodic evaluation cycles",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Modular training configuration options
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes per eval cycle",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for visual encoder and latent predictor",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=3e-4,
        help="Learning rate for twin critic networks",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Initial exploration steps before performing model updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1_000,
        help="Interval in steps for saving periodic checkpoints",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable auto-resumption from existing checkpoint",
    )
    parser.add_argument(
        "--tier2_ratio",
        type=float,
        default=0.30,
        help="Proportion of Tier 2 milestone samples in mixed batches",
    )
    parser.add_argument(
        "--milestone_t1_priority",
        type=float,
        default=0.25,
        help="Sampling priority for milestone transitions in Tier 1 (default 0.25)",
    )
    parser.add_argument(
        "--milestone_t2_priority",
        type=float,
        default=0.80,
        help="Sampling priority for milestone transitions in Tier 2 (default 0.80)",
    )
    parser.add_argument(
        "--critic_intrinsic_weight",
        type=float,
        default=0.0,
        help="Weight for intrinsic curiosity rewards inside critic Bellman target (default 0.0 = pure external reward)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for RL Bellman backup",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.99,
        help="Polyak momentum coefficient for EMA target updates",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval in steps for printing training progress",
    )
    parser.add_argument(
        "--update_every",
        type=int,
        default=2,
        help="Gradient update interval in env steps (default 2)",
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=1,
        help="Action repeat count per env interaction (default 1)",
    )
    parser.add_argument(
        "--actor_intrinsic_weight",
        type=float,
        default=0.1,
        help="Weight for per-action normalized disagreement in actor loss (default 0.1)",
    )
    parser.add_argument(
        "--eps_base",
        type=float,
        default=0.05,
        help="Base epsilon exploration rate (default 0.05)",
    )
    parser.add_argument(
        "--eps_novelty_coef",
        type=float,
        default=0.20,
        help="Coefficient scaling episodic novelty for epsilon modulation (default 0.20)",
    )
    parser.add_argument(
        "--eps_max",
        type=float,
        default=0.50,
        help="Maximum epsilon exploration ceiling (default 0.50)",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=5,
        help="Horizon n for n-step TD critic targets (default 5)",
    )
    # Parallel environment batching arguments (PARALLEL_ENV_PLAN.md)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Number of parallel environments stepped in lockstep (default 8, set 1 for serial)",
    )
    parser.add_argument(
        "--buffer_save_every",
        type=int,
        default=5,
        help="Save replay buffer arrays to disk only every Nth checkpoint (default 5, RAM Rule R2)",
    )
    parser.add_argument(
        "--ram_abort_percent",
        type=float,
        default=80.0,
        help="Emergency save and abort threshold on system RAM percentage (default 80.0, RAM Rule R3)",
    )

    return parser.parse_args(args)


class TrainingPipeline:
    """
    Modular JEPA Training Pipeline coordinating:
    - Environment interaction & Zero-Cheat observation handling
    - Experience accumulation in Two-Tier Replay Buffer
    - Self-supervised World Model updates via VICRegLoss
    - Latent Policy & Twin Critic updates
    - Episodic evaluation & achievement curve tracking
    - Atomic checkpoint persistence and auto-resumption
    """

    def __init__(self, config: Union[argparse.Namespace, Dict[str, Any]]):
        if isinstance(config, argparse.Namespace):
            self.cfg = vars(config)
        else:
            self.cfg = dict(config)

        self.device_str = self.cfg.get("device", "cpu")
        if self.device_str == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)

        self.total_steps: int = int(self.cfg.get("total_steps", 10_000))
        self.batch_size: int = int(self.cfg.get("batch_size", 128))
        self.checkpoint_dir: Path = Path(self.cfg.get("checkpoint_dir", "./checkpoints")).expanduser()
        self.eval_interval: int = int(self.cfg.get("eval_interval", 1_000))
        self.seed: int = int(self.cfg.get("seed", 42))
        self.eval_episodes: int = int(self.cfg.get("eval_episodes", 3))
        self.lr: float = float(self.cfg.get("lr", 1e-4))
        self.critic_lr: float = float(self.cfg.get("critic_lr", 3e-4))
        self.warmup_steps: int = int(self.cfg.get("warmup_steps", 100))
        self.save_interval: int = int(self.cfg.get("save_interval", 1_000))
        self.resume: bool = not bool(self.cfg.get("no_resume", False))
        self.tier2_ratio: float = float(self.cfg.get("tier2_ratio", 0.30))
        self.milestone_t1_priority: float = float(self.cfg.get("milestone_t1_priority", 0.25))
        self.milestone_t2_priority: float = float(self.cfg.get("milestone_t2_priority", 0.80))
        self.critic_intrinsic_weight: float = float(self.cfg.get("critic_intrinsic_weight", 0.0))
        self.gamma: float = float(self.cfg.get("gamma", 0.99))
        self.tau: float = float(self.cfg.get("tau", 0.99))
        self.log_interval: int = int(self.cfg.get("log_interval", 100))
        self.update_every: int = int(self.cfg.get("update_every", 2))
        self.action_repeat: int = int(self.cfg.get("action_repeat", 1))
        self.actor_intrinsic_weight: float = float(self.cfg.get("actor_intrinsic_weight", 0.1))
        self.eps_base: float = float(self.cfg.get("eps_base", 0.05))
        self.eps_novelty_coef: float = float(self.cfg.get("eps_novelty_coef", 0.20))
        self.eps_max: float = float(self.cfg.get("eps_max", 0.50))
        self.nstep: int = int(self.cfg.get("nstep", 5))
        
        # Parallel Environment Configuration (PARALLEL_ENV_PLAN.md)
        self.num_envs: int = int(self.cfg.get("num_envs", 8))
        self.buffer_save_every: int = int(self.cfg.get("buffer_save_every", 5))
        self.ram_abort_percent: float = float(self.cfg.get("ram_abort_percent", 80.0))
        self.checkpoint_count: int = 0
        self.vector_step: int = 0

        self.recent_t1_indices: deque = deque(maxlen=30)
        self.recent_t1_indices_per_env: List[deque] = [
            deque(maxlen=30) for _ in range(self.num_envs)
        ]
        self.env_episode_ids: List[int] = [i for i in range(self.num_envs)]
        self.episode_achievements: List[Set[str]] = [set() for _ in range(self.num_envs)]
        self.episodic_novelties: List[EpisodicNovelty] = [
            EpisodicNovelty(latent_dim=256, seed=self.seed + 100 * i)
            for i in range(self.num_envs)
        ]

        self.total_milestones: int = 0
        self.success_counts: Dict[str, int] = {}
        self.recent_eps: deque = deque(maxlen=1000)

        # C4: Per-action normalized curiosity running statistics and episodic novelty
        self.per_action_disc_stats = RunningMeanStd(shape=(17,)).to(self.device)
        self.episodic_novelty = self.episodic_novelties[0]
        self.episodic_stats = RunningMeanStd(shape=())

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        # 1. Environment Initialization (N parallel envs + 1 standalone eval env)
        self.envs = ParallelCrafterBatch(num_envs=self.num_envs, base_seed=self.seed)
        self.env = self.envs.envs[0]
        self.eval_env = make_crafter_env(seed=self.seed + 1000)

        # 2. Agent Initialization
        self.agent = JEPAAgent(
            latent_dim=256,
            num_actions=17,
            device=self.device,
            tau=self.tau,
            gamma=self.gamma,
        )

        # 3. Two-Tier Replay Buffer (Tier 1 capped at 50,000 to save 1.2 GB RAM in Colab)
        self.buffer = TwoTierReplayBuffer(
            tier1_capacity=int(self.cfg.get("tier1_capacity", 50_000)),
            tier2_capacity=20_000,
            default_tier2_ratio=self.tier2_ratio,
            milestone_t1_priority=self.milestone_t1_priority,
            milestone_t2_priority=self.milestone_t2_priority,
            seed=self.seed,
        )

        # 4. Losses & Optimizers
        self.vicreg_loss = VICRegLoss(
            sim_coeff=1.0,
            var_coeff=1.0,
            cov_coeff=0.04,
            gamma=1.0,
        )

        self.wm_optimizer = torch.optim.Adam(
            list(self.agent.encoder.parameters()) + list(self.agent.predictor.parameters()),
            lr=self.lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.agent.critic.q1_net.parameters()) + list(self.agent.critic.q2_net.parameters()),
            lr=self.critic_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(),
            lr=self.lr,
        )

        # 5. Checkpoint Management & Auto-Resumption
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            keep_last_n=5,
        )

        # Training telemetry tracking
        self.current_step: int = 0
        self.current_episode: int = 0
        self.best_eval_return: float = -float("inf")
        self.achievement_history: Dict[int, Dict[str, Any]] = {}
        self.all_unlocked_achievements: Set[str] = set()
        self.loss_history: List[Dict[str, float]] = []

        # Auto-resumption if configured
        if self.resume:
            self.try_auto_resume()

    def try_auto_resume(self) -> bool:
        """
        Attempts to discover and restore the most recent checkpoint.
        Returns True if a checkpoint was restored, False otherwise.
        """
        latest_ckpt = self.checkpoint_manager.find_latest_checkpoint()
        if latest_ckpt is not None:
            print(f"[*] Found existing checkpoint: {latest_ckpt}")
            try:
                metadata = self.checkpoint_manager.restore(
                    filepath=latest_ckpt,
                    components={
                        "agent": self.agent,
                        "buffer": self.buffer,
                        "wm_optimizer": self.wm_optimizer,
                        "critic_optimizer": self.critic_optimizer,
                        "actor_optimizer": self.actor_optimizer,
                    },
                    device=self.device,
                    strict=False,
                )
                self.current_step = metadata.get("step", 0)
                self.vector_step = metadata.get("vector_step", self.current_step // self.num_envs)
                self.checkpoint_count = metadata.get("checkpoint_count", 0)
                self.current_episode = metadata.get("episode", 0)
                self.env_episode_ids = [self.current_episode + i for i in range(self.num_envs)]
                self.best_eval_return = metadata.get("best_eval_return", -float("inf"))
                self.achievement_history = metadata.get("achievement_history", {})
                self.all_unlocked_achievements = set(metadata.get("all_unlocked_achievements", []))
                if "success_counts" in metadata:
                    self.success_counts = dict(metadata["success_counts"])
                print(
                    f"[+] Resumed successfully at step {self.current_step}, "
                    f"episode {self.current_episode}, best return {self.best_eval_return:.2f}"
                )
                return True
            except Exception as e:
                print(f"[!] Warning: Failed to restore checkpoint at {latest_ckpt}: {e}")
                return False
        return False

    def collect_batch_step(self) -> int:
        """
        Executes one vector step across all N parallel environments in lockstep.
        Batches visual observation encoding and curiosity prediction into single tensor operations.
        
        Returns:
            transitions_added: Number of environment transitions added to the replay buffer.
        """
        obs_np = self.envs.obs  # (N, 3, 64, 64) uint8
        obs_batch = torch.from_numpy(obs_np).to(self.device).float() / 255.0

        with torch.no_grad():
            z = self.agent.encoder(obs_batch)  # (N, D)

            # Epsilon-greedy action selection per environment
            if self.vector_step < self.warmup_steps:
                action_list = [int(self.envs.action_space.sample()) for _ in range(self.num_envs)]
            else:
                probs = self.agent.actor.get_probs(z)  # (N, 17)
                actions_t = torch.multinomial(probs, 1).squeeze(-1)
                action_list = [int(a.item()) for a in actions_t]

                # Per-env novelty-modulated epsilon (PARALLEL_ENV_PLAN.md §3.2)
                for env_i in range(self.num_envs):
                    z_i = z[env_i].detach().cpu().numpy()
                    nov_raw = self.episodic_novelties[env_i].novelty_and_insert(z_i)
                    self.episodic_stats.update(nov_raw)
                    nov = float(np.clip(nov_raw / (float(self.episodic_stats.std) + 1e-4), 0.0, 5.0))
                    eps = min(self.eps_max, self.eps_base + self.eps_novelty_coef * nov)
                    self.recent_eps.append(eps)
                    if float(np.random.random_sample()) < eps:
                        action_list[env_i] = int(self.envs.action_space.sample())

            # Batched predictor forward pass for disagreement and intrinsic curiosity
            actions_tensor = torch.tensor(action_list, device=self.device, dtype=torch.long)
            _, all_preds = self.agent.predictor(z, actions_tensor)
            disagreement = self.agent.curiosity_engine.compute_epistemic_disagreement(all_preds)
            r_int_tensor = self.agent.curiosity_engine.compute_intrinsic_reward(all_preds, update_stats=True)
            r_int_np = r_int_tensor.detach().cpu().numpy()
            disagree_np = disagreement.detach().cpu().numpy()

        # Step all environments in lockstep
        old_obs = obs_np.copy()
        step_results = self.envs.step(action_list)

        transitions_added = 0
        for env_i in range(self.num_envs):
            if self.current_step >= self.total_steps:
                break

            terminal_obs_i, ext_r, done_i, info_i = step_results[env_i]

            for forbidden in FORBIDDEN_CHEAT_KEYS:
                if forbidden in info_i:
                    raise AssertionError(
                        f"CRITICAL ZERO-CHEAT LEAK: Forbidden key '{forbidden}' found in env info!"
                    )

            is_milestone = False
            eval_info = info_i.get("eval_info", {})
            unlocked = eval_info.get("unlocked_achievements", [])
            for ach in unlocked:
                if ach not in self.episode_achievements[env_i]:
                    self.episode_achievements[env_i].add(ach)
                    is_milestone = True
                    if ach not in self.all_unlocked_achievements:
                        self.all_unlocked_achievements.add(ach)
                        print(f" [★ DISCOVERY] Env {env_i} Step {self.current_step + 1}: Unlocked achievement '{ach}'!")

            r_int_i = float(r_int_np[env_i]) if r_int_np.ndim > 0 else float(r_int_np)
            if r_int_i > 0.5:
                is_milestone = True

            act = action_list[env_i]
            if ext_r >= 0.5:
                act_names = getattr(self.env, "action_names", None)
                act_label = act_names[act] if act_names is not None else str(act)
                self.success_counts[act_label] = self.success_counts.get(act_label, 0) + 1

            err_i = float(disagree_np[env_i]) if disagree_np.ndim > 0 else float(disagree_np)
            self.buffer.add(
                obs=old_obs[env_i],
                action=act,
                reward=ext_r,
                next_obs=terminal_obs_i,
                done=done_i,
                r_int=r_int_i,
                is_milestone=is_milestone,
                epistemic_error=err_i,
                episode_id=self.env_episode_ids[env_i],
                step_in_ep=self.envs.episode_step[env_i],
            )
            written_slot = (self.buffer.t1_ptr - 1) % self.buffer.tier1_capacity
            self.recent_t1_indices_per_env[env_i].append(written_slot)
            self.recent_t1_indices.append(written_slot)
            if is_milestone:
                self.total_milestones += 1
                for past_idx in self.recent_t1_indices_per_env[env_i]:
                    self.buffer.flag_milestone_at(past_idx)

            if done_i:
                self.current_episode += 1
                self.env_episode_ids[env_i] = self.current_episode
                self.episode_achievements[env_i].clear()
                self.recent_t1_indices_per_env[env_i].clear()
                self.episodic_novelties[env_i].reset()

            self.current_step += 1
            transitions_added += 1

        self.vector_step += 1
        return transitions_added

    def collect_step(
        self,
        obs: torch.Tensor,
        episode_achievements: Set[str],
        episode_id: int = -1,
        step_in_ep: int = -1,
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Interacts with Crafter for 1 step, computes intrinsic curiosity,
        audits for zero-cheat compliance, and stores transition in TwoTierReplayBuffer.

        Args:
            obs: Normalized observation tensor (3, 64, 64).
            episode_achievements: Set tracking achievements unlocked in current episode.
            episode_id: Unique identifier for current life/episode.
            step_in_ep: Sequential timestep offset within current episode.

        Returns:
            next_obs, reward, done, step_metrics
        """
        # C4.2: Latent encoding, episodic novelty, and novelty-modulated exploration
        with torch.no_grad():
            z_t = self.agent.encode(obs)
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)

        nov_raw = self.episodic_novelty.novelty_and_insert(z_t.detach().squeeze(0).cpu().numpy())
        self.episodic_stats.update(nov_raw)
        nov = float(np.clip(nov_raw / (float(self.episodic_stats.std) + 1e-4), 0.0, 5.0))

        # Select action via novelty-modulated epsilon-greedy
        if self.current_step < self.warmup_steps:
            action = int(self.env.action_space.sample())
        else:
            eps = min(self.eps_max, self.eps_base + self.eps_novelty_coef * nov)
            self.recent_eps.append(eps)
            if float(np.random.random_sample()) < eps:
                action = int(self.env.action_space.sample())
            else:
                action = self.agent.act(obs, eval_mode=False, planning=False)

        # Step environment (supporting action_repeat)
        is_milestone = False
        if self.action_repeat <= 1:
            next_obs, ext_reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            for forbidden in FORBIDDEN_CHEAT_KEYS:
                if forbidden in info:
                    raise AssertionError(
                        f"CRITICAL ZERO-CHEAT LEAK: Forbidden key '{forbidden}' found in env info!"
                    )
            eval_info = info.get("eval_info", {})
            unlocked = eval_info.get("unlocked_achievements", [])
            for ach in unlocked:
                if ach not in episode_achievements:
                    episode_achievements.add(ach)
                    is_milestone = True
                    if ach not in self.all_unlocked_achievements:
                        self.all_unlocked_achievements.add(ach)
                        print(f" [★ DISCOVERY] Step {self.current_step}: Unlocked achievement '{ach}'!")
        else:
            total_ext_reward = 0.0
            done = False
            for _ in range(self.action_repeat):
                next_obs, step_reward, terminated, truncated, info = self.env.step(action)
                total_ext_reward += float(step_reward)
                done = bool(terminated or truncated)
                for forbidden in FORBIDDEN_CHEAT_KEYS:
                    if forbidden in info:
                        raise AssertionError(
                            f"CRITICAL ZERO-CHEAT LEAK: Forbidden key '{forbidden}' found in env info!"
                        )
                eval_info = info.get("eval_info", {})
                unlocked = eval_info.get("unlocked_achievements", [])
                for ach in unlocked:
                    if ach not in episode_achievements:
                        episode_achievements.add(ach)
                        is_milestone = True
                        if ach not in self.all_unlocked_achievements:
                            self.all_unlocked_achievements.add(ach)
                            print(f" [★ DISCOVERY] Step {self.current_step}: Unlocked achievement '{ach}'!")
                if done:
                    break
            ext_reward = total_ext_reward

        # Compute epistemic curiosity reward for exploration
        with torch.no_grad():
            a_t = torch.tensor([action], device=self.device, dtype=torch.long)
            _, all_preds = self.agent.predictor(z_t, a_t)
            r_int_tensor = self.agent.curiosity_engine.compute_intrinsic_reward(all_preds, update_stats=True)
            r_int = float(r_int_tensor.item() if r_int_tensor.numel() == 1 else r_int_tensor[0].item())

        # High epistemic disagreement also qualifies as milestone discovery
        if r_int > 0.5:
            is_milestone = True

        # Insert transition into Two-Tier Buffer
        self.buffer.add(
            obs=obs,
            action=action,
            reward=ext_reward,
            next_obs=next_obs,
            done=done,
            r_int=r_int,
            is_milestone=is_milestone,
            epistemic_error=r_int,
            episode_id=episode_id,
            step_in_ep=step_in_ep,
        )

        # C2.2: Capture written T1 slot and retroactively flag trailing 30-step window on milestone
        written_slot = (self.buffer.t1_ptr - 1) % self.buffer.tier1_capacity
        self.recent_t1_indices.append(written_slot)
        if is_milestone:
            for slot_idx in self.recent_t1_indices:
                self.buffer.flag_milestone_at(slot_idx)
            self.total_milestones += 1

        # C7: Track external reward successes (e.g. chop wood, place table)
        if ext_reward >= 0.5:
            act_names = getattr(self.env, "action_names", None)
            act_label = act_names[action] if act_names is not None else str(action)
            self.success_counts[act_label] = self.success_counts.get(act_label, 0) + 1

        step_metrics = {
            "ext_reward": ext_reward,
            "int_reward": r_int,
            "is_milestone": is_milestone,
            "done": done,
        }

        return next_obs, ext_reward, done, step_metrics

    def update_world_model(self) -> Dict[str, float]:
        """
        Samples a mixed batch (70% Tier 1, 30% Tier 2) and optimizes the JEPA
        visual encoder and 4-head predictor ensemble via VICRegLoss.
        Also performs Polyak EMA update of the target encoder.
        """
        self.agent.train()
        batch = self.buffer.sample(
            batch_size=self.batch_size,
            tier2_ratio=self.tier2_ratio,
            device=self.device,
        )

        # 1. Forward passes
        z_curr = self.agent.encoder(batch["obs"])
        with torch.no_grad():
            z_target = self.agent.target_encoder(batch["next_obs"])

        mean_pred, all_preds = self.agent.predictor(z_curr, batch["actions"])

        # 2. VICReg Loss computation
        loss_dict = self.vicreg_loss(all_preds, z_target)

        # Direct representation variance regularization to guarantee non-collapse of z_curr
        var_z, cov_z, min_std_z, _ = self.vicreg_loss._compute_var_cov(z_curr.unsqueeze(0))
        wm_loss = (
            loss_dict["loss"]
            + self.vicreg_loss.var_coeff * var_z
            + self.vicreg_loss.cov_coeff * cov_z
        )

        # 3. Optimizer backward & step
        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.encoder.parameters()) + list(self.agent.predictor.parameters()),
            max_norm=1.0,
        )
        self.wm_optimizer.step()

        # 4. Polyak EMA update on target encoder
        self.agent.update_target_networks(tau=self.tau)

        return {
            "wm_loss": float(wm_loss.item()),
            "invariance": float(loss_dict["invariance"].item()),
            "variance": float(loss_dict["variance"].item()),
            "covariance": float(loss_dict["covariance"].item()),
            "min_std_z": float(min_std_z.item()),
        }

    def update_actor_critic(self) -> Dict[str, float]:
        """
        Optimizes Twin Critic Q_omega(z, a) and Actor policy prior pi_psi(a | z)
        using latent representations and blended extrinsic + intrinsic rewards.
        """
        self.agent.train()
        batch = self.buffer.sample_nstep(
            batch_size=self.batch_size,
            n=self.nstep,
            gamma=self.gamma,
            tier2_ratio=self.tier2_ratio,
            device=self.device,
        )

        with torch.no_grad():
            z_curr = self.agent.encoder(batch["obs"])
            z_boot = self.agent.target_encoder(batch["boot_obs"])

            # Discrete Double-Q Target with n-step discounted return & bootstrapped future value
            q1b, q2b = self.agent.critic(z_boot)
            best_boot_actions = torch.argmax(torch.min(q1b, q2b), dim=-1)
            boot_q = self.agent.critic.target_q_min(z_boot, best_boot_actions)
            r_total = batch["nstep_return"] + self.critic_intrinsic_weight * batch["int_rewards"]
            target_q = r_total + batch["boot_gamma"] * batch["boot_valid"].float() * boot_q

        # 1. Update Twin Critic
        critic_loss, c_metrics = self.agent.critic.compute_loss(
            z_curr.detach(),
            batch["actions"],
            target_q,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update Target Critic EMA
        self.agent.critic.update_target(tau=self.tau)

        # C4.1: Compute per-action ensemble disagreement under no_grad (exploration term)
        with torch.no_grad():
            B = z_curr.shape[0]
            z_rep = z_curr.unsqueeze(1).expand(B, self.agent.num_actions, -1).reshape(B * self.agent.num_actions, -1)
            a_rep = torch.arange(self.agent.num_actions, device=z_curr.device).repeat(B)
            _, all_preds = self.agent.predictor(z_rep, a_rep)  # (K, B*17, D)
            disc = self.agent.predictor.compute_epistemic_disagreement(all_preds).view(B, -1)  # (B, 17)
            self.per_action_disc_stats.update(disc)
            disc_norm = disc / (self.per_action_disc_stats.std + self.per_action_disc_stats.epsilon)  # ÷std per action column
            disc_norm = torch.clamp(disc_norm, min=0.0, max=5.0)

        # 2. Update Actor Policy Prior via Exact Discrete Expectation
        # Since action space is discrete (17), compute exact expectation under policy
        # allowing direct, rich gradient flow from TwinCritic Q-values into Actor weights
        logits = self.agent.actor(z_curr.detach())
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        q1_all, q2_all = self.agent.critic(z_curr.detach())
        q_all = torch.min(q1_all, q2_all).detach()

        expected_q = (probs * q_all).sum(dim=-1)
        expected_int = (probs * disc_norm).sum(dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        actor_loss = -(expected_q + self.actor_intrinsic_weight * expected_int + 0.05 * entropy).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        q_spread = (q_all.max(dim=-1).values - q_all.min(dim=-1).values).mean()
        q_mean_ext = expected_q.mean()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.mean().item()),
            "q_mean": float(expected_q.mean().item()),
            "q_spread": float(q_spread.item()),
            "q_mean_ext": float(q_mean_ext.item()),
            "expected_int_mean": float(expected_int.mean().item()),
            "eps_mean": float(np.mean(self.recent_eps)) if len(self.recent_eps) > 0 else 0.0,
        }

    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        stochastic: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs evaluation episodes recording both greedy (deterministic) and stochastic policy variants.
        Collects survival time, episodic returns, and sanitized Crafter achievements.

        Args:
            num_episodes: Number of episodes per variant (defaults to self.eval_episodes).
            stochastic: If True, primary top-level return metrics point to stochastic results.

        Returns:
            Dictionary with mean return, mean length, unlocked achievements, and dual variant keys.
        """
        self.agent.eval()
        n_eps = num_episodes if num_episodes is not None else self.eval_episodes

        # 1. Greedy evaluation episodes (eval_mode=True)
        greedy_returns: List[float] = []
        greedy_lengths: List[int] = []
        greedy_achievements: Set[str] = set()

        for ep_i in range(n_eps):
            ep_seed = self.seed + 1000 + self.current_step + ep_i
            obs = self.eval_env.reset(seed=ep_seed)
            ep_ret = 0.0
            ep_len = 0
            done = False

            while not done:
                action = self.agent.act(obs, eval_mode=True, planning=False)
                obs, reward, term, trunc, info = self.eval_env.step(action)
                ep_ret += float(reward)
                ep_len += 1
                done = term or trunc

                eval_info = info.get("eval_info", {})
                unlocked = eval_info.get("unlocked_achievements", [])
                greedy_achievements.update(unlocked)

            greedy_returns.append(ep_ret)
            greedy_lengths.append(ep_len)

        # 2. Stochastic evaluation episodes (eval_mode=False)
        stoch_returns: List[float] = []
        stoch_lengths: List[int] = []
        stoch_achievements: Set[str] = set()

        for ep_i in range(n_eps):
            ep_seed = self.seed + 2000 + self.current_step + ep_i
            obs = self.eval_env.reset(seed=ep_seed)
            ep_ret = 0.0
            ep_len = 0
            done = False

            while not done:
                action = self.agent.act(obs, eval_mode=False, planning=False)
                obs, reward, term, trunc, info = self.eval_env.step(action)
                ep_ret += float(reward)
                ep_len += 1
                done = term or trunc

                eval_info = info.get("eval_info", {})
                unlocked = eval_info.get("unlocked_achievements", [])
                stoch_achievements.update(unlocked)

            stoch_returns.append(ep_ret)
            stoch_lengths.append(ep_len)

        g_mean_ret = float(np.mean(greedy_returns)) if greedy_returns else 0.0
        g_mean_len = float(np.mean(greedy_lengths)) if greedy_lengths else 0.0
        g_ach_list = sorted(list(greedy_achievements))

        s_mean_ret = float(np.mean(stoch_returns)) if stoch_returns else 0.0
        s_mean_len = float(np.mean(stoch_lengths)) if stoch_lengths else 0.0
        s_ach_list = sorted(list(stoch_achievements))

        primary_ret = s_mean_ret if stochastic else g_mean_ret
        primary_len = s_mean_len if stochastic else g_mean_len
        primary_achs = s_ach_list if stochastic else g_ach_list

        eval_summary = {
            "step": self.current_step,
            "mean_return": primary_ret,
            "mean_length": primary_len,
            "achievement_count": len(primary_achs),
            "achievements": primary_achs,
            "greedy_mean_return": g_mean_ret,
            "greedy_mean_length": g_mean_len,
            "greedy_achievement_count": len(g_ach_list),
            "greedy_achievements": g_ach_list,
            "stochastic_mean_return": s_mean_ret,
            "stochastic_mean_length": s_mean_len,
            "stochastic_achievement_count": len(s_ach_list),
            "stochastic_achievements": s_ach_list,
        }

        # C7: Audit representation health / non-collapse
        if self.buffer.size >= min(self.batch_size, 32):
            try:
                sample_sz = min(self.buffer.size, self.batch_size)
                sample_batch = self.buffer.sample(batch_size=sample_sz, device=self.device)
                with torch.no_grad():
                    z_sample = self.agent.encoder(sample_batch["obs"])
                    nc_report = verify_non_collapse(z_sample, strict=False)
                    eval_summary["min_variance"] = float(nc_report.get("min_variance", 0.0))
                    eval_summary["num_collapsed_dims"] = int(nc_report.get("num_collapsed_dims", 0))
                    eval_summary["effective_rank"] = float(nc_report.get("effective_rank", 0.0))
            except Exception:
                pass

        self.achievement_history[self.current_step] = eval_summary

        return eval_summary

    def save_checkpoint(self, is_best: bool = False, force_buffer: bool = False) -> str:
        """
        Atomically saves pipeline components and metadata to checkpoint_dir.
        RAM Rule R2: Saves replay buffer arrays only every buffer_save_every checkpoints
        (or when force_buffer=True) to eliminate transient 1.7GB memory allocation spikes.
        """
        self.checkpoint_count += 1
        save_buffer = force_buffer or (self.checkpoint_count % self.buffer_save_every == 0)

        filename = f"checkpoint_step_{self.current_step}.pt"
        metadata = {
            "step": self.current_step,
            "vector_step": self.vector_step,
            "checkpoint_count": self.checkpoint_count,
            "episode": self.current_episode,
            "num_envs": self.num_envs,
            "best_eval_return": self.best_eval_return,
            "all_unlocked_achievements": sorted(list(self.all_unlocked_achievements)),
            "achievement_history": self.achievement_history,
            "success_counts": self.success_counts,
            "has_buffer": save_buffer,
            "config": self.cfg,
            "timestamp": time.time(),
        }

        components = {
            "agent": self.agent,
            "wm_optimizer": self.wm_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "actor_optimizer": self.actor_optimizer,
        }
        if save_buffer:
            components["buffer"] = self.buffer

        saved_path = self.checkpoint_manager.save(
            filepath=filename,
            components=components,
            metadata=metadata,
            is_best=is_best,
        )
        # Atomically update active verified checkpoint 'state.pt' via zero-copy file clone (Rule 2 & 3)
        # Avoids duplicating massive replay buffer payload in Python RAM to eliminate Colab OOM crashes
        state_path = Path(saved_path).parent / "state.pt"
        state_tmp = Path(saved_path).parent / f"state.pt.tmp.{os.getpid()}"
        try:
            shutil.copyfile(saved_path, str(state_tmp))
            os.replace(str(state_tmp), str(state_path))
        except Exception as e:
            if state_tmp.exists():
                try:
                    state_tmp.unlink()
                except OSError:
                    pass

        # Clean memory immediately
        del components, metadata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return saved_path

    def train(self) -> Dict[str, Any]:
        """
        Main training loop executing environment interaction, experience gathering,
        interleaved model and policy updates, periodic evaluation, and checkpointing.

        Returns:
            Final training report with metrics, curves, and achievement history.
        """
        print("=" * 80)
        print(" T A B U L A   R A S A   J E P A   T R A I N I N G   P I P E L I N E")
        print("=" * 80)
        print(f"Device:           {self.device}")
        print(f"Total Steps:      {self.total_steps}")
        print(f"Num Envs:         {self.num_envs} (Parallel Batching)")
        print(f"Batch Size:       {self.batch_size}")
        print(f"Checkpoint Dir:   {self.checkpoint_dir.resolve()}")
        print(f"Eval Interval:    {self.eval_interval} steps ({self.eval_episodes} episodes)")
        print(f"Seed:             {self.seed}")
        print("=" * 80)

        start_time = time.perf_counter()
        last_log_time = start_time
        next_eval_step = (
            ((self.current_step // self.eval_interval) + 1) * self.eval_interval
            if self.eval_interval > 0
            else float("inf")
        )
        next_save_step = (
            ((self.current_step // self.save_interval) + 1) * self.save_interval
            if self.save_interval > 0
            else float("inf")
        )

        try:
            while self.current_step < self.total_steps:
                # 1. Environment step and experience accumulation (Batched across N parallel envs)
                added = self.collect_batch_step()
                if added == 0:
                    break

                # 2. Interleaved Model & Policy Updates (counts vector steps: C5 & PARALLEL_ENV_PLAN.md §3.2)
                if (
                    self.buffer.size >= self.batch_size
                    and self.vector_step >= self.warmup_steps
                    and (self.vector_step % self.update_every == 0)
                ):
                    wm_metrics = self.update_world_model()
                    ac_metrics = self.update_actor_critic()
                    combined_metrics = {**wm_metrics, **ac_metrics}
                    self.loss_history.append(combined_metrics)

                # 3. Periodic Progress Logging & RAM telemetry (Rule R3)
                if self.vector_step % self.log_interval == 0:
                    now = time.perf_counter()
                    elapsed = max(1e-4, now - last_log_time)
                    last_log_time = now
                    vec_sps = self.log_interval / elapsed
                    env_sps = (self.log_interval * self.num_envs) / elapsed
                    ram_pct = get_ram_percent()

                    if ram_pct > self.ram_abort_percent:
                        emergency_ckpt = self.save_checkpoint(force_buffer=True)
                        raise RuntimeError(
                            f"EMERGENCY ABORT (Rule R3): System RAM at {ram_pct:.1f}% exceeds safe limit of {self.ram_abort_percent}%. "
                            f"Emergency checkpoint saved to {emergency_ckpt}."
                        )

                    loss_info = ""
                    if self.loss_history:
                        latest = self.loss_history[-1]
                        loss_info = (
                            f" | WM Loss: {latest.get('wm_loss', 0.0):.4f} "
                            f"(Inv: {latest.get('invariance', 0.0):.4f}, Var: {latest.get('variance', 0.0):.4f}) "
                            f"| Critic: {latest.get('critic_loss', 0.0):.4f}"
                        )

                    print(
                        f"Step {self.current_step:7d}/{self.total_steps} (Vec: {self.vector_step}) | "
                        f"Buf: {self.buffer.size:5d} (T1:{self.buffer.t1_size}, T2:{self.buffer.t2_size}) | "
                        f"MS: {self.total_milestones:4d} | "
                        f"Ach: {len(self.all_unlocked_achievements):2d} | "
                        f"SPS: {env_sps:5.1f} (Vec: {vec_sps:4.1f}) | "
                        f"RAM: {ram_pct:4.1f}%{loss_info}"
                    )

                # 4. Periodic Episodic Evaluation
                if self.current_step >= next_eval_step or self.current_step >= self.total_steps:
                    next_eval_step = ((self.current_step // self.eval_interval) + 1) * self.eval_interval
                    if self.save_interval > 0:
                        next_save_step = ((self.current_step // self.save_interval) + 1) * self.save_interval
                    eval_metrics = self.evaluate()
                    print("-" * 80)
                    print(
                        f" [EVAL @ Step {self.current_step}] "
                        f"Mean Return: {eval_metrics['mean_return']:.2f} | "
                        f"Mean Length: {eval_metrics['mean_length']:.1f} | "
                        f"Achievements ({eval_metrics['achievement_count']}): {', '.join(eval_metrics['achievements']) or 'None'}"
                    )
                    print("-" * 80)

                    is_best = eval_metrics["mean_return"] > self.best_eval_return
                    if is_best:
                        self.best_eval_return = eval_metrics["mean_return"]

                    saved = self.save_checkpoint(is_best=is_best)
                    print(f" [+] Checkpoint saved: {saved}")

                # 5. Periodic Checkpoint Saving (if not overlapping eval)
                elif self.current_step >= next_save_step:
                    next_save_step = ((self.current_step // self.save_interval) + 1) * self.save_interval
                    saved = self.save_checkpoint(is_best=False)
                    print(f" [+] Checkpoint saved: {saved}")

        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user. Saving emergency checkpoint...")
            saved = self.save_checkpoint(is_best=False, force_buffer=True)
            print(f" [+] Emergency checkpoint saved: {saved}")

        total_time = time.perf_counter() - start_time
        print("=" * 80)
        print(" T R A I N I N G   C O M P L E T E")
        print("=" * 80)
        print(f"Total Steps:        {self.current_step}")
        print(f"Total Episodes:     {self.current_episode}")
        print(f"Total Time:         {total_time:.2f} s")
        print(f"Achievements Count: {len(self.all_unlocked_achievements)}")
        print(f"Unlocked List:      {sorted(list(self.all_unlocked_achievements))}")
        print(f"Success Counts:     {self.success_counts}")
        print("=" * 80)

        # Final checkpoint save (with buffer forced)
        final_ckpt = self.save_checkpoint(is_best=False, force_buffer=True)

        # Close environments
        self.envs.close()
        self.eval_env.close()

        return {
            "total_steps": self.current_step,
            "total_episodes": self.current_episode,
            "total_time_seconds": total_time,
            "best_eval_return": self.best_eval_return,
            "achievements_unlocked": sorted(list(self.all_unlocked_achievements)),
            "achievement_history": self.achievement_history,
            "final_checkpoint": final_ckpt,
            "success_counts": self.success_counts,
        }


def train(
    device: Optional[str] = None,
    total_steps: int = 10_000,
    batch_size: int = 128,
    checkpoint_dir: str = "./checkpoints",
    eval_interval: int = 1_000,
    seed: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run JEPA training programmatically.
    """
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "device": dev,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "checkpoint_dir": checkpoint_dir,
        "eval_interval": eval_interval,
        "seed": seed,
        **kwargs,
    }
    pipeline = TrainingPipeline(config)
    return pipeline.train()


if __name__ == "__main__":
    cli_args = parse_args()
    pipeline = TrainingPipeline(cli_args)
    pipeline.train()
