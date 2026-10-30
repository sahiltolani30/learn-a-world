"""
src/train.py

Production-Grade, Modular Training Pipeline for Crafter Tabula Rasa JEPA World Model.
Coordinates environment stepping, experience accumulation, world model updates,
actor-critic policy updates, episodic evaluation, and preemption-resilient checkpointing.

CLI Arguments:
  --device: Computing device ('cpu', 'cuda', 'mps', or auto-detect).
  --total_steps: Total environment interaction steps (default: 10,000).
  --batch_size: Replay buffer training batch size (default: 64).
  --checkpoint_dir: Path to directory for persistent checkpoints (default: './checkpoints').
  --eval_interval: Frequency of evaluation episodes in steps (default: 1,000).
  --seed: Random seed for environment, buffers, and network initialization (default: 42).
  --eval_episodes: Number of episodes per evaluation cycle (default: 3).
  --lr: Learning rate for visual encoder and world model predictor (default: 1e-4).
  --critic_lr: Learning rate for twin critic heads (default: 3e-4).
  --warmup_steps: Initial random interaction steps before gradient updates (default: 100).
  --save_interval: Step interval for saving persistent checkpoints (default: 1,000).
  --no_resume: Flag to disable auto-resumption from latest checkpoint in checkpoint_dir.

Guarantees:
- R2, R3: Tabula rasa learning from raw pixels without cheats or privileged states.
- R2: Two-Tier persistent memory (70% FIFO, 30% Milestone Archive) preventing catastrophic forgetting.
- R3: Zero-cheat compliance: info dict is completely scrubbed of internal game state.
- R5: Atomic, preemption-resilient checkpoint saving via CheckpointManager.
- Evaluator telemetry: Tracks achievement unlock progression curves without leaking state to the agent.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Set
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
from src.agent.agent import JEPAAgent
from src.curiosity.ensemble import EpistemicCuriosityEngine
from src.memory.buffer import TwoTierReplayBuffer
from src.memory.storage import CheckpointManager
from src.models.losses import VICRegLoss, verify_non_collapse


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
        default=64,
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
        self.batch_size: int = int(self.cfg.get("batch_size", 64))
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
        self.gamma: float = float(self.cfg.get("gamma", 0.99))
        self.tau: float = float(self.cfg.get("tau", 0.99))
        self.log_interval: int = int(self.cfg.get("log_interval", 100))

        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        # 1. Environment Initialization
        self.env = make_crafter_env(seed=self.seed)
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
                self.current_episode = metadata.get("episode", 0)
                self.best_eval_return = metadata.get("best_eval_return", -float("inf"))
                self.achievement_history = metadata.get("achievement_history", {})
                self.all_unlocked_achievements = set(metadata.get("all_unlocked_achievements", []))
                print(
                    f"[+] Resumed successfully at step {self.current_step}, "
                    f"episode {self.current_episode}, best return {self.best_eval_return:.2f}"
                )
                return True
            except Exception as e:
                print(f"[!] Warning: Failed to restore checkpoint at {latest_ckpt}: {e}")
                return False
        return False

    def collect_step(
        self,
        obs: torch.Tensor,
        episode_achievements: Set[str],
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Interacts with Crafter for 1 step, computes intrinsic curiosity,
        audits for zero-cheat compliance, and stores transition in TwoTierReplayBuffer.

        Args:
            obs: Normalized observation tensor (3, 64, 64).
            episode_achievements: Set tracking achievements unlocked in current episode.

        Returns:
            next_obs, reward, done, step_metrics
        """
        # Select action
        if self.current_step < self.warmup_steps:
            action = int(self.env.action_space.sample())
        else:
            action = self.agent.act(obs, eval_mode=False, planning=False)

        next_obs, ext_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Assert Zero-Cheat: Info contains no internal game state
        for forbidden in FORBIDDEN_CHEAT_KEYS:
            if forbidden in info:
                raise AssertionError(
                    f"CRITICAL ZERO-CHEAT LEAK: Forbidden key '{forbidden}' found in env info!"
                )

        # Compute epistemic curiosity reward for exploration
        with torch.no_grad():
            z_t = self.agent.encode(obs)
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
            a_t = torch.tensor([action], device=self.device, dtype=torch.long)
            _, all_preds = self.agent.predictor(z_t, a_t)
            r_int_tensor = self.agent.curiosity_engine.compute_intrinsic_reward(all_preds, update_stats=True)
            r_int = float(r_int_tensor.item() if r_int_tensor.numel() == 1 else r_int_tensor[0].item())

        # Inspect sanitized eval_info for new milestone discoveries
        is_milestone = False
        eval_info = info.get("eval_info", {})
        unlocked = eval_info.get("unlocked_achievements", [])
        for ach in unlocked:
            if ach not in episode_achievements:
                episode_achievements.add(ach)
                is_milestone = True
                if ach not in self.all_unlocked_achievements:
                    self.all_unlocked_achievements.add(ach)
                    print(f" [★ DISCOVERY] Step {self.current_step}: Unlocked achievement '{ach}'!")

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
        )

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
        batch = self.buffer.sample(
            batch_size=self.batch_size,
            tier2_ratio=self.tier2_ratio,
            device=self.device,
        )

        with torch.no_grad():
            z_curr = self.agent.encoder(batch["obs"])
            z_next = self.agent.target_encoder(batch["next_obs"])

            # Compute blended reward: r_total = r_ext + beta * r_int
            r_total = self.agent.curiosity_engine.blend_rewards(
                batch["rewards"],
                batch["int_rewards"],
            )

            # Discrete Double-Q Target: greedy optimal next action from online critic
            # evaluated by clipped double target critic to prevent overestimation
            q1_next, q2_next = self.agent.critic(z_next)
            best_next_actions = torch.argmax(torch.min(q1_next, q2_next), dim=-1)
            target_q_min = self.agent.critic.target_q_min(z_next, best_next_actions)
            not_done = (~batch["dones"]).float()
            target_q = r_total + self.gamma * not_done * target_q_min

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

        # 2. Update Actor Policy Prior via Exact Discrete Expectation
        # Since action space is discrete (17), compute exact expectation under policy
        # allowing direct, rich gradient flow from TwinCritic Q-values into Actor weights
        logits = self.agent.actor(z_curr.detach())
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        q1_all, q2_all = self.agent.critic(z_curr.detach())
        q_all = torch.min(q1_all, q2_all).detach()

        expected_q = (probs * q_all).sum(dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        actor_loss = -(expected_q + 0.05 * entropy).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.mean().item()),
            "q_mean": float(expected_q.mean().item()),
        }

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs evaluation episodes with deterministic action selection (eval_mode=True).
        Collects survival time, episodic returns, and sanitized Crafter achievements.

        Args:
            num_episodes: Number of episodes (defaults to self.eval_episodes).

        Returns:
            Dictionary with mean return, mean length, and unlocked achievements.
        """
        self.agent.eval()
        n_eps = num_episodes if num_episodes is not None else self.eval_episodes
        returns: List[float] = []
        lengths: List[int] = []
        eval_achievements: Set[str] = set()

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

                # Collect unlocked achievements from sanitized eval_info
                eval_info = info.get("eval_info", {})
                unlocked = eval_info.get("unlocked_achievements", [])
                eval_achievements.update(unlocked)

            returns.append(ep_ret)
            lengths.append(ep_len)

        mean_return = float(np.mean(returns)) if returns else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        ach_list = sorted(list(eval_achievements))

        # Record in persistent achievement curve history
        eval_summary = {
            "step": self.current_step,
            "mean_return": mean_return,
            "mean_length": mean_length,
            "achievement_count": len(ach_list),
            "achievements": ach_list,
        }
        self.achievement_history[self.current_step] = eval_summary

        return eval_summary

    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Atomically saves pipeline components and metadata to checkpoint_dir.
        """
        filename = f"checkpoint_step_{self.current_step}.pt"
        metadata = {
            "step": self.current_step,
            "episode": self.current_episode,
            "best_eval_return": self.best_eval_return,
            "all_unlocked_achievements": sorted(list(self.all_unlocked_achievements)),
            "achievement_history": self.achievement_history,
            "config": self.cfg,
            "timestamp": time.time(),
        }

        components = {
            "agent": self.agent,
            "buffer": self.buffer,
            "wm_optimizer": self.wm_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "actor_optimizer": self.actor_optimizer,
        }

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
        print(f"Batch Size:       {self.batch_size}")
        print(f"Checkpoint Dir:   {self.checkpoint_dir.resolve()}")
        print(f"Eval Interval:    {self.eval_interval} steps ({self.eval_episodes} episodes)")
        print(f"Seed:             {self.seed}")
        print("=" * 80)

        obs = self.env.reset()
        ep_achievements: Set[str] = set()
        ep_reward = 0.0
        ep_length = 0

        start_time = time.perf_counter()
        last_log_time = start_time

        try:
            while self.current_step < self.total_steps:
                self.current_step += 1

                # 1. Environment step and experience accumulation
                obs, ext_r, done, step_metrics = self.collect_step(obs, ep_achievements)
                ep_reward += ext_r
                ep_length += 1

                if done:
                    self.current_episode += 1
                    obs = self.env.reset()
                    ep_achievements = set()
                    ep_reward = 0.0
                    ep_length = 0

                # 2. Interleaved Model & Policy Updates (once buffer has enough samples)
                if self.buffer.size >= self.batch_size and self.current_step >= self.warmup_steps:
                    wm_metrics = self.update_world_model()
                    ac_metrics = self.update_actor_critic()
                    combined_metrics = {**wm_metrics, **ac_metrics}
                    self.loss_history.append(combined_metrics)
                    if len(self.loss_history) > 500:
                        self.loss_history = self.loss_history[-500:]

                # 3. Periodic Progress Logging
                if self.current_step % self.log_interval == 0:
                    now = time.perf_counter()
                    sps = self.log_interval / max(1e-4, now - last_log_time)
                    last_log_time = now

                    loss_info = ""
                    if self.loss_history:
                        latest = self.loss_history[-1]
                        loss_info = (
                            f" | WM Loss: {latest.get('wm_loss', 0.0):.4f} "
                            f"(Inv: {latest.get('invariance', 0.0):.4f}, Var: {latest.get('variance', 0.0):.4f}) "
                            f"| Critic: {latest.get('critic_loss', 0.0):.4f}"
                        )

                    print(
                        f"Step {self.current_step:6d}/{self.total_steps} | "
                        f"Buf: {self.buffer.size:5d} (T1:{self.buffer.t1_size}, T2:{self.buffer.t2_size}) | "
                        f"Ach: {len(self.all_unlocked_achievements):2d} | "
                        f"SPS: {sps:5.1f}{loss_info}"
                    )

                # 4. Periodic Episodic Evaluation
                if self.current_step % self.eval_interval == 0 or self.current_step == self.total_steps:
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

                    # Checkpoint saving
                    saved = self.save_checkpoint(is_best=is_best)
                    print(f" [+] Checkpoint saved: {saved}")

                # 5. Periodic Checkpoint Saving (if not overlapping eval)
                elif self.current_step % self.save_interval == 0:
                    saved = self.save_checkpoint(is_best=False)
                    print(f" [+] Checkpoint saved: {saved}")

        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user. Saving emergency checkpoint...")
            saved = self.save_checkpoint(is_best=False)
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
        print("=" * 80)

        # Final checkpoint save
        final_ckpt = self.save_checkpoint(is_best=False)

        # Close environments
        self.env.close()
        self.eval_env.close()

        return {
            "total_steps": self.current_step,
            "total_episodes": self.current_episode,
            "total_time_seconds": total_time,
            "best_eval_return": self.best_eval_return,
            "achievements_unlocked": sorted(list(self.all_unlocked_achievements)),
            "achievement_history": self.achievement_history,
            "final_checkpoint": final_ckpt,
        }


def train(
    device: Optional[str] = None,
    total_steps: int = 10_000,
    batch_size: int = 64,
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
