"""
genesis_jepa.train — Self-Play & Lifelong Tabula Rasa Training Pipeline
======================================================================
Governing Laws: AGENTS.md & GEMINI.md (Zero Cheating, Zero Hardcoding)

Executes the complete self-play learning loop across games N -> N+1:
1. Environment Interaction: Agent acts via strict act(obs, reward_prev, step).
2. Value-Stratified Replay: 50/50 quota equipartition between milestone & routine steps.
3. World Model Optimization: Normalized L1 loss + soft-capped anti-persistence loss.
4. Distributional Critic: 65-bin symlog two-hot simplex projection (|grad| <= 1.0).
5. Advantage-Weighted Actor: Policy optimization weighted by value advantage.
6. Causal Induction & Deficit Budgeting: Consolidates discovered costs into long-term memory.
7. Record Tracking: Agent tracks personal best milestones and aims to surpass them in Game N+1.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import crafter
except ImportError:
    crafter = None

from genesis_jepa.agent import GenesisVisualAgent
from genesis_jepa.config import GenesisConfig
from genesis_jepa.replay import ValueStratifiedReplayBuffer


class GenesisTrainer:
    """
    Self-contained trainer coordinating self-play, replay storage,
    multi-gate optimization, and lifelong memory consolidation across games.
    """

    def __init__(
        self,
        agent: Optional[GenesisVisualAgent] = None,
        config: Optional[GenesisConfig] = None,
        replay_buffer: Optional[ValueStratifiedReplayBuffer] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or GenesisConfig()
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 1. Visual Agent (retina, world model, critic, actor, budgeter)
        self.agent = (agent or GenesisVisualAgent(self.config)).to(self.device)

        # 2. Value-Stratified Replay Buffer (50/50 quota)
        self.replay_buffer = replay_buffer or ValueStratifiedReplayBuffer(
            capacity=100_000,
            milestone_quota=self.config.stratified_milestone_quota,
        )

        # 3. Optimizers
        self.wm_optimizer = torch.optim.AdamW(
            self.agent.world_model.parameters(),
            lr=self.config.lr_world_model,
            weight_decay=self.config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.agent.critic.parameters(),
            lr=self.config.lr_critic,
            weight_decay=self.config.weight_decay,
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.agent.actor.parameters(),
            lr=self.config.lr_actor,
            weight_decay=self.config.weight_decay,
        )
        self.danger_optimizer = torch.optim.AdamW(
            self.agent.danger_head.parameters(),
            lr=1e-4,
            weight_decay=self.config.weight_decay,
        )

        # 4. Lifelong Memory & Record Tracker
        self.best_reward: float = -float("inf")
        self.best_length: int = 0
        self.total_milestones_unlocked: Dict[str, int] = {}
        self.training_steps: int = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executes a single multi-gate gradient step across World Model, Critic, and Actor.
        """
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # -------------------------------------------------------------
        # Gate 1: World Model Step (Normalized L1 + Anti-Persistence)
        # -------------------------------------------------------------
        self.agent.world_model.train()
        loss_wm, wm_metrics = self.agent.world_model.compute_step_loss(obs, action, next_obs)

        self.wm_optimizer.zero_grad()
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.world_model.parameters(), 1.0)
        self.wm_optimizer.step()

        # Update EMA target encoder
        self.agent.world_model.update_target_ema(decay=self.config.target_ema_decay)

        # -------------------------------------------------------------
        # Gate 4: 65-Bin Symlog Distributional Critic Step
        # -------------------------------------------------------------
        self.agent.critic.train()
        with torch.no_grad():
            _, next_slots = self.agent.world_model.target_encoder(next_obs)
            next_pooled = torch.mean(next_slots, dim=1)
            v_next = self.agent.critic.predict_scalar(next_pooled)
            target_return = reward + (1.0 - done) * self.config.gamma * v_next

            # Extract current state representation
            _, curr_slots = self.agent.world_model.encode(obs)
            curr_pooled = torch.mean(curr_slots, dim=1)

        loss_critic, critic_metrics = self.agent.critic.compute_loss(
            curr_pooled.detach(), target_return
        )

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # -------------------------------------------------------------
        # Policy Actor Step (Advantage-Weighted Regression)
        # -------------------------------------------------------------
        self.agent.actor.train()
        logits = self.agent.actor(curr_pooled.detach())
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            v_curr = self.agent.critic.predict_scalar(curr_pooled.detach())
            advantage = target_return - v_curr
            weights = torch.clamp(
                torch.exp(advantage / self.config.awr_temperature),
                np.exp(self.config.awr_clip_min),
                np.exp(self.config.awr_clip_max),
            )

        loss_actor = -torch.mean(weights * selected_log_probs)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # -------------------------------------------------------------
        # Survival Danger Head Step (Hazard on damage: reward < 0.0)
        # -------------------------------------------------------------
        self.agent.danger_head.train()
        danger_labels = (reward < 0.0).float().unsqueeze(-1)
        danger_preds = self.agent.danger_head(curr_pooled.detach())
        loss_danger = F.binary_cross_entropy(danger_preds, danger_labels)

        self.danger_optimizer.zero_grad()
        loss_danger.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.danger_head.parameters(), 1.0)
        self.danger_optimizer.step()

        self.training_steps += 1

        metrics = {
            "loss_wm": float(loss_wm.item()),
            "loss_critic": float(loss_critic.item()),
            "loss_actor": float(loss_actor.item()),
            "critic_std": critic_metrics.get("critic_std", 0.0),
            "max_logit_grad": critic_metrics.get("max_logit_grad", 0.0),
        }
        return metrics

    def self_play_episode(
        self,
        env: Any,
        episode_idx: int,
        max_steps: int = 500,
        train_every: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs one complete self-play episode where agent acts, collects experience,
        trains online, and consolidates knowledge into long-term memory.
        """
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0
        reward_prev = 0.0
        self.agent.reset()

        step_losses: List[Dict[str, float]] = []

        while not done and step_count < max_steps:
            # 1. Unprivileged visual perception & action
            action = self.agent.act(obs, reward_prev=reward_prev, step=step_count)

            # 2. Physical environment transition
            next_obs, reward, done, info = env.step(action)
            total_reward += float(reward)

            # 3. Store in Value-Stratified Replay Buffer
            self.replay_buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                done=bool(done),
            )

            # 4. Online learning step (train world model, critic, actor)
            if len(self.replay_buffer) >= self.config.batch_size and (step_count % train_every == 0):
                batch = self.replay_buffer.sample_batch(self.config.batch_size)
                loss_info = self.train_step(batch)
                step_losses.append(loss_info)

            obs = next_obs
            reward_prev = float(reward)
            step_count += 1

        # Check personal record
        beat_record = total_reward > self.best_reward or (
            total_reward == self.best_reward and step_count > self.best_length
        )
        if beat_record:
            self.best_reward = max(self.best_reward, total_reward)
            self.best_length = max(self.best_length, step_count)

        ep_achievements = info.get("achievements", {})
        for ach, count in ep_achievements.items():
            if count > 0:
                self.total_milestones_unlocked[ach] = (
                    self.total_milestones_unlocked.get(ach, 0) + 1
                )

        mean_wm_loss = (
            float(np.mean([l["loss_wm"] for l in step_losses])) if step_losses else 0.0
        )
        mean_critic_loss = (
            float(np.mean([l["loss_critic"] for l in step_losses])) if step_losses else 0.0
        )

        ep_result = {
            "episode": episode_idx,
            "steps": step_count,
            "total_reward": total_reward,
            "survived_night": step_count >= 220,
            "beat_personal_record": beat_record,
            "mean_wm_loss": mean_wm_loss,
            "mean_critic_loss": mean_critic_loss,
            "achievements": [k for k, v in ep_achievements.items() if v > 0],
        }

        if verbose:
            record_str = " (★ NEW RECORD!)" if beat_record else ""
            print(
                f"[Game {episode_idx}] Steps: {step_count:3d} | "
                f"Reward: {total_reward:4.1f} | "
                f"WM Loss: {mean_wm_loss:5.3f} | "
                f"Critic Loss: {mean_critic_loss:5.3f} | "
                f"Night: {'YES' if step_count >= 220 else ' NO'}{record_str} | "
                f"Milestones: {ep_result['achievements']}"
            )

        return ep_result

    def save_checkpoint(self, path: str) -> None:
        """Saves agent parameters and memory state."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "agent_state": self.agent.state_dict(),
            "best_reward": self.best_reward,
            "best_length": self.best_length,
            "training_steps": self.training_steps,
            "milestones": self.total_milestones_unlocked,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Loads agent parameters and memory state."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.agent.load_state_dict(checkpoint["agent_state"])
            self.best_reward = checkpoint.get("best_reward", 0.0)
            self.best_length = checkpoint.get("best_length", 0)
            self.training_steps = checkpoint.get("training_steps", 0)
            self.total_milestones_unlocked = checkpoint.get("milestones", {})


def run_self_play_training(
    num_episodes: int = 10,
    max_steps_per_episode: int = 300,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Launches autonomous self-play and lifelong learning loop across games.
    """
    if crafter is None:
        raise ImportError("Crafter is not installed. Install via `pip install crafter`.")

    config = GenesisConfig()
    trainer = GenesisTrainer(config=config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming agent from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    env = crafter.Env(seed=seed)
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 75)
    print(f"🚀 LAUNCHING GENESIS JEPA: LIFELONG SELF-PLAY & LEARNING ({num_episodes} GAMES)")
    print("   Governing Laws: Strict Zero Cheating (act(obs, reward, step) unprivileged)")
    print("=" * 75)

    for ep in range(1, num_episodes + 1):
        ep_result = trainer.self_play_episode(
            env=env,
            episode_idx=ep,
            max_steps=max_steps_per_episode,
            train_every=4,
            verbose=True,
        )
        history.append(ep_result)

    if hasattr(env, "close"):
        env.close()

    if checkpoint_path:
        trainer.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    print("\n" + "=" * 75)
    print("🎯 SELF-PLAY SESSION SUMMARY")
    print(f"   Total Games Played: {num_episodes}")
    print(f"   All-Time Best Reward: {trainer.best_reward:.1f}")
    print(f"   All-Time Longest Survival: {trainer.best_length} steps")
    print(f"   Total Training Gradient Steps: {trainer.training_steps}")
    print(f"   Cumulative Milestones Discovered: {trainer.total_milestones_unlocked}")
    print("=" * 75)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis JEPA Self-Play & Learning")
    parser.add_argument("--episodes", type=int, default=10, help="Number of games to self-play")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per game")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--checkpoint", type=str, default="genesis_checkpoints/agent_latest.pt")
    parser.add_argument("--history-json", type=str, default=None, help="Output JSON path for training history")
    args, unknown = parser.parse_known_args()
    if unknown:
        clean_unknown = [u for u in unknown if u.strip() not in ("n", "\\n", "")]
        if clean_unknown:
            print(f"[Warning] Ignoring unrecognized CLI arguments: {clean_unknown}")

    history = run_self_play_training(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )

    if args.history_json:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.history_json)), exist_ok=True)
        with open(args.history_json, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {args.history_json}")


if __name__ == "__main__":
    main()
