"""
genesis_jepa.tests.test_gate6_training_loop
===========================================
Gate 6 Verification Suite:
1. Self-Play Trainer train_step() gradient execution across World Model, Critic, Actor.
2. Weight update verification: parameters change after training step.
3. Full self-play episode rollout with unprivileged Crafter environment.
4. Record tracking and checkpoint saving/loading.
"""
import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from genesis_jepa.agent import GenesisVisualAgent
from genesis_jepa.config import GenesisConfig
from genesis_jepa.replay import ValueStratifiedReplayBuffer
from genesis_jepa.train import GenesisTrainer


class TestGate6TrainingLoop:
    def test_train_step_gradient_and_weight_updates(self):
        """Verifies train_step updates World Model, Critic, and Actor parameters."""
        config = GenesisConfig(token_dim=192)
        trainer = GenesisTrainer(config=config)

        # Record initial weights
        init_wm_weight = next(trainer.agent.world_model.encoder.parameters()).clone().detach()
        init_critic_weight = next(trainer.agent.critic.parameters()).clone().detach()
        init_actor_weight = next(trainer.agent.actor.parameters()).clone().detach()

        # Populate replay buffer with dummy data
        dummy_obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        for i in range(40):
            trainer.replay_buffer.add(
                obs=dummy_obs,
                action=i % 17,
                reward=1.0 if i % 2 == 0 else 0.0,
                next_obs=dummy_obs,
                done=False,
            )

        batch = trainer.replay_buffer.sample_batch(batch_size=16)
        metrics = trainer.train_step(batch)

        assert "loss_wm" in metrics
        assert "loss_critic" in metrics
        assert "loss_actor" in metrics

        # Verify weights were updated
        updated_wm_weight = next(trainer.agent.world_model.encoder.parameters()).detach()
        updated_critic_weight = next(trainer.agent.critic.parameters()).detach()
        updated_actor_weight = next(trainer.agent.actor.parameters()).detach()

        assert not torch.allclose(init_wm_weight, updated_wm_weight), "World Model weights failed to update!"
        assert not torch.allclose(init_critic_weight, updated_critic_weight), "Critic weights failed to update!"
        assert not torch.allclose(init_actor_weight, updated_actor_weight), "Actor weights failed to update!"

    def test_self_play_episode_and_checkpoint_roundtrip(self):
        """Runs a short self-play episode and verifies checkpoint saving/loading."""
        import crafter
        env = crafter.Env(seed=42)

        config = GenesisConfig()
        trainer = GenesisTrainer(config=config)

        # Run 1 quick self-play episode
        ep_result = trainer.self_play_episode(
            env=env,
            episode_idx=1,
            max_steps=20,
            train_every=2,
            verbose=False,
        )

        assert ep_result["steps"] > 0
        assert "mean_wm_loss" in ep_result

        # Test checkpoint roundtrip
        temp_dir = tempfile.mkdtemp()
        ckpt_path = os.path.join(temp_dir, "test_ckpt.pt")
        try:
            trainer.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            new_trainer = GenesisTrainer(config=config)
            new_trainer.load_checkpoint(ckpt_path)
            assert new_trainer.best_length == trainer.best_length
            assert new_trainer.training_steps == trainer.training_steps
        finally:
            shutil.rmtree(temp_dir)
            if hasattr(env, "close"):
                env.close()
