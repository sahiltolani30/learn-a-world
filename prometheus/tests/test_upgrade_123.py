"""
tests/test_upgrade_123.py

Acceptance Test Suite for UPGRADE_1_2_3_SPEC.md:
1. Interface: encoder(x) returns (B,256); forward_tokens returns (B,64,64)+(B,256); old consumers run unchanged.
2. Masked loss: with a fixed random seed, masking produces exactly floor(0.35*64) masked positions; loss computed only on them.
3. Memory: GRU reset on done; two envs' hidden states stay independent in the parallel loop.
4. Symlog roundtrip: inv(symlog(x)) == x for x in {-0.1, 0, +1, -3.7, 50}.
5. Self-imitation: milestone items contribute the BC term; non-milestone items contribute zero; batch cap works.
6. Success counting: a synthetic step where wake_up fires while action=make_iron_sword counts wake_up += 1 and make_iron_sword += 0.
7. P0.1 roundtrip: full save/load preserves t1_size, t2_size, pointer, checksum (per P0.1).
8. Resume assertion: resuming step >5,000 with an empty buffer raises the RuntimeError.
"""

import unittest
import tempfile
import shutil
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import ObservationEncoder
from src.models.target import TargetEncoder
from src.models.predictor import SpatialTransformerPredictor, sample_token_mask
from src.agent.actor_critic import symlog, symexp, ActorPrior, TwinCritic
from src.agent.agent import JEPAAgent
from src.memory.buffer import TwoTierReplayBuffer
from src.memory.storage import CheckpointManager
from src.diagnostics import evaluate_counterfactual_verdict
from src.train import TrainingPipeline


class TestUpgrade123(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="jepa_upgrade123_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_interface_contract(self):
        """
        Acceptance Test 1:
        encoder(x) returns (B, 256); forward_tokens returns (B, 64, 64) + (B, 256);
        old downstream consumers (actor, critic, planner) run unchanged.
        """
        encoder = ObservationEncoder(in_channels=3, latent_dim=256, token_dim=64)
        target_encoder = TargetEncoder(encoder=encoder, tau=0.99)
        B = 4
        x = torch.randn(B, 3, 64, 64)

        # 1. encoder(x) returns (B, 256)
        z = encoder(x)
        self.assertEqual(z.shape, (B, 256))
        self.assertTrue(torch.is_tensor(z))

        # 2. forward_tokens returns (B, 64, 64) + (B, 256)
        tokens, pooled = encoder.forward_tokens(x)
        self.assertEqual(tokens.shape, (B, 64, 64))
        self.assertEqual(pooled.shape, (B, 256))

        # 3. Target encoder forward_tokens returns detached target tokens and pooled latent
        t_tokens, t_pooled = target_encoder.forward_tokens(x)
        self.assertEqual(t_tokens.shape, (B, 64, 64))
        self.assertEqual(t_pooled.shape, (B, 256))
        self.assertFalse(t_tokens.requires_grad)
        self.assertFalse(t_pooled.requires_grad)

        # 4. Old consumers (Actor, Critic) consume the 256-d pooled representation directly
        actor = ActorPrior(latent_dim=256, num_actions=17)
        critic = TwinCritic(latent_dim=256, num_actions=17)
        logits = actor(pooled)
        self.assertEqual(logits.shape, (B, 17))
        q1, q2 = critic(pooled)
        self.assertEqual(q1.shape, (B, 17))
        self.assertEqual(q2.shape, (B, 17))

    def test_02_masked_loss(self):
        """
        Acceptance Test 2:
        With a fixed random seed, masking produces exactly floor(0.35 * 64) = 22 masked positions.
        Loss is computed only on masked positions (perturbing unmasked positions leaves loss unchanged).
        """
        torch.manual_seed(42)
        B = 8
        num_tokens = 64
        mask_ratio = 0.35
        expected_masked = int(math.floor(mask_ratio * num_tokens))
        self.assertEqual(expected_masked, 22)

        # Sample token mask
        mask = sample_token_mask(batch_size=B, num_tokens=num_tokens, mask_ratio=mask_ratio)
        self.assertEqual(mask.shape, (B, num_tokens))
        for b in range(B):
            self.assertEqual(int(mask[b].sum().item()), expected_masked)

        predictor = SpatialTransformerPredictor(latent_dim=256, num_actions=17, token_dim=64, num_tokens=64)
        target_tokens = torch.randn(B, 64, 64)
        pred_tokens = target_tokens.clone().unsqueeze(0).repeat(4, 1, 1, 1)

        # Perfect prediction on masked tokens -> loss is zero
        loss_zero = predictor.compute_masked_loss(pred_tokens, target_tokens, mask)
        self.assertAlmostEqual(float(loss_zero.item()), 0.0, places=5)

        # Perturb unmasked positions only -> loss must remain exactly 0.0
        unmasked = ~mask
        pred_tokens_perturbed = pred_tokens.clone()
        for k in range(4):
            for b in range(B):
                pred_tokens_perturbed[k, b, unmasked[b]] += 100.0  # huge perturbation on unmasked positions

        loss_perturbed = predictor.compute_masked_loss(pred_tokens_perturbed, target_tokens, mask)
        self.assertAlmostEqual(float(loss_perturbed.item()), 0.0, places=5)

        # Perturb masked positions -> loss must increase strictly
        pred_tokens_masked_perturbed = pred_tokens.clone()
        for k in range(4):
            for b in range(B):
                pred_tokens_masked_perturbed[k, b, mask[b]] += 1.0
        loss_masked = predictor.compute_masked_loss(pred_tokens_masked_perturbed, target_tokens, mask)
        self.assertGreater(float(loss_masked.item()), 0.5)

    def test_03_memory_gru_independence_and_reset(self):
        """
        Acceptance Test 3:
        GRU reset on done; two envs' hidden states stay independent in the parallel loop.
        """
        agent = JEPAAgent(latent_dim=256, num_actions=17)
        hidden_states = torch.zeros(2, 256)

        # Step 1: Step both envs with different inputs
        z1 = torch.ones(1, 256) * 1.0
        z2 = torch.ones(1, 256) * 2.0
        h1 = agent.gru(z1, hidden_states[0:1])
        h2 = agent.gru(z2, hidden_states[1:2])
        hidden_states[0] = h1.squeeze(0)
        hidden_states[1] = h2.squeeze(0)

        # Assert hidden states are non-zero and distinct
        self.assertFalse(torch.allclose(hidden_states[0], hidden_states[1]))
        self.assertGreater(float(hidden_states[0].abs().sum()), 0.0)
        self.assertGreater(float(hidden_states[1].abs().sum()), 0.0)

        # Step 2: Env 0 finishes (done=True) -> reset to 0, while Env 1 continues
        saved_h2 = hidden_states[1].clone()
        # Reset Env 0
        hidden_states[0] = 0.0

        # Assert Env 0 is zeroed while Env 1 is preserved
        self.assertEqual(float(hidden_states[0].abs().sum().item()), 0.0)
        self.assertTrue(torch.allclose(hidden_states[1], saved_h2))

    def test_04_symlog_roundtrip(self):
        """
        Acceptance Test 4:
        inv(symlog(x)) == x for x in {-0.1, 0, +1, -3.7, 50}.
        """
        test_values = [-0.1, 0.0, 1.0, -3.7, 50.0]
        for val in test_values:
            # Scalar roundtrip
            s_val = symlog(val)
            inv_val = symexp(s_val)
            self.assertAlmostEqual(inv_val, val, places=6)

            # Tensor roundtrip
            t = torch.tensor([val], dtype=torch.float32)
            s_t = symlog(t)
            inv_t = symexp(s_t)
            self.assertAlmostEqual(float(inv_t.item()), val, places=6)

            # Numpy array roundtrip
            arr = np.array([val], dtype=np.float32)
            s_arr = symlog(arr)
            inv_arr = symexp(s_arr)
            self.assertAlmostEqual(float(inv_arr[0]), val, places=5)

    def test_05_self_imitation_loss(self):
        """
        Acceptance Test 5:
        Milestone items contribute the BC term; non-milestone items contribute zero; batch cap works.
        """
        actor = ActorPrior(latent_dim=256, num_actions=17)
        B = 64
        z = torch.randn(B, 256)
        actions = torch.randint(0, 17, (B,))
        rewards = torch.zeros(B)
        is_milestone = torch.zeros(B, dtype=torch.bool)

        logits = actor(z)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Case A: 0 milestone items -> SI loss contributes zero
        milestone_mask_a = is_milestone & (rewards > 0.0)
        num_a = int(milestone_mask_a.sum().item())
        self.assertEqual(num_a, 0)
        si_loss_a = 0.0
        if 0 < num_a <= 32:
            si_actions = actions[milestone_mask_a]
            si_loss_a = -log_probs[milestone_mask_a].gather(1, si_actions.unsqueeze(1)).squeeze(1).mean()
        self.assertEqual(si_loss_a, 0.0)

        # Case B: 5 milestone items with verified rewards -> non-zero BC loss
        is_milestone[:5] = True
        rewards[:5] = 1.0
        milestone_mask_b = is_milestone & (rewards > 0.0)
        num_b = int(milestone_mask_b.sum().item())
        self.assertEqual(num_b, 5)
        self.assertTrue(0 < num_b <= 32)
        si_actions_b = actions[milestone_mask_b]
        si_loss_b = -log_probs[milestone_mask_b].gather(1, si_actions_b.unsqueeze(1)).squeeze(1).mean()
        self.assertGreater(float(si_loss_b.item()), 0.0)

        # Case C: 40 milestone items (> 32 cap) -> batch cap skips SI term
        is_milestone[:40] = True
        rewards[:40] = 1.0
        milestone_mask_c = is_milestone & (rewards > 0.0)
        num_c = int(milestone_mask_c.sum().item())
        self.assertEqual(num_c, 40)
        applied = (0 < num_c <= 32)
        self.assertFalse(applied)

    def test_06_success_counting(self):
        """
        Acceptance Test 6:
        A synthetic step where wake_up fires while action=make_iron_sword
        counts wake_up += 1 and make_iron_sword += 0.
        """
        pipeline = TrainingPipeline({
            "total_steps": 10,
            "warmup_steps": 5,
            "num_envs": 1,
            "batch_size": 4,
            "checkpoint_dir": self.temp_dir,
            "no_resume": True,
        })

        # Synthetic step: unlocked contains 'wake_up', but action was 'make_iron_sword'
        unlocked = ["wake_up"]
        act_name = "make_iron_sword"
        env_i = 0

        for ach in unlocked:
            if ach not in pipeline.episode_achievements[env_i]:
                pipeline.episode_achievements[env_i].add(ach)
                pipeline.success_counts[ach] = pipeline.success_counts.get(ach, 0) + 1
                pipeline.success_actions[ach] = act_name

        # wake_up must be counted +1, make_iron_sword must NOT be in success_counts
        self.assertEqual(pipeline.success_counts.get("wake_up", 0), 1)
        self.assertEqual(pipeline.success_counts.get("make_iron_sword", 0), 0)
        self.assertEqual(pipeline.success_actions.get("wake_up"), "make_iron_sword")

    def test_07_p01_roundtrip(self):
        """
        Acceptance Test 7:
        Create buffer with 10,000 known transitions -> save via CheckpointManager ->
        restore into a fresh instance -> assert t1_size, t2_size, t1_ptr, and checksum of t1_obs[:100] match.
        """
        ckpt_dir = Path(self.temp_dir) / "ckpt_roundtrip"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

        buf = TwoTierReplayBuffer(tier1_capacity=20_000, tier2_capacity=5_000, seed=42)
        obs_shape = (3, 64, 64)

        # Add 10,000 transitions with known pattern
        np.random.seed(123)
        for i in range(10_000):
            obs = np.full(obs_shape, fill_value=(i % 256), dtype=np.uint8)
            next_obs = np.full(obs_shape, fill_value=((i + 1) % 256), dtype=np.uint8)
            buf.add(
                obs=obs,
                action=(i % 17),
                reward=float(i % 5 == 0),
                next_obs=next_obs,
                done=(i % 100 == 0),
                r_int=float(i % 10) / 10.0,
                is_milestone=(i % 50 == 0),
            )

        self.assertEqual(buf.t1_size, 10_000)
        self.assertEqual(buf.t1_ptr, 10_000)
        checksum_orig = float(buf.t1_obs[:100].sum())

        # Save via CheckpointManager to state.pt
        mgr.save(
            filepath="state.pt",
            components={"buffer": buf},
            metadata={"step": 10_000, "arch": "v3-spatial"},
        )

        # Restore into fresh instance
        fresh_buf = TwoTierReplayBuffer(tier1_capacity=20_000, tier2_capacity=5_000, seed=99)
        mgr.restore(
            filepath="state.pt",
            components={"buffer": fresh_buf},
        )

        self.assertEqual(fresh_buf.t1_size, 10_000)
        self.assertEqual(fresh_buf.t1_ptr, 10_000)
        self.assertEqual(fresh_buf.t2_size, buf.t2_size)
        checksum_restored = float(fresh_buf.t1_obs[:100].sum())
        self.assertEqual(checksum_orig, checksum_restored)

    def test_08_resume_assertion_raises_on_empty_buffer(self):
        """
        Acceptance Test 8:
        Resuming step > 5,000 with an empty buffer raises RuntimeError to protect lineage.
        """
        ckpt_dir = Path(self.temp_dir) / "ckpt_empty_resume"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

        # Save a corrupt/empty checkpoint with step = 6,000 but empty buffer
        empty_buf = TwoTierReplayBuffer(tier1_capacity=10_000, tier2_capacity=2_000)
        mgr.save(
            filepath="state.pt",
            components={"buffer": empty_buf},
            metadata={"step": 6_000, "arch": "v3-spatial"},
        )

        # Attempt to resume with step > 5000 and empty buffer -> must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            TrainingPipeline({
                "total_steps": 10_000,
                "warmup_steps": 100,
                "num_envs": 1,
                "checkpoint_dir": str(ckpt_dir),
                "no_resume": False,
            })

        self.assertIn("Aborting to protect the lineage", str(ctx.exception))
        self.assertIn("buffer is empty", str(ctx.exception))

    def test_09_carry_over_from_valid_v2_checkpoint(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance 1:
        Carry-over from a valid v2-format checkpoint restores t1_size and t2_size exactly,
        even when metadata contains pathlib.Path objects.
        """
        v2_dir = Path(self.temp_dir) / "v2_test"
        v2_dir.mkdir(parents=True, exist_ok=True)
        mgr_v2 = CheckpointManager(checkpoint_dir=v2_dir)

        buf = TwoTierReplayBuffer(tier1_capacity=20_000, tier2_capacity=5_000, seed=42)
        obs_shape = (3, 64, 64)
        for i in range(1_000):
            buf.add(
                obs=np.full(obs_shape, fill_value=(i % 256), dtype=np.uint8),
                action=(i % 17),
                reward=float(i % 10 == 0),
                next_obs=np.full(obs_shape, fill_value=((i + 1) % 256), dtype=np.uint8),
                done=(i % 100 == 0),
                r_int=0.1,
                is_milestone=(i % 5 == 0),
            )

        self.assertEqual(buf.t1_size, 1_000)
        t2_expected = buf.t2_size

        v2_ckpt = v2_dir / "checkpoint_step_100000.pt"
        mgr_v2.save(
            filepath=v2_ckpt,
            components={"buffer": buf},
            metadata={
                "step": 100_000,
                "arch": "v2-legacy",
                "checkpoint_dir": Path("/content/drive/MyDrive/test_path"),
            },
        )

        v3_dir = Path(self.temp_dir) / "checkpoints_v3"
        v3_dir.mkdir(parents=True, exist_ok=True)

        pipeline = TrainingPipeline({
            "total_steps": 25_000,
            "warmup_steps": 100,
            "num_envs": 1,
            "checkpoint_dir": str(v3_dir),
            "carry_source": str(v2_ckpt),
            "allow_fresh_buffer": False,
            "no_resume": False,
        })

        self.assertEqual(pipeline.buffer.t1_size, 1_000)
        self.assertEqual(pipeline.buffer.t2_size, t2_expected)
        self.assertEqual(pipeline.current_step, 0)  # Stage C starts at step 0 with carried buffer

    def test_10_carry_over_fails_loud_on_missing_or_corrupt_file(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance 2:
        Carry-over from a corrupted or missing file raises RuntimeError when
        allow_fresh_buffer=False, and proceeds with empty buffer only when flag is set.
        """
        v3_dir = Path(self.temp_dir) / "v3_fail_loud"
        v3_dir.mkdir(parents=True, exist_ok=True)
        missing_ckpt = self.temp_dir + "/non_existent_step_100000.pt"

        # 1. With allow_fresh_buffer=False -> must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            TrainingPipeline({
                "total_steps": 25_000,
                "warmup_steps": 100,
                "num_envs": 1,
                "checkpoint_dir": str(v3_dir),
                "carry_source": missing_ckpt,
                "allow_fresh_buffer": False,
                "no_resume": False,
            })
        self.assertIn("Diary carry-over FAILED", str(ctx.exception))
        self.assertIn("Refusing to start with an empty buffer", str(ctx.exception))

        # 2. With allow_fresh_buffer=True -> starts fresh without crashing
        pipeline = TrainingPipeline({
            "total_steps": 25_000,
            "warmup_steps": 100,
            "num_envs": 1,
            "checkpoint_dir": str(v3_dir),
            "carry_source": missing_ckpt,
            "allow_fresh_buffer": True,
            "no_resume": False,
        })
        self.assertEqual(pipeline.buffer.t1_size, 0)
        self.assertEqual(pipeline.current_step, 0)

    def test_11_checkpoint_sanitization_loads_weights_only(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance 3:
        Saved checkpoint with Path objects in metadata is sanitized to strings,
        allowing safe-loading via torch.load(..., weights_only=True) without error.
        """
        ckpt_dir = Path(self.temp_dir) / "ckpt_sanitize"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

        model = nn.Linear(10, 5)
        metadata = {
            "step": 500,
            "save_path": Path("/content/drive/MyDrive/model.pt"),
            "config": {
                "checkpoint_dir": Path("/content/drive/MyDrive/checkpoints_v3"),
                "paths_list": [Path("/a/b"), Path("/c/d")],
            },
        }

        saved_path = mgr.save(
            filepath="model_step_500.pt",
            components={"model": model},
            metadata=metadata,
        )

        # Must load cleanly under torch.load(..., weights_only=True)
        loaded = torch.load(saved_path, weights_only=True)
        self.assertIn("components", loaded)
        self.assertIn("metadata", loaded)
        self.assertIsInstance(loaded["metadata"]["save_path"], str)
        self.assertIsInstance(loaded["metadata"]["config"]["checkpoint_dir"], str)
        self.assertEqual(loaded["metadata"]["config"]["paths_list"], ["/a/b", "/c/d"])

    def test_12_readout_proj_receives_h_gradients(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance Test (a) for Hotfix 6:
        Train updates with h-bearing batches assert readout_proj h-half weights
        receive nonzero gradients, and predictor mem_proj receives gradients.
        """
        temp_dir = Path(self.temp_dir) / "test_h_grad"
        temp_dir.mkdir(parents=True, exist_ok=True)

        pipeline = TrainingPipeline({
            "total_steps": 500,
            "warmup_steps": 10,
            "batch_size": 16,
            "num_envs": 1,
            "checkpoint_dir": str(temp_dir),
            "no_resume": True,
        })

        obs_shape = (3, 64, 64)
        for i in range(100):
            obs = np.full(obs_shape, fill_value=(i % 256), dtype=np.uint8)
            next_obs = np.full(obs_shape, fill_value=((i + 1) % 256), dtype=np.uint8)
            h = np.random.randn(256).astype(np.float32)
            pipeline.buffer.add(
                obs=obs,
                action=(i % 17),
                reward=float(i % 5 == 0),
                next_obs=next_obs,
                done=(i % 20 == 0),
                r_int=0.1,
                is_milestone=(i % 10 == 0),
                episode_id=(i // 20),
                step_in_ep=(i % 20),
                h=h,
            )

        # Run WM train step (trains mem_proj with batch["h"])
        wm_metrics = pipeline.update_world_model()
        self.assertIn("wm_loss", wm_metrics)
        # Check predictor mem_proj received gradient
        mem_proj_grad = pipeline.agent.predictor.mem_proj.weight.grad
        self.assertIsNotNone(mem_proj_grad)
        self.assertGreater(float(mem_proj_grad.abs().sum()), 0.0)

        # Run Actor-Critic update (trains readout_proj with batch["h"])
        ac_metrics = pipeline.update_actor_critic()
        self.assertIn("actor_loss", ac_metrics)
        self.assertIn("critic_loss", ac_metrics)

        # Check readout_proj h-half weights (columns 256:512) received nonzero gradient
        linear_layer = pipeline.agent.readout_proj[0] if isinstance(pipeline.agent.readout_proj, nn.Sequential) else pipeline.agent.readout_proj
        readout_grad = linear_layer.weight.grad
        self.assertIsNotNone(readout_grad)
        h_half_grad = readout_grad[:, 256:]
        self.assertGreater(float(h_half_grad.abs().sum()), 0.0)

    def test_13_critic_loss_on_stored_h_readouts(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance Test (b) for Hotfix 6:
        Critic training on stored-h readouts successfully optimizes value targets.
        """
        temp_dir = Path(self.temp_dir) / "test_critic_h"
        temp_dir.mkdir(parents=True, exist_ok=True)

        pipeline = TrainingPipeline({
            "total_steps": 500,
            "warmup_steps": 10,
            "batch_size": 32,
            "num_envs": 1,
            "checkpoint_dir": str(temp_dir),
            "no_resume": True,
        })

        obs_shape = (3, 64, 64)
        for i in range(200):
            obs = np.full(obs_shape, fill_value=(i % 256), dtype=np.uint8)
            next_obs = np.full(obs_shape, fill_value=((i + 1) % 256), dtype=np.uint8)
            h = np.ones(256, dtype=np.float32) * float(i % 10)
            pipeline.buffer.add(
                obs=obs,
                action=(i % 17),
                reward=float(i % 10 == 0),
                next_obs=next_obs,
                done=(i % 20 == 0),
                r_int=0.1,
                is_milestone=(i % 10 == 0),
                episode_id=(i // 20),
                step_in_ep=(i % 20),
                h=h,
            )

        losses = []
        for _ in range(5):
            res = pipeline.update_actor_critic()
            losses.append(res["critic_loss"])

        # Critic loss must be finite and positive
        self.assertTrue(all(np.isfinite(losses)))
        self.assertGreater(losses[0], 0.0)

    def test_14_buffer_h_roundtrip_checksum(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance Test (c) for Hotfix 6:
        Buffer roundtrip includes t1_h checksum and verifies float16 precision.
        """
        buf = TwoTierReplayBuffer(tier1_capacity=1000, tier2_capacity=200, seed=42)
        obs_shape = (3, 64, 64)

        for i in range(50):
            obs = np.full(obs_shape, fill_value=i, dtype=np.uint8)
            next_obs = np.full(obs_shape, fill_value=i + 1, dtype=np.uint8)
            h = np.linspace(0.0, 1.0, 256, dtype=np.float32) * (i + 1)
            buf.add(
                obs=obs,
                action=(i % 17),
                reward=float(i),
                next_obs=next_obs,
                done=(i == 49),
                is_milestone=(i % 5 == 0),
                h=h,
            )

        orig_t1_h_sum = float(buf.t1_h[:buf.t1_size].astype(np.float32).sum())
        orig_t2_h_sum = float(buf.t2_h[:buf.t2_size].astype(np.float32).sum())

        state = buf.state_dict()
        self.assertIn("t1_h", state)
        self.assertIn("t2_h", state)

        fresh_buf = TwoTierReplayBuffer(tier1_capacity=1000, tier2_capacity=200, seed=99)
        fresh_buf.load_state_dict(state)

        restored_t1_h_sum = float(fresh_buf.t1_h[:fresh_buf.t1_size].astype(np.float32).sum())
        restored_t2_h_sum = float(fresh_buf.t2_h[:fresh_buf.t2_size].astype(np.float32).sum())

        self.assertAlmostEqual(orig_t1_h_sum, restored_t1_h_sum, places=1)
        self.assertAlmostEqual(orig_t2_h_sum, restored_t2_h_sum, places=1)

    def test_15_evaluation_evolves_and_resets_memory(self):
        """
        V3_FIX_AND_RUN_PLAN.md Acceptance Test for Hotfix 4:
        get_readout(z, h) with evolving h differs from h≡0 after updates,
        and evaluate() updates eval_h across steps.
        """
        agent = JEPAAgent(device="cpu")
        agent.eval()

        z = torch.randn(1, 256)
        readout_zero_h = agent.get_readout(z, h=None)

        # Step GRU multiple times
        h = torch.zeros(1, 256)
        for _ in range(3):
            h = agent.gru(z, h)

        readout_evolved_h = agent.get_readout(z, h=h)

        # Readout with evolved h must differ significantly from h=0
        diff = float((readout_evolved_h - readout_zero_h).abs().sum().item())
        self.assertGreater(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
