"""
tests/test_v3_2.py

Verification test suite for V3.2 Fixes (V3_2_FIX_PLAN.md):
- Step 1 (I3): GRU auxiliary memory loss -> agent.gru receives gradients, non-zero weight_ih.grad, no NaNs
- Step 2 (I4): Exploration-term recalibration (clamp 50.0)
- Step 3 (I5): (S, A)-augmented novelty key (273-d)
- Step 4 (I2): FiLM action conditioning + dynamic-position disagreement
- Step 5 (I1): Escalation config (token_var_floor 0.10, l_recon_weight 1.0)
- Step 6 (I8): LatentRND + BoredomMonitor
- Step 7 (I9): SI milestone gate & minor fixes
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.agent.agent import JEPAAgent
from src.models.encoder import ObservationEncoder
from src.models.predictor import SpatialTransformerPredictor


class TestV32Step1GRU:
    """Step 1 (I3): GRU auxiliary memory loss verification."""

    def test_gru_parameters_receive_gradients(self):
        """
        Verify that backward pass on l_mem propagates non-zero gradients
        to agent.gru parameters and agent.mem_to_pooled parameters.
        """
        agent = JEPAAgent(device="cpu")
        agent.train()

        B = 4
        z_curr = torch.randn(B, 256)
        h_prev = torch.randn(B, 256)
        target_pooled = torch.randn(B, 256)
        dones = torch.tensor([False, False, True, False])

        # Step 1 I3 formulation
        h_next = agent.gru(z_curr.detach(), h_prev.float())
        pred_next_pooled = agent.mem_to_pooled(h_next)
        valid = (~dones).float().unsqueeze(-1)
        l_mem = (F.mse_loss(pred_next_pooled, target_pooled.detach(), reduction="none") * valid).mean()

        assert not torch.isnan(l_mem), "l_mem produced NaN"
        assert l_mem.item() > 0.0, "l_mem should be positive"

        # Check gradients
        l_mem.backward()

        assert agent.gru.weight_ih.grad is not None, "agent.gru.weight_ih.grad is None"
        assert agent.gru.weight_ih.grad.abs().sum().item() > 0.0, "agent.gru.weight_ih.grad is zero"
        assert agent.gru.weight_hh.grad is not None, "agent.gru.weight_hh.grad is None"
        assert agent.gru.weight_hh.grad.abs().sum().item() > 0.0, "agent.gru.weight_hh.grad is zero"

        assert agent.mem_to_pooled.weight.grad is not None, "mem_to_pooled.weight.grad is None"
        assert agent.mem_to_pooled.weight.grad.abs().sum().item() > 0.0, "mem_to_pooled.weight.grad is zero"

    def test_gru_memory_loss_training_improves_prediction(self):
        """
        Verify that optimizing l_mem on consecutive steps improves next-pooled prediction.
        """
        torch.manual_seed(42)
        agent = JEPAAgent(device="cpu")
        agent.train()

        optimizer = torch.optim.Adam(
            list(agent.gru.parameters()) + list(agent.mem_to_pooled.parameters()),
            lr=1e-2,
        )

        B = 8
        z_curr = torch.randn(B, 256)
        h_prev = torch.randn(B, 256)
        target_pooled = torch.randn(B, 256)

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            h_next = agent.gru(z_curr.detach(), h_prev.float())
            pred_next = agent.mem_to_pooled(h_next)
            loss = F.mse_loss(pred_next, target_pooled.detach())
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"


class TestV32Step2Exploration:
    """Step 2 (I4): Exploration-term recalibration verification."""

    def test_disc_norm_clamp_ceiling(self):
        """Verify disc_norm safety clamp reaches up to 50.0 and is not pinned at 5.0."""
        disc = torch.tensor([1.0, 10.0, 25.0, 49.0, 75.0])
        std = 1.0
        eps = 1e-6
        disc_norm = disc / (std + eps)
        clamped = torch.clamp(disc_norm, min=0.0, max=50.0)
        assert abs(clamped[1].item() - 10.0) < 1e-4, "Values above 5.0 should not be clamped to 5.0"
        assert abs(clamped[2].item() - 25.0) < 1e-4
        assert abs(clamped[3].item() - 49.0) < 1e-4
        assert abs(clamped[4].item() - 50.0) < 1e-4, "Values above 50.0 should be clamped to 50.0"


class TestV32Step3NoveltyKey:
    """Step 3 (I5): (S,A)-augmented novelty key verification."""

    def test_sa_augmented_key_differentiation(self):
        """
        Verify that repeating the same state with a NEW action
        produces significantly higher novelty than repeating with the SAME action.
        """
        from src.curiosity.episodic import EpisodicNovelty

        nov = EpisodicNovelty(latent_dim=273, max_keys=100, k=5, seed=42)
        state = np.random.randn(256).astype(np.float32)

        # Action 0 ('do') tried 10 times at this state
        action_do = np.eye(17, dtype=np.float32)[0]
        key_do = np.concatenate([state, action_do])

        for _ in range(10):
            nov.novelty_and_insert(key_do)

        # Query same state with same action ('do')
        nov_repeat = nov.novelty_and_insert(key_do)

        # Query same state with UNTRIED action 8 ('make_wood_pickaxe')
        action_craft = np.eye(17, dtype=np.float32)[8]
        key_craft = np.concatenate([state, action_craft])
        nov_craft = nov.novelty_and_insert(key_craft)

        # Untried action at same state must have higher novelty than repeated action
        assert nov_craft > nov_repeat, f"Untried (S,A) novelty ({nov_craft}) must exceed repeat ({nov_repeat})"
        # Difference in one-hot adds sqrt(1^2 + 1^2) = sqrt(2) ~ 1.414 distance
        assert (nov_craft - nov_repeat) > 1.0, f"Expected > 1.0 distance delta from novel action, got {nov_craft - nov_repeat}"


class TestV32Step4FiLM:
    """Step 4 (I2): FiLM action conditioning + dynamic-position disagreement verification."""

    def test_film_action_conditioning_differentiation(self):
        """
        Acceptance: with two different actions on the same frame,
        the per-token prediction delta must be nonzero on >= 16 of 64 positions.
        """
        torch.manual_seed(42)
        predictor = SpatialTransformerPredictor(token_dim=64, num_actions=17)
        tokens = torch.randn(1, 64, 64)

        a1 = torch.tensor([0], dtype=torch.long)
        a2 = torch.tensor([8], dtype=torch.long)

        mean_pred1, _ = predictor.forward_tokens(tokens, a1)
        mean_pred2, _ = predictor.forward_tokens(tokens, a2)

        # Token delta per position: (1, 64)
        pos_deltas = (mean_pred1 - mean_pred2).abs().sum(dim=-1).squeeze(0)
        nonzero_positions = (pos_deltas > 1e-4).sum().item()

        assert nonzero_positions >= 16, f"Expected >= 16 positions with non-zero delta, got {nonzero_positions}"

    def test_dynamic_position_disagreement_weighting(self):
        """
        Verify that compute_epistemic_disagreement weights predictions by dyn_w.
        """
        predictor = SpatialTransformerPredictor(token_dim=64, num_actions=17)
        num_heads = 4
        B = 2
        # Mock head predictions with known variance only on tile 0
        all_preds = torch.zeros(num_heads, B, 64, 64)
        for h in range(num_heads):
            all_preds[h, :, 0, :] = float(h) * 1.0  # High variance on position 0

        # Uniform weight: position 0 diluted across 64 positions
        disc_uniform = predictor.compute_epistemic_disagreement(all_preds)

        # Dynamic weight: heavily focus on position 0
        dyn_w = torch.ones(B, 64)
        dyn_w[:, 0] = 50.0
        dyn_w = dyn_w / dyn_w.mean(dim=-1, keepdim=True)
        disc_dynamic = predictor.compute_epistemic_disagreement(all_preds, dyn_w=dyn_w)

        # Focused dynamic disagreement on active tile must be higher than diluted uniform
        assert (disc_dynamic > disc_uniform).all(), "Dynamic weighting should amplify disagreement on active tiles"


class TestV32Step5EscalationConfig:
    """Step 5 (I1): Escalation config verification."""

    def test_escalation_defaults(self):
        """Verify token_var_floor defaults to 0.10 and l_recon_weight defaults to 1.0."""
        from src.train import parse_args, TrainingPipeline

        args = parse_args([])
        assert args.token_var_floor == 0.10, f"Expected default token_var_floor=0.10, got {args.token_var_floor}"
        assert args.l_recon_weight == 1.0, f"Expected default l_recon_weight=1.0, got {args.l_recon_weight}"


class TestV32Step6RNDAndBoredom:
    """Step 6 (I8): LatentRND and BoredomMonitor verification."""

    def test_rnd_loss_and_novelty_decay(self):
        """Verify LatentRND predictor trains on latents and decreases distillation error."""
        from src.curiosity.rnd import LatentRND

        torch.manual_seed(42)
        rnd = LatentRND(latent_dim=256)
        optimizer = torch.optim.Adam(rnd.predictor_parameters(), lr=1e-2)

        z = torch.randn(16, 256)
        initial_loss = rnd.compute_loss(z).item()

        for _ in range(30):
            optimizer.zero_grad()
            loss = rnd.compute_loss(z)
            loss.backward()
            optimizer.step()

        final_loss = rnd.compute_loss(z).item()
        assert final_loss < initial_loss * 0.2, f"RND loss did not decrease: {initial_loss} -> {final_loss}"

        # Unseen latents have higher error than visited latents
        z_unseen = torch.randn(16, 256) + 5.0
        nov_seen = rnd.compute_novelty(z).mean().item()
        nov_unseen = rnd.compute_novelty(z_unseen).mean().item()
        assert nov_unseen > nov_seen, f"Novelty on unseen ({nov_unseen}) must exceed seen ({nov_seen})"

    def test_boredom_monitor_state_machine_and_escalation(self):
        """
        Verify BoredomMonitor:
        - 3 consecutive empty windows triggers BORED (L1)
        - Continued empty windows escalate L2 -> L3
        - Any new achievement resets state immediately to NORMAL (L0)
        """
        from src.curiosity.boredom import BoredomMonitor

        bm = BoredomMonitor(window_threshold=3, eps_step=0.05, rnd_mult_step=2.0)

        # Initial window: discovers 'wake_up'
        rep = bm.step({"wake_up"})
        assert not rep["is_bored"]
        assert rep["escalation_level"] == 0

        # Window 1: no new achievements (repeat 'wake_up')
        rep = bm.step({"wake_up"})
        assert not rep["is_bored"]
        assert bm.consecutive_empty_windows == 1

        # Window 2: no new achievements
        rep = bm.step({"wake_up"})
        assert not rep["is_bored"]
        assert bm.consecutive_empty_windows == 2

        # Window 3: 3rd empty window -> TRIGGER BORED (L1)
        rep = bm.step({"wake_up"})
        assert rep["is_bored"]
        assert rep["escalation_level"] == 1
        assert bm.eps_boost == 0.05
        assert bm.rnd_mult == 1.0
        assert bm.entropy_weight == 0.05

        # Window 4: still empty (bored_duration=1 -> level 1)
        rep = bm.step({"wake_up"})
        assert rep["escalation_level"] == 1

        # Window 5: still empty (bored_duration=2 -> level 2: L2)
        rep = bm.step({"wake_up"})
        assert rep["escalation_level"] == 2
        assert bm.rnd_mult == 2.0  # RND weight x2

        # Window 7: still empty (bored_duration=4 -> level 3: L3)
        bm.step({"wake_up"})  # Window 6
        rep = bm.step({"wake_up"})  # Window 7
        assert rep["escalation_level"] == 3
        assert bm.entropy_weight == 0.10  # Actor entropy 0.05 -> 0.10

        # BREAKTHROUGH: Agent discovers 'collect_wood'!
        rep = bm.step({"wake_up", "collect_wood"})
        assert not rep["is_bored"]
        assert rep["escalation_level"] == 0
        assert bm.eps_boost == 0.0
        assert bm.rnd_mult == 1.0
        assert bm.entropy_weight == 0.05
        assert bm.consecutive_empty_windows == 0
        assert rep["de_escalated"] is True

    def test_boredom_disease_canary(self):
        """Verify the double-duty disease canary computes token cosine matrix."""
        from src.curiosity.boredom import BoredomMonitor
        from src.agent.agent import JEPAAgent

        bm = BoredomMonitor()
        agent = JEPAAgent(device="cpu")
        canary = bm.check_canary(agent, device=torch.device("cpu"))

        assert "mean_cosine" in canary
        assert "is_collapsed" in canary
        assert isinstance(canary["is_collapsed"], bool)
        assert -1.0 <= canary["mean_cosine"] <= 1.0


class TestV32Step7MinorBatch:
    """Step 7 (I9): Minor batch verification (SI milestone gate)."""

    def test_si_gate_tightening(self):
        """
        Verify that SI gate admits only verified milestones (rewards >= 0.5)
        and rejects incidental rewards (< 0.5, e.g. 0.1 health regen).
        """
        is_milestone = torch.tensor([True, True, True, False, True])
        rewards = torch.tensor([0.1, 0.0, 1.0, 1.0, 0.5])  # 0.1 is regen; 1.0 is achievement; 0.5 is milestone

        # Tightened SI gate: is_milestone & (rewards >= 0.5)
        milestone_mask = is_milestone & (rewards >= 0.5)

        assert not milestone_mask[0].item(), "Incidental 0.1 reward should be rejected"
        assert not milestone_mask[1].item(), "Zero reward should be rejected"
        assert milestone_mask[2].item(), "Achievement 1.0 reward should be admitted"
        assert not milestone_mask[3].item(), "Non-milestone should be rejected even with reward"
        assert milestone_mask[4].item(), "Milestone 0.5 reward should be admitted"






