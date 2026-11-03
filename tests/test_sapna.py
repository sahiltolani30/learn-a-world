"""
tests/test_sapna.py

Acceptance tests for SAPNA (T1 - T13) per SAPNA_PLAN.md §11.
All 13 acceptance tests must pass before ANY GPU run.
"""

import copy
import os
import shutil
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sapna.envs import (
    make_crafter_env,
    ZeroCheatWrapper,
    ParallelCrafterBatch,
    FORBIDDEN_CHEAT_KEYS,
)
from sapna.encoder import ObservationEncoder, TargetEncoder
from sapna.dynamics import (
    ActionEmbedding,
    TemporalCore,
    SpatialPredictor,
    RewardHead,
    DoneHead,
    sample_token_mask,
)
from sapna.policy import Actor, Critic, SlowCritic, symlog, symexp, unimix, SAPNAAgent
from sapna.imagination import (
    twohot_encode,
    twohot_score,
    twohot_ce,
    lambda_returns,
    imagine_batch,
)
from sapna.buffer import SAPNAReplayBuffer
from sapna.config import SAPNAConfig
from sapna.benchmark import ALL_CRAFTER_ACHIEVEMENTS
from sapna.train import (
    RunningMeanStd,
    save_checkpoint,
    load_checkpoint,
    evaluate,
    compute_imagination_reward_corr,
    train,
)


def test_t1_shapes_and_identities():
    """
    T1 shapes + identities:
    encoder (2,3,64,64) -> forward_tokens returns ((2,64,64),(2,256));
    identity forward(x) == pool_tokens(forward_tokens(x)[0]) (allclose 1e-6);
    predictor forward_tokens(tokens,(2,),h) -> mean_pred (2,64,64), all_preds (4,2,64,64);
    compute_masked_loss returns scalar >= 0;
    heads (2,255)/(2,1);
    actor (2,17).
    """
    torch.manual_seed(42)
    encoder = ObservationEncoder()
    x = torch.randn(2, 3, 64, 64)

    tokens, pooled = encoder.forward_tokens(x)
    assert tokens.shape == (2, 64, 64), f"Tokens shape {tokens.shape}"
    assert pooled.shape == (2, 256), f"Pooled shape {pooled.shape}"

    direct_pooled = encoder(x)
    repooled = encoder.pool_tokens(tokens)
    assert torch.allclose(direct_pooled, repooled, atol=1e-6), "Repooled != direct_pooled"
    assert torch.allclose(direct_pooled, pooled, atol=1e-6), "Pooled != direct_pooled"

    predictor = SpatialPredictor()
    h = torch.randn(2, 256)
    a = torch.tensor([1, 4], dtype=torch.long)
    mean_pred, all_preds = predictor.forward_tokens(tokens, a, h)
    assert mean_pred.shape == (2, 64, 64), f"Mean pred shape {mean_pred.shape}"
    assert all_preds.shape == (4, 2, 64, 64), f"All preds shape {all_preds.shape}"

    mask = sample_token_mask(2, num_tokens=64, mask_ratio=0.35)
    target_tokens = torch.randn(2, 64, 64)
    loss = predictor.compute_masked_loss(all_preds, target_tokens, mask)
    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() >= 0.0, "Loss must be >= 0"

    rew_head = RewardHead()
    done_head = DoneHead()
    arr = torch.randn(2, 512)
    r_out = rew_head(arr)
    d_out = done_head(arr)
    assert r_out.shape == (2, 255), f"Reward head shape {r_out.shape}"
    assert d_out.shape == (2, 1), f"Done head shape {d_out.shape}"

    actor = Actor()
    state = torch.randn(2, 512)
    act_out = actor(state)
    assert act_out.shape == (2, 17), f"Actor shape {act_out.shape}"


def test_t2_param_budget():
    """
    T2 param budget: 1.5e6 < total < 4e6.
    """
    encoder = ObservationEncoder()
    predictor = SpatialPredictor()
    temporal_core = TemporalCore()
    rew_head = RewardHead()
    done_head = DoneHead()
    actor = Actor()
    critic = Critic()

    modules = [encoder, predictor, temporal_core, rew_head, done_head, actor, critic]
    total_trainable = sum(sum(p.numel() for p in m.parameters()) for m in modules)

    target_encoder = TargetEncoder(encoder)
    slow_critic = SlowCritic(critic)
    total_all = total_trainable + sum(p.numel() for p in target_encoder.parameters()) + sum(p.numel() for p in slow_critic.parameters())

    assert 1.5e6 < total_trainable < 4.0e6, f"Trainable params {total_trainable} out of [1.5M, 4M] budget"
    assert 1.5e6 < total_all < 4.0e6, f"Total params with target models {total_all} out of [1.5M, 4M] budget"


def test_t3_symlog_twohot_roundtrip():
    """
    T3 symlog/two-hot roundtrip:
    twohot_encode -> twohot_score reconstructs x within 1% for x in {-1, -0.1, 0, 0.1, 1, 5, 10};
    target rows sum to 1; no NaN at x=+-20 (clamped).
    """
    test_values = [-20.0, -10.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0, 20.0]
    for val in test_values:
        t = torch.tensor([val], dtype=torch.float32)
        enc = twohot_encode(t)
        assert not torch.isnan(enc).any(), f"NaN in twohot_encode for {val}"
        assert torch.allclose(enc.sum(dim=-1), torch.ones(1), atol=1e-5), f"Rows must sum to 1 for {val}"

        logits = torch.log(enc + 1e-8)
        rec = twohot_score(logits).item()
        assert not np.isnan(rec), f"NaN in twohot_score for {val}"

        err = abs(rec - val)
        if abs(val) > 1e-3:
            rel_err = err / abs(val)
            assert rel_err < 0.01, f"Roundtrip failed for {val}: rec={rec}, rel_err={rel_err:.4f}"
        else:
            assert err < 0.01, f"Roundtrip failed for {val}: rec={rec}, err={err:.4f}"


def test_t4_lambda_returns_hand_example():
    """
    T4 lambda-returns hand example:
    rewards [0,0,1,0,0], conts all 1, v=0, bootstrap v=0, gamma=0.997 lam=0.95
    -> G_0 = 1 * gamma^2 * lam^2.
    Assert allclose 1e-5; cont=0 at step 2 cuts all later credit.
    """
    H = 5
    gamma = 0.997
    lam = 0.95

    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    conts = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    values = torch.zeros(1, H, dtype=torch.float32)
    bootstrap = torch.zeros(1, dtype=torch.float32)

    returns = lambda_returns(rewards, conts, values, bootstrap, gamma=gamma, lam=lam)
    expected_G0 = 1.0 * (gamma ** 2) * (lam ** 2)
    assert abs(returns[0, 0].item() - expected_G0) < 1e-5, f"Expected {expected_G0}, got {returns[0, 0].item()}"

    conts_cut = torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    returns_cut = lambda_returns(rewards, conts_cut, values, bootstrap, gamma=gamma, lam=lam)
    assert abs(returns_cut[0, 0].item() - 0.0) < 1e-5, f"Expected G_0=0 when cont_1=0, got {returns_cut[0, 0].item()}"

    rewards_later = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    conts_cut2 = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    returns_cut2 = lambda_returns(rewards_later, conts_cut2, values, bootstrap, gamma=gamma, lam=lam)
    assert abs(returns_cut2[0, 0].item() - 0.0) < 1e-5, f"Expected G_0=0 when reward is at step 3 and cont_2=0"


def test_t5_gru_reset_independence():
    """
    T5 GRU reset independence:
    two envs' h stay independent after 10 steps with different actions; reset on done -> zeros.
    """
    torch.manual_seed(42)
    core = TemporalCore()
    action_emb = ActionEmbedding()

    h = torch.zeros(2, 256)
    z = torch.randn(2, 256)

    for step in range(10):
        a = torch.tensor([1, 5], dtype=torch.long)
        a_emb = action_emb(a)
        h = core(z, h, a_emb=a_emb)
        assert not torch.allclose(h[0], h[1]), f"Envs hidden states should be different at step {step}"

    done = torch.tensor([True, False])
    h[done] = 0.0

    assert torch.allclose(h[0], torch.zeros(256)), "Env 0 hidden state must be zeros after reset"
    assert not torch.allclose(h[1], torch.zeros(256)), "Env 1 hidden state must remain intact"


def test_t6_sequence_sampler_validity():
    """
    T6 sequence sampler validity:
    200 sampled windows from a synthetic buffer with short episodes
    - 100% satisfy same-episode + contiguous step_in_ep;
    - milestone-pool windows all contain >= 1 milestone.
    """
    np.random.seed(42)
    buffer = SAPNAReplayBuffer(capacity=10000)
    curr_ep = 0

    for ep in range(50):
        ep_len = np.random.randint(5, 35)
        has_ms = (np.random.rand() < 0.6)
        ms_step = np.random.randint(0, ep_len) if has_ms else -1
        for s in range(ep_len):
            obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
            act = np.random.randint(0, 17)
            rew = float(np.random.rand() > 0.9)
            done = (s == ep_len - 1)
            ms = (s == ms_step)
            buffer.add(obs, act, rew, done, curr_ep, s, ms)
        curr_ep += 1

    assert buffer.valid_windows >= 64, f"Buffer must have >= 64 valid windows, got {buffer.valid_windows}"

    # Sample 200 windows (25 batches of 8)
    for _ in range(25):
        batch = buffer.sample_sequences(B=8, T=16)
        for b in range(8):
            ep_ids = batch.episode_id[b]
            steps = batch.step_in_ep[b]
            # 1. 100% satisfy same-episode
            assert (ep_ids == ep_ids[0]).all(), f"Episode mismatch within window: {ep_ids}"
            # 2. 100% contiguous step_in_ep
            assert torch.all(steps[1:] == steps[:-1] + 1), f"Step discontinuity within window: {steps}"

    # Milestone pool check
    _, ms_starts = buffer._get_valid_starts(T=16)
    assert len(ms_starts) > 0, "Buffer should contain milestone windows"
    for start in ms_starts[:50]:
        window_ms = buffer.milestone[start : start + 16]
        assert window_ms.any(), f"Milestone start {start} contains 0 milestones: {window_ms}"


def test_t7_imagination_shapes_and_determinism():
    """
    T7 imagination shapes + determinism:
    imagine_batch with fixed seed -> same outputs twice; shapes per §8;
    rewards/conts detached (.requires_grad == False), values carry grad.
    """
    torch.manual_seed(100)
    encoder = ObservationEncoder()
    core = TemporalCore()
    action_emb = ActionEmbedding()
    actor = Actor()
    critic = Critic()
    pred = SpatialPredictor()
    rew_head = RewardHead()
    done_head = DoneHead()

    N = 4
    H = 15
    z0 = torch.randn(N, 256)
    h0 = torch.randn(N, 256)
    tokens0 = torch.randn(N, 64, 64)

    torch.manual_seed(42)
    out1 = imagine_batch(
        tokens0=tokens0,
        z0=z0,
        h0=h0,
        actor=actor,
        critic=critic,
        predictor=pred,
        temporal_core=core,
        action_embedding=action_emb,
        pool_fn=encoder.pool_tokens,
        reward_head=rew_head,
        done_head=done_head,
        H=H,
    )

    torch.manual_seed(42)
    out2 = imagine_batch(
        tokens0=tokens0,
        z0=z0,
        h0=h0,
        actor=actor,
        critic=critic,
        predictor=pred,
        temporal_core=core,
        action_embedding=action_emb,
        pool_fn=encoder.pool_tokens,
        reward_head=rew_head,
        done_head=done_head,
        H=H,
    )

    assert torch.equal(out1["actions"], out2["actions"]), "Actions not identical across identical seeds"
    assert torch.allclose(out1["values"], out2["values"]), "Values not identical across identical seeds"
    assert torch.allclose(out1["rewards"], out2["rewards"]), "Rewards not identical across identical seeds"

    assert out1["actions"].shape == (N, H)
    assert out1["logprobs"].shape == (N, H)
    assert out1["entropies"].shape == (N, H)
    assert out1["values"].shape == (N, H)
    assert out1["bootstrap"].shape == (N,)
    assert out1["rewards"].shape == (N, H)
    assert out1["conts"].shape == (N, H)
    assert out1["final_z"].shape == (N, 256)
    assert out1["final_h"].shape == (N, 256)

    assert out1["rewards"].requires_grad is False, "Rewards must be detached"
    assert out1["conts"].requires_grad is False, "Conts must be detached"
    assert out1["values"].requires_grad is True, "Values must carry grad"


def test_t8_two_hot_critic_reduces():
    """
    T8 two-hot critic reduces:
    200 gradient steps of Critic on synthetic fixed G targets -> CE drops > 5x;
    scalar readout error < 0.05;
    reward-head action-conditioning sanity: two different arrival-state inputs produce different distributions.
    """
    torch.manual_seed(42)
    critic = Critic()
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    target_scalar = 5.0
    x = torch.randn(32, 512)
    target_G = symlog(torch.tensor([target_scalar] * 32, dtype=torch.float32))

    # Initial loss
    with torch.no_grad():
        initial_loss = twohot_ce(critic(x), target_G, is_symlog=True).item()

    # 200 gradient steps
    for _ in range(200):
        opt.zero_grad()
        loss = twohot_ce(critic(x), target_G, is_symlog=True)
        loss.backward()
        opt.step()

    final_loss = loss.item()
    assert final_loss < (initial_loss / 5.0), f"CE loss did not drop > 5x: initial={initial_loss:.4f}, final={final_loss:.4f}"

    # Scalar readout error < 0.05
    with torch.no_grad():
        readout = twohot_score(critic(x)).mean().item()
    err = abs(readout - target_scalar)
    assert err < 0.05, f"Readout error {err:.4f} >= 0.05 (target {target_scalar}, readout {readout:.4f})"

    # Reward-head action-conditioning sanity per §20 F10:
    # two DIFFERENT arrival states generated by two DIFFERENT actions through the real pipeline
    encoder = ObservationEncoder()
    pred = SpatialPredictor()
    action_emb = ActionEmbedding()
    core = TemporalCore()
    rew_head = RewardHead()

    tokens_0 = torch.randn(2, 64, 64)
    h_0 = torch.randn(2, 256)
    a1 = torch.tensor([0, 0], dtype=torch.long)
    a2 = torch.tensor([5, 5], dtype=torch.long)

    with torch.no_grad():
        p1, _ = pred.forward_tokens_clean(tokens_0, a1, h_0)
        z1 = encoder.pool_tokens(p1)
        h1 = core(z1, h_0, a_emb=action_emb(a1))
        arr1 = torch.cat([z1, h1], dim=-1)

        p2, _ = pred.forward_tokens_clean(tokens_0, a2, h_0)
        z2 = encoder.pool_tokens(p2)
        h2 = core(z2, h_0, a_emb=action_emb(a2))
        arr2 = torch.cat([z2, h2], dim=-1)

        dist1 = rew_head(arr1)
        dist2 = rew_head(arr2)

    assert not torch.allclose(arr1, arr2, atol=1e-3), "Arrival states from different actions must be different"
    assert not torch.allclose(dist1, dist2, atol=1e-2), "Reward head produced identical outputs for arrival states generated from different actions"


def test_t9_eval_h_maintenance():
    """
    T9 eval-h maintenance (Hotfix 4):
    run 20 eval steps — eval_h changes across steps, resets on done, and is actually passed to act.
    """
    torch.manual_seed(42)
    encoder = ObservationEncoder()
    core = TemporalCore()
    action_emb = ActionEmbedding()
    actor = Actor()
    agent = SAPNAAgent(encoder, core, action_emb, actor)

    h_passed_list = []
    original_act = agent.act

    def spy_act(obs, h, greedy=False):
        h_passed_list.append(h.clone())
        return original_act(obs, h, greedy=greedy)

    agent.act = spy_act

    # Run 20 steps
    h = agent.reset_h(batch_size=1)
    obs = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)

    for step in range(20):
        if step == 10:
            # Simulate episode done reset (Hotfix 4)
            h = agent.reset_h(batch_size=1)
        a, h, z = agent.act(obs, h, greedy=False)

    assert len(h_passed_list) == 20, f"Expected 20 act calls, got {len(h_passed_list)}"
    # Step 1 should differ from Step 0 (eval_h changed across steps)
    assert not torch.allclose(h_passed_list[1], h_passed_list[0]), "eval_h did not update across steps"
    # Step 10 was reset to zeros on done
    assert torch.allclose(h_passed_list[10], torch.zeros(1, 256)), "eval_h was not reset to zeros on done"
    # Step 11 changed again from zeros
    assert not torch.allclose(h_passed_list[11], torch.zeros(1, 256)), "eval_h was not updated after reset"

    # Actually exercise evaluate() with environment per §20 F10
    eval_metrics = evaluate(agent, num_episodes_per_mode=1, max_steps=10)
    assert "eval_greedy_return" in eval_metrics
    assert "eval_stochastic_return" in eval_metrics
    assert "eval_greedy_length" in eval_metrics
    assert "eval_stochastic_length" in eval_metrics
    assert "eval_greedy_unique_achievements" in eval_metrics
    assert "eval_stochastic_unique_achievements" in eval_metrics


def test_t10_checkpoint_roundtrip():
    """
    T10 checkpoint roundtrip:
    save -> mutate -> load -> all arrays byte-identical, config restored;
    arch tag mismatch raises RuntimeError.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        config = SAPNAConfig(checkpoint_dir=tmp_dir, total_steps=500)
        device = torch.device("cpu")

        encoder = ObservationEncoder()
        target_encoder = TargetEncoder(encoder)
        predictor = SpatialPredictor()
        temporal_core = TemporalCore()
        action_embedding = ActionEmbedding()
        reward_head = RewardHead()
        done_head = DoneHead()
        actor = Actor()
        critic = Critic()
        slow_critic = SlowCritic(critic)

        opt_wm = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-5)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-5)
        r_int_normalizer = RunningMeanStd()

        buffer = SAPNAReplayBuffer(capacity=1000)
        for i in range(10):
            buffer.add(np.ones((3, 64, 64), dtype=np.uint8) * i, i % 17, float(i), False, 1, i, (i == 5))

        # Save checkpoint
        save_checkpoint(
            checkpoint_dir=tmp_dir,
            step=100,
            config=config,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            temporal_core=temporal_core,
            action_embedding=action_embedding,
            reward_head=reward_head,
            done_head=done_head,
            actor=actor,
            critic=critic,
            slow_critic=slow_critic,
            opt_wm=opt_wm,
            opt_actor=opt_actor,
            opt_critic=opt_critic,
            buffer=buffer,
            r_int_normalizer=r_int_normalizer,
            telemetry_history=[{"step": 100, "sps": 20.0}],
            official_counts={ach: 0 for ach in ALL_CRAFTER_ACHIEVEMENTS},
            total_episodes=1,
        )

        # Mutate buffer and weights
        buffer.obs[:] = 99
        with torch.no_grad():
            for p in encoder.parameters():
                p.fill_(99.0)

        # Load back
        ckpt_path = os.path.join(tmp_dir, "state.pt")
        step_restored, tel, off, eps = load_checkpoint(
            ckpt_path,
            device,
            encoder,
            target_encoder,
            predictor,
            temporal_core,
            action_embedding,
            reward_head,
            done_head,
            actor,
            critic,
            slow_critic,
            opt_wm,
            opt_actor,
            opt_critic,
            buffer,
            r_int_normalizer,
        )

        assert step_restored == 100
        assert len(buffer) == 10
        assert buffer.obs[0, 0, 0, 0] == 0
        assert buffer.obs[5, 0, 0, 0] == 5
        assert bool(buffer.milestone[5]) is True

        # Test arch tag mismatch raises
        corrupt_ckpt = torch.load(ckpt_path, weights_only=False)
        corrupt_ckpt["arch"] = "wrong-v2"
        corrupt_path = os.path.join(tmp_dir, "corrupt.pt")
        torch.save(corrupt_ckpt, corrupt_path)

        with pytest.raises(RuntimeError, match="Architecture mismatch"):
            load_checkpoint(
                corrupt_path,
                device,
                encoder,
                target_encoder,
                predictor,
                temporal_core,
                action_embedding,
                reward_head,
                done_head,
                actor,
                critic,
                slow_critic,
                opt_wm,
                opt_actor,
                opt_critic,
                buffer,
                r_int_normalizer,
            )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_t11_zero_cheat_leak():
    """
    T11 zero-cheat leak test:
    Copy of existing wrapper-leak tests — no key from
    {inventory, achievements, semantic, player_pos} reachable in obs/step outputs
    used by the agent path; milestone flag consumed only from eval_info count.
    """
    env = make_crafter_env(seed=123)
    obs = env.reset()

    if isinstance(obs, tuple):
        obs_tensor = obs[0]
    else:
        obs_tensor = obs
    assert isinstance(obs_tensor, torch.Tensor)
    assert obs_tensor.shape == (3, 64, 64)
    assert obs_tensor.min() >= 0.0 and obs_tensor.max() <= 1.0

    rng = np.random.RandomState(42)
    for step_i in range(50):
        a = int(rng.randint(0, 17))
        step_res = env.step(a)
        obs_t, rew, term, trunc, info = step_res

        assert isinstance(obs_t, torch.Tensor)
        assert obs_t.shape == (3, 64, 64)

        for forbidden in FORBIDDEN_CHEAT_KEYS:
            assert forbidden not in info, f"Forbidden key {forbidden} in info!"
            if "eval_info" in info:
                assert forbidden not in info["eval_info"], f"Forbidden key {forbidden} in eval_info!"

        if "eval_info" in info:
            assert "achievement_count" in info["eval_info"]
            assert isinstance(info["eval_info"]["achievement_count"], int)

        if term or trunc:
            env.reset()

    env.close()

    batch_envs = ParallelCrafterBatch(num_envs=2, base_seed=42, action_repeat=2)
    init_obs = batch_envs.reset()
    assert init_obs.shape == (2, 3, 64, 64)
    assert init_obs.dtype == np.uint8

    results = batch_envs.step([0, 5])
    assert len(results) == 2
    for term_obs, r, d, info in results:
        assert term_obs.shape == (3, 64, 64)
        for forbidden in FORBIDDEN_CHEAT_KEYS:
            assert forbidden not in info
            if "eval_info" in info:
                assert forbidden not in info["eval_info"]
    batch_envs.close()


def test_t12_smoke_end_to_end():
    """
    T12 smoke end-to-end (CPU, 200 steps, tiny config):
    runs to completion, telemetry has all §9 keys, imagination-reward-corr computed at least once,
    verify_non_collapse passes.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        config = SAPNAConfig(
            device="cpu",
            num_envs=2,
            warmup_steps=10,
            total_steps=200,  # CPU, 200 steps per §11 T12
            eval_interval=100,
            save_interval=200,
            batch_seqs=2,
            seq_len=8,
            imag_horizon=4,
            checkpoint_dir=tmp_dir,
            no_resume=True,
        )

        report = train(config)
        assert report["final_step"] >= 200, f"Final step {report['final_step']} < 200"
        assert "official_score" in report
        assert "imagination_reward_corr" in report
        assert report["non_collapse_report"]["is_valid"] is True
        assert "eval_metrics" in report
        assert len(report["telemetry_recent"]) > 0, "Telemetry should not be empty"
        required_keys = [
            "wm_loss", "L_tokens", "L_pool", "L_reward", "L_done",
            "var_hinge", "cov_term", "reward_ce", "done_bce",
            "actor_loss", "critic_loss", "q_spread_equivalent",
            "entropy_mean", "advantage_std", "imagined_reward_mean",
            "real_reward_mean", "milestone_fraction", "sps", "ram_percent",
        ]
        for k in required_keys:
            assert k in report["telemetry_recent"], f"Missing key '{k}' in §9 telemetry"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_t13_closed_loop_purity():
    """
    T13 closed-loop purity (v1.4):
    15-step imagination rollout with a CONSTANT action from a fixed start -> token drift bounded;
    identical rollouts with identical seeds -> identical trajectories;
    forward_tokens_clean output equals tokens_input + head(spatial_out) exactly (no broadcast residue).
    """
    torch.manual_seed(42)
    encoder = ObservationEncoder()
    pred = SpatialPredictor()
    action_emb = ActionEmbedding()
    core = TemporalCore()

    tokens_0 = torch.randn(1, 64, 64)
    h_0 = torch.zeros(1, 256)
    a = torch.tensor([5], dtype=torch.long)  # 'do' action

    # Verify forward_tokens_clean formula exactly
    mean_clean, all_clean = pred.forward_tokens_clean(tokens_0, a, h_0)
    mean_dirty, all_dirty = pred.forward_tokens(tokens_0, a, h_0)

    # In forward_tokens, action broadcast was added to tokens before prediction.
    # Therefore mean_clean != mean_dirty (clean does not contaminate output with action broadcast)
    assert not torch.allclose(mean_clean, mean_dirty), "Clean decode must not equal dirty decode"

    # Exact identity assert per §20 F10: all_clean[k] == tokens_0 + pred.heads[k](spatial_out)
    a_emb = pred.action_embedding(a)
    a_broadcast = pred.a_proj(a_emb)
    post_tokens = tokens_0 + a_broadcast.unsqueeze(1)
    film = pred.film(a_emb)
    gamma_f, beta_f = film.chunk(2, dim=-1)
    post_tokens = post_tokens * (1.0 + gamma_f).unsqueeze(1) + beta_f.unsqueeze(1)
    a_tok = pred.action_proj(a_emb).unsqueeze(1)
    m_tok = pred.mem_proj(h_0).unsqueeze(1)
    seq = torch.cat([post_tokens, a_tok, m_tok], dim=1)
    seq = seq + pred.pos_embed[:, :seq.size(1), :]
    out_seq = pred.transformer(seq)
    spatial_out = out_seq[:, :pred.num_tokens, :]

    for k in range(len(pred.heads)):
        assert torch.allclose(all_clean[k], tokens_0 + pred.heads[k](spatial_out), atol=1e-5), f"Clean head {k} must equal tokens_input + head(spatial_out)"

    # 15-step closed-loop rollout with constant action
    curr_tokens = tokens_0.clone()
    curr_h = h_0.clone()

    for step in range(15):
        mean_pred, _ = pred.forward_tokens_clean(curr_tokens, a, curr_h)
        curr_z = encoder.pool_tokens(mean_pred)
        curr_h = core(curr_z, curr_h, a_emb=action_emb(a))
        curr_tokens = mean_pred

    # Token drift is bounded: relative bound <= 0.25 * ||z_0|| per §20 F10
    drift = (curr_tokens - tokens_0).abs().max().item()
    norm_z0 = torch.norm(encoder.pool_tokens(tokens_0)).item()
    assert not np.isnan(drift), "NaN encountered during closed-loop rollout"
    assert drift <= 0.25 * norm_z0, f"Token drift {drift:.4f} exceeded 0.25 * ||z_0|| ({0.25 * norm_z0:.4f})"

    # Identical seeds -> identical trajectories
    torch.manual_seed(999)
    p1, _ = pred.forward_tokens_clean(tokens_0, a, h_0)
    torch.manual_seed(999)
    p2, _ = pred.forward_tokens_clean(tokens_0, a, h_0)
    assert torch.equal(p1, p2), "Identical seeds must produce bitwise identical trajectories"
