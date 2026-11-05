"""Unit tests for the Zephyr world model, heads, and imagination rollout."""
from __future__ import annotations

import torch

from zephyr_test_utils import tiny_cfg
from zephyr.ac import imagine_trajectory
from zephyr.heads import ZephyrHeads
from zephyr.train_wm import _posterior_states, _prior_rollout
from zephyr.wm import ZephyrWM

CFG = tiny_cfg()


def _random_obs(b, size=64):
    return torch.randint(0, 255, (b, size, size, 3), dtype=torch.uint8)


def test_encoder_and_belief_shapes():
    wm = ZephyrWM(CFG)
    z = wm.encode(_random_obs(2))
    assert z.shape == (2, CFG.world_model.latent_dim)
    h = wm.init_state(2, z.device)
    h2 = wm.belief_step(z, torch.tensor([1, 2]), h)
    assert h2.shape == h.shape
    zh = wm.predict(z, torch.tensor([1, 2]), h)
    assert zh.shape == z.shape


def test_ema_encoder_updates_and_is_frozen():
    wm = ZephyrWM(CFG)
    obs = _random_obs(2)
    before = wm.ema_encode(obs).clone()
    assert not any(p.requires_grad for p in wm.ema_encoder.parameters())
    wm.update_ema(step=0)
    after = wm.ema_encode(obs)
    assert not torch.equal(before, after)  # polyak moved the target
    assert before.shape == after.shape


def test_posterior_and_prior_rollout_shapes():
    wm = ZephyrWM(CFG)
    b, t = 3, 6
    z_now = torch.randn(b, t, CFG.world_model.latent_dim)
    z_next = torch.randn(b, t, CFG.world_model.latent_dim)
    actions = torch.randint(0, 17, (b, t))
    h_pre = _posterior_states(wm, z_now, z_next, actions)
    assert len(h_pre) == t
    zs, hs = _prior_rollout(wm, z_now[:, 0], actions, grad=False)
    assert len(zs) == t and zs[0].shape == (b, CFG.world_model.latent_dim)
    assert len(hs) == t and hs[0].shape == (b, CFG.world_model.belief_dim)


def test_checkpoint_roundtrip_is_exact():
    wm = ZephyrWM(CFG)
    heads = ZephyrHeads(CFG)
    wm.eval(); heads.eval()
    obs = _random_obs(2)
    z = wm.encode(obs)
    h = wm.init_state(2, z.device)
    s = ZephyrHeads.state_input(z, h)

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        wm_path = os.path.join(d, "wm.pt")
        heads_path = os.path.join(d, "heads.pt")
        wm.save_checkpoint(wm_path, step=7, extra={"note": "x"})
        heads.save_checkpoint(heads_path, step=7)
        wm2, meta = ZephyrWM.load_checkpoint(wm_path)
        heads2, meta_h = ZephyrHeads.load_checkpoint(heads_path)
        assert meta["step"] == 7 and meta["extra"]["note"] == "x"
        assert meta_h["step"] == 7
        with torch.no_grad():
            z2 = wm2.encode(obs)
            h2 = wm2.init_state(2, z2.device)
            s2 = ZephyrHeads.state_input(z2, h2)
            assert torch.equal(z, z2)
            assert torch.equal(heads.actor(s), heads2.actor(s2))
            # belief step must also reproduce exactly after reload
            s_step = ZephyrHeads.state_input(z, wm.belief_step(z, torch.tensor([1, 1]), h))
            s2_step = ZephyrHeads.state_input(
                z2, wm2.belief_step(z2, torch.tensor([1, 1]), h2)
            )
            assert torch.equal(heads.actor(s_step), heads2.actor(s2_step))


def test_imagination_trajectory_shapes_and_flow():
    torch.manual_seed(0)
    wm = ZephyrWM(CFG)
    heads = ZephyrHeads(CFG)
    wm.eval(); heads.eval()
    b = 2
    z0 = torch.randn(b, CFG.world_model.latent_dim)
    h0 = wm.init_state(b, z0.device)
    out = imagine_trajectory(wm, heads, z0, h0, horizon=4, cfg=CFG)
    assert out["z"].shape == (4, b, CFG.world_model.latent_dim)
    assert out["h"].shape == (4, b, CFG.world_model.belief_dim)
    assert out["actions"].shape == (4, b)
    assert out["value_logits"].shape == (5, b, CFG.heads.critic_bins)
    assert out["reward_logits"].shape == (4, b, CFG.heads.reward_bins)
    assert (out["intrinsic"] >= 0).all()


def test_deterministic_imagination_is_reproducible():
    torch.manual_seed(0)
    wm = ZephyrWM(CFG)
    heads = ZephyrHeads(CFG)
    wm.eval(); heads.eval()
    z0 = torch.randn(1, CFG.world_model.latent_dim)
    h0 = wm.init_state(1, z0.device)
    a = imagine_trajectory(wm, heads, z0, h0, horizon=3, cfg=CFG, deterministic=True)
    b = imagine_trajectory(wm, heads, z0, h0, horizon=3, cfg=CFG, deterministic=True)
    assert torch.equal(a["actions"], b["actions"])
    assert torch.equal(a["z"], b["z"])
