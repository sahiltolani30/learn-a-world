"""Tests for Project ATLAS v2.2 F4 (Progress-weighted HER) and F5 (Skill-epoch boost)."""

import copy
import time
import numpy as np
import pytest
import torch
import torch.optim as optim

from atlas.config import SkillsConfig, TrainConfig
from atlas.skills import OptionNet, ReachabilityNet, build_her_batch, compute_her_loss


def test_f4_approach_segment_weights_greater():
    """Verification F4: synthetic episode where the agent walks straight toward the goal for one segment and away for another -> approach segment's weight strictly greater."""
    T = 12

    # Approach walk: moves from z=[0, 1, 0...] towards g=[1, 0, 0...]
    z_approach = np.zeros((T, 128), dtype=np.float32)
    z_approach[:, 1] = np.linspace(1.0, 0.0, T)
    z_approach[:, 0] = np.linspace(0.0, 1.0, T)

    # Stagnant / non-approach walk: stays at g=[1, 0, 0...] entire segment (progress = 0)
    z_stagnant = np.zeros((T, 128), dtype=np.float32)
    z_stagnant[:, 0] = 1.0

    ep_approach = {
        "z": z_approach,
        "h": np.zeros((T, 256), dtype=np.float32),
        "actions": np.zeros(T, dtype=np.int64),
        "rewards": np.zeros(T, dtype=np.float32),
        "events": np.zeros((T, 22), dtype=bool),
    }

    ep_stagnant = {
        "z": z_stagnant,
        "h": np.zeros((T, 256), dtype=np.float32),
        "actions": np.zeros(T, dtype=np.int64),
        "rewards": np.zeros(T, dtype=np.float32),
        "events": np.zeros((T, 22), dtype=bool),
    }

    batch_app = build_her_batch([ep_approach], horizons=(5,), progress_weight_gain=3.0)
    batch_stag = build_her_batch([ep_stagnant], horizons=(5,), progress_weight_gain=3.0)

    w_app = float(batch_app["weights_pos"].cpu().numpy().mean())
    w_stag = float(batch_stag["weights_pos"].cpu().numpy().mean())

    assert w_app > w_stag, f"Expected approach weight ({w_app:.3f}) > stagnant/non-approach weight ({w_stag:.3f})"


def test_f4_approach_curriculum_convergence():
    """Verification F4: CE on a curriculum of approach-only synthetic walks drops below 1.0 within 200 steps while uniform-noise walks do not."""
    torch.manual_seed(42)
    np.random.seed(42)

    def _make_app_eps(n_eps: int = 15):
        eps = []
        for i in range(n_eps):
            T = 25
            target_action = i % 4
            z = np.zeros((T, 128), dtype=np.float32)
            g_vec = np.zeros(128, dtype=np.float32)
            g_vec[target_action] = 2.0
            for t in range(T):
                z[t] = g_vec * (t / float(T)) + np.random.randn(128).astype(np.float32) * 0.05
            actions = np.full(T, target_action, dtype=np.int64)
            eps.append({
                "z": z,
                "h": np.zeros((T, 256), dtype=np.float32),
                "actions": actions,
                "rewards": np.zeros(T, dtype=np.float32),
                "events": np.zeros((T, 22), dtype=bool),
            })
        return eps

    def _make_noise_eps(n_eps: int = 15):
        eps = []
        for _ in range(n_eps):
            T = 25
            eps.append({
                "z": np.random.randn(T, 128).astype(np.float32),
                "h": np.zeros((T, 256), dtype=np.float32),
                "actions": np.random.randint(0, 17, size=T, dtype=np.int64),
                "rewards": np.zeros(T, dtype=np.float32),
                "events": np.zeros((T, 22), dtype=bool),
            })
        return eps

    # Train and evaluate on approach walks
    train_app = _make_app_eps(15)
    val_app = _make_app_eps(5)
    b_train_app = build_her_batch(train_app)
    b_val_app = build_her_batch(val_app)

    net_app = OptionNet()
    reach_app = ReachabilityNet()
    opt_app = optim.Adam(list(net_app.parameters()) + list(reach_app.parameters()), lr=3e-3)

    for _ in range(150):
        opt_app.zero_grad()
        loss, _ = compute_her_loss(net_app, reach_app, b_train_app)
        loss.backward()
        opt_app.step()

    with torch.no_grad():
        _, val_metrics_app = compute_her_loss(net_app, reach_app, b_val_app)
    ce_app = val_metrics_app["ce_loss"]

    # Train and evaluate on uniform-noise walks
    train_noise = _make_noise_eps(15)
    val_noise = _make_noise_eps(5)
    b_train_noise = build_her_batch(train_noise)
    b_val_noise = build_her_batch(val_noise)

    net_noise = OptionNet()
    reach_noise = ReachabilityNet()
    opt_noise = optim.Adam(list(net_noise.parameters()) + list(reach_noise.parameters()), lr=3e-3)

    for _ in range(150):
        opt_noise.zero_grad()
        loss_n, _ = compute_her_loss(net_noise, reach_noise, b_train_noise)
        loss_n.backward()
        opt_noise.step()

    with torch.no_grad():
        _, val_metrics_noise = compute_her_loss(net_noise, reach_noise, b_val_noise)
    ce_noise = val_metrics_noise["ce_loss"]

    assert ce_app < 1.0, f"Expected approach CE < 1.0, got {ce_app:.3f}"
    assert ce_noise > 2.0, f"Expected noise CE > 2.0, got {ce_noise:.3f}"


def test_f5_skill_epoch_boost_reaches_lower_ce():
    """Verification F5: one iteration on a fixed synthetic batch reaches lower CE with 16 epochs than 6; total wall time stays low."""
    torch.manual_seed(42)
    np.random.seed(42)

    eps = []
    for _ in range(10):
        T = 20
        eps.append({
            "z": np.random.randn(T, 128).astype(np.float32),
            "h": np.random.randn(T, 256).astype(np.float32),
            "actions": np.random.randint(0, 17, size=T, dtype=np.int64),
            "rewards": np.zeros(T, dtype=np.float32),
            "events": np.zeros((T, 22), dtype=bool),
        })

    batch = build_her_batch(eps)

    net = OptionNet()
    reach = ReachabilityNet()
    init_net_weights = copy.deepcopy(net.state_dict())
    init_reach_weights = copy.deepcopy(reach.state_dict())

    # 1. Train 6 epochs
    opt6 = optim.Adam(list(net.parameters()) + list(reach.parameters()), lr=1e-3)
    for _ in range(6):
        opt6.zero_grad()
        loss, m6 = compute_her_loss(net, reach, batch)
        loss.backward()
        opt6.step()
    ce_6 = m6["ce_loss"]

    # 2. Train 16 epochs from identical start
    net.load_state_dict(init_net_weights)
    reach.load_state_dict(init_reach_weights)
    opt16 = optim.Adam(list(net.parameters()) + list(reach.parameters()), lr=1e-3)
    t0 = time.time()
    for _ in range(16):
        opt16.zero_grad()
        loss, m16 = compute_her_loss(net, reach, batch)
        loss.backward()
        opt16.step()
    ce_16 = m16["ce_loss"]
    dt_16 = time.time() - t0

    assert ce_16 < ce_6, f"Expected CE with 16 epochs ({ce_16:.4f}) < 6 epochs ({ce_6:.4f})"
    assert dt_16 < 60.0, f"16 epochs took {dt_16:.2f}s, expected < 60s"
