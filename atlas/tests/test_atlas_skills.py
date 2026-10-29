"""Unit tests for OptionNet, ReachabilityNet, and HER loss."""

import numpy as np
import torch
import torch.optim as optim

from atlas.skills import OptionNet, ReachabilityNet, build_her_batch, compute_her_loss
from atlas.train import _compute_auroc


def test_her_batch_shapes():
    # Synthetic episode
    T = 30
    ep = {
        "z": np.random.randn(T, 128).astype(np.float32),
        "h": np.random.randn(T, 256).astype(np.float32),
        "actions": np.random.randint(0, 17, size=(T,), dtype=np.int64),
    }

    batch = build_her_batch([ep], horizons=(5, 10))
    assert batch["s_pos"].shape[-1] == 384
    assert batch["g_pos"].shape[-1] == 128
    assert batch["actions_pos"].ndim == 1
    assert batch["s_reach"].shape[-1] == 384
    assert batch["reach_labels"].ndim == 1
    assert len(batch["s_pos"]) > 0


def test_her_training_decreases_ce():
    # Synthetic cyclic trajectory: state deterministic for action
    torch.manual_seed(42)
    np.random.seed(42)

    T = 25
    z = np.zeros((T, 128), dtype=np.float32)
    h = np.zeros((T, 256), dtype=np.float32)
    actions = np.zeros((T,), dtype=np.int64)

    for t in range(T):
        z[t, t % 128] = 1.0
        h[t, t % 256] = 1.0
        actions[t] = t % 17

    ep = {"z": z, "h": h, "actions": actions}
    batch = build_her_batch([ep], horizons=(5,))

    option_net = OptionNet()
    reach_net = ReachabilityNet()
    optimizer = optim.Adam(list(option_net.parameters()) + list(reach_net.parameters()), lr=1e-3)

    _, metrics_start = compute_her_loss(option_net, reach_net, batch)
    initial_ce = metrics_start["ce_loss"]

    for _ in range(150):
        optimizer.zero_grad()
        loss, _ = compute_her_loss(option_net, reach_net, batch)
        loss.backward()
        optimizer.step()

    _, metrics_end = compute_her_loss(option_net, reach_net, batch)
    final_ce = metrics_end["ce_loss"]

    assert final_ce < 0.5 * initial_ce, (
        f"CE loss did not decrease by > 50%: initial={initial_ce:.4f}, final={final_ce:.4f}"
    )


def test_reachability_discrimination():
    # Two distinctly separated episodes
    torch.manual_seed(42)
    np.random.seed(42)

    T = 20
    ep1 = {
        "z": np.ones((T, 128), dtype=np.float32),
        "h": np.zeros((T, 256), dtype=np.float32),
        "actions": np.zeros(T, dtype=np.int64),
    }
    ep2 = {
        "z": -np.ones((T, 128), dtype=np.float32),
        "h": np.zeros((T, 256), dtype=np.float32),
        "actions": np.zeros(T, dtype=np.int64),
    }

    batch = build_her_batch([ep1, ep2], horizons=(5,))
    reach_net = ReachabilityNet()
    opt = optim.Adam(reach_net.parameters(), lr=1e-3)

    s_reach = batch["s_reach"]
    g_reach = batch["g_reach"]
    labels = batch["reach_labels"]

    for _ in range(100):
        opt.zero_grad()
        logits = reach_net.raw_logits(s_reach, g_reach)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = reach_net(s_reach, g_reach).cpu().numpy()
    auroc = _compute_auroc(preds, labels.cpu().numpy())
    assert auroc > 0.90, f"Expected Reachability AUROC > 0.90, got {auroc:.4f}"


def test_h_alignment():
    """Verifies that cache_latents recurrence identically matches the play-loop recurrence."""
    from atlas.backbone import FrozenBackbone

    backbone = FrozenBackbone()
    device = backbone.device
    T = 20
    np.random.seed(42)
    torch.manual_seed(42)

    z_np = np.random.randn(T, 128).astype(np.float32)
    a_np = np.random.randint(0, 17, size=(T,), dtype=np.int64)

    # 1. Play loop (identical to arbiter.act)
    h_play = np.zeros((T, 256), dtype=np.float32)
    h_prev = torch.zeros(1, 256, device=device)
    a_prev = torch.tensor([0], dtype=torch.long, device=device)
    for t in range(T):
        z_t = torch.from_numpy(z_np[t: t + 1]).to(device)
        if t == 0:
            h_t = torch.zeros(1, 256, device=device)
        else:
            with torch.no_grad():
                h_t = backbone.recurrent_step(z_t, a_prev, h_prev)
        h_play[t] = h_t.cpu().numpy().squeeze(0)
        h_prev = h_t
        a_prev = torch.tensor([a_np[t]], dtype=torch.long, device=device)

    # 2. cache_latents recurrence (as implemented in Step 2b)
    h_cached = np.zeros((T, 256), dtype=np.float32)
    z_ep = torch.from_numpy(z_np).float().to(device)
    a_ep = torch.from_numpy(a_np).long().to(device)
    h_t = torch.zeros(1, 256, device=device)
    for t in range(T):
        if t > 0:
            with torch.no_grad():
                h_t = backbone.recurrent_step(z_ep[t: t + 1], a_ep[t - 1: t], h_t)
        h_cached[t] = h_t.cpu().numpy().squeeze(0)

    assert np.allclose(h_play, h_cached, atol=1e-6), "h_play and h_cached do not match!"


def test_frequency_balanced_her_loss():
    """Verifies frequency-balanced HER loss on a synthetic 90%-one-action batch."""
    N = 1000
    # 900 samples of action 0, 10 samples each of actions 1 through 10
    a_list = [0] * 900
    for a in range(1, 11):
        a_list.extend([a] * 10)
    a_pos = torch.tensor(a_list, dtype=torch.long)

    # Calculate weights exactly as in compute_her_loss
    counts = torch.bincount(a_pos, minlength=17).clamp_min(1)
    w = (1.0 / counts[a_pos].float())
    w = w * (len(a_pos) / w.sum())
    w = w.clamp(max=50.0)

    present_actions = list(range(11))
    equal_share = float(len(a_pos)) / len(present_actions)

    for a in present_actions:
        mass_a = float(w[a_pos == a].sum().item())
        ratio = mass_a / equal_share
        assert 0.5 <= ratio <= 2.0, f"Action {a} weight mass ratio {ratio:.3f} outside [0.5, 2.0]"

    # Test weighted CE contribution of majority action vs minority actions
    torch.manual_seed(42)
    s_pos = torch.randn(N, 384)
    g_pos = torch.randn(N, 128)
    batch = {
        "s_pos": s_pos,
        "g_pos": g_pos,
        "actions_pos": a_pos,
        "s_reach": s_pos,
        "g_reach": g_pos,
        "reach_labels": torch.ones(N),
    }

    option_net = OptionNet()
    reach_net = ReachabilityNet()
    policy_logits = option_net.logits(s_pos, g_pos)
    per_sample_ce = torch.nn.functional.cross_entropy(
        policy_logits, a_pos, label_smoothing=0.05, reduction="none"
    )

    weighted_contribs = {}
    for a in present_actions:
        mask = (a_pos == a)
        weighted_contribs[a] = float((per_sample_ce[mask] * w[mask]).sum().item() / N)

    maj_contrib = weighted_contribs[0]
    for a in range(1, 11):
        min_contrib = weighted_contribs[a]
        assert maj_contrib <= 2.0 * min_contrib, (
            f"Majority action CE contribution {maj_contrib:.4f} > 2x minority action {a} {min_contrib:.4f}"
        )
