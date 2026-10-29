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
