"""Unit tests for EpistemicNet and kNN novelty."""

from collections import deque
import numpy as np
import torch
import torch.optim as optim

from atlas.epistemic import EpistemicNet, knn_novelty


def test_epistemic_training_and_disagreement():
    torch.manual_seed(42)
    np.random.seed(42)

    net = EpistemicNet()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    B = 32
    s_train = torch.randn(B, 384)
    a_train = torch.randint(0, 17, (B,))
    target_res = torch.randn(B, 128)

    # Initial loss
    initial_loss = net.compute_loss(s_train, a_train, target_res).item()

    # Train to fit target residuals
    for _ in range(100):
        optimizer.zero_grad()
        loss = net.compute_loss(s_train, a_train, target_res)
        loss.backward()
        optimizer.step()

    final_loss = net.compute_loss(s_train, a_train, target_res).item()
    assert final_loss < 0.3 * initial_loss, (
        f"EpistemicNet loss failed to train down: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    # Disagreement on train distribution vs unseen random pairs
    net.eval()
    with torch.no_grad():
        d_train = net.disagreement(s_train, a_train).mean().item()

        s_unseen = torch.randn(B, 384) * 5.0  # Out of distribution
        a_unseen = torch.randint(0, 17, (B,))
        d_unseen = net.disagreement(s_unseen, a_unseen).mean().item()

    assert d_unseen > d_train, (
        f"Expected unseen disagreement ({d_unseen:.4f}) > train disagreement ({d_train:.4f})"
    )


def test_knn_novelty():
    mem = deque(maxlen=10)
    h_zero = np.zeros(256, dtype=np.float32)

    # Empty memory -> 0.0
    assert knn_novelty(h_zero, mem, k=4) == 0.0

    # Add near-identical states
    for _ in range(5):
        mem.append(np.zeros(256, dtype=np.float32))

    near_nov = knn_novelty(h_zero, mem, k=4)
    assert near_nov < 1e-4

    # Distant query state
    h_far = np.ones(256, dtype=np.float32) * 10.0
    far_nov = knn_novelty(h_far, mem, k=4)
    assert far_nov > 10.0
