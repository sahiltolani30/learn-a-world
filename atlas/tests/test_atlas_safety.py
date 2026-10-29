"""Unit tests for ContinueNet and focal BCE."""

import numpy as np
import torch
import torch.optim as optim

from atlas.safety import ContinueNet, focal_bce


def test_safety_risk_query_shapes():
    net = ContinueNet()
    s = torch.randn(4, 384)
    a = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    risk = net.risk(s, a)
    assert risk.shape == (4,)
    assert torch.all(risk >= 0.0) and torch.all(risk <= 1.0)

    alive = net.alive_prob(s, a)
    assert alive.shape == (4,)
    assert torch.allclose(risk + alive, torch.ones(4), atol=1e-5)

    all_risks = net.risk_all_actions(s)
    assert all_risks.shape == (4, 17)


def test_focal_bce_rare_deaths_recall():
    torch.manual_seed(42)
    np.random.seed(42)

    net = ContinueNet()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)

    # 5% rare deaths (10 deaths, 190 survivals)
    N = 200
    s = torch.randn(N, 384)
    a = torch.randint(0, 17, (N,))
    targets = torch.zeros(N, dtype=torch.float32)

    death_indices = np.random.choice(N, size=10, replace=False)
    targets[death_indices] = 1.0
    # Make death states distinct in feature space
    s[death_indices] += 2.0

    for _ in range(120):
        optimizer.zero_grad()
        logits = net.risk_logits(s, a)
        loss = focal_bce(logits, targets, weight=25.0, gamma=2.0)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        risks = net.risk(s, a).cpu().numpy()

    # Recall on rare death class with 0.5 threshold
    death_preds = risks[death_indices] > 0.5
    recall = np.mean(death_preds)
    assert recall >= 0.50, f"Expected death recall >= 0.50, got {recall:.2f}"
