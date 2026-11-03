"""Tests for EMA target encoder update and stop-gradient properties."""
import torch
from src.models.visual_encoder import VisualEncoder
from src.models.ema_encoder import EmaTargetEncoder


def test_ema_target_update():
    online = VisualEncoder(latent_dim=128)
    target = EmaTargetEncoder(online)

    # All target parameters must not require grad
    for p in target.encoder.parameters():
        assert not p.requires_grad

    # Pick a specific parameter from online and change it
    online_p = next(online.parameters())
    target_p = next(target.encoder.parameters())

    init_val = target_p.data.clone()
    new_online_val = init_val + 10.0
    online_p.data.copy_(new_online_val)

    # Perform EMA update with momentum 0.99
    momentum = 0.99
    target.update(online, momentum=momentum)

    expected = momentum * init_val + (1.0 - momentum) * new_online_val
    assert torch.allclose(target_p.data, expected, atol=1e-5)


def test_target_encoder_stop_gradient():
    online = VisualEncoder(latent_dim=128)
    target = EmaTargetEncoder(online)

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    tokens, pooled = target(x)

    assert not tokens.requires_grad
    assert not pooled.requires_grad
