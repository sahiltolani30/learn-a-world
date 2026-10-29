"""Unit tests for FrozenBackbone."""

from pathlib import Path
import numpy as np
import torch

from atlas.backbone import FrozenBackbone, sha256_of
from atlas.config import AtlasConfig


def test_backbone_load_and_freeze():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")

    # 1. Assert frozen
    fb.assert_frozen()

    # 2. Check provenance
    assert Path(fb.checkpoint_path).exists()
    assert fb.provenance["frozen"] is True
    assert fb.provenance["checkpoint_sha256"] == sha256_of(fb.checkpoint_path)
    assert fb.provenance["step"] >= 0


def test_backbone_encode_determinism():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")

    rng = np.random.RandomState(42)
    obs = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    z1 = fb.encode(obs)
    z2 = fb.encode(obs)
    assert torch.equal(z1, z2), "Backbone encode is not bitwise deterministic!"
    assert z1.shape == (1, 128)


def test_backbone_methods_shapes():
    cfg = AtlasConfig()
    fb = FrozenBackbone(config=cfg, device="cpu")

    z = torch.randn(2, 128)
    h = torch.randn(2, 256)
    a = torch.tensor([1, 2], dtype=torch.int64)
    m = torch.zeros(2, 22)
    s = fb.composite(z, h)
    assert s.shape == (2, 384)

    # Recurrent step
    h_next = fb.recurrent_step(z, a, h)
    assert h_next.shape == (2, 256)

    # Predict next
    next_z, next_h, next_s = fb.predict_next(z, h, a)
    assert next_z.shape == (2, 128)
    assert next_h.shape == (2, 256)
    assert next_s.shape == (2, 384)

    # Event probs
    probs = fb.event_probs(s, a, m)
    assert probs.shape == (2, 22)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    # Event logits all actions
    all_probs = fb.event_logits_all_actions(s, m)
    assert all_probs.shape == (2, 17, 22)

    # Health & Q mean
    vital_r = fb.health(s, a)
    assert vital_r.shape == (2,)

    q = fb.q_mean(s, a)
    assert q.shape == (2,)

    # Prior logits
    logits = fb.prior_logits(s, m)
    assert logits.shape == (2, 17)
