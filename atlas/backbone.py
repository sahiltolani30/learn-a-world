"""Frozen T2 JEPA backbone wrapper.

Loads the already-trained JEPAv3 world model (encoder, GRU belief core, latent predictor,
event head, health head, Q ensemble, behavior-cloned policy prior) and freezes it. Every
new ATLAS head consumes this stable latent geometry; the backbone itself is never trained
(Autopsy Attempt 7: re-encoding destroyed all downstream heads via latent drift).

Old code is imported READ-ONLY. Nothing outside atlas/ is modified.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from atlas.config import (
    LATENT_DIM,
    NUM_ACHIEVEMENTS,
    NUM_ACTIONS,
    RECURRENT_DIM,
    STATE_DIM,
    AtlasConfig,
    resolve_device,
)


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenBackbone:
    """Inference-only facade over the trained JEPAv3 world model."""

    def __init__(self, config: Optional[AtlasConfig] = None, device: Optional[str] = None):
        from atlas.config import AtlasConfig as _AC

        self.config = config or _AC()
        self.device = torch.device(device or resolve_device(self.config.backbone.device))
        self.checkpoint_path = str(self.config.backbone.checkpoint)
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"ATLAS backbone checkpoint not found: {self.checkpoint_path}. "
                "Copy jepa_crafter_v3_state_readout_v2.pt from Drive into checkpoints/ "
                "or point ATLAS_CHECKPOINT at an existing file."
            )

        from dual_store.jepa_v3 import JEPAv3  # read-only import of the trained model class

        self.model = JEPAv3.from_checkpoint(self.checkpoint_path, device=str(self.device))
        self.model.eval()
        if self.config.backbone.frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.provenance: Dict[str, object] = {
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": sha256_of(self.checkpoint_path),
            "loaded_component_keys": sorted(getattr(self.model, "loaded_component_keys", [])),
            "frozen": bool(self.config.backbone.frozen),
            "step": int(torch.load(self.checkpoint_path, map_location="cpu", weights_only=False).get("step", -1)),
            "loaded_at_unix": time.time(),
        }

    # ------------------------------------------------------------------ input
    @staticmethod
    def to_tensor(obs: np.ndarray) -> torch.Tensor:
        """uint8 HWC [64,64,3] (or batched [B,64,64,3]) -> float CHW [B,3,64,64] in [0,1]."""
        if obs.ndim == 3:
            obs = obs[None]
        t = torch.from_numpy(np.ascontiguousarray(obs)).float().div_(255.0)
        return t.permute(0, 3, 1, 2)

    # ------------------------------------------------------------- encoders
    @torch.no_grad()
    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """obs uint8 [...,64,64,3] -> z [...,128] on self.device."""
        x = self.to_tensor(obs).to(self.device)
        return self.model.encoder(x)

    @torch.no_grad()
    def recurrent_step(self, z: torch.Tensor, action: torch.Tensor, h_prev: torch.Tensor):
        """Advance the GRU belief core: (z[B,128], a[B], h_prev[B,256]) -> h_t[B,256]."""
        return self.model.recurrent_core(z, action, h_prev)

    def composite(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, h], dim=-1)

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, h: torch.Tensor, action: torch.Tensor):
        """Imagination step in latent space: -> (next_z, next_h, next_s)."""
        return self.model.predict_next_step(z, h, action)

    # ---------------------------------------------------------------- heads
    @torch.no_grad()
    def event_probs(self, s: torch.Tensor, action: torch.Tensor, mask_before: torch.Tensor) -> torch.Tensor:
        """Calibrated P(unlock i | s, a, mask) -> [B,22]. mask = agent's own belief."""
        return self.model.event_head.predict_proba(s, action, mask_before, use_calibration=True)

    @torch.no_grad()
    def event_logits_all_actions(self, s: torch.Tensor, mask_before: torch.Tensor) -> torch.Tensor:
        """Event probabilities for every action: [B, 17, 22] (PROBE-mode ranking)."""
        a = torch.arange(NUM_ACTIONS, device=s.device).expand(s.shape[0], NUM_ACTIONS)
        m = mask_before.unsqueeze(1).expand(-1, NUM_ACTIONS, -1)
        s_exp = s.unsqueeze(1).expand(-1, NUM_ACTIONS, -1)
        flat_s = s_exp.reshape(-1, STATE_DIM)
        flat_a = a.reshape(-1)
        flat_m = m.reshape(-1, NUM_ACHIEVEMENTS)
        probs = self.model.event_head.predict_proba(flat_s, flat_a, flat_m, use_calibration=True)
        return probs.reshape(s.shape[0], NUM_ACTIONS, NUM_ACHIEVEMENTS)

    @torch.no_grad()
    def health(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicted non-achievement reward delta (vitals effect) -> [B]."""
        return self.model.health_head(s, action)

    @torch.no_grad()
    def q_mean(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model.q_mean(s, action)

    @torch.no_grad()
    def prior_logits(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Behavior-cloned policy prior logits [B,17] over composite state s."""
        return self.model.policy_prior(s, mask)

    # ------------------------------------------------------------- module IO
    def state_dict_of_frozen(self) -> Dict[str, object]:
        return {"path": self.checkpoint_path, "sha256": self.provenance["checkpoint_sha256"]}

    def assert_frozen(self) -> None:
        for name, p in self.model.named_parameters():
            assert not p.requires_grad, f"backbone param {name} is trainable"
            assert p.grad is None, f"backbone param {name} accumulated a gradient"
