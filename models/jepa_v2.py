"""
JEPA v2 Architecture for Learn-a-World.
Wraps the core JEPAWorldModel (frozen encoder and dynamics predictor)
and attaches task heads:
- EventHead: multi-label achievement prediction
- HealthHead: survival vital prediction
Exposes unified reward prediction for the v2 reward-value planner:
predict_reward(z, a, mask) = sum_i(p_i * (1 - m_i)) + h_hat
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.heads import EventHead, HealthHead
from models.jepa import JEPAWorldModel

class JEPAv2(nn.Module):
    """
    JEPA v2 Task World Model.
    Combines self-supervised latent dynamics with calibrated achievement
    and survival prediction heads.
    """

    def __init__(
        self,
        base_model: JEPAWorldModel,
        hidden_dim: int = 256,
        action_emb_dim: int = 32,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.latent_dim = base_model.latent_dim
        num_actions = base_model.predictor.num_actions

        self.event_head = EventHead(
            latent_dim=self.latent_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.health_head = HealthHead(
            latent_dim=self.latent_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
        )

        if freeze_base:
            self.freeze_base_model()

    def freeze_base_model(self) -> None:
        """Freezes encoder and predictor parameters (Phase 1 & 2 diagnostic & planning)."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self) -> None:
        """Unfreezes encoder and predictor parameters for joint fine-tuning (Phase 3)."""
        for param in self.base_model.encoder.parameters():
            param.requires_grad = True
        for param in self.base_model.predictor.parameters():
            param.requires_grad = True
        # Target encoder remains non-gradient, updated via EMA
        for param in self.base_model.target_encoder.parameters():
            param.requires_grad = False

    def encode(self, obs: torch.Tensor, use_target: bool = True) -> torch.Tensor:
        """
        Encodes pixel observation into latent representation z.
        Default use_target=True provides EMA target features for planner consistency.
        """
        return self.base_model.encode(obs, use_target=use_target)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Rolls forward latent dynamics via predictor."""
        return self.base_model.predictor(z, action)

    def predict_reward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        mask_before: torch.Tensor,
        return_decomposition: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Computes composite reward expectation:
        r_hat = sum_i(p_i * (1 - m_i)) + h_hat
        z: [..., latent_dim]
        action: [...] int64
        mask_before: [..., 22] float32
        """
        p = self.event_head.predict_proba(z, action, mask_before)
        h = self.health_head(z, action)

        # Only locked achievements (1 - mask) yield achievement reward
        unlocked_mask = mask_before.float()
        locked_weight = 1.0 - unlocked_mask
        achievement_reward = (p * locked_weight).sum(dim=-1)

        total_reward = achievement_reward + h

        if return_decomposition:
            return total_reward, {
                "event_probs": p,
                "achievement_reward": achievement_reward,
                "health_reward": h,
            }
        return total_reward

    @classmethod
    def from_v1_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        freeze_base: bool = True,
    ) -> "JEPAv2":
        """Factory method: loads trained v1 checkpoint and initializes v2 task heads."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        dev = torch.device(device) if device is not None else torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=dev)

        cfg = config or {}
        m_cfg = cfg.get("model", {})
        h_cfg = cfg.get("heads", {})
        latent_dim = m_cfg.get("latent_dim", 128)
        num_actions = m_cfg.get("num_actions", 17)
        action_emb_dim = m_cfg.get("action_emb_dim", 32)
        predictor_hidden_dim = m_cfg.get("predictor_hidden_dim", 256)
        predictor_layers = m_cfg.get("predictor_layers", 2)
        ema_decay = m_cfg.get("ema_decay", 0.99)
        hidden_dim = h_cfg.get("hidden_dim", 256)

        # Auto-detect architectural dimensions if present in checkpoint state_dict
        model_state = ckpt.get("model_state", {})
        if "predictor.net.0.weight" in model_state:
            predictor_hidden_dim = model_state["predictor.net.0.weight"].shape[0]
        if "predictor.action_emb.weight" in model_state:
            action_emb_dim = model_state["predictor.action_emb.weight"].shape[1]
            num_actions = model_state["predictor.action_emb.weight"].shape[0]
        if "encoder.fc.weight" in model_state:
            latent_dim = model_state["encoder.fc.weight"].shape[0]

        base_model = JEPAWorldModel(
            in_channels=3,
            latent_dim=latent_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_layers=predictor_layers,
            ema_decay=ema_decay,
        )
        base_model.load_state_dict(ckpt["model_state"])
        base_model.to(dev)

        v2_model = cls(
            base_model=base_model,
            hidden_dim=hidden_dim,
            action_emb_dim=action_emb_dim,
            freeze_base=freeze_base,
        ).to(dev)

        return v2_model

    def save_checkpoint(
        self,
        path: Union[str, Path],
        step: int = 0,
        optimizer_state: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves unified v2 model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ckpt_data = {
            "step": step,
            "base_model_state": self.base_model.state_dict(),
            "event_head_state": self.event_head.state_dict(),
            "health_head_state": self.health_head.state_dict(),
        }
        if optimizer_state is not None:
            ckpt_data["optimizer_state"] = optimizer_state
        if extra_metadata is not None:
            ckpt_data["metadata"] = extra_metadata
        torch.save(ckpt_data, p)
        return p

    def load_checkpoint(
        self,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Loads complete v2 model checkpoint."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        ckpt = torch.load(path, map_location=dev)
        self.base_model.load_state_dict(ckpt["base_model_state"])
        self.event_head.load_state_dict(ckpt["event_head_state"])
        self.health_head.load_state_dict(ckpt["health_head_state"])
        self.to(dev)
        return ckpt
