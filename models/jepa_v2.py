"""
JEPA v2 Architecture for Learn-a-World.
Wraps the core JEPAWorldModel (frozen encoder and dynamics predictor)
and attaches task heads:
- EventHead: multi-label achievement prediction
- HealthHead: survival vital prediction
Exposes unified reward prediction for the v2 reward-value planner:
predict_reward(z, a, mask) = sum_i(p_i * (1 - m_i)) + h_hat
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.heads import EventHead, GRUBeliefModule, HealthHead, PolicyPrior, QEnsemble
from models.jepa import JEPAWorldModel

class JEPAv2(nn.Module):
    """
    JEPA v2 Task World Model.
    Combines self-supervised latent dynamics with calibrated achievement,
    survival, Q-value ensemble, policy prior, and recurrent GRU belief heads.
    """

    def __init__(
        self,
        base_model: JEPAWorldModel,
        hidden_dim: int = 256,
        action_emb_dim: int = 32,
        q_heads: int = 3,
        q_bins: int = 51,
        q_max: float = 12.0,
        use_gru: bool = True,
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

        self.q_ensemble = QEnsemble(
            latent_dim=self.latent_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            num_heads=q_heads,
            num_bins=q_bins,
            v_min=0.0,
            v_max=q_max,
            hidden_dim=hidden_dim,
        )

        self.q_target_ensemble = copy.deepcopy(self.q_ensemble)
        for p in self.q_target_ensemble.parameters():
            p.requires_grad = False

        self.policy_prior = PolicyPrior(
            latent_dim=self.latent_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )

        self.use_gru = use_gru
        self.gru_belief = GRUBeliefModule(
            latent_dim=self.latent_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=128,
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

    def update_target_q(self, decay: float = 0.995) -> None:
        """EMA momentum update of target Q ensemble network: xi <- tau*xi + (1-tau)*theta."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.q_target_ensemble.parameters(), self.q_ensemble.parameters()
            ):
                target_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)

    def q_mean(self, z: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Mean expected return across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_mean(z, action)

    def q_min(self, z: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Clipped Double-Q minimum return across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_min(z, action)

    def q_std(self, z: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Epistemic uncertainty (std dev) across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_std(z, action)

    def sample_prior_action(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Samples categorical action indices guided by PolicyPrior."""
        return self.policy_prior.sample(z, mask, temperature=temperature)

    @classmethod
    def from_v1_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        freeze_base: bool = True,
    ) -> "JEPAv2":
        """Factory method: loads a v1 OR v2 checkpoint and returns a JEPAv2 model.

        Auto-detects format:
          - v1 checkpoint: contains key ``model_state``  → builds JEPAv2 from scratch and
            copies the base encoder weights, leaving task heads freshly initialized.
          - v2 checkpoint: contains key ``base_model_state`` → builds a fresh JEPAv2 then
            calls ``load_checkpoint()`` to restore all saved weights.
        """
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
        q_heads = h_cfg.get("q_ensemble", 3)
        q_bins = h_cfg.get("q_bins", 51)
        q_max = h_cfg.get("q_max", 12.0)

        # ------------------------------------------------------------------
        # V2 CHECKPOINT fast-path: all weights already in v2 format → just
        # build a fresh shell and call load_checkpoint to restore everything.
        # ------------------------------------------------------------------
        if "base_model_state" in ckpt:
            print(f"[from_v1_checkpoint] Detected v2 checkpoint at {ckpt_path}; loading all weights.")
            # Peek into base_model_state to recover architecture dims
            base_state = ckpt["base_model_state"]
            if "predictor.net.0.weight" in base_state:
                predictor_hidden_dim = base_state["predictor.net.0.weight"].shape[0]
            if "predictor.action_emb.weight" in base_state:
                action_emb_dim = base_state["predictor.action_emb.weight"].shape[1]
                num_actions = base_state["predictor.action_emb.weight"].shape[0]
            if "encoder.fc.weight" in base_state:
                latent_dim = base_state["encoder.fc.weight"].shape[0]

            base_model = JEPAWorldModel(
                in_channels=3,
                latent_dim=latent_dim,
                num_actions=num_actions,
                action_emb_dim=action_emb_dim,
                predictor_hidden_dim=predictor_hidden_dim,
                predictor_layers=predictor_layers,
                ema_decay=ema_decay,
            )
            v2_model = cls(
                base_model=base_model,
                hidden_dim=hidden_dim,
                action_emb_dim=action_emb_dim,
                q_heads=q_heads,
                q_bins=q_bins,
                q_max=q_max,
                freeze_base=False,  # will be frozen below if requested
            ).to(dev)
            v2_model.load_checkpoint(str(ckpt_path), device=device)
            if freeze_base:
                for p in v2_model.base_model.parameters():
                    p.requires_grad = False
            return v2_model

        # ------------------------------------------------------------------
        # V1 CHECKPOINT: key is ``model_state`` (old format)
        # ------------------------------------------------------------------
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
            q_heads=q_heads,
            q_bins=q_bins,
            q_max=q_max,
            freeze_base=freeze_base,
        ).to(dev)

        return v2_model

    def update_base_target(self, decay: Optional[float] = None) -> None:
        """Momentum update of JEPA base target encoder."""
        self.base_model.update_target_encoder(decay=decay)

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
            "q_ensemble_state": self.q_ensemble.state_dict(),
            "q_target_ensemble_state": self.q_target_ensemble.state_dict(),
            "policy_prior_state": self.policy_prior.state_dict(),
            "gru_belief_state": self.gru_belief.state_dict(),
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
        if "q_ensemble_state" in ckpt:
            self.q_ensemble.load_state_dict(ckpt["q_ensemble_state"])
        if "q_target_ensemble_state" in ckpt:
            self.q_target_ensemble.load_state_dict(ckpt["q_target_ensemble_state"])
        if "policy_prior_state" in ckpt:
            self.policy_prior.load_state_dict(ckpt["policy_prior_state"])
        if "gru_belief_state" in ckpt:
            self.gru_belief.load_state_dict(ckpt["gru_belief_state"])
        self.to(dev)
        return ckpt
