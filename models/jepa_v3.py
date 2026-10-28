"""
JEPA v3 Architecture with Recurrent Memory and Task-Shaped Representation (R2 + R5).

Key Architectural Innovations:
1. RecurrentCore: A stateful GRU cell processing (z_t, a_{t-1}, h_{t-1}) -> h_t.
   Provides persistent belief memory across partial observability, dark night cycles, and occlusions.
2. Composite State: s_t = [z_t, h_t]. All task heads (EventHead, HealthHead, QEnsemble, PolicyPrior)
   evaluate on composite state s_t rather than a single memoryless frame.
3. RecurrentPredictor: Rolls forward both visual latent z_{t+1} and recurrent hidden state h_{t+1}
   in imagination during planning.
4. End-to-End Task Shaping: Gradient from task heads and Q flows into the ConvEncoder from step 1,
   preventing representation collapse on task-irrelevant background features.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ConvEncoder
from models.heads import (
    EventHead,
    HealthHead,
    PolicyPrior,
    QEnsemble,
    ReturnNormalizer,
    TimeToEventHead,
    StateDeltaHead,
    StateReadoutHead,
    TransitionHead,
    NUM_ACHIEVEMENTS,
)


class RecurrentCore(nn.Module):
    """
    GRU Recurrent Belief Core.
    Carries temporal context h_t alongside instant latent observation z_t.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_emb_dim: int = 32,
        recurrent_dim: int = 256,
        num_actions: int = 17,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_emb_dim = action_emb_dim
        self.recurrent_dim = recurrent_dim
        self.num_actions = num_actions

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + action_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, recurrent_dim),
            nn.LayerNorm(recurrent_dim),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(recurrent_dim, recurrent_dim)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step recurrence update:
        z: [B, latent_dim]
        action: [B] (int64) or [B, action_emb_dim]
        h_prev: [B, recurrent_dim]
        Returns: h_t: [B, recurrent_dim]
        """
        if action.dtype in (torch.int64, torch.int32):
            if action.ndim > 1:
                action = action.squeeze(-1)
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        za = torch.cat([z, a_emb], dim=-1)
        feat = self.input_proj(za)
        h_next = self.gru(feat, h_prev)
        return h_next

    def unroll_sequence(
        self,
        z_seq: torch.Tensor,
        action_seq: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        reset_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unrolls recurrence along a sequence of transitions:
        z_seq: [B, T, latent_dim]
        action_seq: [B, T]
        h_0: [B, recurrent_dim] (optional, defaults to zeros)
        reset_masks: [B, T] (optional binary mask where 1 = episode reset / boundary)
        Returns: h_seq: [B, T, recurrent_dim]
        """
        B, T, _ = z_seq.shape
        device = z_seq.device

        if h_0 is None:
            h = torch.zeros(B, self.recurrent_dim, device=device)
        else:
            h = h_0

        h_list = []
        for t in range(T):
            if reset_masks is not None:
                # Mask out hidden state on episode boundaries
                mask_t = (1.0 - reset_masks[:, t].float()).unsqueeze(-1)
                h = h * mask_t

            h = self.forward(z_seq[:, t], action_seq[:, t], h)
            h_list.append(h)

        return torch.stack(h_list, dim=1)


class RecurrentPredictor(nn.Module):
    """
    Action-Conditioned Recurrent Latent Predictor.
    Predicts next latent delta conditioned on current latent z, action a, and recurrent context h:
    hat{z}_{t+1} = z_t + Delta z(z_t, h_t, a_t).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_actions: int = 17,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.action_emb_dim = action_emb_dim

        self.action_emb = nn.Embedding(num_actions, action_emb_dim)

        in_dim = latent_dim + recurrent_dim + action_emb_dim
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: [..., latent_dim]
        h: [..., recurrent_dim]
        action: [...] int64
        Returns: next_z: [..., latent_dim]
        """
        if action.dtype in (torch.int64, torch.int32):
            if action.ndim > 1:
                action = action.squeeze(-1)
            a_emb = self.action_emb(action)
        else:
            a_emb = action

        # Handle batch or sequence dims
        orig_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.latent_dim)
        h_flat = h.reshape(-1, self.recurrent_dim)
        a_flat = a_emb.reshape(-1, self.action_emb_dim)

        feat = torch.cat([z_flat, h_flat, a_flat], dim=-1)
        delta_z = self.net(feat)
        next_z = z_flat + delta_z
        return next_z.reshape(*orig_shape, self.latent_dim)


class JEPAv3(nn.Module):
    """
    Complete JEPA v3 Model with Recurrent Belief and Task-Shaped Representation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        recurrent_dim: int = 256,
        num_actions: int = 17,
        action_emb_dim: int = 32,
        hidden_dim: int = 256,
        q_heads: int = 3,
        q_bins: int = 51,
        q_max: float = 12.0,
        ema_decay: float = 0.99,
        transition_state_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.state_dim = latent_dim + recurrent_dim  # Composite state dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim
        self.ema_decay = ema_decay
        self.transition_state_dim = transition_state_dim

        # 1. Observation Online Encoder
        self.encoder = ConvEncoder(in_channels=3, latent_dim=latent_dim)

        # 2. Observation Target Encoder (EMA consistency target)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 3. Recurrent Core & Predictor
        self.recurrent_core = RecurrentCore(
            latent_dim=latent_dim,
            action_emb_dim=action_emb_dim,
            recurrent_dim=recurrent_dim,
            num_actions=num_actions,
        )

        self.predictor = RecurrentPredictor(
            latent_dim=latent_dim,
            recurrent_dim=recurrent_dim,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )

        # 4. Task Heads operating on Composite State s_t = [z_t, h_t]
        self.event_head = EventHead(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            num_achievements=NUM_ACHIEVEMENTS,
            hidden_dim=hidden_dim,
        )

        self.health_head = HealthHead(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
        )

        # Generic action-conditioned consequences. These heads are auxiliary to
        # JEPA and never receive environment state as an inference input.
        self.state_delta_head = StateDeltaHead(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            state_dim=transition_state_dim,
            hidden_dim=hidden_dim,
        )
        self.state_readout_head = StateReadoutHead(
            latent_dim=self.state_dim,
            state_dim=transition_state_dim,
            hidden_dim=hidden_dim,
        )
        self.transition_head = TransitionHead(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            hidden_dim=hidden_dim,
        )

        self.time_to_event_head = TimeToEventHead(
            state_dim=self.state_dim,
            num_achievements=NUM_ACHIEVEMENTS,
            hidden_dim=hidden_dim,
        )

        self.q_ensemble = QEnsemble(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            action_emb_dim=action_emb_dim,
            num_heads=q_heads,
            num_bins=q_bins,
            v_min=0.0,
            v_max=q_max,
            hidden_dim=hidden_dim,
        )

        # Target Q Ensemble for Clipped Double-Q
        self.q_target_ensemble = copy.deepcopy(self.q_ensemble)
        for p in self.q_target_ensemble.parameters():
            p.requires_grad = False

        self.policy_prior = PolicyPrior(
            latent_dim=self.state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )

        # Return normalizer for scale stability
        self.return_normalizer = ReturnNormalizer(
            low_percentile=5.0,
            high_percentile=95.0,
            target_max=q_max,
        )

    def encode(self, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Encodes [B, 3, 64, 64] image into latent z [B, latent_dim]."""
        if use_target:
            with torch.no_grad():
                return self.target_encoder(obs)
        return self.encoder(obs)

    def recurrent_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advances recurrent state by one step and returns composite state:
        Returns: (s_t, h_t) where s_t = [z, h_t]
        """
        h_t = self.recurrent_core(z, action, h_prev)
        s_t = torch.cat([z, h_t], dim=-1)
        return s_t, h_t

    def predict_next_step(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagines next state in latent space:
        Returns: (next_z, next_h, next_s)
        """
        next_z = self.predictor(z, h, action)
        next_h = self.recurrent_core(next_z, action, h)
        next_s = torch.cat([next_z, next_h], dim=-1)
        return next_z, next_h, next_s

    def predict_reward(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
        mask_before: torch.Tensor,
        m_soft: Optional[torch.Tensor] = None,
        return_decomposition: bool = False,
        use_calibration: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Predicts expected reward from composite state s = [z, h] using dual masks:
        r_hat = sum_i(clamp(p_i - m_soft_i, min=0)) + h_hat
        where p is computed using the binary mask m_bin (mask_before) fed to EventHead.
        """
        p = self.event_head.predict_proba(s, action, mask_before, use_calibration=use_calibration)
        h = self.health_head(s, action)

        if m_soft is None:
            m_soft = mask_before.float()

        achievement_reward = torch.clamp(p - m_soft.float(), min=0.0).sum(dim=-1)
        total_reward = achievement_reward + h

        if return_decomposition:
            return total_reward, {
                "event_probs": p,
                "achievement_reward": achievement_reward,
                "health_reward": h,
            }
        return total_reward

    def predict_transition(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict generic state delta, reward, termination, and uncertainty."""
        state_delta = self.state_delta_head(s, action)
        reward, done_logits, uncertainty = self.transition_head(s, action)
        return {
            "state_delta": state_delta,
            "reward": reward,
            "done_logits": done_logits,
            "done_prob": torch.sigmoid(done_logits),
            "uncertainty": uncertainty,
        }

    def predict_state(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict generic current state estimate and uncertainty from composite state s_t = [z_t, h_t]."""
        if hasattr(self, "loaded_component_keys") and "state_readout_head" not in self.loaded_component_keys:
            # Checkpoint did not contain a trained state_readout_head; return zero state to prevent Gaussian noise corruption
            if s.dim() > 1:
                zeros = torch.zeros((s.shape[0], 16), device=s.device, dtype=torch.float32)
                uncertainty = torch.ones((s.shape[0], 16), device=s.device, dtype=torch.float32)
            else:
                zeros = torch.zeros(16, device=s.device, dtype=torch.float32)
                uncertainty = torch.ones(16, device=s.device, dtype=torch.float32)
            return zeros, uncertainty
        return self.state_readout_head(s)

    def q_min(self, s: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Clipped Double-Q minimum return across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_min(s, action)

    def q_mean(self, s: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Mean expected return across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_mean(s, action)

    def q_std(self, s: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Epistemic uncertainty (std dev) across ensemble heads."""
        ensemble = self.q_target_ensemble if use_target else self.q_ensemble
        return ensemble.q_std(s, action)

    def sample_prior_action(
        self,
        s: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Samples categorical action indices guided by PolicyPrior."""
        return self.policy_prior.sample(s, mask, temperature=temperature)

    def set_calibrator(self, calibrator: Optional[Any] = None) -> None:
        """Attaches an EventHeadCalibrator to the event head."""
        self.event_head.set_calibrator(calibrator)

    def load_calibrator(self, calibrator_path: Union[str, Path]) -> None:
        """Loads and attaches an EventHeadCalibrator from JSON."""
        from models.calibration import EventHeadCalibrator
        cal = EventHeadCalibrator.load(calibrator_path)
        self.set_calibrator(cal)

    def load_default_calibrator(self, calibrator_rel_path: str = "results/calibration/event_head_calibrator.json") -> bool:
        """
        Loads the event-head calibrator from results/calibration/event_head_calibrator.json
        (resolved relative to the repo root or Drive, exactly as the autopsy script does)
        and attaches it with set_calibrator.
        If the file is missing, prints a loud warning, does not crash, and does not continue silently.
        Logs whether it loaded.
        """
        repo_root = Path(__file__).resolve().parent.parent
        drive_dir = Path("/content/drive/MyDrive/learn-a-world")
        candidates = [
            Path(calibrator_rel_path),
            repo_root / calibrator_rel_path,
            drive_dir / calibrator_rel_path,
        ]

        resolved: Optional[Path] = None
        for c in candidates:
            if c.exists():
                resolved = c.resolve()
                break

        if resolved is not None:
            try:
                from models.calibration import EventHeadCalibrator
                cal = EventHeadCalibrator.load(str(resolved))
                self.set_calibrator(cal)
                print(f"[JEPAv3] Event-head calibrator successfully loaded from {resolved} and attached.")
                return True
            except Exception as e:
                print("!" * 80)
                print(f"[LOUD WARNING] Failed to parse EventHeadCalibrator from {resolved}: {e}")
                print("[LOUD WARNING] The event-head calibrator was NOT loaded! Continuing uncalibrated.")
                print("!" * 80, flush=True)
                return False
        else:
            print("!" * 80)
            print(f"[LOUD WARNING] Event-head calibrator was NOT loaded! File not found at {calibrator_rel_path} "
                  f"(checked repo root: {repo_root}, drive: {drive_dir}). Continuing uncalibrated.")
            print("!" * 80, flush=True)
            return False

    def update_target_networks(self, decay_encoder: float = 0.99, decay_q: float = 0.995) -> None:
        """Momentum update of target encoder and target Q ensemble."""
        with torch.no_grad():
            # Update target encoder
            for t_p, o_p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                t_p.data.mul_(decay_encoder).add_(o_p.data, alpha=1.0 - decay_encoder)
            # Update target Q ensemble
            for t_p, o_p in zip(self.q_target_ensemble.parameters(), self.q_ensemble.parameters()):
                t_p.data.mul_(decay_q).add_(o_p.data, alpha=1.0 - decay_q)

    def save_checkpoint(
        self,
        path: str,
        step: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves complete v3 checkpoint with full training state (optimizer, RNG, config)."""
        ckpt = {
            "version": "v3",
            "step": step,
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "recurrent_core": self.recurrent_core.state_dict(),
            "predictor": self.predictor.state_dict(),
            "event_head": self.event_head.state_dict(),
            "health_head": self.health_head.state_dict(),
            "state_delta_head": self.state_delta_head.state_dict(),
            "state_readout_head": self.state_readout_head.state_dict(),
            "transition_head": self.transition_head.state_dict(),
            "q_ensemble": self.q_ensemble.state_dict(),
            "q_target_ensemble": self.q_target_ensemble.state_dict(),
            "policy_prior": self.policy_prior.state_dict(),
            "time_to_event_head": self.time_to_event_head.state_dict(),
            "normalizer": {
                "low": self.return_normalizer.low,
                "high": self.return_normalizer.high,
                "initialized": self.return_normalizer.initialized,
            },
            "rng_state": torch.get_rng_state(),
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if config is not None:
            ckpt["config"] = config
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[Union[str, torch.device]] = None) -> "JEPAv3":
        """Instantiates and loads a JEPAv3 model directly from checkpoint, attaching the calibrator."""
        model = cls()
        model.load_checkpoint(path, device=device)
        return model

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Audits parameter norms and initialization state across all sub-modules.
        Returns a dictionary of module norms and status flags.
        """
        modules = {
            "encoder": self.encoder,
            "target_encoder": self.target_encoder,
            "recurrent_core": self.recurrent_core,
            "predictor": self.predictor,
            "event_head": self.event_head,
            "health_head": self.health_head,
            "state_delta_head": self.state_delta_head,
            "state_readout_head": self.state_readout_head,
            "transition_head": self.transition_head,
            "q_ensemble": self.q_ensemble,
            "q_target_ensemble": self.q_target_ensemble,
            "policy_prior": self.policy_prior,
        }
        if hasattr(self, "time_to_event_head") and self.time_to_event_head is not None:
            modules["time_to_event_head"] = self.time_to_event_head

        status: Dict[str, Any] = {}
        for name, mod in modules.items():
            total_norm = 0.0
            param_count = 0
            for p in mod.parameters():
                param_count += p.numel()
                total_norm += float(p.detach().norm(2).item() ** 2)
            total_norm = float(total_norm ** 0.5)
            status[name] = {
                "num_params": param_count,
                "l2_norm": round(total_norm, 4),
                "is_healthy": bool(total_norm > 1e-4 and not np.isnan(total_norm)),
            }
        status["all_healthy"] = all(v["is_healthy"] for v in status.values())
        return status

    def load_checkpoint(
        self,
        path: str,
        device: Optional[Union[str, torch.device]] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        strict_v3: bool = False,
    ) -> Dict[str, Any]:
        """
        Loads v3 checkpoint or transfers weights from legacy checkpoints with schema normalization.
        Attaches calibrator and verifies parameter integrity.
        """
        target_dev = map_location if map_location is not None else device
        dev = torch.device(target_dev) if target_dev is not None else torch.device("cpu")
        self.to(dev)
        ckpt = torch.load(path, map_location=dev)

        loaded_components = []
        skipped_components = []

        is_native_v3 = ckpt.get("version") == "v3"

        # Direct component name mapping: target_attr -> list of possible checkpoint keys
        key_mappings = {
            "encoder": ["encoder", "base_model_state"],
            "target_encoder": ["target_encoder"],
            "recurrent_core": ["recurrent_core", "gru_belief_state"],
            "predictor": ["predictor"],
            "event_head": ["event_head", "event_head_state"],
            "health_head": ["health_head", "health_head_state"],
            "state_delta_head": ["state_delta_head", "state_delta_head_state"],
            "state_readout_head": ["state_readout_head", "state_readout_head_state"],
            "transition_head": ["transition_head", "transition_head_state"],
            "q_ensemble": ["q_ensemble", "q_ensemble_state"],
            "q_target_ensemble": ["q_target_ensemble", "q_target_ensemble_state"],
            "policy_prior": ["policy_prior", "policy_prior_state"],
            "time_to_event_head": ["time_to_event_head", "tte_head_state"],
        }

        for attr, possible_keys in key_mappings.items():
            target_mod = getattr(self, attr, None)
            if target_mod is None:
                continue

            found_state = None
            found_key = None
            for k in possible_keys:
                if k in ckpt and isinstance(ckpt[k], (dict, torch.nn.modules.container.Sequential)):
                    found_state = ckpt[k]
                    found_key = k
                    break

            if found_state is not None:
                # If loading encoder from base_model_state, strip prefix
                if found_key == "base_model_state" and attr in ("encoder", "target_encoder"):
                    found_state = {
                        k.replace("encoder.", ""): v
                        for k, v in found_state.items()
                        if k.startswith("encoder.")
                    }

                try:
                    target_mod.load_state_dict(found_state, strict=True)
                    loaded_components.append(attr)
                except Exception as e_strict:
                    # Try non-strict load if strict fails
                    try:
                        incompatible = target_mod.load_state_dict(found_state, strict=False)
                        missing = len(incompatible.missing_keys)
                        unexpected = len(incompatible.unexpected_keys)
                        if missing > 0 or unexpected > 0:
                            print(
                                f"[JEPAv3 Partial Load] '{attr}' loaded partially from '{found_key}' "
                                f"(missing: {missing}, unexpected: {unexpected})."
                            )
                        loaded_components.append(f"{attr} (partial)")
                    except Exception as e_nonstrict:
                        skipped_components.append(f"{attr} (incompatible: {e_nonstrict})")
                        if strict_v3:
                            raise ValueError(
                                f"[JEPAv3 Strict Error] Failed loading component '{attr}' from '{found_key}': {e_nonstrict}"
                            )
            else:
                skipped_components.append(f"{attr} (not in checkpoint)")
                if strict_v3 and is_native_v3:
                    raise KeyError(f"[JEPAv3 Strict Error] Expected key for '{attr}' not found in {path}")

        # Normalizer
        if "normalizer" in ckpt:
            self.return_normalizer.low = ckpt["normalizer"].get("low", 0.0)
            self.return_normalizer.high = ckpt["normalizer"].get("high", 1.0)
            self.return_normalizer.initialized = ckpt["normalizer"].get("initialized", True)
            loaded_components.append("normalizer")

        # Provenance: which components actually came from the file vs were left at
        # random init. Consumers (e.g. MPC's transition consequence term) must gate
        # on this — a random-init head's outputs are noise, not predictions.
        self.loaded_component_keys = {
            name.split(" ")[0] for name in loaded_components
        }
        self.to(dev)

        # Integrity verification
        audit = self.verify_integrity()
        print(f"[JEPAv3 Checkpoint Audit] Loaded from: {path} (version: {ckpt.get('version', 'legacy')})")
        print(f"  -> Successfully Loaded ({len(loaded_components)}): {', '.join(loaded_components)}")
        if skipped_components:
            print(f"  -> Skipped / Fresh ({len(skipped_components)}): {', '.join(skipped_components)}")

        # Attach calibrator
        self.load_default_calibrator()

        return {
            "loaded_components": loaded_components,
            "skipped_components": skipped_components,
            "audit": audit,
        }
