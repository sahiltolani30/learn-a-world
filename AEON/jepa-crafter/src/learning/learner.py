"""JEPA Learner orchestrating model updates, target EMA, and mixed-precision training."""
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.models.jepa_world_model import JepaWorldModel
from src.learning.jepa_losses import JepaLoss
from src.learning.feature_diagnostics import compute_feature_diagnostics
from src.utils.safety_checks import assert_finite_tensor


class JepaLearner:
    """Manages forward/backward updates, target EMA updates, diagnostics, and gradient scaling."""

    def __init__(
        self,
        model: JepaWorldModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[JepaLoss] = None,
        device: torch.device = torch.device("cpu"),
        precision: str = "fp16",
        grad_clip_norm: float = 1.0,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 0.999,
        ema_momentum_ramp: int = 50000,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else JepaLoss()
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.ema_momentum_ramp = ema_momentum_ramp

        self.use_amp = (precision == "fp16" and device.type == "cuda")
        dev_str = "cuda" if device.type == "cuda" else "cpu"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(dev_str, enabled=self.use_amp)
        else:
            self.scaler = GradScaler(enabled=self.use_amp)
        self.update_count = 0

    def get_current_ema_momentum(self) -> float:
        """Calculate current target EMA momentum based on update schedule."""
        progress = min(1.0, float(self.update_count) / float(max(1, self.ema_momentum_ramp)))
        return self.ema_momentum_start + (self.ema_momentum_end - self.ema_momentum_start) * progress

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single JEPA training update from a sampled batch."""
        self.model.train()
        self.model.target_encoder.eval()

        obs_context = batch["obs_context"].to(self.device)         # [B, 4, 3, 64, 64]
        context_mask = batch["context_mask"].to(self.device)       # [B, 4]
        previous_actions = batch["previous_actions"].to(self.device)# [B, 4]
        actions_future = batch["actions_future"].to(self.device)   # [B, 8]
        target_1 = batch["target_1"].to(self.device)               # [B, 3, 64, 64]
        target_2 = batch["target_2"].to(self.device)               # [B, 3, 64, 64]
        target_4 = batch["target_4"].to(self.device)               # [B, 3, 64, 64]
        target_8 = batch["target_8"].to(self.device)               # [B, 3, 64, 64]
        rewards_first = batch["reward_first"].to(self.device)       # [B, 1]
        done_first = batch["done_first"].to(self.device)           # [B, 1]

        B = obs_context.size(0)
        dev_str = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_ctx = (
            torch.amp.autocast(dev_str, enabled=self.use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else autocast(enabled=self.use_amp)
        )

        with autocast_ctx:
            # 1. Online context encoding
            z_t, h_t = self.model.encode_actual_history(obs_context, previous_actions, context_mask)

            # 2. Stop-gradient EMA target latents
            with torch.no_grad():
                targ_tokens_1, _ = self.model.target_encoder(target_1)
                targ_tokens_2, _ = self.model.target_encoder(target_2)
                targ_tokens_4, _ = self.model.target_encoder(target_4)
                targ_tokens_8, _ = self.model.target_encoder(target_8)

            targets_by_h = {
                1: targ_tokens_1,
                2: targ_tokens_2,
                4: targ_tokens_4,
                8: targ_tokens_8,
            }

            # 3. Rollout across ensemble members
            predictions_by_member = []
            aux_outputs_by_member = []
            bootstrap_masks = []

            for m_idx in range(self.model.num_predictor_members):
                # Independent bootstrap mask per member (Bernoulli p=0.8 or full)
                b_mask = (torch.rand(B, device=self.device) > 0.1).float()
                bootstrap_masks.append(b_mask)

                z_hat = z_t
                h = h_t
                member_preds = {}
                aux_first = None

                for k in range(1, 9):
                    a_idx = actions_future[:, k - 1]
                    a_emb = self.model.action_embedding(a_idx)
                    z_hat = self.model.predictor_ensemble.forward_member(m_idx, z_hat, h, a_emb)
                    u_hat = self.model.online_encoder.pool(z_hat)
                    h = self.model.temporal_context.step(u_hat, a_emb, h)

                    if k == 1:
                        aux_first = self.model.auxiliary_heads(u_hat, h)
                    if k in {1, 2, 4, 8}:
                        member_preds[k] = z_hat

                predictions_by_member.append(member_preds)
                aux_outputs_by_member.append(aux_first)

            # 4. Multi-horizon weighted loss
            loss_dict = self.loss_fn(
                predictions_by_member=predictions_by_member,
                targets_by_horizon=targets_by_h,
                bootstrap_masks=bootstrap_masks,
                auxiliary_outputs=aux_outputs_by_member,
                rewards_first=rewards_first,
                done_first=done_first,
            )
            total_loss = loss_dict["loss"]

        # 5. Optimization step with gradient scaling
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, self.grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        # 6. Target EMA update
        momentum = self.get_current_ema_momentum()
        self.model.target_encoder.update(self.model.online_encoder, momentum=momentum)

        self.update_count += 1

        # Collect metrics
        metrics = {
            "loss": total_loss.item(),
            "jepa_loss": loss_dict["jepa_loss"].item(),
            "reward_loss": loss_dict["reward_loss"].item(),
            "termination_loss": loss_dict["termination_loss"].item(),
            "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
            "ema_momentum": momentum,
            "update_count": self.update_count,
        }
        for h in [1, 2, 4, 8]:
            if f"loss_h{h}" in loss_dict:
                metrics[f"loss_h{h}"] = loss_dict[f"loss_h{h}"].item()

        return metrics
