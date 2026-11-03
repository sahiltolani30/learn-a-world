"""Optimizer and learning rate scheduler configuration for JEPA."""
import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.models.jepa_world_model import JepaWorldModel


def build_optimizer_and_scheduler(
    model: JepaWorldModel,
    encoder_lr: float = 1e-4,
    predictor_lr: float = 3e-4,
    auxiliary_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_updates: int = 1000,
    max_updates: int = 250000,
    min_lr_ratio: float = 0.1,
) -> Tuple[AdamW, LambdaLR]:
    """Build AdamW optimizer with parameter groups and warmup-cosine scheduler."""
    # Parameter groups with distinct learning rates
    param_groups = [
        {
            "params": model.online_encoder.parameters(),
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder",
        },
        {
            "params": list(model.action_embedding.parameters())
            + list(model.temporal_context.parameters())
            + list(model.predictor_ensemble.parameters()),
            "lr": predictor_lr,
            "weight_decay": weight_decay,
            "name": "predictor_and_context",
        },
        {
            "params": model.auxiliary_heads.parameters(),
            "lr": auxiliary_lr,
            "weight_decay": weight_decay,
            "name": "auxiliary",
        },
    ]

    optimizer = AdamW(param_groups)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_updates:
            return float(current_step + 1) / float(max(1, warmup_updates))
        # Cosine decay
        progress = float(current_step - warmup_updates) / float(max(1, max_updates - warmup_updates))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
