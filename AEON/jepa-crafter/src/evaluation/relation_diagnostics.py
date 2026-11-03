"""Post-hoc relation diagnostics and frozen probe analysis (Tests H & I)."""
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.jepa_world_model import JepaWorldModel


class RelationDiagnostics:
    """Evaluates post-hoc representational capacity (Test H) and context-conditional dynamics (Test I)."""

    def __init__(self, model: JepaWorldModel):
        self.model = model

    @torch.no_grad()
    def evaluate_conditional_relationships(
        self,
        contexts_a: torch.Tensor,
        contexts_b: torch.Tensor,
        action_token: int,
    ) -> Dict[str, Any]:
        """Test whether the model predicts distinct outcomes for the SAME action in DIFFERENT visual contexts (Test I).
        
        Args:
            contexts_a: [B, 4, 3, 64, 64]
            contexts_b: [B, 4, 3, 64, 64] (distinct scenes)
            action_token: opaque action token
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        ca = contexts_a.to(device)
        cb = contexts_b.to(device)
        B = ca.size(0)

        prev_actions = torch.zeros(B, 4, dtype=torch.long, device=device)
        za, ha = self.model.encode_actual_history(ca, prev_actions)
        zb, hb = self.model.encode_actual_history(cb, prev_actions)

        act = torch.tensor([action_token], device=device).expand(B)
        a_emb = self.model.action_embedding(act)

        pred_a = self.model.predictor_ensemble.forward_member(0, za, ha, a_emb)
        pred_b = self.model.predictor_ensemble.forward_member(0, zb, hb, a_emb)

        pa_norm = F.normalize(pred_a.mean(dim=1), dim=-1)
        pb_norm = F.normalize(pred_b.mean(dim=1), dim=-1)
        cos_dist = (1.0 - (pa_norm * pb_norm).sum(dim=-1)).mean().item()

        return {
            "conditional_outcome_divergence": float(cos_dist),
            "context_sensitive": (cos_dist > 1e-4),
        }

    def train_linear_probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 50,
        lr: float = 1e-2,
    ) -> float:
        """Post-hoc linear probe on frozen features (Test H). Evaluator-only diagnostic."""
        device = features.device
        D = features.size(1)
        probe = nn.Linear(D, 1).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.MSELoss()

        feat = features.detach()
        lab = labels.float().unsqueeze(-1).detach()

        for _ in range(num_epochs):
            optimizer.zero_grad()
            out = probe(feat)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = probe(feat).squeeze(-1)
            mae = F.l1_loss(preds, labels.float()).item()
        return float(mae)
