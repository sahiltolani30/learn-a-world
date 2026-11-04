"""Evaluation of held-out future latent predictions against controlled baselines."""
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

from src.models.jepa_world_model import JepaWorldModel


class HeldoutPredictorEvaluator:
    """Evaluates world model predictions against persistence, no-action, and shuffled-action baselines."""

    def __init__(self, model: JepaWorldModel, horizons: List[int] = [1, 2, 4, 8]):
        self.model = model
        self.horizons = horizons

    @torch.no_grad()
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run multi-horizon prediction on a held-out batch and compare against baselines."""
        self.model.eval()
        device = next(self.model.parameters()).device

        obs_context = batch["obs_context"].to(device)
        previous_actions = batch["previous_actions"].to(device)
        context_mask = batch["context_mask"].to(device)
        actions_future = batch["actions_future"].to(device)

        B = obs_context.size(0)

        # Encode context
        z_t, h_t = self.model.encode_actual_history(obs_context, previous_actions, context_mask)
        
        # Target latents at 1, 2, 4, 8
        target_latents = {}
        for h in self.horizons:
            targ_raw = batch[f"target_{h}"].to(device)
            targ_tokens, _ = self.model.target_encoder(targ_raw)
            target_latents[h] = targ_tokens

        # Baseline 1: Persistence baseline (predict z_t for all future frames)
        persistence_errors = {}
        for h in self.horizons:
            cos_dist = self._compute_token_cosine_distance(z_t, target_latents[h])
            persistence_errors[h] = float(cos_dist.mean().item())

        # Baseline 2: Action-conditioned model (active model)
        model_rollout = self.model.autoregressive_rollout(z_t, h_t, actions_future, member_idx=0)
        model_errors = {}
        model_cos_sims = {}
        model_mses = {}
        for h in self.horizons:
            z_hat = model_rollout["predictions_by_horizon"][h]
            cos_dist = self._compute_token_cosine_distance(z_hat, target_latents[h])
            cos_sim = 1.0 - cos_dist
            mse = F.mse_loss(z_hat, target_latents[h])

            model_errors[h] = float(cos_dist.mean().item())
            model_cos_sims[h] = float(cos_sim.mean().item())
            model_mses[h] = float(mse.item())

        # Baseline 3: No-action baseline (true zero embeddings ablation)
        zero_action_embs = torch.zeros(B, actions_future.size(1), self.model.action_dim, device=device)
        no_act_rollout = self.model.autoregressive_rollout(z_t, h_t, future_action_embeddings=zero_action_embs, member_idx=0)
        no_action_errors = {}
        for h in self.horizons:
            z_hat_na = no_act_rollout["predictions_by_horizon"][h]
            cos_dist_na = self._compute_token_cosine_distance(z_hat_na, target_latents[h])
            no_action_errors[h] = float(cos_dist_na.mean().item())

        # Baseline 4: Shuffled-action baseline
        shuffled_actions = actions_future[:, torch.randperm(actions_future.size(1))]
        shuf_rollout = self.model.autoregressive_rollout(z_t, h_t, shuffled_actions, member_idx=0)
        shuffled_errors = {}
        for h in self.horizons:
            z_hat_shuf = shuf_rollout["predictions_by_horizon"][h]
            cos_dist_shuf = self._compute_token_cosine_distance(z_hat_shuf, target_latents[h])
            shuffled_errors[h] = float(cos_dist_shuf.mean().item())

        # Pass condition checks for gate
        # 1-step model beats persistence
        one_step_beats_persistence = (model_errors[1] < persistence_errors[1])
        pct_improvement_over_persistence = ((persistence_errors[1] - model_errors[1]) / max(1e-6, persistence_errors[1])) * 100.0

        return {
            "model_cosine_distance": model_errors,
            "model_cosine_similarity": model_cos_sims,
            "model_feature_mse": model_mses,
            "persistence_cosine_distance": persistence_errors,
            "no_action_cosine_distance": no_action_errors,
            "shuffled_action_cosine_distance": shuffled_errors,
            "one_step_beats_persistence": one_step_beats_persistence,
            "pct_improvement_over_persistence": pct_improvement_over_persistence,
        }

    @staticmethod
    def _compute_token_cosine_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance [B] averaged across tokens."""
        p_norm = F.normalize(pred, dim=-1, eps=1e-6)
        t_norm = F.normalize(target.detach(), dim=-1, eps=1e-6)
        # Cosine distance per token: 1.0 - cos_sim
        cos_dist_per_token = 1.0 - (p_norm * t_norm).sum(dim=-1)  # [B, 64]
        return cos_dist_per_token.mean(dim=-1)                    # [B]
