"""Paired action-conditioning validation tests (Test C)."""
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

from src.models.jepa_world_model import JepaWorldModel
from src.env.crafter_adapter import CrafterAdapter
from src.utils.safety_checks import assert_finite_tensor


class ActionConditioningTester:
    """Evaluates whether the latent predictor responds specifically and accurately to action tokens."""

    def __init__(self, model: JepaWorldModel):
        self.model = model

    @torch.no_grad()
    def run_paired_action_test(
        self,
        seeds: List[int] = [1000, 1001, 1002, 1003, 1004],
        test_actions: List[int] = [0, 1, 2, 3, 4],
    ) -> Dict[str, Any]:
        """From matched identical initial states, step different action tokens and measure prediction fidelity."""
        self.model.eval()
        device = next(self.model.parameters()).device

        paired_model_errors = []
        paired_no_action_errors = []
        paired_shuffled_errors = []
        action_prediction_divergence = []

        for seed in seeds:
            # 1. Reset from fixed seed
            adapter_root = CrafterAdapter(seed=seed)
            root_obs = adapter_root.reset()
            adapter_root.close()

            # Prepare root tensor
            root_ch = np.transpose(root_obs, (2, 0, 1))  # [3, 64, 64]
            root_tensor = torch.from_numpy(root_ch).float().div_(255.0).unsqueeze(0).to(device)
            # Dummy context window: single frame
            obs_ctx = root_tensor.unsqueeze(1)  # [1, 1, 3, 64, 64]
            prev_act = torch.zeros(1, 1, dtype=torch.long, device=device)
            z_t, h_t = self.model.encode_actual_history(obs_ctx, prev_act)

            predicted_latents_for_actions = []

            for act in test_actions:
                # 2. Branch: execute action `act` in clean env instance with same seed
                adapter_branch = CrafterAdapter(seed=seed)
                adapter_branch.reset()
                next_obs, _, _, _ = adapter_branch.step(act)
                adapter_branch.close()

                next_ch = np.transpose(next_obs, (2, 0, 1))
                next_tensor = torch.from_numpy(next_ch).float().div_(255.0).unsqueeze(0).to(device)
                target_latent, _ = self.model.target_encoder(next_tensor)

                # Active action prediction
                act_tensor = torch.tensor([act], device=device)
                a_emb = self.model.action_embedding(act_tensor)
                z_hat = self.model.predictor_ensemble.forward_member(0, z_t, h_t, a_emb)
                predicted_latents_for_actions.append(z_hat)

                # Error of conditioned model
                err_cond = (1.0 - (F.normalize(z_hat, dim=-1) * F.normalize(target_latent, dim=-1)).sum(dim=-1)).mean().item()
                paired_model_errors.append(err_cond)

                # Control 1: No-action error
                zero_emb = torch.zeros_like(a_emb)
                z_hat_na = self.model.predictor_ensemble.forward_member(0, z_t, h_t, zero_emb)
                err_na = (1.0 - (F.normalize(z_hat_na, dim=-1) * F.normalize(target_latent, dim=-1)).sum(dim=-1)).mean().item()
                paired_no_action_errors.append(err_na)

                # Control 2: Shuffled action (different random action)
                shuf_act = (act + 1) % 17
                shuf_emb = self.model.action_embedding(torch.tensor([shuf_act], device=device))
                z_hat_shuf = self.model.predictor_ensemble.forward_member(0, z_t, h_t, shuf_emb)
                err_shuf = (1.0 - (F.normalize(z_hat_shuf, dim=-1) * F.normalize(target_latent, dim=-1)).sum(dim=-1)).mean().item()
                paired_shuffled_errors.append(err_shuf)

            # Measure pairwise divergence between predictions for different actions from same context
            for i in range(len(predicted_latents_for_actions)):
                for j in range(i + 1, len(predicted_latents_for_actions)):
                    p_i = F.normalize(predicted_latents_for_actions[i], dim=-1)
                    p_j = F.normalize(predicted_latents_for_actions[j], dim=-1)
                    div = (1.0 - (p_i * p_j).sum(dim=-1)).mean().item()
                    action_prediction_divergence.append(div)

        mean_model_err = float(np.mean(paired_model_errors))
        mean_na_err = float(np.mean(paired_no_action_errors))
        mean_shuf_err = float(np.mean(paired_shuffled_errors))
        mean_divergence = float(np.mean(action_prediction_divergence))

        # Check pass conditions
        beats_no_action = mean_model_err < mean_na_err
        beats_shuffled = mean_model_err < mean_shuf_err
        action_sensitive = mean_divergence > 1e-4

        return {
            "mean_action_conditioned_error": mean_model_err,
            "mean_no_action_error": mean_na_err,
            "mean_shuffled_action_error": mean_shuf_err,
            "action_prediction_divergence": mean_divergence,
            "beats_no_action": beats_no_action,
            "beats_shuffled": beats_shuffled,
            "action_sensitive": action_sensitive,
            "test_c_passed": (beats_no_action and beats_shuffled and action_sensitive),
        }
