"""Staged action selection: warmup, curiosity-driven exploration, and MPC."""
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

from src.models.jepa_world_model import JepaWorldModel
from src.exploration.novelty import NoveltyEstimator
from src.exploration.disagreement import DisagreementEstimator, LearningProgressEstimator
from src.exploration.role_definitions import ROLE_WEIGHTS, RoleWeights


class ActionSelector:
    """Selects discrete actions using random warm-up, curiosity scoring, or MPC planning."""

    def __init__(
        self,
        num_actions: int = 17,
        random_action_floor: float = 0.10,
        temperature: float = 1.0,
        novelty_estimator: Optional[NoveltyEstimator] = None,
        disagreement_estimator: Optional[DisagreementEstimator] = None,
        progress_estimator: Optional[LearningProgressEstimator] = None,
    ):
        self.num_actions = num_actions
        self.random_action_floor = random_action_floor
        self.temperature = temperature
        
        self.novelty = novelty_estimator if novelty_estimator is not None else NoveltyEstimator()
        self.disagreement = disagreement_estimator if disagreement_estimator is not None else DisagreementEstimator()
        self.progress = progress_estimator if progress_estimator is not None else LearningProgressEstimator()

    def select_action(
        self,
        latent: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        model: Optional[JepaWorldModel],
        mode: str = "warmup",
        role: str = "explorer",
        planner: Optional[Any] = None,
    ) -> int:
        """Select action token in [0, 16] according to current policy mode and role."""
        # 1. Warm-up mode
        if mode == "warmup" or model is None or latent is None or context is None:
            return int(np.random.randint(0, self.num_actions))

        # 2. Random action floor for guaranteed exploration
        if np.random.rand() < self.random_action_floor:
            return int(np.random.randint(0, self.num_actions))

        weights = ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["explorer"])

        # 3. Model predictive control mode
        if mode == "mpc" and planner is not None:
            return planner.plan_first_action(
                latent=latent,
                context=context,
                role_weights=weights,
            )

        # 4. Curiosity-guided pre-MPC exploration: score each opaque token
        model.eval()
        scores = []
        with torch.no_grad():
            for token in range(self.num_actions):
                t_tensor = torch.tensor([token], device=latent.device)
                predictions, mean_pooled, _ = model.predict_one_step_ensemble(latent, context, t_tensor)
                
                nov_val = self.novelty.evaluate(mean_pooled, update_memory=False)
                dis_val = self.disagreement.evaluate_predictions(predictions)
                prog_val = self.progress.score()

                score = (
                    weights.novelty * nov_val
                    + weights.disagreement * dis_val
                    + weights.progress * prog_val
                )
                scores.append(score)

        # Softmax categorical sampling
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = F.softmax(score_tensor / max(1e-4, self.temperature), dim=-1).numpy()
        return int(np.random.choice(self.num_actions, p=probs))
