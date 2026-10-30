"""
src/curiosity/ensemble.py

Epistemic Curiosity Engine with Predictor Ensemble Disagreement and Online
Welford Normalization for JEPA World Model.

Fulfills requirements:
- R2: Tabula rasa intrinsic motivation driven by epistemic prediction disagreement.
- R2: Eliminates the "Noisy TV" aleatoric noise trap (aleatoric uncertainty converges
      to identical conditional means across heads -> Var_k -> 0; novel causal affordances diverge).
- R2: Online Welford running mean and standard deviation normalization for stationary scaling.
- R2: Blended total reward formulation: r_total = r_ext + beta * r_int.
"""

from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """
    Online Welford Running Mean and Variance Normalizer.
    
    Tracks empirical first and second moments in an online, numerically stable manner
    using Chan's parallel update formulation of Welford's algorithm.
    
    Supports:
    - Scalar and multidimensional tracking
    - PyTorch tensors and NumPy arrays
    - Device migration (.to(device)) and state_dict serialization
    - Safe normalization with epsilon floor preventing zero-division
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        epsilon: float = 1e-4,
        clip_max: Optional[float] = None,
    ):
        """
        Initialize the online Welford normalizer.
        
        Args:
            shape: Shape of the statistic tensors (empty tuple for scalar).
            epsilon: Numerical stability constant added to standard deviation.
            clip_max: Optional absolute clipping threshold for normalized values.
        """
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon
        self.clip_max = clip_max

        # Registered as non-parameter persistent buffers for seamless state_dict serialization
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float32))

    @property
    def mean(self) -> torch.Tensor:
        return self.running_mean

    @property
    def var(self) -> torch.Tensor:
        return self.running_var

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.running_var, min=1e-8))

    def update(self, x: Union[torch.Tensor, np.ndarray, float, int]) -> None:
        """
        Update running statistics using online parallel Welford algorithm.
        
        Args:
            x: Batch of values to incorporate into running moments.
        """
        with torch.no_grad():
            if isinstance(x, (float, int)):
                x_t = torch.tensor([float(x)], dtype=torch.float32, device=self.running_mean.device)
            elif isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x).to(dtype=torch.float32, device=self.running_mean.device)
            elif isinstance(x, torch.Tensor):
                x_t = x.detach().to(dtype=torch.float32, device=self.running_mean.device)
            else:
                raise TypeError(f"Unsupported data type for RunningMeanStd: {type(x)}")

            if x_t.numel() == 0:
                return

            # Determine batch dimension(s)
            shape_len = len(self.shape)
            if x_t.dim() > shape_len:
                batch_dims = tuple(range(x_t.dim() - shape_len))
                batch_mean = x_t.mean(dim=batch_dims)
                batch_var = x_t.var(dim=batch_dims, unbiased=False) if x_t.numel() > 1 else torch.zeros_like(batch_mean)
                batch_count = float(np.prod([x_t.shape[d] for d in batch_dims]))
            else:
                batch_mean = x_t.clone()
                batch_var = torch.zeros_like(x_t)
                batch_count = 1.0

            self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: float,
    ) -> None:
        """
        Combine current running moments with batch moments via Chan's parallel algorithm.
        """
        if self.count.item() == 0.0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(torch.clamp(batch_var, min=1e-8))
            self.count.copy_(torch.tensor(batch_count, device=self.count.device, dtype=self.count.dtype))
            return

        delta = batch_mean - self.running_mean
        total_count = self.count.item() + batch_count

        # New mean: mean_A + delta * (count_B / total_count)
        new_mean = self.running_mean + delta * (batch_count / total_count)

        # M2 update: M2_A + M2_B + delta^2 * (count_A * count_B / total_count)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / total_count)

        new_var = torch.clamp(m2 / total_count, min=1e-8)

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(torch.tensor(total_count, device=self.count.device, dtype=self.count.dtype))

    def normalize(
        self,
        x: Union[torch.Tensor, np.ndarray, float],
        center: bool = False,
        clip: Optional[float] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Normalize input values using running moments.
        
        Args:
            x: Values to normalize.
            center: If True, subtracts mean: (x - mean) / (std + eps).
                    If False, scales only: x / (std + eps), preserving non-negativity.
            clip: Optional threshold to clamp result to [-clip, clip] (or [0, clip] if center=False).
            
        Returns:
            Normalized tensor or numpy array matching input type.
        """
        is_numpy = isinstance(x, np.ndarray)
        is_scalar = isinstance(x, (float, int))

        if is_scalar:
            x_t = torch.tensor(float(x), dtype=torch.float32, device=self.running_mean.device)
        elif is_numpy:
            x_t = torch.from_numpy(x).to(dtype=torch.float32, device=self.running_mean.device)
        else:
            x_t = x

        std = torch.sqrt(torch.clamp(self.running_var, min=1e-8)) + self.epsilon
        if center:
            norm_x = (x_t - self.running_mean) / std
        else:
            norm_x = x_t / std

        clip_val = clip if clip is not None else self.clip_max
        if clip_val is not None:
            if center:
                norm_x = torch.clamp(norm_x, min=-clip_val, max=clip_val)
            else:
                norm_x = torch.clamp(norm_x, min=0.0, max=clip_val)

        if is_numpy:
            return norm_x.cpu().numpy()
        elif is_scalar:
            return norm_x.item()
        return norm_x

    def denormalize(
        self,
        x_norm: Union[torch.Tensor, np.ndarray, float],
        center: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Invert normalization: restores original scale.
        """
        is_numpy = isinstance(x_norm, np.ndarray)
        if is_numpy:
            x_t = torch.from_numpy(x_norm).to(dtype=torch.float32, device=self.running_mean.device)
        else:
            x_t = x_norm

        std = torch.sqrt(torch.clamp(self.running_var, min=1e-8)) + self.epsilon
        if center:
            orig = x_t * std + self.running_mean
        else:
            orig = x_t * std

        if is_numpy:
            return orig.cpu().numpy()
        return orig

    def reset(self) -> None:
        """Reset running statistics to initial state."""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.count.zero_()


class EpistemicCuriosityEngine(nn.Module):
    """
    Epistemic Intrinsic Curiosity Engine for JEPA World Models.
    
    Computes intrinsic exploration rewards from the variance across K predictor heads:
        r_int = (1 / K) * sum_{k=1}^K || z_hat^{(k)} - z_bar ||_2^2  (normalized by D)
        
    Mathematical Properties:
    1. Aleatoric Noise Immunity ("Noisy TV" trap elimination):
       When environment transitions have stochastic noise (e.g. night-time zombie paths,
       water animation), predictor heads trained with squared error converge to the
       identical conditional expectation E[z_{t+1}|z_t, a_t].
       Therefore: z_hat^{(1)} ~ z_hat^{(2)} ~ ... ~ z_hat^{(K)} -> Var_k(z_hat) -> 0.
       The agent completely ignores unlearnable aleatoric noise!
       
    2. Novel Causal Affordance Amplification:
       When encountering a novel transition (e.g., harvesting wood, crafting table),
       heads with diverse initializations make sharply divergent predictions:
       z_hat^{(i)} != z_hat^{(j)} -> Var_k(z_hat) >> 0.
       The agent receives a strong curiosity surge to explore causal discoveries.
       
    3. Stationary Running Normalization:
       Applies online Welford standardization to maintain stable reward scales across
       training progression as model accuracy improves.
       
    4. Reward Blending:
       Computes combined reward: r_total = r_ext + beta * r_int.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_heads: int = 4,
        beta: float = 0.05,
        normalize_by_dim: bool = True,
        clip_reward: Optional[float] = 10.0,
        use_welford: bool = True,
        center_reward: bool = False,
        epsilon: float = 1e-4,
    ):
        """
        Initialize Epistemic Curiosity Engine.
        
        Args:
            latent_dim: Dimension of JEPA latent state representation D.
            num_heads: Number of predictor ensemble heads K (default 4).
            beta: Intrinsic reward blending coefficient.
            normalize_by_dim: If True, divides sum of squared deviations by D.
            clip_reward: Maximum upper bound for normalized intrinsic reward.
            use_welford: If True, normalizes rewards using online Welford statistics.
            center_reward: If True, centers rewards by subtracting mean.
                           If False, scales by std only to preserve r_int >= 0.
            epsilon: Numerical stability floor for variance scaling.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.beta = beta
        self.normalize_by_dim = normalize_by_dim
        self.clip_reward = clip_reward
        self.use_welford = use_welford
        self.center_reward = center_reward
        self.epsilon = epsilon

        # Running normalizer for intrinsic reward scalars
        self.running_stats = RunningMeanStd(
            shape=(),
            epsilon=epsilon,
            clip_max=clip_reward,
        )

    def compute_epistemic_disagreement(
        self,
        all_preds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute epistemic disagreement across ensemble predictions.
        
        Formula:
            r_int = (1 / K) * sum_{k=1}^K || z_hat^{(k)} - z_bar ||_2^2
            (Optionally normalized by latent_dim D)
            
        Args:
            all_preds: Stacked predictions across heads of shape (K, B, D) or (K, D).
            
        Returns:
            disagreement: Non-negative variance tensor of shape (B,) or scalar.
        """
        if all_preds.dim() == 2:
            # (K, D) single transition -> expand to (K, 1, D)
            all_preds = all_preds.unsqueeze(1)
            was_single = True
        elif all_preds.dim() == 3:
            was_single = False
        else:
            raise ValueError(
                f"Expected all_preds of shape (K, B, D) or (K, D), got {all_preds.shape}"
            )

        k_heads, b_size, d_dim = all_preds.shape
        if k_heads < 2:
            raise ValueError(
                f"Epistemic disagreement requires at least 2 ensemble heads, got {k_heads}"
            )

        # Ensemble consensus mean: z_bar in R^{B x D}
        mean_pred = all_preds.mean(dim=0, keepdim=True)  # (1, B, D)

        # Squared difference from consensus
        diff_sq = (all_preds - mean_pred) ** 2  # (K, B, D)

        if self.normalize_by_dim:
            # Mean over latent dimension D, then mean over heads K
            disagreement = diff_sq.mean(dim=-1).mean(dim=0)  # (B,)
        else:
            # Sum over latent dimension D, then mean over heads K
            disagreement = diff_sq.sum(dim=-1).mean(dim=0)  # (B,)

        # Guarantee non-negativity
        disagreement = torch.clamp(disagreement, min=0.0)

        if was_single:
            return disagreement.squeeze(0)
        return disagreement

    def compute_intrinsic_reward(
        self,
        all_preds: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """
        Compute epistemic curiosity reward, applying online Welford normalization.
        
        Args:
            all_preds: Ensemble predictions of shape (K, B, D) or (K, D).
            update_stats: Whether to update running Welford statistics with this batch.
            
        Returns:
            intrinsic_reward: Normalized intrinsic reward tensor of shape (B,) or scalar.
        """
        disagreement = self.compute_epistemic_disagreement(all_preds)

        if self.use_welford:
            if update_stats and self.training:
                self.running_stats.update(disagreement.detach())

            norm_reward = self.running_stats.normalize(
                disagreement,
                center=self.center_reward,
                clip=self.clip_reward,
            )
            # Ensure strictly non-negative
            norm_reward = torch.clamp(norm_reward, min=0.0)
            if self.clip_reward is not None:
                norm_reward = torch.clamp(norm_reward, max=self.clip_reward)
            return norm_reward

        if self.clip_reward is not None:
            return torch.clamp(disagreement, min=0.0, max=self.clip_reward)
        return disagreement

    def forward(
        self,
        all_preds: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """
        Forward method alias to compute_intrinsic_reward.
        
        Args:
            all_preds: (K, B, D) tensor of ensemble predictions.
            update_stats: Whether to update running stats.
            
        Returns:
            Intrinsic reward tensor of shape (B,).
        """
        return self.compute_intrinsic_reward(all_preds, update_stats=update_stats)

    def blend_rewards(
        self,
        ext_reward: Union[torch.Tensor, np.ndarray, float],
        int_reward: Union[torch.Tensor, np.ndarray, float],
        beta: Optional[float] = None,
    ) -> Union[torch.Tensor, np.ndarray, float]:
        """
        Compute blended total reward: r_total = r_ext + beta * r_int.
        
        Args:
            ext_reward: Task environment reward.
            int_reward: Epistemic intrinsic curiosity reward.
            beta: Blending coefficient (defaults to self.beta).
            
        Returns:
            Blended reward with type matching inputs.
        """
        b = beta if beta is not None else self.beta

        if isinstance(ext_reward, torch.Tensor) or isinstance(int_reward, torch.Tensor):
            if not isinstance(ext_reward, torch.Tensor):
                ext_t = torch.tensor(ext_reward, dtype=torch.float32, device=int_reward.device)
            else:
                ext_t = ext_reward

            if not isinstance(int_reward, torch.Tensor):
                int_t = torch.tensor(int_reward, dtype=torch.float32, device=ext_t.device)
            else:
                int_t = int_reward

            return ext_t + b * int_t

        elif isinstance(ext_reward, np.ndarray) or isinstance(int_reward, np.ndarray):
            return np.asarray(ext_reward, dtype=np.float32) + b * np.asarray(int_reward, dtype=np.float32)

        else:
            return float(ext_reward) + b * float(int_reward)

    def compute_from_predictor(
        self,
        predictor: nn.Module,
        z: torch.Tensor,
        a: torch.Tensor,
        update_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience integration with LatentPredictor module.
        
        Args:
            predictor: LatentPredictor / ResidualMLPPredictor instance.
            z: Latent state tensor (B, latent_dim).
            a: Action tensor.
            update_stats: Whether to update running Welford stats.
            
        Returns:
            mean_pred: Consensus prediction (B, latent_dim).
            int_reward: Normalized intrinsic reward (B,).
        """
        mean_pred, all_preds = predictor(z, a)
        int_reward = self.compute_intrinsic_reward(all_preds, update_stats=update_stats)
        return mean_pred, int_reward
