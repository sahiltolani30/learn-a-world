"""Phase P2 World Model training and evaluation runner for MOSAIC."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim

from .checkpoint import save_atomic
from .env_boundary import CrafterPixelEnv
from .models.world_model import MosaicWorldModel
from .models.diagnostics import (
    evaluate_action_conditioning,
    evaluate_heldout_prediction,
    evaluate_representation_collapse,
)


@dataclass(frozen=True)
class WorldModelConfig:
    num_actions: int = 17
    latent_dim: int = 128
    action_dim: int = 64
    reward_dim: int = 16
    hidden_dim: int = 256
    stochastic_dim: int = 32
    num_ensemble_members: int = 3
    ema_decay: float = 0.99
    use_anchor_decoder: bool = True
    learning_rate: float = 3e-4
    lambda_dyn: float = 1.0
    lambda_anchor: float = 0.1
    lambda_r: float = 1.0
    lambda_c: float = 1.0
    lambda_kl: float = 0.1
    free_bits: float = 0.1
    batch_size: int = 16
    seq_len: int = 8
    train_steps: int = 500
    collect_episodes: int = 10
    eval_episodes: int = 5
    seed: int = 12345
    eval_seed_start: int = 900_000
    permutation_seed: int = 12345
    device: str = "auto"
    output_dir: str = "runs/mosaic/p2_world_model"


class SequenceReplayBuffer:
    """Stores complete episodes and samples boundary-safe contiguous sequence segments."""

    def __init__(self, capacity_episodes: int = 100) -> None:
        self.capacity = int(capacity_episodes)
        self.episodes: List[Dict[str, np.ndarray]] = []

    def add_episode(
        self,
        observations: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        continues: List[float],
    ) -> None:
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)

        self.episodes.append({
            "observations": np.stack(observations, axis=0),  # [T, 64, 64, 3] uint8
            "actions": np.array(actions, dtype=np.int64),      # [T]
            "rewards": np.array(rewards, dtype=np.float32),    # [T]
            "continues": np.array(continues, dtype=np.float32),# [T]
        })

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of shape [B, T, ...]."""
        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_cont = []

        valid_episodes = [ep for ep in self.episodes if len(ep["actions"]) >= seq_len]
        if not valid_episodes:
            raise ValueError(f"No episode has length >= {seq_len}")

        for _ in range(batch_size):
            ep = valid_episodes[int(rng.integers(0, len(valid_episodes)))]
            max_start = len(ep["actions"]) - seq_len
            start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            end_idx = start_idx + seq_len

            batch_obs.append(ep["observations"][start_idx:end_idx])
            batch_act.append(ep["actions"][start_idx:end_idx])
            batch_rew.append(ep["rewards"][start_idx:end_idx])
            batch_cont.append(ep["continues"][start_idx:end_idx])

        # Convert to tensors: [B, T, 3, 64, 64] float in [0, 1]
        obs_np = np.stack(batch_obs, axis=0)  # [B, T, 64, 64, 3]
        obs_tensor = torch.from_numpy(obs_np).permute(0, 1, 4, 2, 3).float() / 255.0
        act_tensor = torch.from_numpy(np.stack(batch_act, axis=0)).long()
        rew_tensor = torch.from_numpy(np.stack(batch_rew, axis=0)).float()
        cont_tensor = torch.from_numpy(np.stack(batch_cont, axis=0)).float()

        return obs_tensor, act_tensor, rew_tensor, cont_tensor


def collect_dataset(
    num_episodes: int,
    seed_start: int,
    permutation_seed: int,
) -> SequenceReplayBuffer:
    """Collect episodes using the sanitized Crafter boundary."""
    buffer = SequenceReplayBuffer(capacity_episodes=num_episodes + 10)
    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        env = CrafterPixelEnv(seed=seed, permutation_seed=permutation_seed)
        obs = env.reset(seed=seed)
        observations = [obs]
        actions = []
        rewards = []
        continues = []

        rng = np.random.default_rng(seed)
        step = 0
        while True:
            # Exploratory action
            action = int(rng.integers(0, env.num_actions))
            next_obs, reward, terminated, truncated = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            continues.append(0.0 if terminated else 1.0)
            observations.append(next_obs)
            step += 1
            if terminated or truncated:
                break
        env.close()

        # Drop the final extra observation so lengths match actions
        buffer.add_episode(
            observations=observations[:-1],
            actions=actions,
            rewards=rewards,
            continues=continues,
        )
    return buffer


def run_world_model_training(config: WorldModelConfig) -> Dict[str, Any]:
    """Train the MOSAIC Phase P2 world model and compute representation/dynamics gates."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(config.device)

    # 1. Collect training and held-out evaluation datasets
    train_buffer = collect_dataset(
        num_episodes=config.collect_episodes,
        seed_start=config.seed,
        permutation_seed=config.permutation_seed,
    )
    eval_buffer = collect_dataset(
        num_episodes=config.eval_episodes,
        seed_start=config.eval_seed_start,
        permutation_seed=config.permutation_seed,
    )

    # 2. Instantiate World Model
    wm = MosaicWorldModel(
        num_actions=config.num_actions,
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        reward_dim=config.reward_dim,
        hidden_dim=config.hidden_dim,
        stochastic_dim=config.stochastic_dim,
        num_ensemble_members=config.num_ensemble_members,
        ema_decay=config.ema_decay,
        use_anchor_decoder=config.use_anchor_decoder,
    ).to(device)

    optimizer = optim.Adam(wm.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)

    # 3. Training Loop
    wm.train()
    history = []
    for step in range(config.train_steps):
        obs_b, act_b, rew_b, cont_b = train_buffer.sample_batch(
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            rng=rng,
        )
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)
        rew_b = rew_b.to(device)
        cont_b = cont_b.to(device)

        optimizer.zero_grad()
        losses = wm.compute_losses(
            obs_seq=obs_b,
            action_seq=act_b,
            reward_seq=rew_b,
            continue_seq=cont_b,
            lambda_dyn=config.lambda_dyn,
            lambda_anchor=config.lambda_anchor,
            lambda_r=config.lambda_r,
            lambda_c=config.lambda_c,
            lambda_kl=config.lambda_kl,
            free_bits=config.free_bits,
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=1.0)
        optimizer.step()
        wm.update_target_network()

        if (step + 1) % max(1, config.train_steps // 10) == 0 or step == config.train_steps - 1:
            log_entry = {
                "step": step + 1,
                "total_loss": float(losses["total_loss"].item()),
                "loss_dyn": float(losses["loss_dyn"].item()),
                "loss_anchor": float(losses["loss_anchor"].item()),
                "loss_r": float(losses["loss_r"].item()),
                "loss_c": float(losses["loss_c"].item()),
                "loss_kl": float(losses["loss_kl"].item()),
                "disagreement": float(losses["disagreement"].item()),
            }
            history.append(log_entry)

    # 4. Evaluation and Gates on Held-Out Seed Data
    wm.eval()
    with torch.no_grad():
        eval_batch_size = max(16, config.batch_size)
        eval_obs, eval_act, eval_rew, eval_cont = eval_buffer.sample_batch(
            batch_size=eval_batch_size,
            seq_len=2,
            rng=np.random.default_rng(config.eval_seed_start),
        )
        eval_obs = eval_obs.to(device)
        eval_act = eval_act.to(device)

        # A. Action conditioning test
        h_0, z_0, b_0 = wm.initial_belief(eval_obs.size(0), device=device)
        action_cond_diag = evaluate_action_conditioning(wm, b_0, action_token_1=0, action_token_2=1)

        # B. Held-out prediction vs persistence test
        heldout_pred_diag = evaluate_heldout_prediction(
            world_model=wm,
            obs_t=eval_obs[:, 0],
            action_t=eval_act[:, 0],
            obs_next=eval_obs[:, 1],
        )

        # C. Representation collapse check
        flat_eval_obs = eval_obs.reshape(-1, 3, 64, 64)
        _, eval_pooled = wm.encoder(flat_eval_obs)
        collapse_diag = evaluate_representation_collapse(eval_pooled)

    results = {
        "format": "mosaic_p2_world_model_v1",
        "config": asdict(config),
        "history": history,
        "diagnostics": {
            "action_conditioning": action_cond_diag,
            "heldout_prediction": heldout_pred_diag,
            "representation_collapse": collapse_diag,
        },
        "gates": {
            "action_conditioning_passed": bool(action_cond_diag["is_action_sensitive"]),
            "collapse_avoided": bool(not collapse_diag["is_collapsed"]),
            "prediction_measured": True,
        },
    }

    # 5. Atomic checkpoint save
    save_atomic(
        output_dir / "world_model.pt",
        {
            "format": "mosaic_runtime_v1",
            "phase": "p2_world_model",
            "state_dict": wm.state_dict(),
            "config": asdict(config),
            "results": results,
        },
    )
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    return results
