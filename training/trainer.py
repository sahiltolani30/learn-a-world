"""
Modular JEPA World Model Trainer for Learn-a-World.
Supports:
- Local Mac (Apple Silicon MPS / CPU) and Google Colab (CUDA T4).
- Seeding replay buffer via self-play interactions.
- Interleaved mini-batch training with EMA target encoder momentum.
- Latent variance monitoring to prevent representation collapse.
- Checkpointing and automatic resumption for long runs.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

import envs  # registers environments
from experiments.utils import get_device, get_output_dir, load_config, set_seed
from models.jepa import JEPAWorldModel
from training.buffer import ReplayBuffer

def seed_buffer(
    env: gym.Env,
    buffer: ReplayBuffer,
    num_steps: int,
    random_start: bool = True,
) -> None:
    """
    Fills the replay buffer with initial self-play transitions.
    For Crafter, uses a structured exploration distribution (movement + interaction)
    to discover diverse biomes, resources, and night cycles instead of idle loops.
    """
    print(f"--> Seeding replay buffer with {num_steps} transitions...")
    obs, info = env.reset(options={"random_start": random_start})

    is_crafter = getattr(env.action_space, "n", 0) == 17

    for step in range(num_steps):
        if is_crafter:
            # Crafter exploration distribution: 60% movement across biomes, 25% interact/Do, 15% other
            p = np.random.random()
            if p < 0.60:
                action = int(np.random.randint(1, 5))  # Move Left, Right, Up, Down
            elif p < 0.85:
                action = 5  # "Do" (harvest wood/stone, drink, eat, attack)
            else:
                action = int(env.action_space.sample())
        else:
            action = int(env.action_space.sample())

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, next_obs, reward, terminated)

        if terminated or truncated:
            obs, info = env.reset(options={"random_start": random_start})
        else:
            obs = next_obs

    print(f"    [DONE] Buffer contains {len(buffer)} transitions.")

def evaluate_prediction_error(
    model: JEPAWorldModel,
    val_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Dict[str, float]:
    """Computes prediction error on a held-out validation batch."""
    model.eval()
    obs, actions, next_obs, _, _ = val_batch
    with torch.no_grad():
        _, metrics = model(obs, actions, next_obs)
    model.train()
    return metrics

def train_jepa(
    config_path: str = "experiments/configs/maze.yaml",
    device_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    total_steps_override: Optional[int] = None,
    seed_steps_override: Optional[int] = None,
    resume: bool = False,
) -> Tuple[JEPAWorldModel, Dict[str, Any]]:
    """
    Main training execution function.
    Returns the trained model and execution logs.
    """
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = get_device(device_override)
    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else get_output_dir("./checkpoints")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / cfg["training"].get("checkpoint_name", "jepa_maze.pt")

    print(f"==================================================")
    print(f"   Learn-a-World: JEPA Training ({cfg['project']})")
    print(f"   Device: {device} | Output Dir: {output_dir}")
    print(f"==================================================")

    # 1. Initialize Environment
    env_cfg = cfg["env"]
    env = gym.make(
        env_cfg["name"],
        render_size=env_cfg.get("render_size", 64),
        max_steps=env_cfg.get("max_steps", 100),
        slippery_active=env_cfg.get("slippery_active", False),
    )

    # 2. Initialize Buffer
    train_cfg = cfg["training"]
    buffer = ReplayBuffer(
        capacity=train_cfg.get("buffer_capacity", 100_000),
        obs_shape=(env_cfg.get("render_size", 64), env_cfg.get("render_size", 64), 3),
        action_dim=1,
        device=device,
    )

    # 3. Seed Buffer
    seed_steps = (
        seed_steps_override
        if seed_steps_override is not None
        else train_cfg.get("seed_steps", 2000)
    )
    seed_buffer(env, buffer, seed_steps)

    # 4. Hold out validation transitions
    val_batch = buffer.sample(min(128, len(buffer)), device=device)

    # 5. Initialize Model & Optimizer
    m_cfg = cfg["model"]
    model = JEPAWorldModel(
        in_channels=3,
        latent_dim=m_cfg.get("latent_dim", 128),
        num_actions=m_cfg.get("num_actions", 4),
        action_emb_dim=m_cfg.get("action_emb_dim", 32),
        predictor_hidden_dim=m_cfg.get("predictor_hidden_dim", 256),
        predictor_layers=m_cfg.get("predictor_layers", 2),
        ema_decay=m_cfg.get("ema_decay", 0.99),
        var_weight=m_cfg.get("var_weight", 1.0),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    start_step = 0
    history = {"train_loss": [], "val_loss": [], "variance": [], "steps": []}

    # Resume from checkpoint if requested and file exists
    if resume and checkpoint_file.exists():
        print(f"--> Resuming from checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0)
        history = ckpt.get("history", history)
        print(f"    Resumed successfully at step {start_step}.")

    total_steps = (
        total_steps_override
        if total_steps_override is not None
        else train_cfg.get("train_steps", 1000)
    )
    batch_size = train_cfg.get("batch_size", 64)
    log_interval = train_cfg.get("log_interval", 100)
    eval_interval = train_cfg.get("eval_interval", 200)
    ckpt_interval = train_cfg.get("checkpoint_interval", 500)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    # Initial validation score
    init_val = evaluate_prediction_error(model, val_batch)
    if start_step == 0:
        history["train_loss"].append(init_val["loss"])
        history["val_loss"].append(init_val["loss"])
        history["variance"].append(init_val["online_latent_variance"])
        history["steps"].append(0)

    print(f"--> Initial Val Loss: {init_val['loss']:.4f} | Latent Std-Dev: {init_val['online_latent_variance']:.4f}")

    print(f"--> Starting gradient optimization ({start_step} -> {total_steps} steps)...")
    model.train()

    for step in range(start_step + 1, total_steps + 1):
        obs, actions, next_obs, rewards, dones = buffer.sample(batch_size, device=device)

        optimizer.zero_grad()
        loss, diag = model(obs, actions, next_obs)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        model.update_target_encoder()

        # Periodic logging
        if step % log_interval == 0 or step == total_steps:
            val_metrics = evaluate_prediction_error(model, val_batch)
            history["train_loss"].append(diag["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["variance"].append(val_metrics["online_latent_variance"])
            history["steps"].append(step)

            print(
                f"    Step {step:5d}/{total_steps} | "
                f"Train Loss: {diag['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Latent Std: {val_metrics['online_latent_variance']:.4f}"
            )

        # Checkpointing
        if step % ckpt_interval == 0 or step == total_steps:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "history": history,
                },
                checkpoint_file,
            )
            print(f"    [CHECKPOINT] Saved state to {checkpoint_file}")

    env.close()
    print("==================================================")
    print(f"   TRAINING COMPLETE: Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final Latent Std: {history['variance'][-1]:.4f}")
    print("==================================================")

    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train Learn-a-World JEPA World Model")
    parser.add_argument("--config", type=str, default="experiments/configs/maze.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device override (mps, cuda, cpu)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps override")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available")
    args = parser.parse_args()

    train_jepa(
        config_path=args.config,
        device_override=args.device,
        output_dir_override=args.output_dir,
        total_steps_override=args.steps,
        resume=args.resume,
    )

if __name__ == "__main__":
    main()
