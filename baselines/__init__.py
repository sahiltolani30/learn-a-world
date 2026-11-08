"""Baselines module for Learn-a-World."""

try:
    from baselines.ppo_baseline import train_ppo
except ImportError:
    train_ppo = None

__all__ = ["train_ppo"]
