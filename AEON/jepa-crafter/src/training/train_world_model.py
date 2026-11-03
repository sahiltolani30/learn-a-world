"""World model offline/online training routines."""
from typing import Dict, Any, Optional
import torch

from src.models.jepa_world_model import JepaWorldModel
from src.learning.learner import JepaLearner
from src.data.sequence_sampler import SequenceSampler
from src.utils.logging import MetricLogger


def train_world_model_epoch(
    learner: JepaLearner,
    sampler: SequenceSampler,
    num_updates: int,
    batch_size: int = 32,
    logger: Optional[MetricLogger] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Execute a fixed number of learner updates using the sequence sampler."""
    total_loss = 0.0
    total_jepa = 0.0
    last_metrics = {}

    for u in range(num_updates):
        batch = sampler.sample_batch(batch_size=batch_size, device=learner.device)
        metrics = learner.train_step(batch)
        total_loss += metrics["loss"]
        total_jepa += metrics["jepa_loss"]
        last_metrics = metrics

        if logger is not None and (u + 1) % log_interval == 0:
            logger.log(step=learner.update_count, metrics=metrics)

    avg_loss = total_loss / max(1, num_updates)
    avg_jepa = total_jepa / max(1, num_updates)
    return {
        "avg_loss": avg_loss,
        "avg_jepa_loss": avg_jepa,
        "final_update_count": learner.update_count,
        "last_metrics": last_metrics,
    }
