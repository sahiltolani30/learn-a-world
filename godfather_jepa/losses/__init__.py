"""Losses package for Godfather JEPA."""
from godfather_jepa.losses.vicreg import (
    decoupled_vicreg_loss,
    vicreg_loss,
    DecoupledVICRegLoss,
)

__all__ = ["decoupled_vicreg_loss", "vicreg_loss", "DecoupledVICRegLoss"]

