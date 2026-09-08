from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float

from gridfoam.core.grid.base import GridBase

logger = logging.getLogger(__name__)


class TransportModel(ABC):
    """
    Abstract base class for molecular transport models.

    Parameters
    ----------
    grid : GridBase
        Computational grid.

    Attributes
    ----------
    grid : GridBase
        Computational grid.
    """

    grid: GridBase
    """Computational grid."""

    def __init__(self, grid: GridBase):
        self.grid = grid

    @abstractmethod
    def nu(self) -> Float[torch.Tensor, ""]:
        """
        Return molecular kinematic viscosity.

        Returns
        -------
        torch.Tensor
            Kinematic viscosity ``nu`` with shape ``()``. Broadcasts
            against cell fields such as ``nu_t`` with shape ``[C]``.
        """
        pass
