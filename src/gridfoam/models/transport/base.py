from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float

from gridfoam.core.grid.base import IGridBase

logger = logging.getLogger(__name__)


class TransportModel(ABC):
    """
    Abstract base class for molecular transport models.

    Parameters
    ----------
    grid : IGridBase
        Computational grid.

    Attributes
    ----------
    grid : IGridBase
        Computational grid.
    """

    grid: IGridBase
    """Computational grid."""

    def __init__(self, grid: IGridBase):
        self.grid = grid

    @abstractmethod
    def nu(self) -> Float[torch.Tensor, " 1"]:
        """
        Return molecular kinematic viscosity.

        Returns
        -------
        torch.Tensor
            Kinematic viscosity ``nu`` with shape ``[1]``. Broadcasts
            against cell fields such as ``nu_t`` with shape ``[C, 1]``.
        """
        pass
