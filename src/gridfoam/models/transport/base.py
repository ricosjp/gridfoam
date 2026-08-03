import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class TransportModel(ABC):
    """
    Abstract base class for molecular transport models.
    """

    @abstractmethod
    def nu(self) -> float | torch.Tensor:
        """
        Return molecular kinematic viscosity.

        Returns
        -------
        float or torch.Tensor
            Kinematic viscosity ``nu``.
        """
        pass
