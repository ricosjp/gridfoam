import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class TransportModel(ABC):
    @abstractmethod
    def nu(self) -> float | torch.Tensor:
        pass
