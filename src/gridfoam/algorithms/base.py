from abc import ABC, abstractmethod

from gridfoam.core.grid.base import IGridBase
from gridfoam.models.turbulence.base import TurbulenceModel


class AlgorithmBase(ABC):
    """
    Abstract base class for CFD algorithms (macro solvers).
    """

    @property
    @abstractmethod
    def grid(self) -> IGridBase:
        """
        Return the computational grid.
        """
        pass

    @property
    @abstractmethod
    def turbulence(self) -> TurbulenceModel:
        """
        Return the turbulence model.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Advance the simulation by one algorithm step.
        """
        pass
