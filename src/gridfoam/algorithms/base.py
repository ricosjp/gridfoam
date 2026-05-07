from abc import ABC, abstractmethod


class AlgorithmBase(ABC):
    """
    Abstract base class for CFD algorithms (macro solvers).
    """

    @abstractmethod
    def step(self):
        """
        Advance the simulation by one algorithm step.
        """
        pass
