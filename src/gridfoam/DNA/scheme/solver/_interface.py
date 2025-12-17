import abc

from gridfoam.DNA._gridhandle import IGridHandle
from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


class ILinearSolver(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, solver_choice: SolverChoice, eq_meta: EquationMeta
    ) -> None:
        pass

    @property
    @abc.abstractmethod
    def required_fields(self) -> list[FieldMeta]:
        pass

    @abc.abstractmethod
    def solve(
        self,
        grid_handle: IGridHandle,
    ) -> None:
        """
        Solve the equation.

        Parameters
        ----------
        grid_handle : GridHandle
            Grid handle.
        """
        pass
