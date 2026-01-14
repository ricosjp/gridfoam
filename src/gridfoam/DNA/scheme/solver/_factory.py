from gridfoam.DNA.config import SolverChoice
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.scheme.solver._BiCGSTAB import BiCGSTAB
from gridfoam.DNA.scheme.solver._CG import CG
from gridfoam.DNA.scheme.solver._choice import SolverMethodChoice
from gridfoam.DNA.scheme.solver._interface import ILinearSolver


class SolverFactory:
    registry: dict[SolverMethodChoice, ILinearSolver] = {
        SolverMethodChoice.CG: CG,
        SolverMethodChoice.BICGSTAB: BiCGSTAB,
    }

    @classmethod
    def create(
        cls, choice: SolverChoice, eq_meta: EquationMeta
    ) -> ILinearSolver:
        method = choice.method
        if method not in cls.registry:
            raise ValueError(f"Unknown solver method choice: {method}")
        return cls.registry[method](choice, eq_meta)

    @classmethod
    def register(cls, choice: SolverChoice, impl: ILinearSolver) -> None:
        cls.registry[choice] = impl
