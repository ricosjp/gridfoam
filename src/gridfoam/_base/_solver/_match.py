from gridfoam._base._field._registry import FieldRegistry
from gridfoam._base._interface._solver import ISolver
from gridfoam._base._solver._BiCGSTAB import BiCGSTAB
from gridfoam._base._solver._CG import CG
from gridfoam.config import SolverChoice
from gridfoam.utils.enums import SolverName


def match_choice_to_solver(
    choice: SolverChoice, field_registry: FieldRegistry
) -> ISolver:
    match choice.type:
        case SolverName.CG:
            return CG(
                preconditioner=choice.preconditioner,
                tolerance=choice.tolerance,
                rel_tolerance=choice.rel_tolerance,
                max_iter=choice.max_iter,
                field_registry=field_registry,
            )
        case SolverName.BICGSTAB:
            return BiCGSTAB(
                preconditioner=choice.preconditioner,
                tolerance=choice.tolerance,
                rel_tolerance=choice.rel_tolerance,
                max_iter=choice.max_iter,
                field_registry=field_registry,
            )
        case _:
            raise ValueError(f"Unknown solver: {choice.type}")
