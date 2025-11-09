from gridfoam._base._equation._equation import Equation
from gridfoam._base._field._registry import FieldRegistry
from gridfoam._base._interface._solver import ISolver
from gridfoam._base._solver._match import match_choice_to_solver
from gridfoam.config import SolverChoice, fvSolutionConfig


class SolverRegistry:
    def __init__(self, config: fvSolutionConfig, field_registry: FieldRegistry):
        # key: equation tag
        self._registry: dict[str, ISolver] = {}
        self._register_solver(config.solvers, field_registry)

    def _register_solver(
        self, solvers: dict[str, SolverChoice], field_registry: FieldRegistry
    ) -> None:
        for key, solver_choice in solvers.items():
            self._registry[key] = match_choice_to_solver(
                solver_choice, field_registry
            )

    def resolve(self, equation: Equation) -> ISolver:
        return self._registry[equation.tag]
