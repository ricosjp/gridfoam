from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import SolverType
from gridfoam.solvers.base import LinearSolver
from gridfoam.solvers.bicgstab import BiCGSTABSolver
from gridfoam.solvers.cg import CGSolver
from gridfoam.solvers.pyamg_bridge import PyamgBridgeSolver


def create_solver(config: SolverConfig) -> LinearSolver:
    """
    Create a solver based on the given name.
    """
    match config.method:
        case SolverType.CG:
            return CGSolver(config)
        case SolverType.BiCGSTAB:
            return BiCGSTABSolver(config)
        case SolverType.PyAMG:
            return PyamgBridgeSolver(config)
        case _:
            raise ValueError(f"Unknown solver method: {config.method}")
