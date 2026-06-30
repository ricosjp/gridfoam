from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import AlgorithmType


def create_algorithm(
    grid: IGridBase, phase: str | None = None
) -> AlgorithmBase:
    match grid.sim_config.fvSolution.algorithm.type:
        case AlgorithmType.SIMPLE:
            return SIMPLE(grid, phase=phase)
        case AlgorithmType.PIMPLE:
            return PIMPLE(grid, phase=phase)
        case AlgorithmType.PISO:
            return PISO(grid, phase=phase)
        case AlgorithmType.MANUAL:
            raise ValueError("Manual algorithm has no step sequence")
