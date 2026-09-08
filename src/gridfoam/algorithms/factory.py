from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.base import GridBase
from gridfoam.meta.enums import AlgorithmType


def create_algorithm(
    grid: GridBase, phase: str | None = None
) -> AlgorithmBase:
    """
    Create a pressure--velocity algorithm from ``fvSolution`` configuration.

    Parameters
    ----------
    grid : GridBase
        Computational grid whose ``sim_config`` selects the algorithm type.
    phase : str or None, optional
        Optional phase name used when constructing field names.

    Returns
    -------
    AlgorithmBase
        Constructed SIMPLE, PISO, or PIMPLE instance.

    Raises
    ------
    ValueError
        If the configured algorithm type is ``MANUAL``.
    """
    match grid.sim_config.fvSolution.algorithm.type:
        case AlgorithmType.SIMPLE:
            return SIMPLE(grid, phase=phase)
        case AlgorithmType.PIMPLE:
            return PIMPLE(grid, phase=phase)
        case AlgorithmType.PISO:
            return PISO(grid, phase=phase)
        case AlgorithmType.MANUAL:
            raise ValueError("Manual algorithm has no step sequence")
