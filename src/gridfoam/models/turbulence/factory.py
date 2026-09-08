from gridfoam.core.grid.base import GridBase
from gridfoam.meta.enums import TurbulenceType
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.models.turbulence.laminar import Laminar


def create_turbulence_model(grid: GridBase) -> TurbulenceModel:
    """
    Create a turbulence model from ``properties.turbulence`` configuration.

    Parameters
    ----------
    grid : GridBase
        Computational grid whose simulator configuration selects the model.

    Returns
    -------
    TurbulenceModel
        Constructed turbulence-model instance.

    Raises
    ------
    ValueError
        If the configured turbulence model type is unsupported.
    """
    type = grid.sim_config.properties.turbulence.type
    match type:
        case TurbulenceType.LAMINAR:
            return Laminar(grid)
        case _:
            raise ValueError(f"Unsupported turbulence model type: {type}")
