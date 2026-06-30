from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import TurbulenceType
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.models.turbulence.laminar import Laminar


def create_turbulence_model(grid: IGridBase) -> TurbulenceModel:
    type = grid.sim_config.properties.turbulence.type
    match type:
        case TurbulenceType.LAMINAR:
            return Laminar(grid)
        case _:
            raise ValueError(f"Unsupported turbulence model type: {type}")
