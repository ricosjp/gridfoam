from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import TransportModelType
from gridfoam.models.transport.base import TransportModel
from gridfoam.models.transport.newtonian import NewtonianTransport


def create_transport_model(grid: IGridBase) -> TransportModel:
    type = grid.sim_config.properties.transport.type
    match type:
        case TransportModelType.NEWTONIAN:
            return NewtonianTransport(grid)
        case _:
            raise ValueError(f"Unsupported transport model type: {type}")
