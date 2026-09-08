from __future__ import annotations

from gridfoam.core.grid.base import GridBase
from gridfoam.meta.enums import TransportModelType
from gridfoam.models.transport.base import TransportModel
from gridfoam.models.transport.newtonian import NewtonianTransport


def create_transport_model(grid: GridBase) -> TransportModel:
    """
    Create a transport model from ``properties.transport`` configuration.

    Parameters
    ----------
    grid : GridBase
        Computational grid whose simulator configuration selects the model.

    Returns
    -------
    TransportModel
        Constructed transport-model instance.

    Raises
    ------
    ValueError
        If the configured transport model type is unsupported.
    """
    type = grid.sim_config.properties.transport.type
    match type:
        case TransportModelType.NEWTONIAN:
            return NewtonianTransport(grid)
        case _:
            raise ValueError(f"Unsupported transport model type: {type}")
