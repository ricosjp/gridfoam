import logging

from gridfoam.core.grid.base import IGridBase
from gridfoam.models.transport.base import TransportModel

logger = logging.getLogger(__name__)


class NewtonianTransport(TransportModel):
    def __init__(self, grid: IGridBase):
        self.grid = grid
        self._nu = grid.sim_config.properties.transport.nu

    def nu(self) -> float:
        return self._nu
