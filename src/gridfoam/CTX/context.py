from dataclasses import dataclass

from gridfoam.RNA.grid_handle import GridHandle
from gridfoam.RNA.registry import SimulationMetaRegistry


@dataclass
class SimulationContext:
    registry: SimulationMetaRegistry
    grid_handle: GridHandle
