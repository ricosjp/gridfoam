import dataclasses as dc
from enum import Enum


@dc.dataclass(init=False, frozen=True)
class Constants:
    """
    Constants for the gridfoam package
    Changing these values breaks the code
    (e.g. the morton code)
    """

    MAX_OCTREE_DEPTH = 20
    MORTON_CODE_BIT_LENGTH = 64
    ROOT_CODE_BIT_LENGTH = 20


class AddressMode(Enum):
    """
    Addressing mode for out-of-domain neighbors.
    Let k be an index outside the valid range:
    for 'WRAP', return k % n;
    for 'CLAMP', return 0 for k < 0 and n-1 for k >= n;
    for 'BORDER', return -1 for k < 0 or k >= n.
    """

    WRAP = 0
    CLAMP = 1
    BORDER = 2


class GridCalculationMode(Enum):
    """
    Calculation mode for the grid.
    """

    CELL = 0
    NODE = 1


class CubeType(Enum):
    """
    Type of cube.
    """

    LEAF = 0
    GHOST_FROM_PARENT = 1
    GHOST_FROM_CHILD = 2
