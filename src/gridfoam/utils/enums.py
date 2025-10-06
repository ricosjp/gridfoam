from enum import Enum


class GridMode(Enum):
    """
    Grid mode for saving the grid.
    """

    CELL = "cell"
    CUBE = "cube"


class Direction(Enum):
    """
    Direction of the grid.
    """

    ZM = 4
    YM = 10
    XM = 12
    CENTER = 13
    XP = 14
    YP = 16
    ZP = 22


FACE_NEIGHBOR_INDEX = [
    Direction.ZM.value,
    Direction.ZP.value,
    Direction.YM.value,
    Direction.YP.value,
    Direction.XM.value,
    Direction.XP.value,
]

class TVDScheme(Enum):
    """
    TVD scheme for interpolation.
    """

    SUPERBEE = "superbee"
    MINMOD = "minmod"
    LIMITED_LINEAR = "limited_linear"
    VAN_LEER = "van_leer"
    VAN_ALBADA = "van_albada"
    UPWIND = "upwind"
