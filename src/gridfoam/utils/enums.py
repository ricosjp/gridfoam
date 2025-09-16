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
