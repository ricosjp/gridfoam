from enum import Enum


class CoordinateType(Enum):
    GLOBAL = "global"
    LOCAL = "local"
    CELL_CENTER = "cell_center"
    CELL_INDEX = "cell_index"
