from importlib.metadata import version

from beartype.claw import beartype_this_package

from gridfoam.cubion import (
    NodeType,
    PyBBox,
    PyGrid,
    PyOctreeLevel,
    PyOctreeNode,
    generate_grid_from_polydata,
)

beartype_this_package()
__version__ = version("gridfoam")
