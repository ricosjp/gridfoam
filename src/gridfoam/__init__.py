from importlib.metadata import version

from beartype.claw import beartype_this_package

from gridfoam._base import TensorGrid
from gridfoam._io import save_grid

beartype_this_package()
__version__ = version("gridfoam")
