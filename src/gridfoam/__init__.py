from importlib.metadata import version

from beartype.claw import beartype_this_package

beartype_this_package()
__version__ = version("gridfoam")
