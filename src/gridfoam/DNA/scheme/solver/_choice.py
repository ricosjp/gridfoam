from enum import Enum


class SolverMethodChoice(Enum):
    """
    Choice of the solver method.
    """

    CG = "CG"
    """
    CG solver.
    """
    BICGSTAB = "BiCGSTAB"
    """
    BiCGSTAB solver.
    """
