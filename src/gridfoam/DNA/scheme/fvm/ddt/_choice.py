from __future__ import annotations

from enum import Enum


class FVMDdtSchemeChoice(Enum):
    """
    Choice of the ddt scheme.
    """

    EULER = "Euler"
    """
    Euler scheme.
    """
    # BACKWARD_EULER = "BackwardEuler"
    # """
    # Backward Euler scheme.
    # """
    # CRANK_NICHOLSON = "CrankNicholson"
    # """
    # Crank-Nicolson scheme.
    # """


