from __future__ import annotations

from enum import Enum


class FVMDivSchemeChoice(Enum):
    """
    Choice of the div scheme.
    """

    UPWIND = "Upwind"
    """
    Upwind scheme.
    """
    # LIMITED_LINEAR = "LimitedLinear"
    # """
    # Limited linear scheme.
    # """

