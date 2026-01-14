from __future__ import annotations

from enum import Enum


class FVMGradSchemeChoice(Enum):
    """
    Choice of the grad scheme.
    """

    LINEAR = "GaussLinear"
    """
    Linear scheme.
    """

