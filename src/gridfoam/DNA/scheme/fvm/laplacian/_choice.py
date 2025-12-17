from __future__ import annotations

from enum import Enum


class FVMLaplacianSchemeChoice(Enum):
    """
    Choice of the laplacian scheme.
    """

    LINEAR = "GaussLinear"
    """
    Linear scheme.
    """
