"""Shared helpers for incompressible flow algorithms."""

from gridfoam.algorithms.utils.reference_value import (
    needs_reference_value,
    set_reference_value,
)

__all__ = [
    "needs_reference_value",
    "set_reference_value",
]
