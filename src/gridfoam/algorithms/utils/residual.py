"""Residual evaluation helpers for OpenFOAM-style convergence control."""

from __future__ import annotations

import torch

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.fv import fvc


def field_initial_residual(
    mat: FvMatrix,
    field: CellField,
) -> float:
    """
    Compute an OpenFOAM-style normalized linear-system residual.

    ``|A*psi - b|_2 / (|b|_2 + eps)`` using the current field values as
    ``psi``.

    Parameters
    ----------
    mat : FvMatrix
        Assembled finite-volume matrix.
    field : CellField
        Field whose current values are used in the matrix product.

    Returns
    -------
    float
        Normalized residual.
    """
    psi = field.data
    residual = mat.multiply(psi) - mat.source
    res_norm = torch.linalg.vector_norm(residual, ord=2)
    src_norm = torch.linalg.vector_norm(mat.source, ord=2)
    denom = src_norm + torch.finfo(residual.dtype).eps
    return (res_norm / denom).item()


def continuity_residual(phi: FaceField) -> float:
    """
    Compute the global continuity residual as ``|div(phi)|_2``.

    Parameters
    ----------
    phi : FaceField
        Face flux field.

    Returns
    -------
    float
        L2 norm of the cell-centered divergence.
    """
    return torch.linalg.vector_norm(fvc.div(phi).data, ord=2).item()


def residual_satisfied(
    residual: float,
    tolerance: float,
    rel_tolerance: float,
    initial_residual: float,
) -> bool:
    """
    Check whether a residual meets OpenFOAM ``residualControl`` criteria.

    Convergence requires both:

    - ``residual <= tolerance`` (absolute), and
    - ``residual <= rel_tolerance * initial_residual`` when
      ``rel_tolerance > 0``.

    Parameters
    ----------
    residual : float
        Current residual value.
    tolerance : float
        Absolute tolerance.
    rel_tolerance : float
        Relative tolerance multiplier applied to the initial residual.
    initial_residual : float
        Initial residual used for relative checks.

    Returns
    -------
    bool
        ``True`` when the absolute and relative criteria are satisfied.
    """
    if residual > tolerance:
        return False
    if rel_tolerance > 0.0 and residual > rel_tolerance * initial_residual:
        return False
    return True
