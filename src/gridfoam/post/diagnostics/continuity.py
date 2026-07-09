"""Continuity error metrics compatible with OpenFOAM continuityError."""

from __future__ import annotations

import torch

from gridfoam.core.field import FaceField
from gridfoam.fv import fvc


def compute_continuity_error(
    phi: FaceField,
    cell_volumes: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute local and global continuity errors from a face flux field.

    OpenFOAM ``continuityError`` uses volume-weighted sums of the cell-centered
    divergence magnitude (local) and signed divergence (global).

    Parameters
    ----------
    phi : FaceField
        Face flux field.
    cell_volumes : torch.Tensor
        Cell volumes with shape ``[C, 1]``.

    Returns
    -------
    tuple[float, float]
        ``(local_error, global_error)`` where local is
        ``sum(|div(phi)| * V)`` and global is ``sum(div(phi) * V)``.
    """
    div_phi = fvc.div(phi).data
    local_error = torch.sum(torch.abs(div_phi) * cell_volumes).item()
    global_error = torch.sum(div_phi * cell_volumes).item()
    return local_error, global_error
