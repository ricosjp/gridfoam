"""Continuity error metrics compatible with OpenFOAM continuityError."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import FaceField
from gridfoam.fv import fvc


def compute_continuity_error(
    phi: FaceField,
    cell_volumes: Float[torch.Tensor, " C"],
    delta_t: float,
) -> tuple[float, float]:
    """
    Compute local and global continuity errors from a face flux field.

    Mirrors OpenFOAM ``continuityErrs.H``: both errors are volume-weighted
    averages of the cell-centered divergence ``contErr = div(phi)``, scaled
    by the time-step size. The local error uses the magnitude and the global
    error the signed value:

    ``localError = dt * sum(|contErr| * V) / sum(V)``,
    ``globalError = dt * sum(contErr * V) / sum(V)``.

    Parameters
    ----------
    phi : FaceField
        Face flux field.
    cell_volumes : torch.Tensor
        Cell volumes with shape ``[C]``.
    delta_t : float
        Time-step size.

    Returns
    -------
    tuple[float, float]
        ``(local_error, global_error)``.
    """
    cont_err = fvc.div(phi).data
    total_volume = torch.sum(cell_volumes)
    local_error = torch.sum(torch.abs(cont_err) * cell_volumes) / total_volume
    global_error = torch.sum(cont_err * cell_volumes) / total_volume
    return (
        (delta_t * local_error).item(),
        (delta_t * global_error).item(),
    )
