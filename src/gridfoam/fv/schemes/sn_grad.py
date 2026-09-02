"""Surface-normal gradient scheme."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.fv.kernels.face_interpolation import single_internal_mask
from gridfoam.fv.schemes.grad import eval_grad


def corrected(field: CellField) -> Float[torch.Tensor, " F_single k"]:
    """
    Skewness-aware surface-normal gradient on single-sided internal faces.

    Each cell value is first reconstructed onto the normal line passing
    through the face centre ``C_f`` using the cell gradient, removing the
    tangential (skew) offset of the cell centre from that line:

    ``psi_O* = psi_O + grad_O & (C_f - C_O)_tangential``,
    ``psi_N* = psi_N + grad_N & (C_f - C_N)_tangential``,

    and the normal gradient is then the compact difference along the face
    normal ``(psi_N* - psi_O*) / |(C_N - C_O) . n|``. Unlike a correction based
    only on the owner--neighbour vector, this uses the true face centre ``C_f``
    explicitly, so it stays exact for linear fields on hanging-node (2:1)
    octree interfaces where ``C_f`` is offset from the owner--neighbour line.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    torch.Tensor
        Corrected surface-normal gradient with shape ``[F_single, k]``.
    """
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]
    axis_idx = grid.axis[single_mask, None]

    c_own = grid.cell_centers[owner]
    c_nei = grid.cell_centers[neighbour]
    c_face = grid.face_centers[single_mask]

    d_ON_vec = c_nei - c_own
    mag_d = torch.abs(d_ON_vec.gather(1, axis_idx))

    grad_data = eval_grad(field)
    grad_O = grad_data[owner]
    grad_N = grad_data[neighbour]

    # Tangential (in-face) offset from each cell centre to the face centre.
    # The axis/normal component is removed; these vanish on non-skewed faces.
    move_O = c_face - c_own
    move_N = c_face - c_nei
    move_O.scatter_(1, axis_idx, 0.0)
    move_N.scatter_(1, axis_idx, 0.0)

    psi_O = field.data[owner] + torch.sum(grad_O * move_O[:, None, :], dim=2)
    psi_N = field.data[neighbour] + torch.sum(
        grad_N * move_N[:, None, :], dim=2
    )

    return (psi_N - psi_O) / mag_d
