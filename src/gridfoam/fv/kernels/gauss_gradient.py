"""Green-Gauss assembly for cell-centered gradients."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase


def assemble_gauss_gradient(
    grid: IGridBase,
    psi_f: FaceField,
) -> Float[torch.Tensor, " C k 3"]:
    """
    Assemble a cell-centered Green-Gauss gradient from face values.

    Parameters
    ----------
    grid : IGridBase
        Grid providing face topology and geometry.
    psi_f : FaceField
        Face field supplying single-sided, domain, and immersed values.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, k, 3]``.
    """
    k = psi_f.num_components
    single_mask = psi_f.single_mask
    grad_data = torch.zeros(
        (grid.num_cells, k, 3), dtype=grid.dtype, device=grid.device
    )

    # Internal single-sided faces: owner +Sf*psi_f, neighbour -Sf*psi_f.
    Sf = grid.Sf[single_mask]
    flux = Sf[:, None, :] * psi_f.single_data[:, :, None]
    grad_data.index_add_(0, grid.owner[single_mask], flux)
    grad_data.index_add_(0, grid.neighbour[single_mask], -flux)

    # Domain boundaries.
    flux_domain = (
        grid.domain_bnd_Sf[:, None, :] * psi_f.domain_bnd_data[:, :, None]
    )
    grad_data.index_add_(0, grid.domain_bnd_owner, flux_domain)

    # Immersed boundaries: upper uses +Sf on owner, lower uses -Sf on neighbour.
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed_owner = grid.owner[grid.ap_is_immersed_faces]
        immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
        grad_data.index_add_(
            0,
            immersed_owner,
            immersed_Sf[:, None, :] * psi_f.immersed_upper[:, :, None],
        )
        grad_data.index_add_(
            0,
            immersed_neighbour,
            -immersed_Sf[:, None, :] * psi_f.immersed_lower[:, :, None],
        )

    return grad_data / grid.cell_volumes.unsqueeze(-1)
