"""Green-Gauss assembly for cell-centered gradients."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import GridBase
from gridfoam.core.shapes import broadcast_entity
from gridfoam.fv.kernels.face_geometry import FaceGeometry, face_geometry


def assemble_gauss_gradient(
    grid: GridBase,
    psi_f: FaceField,
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " C *component_shape 3"]:
    """
    Assemble a cell-centered Green-Gauss gradient from face values.

    Parameters
    ----------
    grid : GridBase
        Grid providing face topology and geometry.
    psi_f : FaceField
        Face field supplying single-sided, domain, and immersed values.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, *component_shape, 3]``.
    """
    geo = geometry if geometry is not None else face_geometry(grid)
    component_shape = psi_f.component_shape
    grad_data = torch.zeros(
        (grid.num_cells, *component_shape, 3),
        dtype=grid.dtype,
        device=grid.device,
    )

    # Internal single-sided faces: owner +Sf*psi_f, neighbour -Sf*psi_f.
    # Append the spatial direction after the field's physical component axes.
    flux = torch.einsum("n...,nj->n...j", psi_f.single_data, geo.Sf_s)
    grad_data.index_add_(0, geo.owner_s, flux)
    grad_data.index_add_(0, geo.neighbour_s, -flux)

    # Domain boundaries.
    flux_domain = torch.einsum(
        "n...,nj->n...j", psi_f.domain_bnd_data, grid.domain_bnd_Sf
    )
    grad_data.index_add_(0, grid.domain_bnd_owner, flux_domain)

    # Immersed boundaries: upper uses +Sf on owner, lower uses -Sf on neighbour.
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed = grid.ap_is_immersed_faces
        immersed_Sf = grid.Sf[immersed]
        grad_data.index_add_(
            0,
            grid.owner[immersed],
            torch.einsum("n...,nj->n...j", psi_f.immersed_upper, immersed_Sf),
        )
        grad_data.index_add_(
            0,
            grid.neighbour[immersed],
            torch.einsum("n...,nj->n...j", psi_f.immersed_lower, -immersed_Sf),
        )

    volumes = broadcast_entity(grid.cell_volumes, grad_data)
    return grad_data / volumes
