"""Weighted least-squares cell gradient kernels."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.fv.boundary_ops import (
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.kernels.face_geometry import FaceGeometry, face_geometry


def _accumulate_atb(
    atb: Float[torch.Tensor, " C 3 k"],
    cells: torch.Tensor,
    d: Float[torch.Tensor, " N 3"],
    dpsi: Float[torch.Tensor, " N k"],
) -> None:
    w2 = 1.0 / torch.sum(d * d, dim=1, keepdim=True)
    atb.index_add_(0, cells, w2[:, :, None] * d[:, :, None] * dpsi[:, None, :])


def least_squares_gradient(
    field: CellField,
    geometry: FaceGeometry | None = None,
    *,
    hanging_cells_only: bool = False,
) -> Float[torch.Tensor, " C k 3"]:
    """
    Inverse-distance-squared weighted least-squares gradient.

    The normal matrix is purely geometric and cached on the grid
    (:attr:`FaceGeometry.lsq_ata_inv`); only the right-hand side is
    assembled here. Internal equations use single-sided faces only.
    Boundary faces with a boundary condition contribute
    ``psi_b - psi_O``; faces without one act as zero-gradient equations,
    consistent with the zero-gradient extrapolation used elsewhere.

    Parameters
    ----------
    field : CellField
        Cell-centered field with ``k`` components.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.
    hanging_cells_only : bool, default False
        If True, assemble only the stencils of cells adjacent to hanging
        faces. The result is then valid only on
        :attr:`FaceGeometry.hang_cell_mask` cells; other rows are partial.

    Returns
    -------
    torch.Tensor
        Cell gradient with shape ``[C, k, 3]``.
    """
    grid = field.grid
    geo = geometry if geometry is not None else face_geometry(grid)
    k = field.num_components
    psi = field.data
    centers = grid.cell_centers

    atb = torch.zeros(
        (grid.num_cells, 3, k), dtype=grid.dtype, device=grid.device
    )

    if hanging_cells_only:
        owner = geo.owner_s[geo.lsq_face_idx]
        neighbour = geo.neighbour_s[geo.lsq_face_idx]
    else:
        owner = geo.owner_s
        neighbour = geo.neighbour_s
    d = centers[neighbour] - centers[owner]
    dpsi = psi[neighbour] - psi[owner]
    w2 = 1.0 / torch.sum(d * d, dim=1, keepdim=True)
    atb_face = w2[:, :, None] * d[:, :, None] * dpsi[:, None, :]
    atb.index_add_(0, owner, atb_face)
    atb.index_add_(0, neighbour, atb_face)

    for batch in iter_boundary_batches(field):
        cells = batch.target_cells
        d_bnd = batch.d_vec
        _, _, _, psi_b = evaluate_boundary_state(field, batch)
        if hanging_cells_only:
            select = geo.hang_cell_mask[cells]
            if not bool(torch.any(select)):
                continue
            cells = cells[select]
            d_bnd = d_bnd[select]
            psi_b = psi_b[select]
        _accumulate_atb(atb, cells, d_bnd, psi_b - psi[cells])

    return torch.bmm(geo.lsq_ata_inv, atb).transpose(1, 2)
