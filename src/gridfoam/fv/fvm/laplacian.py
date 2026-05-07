from typing import cast

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)


def _interpolate_gamma(
    grid: IGridBase, gamma: Float[torch.Tensor, " C 1"] | float
) -> Float[torch.Tensor, " F 1"]:
    c_own = grid.cell_centers[grid.owner]
    c_nei = grid.cell_centers[grid.neighbour]
    axis_idx = grid.axis[:, None]
    if isinstance(gamma, torch.Tensor):
        d_ON_vec = c_nei - c_own
        d_fN_vec = c_nei - grid.face_centers

        # Extract only axis-direction distance normal to the face.
        d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
        d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
        w = d_fN / d_ON

        gamma_f = w * gamma[grid.owner] + (1.0 - w) * gamma[grid.neighbour]
    else:
        gamma_f = gamma

    return gamma_f


def laplacian(
    gamma: Float[torch.Tensor, " C 1"] | float, field: CellField
) -> FvMatrix:
    """
    Build the diffusion (Laplacian) matrix term.

    Represents div(gamma * grad(phi)).

    Parameters
    ----------
    gamma : torch.Tensor | float
        Diffusion coefficient (e.g., kinematic viscosity or conductivity).
        If tensor, values are cell-centered with size ``num_cells``.
        If float, a uniform value is used for all cells.
    field : CellField
        Target field.

    Returns
    -------
    FvMatrix
        Coefficient matrix assembled from diffusion term.
    """
    mat = FvMatrix(field)
    grid = field.grid

    # Distance vector d between owner and neighbor cell centers
    c_own = grid.cell_centers[grid.owner]
    c_nei = grid.cell_centers[grid.neighbour]
    d = c_nei - c_own

    # Extract only axis-direction distance normal to the face.
    axis_idx = grid.axis[:, None]
    mag_d = torch.abs(d.gather(1, axis_idx))
    mag_Sf = cast(
        torch.Tensor, torch.linalg.vector_norm(grid.Sf, dim=1, keepdim=True)
    )

    # Interpolate gamma to face centers.
    gamma_f = _interpolate_gamma(grid, gamma)

    # Face diffusion coefficient: gamma * |Sf| / |d|
    coeff = gamma_f * mag_Sf / mag_d  # [F_internal 1]

    # Cut immersed split faces for dual-sided IBM treatment.
    if isinstance(grid, AxisProjectedGrid):
        coeff[grid.ap_is_immersed_faces] = 0.0

    mat.upper = coeff
    mat.lower = coeff

    # Subtract from owner and neighbor diagonal contributions.
    mat.diag.index_add_(0, grid.owner, -coeff)
    mat.diag.index_add_(0, grid.neighbour, -coeff)

    # Domain boundaries
    for batch in iter_boundary_batches(field):
        f, ref_v, ref_g, _ = evaluate_boundary_state(field, batch)
        gamma_bnd = (
            gamma[batch.target_cells]
            if isinstance(gamma, torch.Tensor)
            else gamma
        )
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            mag_Sf_bnd = torch.linalg.vector_norm(
                grid.domain_bnd_Sf[batch.face_mask], dim=1, keepdim=True
            )
            c_dirichlet = gamma_bnd * mag_Sf_bnd / batch.mag_d
            c_neumann = gamma_bnd * mag_Sf_bnd
            diag = -f * c_dirichlet
            src = f * c_dirichlet * ref_v + (1.0 - f) * c_neumann * ref_g
            mat.diag.index_add_(0, batch.target_cells, diag)
            mat.source.index_add_(0, batch.target_cells, -src)
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
            mag_Sf_bnd = torch.linalg.vector_norm(
                immersed_Sf[batch.face_mask], dim=1, keepdim=True
            )
            c_dirichlet = gamma_bnd * mag_Sf_bnd / batch.mag_d
            c_neumann = gamma_bnd * mag_Sf_bnd
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                wb = grid.ap_owner_weights[batch.face_mask, 0:1]
                w = grid.ap_owner_weights[batch.face_mask, 1:2]

                diag = c_dirichlet * (w - 1.0 + wb * (1.0 - f))
                src = wb * (
                    f * c_dirichlet * ref_v + (1.0 - f) * c_neumann * ref_g
                )
                mat.diag.index_add_(0, batch.target_cells, diag)
                mat.source.index_add_(0, batch.target_cells, -src)
                continue

            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                wb = grid.ap_neighbour_weights[batch.face_mask, 0:1]
                w = grid.ap_neighbour_weights[batch.face_mask, 1:2]

                diag = c_dirichlet * (w - 1.0 + wb * (1.0 - f))
                src = wb * (
                    f * c_dirichlet * ref_v + (1.0 - f) * c_neumann * ref_g
                )
                mat.diag.index_add_(0, batch.target_cells, diag)
                mat.source.index_add_(0, batch.target_cells, -src)
                continue

    return mat
