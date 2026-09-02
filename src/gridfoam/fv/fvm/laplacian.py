from typing import cast

import torch
from jaxtyping import Float, Int

from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.fvc.grad import grad
from gridfoam.fv.fvc.interpolate import interpolate
from gridfoam.fv.kernels.face_interpolation import linear_face_weights
from gridfoam.fv.kernels.geometry import (
    non_orth_correction_vectors,
    non_orth_delta_coeffs,
)


def _interpolate_gamma(
    grid: IGridBase, gamma: Float[torch.Tensor, " C 1"] | float
) -> Float[torch.Tensor, " F 1"] | float:
    if not isinstance(gamma, torch.Tensor):
        return gamma
    # All internal faces, including immersed; those coefficients are zeroed
    # later so the LDU arrays keep internal-face length.
    all_internal = torch.ones(
        grid.num_internal_faces, dtype=torch.bool, device=grid.device
    )
    owner, neighbour, w = linear_face_weights(grid, all_internal)
    return w * gamma[owner] + (1.0 - w) * gamma[neighbour]


def _non_orthogonal_correction_source(
    field: CellField,
    gamma_f: Float[torch.Tensor, " F 1"] | float,
    d: Float[torch.Tensor, " F 3"],
    delta_coeffs: Float[torch.Tensor, " F 1"],
    mag_Sf: Float[torch.Tensor, " F 1"],
) -> tuple[
    Int[torch.Tensor, " F_single"],
    Int[torch.Tensor, " F_single"],
    Float[torch.Tensor, " F_single k"],
]:
    """
    Build the explicit non-orthogonal Laplacian source on single-sided faces.

    Returns owner/neighbour cell indices and
    ``gamma_f * |Sf| * (k & grad(psi)_f)`` for each face.
    """
    grid = field.grid
    grad_psi_f = interpolate(grad(field))
    single_mask = grad_psi_f.single_mask

    d_vec = d[single_mask]
    delta_coeffs_face = delta_coeffs[single_mask]
    Sf = grid.Sf[single_mask]
    mag_Sf_face = mag_Sf[single_mask]
    corr_vec = non_orth_correction_vectors(
        d_vec, delta_coeffs_face, mag_Sf_face, Sf
    )

    grad_f = grad_psi_f.single_data.reshape(-1, field.num_components, 3)
    correction = torch.sum(corr_vec[:, None, :] * grad_f, dim=2)

    if isinstance(gamma_f, torch.Tensor):
        gamma_face = gamma_f[single_mask]
    else:
        gamma_face = gamma_f
    source = gamma_face * mag_Sf_face * correction

    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]
    return owner, neighbour, source


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

    mag_Sf = cast(
        torch.Tensor, torch.linalg.vector_norm(grid.Sf, dim=1, keepdim=True)
    )
    delta_coeffs = non_orth_delta_coeffs(d, mag_Sf, grid.Sf)

    # Interpolate gamma to face centers.
    gamma_f = _interpolate_gamma(grid, gamma)

    # Face diffusion coefficient: gamma * |Sf| * nonOrthDeltaCoeffs
    coeff = gamma_f * mag_Sf * delta_coeffs

    # Cut immersed split faces for dual-sided IBM treatment.
    if isinstance(grid, AxisProjectedGrid):
        coeff[grid.ap_is_immersed_faces] = 0.0

    mat.upper = coeff
    mat.lower = coeff

    # Subtract from owner and neighbor diagonal contributions.
    mat.diag.index_add_(0, grid.owner, -coeff)
    mat.diag.index_add_(0, grid.neighbour, -coeff)

    # Non-orthogonal correction source
    owner_single, neighbour_single, correction_src = (
        _non_orthogonal_correction_source(
            field, gamma_f, d, delta_coeffs, mag_Sf
        )
    )
    mat.source.index_add_(0, owner_single, -correction_src)
    mat.source.index_add_(0, neighbour_single, correction_src)

    # Store the explicit correction so that ``FvMatrix.flux`` can reproduce the
    # full discrete face flux (OpenFOAM ``faceFluxCorrectionPtr`` equivalent).
    mat.face_flux_correction = correction_src

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
