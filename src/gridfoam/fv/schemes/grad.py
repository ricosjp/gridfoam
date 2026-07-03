from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.fvc.interpolate import (
    interpolate,
    linear_internal_face_values,
    single_face_linear_weights,
    single_internal_mask,
)
from gridfoam.meta.enums import GradScheme

GradSchemeFunc = Callable[[CellField], Float[torch.Tensor, " C 3"]]


# =============================================================================
# Green-Gauss assembly
# =============================================================================
def _gauss_assemble(
    field: CellField,
    psi_f: FaceField,
    internal_single_data: Float[torch.Tensor, " F_single k"],
) -> Float[torch.Tensor, " C 3"]:
    """
    Assemble a scalar Green-Gauss gradient from face values.

    Parameters
    ----------
    field : CellField
        Cell-centered scalar field.
    psi_f : FaceField
        Interpolated face field supplying boundary face values.
    internal_single_data : torch.Tensor
        Face values on single-sided internal faces, overriding
        ``psi_f.single_data``.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, 3]``.
    """
    grid = field.grid
    single_mask = single_internal_mask(grid)

    grad_data = torch.zeros(
        (grid.num_cells, 3), dtype=grid.dtype, device=grid.device
    )

    # Internal single-sided faces
    flux = grid.Sf[single_mask] * internal_single_data
    grad_data.index_add_(0, grid.owner[single_mask], flux)
    grad_data.index_add_(0, grid.neighbour[single_mask], -flux)

    # Domain boundaries
    flux_domain_bnd = grid.domain_bnd_Sf * psi_f.domain_bnd_data
    grad_data.index_add_(0, grid.domain_bnd_owner, flux_domain_bnd)

    # Immersed boundaries
    if isinstance(grid, AxisProjectedGrid):
        immersed_owner = grid.owner[grid.ap_is_immersed_faces]
        immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
        grad_data.index_add_(
            0, immersed_owner, immersed_Sf * psi_f.immersed_upper
        )
        grad_data.index_add_(
            0, immersed_neighbour, -immersed_Sf * psi_f.immersed_lower
        )

    return grad_data / grid.cell_volumes


# =============================================================================
# Face-centre offset correction
# =============================================================================
def corrected_linear_internal_face_values(
    field: CellField,
    grad_data: Float[torch.Tensor, " C 3"],
) -> Float[torch.Tensor, " F_single k"]:
    """
    Linear face values with face-centre offset correction.

    Correct the axis-aligned linear estimate using a provisional
    cell-centered gradient:

    ``psi_f = w * psi_O + (1 - w) * psi_N + grad_f & (C_f - C_w)``,

    where ``C_w`` is the linearly interpolated point between the two cell
    centers and ``C_f`` is the face center. On uniform meshes ``C_f = C_w``
    and the correction vanishes identically.

    Parameters
    ----------
    field : CellField
        Cell-centered scalar field.
    grad_data : torch.Tensor
        Provisional cell-centered gradient with shape ``[C, 3]``.

    Returns
    -------
    torch.Tensor
        Corrected face values on single-sided internal faces.
    """
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner, neighbour, w = single_face_linear_weights(grid, single_mask)
    psi_linear = w * field.data[owner] + (1.0 - w) * field.data[neighbour]

    centroid = (
        w * grid.cell_centers[owner]
        + (1.0 - w) * grid.cell_centers[neighbour]
    )
    grad_face = w * grad_data[owner] + (1.0 - w) * grad_data[neighbour]
    face_offset = grid.face_centers[single_mask] - centroid
    correction = torch.sum(grad_face * face_offset, dim=1, keepdim=True)
    return psi_linear + correction


# =============================================================================
# Least-squares reconstruction
# =============================================================================
def _unit_normals(
    vectors: Float[torch.Tensor, " N 3"],
) -> Float[torch.Tensor, " N 3"]:
    mag = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return vectors / mag


def _add_boundary_least_square_terms(
    field: CellField,
    ata: Float[torch.Tensor, " C 3 3"],
    atb: Float[torch.Tensor, " C 3 k"],
) -> None:
    grid = field.grid
    for batch in iter_boundary_batches(field):
        _, _, _, psi_b = evaluate_boundary_state(field, batch)
        psi_O = field.data[batch.target_cells]
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            face_centers = grid.domain_bnd_face_centers[batch.face_mask]
        elif isinstance(grid, AxisProjectedGrid):
            immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                n_hat = _unit_normals(immersed_Sf[batch.face_mask])
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                n_hat = _unit_normals(-immersed_Sf[batch.face_mask])
            else:
                continue
            face_centers = (
                grid.cell_centers[batch.target_cells] + n_hat * batch.mag_d
            )
        else:
            continue

        d = face_centers - grid.cell_centers[batch.target_cells]
        dpsi = psi_b - psi_O
        w2 = 1.0 / torch.sum(d * d, dim=1, keepdim=True)
        ata_face = w2[:, :, None] * d[:, :, None] * d[:, None, :]
        atb_face = w2[:, :, None] * d[:, :, None] * dpsi[:, None, :]
        ata.index_add_(0, batch.target_cells, ata_face)
        atb.index_add_(0, batch.target_cells, atb_face)


def _least_square_grad_data(
    field: CellField,
) -> Float[torch.Tensor, " C k 3"]:
    grid = field.grid
    owner = grid.owner
    neighbour = grid.neighbour
    d = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    dpsi = field.data[neighbour] - field.data[owner]
    w2 = 1.0 / torch.sum(d * d, dim=1, keepdim=True)

    ata_face = w2[:, :, None] * d[:, :, None] * d[:, None, :]
    atb_face = w2[:, :, None] * d[:, :, None] * dpsi[:, None, :]

    ata = torch.zeros(
        (grid.num_cells, 3, 3), dtype=grid.dtype, device=grid.device
    )
    atb = torch.zeros(
        (grid.num_cells, 3, field.num_components),
        dtype=grid.dtype,
        device=grid.device,
    )
    ata.index_add_(0, owner, ata_face)
    ata.index_add_(0, neighbour, ata_face)
    atb.index_add_(0, owner, atb_face)
    atb.index_add_(0, neighbour, atb_face)

    _add_boundary_least_square_terms(field, ata, atb)

    return torch.bmm(torch.linalg.pinv(ata), atb).transpose(1, 2)


# =============================================================================
# Gradient schemes
# =============================================================================
def linear(field: CellField) -> Float[torch.Tensor, " C 3"]:
    """
    Green-Gauss gradient with face-centre offset correction.

    A provisional (uncorrected) linear Gauss gradient is computed first, then
    used to correct internal face values by the offset between the face center
    and the linearly interpolated centroid ``C_w``. The Gauss gradient is
    finally reassembled from the corrected internal face values. On uniform
    meshes ``C_f = C_w`` and this reduces to the standard linear Green-Gauss
    gradient.

    Parameters
    ----------
    field : CellField
        Cell-centered scalar field.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, 3]``.
    """
    psi_f = interpolate(field)
    linear_faces = linear_internal_face_values(field)
    provisional = _gauss_assemble(field, psi_f, linear_faces)
    corrected = corrected_linear_internal_face_values(field, provisional)
    return _gauss_assemble(field, psi_f, corrected)


def leastsquare(field: CellField) -> Float[torch.Tensor, " C 3"]:
    """
    Weighted least-squares gradient.

    Parameters
    ----------
    field : CellField
        Cell-centered scalar field.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, 3]``.
    """
    return _least_square_grad_data(field).reshape(field.grid.num_cells, 3)


# Scheme dispatch table
GRAD_SCHEMES: dict[GradScheme, GradSchemeFunc] = {
    GradScheme.LINEAR: linear,
    GradScheme.LEASTSQUARE: leastsquare,
}


def get_grad_scheme(scheme: GradScheme) -> GradSchemeFunc:
    """Return the gradient-scheme function for the given enum."""
    return GRAD_SCHEMES[scheme]
