import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)


def single_internal_mask(grid: IGridBase) -> torch.Tensor:
    single_mask = torch.ones(
        grid.num_internal_faces, dtype=torch.bool, device=grid.device
    )
    if isinstance(grid, AxisProjectedGrid):
        single_mask[grid.ap_is_immersed_faces] = False
    return single_mask


def linear_internal_face_values(
    field: CellField,
) -> Float[torch.Tensor, " F_single k"]:
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]

    axis_idx = grid.axis[single_mask, None]
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    d_fN_vec = grid.cell_centers[neighbour] - grid.face_centers[single_mask]
    d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
    d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
    w = d_fN / d_ON
    return w * field.data[owner] + (1.0 - w) * field.data[neighbour]


def least_square_grad_data(field: CellField) -> Float[torch.Tensor, " C k 3"]:
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


def skew_corrected_internal_face_values(
    field: CellField,
) -> Float[torch.Tensor, " F_single k"]:
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]
    face_centers = grid.face_centers[single_mask]

    grad_data = least_square_grad_data(field)
    psi_O = field.data[owner]
    psi_N = field.data[neighbour]
    grad_O = grad_data[owner]
    grad_N = grad_data[neighbour]

    d_OF = face_centers - grid.cell_centers[owner]
    d_NF = face_centers - grid.cell_centers[neighbour]
    psi_OF = psi_O + torch.sum(grad_O * d_OF[:, None, :], dim=2)
    psi_NF = psi_N + torch.sum(grad_N * d_NF[:, None, :], dim=2)

    axis_idx = grid.axis[single_mask, None]
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    d_fN_vec = grid.cell_centers[neighbour] - face_centers
    d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
    d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
    w = d_fN / d_ON
    return w * psi_OF + (1.0 - w) * psi_NF


def corrected_internal_sn_grad_values(
    field: CellField,
) -> Float[torch.Tensor, " F_single k"]:
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]
    axis_idx = grid.axis[single_mask, None]

    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    mag_d = torch.abs(d_ON_vec.gather(1, axis_idx))

    grad_data = least_square_grad_data(field)
    grad_O = grad_data[owner]
    grad_N = grad_data[neighbour]

    d_fN_vec = grid.cell_centers[neighbour] - grid.face_centers[single_mask]
    d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
    d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
    w = d_fN / d_ON
    grad_f = w[:, :, None] * grad_O + (1.0 - w)[:, :, None] * grad_N

    d_tangent = d_ON_vec.clone()
    d_tangent.scatter_(1, axis_idx, 0.0)
    tangential_delta = torch.sum(grad_f * d_tangent[:, None, :], dim=2)

    psi_N = field.data[neighbour]
    psi_O = field.data[owner]
    return (psi_N - psi_O - tangential_delta) / mag_d


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


def _unit_normals(
    vectors: Float[torch.Tensor, " N 3"],
) -> Float[torch.Tensor, " N 3"]:
    mag = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return vectors / mag
