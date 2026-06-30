import torch
from jaxtyping import Float

from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_facefield,
)
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.fvc.grad import grad
from gridfoam.fv.fvc.reconstruction import single_internal_mask


def sn_grad(field: CellField) -> FaceField:
    """
    Compute the surface-normal gradient on faces.

    Uses the OpenFOAM corrected scheme:

    snGrad(psi) = nonOrthDeltaCoeffs * (psi_N - psi_O)
                  + nonOrthCorrectionVectors & grad(psi)_f.

    Parameters
    ----------
    field : CellField
        Cell-centered scalar field.

    Returns
    -------
    FaceField
        Face-centered scalar field.
    """
    grid = field.grid

    # TODO: fix dimension L: -1
    sn_grad_field = get_or_create_facefield(
        grid,
        f"snGrad({field.name})",
        FieldRole.LOCAL,
        field.num_components,
    )
    # Internal faces
    sn_grad_field.single_data = corrected_internal_sn_grad_values(field)

    for batch in iter_boundary_batches(field):
        _, _, ref_g, _ = evaluate_boundary_state(field, batch)
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            sn_grad_field.domain_bnd_data[batch.face_mask] = ref_g
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                sn_grad_field.immersed_upper[batch.face_mask] = ref_g
                continue
            if batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                sn_grad_field.immersed_lower[batch.face_mask] = ref_g
                continue

    return sn_grad_field


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

    grad_data = grad(field).data.reshape(
        grid.num_cells, field.num_components, 3
    )
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
