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
from gridfoam.fv.fvc.interpolate import single_internal_mask


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

    grad_data = grad(field).data.reshape(
        grid.num_cells, field.num_components, 3
    )
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
