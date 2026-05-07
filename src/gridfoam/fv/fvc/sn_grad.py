import torch

from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)


def sn_grad(field: CellField) -> FaceField:
    """
    Compute the surface-normal gradient on faces.

    snGrad(psi) ~= (psi_N - psi_O) / |d|.

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

    sn_grad_field = grid.get_field(f"snGrad({field.name})")
    if sn_grad_field is None:
        sn_grad_field = FaceField(
            grid,
            name=f"snGrad({field.name})",
            role=FieldRole.LOCAL,
            num_components=field.num_components,
            dimension=field.dimension,  # TODO: fix L: -1
            export=False,
        )
    owner_single = grid.owner[sn_grad_field.single_mask]
    neighbour_single = grid.neighbour[sn_grad_field.single_mask]

    # Internal faces
    axis_single = grid.axis[sn_grad_field.single_mask, None]
    d_vec_single = (
        grid.cell_centers[neighbour_single] - grid.cell_centers[owner_single]
    )
    mag_d_single = torch.abs(d_vec_single.gather(1, axis_single))
    psi_N_single = field.data[neighbour_single]
    psi_O_single = field.data[owner_single]
    sn_grad_field.single_data = (psi_N_single - psi_O_single) / mag_d_single

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
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                sn_grad_field.immersed_lower[batch.face_mask] = ref_g
                continue

    return sn_grad_field
